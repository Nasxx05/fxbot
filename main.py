"""Main entry point for the forex trading bot.

Two operating modes, selected via ``mode`` in ``config/config.yaml``:

* ``live``          — fully automated. ExecutionEngine places orders on MT5
                      and TradeManager manages positions.
* ``signal_only``   — analysis only. The 11-gate pipeline is run as usual,
                      but instead of placing orders the bot pushes a
                      NEW SIGNAL message to Telegram and tracks the signal's
                      lifecycle (entry hit / TP1 / TP2 / SL / invalidated)
                      via SignalMonitor, emitting follow-up alerts.

All analysis engines (features, liquidity, structure, volatility, news,
session, spread, correlation, circuit breaker) run identically in both
modes so signal quality is unchanged.
"""

import json
import os
import subprocess
import sys
import traceback

import yaml
from dotenv import load_dotenv

from src.alert_system import AlertSystem
from src.circuit_breaker import CircuitBreaker
from src.correlation_guard import CorrelationGuard
from src.data_engine import DataEngine
from src.execution_engine import ExecutionEngine
from src.feature_engine import FeatureEngine
from src.liquidity_engine import LiquidityEngine
from src.logger import BotLogger
from src.news_filter import NewsFilter
from src.risk_engine import RiskEngine
from src.session_filter import SessionFilter
from src.signal_monitor import SignalMonitor
from src.spread_controller import SpreadController
from src.strategy_engine import StrategyEngine
from src.structure_engine import StructureEngine
from src.trade_manager import TradeManager
from src.volatility_engine import VolatilityEngine


def main():
    """Initialize and start the forex trading bot."""
    load_dotenv()

    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mode = (config.get("mode") or "live").strip().lower()
    if mode not in ("live", "signal_only"):
        raise ValueError(
            f"Invalid mode '{mode}' in config.yaml — must be 'live' or 'signal_only'")

    logger = BotLogger(config)
    logger.log("INFO", "main",
               f"Starting Forex Bot (MetaTrader 5) — mode={mode}")

    # 1. Initialize all analysis modules (same in both modes).
    data_engine = DataEngine(config)
    feature_engine = FeatureEngine(config)
    liquidity_engine = LiquidityEngine(config, logger)
    structure_engine = StructureEngine(config, logger)
    volatility_engine = VolatilityEngine(config, logger)
    news_filter = NewsFilter(config, logger)
    session_filter = SessionFilter(config, logger)
    spread_controller = SpreadController(config, logger, data_engine)
    correlation_guard = CorrelationGuard(config, logger)
    risk_engine = RiskEngine(config, logger, data_engine)
    circuit_breaker = CircuitBreaker(config, logger)
    alert_system = AlertSystem(config, logger)

    # 2. Mode-specific execution/monitoring layer.
    execution_engine = None
    trade_manager = None
    signal_monitor = None

    if mode == "live":
        execution_engine = ExecutionEngine(config, logger, data_engine,
                                           risk_engine, spread_controller)
        trade_manager = TradeManager(config, logger, execution_engine,
                                     volatility_engine)
    else:  # signal_only
        signal_monitor = SignalMonitor(config, logger, data_engine,
                                       alert_system, correlation_guard)
        signal_monitor.start()

    strategy_engine = StrategyEngine(
        config=config,
        logger=logger,
        data_engine=data_engine,
        feature_engine=feature_engine,
        liquidity_engine=liquidity_engine,
        structure_engine=structure_engine,
        volatility_engine=volatility_engine,
        news_filter=news_filter,
        session_filter=session_filter,
        spread_controller=spread_controller,
        correlation_guard=correlation_guard,
    )

    instruments = config.get("broker", {}).get("instruments", [])
    timeframes = [
        config.get("broker", {}).get("primary_timeframe", "M15"),
        config.get("broker", {}).get("confirmation_timeframe", "H1"),
        config.get("broker", {}).get("context_timeframe", "H4"),
    ]

    # 3. Warm up historical data.
    logger.log("INFO", "main", "Warming up historical data...")
    data_engine.warm_up(instruments, timeframes)

    # 4. Restore circuit breaker state (already loaded in __init__).
    logger.log("INFO", "main",
               f"Circuit breaker status: {circuit_breaker.state['status']}")

    # 5. Restore open trades (live mode only).
    if trade_manager is not None:
        open_count = len(trade_manager.open_trades)
        logger.log("INFO", "main", f"Restored {open_count} open trades")

    # 6. Send bot online alert.
    if mode == "signal_only":
        alert_system.send(
            "*BOT ONLINE — SIGNAL-ONLY MODE*\n"
            f"Instruments: {', '.join(i.replace('_', '/') for i in instruments)}\n"
            "You will receive trade signals here for manual entry.\n"
            "Follow-ups: entry hit, TP1, TP2, SL, invalidation."
        )
    else:
        alert_system.alert_bot_online(instruments)

    # 7. Start dashboard in subprocess.
    dashboard_process = None
    try:
        dashboard_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
             "dashboard/dashboard.py",
             "--server.headless", "true",
             "--server.port", "8501"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.log("INFO", "main", "Dashboard started on port 8501")
    except Exception as e:
        logger.log_error("main", f"Dashboard failed to start: {e}")

    # 8. Define the candle callback.
    def on_new_candle(instrument, candle_data):
        """Process a new completed candle."""
        try:
            # Check circuit breaker.
            paused = circuit_breaker.is_paused()
            if paused or circuit_breaker.state["status"] in ("STOPPED", "DAILY_PAUSED"):
                return

            # Fetch latest candles and compute features.
            df = data_engine.fetch_historical_candles(
                instrument,
                config["broker"]["primary_timeframe"],
                count=500,
            )
            if df.empty:
                return

            df = feature_engine.compute_all(df)

            # Write live spread data for dashboard.
            try:
                spread_val = data_engine.get_current_spread(instrument)
                spread_path = os.path.join("data", "live_spreads.json")
                spreads = {}
                if os.path.exists(spread_path):
                    with open(spread_path, "r") as sf:
                        spreads = json.load(sf)
                spreads[instrument] = spread_val
                with open(spread_path, "w") as sf:
                    json.dump(spreads, sf)
            except Exception:
                pass

            # Write live volatility data for dashboard.
            try:
                vol_state = volatility_engine.get_volatility_state(df)
                atr_val = volatility_engine.get_atr_value(df)
                vol_path = os.path.join("data", "live_volatility.json")
                vols = {}
                if os.path.exists(vol_path):
                    with open(vol_path, "r") as vf:
                        vols = json.load(vf)
                vols[instrument] = {
                    "regime": vol_state.get("regime", "N/A"),
                    "atr": float(atr_val) if atr_val else 0,
                }
                with open(vol_path, "w") as vf:
                    json.dump(vols, vf)
            except Exception:
                pass

            # Live-mode trade management happens per candle.
            if mode == "live" and trade_manager is not None and not df.empty:
                current_price = df["close"].iloc[-1]
                trade_manager.on_price_update(instrument, current_price, df)

            # Run the strategy pipeline — identical in both modes.
            signal = strategy_engine.on_new_candle(instrument, df)
            if signal is None:
                return

            # Attach position size (so the user sees suggested lots).
            sized_signal = risk_engine.attach_position_size_to_signal(signal)
            if sized_signal is None:
                return

            # -------------------------------------------------------------
            # Mode fork: live places an order; signal_only pushes to Telegram
            # via the SignalMonitor, which also watches the lifecycle.
            # -------------------------------------------------------------
            if mode == "live":
                confirmed = execution_engine.execute_signal(sized_signal)
                if confirmed is not None:
                    trade_manager.register_trade(confirmed)
                    correlation_guard.register_open_trade(instrument)
                    alert_system.alert_trade_opened(confirmed)
            else:
                # SignalMonitor takes ownership of the signal: sends the
                # initial Telegram alert and watches for entry/TP/SL hits.
                signal_monitor.register_signal(sized_signal)
                # Register with correlation guard so we do not push a
                # second signal on a correlated pair while this one is
                # still active.
                correlation_guard.register_open_trade(instrument)

        except Exception as e:
            logger.log_error("main", f"Candle callback error: {e}",
                             {"instrument": instrument,
                              "traceback": traceback.format_exc()})
            alert_system.alert_bot_error(str(e), "on_new_candle")

    # 9. Start the main loop.
    logger.log("INFO", "main", "Starting main trading loop (MT5 polling)...")
    print(f"Forex Bot started — monitoring {instruments}")
    print(f"Mode: {mode.upper()} / "
          f"{config.get('broker', {}).get('environment', 'demo').upper()}")
    print("Broker: MetaTrader 5")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            data_engine.stream_candles(instruments, on_new_candle)
        except KeyboardInterrupt:
            logger.log("INFO", "main", "Shutdown requested by user")
            print("\nShutting down gracefully...")
            if signal_monitor is not None:
                signal_monitor.stop()
            data_engine.shutdown()
            alert_system.send("*BOT OFFLINE* — Manual shutdown")
            break
        except Exception as e:
            logger.log_error("main", f"Main loop error: {e}",
                             {"traceback": traceback.format_exc()})
            alert_system.alert_bot_error(str(e), "main_loop")
            logger.log("INFO", "main", "Restarting main loop in 10 seconds...")
            import time
            time.sleep(10)

    # Cleanup.
    if dashboard_process:
        dashboard_process.terminate()
        logger.log("INFO", "main", "Dashboard process terminated")

    logger.log("INFO", "main", "Forex Bot shutdown complete")


if __name__ == "__main__":
    main()
