"""Backtest engine module — historical strategy backtesting and performance analysis."""

import json
import math
import os
import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.feature_engine import FeatureEngine
from src.liquidity_engine import LiquidityEngine
from src.logger import BotLogger
from src.structure_engine import StructureEngine
from src.volatility_engine import VolatilityEngine


class BacktestEngine:
    """Runs historical backtests with walk-forward validation."""

    def __init__(self, config: dict, logger: BotLogger):
        """Initialize the BacktestEngine.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance (can be None).
        """
        self.config = config
        self.logger = logger
        self.results = []

        risk = config.get("risk", {})
        self.max_trades_per_day = risk.get("max_trades_per_day", 3)
        self.min_rr_after_spread = risk.get("min_risk_reward_after_spread", 2.5)
        self.min_sl_atr_multiple = risk.get("min_sl_atr_multiple", 1.0)
        self.max_sl_atr_multiple = risk.get("max_sl_atr_multiple", 3.0)

        tm = config.get("trade_management", {})
        self.breakeven_trigger_r = tm.get("breakeven_trigger_r", 1.0)
        self.partial_close_trigger_r = tm.get("partial_close_trigger_r", 2.0)
        self.partial_close_pct = tm.get("partial_close_pct", 0.50)
        self.full_close_trigger_r = tm.get("full_close_trigger_r", 3.0)
        self.max_trade_duration_hours = tm.get("max_trade_duration_hours", 24)

        strategy = config.get("strategy", {})
        self.pullback_candle_limit = strategy.get("pullback_candle_limit", 5)

        sessions = config.get("sessions", {})
        self.allowed_sessions = sessions.get("allowed_sessions",
                                             ["LONDON", "NEW_YORK", "OVERLAP"])

        self.db_path = os.path.join("data", "historical", "candles.db")

    def load_historical_data(self, instrument: str, timeframe: str,
                             start_date: str, end_date: str) -> pd.DataFrame:
        """Load OHLC data from SQLite for the given range.

        Args:
            instrument: OANDA instrument name.
            timeframe: Candle granularity.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).

        Returns:
            DataFrame sorted by timestamp ascending.
        """
        if not os.path.exists(self.db_path):
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"])

        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE instrument = ? AND timeframe = ?
              AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(query, conn,
                               params=(instrument, timeframe, start_date, end_date))
        conn.close()

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def run_backtest(self, instrument: str, start_date: str, end_date: str,
                     spread_pips: float = None,
                     slippage_pips: float = None) -> dict:
        """Run a full backtest for a single instrument over a date range.

        Args:
            instrument: OANDA instrument name.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            spread_pips: Fixed spread to use, or None for simulated spread.
            slippage_pips: Fixed slippage, or None for random.

        Returns:
            Summary metrics dict.
        """
        df = self.load_historical_data(instrument, "M15", start_date, end_date)
        if df.empty or len(df) < 100:
            return self.calculate_metrics([], 10000)

        fe = FeatureEngine(self.config)
        df = fe.compute_all(df)

        le = LiquidityEngine(self.config, self.logger)
        se = StructureEngine(self.config, self.logger)
        ve = VolatilityEngine(self.config, self.logger)

        self.results = []
        daily_count = {}

        warmup = 100
        for i in range(warmup, len(df)):
            candle_df = df.iloc[:i + 1].copy()
            current = candle_df.iloc[-1]

            # Session filter
            session = current.get("session", "OFF_HOURS")
            if session not in self.allowed_sessions:
                continue

            # Daily limit
            trade_date = str(current["timestamp"].date())
            if daily_count.get(trade_date, 0) >= self.max_trades_per_day:
                continue

            # Volatility regime
            if "volatility_regime" in candle_df.columns:
                regime = candle_df["volatility_regime"].iloc[-1]
                if regime == "RANGING":
                    continue

            # Sweep detection
            zones = le.get_active_zones(candle_df)
            sweep = None
            for zone in zones:
                s = le.detect_sweep(candle_df, zone)
                if s is not None:
                    sweep = s
                    break

            if sweep is None:
                continue

            # BOS detection
            bos = se.detect_break_of_structure(candle_df, sweep)
            if bos is None:
                continue

            # Pullback zone
            pullback = se.detect_pullback_zone(candle_df, sweep, bos)
            if pullback is None:
                continue

            # Volatility confirmation
            if not ve.is_volatility_sufficient(candle_df, sweep=sweep):
                continue

            # Determine direction and entry
            direction = "LONG" if bos["direction"] == "BULLISH" else "SHORT"
            entry_price = pullback["entry_price"]

            # Calculate SL and TP
            atr = candle_df["atr_14"].iloc[-1]
            if pd.isna(atr) or atr <= 0:
                continue

            eff_spread = spread_pips if spread_pips is not None else self._sim_spread(current["timestamp"])
            spread_price = self._pips_to_price(instrument, eff_spread)

            if direction == "LONG":
                sl = sweep["sweep_low"] - (0.5 * atr) - spread_price
            else:
                sl = sweep["sweep_high"] + (0.5 * atr) + spread_price

            sl_distance = abs(entry_price - sl)
            if sl_distance < self.min_sl_atr_multiple * atr:
                sl_distance = self.min_sl_atr_multiple * atr
                sl = entry_price - sl_distance if direction == "LONG" else entry_price + sl_distance
            if sl_distance > self.max_sl_atr_multiple * atr:
                continue

            risk = abs(entry_price - sl)
            if direction == "LONG":
                tp1 = entry_price + risk * 2
                tp2 = entry_price + risk * 3
            else:
                tp1 = entry_price - risk * 2
                tp2 = entry_price - risk * 3

            # Spread-adjusted RR
            adjusted_risk = risk + self._pips_to_price(instrument, eff_spread)
            rr_adjusted = abs(tp2 - entry_price) / adjusted_risk if adjusted_risk > 0 else 0
            if rr_adjusted < self.min_rr_after_spread:
                continue

            signal = {
                "instrument": instrument,
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": sl,
                "take_profit_1": tp1,
                "take_profit_2": tp2,
                "entry_zone_high": pullback.get("zone_high", entry_price + 0.001),
                "entry_zone_low": pullback.get("zone_low", entry_price - 0.001),
                "session": session,
                "atr": atr,
            }

            eff_slippage = slippage_pips if slippage_pips is not None else max(
                0, random.gauss(0.3, 0.2))

            result = self.simulate_trade_outcome(
                signal, df, i, eff_spread, eff_slippage)

            self.results.append(result)
            if result["status"] != "EXPIRED":
                daily_count[trade_date] = daily_count.get(trade_date, 0) + 1

        return self.calculate_metrics(self.results)

    def simulate_trade_outcome(self, signal: dict, df: pd.DataFrame,
                               entry_index: int, spread_pips: float,
                               slippage_pips: float) -> dict:
        """Simulate a single trade outcome using future candles.

        Args:
            signal: Trade signal dict.
            df: Full DataFrame.
            entry_index: Index where the signal was generated.
            spread_pips: Spread in pips.
            slippage_pips: Slippage in pips.

        Returns:
            Trade result dict.
        """
        instrument = signal["instrument"]
        direction = signal["direction"]
        entry_price = signal["entry_price"]
        sl = signal["stop_loss"]
        tp1 = signal["take_profit_1"]
        tp2 = signal["take_profit_2"]
        zone_high = signal.get("entry_zone_high", entry_price + 0.001)
        zone_low = signal.get("entry_zone_low", entry_price - 0.001)
        entry_time = df["timestamp"].iloc[entry_index]

        spread_cost = self._pips_to_price(instrument, spread_pips)
        slip_cost = self._pips_to_price(instrument, slippage_pips)

        # Apply slippage to entry
        if direction == "LONG":
            entry_price += slip_cost
        else:
            entry_price -= slip_cost

        # Check if price reaches entry zone
        fill_index = None
        limit = min(entry_index + self.pullback_candle_limit, len(df))
        for j in range(entry_index + 1, limit):
            if df["low"].iloc[j] <= zone_high and df["high"].iloc[j] >= zone_low:
                fill_index = j
                break

        if fill_index is None:
            return {
                "status": "EXPIRED",
                "entry_price": signal["entry_price"],
                "exit_price": 0.0,
                "entry_time": str(entry_time),
                "exit_time": str(entry_time),
                "pnl_pips": 0.0,
                "pnl_usd": 0.0,
                "r_multiple": 0.0,
                "exit_reason": "EXPIRED",
                "spread_cost_pips": spread_pips,
                "slippage_cost_pips": slippage_pips,
                "commission_usd": 0.0,
                "net_pnl_usd": 0.0,
                "session": signal.get("session", ""),
                "duration_hours": 0.0,
            }

        fill_time = df["timestamp"].iloc[fill_index]
        risk = abs(entry_price - sl)
        if risk == 0:
            risk = 0.0001

        breakeven_moved = False
        partial_closed = False
        current_sl = sl

        # Simulate bar by bar from fill
        for k in range(fill_index + 1, len(df)):
            candle = df.iloc[k]
            candle_high = candle["high"]
            candle_low = candle["low"]
            candle_close = candle["close"]

            # Check duration
            duration = (candle["timestamp"] - fill_time).total_seconds() / 3600
            if duration >= self.max_trade_duration_hours:
                return self._build_result(
                    signal, entry_price, candle_close, fill_time,
                    candle["timestamp"], risk, spread_pips, slippage_pips,
                    instrument, "MAX_DURATION")

            # Check SL hit
            if direction == "LONG" and candle_low <= current_sl:
                exit_price = current_sl
                status = "BREAKEVEN" if breakeven_moved and abs(current_sl - entry_price) < risk * 0.1 else "LOSS"
                return self._build_result(
                    signal, entry_price, exit_price, fill_time,
                    candle["timestamp"], risk, spread_pips, slippage_pips,
                    instrument, "SL" if status == "LOSS" else "BREAKEVEN")

            if direction == "SHORT" and candle_high >= current_sl:
                exit_price = current_sl
                status = "BREAKEVEN" if breakeven_moved and abs(current_sl - entry_price) < risk * 0.1 else "LOSS"
                return self._build_result(
                    signal, entry_price, exit_price, fill_time,
                    candle["timestamp"], risk, spread_pips, slippage_pips,
                    instrument, "SL" if status == "LOSS" else "BREAKEVEN")

            # Calculate R
            if direction == "LONG":
                r = (candle_close - entry_price) / risk
            else:
                r = (entry_price - candle_close) / risk

            # Check TP2
            if direction == "LONG" and candle_high >= tp2:
                return self._build_result(
                    signal, entry_price, tp2, fill_time,
                    candle["timestamp"], risk, spread_pips, slippage_pips,
                    instrument, "TP2")

            if direction == "SHORT" and candle_low <= tp2:
                return self._build_result(
                    signal, entry_price, tp2, fill_time,
                    candle["timestamp"], risk, spread_pips, slippage_pips,
                    instrument, "TP2")

            # Breakeven
            if r >= self.breakeven_trigger_r and not breakeven_moved:
                if direction == "LONG":
                    current_sl = entry_price + spread_cost
                else:
                    current_sl = entry_price - spread_cost
                breakeven_moved = True

            # Trailing stop
            if breakeven_moved and r > 1.0:
                atr = candle.get("atr_14", 0.005)
                if not pd.isna(atr) and atr > 0:
                    if direction == "LONG":
                        new_sl = candle_close - atr
                        if new_sl > current_sl:
                            current_sl = new_sl
                    else:
                        new_sl = candle_close + atr
                        if new_sl < current_sl:
                            current_sl = new_sl

        # If we reach end of data without exit
        last = df.iloc[-1]
        return self._build_result(
            signal, entry_price, last["close"], fill_time,
            last["timestamp"], risk, spread_pips, slippage_pips,
            instrument, "DATA_END")

    def _build_result(self, signal, entry_price, exit_price, fill_time,
                      exit_time, risk, spread_pips, slippage_pips,
                      instrument, exit_reason):
        """Build a trade result dict."""
        direction = signal["direction"]
        if direction == "LONG":
            pnl_price = exit_price - entry_price
        else:
            pnl_price = entry_price - exit_price

        pnl_pips = self._price_to_pips(instrument, pnl_price)
        r_multiple = pnl_price / risk if risk > 0 else 0.0
        pnl_usd = pnl_pips * 10  # Approximate for 1 lot
        commission = 7.0  # Per round trip per lot

        if pnl_price > 0:
            status = "WIN"
        elif abs(pnl_price) < risk * 0.1:
            status = "BREAKEVEN"
        else:
            status = "LOSS"

        if exit_reason in ("TP2", "TP1") and pnl_price > 0:
            status = "WIN"
        elif exit_reason == "BREAKEVEN":
            status = "BREAKEVEN"
        elif exit_reason == "EXPIRED":
            status = "EXPIRED"

        duration = 0.0
        try:
            duration = (exit_time - fill_time).total_seconds() / 3600
        except Exception:
            pass

        return {
            "status": status,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_time": str(fill_time),
            "exit_time": str(exit_time),
            "pnl_pips": pnl_pips,
            "pnl_usd": pnl_usd - commission,
            "r_multiple": r_multiple,
            "exit_reason": exit_reason,
            "spread_cost_pips": spread_pips,
            "slippage_cost_pips": slippage_pips,
            "commission_usd": commission,
            "net_pnl_usd": pnl_usd - commission - (spread_pips * 10) - (slippage_pips * 10),
            "session": signal.get("session", ""),
            "duration_hours": duration,
        }

    def calculate_metrics(self, results: list,
                          initial_balance: float = 10000) -> dict:
        """Calculate performance metrics from backtest results.

        Args:
            results: List of trade result dicts.
            initial_balance: Starting account balance.

        Returns:
            Dict with all calculated metrics.
        """
        if not results:
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "expired_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "total_net_pnl_usd": 0.0, "total_return_pct": 0.0,
                "max_drawdown_usd": 0.0, "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0, "expectancy_per_trade_r": 0.0,
                "average_win_r": 0.0, "average_loss_r": 0.0,
                "largest_win_usd": 0.0, "largest_loss_usd": 0.0,
                "average_trade_duration_hours": 0.0,
                "trades_by_session": {}, "trades_by_day_of_week": {},
            }

        trades = [r for r in results if r["status"] != "EXPIRED"]
        expired = [r for r in results if r["status"] == "EXPIRED"]
        wins = [r for r in trades if r["status"] == "WIN"]
        losses = [r for r in trades if r["status"] == "LOSS"]

        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        expired_trades = len(expired)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        gross_profit = sum(r["net_pnl_usd"] for r in wins) if wins else 0
        gross_loss = abs(sum(r["net_pnl_usd"] for r in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        total_net_pnl = sum(r["net_pnl_usd"] for r in trades)
        total_return_pct = (total_net_pnl / initial_balance) * 100

        # Max drawdown
        equity_curve = [initial_balance]
        for r in trades:
            equity_curve.append(equity_curve[-1] + r["net_pnl_usd"])

        peak = initial_balance
        max_dd_usd = 0.0
        max_dd_pct = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd_usd:
                max_dd_usd = dd
                max_dd_pct = (dd / peak * 100) if peak > 0 else 0.0

        # Sharpe ratio (annualized)
        daily_returns = [r["net_pnl_usd"] / initial_balance for r in trades]
        if len(daily_returns) > 1:
            mean_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns, ddof=1)
            sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        # R-multiples
        r_multiples = [r["r_multiple"] for r in trades]
        win_rs = [r["r_multiple"] for r in wins]
        loss_rs = [r["r_multiple"] for r in losses]

        expectancy = np.mean(r_multiples) if r_multiples else 0.0
        avg_win_r = np.mean(win_rs) if win_rs else 0.0
        avg_loss_r = np.mean(loss_rs) if loss_rs else 0.0

        # Extremes
        pnls = [r["net_pnl_usd"] for r in trades]
        largest_win = max(pnls) if pnls else 0.0
        largest_loss = min(pnls) if pnls else 0.0

        # Duration
        durations = [r.get("duration_hours", 0) for r in trades]
        avg_duration = np.mean(durations) if durations else 0.0

        # By session/day
        trades_by_session = defaultdict(int)
        trades_by_day = defaultdict(int)
        for r in trades:
            trades_by_session[r.get("session", "UNKNOWN")] += 1
            try:
                dt = pd.to_datetime(r["entry_time"])
                trades_by_day[dt.strftime("%A")] += 1
            except Exception:
                pass

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "expired_trades": expired_trades,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
            "total_net_pnl_usd": round(total_net_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_usd": round(max_dd_usd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "expectancy_per_trade_r": round(expectancy, 2),
            "average_win_r": round(avg_win_r, 2),
            "average_loss_r": round(avg_loss_r, 2),
            "largest_win_usd": round(largest_win, 2),
            "largest_loss_usd": round(largest_loss, 2),
            "average_trade_duration_hours": round(avg_duration, 2),
            "trades_by_session": dict(trades_by_session),
            "trades_by_day_of_week": dict(trades_by_day),
        }

    def run_walk_forward_validation(self, instrument: str, full_start: str,
                                    full_end: str, is_pct: float = 0.70,
                                    oos_increment_months: int = 3) -> dict:
        """Run walk-forward validation splitting IS and OOS periods.

        Args:
            instrument: OANDA instrument name.
            full_start: Start date (YYYY-MM-DD).
            full_end: End date (YYYY-MM-DD).
            is_pct: In-sample percentage (default 0.70).
            oos_increment_months: OOS window size in months.

        Returns:
            Dict with IS metrics, OOS windows, and verdict.
        """
        start = pd.Timestamp(full_start)
        end = pd.Timestamp(full_end)
        total_days = (end - start).days

        is_days = int(total_days * is_pct)
        is_end = start + pd.Timedelta(days=is_days)

        # In-sample
        is_metrics = self.run_backtest(instrument, str(start.date()),
                                       str(is_end.date()))

        # Out-of-sample windows
        oos_windows = []
        oos_start = is_end
        increment = pd.DateOffset(months=oos_increment_months)

        while oos_start < end:
            oos_end_dt = min(oos_start + increment, end)
            window_metrics = self.run_backtest(instrument,
                                               str(oos_start.date()),
                                               str(oos_end_dt.date()))
            window_metrics["window_start"] = str(oos_start.date())
            window_metrics["window_end"] = str(oos_end_dt.date())
            oos_windows.append(window_metrics)
            oos_start = oos_end_dt

        # Combined OOS
        all_oos_results = []
        for w in oos_windows:
            all_oos_results.append(w)

        oos_combined = self._combine_oos_metrics(oos_windows) if oos_windows else self.calculate_metrics([])

        # Verdict
        verdict = "VALID"
        verdict_reason = "Strategy performance is consistent."

        is_pf = is_metrics.get("profit_factor", 0)
        oos_pf = oos_combined.get("profit_factor", 0)
        is_wr = is_metrics.get("win_rate", 0)
        oos_wr = oos_combined.get("win_rate", 0)

        if is_pf > 2.0 and oos_pf < 1.0:
            verdict = "OVERFIT"
            verdict_reason = (f"IS profit factor {is_pf} vs OOS {oos_pf}. "
                              "Strategy is likely overfit to historical data.")
        elif oos_pf < is_pf * 0.70 or oos_wr < is_wr * 0.70:
            verdict = "SUSPECT"
            verdict_reason = (f"OOS performance degrades >30% from IS. "
                              f"IS PF={is_pf}, OOS PF={oos_pf}. "
                              f"IS WR={is_wr}%, OOS WR={oos_wr}%.")

        return {
            "is_metrics": is_metrics,
            "oos_windows": oos_windows,
            "overall_oos_metrics": oos_combined,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
        }

    def _combine_oos_metrics(self, windows: list) -> dict:
        """Combine multiple OOS window metrics into one summary."""
        total = sum(w.get("total_trades", 0) for w in windows)
        wins = sum(w.get("winning_trades", 0) for w in windows)
        losses = sum(w.get("losing_trades", 0) for w in windows)
        net_pnl = sum(w.get("total_net_pnl_usd", 0) for w in windows)

        win_rate = (wins / total * 100) if total > 0 else 0
        # Approximate profit factor from window-level data
        gross_p = sum(w.get("total_net_pnl_usd", 0)
                      for w in windows if w.get("total_net_pnl_usd", 0) > 0)
        gross_l = abs(sum(w.get("total_net_pnl_usd", 0)
                          for w in windows if w.get("total_net_pnl_usd", 0) < 0))
        pf = gross_p / gross_l if gross_l > 0 else 0

        return {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(pf, 2),
            "total_net_pnl_usd": round(net_pnl, 2),
        }

    def save_results(self, results: dict, filename: str):
        """Save results to JSON and trades to CSV.

        Args:
            results: Metrics dict from run_backtest.
            filename: Base filename.
        """
        os.makedirs("data", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        json_path = f"data/backtest_{filename}_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if self.results:
            csv_path = f"data/backtest_{filename}_{ts}_trades.csv"
            pd.DataFrame(self.results).to_csv(csv_path, index=False)

    def print_report(self, metrics: dict):
        """Print a formatted summary report.

        Args:
            metrics: Metrics dict from calculate_metrics.
        """
        print("\n" + "=" * 60)
        print("BACKTEST REPORT")
        print("=" * 60)
        print(f"Total Trades:    {metrics.get('total_trades', 0)}")
        print(f"Win/Loss:        {metrics.get('winning_trades', 0)}/"
              f"{metrics.get('losing_trades', 0)}")
        print(f"Win Rate:        {metrics.get('win_rate', 0):.1f}%")
        print(f"Profit Factor:   {metrics.get('profit_factor', 0):.2f}")
        print(f"Net PnL:         ${metrics.get('total_net_pnl_usd', 0):.2f}")
        print(f"Return:          {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Max Drawdown:    ${metrics.get('max_drawdown_usd', 0):.2f} "
              f"({metrics.get('max_drawdown_pct', 0):.2f}%)")
        print(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Expectancy (R):  {metrics.get('expectancy_per_trade_r', 0):.2f}")
        print(f"Avg Win (R):     {metrics.get('average_win_r', 0):.2f}")
        print(f"Avg Loss (R):    {metrics.get('average_loss_r', 0):.2f}")
        print(f"Largest Win:     ${metrics.get('largest_win_usd', 0):.2f}")
        print(f"Largest Loss:    ${metrics.get('largest_loss_usd', 0):.2f}")
        print(f"Avg Duration:    {metrics.get('average_trade_duration_hours', 0):.1f}h")
        print(f"Expired:         {metrics.get('expired_trades', 0)}")
        print("=" * 60)

    def _sim_spread(self, timestamp) -> float:
        """Simulate realistic spread based on time of day."""
        try:
            hour = timestamp.hour
        except Exception:
            return 1.5

        if hour in (0, 1, 2, 3, 4, 5):
            return 2.0 + random.uniform(0, 1.0)
        elif hour in (7, 8, 12, 13):
            return 0.8 + random.uniform(0, 0.4)
        elif hour in (20, 21, 22, 23):
            return 1.8 + random.uniform(0, 0.8)
        else:
            return 1.0 + random.uniform(0, 0.5)

    def _pips_to_price(self, instrument: str, pips: float) -> float:
        """Convert pips to price units."""
        if "JPY" in instrument:
            return pips / 100
        elif instrument == "XAU_USD":
            return pips / 10
        else:
            return pips / 10000

    def _price_to_pips(self, instrument: str, price_diff: float) -> float:
        """Convert price difference to pips."""
        if "JPY" in instrument:
            return price_diff * 100
        elif instrument == "XAU_USD":
            return price_diff * 10
        else:
            return price_diff * 10000
