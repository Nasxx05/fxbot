"""Trade manager module — manages open trades, breakeven, partial closes, and trailing stops."""

import json
import os
from datetime import datetime, timezone

import pandas as pd

from src.logger import BotLogger


class TradeManager:
    """Manages the lifecycle of open trades including SL moves and exits."""

    def __init__(self, config: dict, logger: BotLogger, execution_engine,
                 volatility_engine):
        """Initialize the TradeManager.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance.
            execution_engine: ExecutionEngine instance.
            volatility_engine: VolatilityEngine instance.
        """
        self.config = config
        self.logger = logger
        self.execution_engine = execution_engine
        self.volatility_engine = volatility_engine

        tm = config.get("trade_management", {})
        self.breakeven_trigger_r = tm.get("breakeven_trigger_r", 1.0)
        self.partial_close_trigger_r = tm.get("partial_close_trigger_r", 2.0)
        self.partial_close_pct = tm.get("partial_close_pct", 0.50)
        self.full_close_trigger_r = tm.get("full_close_trigger_r", 3.0)
        self.trailing_stop_atr_multiple = tm.get("trailing_stop_atr_multiple", 1.0)
        self.max_trade_duration_hours = tm.get("max_trade_duration_hours", 24)

        self.open_trades = {}
        self.trade_history = []

        self._open_trades_path = os.path.join("data", "open_trades.json")
        self._trade_history_path = os.path.join("data", "trade_history.json")
        os.makedirs("data", exist_ok=True)

        self._load_open_trades()

    def _load_open_trades(self):
        """Load persisted open trades from JSON file."""
        if os.path.exists(self._open_trades_path):
            try:
                with open(self._open_trades_path, "r") as f:
                    trades = json.load(f)
                self.open_trades = {t["trade_id"]: t for t in trades}
                self.logger.log("INFO", "trade_manager",
                                f"Loaded {len(self.open_trades)} open trades from disk")
            except Exception as e:
                self.logger.log_error("trade_manager",
                                      f"Failed to load open trades: {e}")

    def _persist_open_trades(self):
        """Save open trades to JSON file."""
        try:
            with open(self._open_trades_path, "w") as f:
                json.dump(list(self.open_trades.values()), f, indent=2, default=str)
        except Exception as e:
            self.logger.log_error("trade_manager",
                                  f"Failed to persist open trades: {e}")

    def _persist_trade_history(self):
        """Append trade history to JSON file."""
        try:
            existing = []
            if os.path.exists(self._trade_history_path):
                with open(self._trade_history_path, "r") as f:
                    existing = json.load(f)
            existing.extend(self.trade_history)
            with open(self._trade_history_path, "w") as f:
                json.dump(existing, f, indent=2, default=str)
            self.trade_history = []
        except Exception as e:
            self.logger.log_error("trade_manager",
                                  f"Failed to persist trade history: {e}")

    def register_trade(self, trade: dict):
        """Add a trade to open_trades and persist.

        Args:
            trade: Confirmed trade dict from ExecutionEngine.
        """
        self.open_trades[trade["trade_id"]] = trade
        self._persist_open_trades()
        self.logger.log("INFO", "trade_manager", "Trade registered",
                        {"trade_id": trade["trade_id"],
                         "instrument": trade["instrument"]})

    def on_price_update(self, instrument: str, current_price: float,
                        df: pd.DataFrame):
        """Process a price update for all open trades on this instrument.

        Args:
            instrument: Config instrument name.
            current_price: Current market price.
            df: DataFrame with latest candle data.
        """
        for trade_id, trade in list(self.open_trades.items()):
            if trade["instrument"] == instrument and trade["status"] == "OPEN":
                self.manage_trade(trade, current_price, df)

    def manage_trade(self, trade: dict, current_price: float, df: pd.DataFrame):
        """Run all trade management checks on a single trade.

        Args:
            trade: Open trade dict.
            current_price: Current market price.
            df: DataFrame with latest candle data.
        """
        risk = abs(trade["entry_price"] - trade["stop_loss"])
        if risk == 0:
            return

        direction = trade["direction"]
        if direction == "LONG":
            r = (current_price - trade["entry_price"]) / risk
        else:
            r = (trade["entry_price"] - current_price) / risk

        trade["r_multiple_current"] = r

        # Check max duration
        try:
            entry_time = datetime.fromisoformat(trade["entry_time"])
            now = datetime.now(timezone.utc)
            hours_open = (now - entry_time).total_seconds() / 3600
            if hours_open >= self.max_trade_duration_hours:
                self.execution_engine.close_trade_at_market(
                    trade["trade_id"], trade["instrument"],
                    trade["direction"], trade["position_size"],
                    "MAX_DURATION_EXCEEDED")
                self.register_closed_trade(trade["trade_id"], current_price,
                                           "MAX_DURATION_EXCEEDED")
                return
        except (ValueError, TypeError):
            pass

        # Check volatility collapse
        if self.volatility_engine.is_volatility_collapsing(df):
            self.execution_engine.close_trade_at_market(
                trade["trade_id"], trade["instrument"],
                trade["direction"], trade["position_size"],
                "VOLATILITY_COLLAPSE")
            self.register_closed_trade(trade["trade_id"], current_price,
                                       "VOLATILITY_COLLAPSE")
            return

        # Check full close at TP2
        if r >= self.full_close_trigger_r:
            self.execution_engine.close_trade_at_market(
                trade["trade_id"], trade["instrument"],
                trade["direction"], trade["position_size"],
                "TP2_HIT")
            self.register_closed_trade(trade["trade_id"], current_price,
                                       "TP2_HIT")
            return

        # Check partial close at TP1
        if r >= self.partial_close_trigger_r and not trade["partial_closed"]:
            units_to_close = trade["position_size"] * self.partial_close_pct
            self.partial_close(trade, units_to_close)
            trade["partial_closed"] = True
            self._persist_open_trades()
            self.logger.log("INFO", "trade_manager", "Partial close executed",
                            {"trade_id": trade["trade_id"], "r": r,
                             "units_closed": units_to_close})

        # Check breakeven
        if r >= self.breakeven_trigger_r and not trade["breakeven_moved"]:
            spread_buffer = self._get_spread_buffer(trade["instrument"])
            if direction == "LONG":
                new_sl = trade["entry_price"] + spread_buffer
            else:
                new_sl = trade["entry_price"] - spread_buffer
            self.modify_trade_sl(trade, new_sl)
            trade["breakeven_moved"] = True
            trade["stop_loss"] = new_sl
            self._persist_open_trades()
            self.logger.log("INFO", "trade_manager", "Breakeven move executed",
                            {"trade_id": trade["trade_id"], "new_sl": new_sl})

        # Check trailing stop
        if trade["breakeven_moved"] and r > 1.0:
            atr = self.volatility_engine.get_atr_value(df)
            if not pd.isna(atr) and atr > 0:
                if direction == "LONG":
                    new_sl = current_price - (atr * self.trailing_stop_atr_multiple)
                    if new_sl > trade["stop_loss"]:
                        self.modify_trade_sl(trade, new_sl)
                        trade["stop_loss"] = new_sl
                        self._persist_open_trades()
                else:
                    new_sl = current_price + (atr * self.trailing_stop_atr_multiple)
                    if new_sl < trade["stop_loss"]:
                        self.modify_trade_sl(trade, new_sl)
                        trade["stop_loss"] = new_sl
                        self._persist_open_trades()

    def _get_spread_buffer(self, instrument: str) -> float:
        """Get spread buffer in price units for breakeven moves."""
        try:
            spread_pips = self.execution_engine.data_engine.get_current_spread(instrument)
            return self.execution_engine.spread_controller.pips_to_price(
                instrument, spread_pips)
        except Exception:
            return 0.0001

    def register_closed_trade(self, trade_id: str, close_price: float, reason: str):
        """Record a trade closure.

        Args:
            trade_id: MT5 position ticket as string.
            close_price: Price at which the trade was closed.
            reason: Reason for closure.
        """
        trade = self.open_trades.get(trade_id)
        if trade is None:
            return

        trade["status"] = "CLOSED"
        trade["close_price"] = close_price
        trade["close_reason"] = reason
        trade["close_time"] = datetime.now(timezone.utc).isoformat()

        risk = abs(trade["entry_price"] - trade.get("signal_reference", {}).get("stop_loss", trade["stop_loss"]))
        if risk > 0:
            if trade["direction"] == "LONG":
                pnl_price = close_price - trade["entry_price"]
            else:
                pnl_price = trade["entry_price"] - close_price
            trade["final_r_multiple"] = pnl_price / risk
        else:
            trade["final_r_multiple"] = 0.0

        # Calculate PnL in pips
        price_diff = abs(close_price - trade["entry_price"])
        if "JPY" in trade["instrument"]:
            trade["pnl_pips"] = price_diff * 100
        elif trade["instrument"] == "XAU_USD":
            trade["pnl_pips"] = price_diff * 10
        else:
            trade["pnl_pips"] = price_diff * 10000

        if trade.get("final_r_multiple", 0) < 0:
            trade["pnl_pips"] = -trade["pnl_pips"]

        self.logger.log_trade_close(trade)

        self.trade_history.append(trade)
        del self.open_trades[trade_id]
        self._persist_open_trades()
        self._persist_trade_history()

    def modify_trade_sl(self, trade: dict, new_sl: float):
        """Modify the stop loss on an open trade via MT5.

        Only allows SL moves that are strictly better (closer to profit).

        Args:
            trade: Open trade dict.
            new_sl: New stop loss price.
        """
        trade_id = trade["trade_id"]
        current_sl = trade["stop_loss"]
        direction = trade["direction"]

        # Never worsen the SL
        if direction == "LONG" and new_sl < current_sl:
            self.logger.log("WARNING", "trade_manager",
                            "Rejected SL modification: would worsen LONG SL",
                            {"trade_id": trade_id, "current_sl": current_sl,
                             "proposed_sl": new_sl})
            return
        if direction == "SHORT" and new_sl > current_sl:
            self.logger.log("WARNING", "trade_manager",
                            "Rejected SL modification: would worsen SHORT SL",
                            {"trade_id": trade_id, "current_sl": current_sl,
                             "proposed_sl": new_sl})
            return

        current_tp = trade.get("take_profit_2", 0.0)
        self.execution_engine.modify_trade_sl(
            trade_id, trade["instrument"], new_sl, current_tp)

    def partial_close(self, trade: dict, units_to_close: float):
        """Partially close a trade via MT5.

        Args:
            trade: Open trade dict.
            units_to_close: Number of lots to close.
        """
        self.execution_engine.close_trade_at_market(
            trade["trade_id"], trade["instrument"],
            trade["direction"], units_to_close,
            "PARTIAL_CLOSE")

    def get_open_trades_summary(self) -> list:
        """Return a summary of all open trades.

        Returns:
            List of dicts with trade summaries.
        """
        summaries = []
        for trade_id, trade in self.open_trades.items():
            summaries.append({
                "trade_id": trade_id,
                "instrument": trade["instrument"],
                "direction": trade["direction"],
                "entry_price": trade["entry_price"],
                "stop_loss": trade["stop_loss"],
                "r_multiple_current": trade.get("r_multiple_current", 0.0),
                "breakeven_moved": trade.get("breakeven_moved", False),
                "partial_closed": trade.get("partial_closed", False),
                "position_size": trade.get("position_size", 0),
                "entry_time": trade.get("entry_time"),
            })
        return summaries
