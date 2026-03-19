"""Unit tests for the TradeManager module (MetaTrader 5)."""

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.trade_manager import TradeManager
from src.logger import BotLogger


def _make_config():
    return {
        "trade_management": {
            "breakeven_trigger_r": 1.0,
            "partial_close_trigger_r": 2.0,
            "partial_close_pct": 0.50,
            "full_close_trigger_r": 3.0,
            "trailing_stop_atr_multiple": 1.0,
            "max_trade_duration_hours": 24,
            "volatility_collapse_atr_multiple": 0.7,
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


def _make_trade(direction="LONG", entry=1.2000, sl=1.1950, tp1=1.2100,
                tp2=1.2150, size=0.50, trade_id="T001"):
    return {
        "trade_id": trade_id,
        "instrument": "EUR_USD",
        "direction": direction,
        "entry_price": entry,
        "intended_entry": entry,
        "stop_loss": sl,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "position_size": size,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "status": "OPEN",
        "breakeven_moved": False,
        "partial_closed": False,
        "r_multiple_current": 0.0,
        "signal_reference": {"stop_loss": sl},
    }


def _make_df():
    n = 20
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": [1.20] * n,
        "high": [1.21] * n,
        "low": [1.19] * n,
        "close": [1.20] * n,
        "volume": [100] * n,
        "atr_14": [0.005] * n,
        "atr_average": [0.004] * n,
    })


class TestTradeManager(unittest.TestCase):

    def setUp(self):
        self.config = _make_config()
        self.logger = BotLogger(self.config)
        self.execution_engine = MagicMock()
        self.execution_engine.data_engine = MagicMock()
        self.execution_engine.data_engine.get_current_spread.return_value = 1.0
        self.execution_engine.spread_controller = MagicMock()
        self.execution_engine.spread_controller.pips_to_price.return_value = 0.0001
        self.execution_engine.close_trade_at_market.return_value = True
        self.execution_engine.modify_trade_sl.return_value = True

        self.volatility_engine = MagicMock()
        self.volatility_engine.is_volatility_collapsing.return_value = False
        self.volatility_engine.get_atr_value.return_value = 0.005

        # Clean up persisted files
        for f in ["data/open_trades.json", "data/trade_history.json"]:
            if os.path.exists(f):
                os.remove(f)

        self.tm = TradeManager(self.config, self.logger,
                               self.execution_engine, self.volatility_engine)

    def test_breakeven_at_1r(self):
        """manage_trade moves SL to breakeven at 1R profit."""
        trade = _make_trade(direction="LONG", entry=1.2000, sl=1.1950)
        self.tm.open_trades[trade["trade_id"]] = trade

        # Price at 1R = entry + risk = 1.2000 + 0.0050 = 1.2050
        self.tm.manage_trade(trade, 1.2050, _make_df())

        self.assertTrue(trade["breakeven_moved"])

    def test_breakeven_not_moved_twice(self):
        """manage_trade does NOT move SL to breakeven a second time."""
        trade = _make_trade(direction="LONG", entry=1.2000, sl=1.1950)
        trade["breakeven_moved"] = True
        self.tm.open_trades[trade["trade_id"]] = trade

        self.volatility_engine.get_atr_value.return_value = 0.005
        self.tm.manage_trade(trade, 1.2055, _make_df())

        self.assertTrue(trade["breakeven_moved"])

    def test_partial_close_at_2r(self):
        """manage_trade closes 50% at 2R."""
        trade = _make_trade(direction="LONG", entry=1.2000, sl=1.1950)
        self.tm.open_trades[trade["trade_id"]] = trade

        # Price at 2R = 1.2000 + 0.0100 = 1.2100
        self.tm.manage_trade(trade, 1.2100, _make_df())

        self.assertTrue(trade["partial_closed"])

    def test_full_close_at_3r(self):
        """manage_trade closes fully at 3R."""
        trade = _make_trade(direction="LONG", entry=1.2000, sl=1.1950)
        self.tm.open_trades[trade["trade_id"]] = trade

        # Price at 3R = 1.2000 + 0.0150 = 1.2150
        self.tm.manage_trade(trade, 1.2150, _make_df())

        self.execution_engine.close_trade_at_market.assert_called_once_with(
            trade["trade_id"], trade["instrument"],
            trade["direction"], trade["position_size"],
            "TP2_HIT")

    def test_close_on_volatility_collapse(self):
        """manage_trade closes trade on volatility collapse."""
        trade = _make_trade(direction="LONG", entry=1.2000, sl=1.1950)
        self.tm.open_trades[trade["trade_id"]] = trade
        self.volatility_engine.is_volatility_collapsing.return_value = True

        self.tm.manage_trade(trade, 1.2010, _make_df())

        self.execution_engine.close_trade_at_market.assert_called_once_with(
            trade["trade_id"], trade["instrument"],
            trade["direction"], trade["position_size"],
            "VOLATILITY_COLLAPSE")

    def test_close_on_max_duration(self):
        """manage_trade closes trade on max duration exceeded."""
        trade = _make_trade(direction="LONG", entry=1.2000, sl=1.1950)
        trade["entry_time"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        self.tm.open_trades[trade["trade_id"]] = trade

        self.tm.manage_trade(trade, 1.2010, _make_df())

        self.execution_engine.close_trade_at_market.assert_called_once_with(
            trade["trade_id"], trade["instrument"],
            trade["direction"], trade["position_size"],
            "MAX_DURATION_EXCEEDED")

    def test_sl_modification_never_worsens_long(self):
        """SL modification is rejected if it would worsen LONG SL."""
        trade = _make_trade(direction="LONG", entry=1.2000, sl=1.2010)
        self.tm.open_trades[trade["trade_id"]] = trade

        # Try to move SL lower (worse for LONG)
        self.tm.modify_trade_sl(trade, 1.2005)

        # Should NOT call execution engine
        self.execution_engine.modify_trade_sl.assert_not_called()

    def test_sl_modification_allowed_when_better(self):
        """SL modification succeeds when moving in profitable direction."""
        trade = _make_trade(direction="LONG", entry=1.2000, sl=1.1950)
        self.tm.open_trades[trade["trade_id"]] = trade

        self.tm.modify_trade_sl(trade, 1.1980)

        self.execution_engine.modify_trade_sl.assert_called_once()

    def test_register_trade_persists(self):
        """register_trade saves to open_trades.json."""
        trade = _make_trade()
        self.tm.register_trade(trade)

        self.assertIn(trade["trade_id"], self.tm.open_trades)
        self.assertTrue(os.path.exists("data/open_trades.json"))


if __name__ == "__main__":
    unittest.main()
