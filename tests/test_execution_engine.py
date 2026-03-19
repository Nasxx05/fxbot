"""Unit tests for the ExecutionEngine module (MetaTrader 5)."""

import unittest
from unittest.mock import MagicMock, patch

from src.execution_engine import ExecutionEngine, ExecutionError
from src.logger import BotLogger


def _make_config():
    return {
        "slippage": {
            "fill_confirmation_timeout_seconds": 30,
            "max_pips": {"majors": 1.5, "exotics": 3.0},
            "partial_fill_min_pct": 0.80,
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


def _make_signal():
    return {
        "instrument": "EUR_USD",
        "direction": "SHORT",
        "entry_price": 1.2100,
        "stop_loss": 1.2150,
        "take_profit_1": 1.2000,
        "take_profit_2": 1.1950,
        "position_size": 0.50,
        "risk_reward_raw": 3.0,
        "risk_reward_adjusted": 2.8,
        "sweep_reference": {},
        "bos_reference": {},
        "entry_method": "RETRACE_50",
        "signal_time": "2024-01-15T10:00:00+00:00",
        "current_spread_pips": 1.0,
        "session": "LONDON",
        "atr_at_signal": 0.005,
    }


class TestExecutionEngine(unittest.TestCase):

    def setUp(self):
        self.config = _make_config()
        self.logger = BotLogger(self.config)
        self.data_engine = MagicMock()
        self.data_engine.to_mt5_symbol.side_effect = lambda x: x.replace("_", "")
        self.risk_engine = MagicMock()
        self.spread_controller = MagicMock()

        self.ee = ExecutionEngine(
            self.config, self.logger, self.data_engine,
            self.risk_engine, self.spread_controller)

    def test_execute_signal_blocked_by_spread(self):
        """execute_signal returns None when spread check fails."""
        self.spread_controller.check_spread.return_value = (False, 3.5, "SPREAD_TOO_HIGH")
        result = self.ee.execute_signal(_make_signal())
        self.assertIsNone(result)

    def test_execute_signal_blocked_by_sizing(self):
        """execute_signal returns None when position sizing fails."""
        self.spread_controller.check_spread.return_value = (True, 1.0, None)
        self.risk_engine.attach_position_size_to_signal.return_value = None
        result = self.ee.execute_signal(_make_signal())
        self.assertIsNone(result)

    @patch("src.execution_engine.mt5")
    def test_execute_signal_slippage_reject(self, mock_mt5):
        """execute_signal returns None when slippage exceeds threshold."""
        self.spread_controller.check_spread.return_value = (True, 1.0, None)
        self.risk_engine.attach_position_size_to_signal.return_value = _make_signal()
        self.spread_controller.check_slippage.return_value = (False, 5.0)

        # Mock successful order send
        mock_result = MagicMock()
        mock_result.retcode = 10009  # TRADE_RETCODE_DONE
        mock_result.order = 12345
        mock_result.price = 1.2100
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.TRADE_ACTION_PENDING = 5
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_TIME_GTC = 2
        mock_mt5.ORDER_FILLING_IOC = 1

        # Mock tick for close
        mock_tick = MagicMock()
        mock_tick.bid = 1.2100
        mock_tick.ask = 1.2101
        mock_mt5.symbol_info_tick.return_value = mock_tick

        result = self.ee.execute_signal(_make_signal())
        self.assertIsNone(result)

    @patch("src.execution_engine.mt5")
    def test_execute_signal_partial_fill_reject(self, mock_mt5):
        """execute_signal returns None when partial fill is below 80%."""
        self.spread_controller.check_spread.return_value = (True, 1.0, None)
        self.risk_engine.attach_position_size_to_signal.return_value = _make_signal()
        self.spread_controller.check_slippage.return_value = (True, 0.5)
        self.spread_controller.handle_partial_fill.return_value = ("REJECT", 0.50)

        # Mock successful order send
        mock_result = MagicMock()
        mock_result.retcode = 10009
        mock_result.order = 12345
        mock_result.price = 1.2100
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.TRADE_ACTION_PENDING = 5
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_TIME_GTC = 2
        mock_mt5.ORDER_FILLING_IOC = 1

        # Mock tick for close
        mock_tick = MagicMock()
        mock_tick.bid = 1.2100
        mock_tick.ask = 1.2101
        mock_mt5.symbol_info_tick.return_value = mock_tick

        result = self.ee.execute_signal(_make_signal())
        self.assertIsNone(result)

    def test_build_confirmed_trade_has_all_keys(self):
        """build_confirmed_trade returns dict with all required keys."""
        signal = _make_signal()
        trade = self.ee.build_confirmed_trade(signal, 1.2100, 0.50, "MT5_123")

        required_keys = [
            "trade_id", "instrument", "direction", "entry_price",
            "intended_entry", "stop_loss", "take_profit_1", "take_profit_2",
            "position_size", "entry_time", "status", "breakeven_moved",
            "partial_closed", "r_multiple_current", "signal_reference",
        ]
        for key in required_keys:
            self.assertIn(key, trade, f"Missing key: {key}")

        self.assertEqual(trade["status"], "OPEN")
        self.assertFalse(trade["breakeven_moved"])
        self.assertFalse(trade["partial_closed"])
        self.assertEqual(trade["r_multiple_current"], 0.0)

    @patch("src.execution_engine.mt5")
    def test_cancel_order_success(self, mock_mt5):
        """cancel_order returns True on success."""
        mock_result = MagicMock()
        mock_result.retcode = 10009
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.TRADE_ACTION_REMOVE = 8

        result = self.ee.cancel_order("123")
        self.assertTrue(result)

    @patch("src.execution_engine.mt5")
    def test_cancel_order_failure(self, mock_mt5):
        """cancel_order returns False on failure."""
        mock_result = MagicMock()
        mock_result.retcode = 10013
        mock_result.comment = "Invalid order"
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.TRADE_ACTION_REMOVE = 8

        result = self.ee.cancel_order("123")
        self.assertFalse(result)

    @patch("src.execution_engine.mt5")
    def test_close_trade_at_market_success(self, mock_mt5):
        """close_trade_at_market returns True on success."""
        mock_tick = MagicMock()
        mock_tick.bid = 1.2100
        mock_tick.ask = 1.2101
        mock_mt5.symbol_info_tick.return_value = mock_tick

        mock_result = MagicMock()
        mock_result.retcode = 10009
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_FILLING_IOC = 1

        result = self.ee.close_trade_at_market("123", "EUR_USD", "LONG", 0.10, "TEST")
        self.assertTrue(result)

    @patch("src.execution_engine.mt5")
    def test_modify_trade_sl_success(self, mock_mt5):
        """modify_trade_sl returns True on success."""
        mock_result = MagicMock()
        mock_result.retcode = 10009
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.TRADE_ACTION_SLTP = 6

        result = self.ee.modify_trade_sl("123", "EUR_USD", 1.2050, 1.1950)
        self.assertTrue(result)

    @patch("src.execution_engine.mt5")
    def test_submit_limit_order_retries_on_failure(self, mock_mt5):
        """submit_limit_order raises ExecutionError after 3 failures."""
        mock_result = MagicMock()
        mock_result.retcode = 10013
        mock_result.comment = "Rejected"
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
        mock_mt5.TRADE_ACTION_PENDING = 5
        mock_mt5.ORDER_TIME_GTC = 2
        mock_mt5.ORDER_FILLING_IOC = 1

        with self.assertRaises(ExecutionError):
            self.ee.submit_limit_order("EUR_USD", "LONG", 1.1000, 1.0950, 1.1100, 0.10)

        self.assertEqual(mock_mt5.order_send.call_count, 3)

    @patch("src.execution_engine.mt5")
    def test_submit_market_order_success(self, mock_mt5):
        """submit_market_order returns dict on success."""
        mock_tick = MagicMock()
        mock_tick.ask = 1.1001
        mock_tick.bid = 1.1000
        mock_mt5.symbol_info_tick.return_value = mock_tick

        mock_result = MagicMock()
        mock_result.retcode = 10009
        mock_result.order = 99999
        mock_result.price = 1.1001
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_FILLING_IOC = 1

        result = self.ee.submit_market_order("EUR_USD", "LONG", 0.10, reason="TEST")
        self.assertEqual(result["order_id"], 99999)
        self.assertEqual(result["fill_price"], 1.1001)


if __name__ == "__main__":
    unittest.main()
