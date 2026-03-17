"""Unit tests for the ExecutionEngine module."""

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
        self.data_engine.base_url = "https://api-fxpractice.oanda.com"
        self.data_engine.account_id = "TEST"
        self.data_engine.headers = {}
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

    def test_execute_signal_fill_timeout(self):
        """execute_signal returns None when fill timeout is exceeded."""
        self.spread_controller.check_spread.return_value = (True, 1.0, None)
        self.risk_engine.attach_position_size_to_signal.return_value = _make_signal()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"orderCreateTransaction": {"id": "123"}}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.execution_engine.requests.post", return_value=mock_resp):
            # Override fill timeout to 0 so it times out immediately
            self.ee.fill_timeout = 0
            cancel_resp = MagicMock()
            cancel_resp.raise_for_status = MagicMock()
            with patch("src.execution_engine.requests.put", return_value=cancel_resp):
                result = self.ee.execute_signal(_make_signal())

        self.assertIsNone(result)

    def test_execute_signal_slippage_reject(self):
        """execute_signal returns None when slippage exceeds threshold."""
        self.spread_controller.check_spread.return_value = (True, 1.0, None)
        self.risk_engine.attach_position_size_to_signal.return_value = _make_signal()
        self.spread_controller.check_slippage.return_value = (False, 5.0)
        self.spread_controller.handle_partial_fill.return_value = ("ACCEPT", 1.0)

        # Mock submit and fill
        mock_post = MagicMock()
        mock_post.json.return_value = {"orderCreateTransaction": {"id": "123"}}
        mock_post.raise_for_status = MagicMock()

        mock_get = MagicMock()
        mock_get.json.return_value = {"order": {
            "state": "FILLED", "price": "1.2110",
            "fillingTransactionID": "124", "tradeOpenedID": "125",
            "filledTime": "2024-01-15T10:00:01Z"
        }}
        mock_get.raise_for_status = MagicMock()

        mock_put = MagicMock()
        mock_put.raise_for_status = MagicMock()

        with patch("src.execution_engine.requests.post", return_value=mock_post):
            with patch("src.execution_engine.requests.get", return_value=mock_get):
                with patch("src.execution_engine.requests.put", return_value=mock_put):
                    result = self.ee.execute_signal(_make_signal())

        self.assertIsNone(result)

    def test_execute_signal_partial_fill_reject(self):
        """execute_signal returns None when partial fill is below 80%."""
        self.spread_controller.check_spread.return_value = (True, 1.0, None)
        self.risk_engine.attach_position_size_to_signal.return_value = _make_signal()
        self.spread_controller.check_slippage.return_value = (True, 0.5)
        self.spread_controller.handle_partial_fill.return_value = ("REJECT", 0.50)

        mock_post = MagicMock()
        mock_post.json.return_value = {"orderCreateTransaction": {"id": "123"}}
        mock_post.raise_for_status = MagicMock()

        mock_get = MagicMock()
        mock_get.json.return_value = {"order": {
            "state": "FILLED", "price": "1.2100",
            "fillingTransactionID": "124", "tradeOpenedID": "125",
            "filledTime": "2024-01-15T10:00:01Z"
        }}
        mock_get.raise_for_status = MagicMock()

        mock_put = MagicMock()
        mock_put.raise_for_status = MagicMock()

        with patch("src.execution_engine.requests.post", return_value=mock_post):
            with patch("src.execution_engine.requests.get", return_value=mock_get):
                with patch("src.execution_engine.requests.put", return_value=mock_put):
                    result = self.ee.execute_signal(_make_signal())

        self.assertIsNone(result)

    def test_build_confirmed_trade_has_all_keys(self):
        """build_confirmed_trade returns dict with all required keys."""
        signal = _make_signal()
        trade = self.ee.build_confirmed_trade(signal, 1.2100, 0.50, "OANDA_123")

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

    def test_cancel_order_success(self):
        """cancel_order returns True on success."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch("src.execution_engine.requests.put", return_value=mock_resp):
            result = self.ee.cancel_order("123")
        self.assertTrue(result)

    def test_close_trade_at_market_success(self):
        """close_trade_at_market returns True on success."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch("src.execution_engine.requests.put", return_value=mock_resp):
            result = self.ee.close_trade_at_market("123", "TEST")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
