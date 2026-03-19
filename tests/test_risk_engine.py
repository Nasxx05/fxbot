"""Unit tests for the RiskEngine module (MetaTrader 5)."""

import unittest
from unittest.mock import MagicMock, patch

from src.risk_engine import RiskEngine
from src.logger import BotLogger


def _make_config():
    return {
        "risk": {
            "per_trade_pct": 0.01,
            "daily_max_pct": 0.03,
            "weekly_max_pct": 0.08,
            "monthly_max_pct": 0.15,
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        self.config = _make_config()
        self.logger = BotLogger(self.config)
        self.data_engine = MagicMock()
        self.re = RiskEngine(self.config, self.logger, self.data_engine)

    def test_position_size_within_limits(self):
        """Position size must be between 0.01 and 10.0 lots."""
        size = self.re.calculate_position_size("EUR_USD", 1.2000, 1.1950, 10000.0)
        self.assertGreaterEqual(size, 0.01)
        self.assertLessEqual(size, 10.0)

    def test_position_size_rounds_down(self):
        """Position size must round DOWN, never up."""
        # 10000 * 0.01 = 100 risk. 50 pips * 10 pip_value = 500. 100/500 = 0.20
        size = self.re.calculate_position_size("EUR_USD", 1.2000, 1.1950, 10000.0)
        self.assertEqual(size, 0.20)

        # With a balance that would give a fractional lot
        # 15000 * 0.01 = 150. 50 pips * 10 = 500. 150/500 = 0.30
        size2 = self.re.calculate_position_size("EUR_USD", 1.2000, 1.1950, 15000.0)
        self.assertEqual(size2, 0.30)

        # 12345 * 0.01 = 123.45. 50 * 10 = 500. 123.45/500 = 0.2469 -> floor to 0.24
        size3 = self.re.calculate_position_size("EUR_USD", 1.2000, 1.1950, 12345.0)
        self.assertEqual(size3, 0.24)

    def test_check_daily_limit_false(self):
        """Daily limit returns False when loss exceeds 3%."""
        allowed, reason = self.re.check_daily_limit(-350.0, 10000.0)
        self.assertFalse(allowed)
        self.assertEqual(reason, "DAILY_LIMIT_HIT")

    def test_check_daily_limit_true(self):
        """Daily limit returns True when loss is within limit."""
        allowed, reason = self.re.check_daily_limit(-100.0, 10000.0)
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_check_weekly_limit_false(self):
        """Weekly limit returns False when loss exceeds 8%."""
        allowed, reason = self.re.check_weekly_limit(-850.0, 10000.0)
        self.assertFalse(allowed)
        self.assertEqual(reason, "WEEKLY_LIMIT_HIT")

    def test_check_weekly_limit_true(self):
        """Weekly limit returns True when loss is within limit."""
        allowed, reason = self.re.check_weekly_limit(-500.0, 10000.0)
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_check_monthly_limit_false(self):
        """Monthly limit returns False when loss exceeds 15%."""
        allowed, reason = self.re.check_monthly_limit(-1600.0, 10000.0)
        self.assertFalse(allowed)
        self.assertEqual(reason, "MONTHLY_LIMIT_HIT")

    @patch("src.risk_engine.mt5")
    def test_get_account_balance_from_mt5(self, mock_mt5):
        """get_account_balance fetches from MT5 account_info."""
        mock_account = MagicMock()
        mock_account.balance = 50000.0
        mock_mt5.account_info.return_value = mock_account

        balance = self.re.get_account_balance()
        self.assertEqual(balance, 50000.0)

    @patch("src.risk_engine.mt5")
    def test_get_account_balance_uses_cache(self, mock_mt5):
        """get_account_balance returns cached value within TTL."""
        self.re._balance_cache = 25000.0
        self.re._balance_cache_time = 9999999999  # Far future

        balance = self.re.get_account_balance()
        self.assertEqual(balance, 25000.0)
        mock_mt5.account_info.assert_not_called()

    @patch("src.risk_engine.mt5")
    def test_get_account_balance_handles_none(self, mock_mt5):
        """get_account_balance returns 0.0 when MT5 returns None."""
        mock_mt5.account_info.return_value = None
        self.re._balance_cache = None

        balance = self.re.get_account_balance()
        self.assertEqual(balance, 0.0)

    def test_attach_position_size_to_signal(self):
        """attach_position_size_to_signal fills in position_size."""
        self.re._balance_cache = 10000.0
        self.re._balance_cache_time = 9999999999
        signal = {
            "instrument": "EUR_USD",
            "entry_price": 1.2000,
            "stop_loss": 1.1950,
        }
        result = self.re.attach_position_size_to_signal(signal)
        self.assertIsNotNone(result)
        self.assertIn("position_size", result)
        self.assertGreater(result["position_size"], 0)

    def test_position_size_jpy_pair(self):
        """Position size calculates correctly for JPY pair."""
        size = self.re.calculate_position_size("USD_JPY", 150.00, 149.50, 10000.0)
        self.assertGreaterEqual(size, 0.01)
        self.assertLessEqual(size, 10.0)

    def test_position_size_max_cap(self):
        """Position size caps at 10 lots."""
        size = self.re.calculate_position_size("EUR_USD", 1.2000, 1.1999, 5000000.0)
        self.assertEqual(size, 10.0)


if __name__ == "__main__":
    unittest.main()
