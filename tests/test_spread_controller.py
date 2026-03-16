"""Unit tests for the SpreadController module."""

import unittest
from unittest.mock import MagicMock

from src.spread_controller import SpreadController
from src.logger import BotLogger


def _make_config():
    """Return a minimal config dict for testing."""
    return {
        "spread": {
            "max_pips": {
                "EUR_USD": 1.5,
                "GBP_USD": 2.0,
                "GBP_JPY": 3.0,
                "XAU_USD": 35.0,
            },
            "spike_multiplier": 3.0,
            "rolling_average_period": 20,
        },
        "slippage": {
            "max_pips": {
                "majors": 1.5,
                "exotics": 3.0,
            },
            "partial_fill_min_pct": 0.80,
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


class TestSpreadController(unittest.TestCase):
    """Tests for SpreadController class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = _make_config()
        self.logger = BotLogger(self.config)
        self.mock_data_engine = MagicMock()
        self.sc = SpreadController(self.config, self.logger, self.mock_data_engine)

    def test_check_spread_false_exceeds_max(self):
        """Test check_spread returns False when spread exceeds max_pips."""
        self.mock_data_engine.get_current_spread.return_value = 2.5  # > 1.5 max
        ok, spread, reason = self.sc.check_spread("EUR_USD")
        self.assertFalse(ok)
        self.assertEqual(reason, "SPREAD_TOO_HIGH")
        self.assertAlmostEqual(spread, 2.5)

    def test_check_spread_false_spike(self):
        """Test check_spread returns False when spread is 3x the rolling average."""
        # Fill history with 19 normal readings so adding one more won't dilute much
        history = self.sc._get_history("GBP_USD")
        for _ in range(19):
            history.append(0.3)
        # Rolling avg ≈ 0.3, spike threshold = 0.3 * 3.0 = 0.9
        # New spread = 1.8, new avg ≈ (0.3*19+1.8)/20 = 0.375, threshold = 1.125
        # 1.8 > 1.125 => SPIKE, and 1.8 < max_pips 2.0
        self.mock_data_engine.get_current_spread.return_value = 1.8
        ok, spread, reason = self.sc.check_spread("GBP_USD")
        self.assertFalse(ok)
        self.assertEqual(reason, "SPREAD_SPIKE")

    def test_check_spread_true_normal(self):
        """Test check_spread returns True when spread is within normal range."""
        self.mock_data_engine.get_current_spread.return_value = 1.0  # < 1.5 max
        ok, spread, reason = self.sc.check_spread("EUR_USD")
        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertAlmostEqual(spread, 1.0)

    def test_calculate_spread_adjusted_rr(self):
        """Test that spread-adjusted RR is lower than original for LONG."""
        self.mock_data_engine.get_current_spread.return_value = 1.5  # 1.5 pips
        # Entry=1.2000, SL=1.1980, TP=1.2060 => original risk=20 pips, reward=60 pips, RR=3.0
        # Spread = 1.5 pips = 0.00015 price
        # Effective entry = 1.20015
        # Adjusted risk = |1.20015 - 1.198| = 0.00215 => 21.5 pips
        # Adjusted reward = |1.206 - 1.20015| = 0.00585 => 58.5 pips
        # Adjusted RR = 58.5 / 21.5 ≈ 2.72
        rr = self.sc.calculate_spread_adjusted_rr(
            entry=1.2000, stop_loss=1.1980, take_profit=1.2060,
            instrument="EUR_USD", direction="LONG"
        )
        self.assertLess(rr, 3.0)  # Must be reduced from original 3.0
        self.assertGreater(rr, 2.5)

    def test_check_slippage_false_exceeds_max(self):
        """Test check_slippage returns False when slippage exceeds max for majors."""
        # 2 pips slippage for EUR_USD (major, max 1.5 pips)
        ok, slip = self.sc.check_slippage("EUR_USD", 1.20000, 1.20020, "LONG")
        self.assertFalse(ok)
        self.assertAlmostEqual(slip, 2.0)

    def test_check_slippage_true_within_limit(self):
        """Test check_slippage returns True when slippage is within limit."""
        ok, slip = self.sc.check_slippage("EUR_USD", 1.20000, 1.20005, "LONG")
        self.assertTrue(ok)
        self.assertAlmostEqual(slip, 0.5)

    def test_handle_partial_fill_reject(self):
        """Test handle_partial_fill returns REJECT when fill < 80%."""
        decision, pct = self.sc.handle_partial_fill(100000, 70000)
        self.assertEqual(decision, "REJECT")
        self.assertAlmostEqual(pct, 0.7)

    def test_handle_partial_fill_accept(self):
        """Test handle_partial_fill returns ACCEPT when fill >= 80%."""
        decision, pct = self.sc.handle_partial_fill(100000, 85000)
        self.assertEqual(decision, "ACCEPT")
        self.assertAlmostEqual(pct, 0.85)

    def test_pips_to_price_standard(self):
        """Test pips_to_price for standard pair."""
        self.assertAlmostEqual(self.sc.pips_to_price("EUR_USD", 1.5), 0.00015)

    def test_pips_to_price_jpy(self):
        """Test pips_to_price for JPY pair."""
        self.assertAlmostEqual(self.sc.pips_to_price("GBP_JPY", 3.0), 0.03)

    def test_pips_to_price_xau(self):
        """Test pips_to_price for XAU_USD."""
        self.assertAlmostEqual(self.sc.pips_to_price("XAU_USD", 10.0), 1.0)

    def test_price_to_pips_standard(self):
        """Test price_to_pips for standard pair."""
        self.assertAlmostEqual(self.sc.price_to_pips("EUR_USD", 0.00015), 1.5)

    def test_get_rolling_average_none_few_readings(self):
        """Test get_rolling_average_spread returns None with < 5 readings."""
        self.sc._get_history("EUR_USD").append(1.0)
        result = self.sc.get_rolling_average_spread("EUR_USD")
        self.assertIsNone(result)

    def test_check_slippage_exotic(self):
        """Test check_slippage uses exotic threshold for non-major pairs."""
        # XAU_USD is exotic, max 3.0 pips
        # price_diff = 0.2, XAU pips = 0.2 * 10 = 2.0 pips < 3.0
        ok, slip = self.sc.check_slippage("XAU_USD", 2000.00, 2000.20, "LONG")
        self.assertTrue(ok)
        self.assertAlmostEqual(slip, 2.0)


if __name__ == "__main__":
    unittest.main()
