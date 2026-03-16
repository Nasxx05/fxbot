"""Unit tests for the CorrelationGuard module."""

import unittest

from src.correlation_guard import CorrelationGuard
from src.logger import BotLogger


def _make_config():
    """Return a config dict with correlation groups."""
    return {
        "correlation_groups": [
            ["EUR_USD", "GBP_USD"],
            ["EUR_USD", "USD_CHF"],
            ["XAU_USD", "AUD_USD"],
        ],
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


class TestCorrelationGuard(unittest.TestCase):
    """Tests for CorrelationGuard class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = _make_config()
        self.logger = BotLogger(self.config)
        self.cg = CorrelationGuard(self.config, self.logger)

    def test_is_correlated_trade_open_true(self):
        """Test returns True when EUR_USD is open and GBP_USD is checked."""
        self.cg.register_open_trade("EUR_USD")
        blocked, blocker = self.cg.is_correlated_trade_open("GBP_USD")
        self.assertTrue(blocked)
        self.assertEqual(blocker, "EUR_USD")

    def test_is_correlated_trade_open_false(self):
        """Test returns False when no correlated trade is open."""
        blocked, blocker = self.cg.is_correlated_trade_open("GBP_USD")
        self.assertFalse(blocked)
        self.assertIsNone(blocker)

    def test_get_best_setup_higher_rr(self):
        """Test returns the setup with the higher RR from two competing setups."""
        setups = [
            {"instrument": "EUR_USD", "risk_reward_adjusted": 2.8,
             "sweep_reference": {"sweep_high": 1.22, "sweep_low": 1.19}},
            {"instrument": "GBP_USD", "risk_reward_adjusted": 3.5,
             "sweep_reference": {"sweep_high": 1.32, "sweep_low": 1.29}},
        ]
        best = self.cg.get_best_setup(setups)
        self.assertEqual(best["instrument"], "GBP_USD")

    def test_register_closed_trade_frees_instrument(self):
        """Test that register_closed_trade correctly frees up the instrument."""
        self.cg.register_open_trade("EUR_USD")
        blocked, _ = self.cg.is_correlated_trade_open("GBP_USD")
        self.assertTrue(blocked)

        self.cg.register_closed_trade("EUR_USD")
        blocked, _ = self.cg.is_correlated_trade_open("GBP_USD")
        self.assertFalse(blocked)

    def test_get_best_setup_empty(self):
        """Test get_best_setup returns None for empty list."""
        self.assertIsNone(self.cg.get_best_setup([]))

    def test_uncorrelated_pair_not_blocked(self):
        """Test that uncorrelated pairs do not block each other."""
        self.cg.register_open_trade("XAU_USD")
        blocked, _ = self.cg.is_correlated_trade_open("EUR_USD")
        self.assertFalse(blocked)

    def test_same_instrument_not_blocked_by_self(self):
        """Test that an instrument is not blocked by its own open trade."""
        self.cg.register_open_trade("EUR_USD")
        blocked, blocker = self.cg.is_correlated_trade_open("EUR_USD")
        # EUR_USD correlates with GBP_USD and USD_CHF, not itself
        self.assertFalse(blocked)


if __name__ == "__main__":
    unittest.main()
