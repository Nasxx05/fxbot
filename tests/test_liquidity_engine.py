"""Unit tests for the LiquidityEngine module."""

import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.liquidity_engine import LiquidityEngine
from src.logger import BotLogger


def _make_config():
    """Return a minimal config dict for testing."""
    return {
        "strategy": {
            "sweep_wick_body_ratio": 1.5,
            "sweep_candle_lookback": 3,
            "liquidity_threshold_pct": 0.02,
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


def _make_zone(zone_type="EQUAL_HIGH", price=1.2100, active=True):
    """Create a test zone dict."""
    return {
        "type": zone_type,
        "price": price,
        "strength": 2,
        "created_at": datetime.now(timezone.utc),
        "active": active,
    }


class TestLiquidityEngine(unittest.TestCase):
    """Tests for LiquidityEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = _make_config()
        self.logger = BotLogger(self.config)
        self.engine = LiquidityEngine(self.config, self.logger)

    def test_detect_sweep_returns_none_body_closes_beyond(self):
        """Test detect_sweep returns None when candle body closes beyond the zone."""
        zone = _make_zone("EQUAL_HIGH", price=1.2100)
        # Candle wick goes above 1.21, but body ALSO closes above — not a rejection
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "open":  [1.2050, 1.2060, 1.2080],
            "high":  [1.2070, 1.2080, 1.2150],
            "low":   [1.2040, 1.2050, 1.2070],
            "close": [1.2060, 1.2070, 1.2130],  # close is ABOVE 1.21 — not back inside
            "volume": [100, 100, 100],
        })
        result = self.engine.detect_sweep(df, zone)
        self.assertIsNone(result)

    def test_detect_sweep_returns_valid_dict_all_conditions_met(self):
        """Test detect_sweep returns a valid sweep dict when all 4 conditions are met."""
        zone = _make_zone("EQUAL_HIGH", price=1.2100)
        # Candle wicks above 1.21, closes back below, big wick relative to body
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "open":  [1.2050, 1.2060, 1.2090],
            "high":  [1.2070, 1.2080, 1.2180],  # wick to 1.218, way above 1.21
            "low":   [1.2040, 1.2050, 1.2070],
            "close": [1.2060, 1.2070, 1.2080],  # close at 1.208, back below 1.21
            "volume": [100, 100, 100],
        })
        # body = |1.209 - 1.208| = 0.001, wick = 1.218 - 1.209 = 0.009
        # wick/body = 9.0 > 1.5 threshold
        result = self.engine.detect_sweep(df, zone)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BEARISH_SWEEP")
        self.assertIn("zone", result)
        self.assertIn("sweep_candle_index", result)
        self.assertIn("sweep_high", result)
        self.assertIn("sweep_low", result)
        self.assertIn("timestamp", result)

    def test_detect_sweep_returns_none_wick_ratio_below_threshold(self):
        """Test detect_sweep returns None when the wick ratio is below the threshold."""
        zone = _make_zone("EQUAL_HIGH", price=1.2100)
        # Candle wick barely above zone, body is large, so wick ratio < 1.5
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "open":  [1.2050, 1.2060, 1.2000],
            "high":  [1.2070, 1.2080, 1.2110],  # wick barely above 1.21
            "low":   [1.2040, 1.2050, 1.1990],
            "close": [1.2060, 1.2070, 1.2090],  # close below 1.21
            "volume": [100, 100, 100],
        })
        # body = |1.200 - 1.209| = 0.009, upper wick = 1.211 - 1.209 = 0.002
        # wick/body = 0.22 < 1.5 — should be rejected
        result = self.engine.detect_sweep(df, zone)
        self.assertIsNone(result)

    def test_get_active_zones_returns_only_active(self):
        """Test that get_active_zones returns only zones where active is True."""
        n = 20
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": [1.20] * n,
            "high": [1.21] * n,
            "low": [1.19] * n,
            "close": [1.20] * n,
            "volume": [100] * n,
            "equal_high_level": [np.nan] * n,
            "equal_low_level": [np.nan] * n,
            "prev_day_high": [1.2200] * n,
            "prev_day_low": [1.1800] * n,
            "asian_session_high": [np.nan] * n,
            "asian_session_low": [np.nan] * n,
            "london_session_high": [np.nan] * n,
            "london_session_low": [np.nan] * n,
        })
        zones = self.engine.get_active_zones(df)
        for z in zones:
            self.assertTrue(z["active"])

    def test_detect_sweep_bullish_sweep_of_lows(self):
        """Test detect_sweep on an EQUAL_LOW zone (bullish sweep)."""
        zone = _make_zone("EQUAL_LOW", price=1.1900)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "open":  [1.1950, 1.1940, 1.1910],
            "high":  [1.1960, 1.1960, 1.1930],
            "low":   [1.1930, 1.1920, 1.1850],  # wick below 1.19
            "close": [1.1940, 1.1930, 1.1920],  # close back above 1.19
            "volume": [100, 100, 100],
        })
        # body = |1.191 - 1.192| = 0.001, lower wick = 1.191 - 1.185 = 0.006
        # wick/body = 6.0 > 1.5
        result = self.engine.detect_sweep(df, zone)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH_SWEEP")

    def test_invalidate_zone_beyond_2_atr(self):
        """Test that invalidate_zone returns True when price is > 2 ATR away."""
        zone = _make_zone("EQUAL_HIGH", price=1.2100)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC"),
            "close": [1.2500, 1.2500, 1.2500, 1.2500, 1.2500],
            "atr_14": [0.005, 0.005, 0.005, 0.005, 0.005],
        })
        # distance = |1.25 - 1.21| = 0.04, 2*ATR = 0.01 => should invalidate
        self.assertTrue(self.engine.invalidate_zone(zone, df))

    def test_invalidate_zone_within_range(self):
        """Test that invalidate_zone returns False when price is within 2 ATR."""
        zone = _make_zone("EQUAL_HIGH", price=1.2100)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC"),
            "close": [1.2105, 1.2105, 1.2105, 1.2105, 1.2105],
            "atr_14": [0.005, 0.005, 0.005, 0.005, 0.005],
        })
        # distance = 0.0005, 2*ATR = 0.01 => within range
        self.assertFalse(self.engine.invalidate_zone(zone, df))

    def test_detect_sweep_inactive_zone_returns_none(self):
        """Test that detect_sweep returns None for an inactive zone."""
        zone = _make_zone("EQUAL_HIGH", price=1.2100, active=False)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "open": [1.20, 1.20, 1.20],
            "high": [1.22, 1.22, 1.22],
            "low": [1.19, 1.19, 1.19],
            "close": [1.20, 1.20, 1.20],
            "volume": [100, 100, 100],
        })
        result = self.engine.detect_sweep(df, zone)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
