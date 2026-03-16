"""Unit tests for the StructureEngine module."""

import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.structure_engine import StructureEngine
from src.logger import BotLogger


def _make_config():
    """Return a minimal config dict for testing."""
    return {
        "strategy": {
            "pullback_candle_limit": 5,
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


class TestStructureEngine(unittest.TestCase):
    """Tests for StructureEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = _make_config()
        self.logger = BotLogger(self.config)
        self.engine = StructureEngine(self.config, self.logger)

    def _make_bullish_bos_df(self):
        """Create a DataFrame with a clear bullish BOS pattern.

        The sweep is at index 5 (sweep of lows), and after that there's a
        swing high at index 10 (with pivot lookback=2), then a candle at
        index 15 that closes above the swing high.
        """
        n = 20
        # Base prices trending up after a dip
        prices = [1.20] * n
        highs = [1.21] * n
        lows = [1.19] * n
        closes = [1.20] * n

        # Swing high at index 10: high is higher than neighbors
        highs[10] = 1.2200
        closes[10] = 1.2150

        # Mark it as a swing high
        is_swing_high = [False] * n
        is_swing_high[10] = True
        swing_high_price = [np.nan] * n
        swing_high_price[10] = 1.2200

        is_swing_low = [False] * n
        swing_low_price = [np.nan] * n

        # BOS candle at index 15: closes above 1.22
        closes[15] = 1.2250
        highs[15] = 1.2260

        # Non-BOS candle at index 12: wicks above but doesn't close above
        highs[12] = 1.2230
        closes[12] = 1.2180

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": prices,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100] * n,
            "is_swing_high": is_swing_high,
            "swing_high_price": swing_high_price,
            "is_swing_low": is_swing_low,
            "swing_low_price": swing_low_price,
            "atr_14": [0.005] * n,
        })
        return df

    def test_detect_bos_returns_none_wick_only(self):
        """Test that BOS returns None when candle only wicks through (no close)."""
        df = self._make_bullish_bos_df()
        sweep = {
            "direction": "BULLISH_SWEEP",
            "sweep_candle_index": 5,
            "sweep_high": 1.21,
            "sweep_low": 1.18,
            "timestamp": datetime.now(timezone.utc),
            "zone": {},
        }

        # Modify so no candle actually closes above swing high 1.22
        df.loc[15, "close"] = 1.2180  # below 1.22
        df.loc[15, "high"] = 1.2230  # wick above, but close below

        result = self.engine.detect_break_of_structure(df, sweep)
        self.assertIsNone(result)

    def test_detect_bos_returns_valid_dict_on_close(self):
        """Test that BOS returns valid dict when candle closes beyond the level."""
        df = self._make_bullish_bos_df()
        sweep = {
            "direction": "BULLISH_SWEEP",
            "sweep_candle_index": 5,
            "sweep_high": 1.21,
            "sweep_low": 1.18,
            "timestamp": datetime.now(timezone.utc),
            "zone": {},
        }

        result = self.engine.detect_break_of_structure(df, sweep)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BULLISH")
        self.assertAlmostEqual(result["bos_level"], 1.2200)
        self.assertEqual(result["bos_candle_index"], 15)
        self.assertIn("sweep_reference", result)
        self.assertIn("timestamp", result)

    def test_detect_pullback_zone_retrace_50(self):
        """Test detect_pullback_zone returns valid zone using 50% retrace."""
        df = self._make_bullish_bos_df()
        sweep = {
            "direction": "BULLISH_SWEEP",
            "sweep_candle_index": 5,
            "sweep_high": 1.2100,
            "sweep_low": 1.1800,
            "timestamp": datetime.now(timezone.utc),
            "zone": {},
        }
        bos = {
            "direction": "BULLISH",
            "bos_level": 1.2200,
            "bos_candle_index": 15,
            "bos_candle_close": 1.2250,
            "sweep_reference": sweep,
            "timestamp": datetime.now(timezone.utc),
        }

        result = self.engine.detect_pullback_zone(df, sweep, bos)
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "RETRACE_50")
        self.assertIn("entry_price", result)
        self.assertIn("zone_high", result)
        self.assertIn("zone_low", result)
        self.assertIn("expiry_candles", result)
        # For bullish: zone_low = 1.21 - 0.03*0.5 = 1.195, zone_high = 1.21 - 0.03*0.3 = 1.201
        self.assertGreater(result["zone_high"], result["zone_low"])

    def test_detect_pullback_zone_fvg(self):
        """Test detect_pullback_zone detects a Fair Value Gap in a known sequence."""
        # Bullish FVG: candle[N-1].high < candle[N+1].low
        n = 20
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": [1.20] * n,
            "high": [1.21] * n,
            "low": [1.19] * n,
            "close": [1.20] * n,
            "volume": [100] * n,
        })
        # Create FVG around index 10 (BOS candle)
        # candle 9: high = 1.2050
        # candle 11: low = 1.2100
        # Gap from 1.2050 to 1.2100
        df.loc[9, "high"] = 1.2050
        df.loc[11, "low"] = 1.2100

        sweep = {
            "direction": "BULLISH_SWEEP",
            "sweep_candle_index": 5,
            "sweep_high": 1.21,
            "sweep_low": 1.18,
            "timestamp": datetime.now(timezone.utc),
            "zone": {},
        }
        bos = {
            "direction": "BULLISH",
            "bos_level": 1.2200,
            "bos_candle_index": 10,
            "bos_candle_close": 1.2250,
            "sweep_reference": sweep,
            "timestamp": datetime.now(timezone.utc),
        }

        result = self.engine.detect_pullback_zone(df, sweep, bos)
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "FVG")
        self.assertAlmostEqual(result["zone_low"], 1.2050)
        self.assertAlmostEqual(result["zone_high"], 1.2100)

    def test_check_setup_expired_after_candle_limit(self):
        """Test check_setup_expired returns True after pullback_candle_limit exceeded."""
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": [1.20] * n,
            "high": [1.21] * n,
            "low": [1.19] * n,
            "close": [1.20] * n,
            "volume": [100] * n,
            "atr_14": [0.005] * n,
        })
        setup = {
            "entry_price": 1.2050,
            "zone_high": 1.2060,
            "zone_low": 1.2040,
            "method": "RETRACE_50",
            "expiry_candles": 10,  # expires at index 10
        }
        # Current index is 29, well past expiry of 10
        self.assertTrue(self.engine.check_setup_expired(df, setup))

    def test_check_setup_not_expired(self):
        """Test check_setup_expired returns False when within limits."""
        n = 5
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": [1.2050] * n,
            "high": [1.2060] * n,
            "low": [1.2040] * n,
            "close": [1.2050] * n,
            "volume": [100] * n,
            "atr_14": [0.010] * n,
        })
        setup = {
            "entry_price": 1.2050,
            "zone_high": 1.2060,
            "zone_low": 1.2040,
            "method": "RETRACE_50",
            "expiry_candles": 100,
        }
        self.assertFalse(self.engine.check_setup_expired(df, setup))

    def test_get_last_swing_high(self):
        """Test get_last_swing_high returns correct price."""
        df = self._make_bullish_bos_df()
        result = self.engine.get_last_swing_high(df, after_index=5)
        self.assertAlmostEqual(result, 1.2200)

    def test_get_last_swing_low_none_when_missing(self):
        """Test get_last_swing_low returns None when no swing low exists."""
        df = self._make_bullish_bos_df()
        result = self.engine.get_last_swing_low(df, after_index=5)
        self.assertIsNone(result)

    def test_detect_bos_bearish(self):
        """Test bearish BOS detection."""
        n = 20
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": [1.20] * n,
            "high": [1.21] * n,
            "low": [1.19] * n,
            "close": [1.20] * n,
            "volume": [100] * n,
            "is_swing_high": [False] * n,
            "swing_high_price": [np.nan] * n,
            "is_swing_low": [False] * n,
            "swing_low_price": [np.nan] * n,
            "atr_14": [0.005] * n,
        })
        # Swing low at index 10
        df.loc[10, "is_swing_low"] = True
        df.loc[10, "swing_low_price"] = 1.1800
        df.loc[10, "low"] = 1.1800

        # BOS candle at index 15 closes below 1.18
        df.loc[15, "close"] = 1.1750

        sweep = {
            "direction": "BEARISH_SWEEP",
            "sweep_candle_index": 5,
            "sweep_high": 1.22,
            "sweep_low": 1.19,
            "timestamp": datetime.now(timezone.utc),
            "zone": {},
        }

        result = self.engine.detect_break_of_structure(df, sweep)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "BEARISH")


if __name__ == "__main__":
    unittest.main()
