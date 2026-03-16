"""Unit tests for the FeatureEngine module."""

import unittest

import numpy as np
import pandas as pd

from src.feature_engine import FeatureEngine


def _make_config():
    """Return a minimal config dict for testing."""
    return {
        "strategy": {
            "atr_period": 14,
            "atr_average_period": 50,
            "pivot_lookback": 2,
            "volatility_multiplier": 1.3,
            "ranging_candle_threshold": 5,
            "liquidity_threshold_pct": 0.02,
        },
        "sessions": {
            "asian_start_utc": "00:00",
            "asian_end_utc": "07:00",
            "london_start_utc": "07:00",
            "london_end_utc": "12:00",
            "ny_start_utc": "12:00",
            "ny_end_utc": "20:00",
            "overlap_start_utc": "12:00",
            "overlap_end_utc": "16:00",
        },
    }


def _make_ohlcv_df(n=100):
    """Create a random OHLCV DataFrame for testing."""
    np.random.seed(42)
    close = 1.2000 + np.cumsum(np.random.randn(n) * 0.001)
    open_ = close + np.random.randn(n) * 0.0005
    high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.0003)
    low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.0003)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    })
    return df


class TestFeatureEngine(unittest.TestCase):
    """Tests for FeatureEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = _make_config()
        self.fe = FeatureEngine(self.config)

    def test_compute_atr_correct_length_no_nulls_after_warmup(self):
        """Test compute_atr returns a Series with correct length and no nulls after warmup."""
        df = _make_ohlcv_df(100)
        result = self.fe.compute_atr(df)

        self.assertIn("atr_14", result.columns)
        self.assertEqual(len(result["atr_14"]), 100)
        # After warmup period (14 candles), there should be no nulls
        after_warmup = result["atr_14"].iloc[14:]
        self.assertFalse(after_warmup.isna().any())

    def test_compute_swing_points_identifies_swing_high(self):
        """Test that compute_swing_points correctly identifies a swing high."""
        # Create a known price sequence with a clear swing high at index 2
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=7, freq="15min", tz="UTC"),
            "open": [1.10, 1.11, 1.15, 1.14, 1.12, 1.11, 1.10],
            "high": [1.11, 1.12, 1.18, 1.15, 1.13, 1.12, 1.11],
            "low":  [1.09, 1.10, 1.14, 1.13, 1.11, 1.10, 1.09],
            "close":[1.10, 1.11, 1.16, 1.14, 1.12, 1.11, 1.10],
            "volume": [100] * 7,
        })

        result = self.fe.compute_swing_points(df)

        self.assertTrue(result["is_swing_high"].iloc[2])
        self.assertAlmostEqual(result["swing_high_price"].iloc[2], 1.18)

    def test_compute_session_tag_london(self):
        """Test that a timestamp at 08:00 UTC is tagged as LONDON."""
        df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01 08:00:00", tz="UTC")],
            "open": [1.10], "high": [1.11], "low": [1.09], "close": [1.10],
            "volume": [100],
        })
        result = self.fe.compute_session_tag(df)
        self.assertEqual(result["session"].iloc[0], "LONDON")

    def test_compute_session_tag_asian(self):
        """Test that a timestamp at 03:00 UTC is tagged as ASIAN."""
        df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01 03:00:00", tz="UTC")],
            "open": [1.10], "high": [1.11], "low": [1.09], "close": [1.10],
            "volume": [100],
        })
        result = self.fe.compute_session_tag(df)
        self.assertEqual(result["session"].iloc[0], "ASIAN")

    def test_compute_equal_levels_detects_equal_highs(self):
        """Test that compute_equal_levels detects two highs within threshold."""
        n = 20
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": [1.20] * n,
            "high": [1.21] * n,
            "low": [1.19] * n,
            "close": [1.20] * n,
            "volume": [100] * n,
        })

        # Manually set swing highs at known positions with nearly equal prices
        df["is_swing_high"] = False
        df["is_swing_low"] = False
        df.loc[5, "is_swing_high"] = True
        df.loc[15, "is_swing_high"] = True
        df["swing_high_price"] = np.nan
        df["swing_low_price"] = np.nan
        df.loc[5, "swing_high_price"] = 1.2100
        df.loc[15, "swing_high_price"] = 1.2102  # Within 2% of 1.20 = 0.024

        result = self.fe.compute_equal_levels(df)
        self.assertTrue(result["near_equal_high"].iloc[-1])

    def test_compute_volatility_regime_ranging(self):
        """Test that RANGING is returned when ATR has been below average for N consecutive candles."""
        n = 20
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": [1.20] * n,
            "high": [1.21] * n,
            "low": [1.19] * n,
            "close": [1.20] * n,
            "volume": [100] * n,
        })
        # ATR below average for all candles
        df["atr_14"] = 0.005
        df["atr_average"] = 0.010

        result = self.fe.compute_volatility_regime(df)
        # After ranging_candle_threshold (5) consecutive candles below, should be RANGING
        self.assertEqual(result["volatility_regime"].iloc[-1], "RANGING")

    def test_compute_market_bias_bullish(self):
        """Test that BULLISH is returned for rising swing highs and lows."""
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": [1.20] * n,
            "high": [1.21] * n,
            "low": [1.19] * n,
            "close": [1.20] * n,
            "volume": [100] * n,
        })

        # Create rising swing highs and lows
        df["swing_high_price"] = np.nan
        df["swing_low_price"] = np.nan
        df.loc[5, "swing_high_price"] = 1.2100
        df.loc[10, "swing_high_price"] = 1.2200
        df.loc[15, "swing_high_price"] = 1.2300
        df.loc[20, "swing_high_price"] = 1.2400

        df.loc[3, "swing_low_price"] = 1.1900
        df.loc[8, "swing_low_price"] = 1.1950
        df.loc[13, "swing_low_price"] = 1.2000
        df.loc[18, "swing_low_price"] = 1.2050

        result = self.fe.compute_market_bias(df)
        self.assertEqual(result["market_bias"].iloc[-1], "BULLISH")

    def test_compute_all_adds_all_columns(self):
        """Test that compute_all adds all expected feature columns."""
        df = _make_ohlcv_df(200)
        result = self.fe.compute_all(df)

        expected_columns = [
            "atr_14", "atr_average", "is_swing_high", "is_swing_low",
            "swing_high_price", "swing_low_price", "market_bias",
            "body_size", "total_range", "upper_wick", "lower_wick",
            "body_ratio", "is_bullish", "is_bearish", "volatility_regime",
            "session", "near_equal_high", "near_equal_low",
            "prev_day_high", "prev_day_low",
            "asian_session_high", "asian_session_low",
            "london_session_high", "london_session_low",
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_compute_candle_properties(self):
        """Test candle property calculations."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="15min", tz="UTC"),
            "open": [1.1000],
            "high": [1.1050],
            "low": [1.0950],
            "close": [1.1030],
            "volume": [500],
        })
        result = self.fe.compute_candle_properties(df)

        self.assertAlmostEqual(result["body_size"].iloc[0], 0.003, places=4)
        self.assertAlmostEqual(result["total_range"].iloc[0], 0.01, places=4)
        self.assertTrue(result["is_bullish"].iloc[0])
        self.assertFalse(result["is_bearish"].iloc[0])


if __name__ == "__main__":
    unittest.main()
