"""Integration test for Phase 2 — full pipeline from FeatureEngine through signal detection."""

import unittest

import numpy as np
import pandas as pd

from src.feature_engine import FeatureEngine
from src.liquidity_engine import LiquidityEngine
from src.structure_engine import StructureEngine
from src.volatility_engine import VolatilityEngine
from src.logger import BotLogger


def _make_config():
    """Return a full config dict for integration testing."""
    return {
        "strategy": {
            "atr_period": 14,
            "atr_average_period": 50,
            "pivot_lookback": 2,
            "volatility_multiplier": 1.3,
            "ranging_candle_threshold": 10,
            "liquidity_threshold_pct": 0.02,
            "sweep_wick_body_ratio": 1.5,
            "sweep_candle_lookback": 5,
            "pullback_candle_limit": 10,
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
        "trade_management": {
            "volatility_collapse_atr_multiple": 0.7,
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


def _build_synthetic_df():
    """Build a 200-candle DataFrame with a sweep and BOS pattern embedded.

    The pattern:
    - Candles 0-149: trending up with natural swing points
    - Around candle 150: equal highs form a liquidity zone
    - Around candle 160: sweep of those highs (wick above, close back below)
    - Around candle 170: swing low forms after the sweep
    - Around candle 180: bearish BOS — close below the swing low
    """
    n = 200
    np.random.seed(123)

    base = 1.2000 + np.cumsum(np.random.randn(n) * 0.0005)
    open_ = base.copy()
    close = base + np.random.randn(n) * 0.0003
    high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.0008)
    low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.0008)

    # Create a clear high level around 1.22 at indices 100 and 120 (equal highs)
    target_high = 1.2200
    for idx in [100, 120]:
        high[idx] = target_high + 0.0001 * (idx == 120)  # Nearly equal
        open_[idx] = target_high - 0.002
        close[idx] = target_high - 0.001
        low[idx] = target_high - 0.003

    # Sweep candle at index 160: wick above, close back below
    high[160] = target_high + 0.0050  # wick well above
    open_[160] = target_high - 0.0010
    close[160] = target_high - 0.0020  # close below the level
    low[160] = target_high - 0.0030

    # Swing low after sweep at index 170
    low[170] = target_high - 0.0100
    open_[170] = target_high - 0.005
    close[170] = target_high - 0.006
    high[170] = target_high - 0.004

    # BOS candle at index 180: close below the swing low
    close[180] = target_high - 0.0120
    open_[180] = target_high - 0.005
    high[180] = target_high - 0.004
    low[180] = target_high - 0.013

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    })
    return df


class TestIntegrationPhase2(unittest.TestCase):
    """Integration test: full pipeline from features through signal detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = _make_config()
        self.logger = BotLogger(self.config)

    def test_full_pipeline_completes_without_errors(self):
        """Test that the full pipeline runs without errors and returns expected types."""
        df = _build_synthetic_df()

        # Step 1: Compute all features
        fe = FeatureEngine(self.config)
        df = fe.compute_all(df)

        self.assertIn("atr_14", df.columns)
        self.assertIn("is_swing_high", df.columns)
        self.assertIn("volatility_regime", df.columns)

        # Step 2: Detect liquidity zones
        le = LiquidityEngine(self.config, self.logger)
        zones = le.detect_liquidity_zones(df)
        self.assertIsInstance(zones, list)

        # Step 3: Attempt sweep detection on each zone
        sweep_results = []
        for zone in zones:
            result = le.detect_sweep(df, zone)
            if result is not None:
                sweep_results.append(result)
                self.assertIsInstance(result, dict)
                self.assertIn("direction", result)
                self.assertIn("sweep_candle_index", result)

        # Step 4: Attempt BOS detection on any sweep found
        se = StructureEngine(self.config, self.logger)
        bos_results = []
        for sweep in sweep_results:
            bos = se.detect_break_of_structure(df, sweep)
            if bos is not None:
                bos_results.append(bos)
                self.assertIsInstance(bos, dict)
                self.assertIn("direction", bos)
                self.assertIn("bos_level", bos)

        # Step 5: Attempt pullback zone on any BOS found
        for bos in bos_results:
            sweep = bos["sweep_reference"]
            pullback = se.detect_pullback_zone(df, sweep, bos)
            if pullback is not None:
                self.assertIsInstance(pullback, dict)
                self.assertIn("entry_price", pullback)
                self.assertIn("method", pullback)

        # Step 6: Check volatility
        ve = VolatilityEngine(self.config, self.logger)
        vol_sufficient = ve.is_volatility_sufficient(df)
        self.assertIsInstance(vol_sufficient, bool)

        state = ve.get_volatility_state(df)
        self.assertIn("current_atr", state)
        self.assertIn("regime", state)

    def test_volatility_engine_on_computed_df(self):
        """Test VolatilityEngine methods work on FeatureEngine output."""
        df = _build_synthetic_df()
        fe = FeatureEngine(self.config)
        df = fe.compute_all(df)

        ve = VolatilityEngine(self.config, self.logger)

        atr = ve.get_atr_value(df)
        self.assertIsInstance(atr, float)
        self.assertGreater(atr, 0)

        avg = ve.get_atr_average(df)
        self.assertIsInstance(avg, float)
        self.assertGreater(avg, 0)

        collapsing = ve.is_volatility_collapsing(df)
        self.assertIsInstance(collapsing, bool)


if __name__ == "__main__":
    unittest.main()
