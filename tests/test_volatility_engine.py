"""Unit tests for the VolatilityEngine module."""

import unittest

import numpy as np
import pandas as pd

from src.volatility_engine import VolatilityEngine
from src.logger import BotLogger


def _make_config():
    """Return a minimal config dict for testing."""
    return {
        "strategy": {
            "volatility_multiplier": 1.3,
        },
        "trade_management": {
            "volatility_collapse_atr_multiple": 0.7,
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


def _make_df(atr=0.015, atr_avg=0.010, regime="NORMAL"):
    """Create a test DataFrame with ATR columns."""
    n = 10
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": [1.20] * n,
        "high": [1.21] * n,
        "low": [1.19] * n,
        "close": [1.20] * n,
        "volume": [100] * n,
        "atr_14": [atr] * n,
        "atr_average": [atr_avg] * n,
        "volatility_regime": [regime] * n,
    })


class TestVolatilityEngine(unittest.TestCase):
    """Tests for VolatilityEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = _make_config()
        self.logger = BotLogger(self.config)
        self.engine = VolatilityEngine(self.config, self.logger)

    def test_is_volatility_sufficient_false_atr_below_threshold(self):
        """Test is_volatility_sufficient returns False when ATR is below threshold."""
        # ATR = 0.010, avg = 0.010, threshold = 0.013 => ATR < threshold
        df = _make_df(atr=0.010, atr_avg=0.010, regime="NORMAL")
        self.assertFalse(self.engine.is_volatility_sufficient(df))

    def test_is_volatility_sufficient_false_when_ranging(self):
        """Test is_volatility_sufficient returns False when regime is RANGING."""
        # ATR above threshold but regime is RANGING
        df = _make_df(atr=0.020, atr_avg=0.010, regime="RANGING")
        self.assertFalse(self.engine.is_volatility_sufficient(df))

    def test_is_volatility_sufficient_true(self):
        """Test is_volatility_sufficient returns True when all conditions met."""
        df = _make_df(atr=0.020, atr_avg=0.010, regime="EXPANDING")
        self.assertTrue(self.engine.is_volatility_sufficient(df))

    def test_is_volatility_sufficient_false_sweep_range_too_small(self):
        """Test is_volatility_sufficient returns False when sweep range is too small."""
        df = _make_df(atr=0.020, atr_avg=0.010, regime="EXPANDING")
        sweep = {"sweep_high": 1.2010, "sweep_low": 1.2000}  # range = 0.001, threshold = 0.015
        self.assertFalse(self.engine.is_volatility_sufficient(df, sweep=sweep))

    def test_is_volatility_collapsing_true(self):
        """Test is_volatility_collapsing returns True when ATR drops below collapse threshold."""
        # ATR = 0.005, avg = 0.010, collapse threshold = 0.007 => collapsing
        df = _make_df(atr=0.005, atr_avg=0.010)
        self.assertTrue(self.engine.is_volatility_collapsing(df))

    def test_is_volatility_collapsing_false(self):
        """Test is_volatility_collapsing returns False when ATR is above collapse threshold."""
        df = _make_df(atr=0.015, atr_avg=0.010)
        self.assertFalse(self.engine.is_volatility_collapsing(df))

    def test_get_volatility_state_all_keys(self):
        """Test get_volatility_state returns a dict with all required keys."""
        df = _make_df(atr=0.015, atr_avg=0.010, regime="NORMAL")
        state = self.engine.get_volatility_state(df)

        required_keys = {"current_atr", "atr_average", "regime", "is_sufficient", "ratio"}
        self.assertEqual(set(state.keys()), required_keys)
        self.assertIsInstance(state["current_atr"], float)
        self.assertIsInstance(state["atr_average"], float)
        self.assertIsInstance(state["regime"], str)
        self.assertIsInstance(state["is_sufficient"], bool)
        self.assertIsInstance(state["ratio"], float)

    def test_get_atr_value(self):
        """Test get_atr_value returns correct float."""
        df = _make_df(atr=0.0123)
        self.assertAlmostEqual(self.engine.get_atr_value(df), 0.0123)

    def test_get_atr_average(self):
        """Test get_atr_average returns correct float."""
        df = _make_df(atr_avg=0.0456)
        self.assertAlmostEqual(self.engine.get_atr_average(df), 0.0456)

    def test_get_atr_value_empty_df(self):
        """Test get_atr_value returns NaN for empty DataFrame."""
        df = pd.DataFrame()
        result = self.engine.get_atr_value(df)
        self.assertTrue(pd.isna(result))

    def test_get_volatility_state_ratio(self):
        """Test that ratio is correctly calculated."""
        df = _make_df(atr=0.020, atr_avg=0.010)
        state = self.engine.get_volatility_state(df)
        self.assertAlmostEqual(state["ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
