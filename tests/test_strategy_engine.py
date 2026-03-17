"""Unit tests for the StrategyEngine module."""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd

from src.strategy_engine import StrategyEngine
from src.logger import BotLogger


def _make_config():
    """Return a full config dict for testing."""
    return {
        "risk": {
            "max_trades_per_day": 3,
            "min_risk_reward": 3.0,
            "min_risk_reward_after_spread": 2.5,
            "min_sl_atr_multiple": 1.0,
            "max_sl_atr_multiple": 3.0,
        },
        "strategy": {
            "pullback_candle_limit": 5,
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


def _make_df(n=50):
    """Create a test DataFrame with required columns."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": [1.20] * n,
        "high": [1.21] * n,
        "low": [1.19] * n,
        "close": [1.20] * n,
        "volume": [100] * n,
        "atr_14": [0.005] * n,
        "atr_average": [0.004] * n,
        "volatility_regime": ["NORMAL"] * n,
        "is_swing_high": [False] * n,
        "is_swing_low": [False] * n,
        "swing_high_price": [np.nan] * n,
        "swing_low_price": [np.nan] * n,
    })


def _build_strategy_engine(config=None, **overrides):
    """Build a StrategyEngine with mocked dependencies."""
    config = config or _make_config()
    logger = BotLogger(config)
    deps = {
        "data_engine": MagicMock(),
        "feature_engine": MagicMock(),
        "liquidity_engine": MagicMock(),
        "structure_engine": MagicMock(),
        "volatility_engine": MagicMock(),
        "news_filter": MagicMock(),
        "session_filter": MagicMock(),
        "spread_controller": MagicMock(),
        "correlation_guard": MagicMock(),
    }
    deps.update(overrides)

    # Default mock returns: all gates pass
    deps["session_filter"].is_trading_allowed.return_value = (True, "LONDON")
    deps["news_filter"].is_trading_blocked.return_value = (False, None)
    deps["volatility_engine"].get_volatility_state.return_value = {
        "regime": "NORMAL", "current_atr": 0.005, "atr_average": 0.004,
        "is_sufficient": True, "ratio": 1.25,
    }
    deps["volatility_engine"].is_volatility_sufficient.return_value = True
    deps["volatility_engine"].get_atr_value.return_value = 0.005
    deps["spread_controller"].check_spread.return_value = (True, 1.0, None)
    deps["spread_controller"].calculate_spread_adjusted_rr.return_value = 3.0
    deps["spread_controller"].pips_to_price.return_value = 0.00010
    deps["spread_controller"].data_engine = deps["data_engine"]
    deps["data_engine"].get_current_spread.return_value = 1.0
    deps["correlation_guard"].is_correlated_trade_open.return_value = (False, None)
    deps["liquidity_engine"].get_active_zones.return_value = [{"type": "EQUAL_HIGH", "price": 1.21, "active": True}]
    deps["liquidity_engine"].detect_sweep.return_value = {
        "direction": "BEARISH_SWEEP",
        "sweep_candle_index": 10,
        "sweep_high": 1.2150,
        "sweep_low": 1.2050,
        "timestamp": datetime.now(timezone.utc),
        "zone": {"type": "EQUAL_HIGH", "price": 1.21},
    }
    deps["structure_engine"].detect_break_of_structure.return_value = {
        "direction": "BEARISH",
        "bos_level": 1.1950,
        "bos_candle_index": 20,
        "bos_candle_close": 1.1940,
        "sweep_reference": deps["liquidity_engine"].detect_sweep.return_value,
        "timestamp": datetime.now(timezone.utc),
    }
    deps["structure_engine"].detect_pullback_zone.return_value = {
        "entry_price": 1.2100,
        "zone_high": 1.2120,
        "zone_low": 1.2080,
        "method": "RETRACE_50",
        "expiry_candles": 100,
    }
    deps["structure_engine"].check_setup_expired.return_value = False

    se = StrategyEngine(config, logger, **deps)
    return se, deps


class TestStrategyEngine(unittest.TestCase):
    """Tests for StrategyEngine class."""

    def test_gate1_daily_limit_blocks(self):
        """Test Gate 1 blocks when 3 trades already taken."""
        se, deps = _build_strategy_engine()
        se.daily_trade_count = 3
        result = se.run_signal_pipeline("EUR_USD", _make_df())
        self.assertIsNone(result)

    def test_gate2_session_blocks_asian(self):
        """Test Gate 2 blocks during Asian session."""
        se, deps = _build_strategy_engine()
        deps["session_filter"].is_trading_allowed.return_value = (False, "ASIAN")
        result = se.run_signal_pipeline("EUR_USD", _make_df())
        self.assertIsNone(result)

    def test_gate3_news_blocks(self):
        """Test Gate 3 blocks when news event is imminent."""
        se, deps = _build_strategy_engine()
        deps["news_filter"].is_trading_blocked.return_value = (True, "NFP in 10 minutes")
        result = se.run_signal_pipeline("EUR_USD", _make_df())
        self.assertIsNone(result)

    def test_gate4_ranging_blocks(self):
        """Test Gate 4 blocks when volatility regime is RANGING."""
        se, deps = _build_strategy_engine()
        deps["volatility_engine"].get_volatility_state.return_value = {
            "regime": "RANGING", "current_atr": 0.003, "atr_average": 0.005,
            "is_sufficient": False, "ratio": 0.6,
        }
        result = se.run_signal_pipeline("EUR_USD", _make_df())
        self.assertIsNone(result)

    def test_gate5_spread_blocks(self):
        """Test Gate 5 blocks when spread exceeds threshold."""
        se, deps = _build_strategy_engine()
        deps["spread_controller"].check_spread.return_value = (False, 3.5, "SPREAD_TOO_HIGH")
        result = se.run_signal_pipeline("EUR_USD", _make_df())
        self.assertIsNone(result)

    def test_gate11_rr_too_low_blocks(self):
        """Test Gate 11 blocks when spread-adjusted RR is below minimum."""
        se, deps = _build_strategy_engine()
        deps["spread_controller"].calculate_spread_adjusted_rr.return_value = 1.5
        df = _make_df()
        # Ensure pullback zone puts price in zone
        df["low"] = 1.2080
        df["high"] = 1.2120
        result = se.run_signal_pipeline("EUR_USD", df)
        self.assertIsNone(result)

    def test_all_gates_pass_returns_signal(self):
        """Test all gates passing returns a valid trade_signal dict."""
        se, deps = _build_strategy_engine()
        df = _make_df()
        # Ensure price is in the pullback zone
        df["low"] = 1.2080
        df["high"] = 1.2120
        df["close"] = 1.2100

        result = se.run_signal_pipeline("EUR_USD", df)
        self.assertIsNotNone(result)

        required_keys = [
            "instrument", "direction", "entry_price", "stop_loss",
            "take_profit_1", "take_profit_2", "position_size",
            "risk_reward_raw", "risk_reward_adjusted", "sweep_reference",
            "bos_reference", "entry_method", "signal_time",
            "current_spread_pips", "session", "atr_at_signal",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

        self.assertEqual(result["instrument"], "EUR_USD")
        self.assertEqual(result["direction"], "SHORT")
        self.assertIsNone(result["position_size"])

    def test_calculate_stop_loss_returns_none_when_too_wide(self):
        """Test calculate_stop_loss returns None when SL exceeds max ATR multiple."""
        se, deps = _build_strategy_engine()
        sweep = {
            "sweep_high": 1.3000,  # Very far from current price
            "sweep_low": 1.1000,
            "zone": {"type": "EQUAL_HIGH"},
        }
        df = _make_df()
        df["atr_14"] = 0.005
        df["close"] = 1.20

        result = se.calculate_stop_loss(sweep, "SHORT", df)
        self.assertIsNone(result)

    def test_calculate_take_profits_long(self):
        """Test take profit calculation for LONG direction."""
        se, _ = _build_strategy_engine()
        tp1, tp2 = se.calculate_take_profits(1.2000, 1.1950, "LONG")
        # risk = 0.005, tp1 = 1.21, tp2 = 1.215
        self.assertAlmostEqual(tp1, 1.2100, places=4)
        self.assertAlmostEqual(tp2, 1.2150, places=4)

    def test_calculate_take_profits_short(self):
        """Test take profit calculation for SHORT direction."""
        se, _ = _build_strategy_engine()
        tp1, tp2 = se.calculate_take_profits(1.2000, 1.2050, "SHORT")
        # risk = 0.005, tp1 = 1.19, tp2 = 1.185
        self.assertAlmostEqual(tp1, 1.1900, places=4)
        self.assertAlmostEqual(tp2, 1.1850, places=4)

    def test_invalidate_setup_clears_state(self):
        """Test invalidate_setup removes all active state for instrument."""
        se, _ = _build_strategy_engine()
        se.active_sweeps["EUR_USD"] = [{"some": "sweep"}]
        se.active_setups["EUR_USD"] = [{"some": "setup"}]

        se.invalidate_setup("EUR_USD", "TEST_REASON")

        self.assertNotIn("EUR_USD", se.active_sweeps)
        self.assertNotIn("EUR_USD", se.active_setups)

    def test_reset_daily_count(self):
        """Test reset_daily_count resets counter."""
        se, _ = _build_strategy_engine()
        se.daily_trade_count = 5
        se.reset_daily_count()
        self.assertEqual(se.daily_trade_count, 0)

    def test_gate6_correlation_blocks(self):
        """Test Gate 6 blocks when correlated trade is open."""
        se, deps = _build_strategy_engine()
        deps["correlation_guard"].is_correlated_trade_open.return_value = (True, "GBP_USD")
        result = se.run_signal_pipeline("EUR_USD", _make_df())
        self.assertIsNone(result)

    def test_signal_increments_daily_count(self):
        """Test that a successful signal increments the daily trade count."""
        se, deps = _build_strategy_engine()
        df = _make_df()
        df["low"] = 1.2080
        df["high"] = 1.2120
        df["close"] = 1.2100

        self.assertEqual(se.daily_trade_count, 0)
        se.run_signal_pipeline("EUR_USD", df)
        self.assertEqual(se.daily_trade_count, 1)


if __name__ == "__main__":
    unittest.main()
