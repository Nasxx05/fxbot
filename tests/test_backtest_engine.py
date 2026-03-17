"""Unit tests for the BacktestEngine module."""

import unittest

import numpy as np
import pandas as pd

from backtest.backtest_engine import BacktestEngine


def _make_config():
    return {
        "risk": {
            "max_trades_per_day": 3,
            "min_risk_reward": 3.0,
            "min_risk_reward_after_spread": 2.5,
            "min_sl_atr_multiple": 1.0,
            "max_sl_atr_multiple": 3.0,
        },
        "strategy": {
            "atr_period": 14,
            "atr_average_period": 50,
            "pivot_lookback": 2,
            "volatility_multiplier": 1.3,
            "ranging_candle_threshold": 10,
            "liquidity_threshold_pct": 0.02,
            "sweep_wick_body_ratio": 1.5,
            "sweep_candle_lookback": 5,
            "pullback_candle_limit": 5,
        },
        "trade_management": {
            "breakeven_trigger_r": 1.0,
            "partial_close_trigger_r": 2.0,
            "partial_close_pct": 0.50,
            "full_close_trigger_r": 3.0,
            "trailing_stop_atr_multiple": 1.0,
            "max_trade_duration_hours": 24,
            "volatility_collapse_atr_multiple": 0.7,
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
            "allowed_sessions": ["LONDON", "NEW_YORK", "OVERLAP"],
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/",
                    "max_file_size_mb": 50, "backup_count": 10},
    }


def _make_df(n=50, base_price=1.2000):
    """Create a basic test DataFrame."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-15 08:00", periods=n,
                                   freq="15min", tz="UTC"),
        "open": [base_price] * n,
        "high": [base_price + 0.001] * n,
        "low": [base_price - 0.001] * n,
        "close": [base_price] * n,
        "volume": [100] * n,
        "atr_14": [0.005] * n,
    })


class TestBacktestEngine(unittest.TestCase):

    def setUp(self):
        self.config = _make_config()
        self.be = BacktestEngine(self.config, None)

    def test_simulate_trade_expired(self):
        """simulate_trade_outcome returns EXPIRED when price never reaches entry zone."""
        df = _make_df(20)
        signal = {
            "instrument": "EUR_USD",
            "direction": "LONG",
            "entry_price": 1.1500,  # Far below — price never gets here
            "stop_loss": 1.1450,
            "take_profit_1": 1.1600,
            "take_profit_2": 1.1650,
            "entry_zone_high": 1.1510,
            "entry_zone_low": 1.1490,
            "session": "LONDON",
        }
        result = self.be.simulate_trade_outcome(signal, df, 5, 1.0, 0.3)
        self.assertEqual(result["status"], "EXPIRED")
        self.assertEqual(result["exit_reason"], "EXPIRED")

    def test_simulate_trade_win_tp2(self):
        """simulate_trade_outcome returns WIN when price hits TP2."""
        n = 30
        prices = [1.2000] * 10
        # Candle 10-14: price enters entry zone
        prices += [1.2000] * 5
        # Candle 15-29: price rallies to TP2
        for i in range(15):
            prices.append(1.2000 + i * 0.002)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-15 08:00", periods=n,
                                       freq="15min", tz="UTC"),
            "open": prices,
            "high": [p + 0.003 for p in prices],
            "low": [p - 0.001 for p in prices],
            "close": prices,
            "volume": [100] * n,
            "atr_14": [0.005] * n,
        })

        signal = {
            "instrument": "EUR_USD",
            "direction": "LONG",
            "entry_price": 1.2000,
            "stop_loss": 1.1950,
            "take_profit_1": 1.2100,
            "take_profit_2": 1.2150,
            "entry_zone_high": 1.2010,
            "entry_zone_low": 1.1990,
            "session": "LONDON",
        }
        result = self.be.simulate_trade_outcome(signal, df, 5, 1.0, 0.0)
        self.assertEqual(result["status"], "WIN")
        self.assertEqual(result["exit_reason"], "TP2")

    def test_simulate_trade_applies_costs(self):
        """simulate_trade_outcome correctly applies spread and slippage costs."""
        df = _make_df(20)
        signal = {
            "instrument": "EUR_USD",
            "direction": "LONG",
            "entry_price": 1.1500,
            "stop_loss": 1.1450,
            "take_profit_1": 1.1600,
            "take_profit_2": 1.1650,
            "entry_zone_high": 1.1510,
            "entry_zone_low": 1.1490,
            "session": "LONDON",
        }
        result = self.be.simulate_trade_outcome(signal, df, 5, 2.0, 0.5)
        self.assertEqual(result["spread_cost_pips"], 2.0)
        self.assertEqual(result["slippage_cost_pips"], 0.5)
        # EXPIRED trades have zero commission; non-expired have > 0
        self.assertGreaterEqual(result["commission_usd"], 0)

    def test_calculate_metrics_win_rate(self):
        """calculate_metrics returns correct win_rate for a known set."""
        results = [
            {"status": "WIN", "net_pnl_usd": 100, "r_multiple": 2.0,
             "pnl_pips": 50, "session": "LONDON", "entry_time": "2024-01-15",
             "duration_hours": 2.0},
            {"status": "WIN", "net_pnl_usd": 80, "r_multiple": 1.5,
             "pnl_pips": 40, "session": "LONDON", "entry_time": "2024-01-16",
             "duration_hours": 3.0},
            {"status": "LOSS", "net_pnl_usd": -50, "r_multiple": -1.0,
             "pnl_pips": -25, "session": "NEW_YORK", "entry_time": "2024-01-17",
             "duration_hours": 1.0},
        ]
        metrics = self.be.calculate_metrics(results)
        # 2 wins out of 3 = 66.67%
        self.assertAlmostEqual(metrics["win_rate"], 66.67, places=1)
        self.assertEqual(metrics["total_trades"], 3)
        self.assertEqual(metrics["winning_trades"], 2)
        self.assertEqual(metrics["losing_trades"], 1)

    def test_calculate_metrics_max_drawdown(self):
        """calculate_metrics returns correct max_drawdown for a known sequence."""
        # +100, -200, -100 -> equity: 10100, 9900, 9800
        # Peak was 10100, trough 9800 -> dd = 300
        results = [
            {"status": "WIN", "net_pnl_usd": 100, "r_multiple": 2.0,
             "pnl_pips": 50, "session": "LONDON", "entry_time": "2024-01-15",
             "duration_hours": 2.0},
            {"status": "LOSS", "net_pnl_usd": -200, "r_multiple": -1.0,
             "pnl_pips": -100, "session": "LONDON", "entry_time": "2024-01-16",
             "duration_hours": 1.0},
            {"status": "LOSS", "net_pnl_usd": -100, "r_multiple": -1.0,
             "pnl_pips": -50, "session": "LONDON", "entry_time": "2024-01-17",
             "duration_hours": 1.0},
        ]
        metrics = self.be.calculate_metrics(results, initial_balance=10000)
        self.assertAlmostEqual(metrics["max_drawdown_usd"], 300.0, places=0)

    def test_walk_forward_overfit(self):
        """run_walk_forward_validation returns OVERFIT when IS PF is high and OOS is low."""
        # We can't easily generate real data, so test the verdict logic directly
        be = BacktestEngine(self.config, None)

        # Mock the run_backtest to return controlled results
        original_run = be.run_backtest

        call_count = [0]

        def mock_run_backtest(instrument, start, end, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # IS: high performance
                return {
                    "total_trades": 50, "winning_trades": 40,
                    "losing_trades": 10, "expired_trades": 0,
                    "win_rate": 80.0, "profit_factor": 3.0,
                    "total_net_pnl_usd": 5000, "total_return_pct": 50,
                    "max_drawdown_usd": 500, "max_drawdown_pct": 5,
                    "sharpe_ratio": 2.0, "expectancy_per_trade_r": 1.5,
                    "average_win_r": 2.0, "average_loss_r": -1.0,
                    "largest_win_usd": 500, "largest_loss_usd": -200,
                    "average_trade_duration_hours": 4,
                    "trades_by_session": {}, "trades_by_day_of_week": {},
                }
            else:
                # OOS: poor performance
                return {
                    "total_trades": 15, "winning_trades": 5,
                    "losing_trades": 10, "expired_trades": 0,
                    "win_rate": 33.0, "profit_factor": 0.8,
                    "total_net_pnl_usd": -200, "total_return_pct": -2,
                    "max_drawdown_usd": 400, "max_drawdown_pct": 4,
                    "sharpe_ratio": -0.5, "expectancy_per_trade_r": -0.3,
                    "average_win_r": 1.5, "average_loss_r": -1.0,
                    "largest_win_usd": 100, "largest_loss_usd": -150,
                    "average_trade_duration_hours": 3,
                    "trades_by_session": {}, "trades_by_day_of_week": {},
                }

        be.run_backtest = mock_run_backtest

        result = be.run_walk_forward_validation(
            "EUR_USD", "2023-01-01", "2024-01-01")

        self.assertEqual(result["verdict"], "OVERFIT")

    def test_calculate_metrics_empty(self):
        """calculate_metrics handles empty results list."""
        metrics = self.be.calculate_metrics([])
        self.assertEqual(metrics["total_trades"], 0)
        self.assertEqual(metrics["win_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
