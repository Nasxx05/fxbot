"""Unit tests for the DataEngine module (MetaTrader 5)."""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import yaml


def _load_config():
    """Load the test configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _make_mt5_rates(count=2, start_time=1704067200):
    """Create a mock MT5 rates numpy structured array."""
    dtype = np.dtype([
        ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
        ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8"),
    ])
    data = []
    for i in range(count):
        data.append((
            start_time + i * 900,
            1.1000 + i * 0.002,
            1.1050 + i * 0.002,
            1.0950 + i * 0.002,
            1.1020 + i * 0.002,
            500 + i * 100,
            12,
            0,
        ))
    return np.array(data, dtype=dtype)


class TestDataEngineHelpers(unittest.TestCase):
    """Tests for DataEngine static helpers and validation (no MT5 connection needed)."""

    def test_to_mt5_symbol(self):
        """to_mt5_symbol strips underscores."""
        from src.data_engine import DataEngine
        self.assertEqual(DataEngine.to_mt5_symbol("EUR_USD"), "EURUSD")
        self.assertEqual(DataEngine.to_mt5_symbol("GBP_JPY"), "GBPJPY")
        self.assertEqual(DataEngine.to_mt5_symbol("XAU_USD"), "XAUUSD")

    def test_from_mt5_symbol(self):
        """from_mt5_symbol adds underscore after 3rd character."""
        from src.data_engine import DataEngine
        self.assertEqual(DataEngine.from_mt5_symbol("EURUSD"), "EUR_USD")
        self.assertEqual(DataEngine.from_mt5_symbol("GBPJPY"), "GBP_JPY")
        self.assertEqual(DataEngine.from_mt5_symbol("XAUUSD"), "XAU_USD")

    def test_roundtrip_symbol_conversion(self):
        """Converting to MT5 and back should return the original."""
        from src.data_engine import DataEngine
        for sym in ["EUR_USD", "GBP_USD", "GBP_JPY", "XAU_USD"]:
            self.assertEqual(DataEngine.from_mt5_symbol(DataEngine.to_mt5_symbol(sym)), sym)


class TestDataEngineWithMockedMT5(unittest.TestCase):
    """Tests for DataEngine with mocked MT5 module."""

    @patch("src.data_engine.mt5")
    def setUp(self, mock_mt5):
        """Set up test fixtures with mocked MT5."""
        mock_mt5.initialize.return_value = True
        mock_account = MagicMock()
        mock_account.login = 12345
        mock_account.server = "TestServer"
        mock_mt5.account_info.return_value = mock_account
        mock_mt5.TIMEFRAME_M15 = 15
        mock_mt5.TIMEFRAME_H1 = 60
        mock_mt5.TIMEFRAME_H4 = 240

        self.config = _load_config()
        self.engine = self._create_engine(mock_mt5)
        self.mock_mt5 = mock_mt5

        # Use temp db
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.tmp_db = tmp.name
        self.engine.db_path = self.tmp_db
        self.engine._init_db()

    def _create_engine(self, mock_mt5):
        """Create DataEngine with mocked MT5."""
        from src.data_engine import DataEngine
        return DataEngine(self.config)

    def tearDown(self):
        """Clean up temp db."""
        if hasattr(self, "tmp_db") and os.path.exists(self.tmp_db):
            os.unlink(self.tmp_db)

    def test_validate_candle_rejects_high_less_than_low(self):
        """Candle validation rejects a candle where high < low."""
        bad_candle = {"open": 1.10, "high": 1.05, "low": 1.09, "close": 1.08, "volume": 100}
        self.assertFalse(self.engine._validate_candle(bad_candle))

    def test_validate_candle_rejects_null_value(self):
        """Candle validation rejects a candle with a null value."""
        null_candle = {"open": 1.10, "high": None, "low": 1.09, "close": 1.10, "volume": 100}
        self.assertFalse(self.engine._validate_candle(null_candle))

    def test_validate_candle_accepts_valid(self):
        """A valid candle passes validation."""
        good_candle = {"open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11, "volume": 500}
        self.assertTrue(self.engine._validate_candle(good_candle))

    def test_calculate_spread_pips_standard(self):
        """Pip calculation for standard pairs."""
        spread = self.engine._calculate_spread_pips("EUR_USD", 1.10000, 1.10015)
        self.assertAlmostEqual(spread, 1.5, places=1)

    def test_calculate_spread_pips_jpy(self):
        """Pip calculation for JPY pairs."""
        spread = self.engine._calculate_spread_pips("GBP_JPY", 186.500, 186.530)
        self.assertAlmostEqual(spread, 3.0, places=1)

    def test_calculate_spread_pips_xau(self):
        """Pip calculation for XAU_USD."""
        spread = self.engine._calculate_spread_pips("XAU_USD", 2000.00, 2000.50)
        self.assertAlmostEqual(spread, 5.0, places=1)

    def test_store_and_get_candles_from_db(self):
        """Storing and retrieving candles from SQLite."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:15:00Z"], utc=True),
            "open": [1.10, 1.11],
            "high": [1.12, 1.13],
            "low": [1.09, 1.10],
            "close": [1.11, 1.12],
            "volume": [500, 600],
        })
        self.engine._store_candles("EUR_USD", "M15", df)
        result = self.engine.get_candles_from_db("EUR_USD", "M15", 10)

        self.assertEqual(len(result), 2)
        self.assertIn("timestamp", result.columns)

    @patch("src.data_engine.mt5")
    def test_fetch_historical_candles_returns_correct_columns(self, mock_mt5):
        """fetch_historical_candles returns DataFrame with correct columns."""
        rates = _make_mt5_rates(3)
        mock_mt5.copy_rates_from_pos.return_value = rates
        mock_mt5.symbol_select.return_value = True
        mock_mt5.TIMEFRAME_M15 = 15

        df = self.engine.fetch_historical_candles("EUR_USD", "M15", 3)

        self.assertIsInstance(df, pd.DataFrame)
        expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        # Last candle is dropped (incomplete), so we get count-1
        self.assertEqual(len(df), 2)

    @patch("src.data_engine.mt5")
    def test_fetch_historical_candles_handles_none(self, mock_mt5):
        """fetch_historical_candles returns empty DataFrame on MT5 failure."""
        mock_mt5.copy_rates_from_pos.return_value = None
        mock_mt5.symbol_select.return_value = True
        mock_mt5.last_error.return_value = (-1, "Test error")
        mock_mt5.TIMEFRAME_M15 = 15

        df = self.engine.fetch_historical_candles("EUR_USD", "M15", 10)
        self.assertTrue(df.empty)

    @patch("src.data_engine.mt5")
    def test_get_current_spread_returns_float(self, mock_mt5):
        """get_current_spread returns a positive float."""
        mock_info = MagicMock()
        mock_info.spread = 12  # 12 points = 1.2 pips
        mock_mt5.symbol_info.return_value = mock_info

        spread = self.engine.get_current_spread("EUR_USD")
        self.assertIsInstance(spread, float)
        self.assertAlmostEqual(spread, 1.2, places=1)

    @patch("src.data_engine.mt5")
    def test_get_current_spread_handles_none(self, mock_mt5):
        """get_current_spread returns 0.0 when symbol_info returns None."""
        mock_mt5.symbol_info.return_value = None

        spread = self.engine.get_current_spread("EUR_USD")
        self.assertEqual(spread, 0.0)

    @patch("src.data_engine.mt5")
    def test_get_bid_ask_returns_tuple(self, mock_mt5):
        """get_bid_ask returns a tuple of floats."""
        mock_tick = MagicMock()
        mock_tick.bid = 1.10000
        mock_tick.ask = 1.10015
        mock_mt5.symbol_info_tick.return_value = mock_tick

        bid, ask = self.engine.get_bid_ask("EUR_USD")
        self.assertIsInstance(bid, float)
        self.assertIsInstance(ask, float)
        self.assertGreater(ask, bid)

    @patch("src.data_engine.mt5")
    def test_get_bid_ask_handles_none(self, mock_mt5):
        """get_bid_ask returns (0.0, 0.0) when tick data unavailable."""
        mock_mt5.symbol_info_tick.return_value = None

        bid, ask = self.engine.get_bid_ask("EUR_USD")
        self.assertEqual(bid, 0.0)
        self.assertEqual(ask, 0.0)

    @patch("src.data_engine.mt5")
    def test_get_open_positions_empty(self, mock_mt5):
        """get_open_positions returns empty list when no positions."""
        mock_mt5.positions_get.return_value = None

        positions = self.engine.get_open_positions()
        self.assertEqual(positions, [])

    @patch("src.data_engine.mt5")
    def test_get_open_positions_returns_list(self, mock_mt5):
        """get_open_positions converts MT5 positions to dicts."""
        mock_pos = MagicMock()
        mock_pos.ticket = 123456
        mock_pos.symbol = "EURUSD"
        mock_pos.type = 0  # ORDER_TYPE_BUY
        mock_pos.volume = 0.10
        mock_pos.price_open = 1.1000
        mock_pos.sl = 1.0950
        mock_pos.tp = 1.1100
        mock_pos.profit = 25.0
        mock_pos.time = 1704067200
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.positions_get.return_value = [mock_pos]

        positions = self.engine.get_open_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "EUR_USD")
        self.assertEqual(positions[0]["type"], "LONG")
        self.assertEqual(positions[0]["ticket"], 123456)

    @patch("src.data_engine.mt5")
    def test_warm_up_calls_fetch(self, mock_mt5):
        """warm_up fetches candles for all instrument/timeframe combos."""
        rates = _make_mt5_rates(3)
        mock_mt5.copy_rates_from_pos.return_value = rates
        mock_mt5.symbol_select.return_value = True
        mock_mt5.TIMEFRAME_M15 = 15
        mock_mt5.TIMEFRAME_H1 = 60

        self.engine.warm_up(["EUR_USD"], ["M15", "H1"])
        self.assertEqual(mock_mt5.copy_rates_from_pos.call_count, 2)


if __name__ == "__main__":
    unittest.main()
