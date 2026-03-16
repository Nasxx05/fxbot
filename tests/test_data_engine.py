"""Unit tests for the DataEngine module."""

import json
import os
import sqlite3
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

from src.data_engine import DataEngine


def _load_config():
    """Load the test configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _make_oanda_candle(time_str, o, h, l, c, vol, complete=True):
    """Create a mock OANDA candle response dict."""
    return {
        "complete": complete,
        "volume": vol,
        "time": time_str,
        "mid": {"o": str(o), "h": str(h), "l": str(l), "c": str(c)},
    }


class TestDataEngine(unittest.TestCase):
    """Tests for DataEngine class."""

    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def setUp(self):
        """Set up test fixtures."""
        self.config = _load_config()
        self.engine = DataEngine(self.config)
        self.engine.db_path = ":memory:"

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_fetch_historical_candles_returns_correct_columns(self, mock_get):
        """Test that fetch_historical_candles returns a DataFrame with the correct columns."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candles": [
                _make_oanda_candle("2024-01-01T00:00:00.000000000Z", 1.1000, 1.1050, 1.0950, 1.1020, 500),
                _make_oanda_candle("2024-01-01T00:15:00.000000000Z", 1.1020, 1.1060, 1.0980, 1.1040, 600),
            ]
        }
        mock_get.return_value = mock_response

        df = self.engine.fetch_historical_candles("EUR_USD", "M15", 2)

        self.assertIsInstance(df, pd.DataFrame)
        expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        self.assertEqual(len(df), 2)

    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_validate_candle_rejects_high_less_than_low(self):
        """Test that candle validation rejects a candle where high < low."""
        bad_candle = {"open": 1.10, "high": 1.05, "low": 1.09, "close": 1.08, "volume": 100}
        self.assertFalse(self.engine._validate_candle(bad_candle))

    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_validate_candle_rejects_null_value(self):
        """Test that candle validation rejects a candle with a null value."""
        null_candle = {"open": 1.10, "high": None, "low": 1.09, "close": 1.10, "volume": 100}
        self.assertFalse(self.engine._validate_candle(null_candle))

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_get_current_spread_returns_positive_float(self, mock_get):
        """Test that get_current_spread returns a positive float."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "prices": [
                {
                    "instrument": "EUR_USD",
                    "bids": [{"price": "1.10000"}],
                    "asks": [{"price": "1.10015"}],
                }
            ]
        }
        mock_get.return_value = mock_response

        spread = self.engine.get_current_spread("EUR_USD")

        self.assertIsInstance(spread, float)
        self.assertGreater(spread, 0)

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_get_current_spread_jpy_pair(self, mock_get):
        """Test spread calculation for JPY pairs uses correct multiplier."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "prices": [
                {
                    "instrument": "GBP_JPY",
                    "bids": [{"price": "186.500"}],
                    "asks": [{"price": "186.530"}],
                }
            ]
        }
        mock_get.return_value = mock_response

        spread = self.engine.get_current_spread("GBP_JPY")
        self.assertAlmostEqual(spread, 3.0, places=1)

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_fetch_historical_rejects_invalid_candles(self, mock_get):
        """Test that invalid candles are filtered out from fetch results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candles": [
                _make_oanda_candle("2024-01-01T00:00:00.000000000Z", 1.10, 1.12, 1.09, 1.11, 500),
                _make_oanda_candle("2024-01-01T00:15:00.000000000Z", 1.10, 1.05, 1.09, 1.08, 100),
            ]
        }
        mock_get.return_value = mock_response

        df = self.engine.fetch_historical_candles("EUR_USD", "M15", 2)
        self.assertEqual(len(df), 1)

    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_validate_candle_accepts_valid(self):
        """Test that a valid candle passes validation."""
        good_candle = {"open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11, "volume": 500}
        self.assertTrue(self.engine._validate_candle(good_candle))

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_get_bid_ask_returns_tuple(self, mock_get):
        """Test that get_bid_ask returns a tuple of floats."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "prices": [
                {
                    "instrument": "EUR_USD",
                    "bids": [{"price": "1.10000"}],
                    "asks": [{"price": "1.10015"}],
                }
            ]
        }
        mock_get.return_value = mock_response

        bid, ask = self.engine.get_bid_ask("EUR_USD")
        self.assertIsInstance(bid, float)
        self.assertIsInstance(ask, float)
        self.assertGreater(ask, bid)

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_get_bid_ask_handles_error(self, mock_get):
        """Test that get_bid_ask returns (0.0, 0.0) on API error."""
        mock_get.side_effect = Exception("Network error")
        bid, ask = self.engine.get_bid_ask("EUR_USD")
        self.assertEqual(bid, 0.0)
        self.assertEqual(ask, 0.0)

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_get_current_spread_handles_error(self, mock_get):
        """Test that get_current_spread returns 0.0 on API error."""
        mock_get.side_effect = Exception("Network error")
        spread = self.engine.get_current_spread("EUR_USD")
        self.assertEqual(spread, 0.0)

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_get_current_spread_empty_prices(self, mock_get):
        """Test that get_current_spread handles empty prices list."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"prices": []}
        mock_get.return_value = mock_response

        spread = self.engine.get_current_spread("EUR_USD")
        self.assertEqual(spread, 0.0)

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_get_current_spread_xau(self, mock_get):
        """Test spread calculation for XAU_USD uses correct multiplier."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "prices": [
                {
                    "instrument": "XAU_USD",
                    "bids": [{"price": "2000.00"}],
                    "asks": [{"price": "2000.50"}],
                }
            ]
        }
        mock_get.return_value = mock_response

        spread = self.engine.get_current_spread("XAU_USD")
        self.assertAlmostEqual(spread, 5.0, places=1)

    @patch("src.data_engine.time.sleep")
    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_fetch_historical_api_failure_retries(self, mock_get, mock_sleep):
        """Test that fetch_historical_candles retries on API failure."""
        mock_get.side_effect = Exception("Connection timeout")

        df = self.engine.fetch_historical_candles("EUR_USD", "M15", 10)
        self.assertTrue(df.empty)
        self.assertEqual(mock_get.call_count, 3)

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_fetch_historical_skips_incomplete_candles(self, mock_get):
        """Test that incomplete candles are skipped."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candles": [
                _make_oanda_candle("2024-01-01T00:00:00Z", 1.10, 1.12, 1.09, 1.11, 500, complete=True),
                _make_oanda_candle("2024-01-01T00:15:00Z", 1.10, 1.12, 1.09, 1.11, 500, complete=False),
            ]
        }
        mock_get.return_value = mock_response

        df = self.engine.fetch_historical_candles("EUR_USD", "M15", 2)
        self.assertEqual(len(df), 1)

    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_store_and_get_candles_from_db(self):
        """Test storing and retrieving candles from SQLite."""
        # Use a temp db
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.engine.db_path = tmp.name
        self.engine._init_db()

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
        os.unlink(tmp.name)

    @patch("src.data_engine.requests.get")
    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_warm_up_calls_fetch(self, mock_get):
        """Test that warm_up fetches candles for all instrument/timeframe combos."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"candles": []}
        mock_get.return_value = mock_response

        self.engine.warm_up(["EUR_USD"], ["M15", "H1"])
        self.assertEqual(mock_get.call_count, 2)

    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_calculate_spread_pips_standard(self):
        """Test pip calculation for standard pairs."""
        spread = self.engine._calculate_spread_pips("EUR_USD", 1.10000, 1.10015)
        self.assertAlmostEqual(spread, 1.5, places=1)

    @patch.dict(os.environ, {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"})
    def test_init_creates_live_url(self):
        """Test that live environment uses live URL."""
        config = _load_config()
        config["broker"]["environment"] = "live"
        engine = DataEngine(config)
        self.assertIn("fxtrade", engine.base_url)


if __name__ == "__main__":
    unittest.main()
