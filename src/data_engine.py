"""Data engine module — connects to OANDA v20 API, fetches and streams candle data."""

import os
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from src.logger import BotLogger

load_dotenv()


class DataEngine:
    """Handles all data operations: OANDA API communication, candle storage, and streaming."""

    def __init__(self, config: dict):
        """Initialize the DataEngine with configuration.

        Args:
            config: Full bot configuration dictionary.
        """
        self.config = config
        self.logger = BotLogger(config)

        self.api_key = os.environ.get("OANDA_API_KEY", "")
        self.account_id = os.environ.get("OANDA_ACCOUNT_ID", "")

        broker_config = config.get("broker", {})
        env = broker_config.get("environment", "demo")
        if env == "live":
            self.base_url = broker_config.get("base_url_live", "https://api-fxtrade.oanda.com")
            self.stream_url = "https://stream-fxtrade.oanda.com"
        else:
            self.base_url = broker_config.get("base_url_demo", "https://api-fxpractice.oanda.com")
            self.stream_url = "https://stream-fxpractice.oanda.com"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.db_path = os.path.join("data", "historical", "candles.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the candles table and indexes if they do not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                spread REAL,
                created_at TEXT NOT NULL,
                UNIQUE(instrument, timeframe, timestamp)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_candles_lookup
            ON candles (instrument, timeframe, timestamp)
        """)
        conn.commit()
        conn.close()

    def _validate_candle(self, row: dict) -> bool:
        """Validate a single candle's OHLCV data.

        Args:
            row: Dictionary with open, high, low, close, volume keys.

        Returns:
            True if the candle is valid, False otherwise.
        """
        required = ["open", "high", "low", "close", "volume"]
        for key in required:
            if row.get(key) is None:
                return False

        h, l, o, c = row["high"], row["low"], row["open"], row["close"]
        if h < l or h < o or h < c or l > o or l > c:
            return False

        return True

    def fetch_historical_candles(self, instrument: str, timeframe: str, count: int) -> pd.DataFrame:
        """Fetch historical candles from OANDA v20 REST API.

        Args:
            instrument: OANDA instrument name (e.g. EUR_USD).
            timeframe: Candle granularity (e.g. M15, H1, H4).
            count: Number of candles to fetch.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume,
            sorted by timestamp ascending.
        """
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {
            "count": count,
            "granularity": timeframe,
            "price": "MBA",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as e:
                self.logger.log_error(
                    "data_engine",
                    f"API call failed (attempt {attempt + 1}/{max_retries}): {e}",
                    {"instrument": instrument, "timeframe": timeframe},
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        candles = []
        for c in data.get("candles", []):
            if not c.get("complete", False):
                continue
            mid = c.get("mid", {})
            row = {
                "timestamp": c["time"],
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            }
            if self._validate_candle(row):
                candles.append(row)

        df = pd.DataFrame(candles)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        self._store_candles(instrument, timeframe, df)

        return df

    def _store_candles(self, instrument: str, timeframe: str, df: pd.DataFrame):
        """Store validated candles into SQLite, skipping duplicates.

        Args:
            instrument: OANDA instrument name.
            timeframe: Candle granularity.
            df: DataFrame of candles to store.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        for _, row in df.iterrows():
            try:
                cursor.execute(
                    """INSERT OR IGNORE INTO candles
                    (instrument, timeframe, timestamp, open, high, low, close, volume, spread, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        instrument,
                        timeframe,
                        row["timestamp"].isoformat(),
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                        row.get("spread"),
                        now,
                    ),
                )
            except sqlite3.Error as e:
                self.logger.log_error("data_engine", f"SQLite insert error: {e}")

        conn.commit()
        conn.close()

    def get_candles_from_db(self, instrument: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Query the most recent N candles from SQLite.

        Args:
            instrument: OANDA instrument name.
            timeframe: Candle granularity.
            limit: Maximum number of candles to return.

        Returns:
            DataFrame sorted by timestamp ascending.
        """
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT timestamp, open, high, low, close, volume, spread
            FROM candles
            WHERE instrument = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(instrument, timeframe, limit))
        conn.close()

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def get_current_spread(self, instrument: str) -> float:
        """Get the current spread in pips for an instrument.

        Args:
            instrument: OANDA instrument name.

        Returns:
            Spread as a positive float in pips.
        """
        url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            prices = data.get("prices", [])
            if not prices:
                self.logger.log_error("data_engine", f"No pricing data for {instrument}")
                return 0.0

            price = prices[0]
            bid = float(price["bids"][0]["price"])
            ask = float(price["asks"][0]["price"])

            return self._calculate_spread_pips(instrument, bid, ask)

        except Exception as e:
            self.logger.log_error(
                "data_engine",
                f"Spread fetch error: {e}",
                {"instrument": instrument},
            )
            return 0.0

    def _calculate_spread_pips(self, instrument: str, bid: float, ask: float) -> float:
        """Calculate spread in pips based on instrument type.

        Args:
            instrument: OANDA instrument name.
            bid: Current bid price.
            ask: Current ask price.

        Returns:
            Spread in pips as a positive float.
        """
        raw_spread = ask - bid
        if "JPY" in instrument:
            return abs(raw_spread * 100)
        elif instrument == "XAU_USD":
            return abs(raw_spread * 10)
        else:
            return abs(raw_spread * 10000)

    def get_bid_ask(self, instrument: str) -> tuple:
        """Get the current bid and ask prices for an instrument.

        Args:
            instrument: OANDA instrument name.

        Returns:
            Tuple of (bid, ask) as floats.
        """
        url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            prices = data.get("prices", [])
            if not prices:
                self.logger.log_error("data_engine", f"No pricing data for {instrument}")
                return (0.0, 0.0)

            price = prices[0]
            bid = float(price["bids"][0]["price"])
            ask = float(price["asks"][0]["price"])
            return (bid, ask)

        except Exception as e:
            self.logger.log_error(
                "data_engine",
                f"Bid/ask fetch error: {e}",
                {"instrument": instrument},
            )
            return (0.0, 0.0)

    def stream_candles(self, instruments: list, callback: callable):
        """Open a streaming connection to OANDA for real-time pricing.

        Detects when a new M15 candle completes and calls the callback.

        Args:
            instruments: List of OANDA instrument names.
            callback: Function to call with (instrument, candle_data) when a candle completes.
        """
        url = f"{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream"
        params = {"instruments": ",".join(instruments)}

        reconnect_delay = 5
        current_candle_times = {}

        while True:
            try:
                self.logger.log("INFO", "data_engine", "Opening streaming connection", {"instruments": instruments})
                response = requests.get(url, headers=self.headers, params=params, stream=True, timeout=None)
                response.raise_for_status()

                reconnect_delay = 5

                for line in response.iter_lines():
                    if not line:
                        continue

                    import json
                    try:
                        tick = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                    if tick.get("type") != "PRICE":
                        continue

                    instrument = tick.get("instrument")
                    tick_time = pd.to_datetime(tick.get("time"), utc=True)

                    candle_open_time = tick_time.floor("15min")

                    prev_candle_time = current_candle_times.get(instrument)
                    if prev_candle_time and candle_open_time > prev_candle_time:
                        candle_data = {
                            "instrument": instrument,
                            "timeframe": "M15",
                            "candle_open_time": prev_candle_time.isoformat(),
                        }
                        try:
                            callback(instrument, candle_data)
                        except Exception as e:
                            self.logger.log_error("data_engine", f"Callback error: {e}")

                    current_candle_times[instrument] = candle_open_time

            except requests.RequestException as e:
                self.logger.log_error(
                    "data_engine",
                    f"Stream disconnected: {e}",
                    {"reconnect_delay": reconnect_delay},
                )
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)
            except Exception as e:
                self.logger.log_error("data_engine", f"Stream fatal error: {e}")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

    def warm_up(self, instruments: list, timeframes: list):
        """Fetch historical candles for all instrument/timeframe combinations on startup.

        Args:
            instruments: List of OANDA instrument names.
            timeframes: List of timeframe granularities.
        """
        for instrument in instruments:
            for timeframe in timeframes:
                self.logger.log(
                    "INFO",
                    "data_engine",
                    f"Warming up {instrument} {timeframe}",
                )
                self.fetch_historical_candles(instrument, timeframe, count=500)
                self.logger.log(
                    "INFO",
                    "data_engine",
                    f"Warm-up complete: {instrument} {timeframe}",
                )
