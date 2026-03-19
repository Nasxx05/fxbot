"""Data engine module — connects to MetaTrader 5 terminal, fetches and polls candle data."""

import os
import sqlite3
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.logger import BotLogger

load_dotenv()

# MT5 timeframe mapping
MT5_TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}


class DataEngine:
    """Handles all data operations: MT5 connection, candle fetching, and polling."""

    def __init__(self, config: dict):
        """Initialize the DataEngine with configuration.

        Args:
            config: Full bot configuration dictionary.
        """
        self.config = config
        self.logger = BotLogger(config)

        # Initialize MT5 connection
        login = os.environ.get("MT5_LOGIN", "")
        password = os.environ.get("MT5_PASSWORD", "")
        server = os.environ.get("MT5_SERVER", "")

        init_kwargs = {}
        if login:
            init_kwargs["login"] = int(login)
        if password:
            init_kwargs["password"] = password
        if server:
            init_kwargs["server"] = server

        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            raise RuntimeError(
                f"MT5 initialize failed: {error}. "
                "Make sure MetaTrader 5 terminal is open and logged in."
            )

        account_info = mt5.account_info()
        if account_info is None:
            raise RuntimeError("MT5: Could not retrieve account info. Check login credentials.")

        self.logger.log("INFO", "data_engine",
                        f"MT5 connected — Account #{account_info.login} "
                        f"on {account_info.server}")

        self.db_path = os.path.join("data", "historical", "candles.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @staticmethod
    def to_mt5_symbol(instrument: str) -> str:
        """Convert config instrument name to MT5 symbol.

        EUR_USD -> EURUSD, XAU_USD -> XAUUSD
        """
        return instrument.replace("_", "")

    @staticmethod
    def from_mt5_symbol(symbol: str) -> str:
        """Convert MT5 symbol back to config instrument name.

        EURUSD -> EUR_USD, XAUUSD -> XAU_USD
        """
        return symbol[:3] + "_" + symbol[3:]

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
        """Fetch historical candles from MetaTrader 5.

        Args:
            instrument: Config instrument name (e.g. EUR_USD).
            timeframe: Candle granularity (e.g. M15, H1, H4).
            count: Number of candles to fetch.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume,
            sorted by timestamp ascending.
        """
        mt5_symbol = self.to_mt5_symbol(instrument)
        mt5_tf = MT5_TIMEFRAMES.get(timeframe)
        if mt5_tf is None:
            self.logger.log_error("data_engine", f"Unknown timeframe: {timeframe}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Ensure symbol is available in Market Watch
        if not mt5.symbol_select(mt5_symbol, True):
            self.logger.log_error("data_engine", f"Symbol {mt5_symbol} not available in MT5")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            self.logger.log_error("data_engine",
                                  f"MT5 copy_rates failed: {error}",
                                  {"instrument": instrument, "timeframe": timeframe})
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert numpy structured array to DataFrame
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={
            "tick_volume": "volume",
        })
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Remove the last (incomplete) candle
        if len(df) > 1:
            df = df.iloc[:-1]

        # Validate candles
        valid_rows = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            if self._validate_candle(row_dict):
                valid_rows.append(row_dict)

        df = pd.DataFrame(valid_rows)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        self._store_candles(instrument, timeframe, df)
        return df

    def _store_candles(self, instrument: str, timeframe: str, df: pd.DataFrame):
        """Store validated candles into SQLite, skipping duplicates.

        Args:
            instrument: Config instrument name.
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
            instrument: Config instrument name.
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
            instrument: Config instrument name.

        Returns:
            Spread as a positive float in pips.
        """
        mt5_symbol = self.to_mt5_symbol(instrument)

        try:
            info = mt5.symbol_info(mt5_symbol)
            if info is None:
                self.logger.log_error("data_engine", f"No symbol info for {mt5_symbol}")
                return 0.0

            spread_points = info.spread

            # Convert points to pips
            # For JPY pairs: 1 point = 1 pip (3-digit pricing)
            # For most pairs: 1 pip = 10 points (5-digit pricing)
            # For XAU_USD: 1 pip = 10 points
            if "JPY" in instrument:
                return spread_points / 10.0
            else:
                return spread_points / 10.0

        except Exception as e:
            self.logger.log_error("data_engine",
                                  f"Spread fetch error: {e}",
                                  {"instrument": instrument})
            return 0.0

    def _calculate_spread_pips(self, instrument: str, bid: float, ask: float) -> float:
        """Calculate spread in pips based on instrument type.

        Args:
            instrument: Config instrument name.
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
            instrument: Config instrument name.

        Returns:
            Tuple of (bid, ask) as floats.
        """
        mt5_symbol = self.to_mt5_symbol(instrument)

        try:
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                self.logger.log_error("data_engine", f"No tick data for {mt5_symbol}")
                return (0.0, 0.0)

            return (tick.bid, tick.ask)

        except Exception as e:
            self.logger.log_error("data_engine",
                                  f"Bid/ask fetch error: {e}",
                                  {"instrument": instrument})
            return (0.0, 0.0)

    def stream_candles(self, instruments: list, callback: callable):
        """Poll for new M15 candle completions by checking every 5 seconds.

        MT5 does not have a push streaming API, so we poll the last 2 candles
        and detect when a new completed candle appears.

        Args:
            instruments: List of config instrument names.
            callback: Function to call with (instrument, candle_data) when a candle completes.
        """
        self.logger.log("INFO", "data_engine",
                        "Starting candle polling loop",
                        {"instruments": instruments, "poll_interval": 5})

        last_seen_candle_times = {}

        while True:
            for instrument in instruments:
                try:
                    mt5_symbol = self.to_mt5_symbol(instrument)
                    mt5_tf = MT5_TIMEFRAMES.get("M15", mt5.TIMEFRAME_M15)

                    rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, 2)
                    if rates is None or len(rates) < 2:
                        continue

                    # The second-to-last candle is the most recent completed one
                    completed_candle = rates[-2]
                    candle_time = pd.Timestamp(completed_candle["time"], unit="s", tz="UTC")

                    prev_time = last_seen_candle_times.get(instrument)
                    if prev_time is None:
                        last_seen_candle_times[instrument] = candle_time
                        continue

                    if candle_time > prev_time:
                        last_seen_candle_times[instrument] = candle_time
                        candle_data = {
                            "instrument": instrument,
                            "timeframe": "M15",
                            "candle_open_time": candle_time.isoformat(),
                        }
                        try:
                            callback(instrument, candle_data)
                        except Exception as e:
                            self.logger.log_error("data_engine", f"Callback error: {e}")

                except Exception as e:
                    self.logger.log_error("data_engine",
                                          f"Poll error for {instrument}: {e}")

            time.sleep(5)

    def get_open_positions(self) -> list:
        """Get all open positions from MT5.

        Returns:
            List of position dicts with symbol, ticket, type, volume, etc.
        """
        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "symbol": self.from_mt5_symbol(pos.symbol),
                "mt5_symbol": pos.symbol,
                "type": "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "time": datetime.fromtimestamp(pos.time, tz=timezone.utc).isoformat(),
            })
        return result

    def warm_up(self, instruments: list, timeframes: list):
        """Fetch historical candles for all instrument/timeframe combinations on startup.

        Args:
            instruments: List of config instrument names.
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

    def shutdown(self):
        """Cleanly shut down the MT5 connection."""
        mt5.shutdown()
        self.logger.log("INFO", "data_engine", "MT5 connection closed")
