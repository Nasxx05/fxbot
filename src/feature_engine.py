"""Feature engine module — computes all technical indicators and market features."""

import numpy as np
import pandas as pd


class FeatureEngine:
    """Computes technical indicators, swing points, session tags, and market bias."""

    def __init__(self, config: dict):
        """Initialize the FeatureEngine with configuration.

        Args:
            config: Full bot configuration dictionary.
        """
        strategy = config.get("strategy", {})
        self.atr_period = strategy.get("atr_period", 14)
        self.atr_average_period = strategy.get("atr_average_period", 50)
        self.pivot_lookback = strategy.get("pivot_lookback", 5)
        self.volatility_multiplier = strategy.get("volatility_multiplier", 1.3)
        self.ranging_candle_threshold = strategy.get("ranging_candle_threshold", 10)
        self.liquidity_threshold_pct = strategy.get("liquidity_threshold_pct", 0.02)

        sessions = config.get("sessions", {})
        self.asian_start = self._parse_time(sessions.get("asian_start_utc", "00:00"))
        self.asian_end = self._parse_time(sessions.get("asian_end_utc", "07:00"))
        self.london_start = self._parse_time(sessions.get("london_start_utc", "07:00"))
        self.london_end = self._parse_time(sessions.get("london_end_utc", "12:00"))
        self.ny_start = self._parse_time(sessions.get("ny_start_utc", "12:00"))
        self.ny_end = self._parse_time(sessions.get("ny_end_utc", "20:00"))
        self.overlap_start = self._parse_time(sessions.get("overlap_start_utc", "12:00"))
        self.overlap_end = self._parse_time(sessions.get("overlap_end_utc", "16:00"))

    @staticmethod
    def _parse_time(time_str: str) -> int:
        """Parse a HH:MM time string into minutes since midnight.

        Args:
            time_str: Time string in HH:MM format.

        Returns:
            Total minutes since midnight.
        """
        parts = time_str.split(":")
        return int(parts[0]) * 60 + int(parts[1])

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features on the DataFrame.

        This is the only method that should be called externally after each new candle.

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume.

        Returns:
            DataFrame with all computed feature columns added.
        """
        df = df.copy()
        df = self.compute_atr(df)
        df = self.compute_atr_average(df)
        df = self.compute_swing_points(df)
        df = self.compute_market_bias(df)
        df = self.compute_candle_properties(df)
        df = self.compute_volatility_regime(df)
        df = self.compute_session_tag(df)
        df = self.compute_equal_levels(df)
        df = self.compute_previous_day_levels(df)
        df = self.compute_session_levels(df)
        return df

    def compute_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the Average True Range (ATR).

        True Range = max of: (high - low), abs(high - prev_close), abs(low - prev_close).
        ATR = rolling mean of True Range over atr_period.

        Args:
            df: DataFrame with high, low, close columns.

        Returns:
            DataFrame with atr_14 column added.
        """
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(window=self.atr_period, min_periods=1).mean()
        return df

    def compute_atr_average(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the rolling average of ATR.

        Args:
            df: DataFrame with atr_14 column.

        Returns:
            DataFrame with atr_average column added.
        """
        df["atr_average"] = df["atr_14"].rolling(
            window=self.atr_average_period, min_periods=1
        ).mean()
        return df

    def compute_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify swing highs and swing lows using pivot lookback.

        A swing high has a higher high than N candles before and after it.
        A swing low has a lower low than N candles before and after it.

        Args:
            df: DataFrame with high, low columns.

        Returns:
            DataFrame with is_swing_high, is_swing_low, swing_high_price,
            swing_low_price columns added.
        """
        n = self.pivot_lookback
        highs = df["high"].values
        lows = df["low"].values
        length = len(df)

        is_swing_high = np.zeros(length, dtype=bool)
        is_swing_low = np.zeros(length, dtype=bool)

        for i in range(n, length - n):
            if all(highs[i] > highs[i - j] for j in range(1, n + 1)) and \
               all(highs[i] > highs[i + j] for j in range(1, n + 1)):
                is_swing_high[i] = True

            if all(lows[i] < lows[i - j] for j in range(1, n + 1)) and \
               all(lows[i] < lows[i + j] for j in range(1, n + 1)):
                is_swing_low[i] = True

        df["is_swing_high"] = is_swing_high
        df["is_swing_low"] = is_swing_low
        df["swing_high_price"] = np.where(is_swing_high, highs, np.nan)
        df["swing_low_price"] = np.where(is_swing_low, lows, np.nan)

        return df

    def compute_market_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Determine market bias from swing point structure.

        BULLISH: last 2 swing highs rising AND last 2 swing lows rising.
        BEARISH: last 2 swing highs falling AND last 2 swing lows falling.
        NEUTRAL: anything else.

        Args:
            df: DataFrame with swing_high_price, swing_low_price columns.

        Returns:
            DataFrame with market_bias column added.
        """
        swing_highs = df["swing_high_price"].dropna().values
        swing_lows = df["swing_low_price"].dropna().values

        bias = "NEUTRAL"

        if len(swing_highs) >= 4 and len(swing_lows) >= 4:
            last_4_highs = swing_highs[-4:]
            last_4_lows = swing_lows[-4:]

            highs_rising = last_4_highs[-1] > last_4_highs[-2] and last_4_highs[-2] > last_4_highs[-3]
            lows_rising = last_4_lows[-1] > last_4_lows[-2] and last_4_lows[-2] > last_4_lows[-3]

            highs_falling = last_4_highs[-1] < last_4_highs[-2] and last_4_highs[-2] < last_4_highs[-3]
            lows_falling = last_4_lows[-1] < last_4_lows[-2] and last_4_lows[-2] < last_4_lows[-3]

            if highs_rising and lows_rising:
                bias = "BULLISH"
            elif highs_falling and lows_falling:
                bias = "BEARISH"

        df["market_bias"] = bias
        return df

    def compute_candle_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute candle body, wick, and ratio properties.

        Args:
            df: DataFrame with open, high, low, close columns.

        Returns:
            DataFrame with body_size, total_range, upper_wick, lower_wick,
            body_ratio, upper_wick_ratio, lower_wick_ratio, is_bullish,
            is_bearish columns added.
        """
        df["body_size"] = (df["close"] - df["open"]).abs()
        df["total_range"] = df["high"] - df["low"]
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

        safe_range = df["total_range"].replace(0, np.nan)
        df["body_ratio"] = (df["body_size"] / safe_range).fillna(0)
        df["upper_wick_ratio"] = (df["upper_wick"] / safe_range).fillna(0)
        df["lower_wick_ratio"] = (df["lower_wick"] / safe_range).fillna(0)

        df["is_bullish"] = df["close"] > df["open"]
        df["is_bearish"] = df["close"] < df["open"]

        return df

    def compute_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify the current volatility regime.

        EXPANDING: atr_14 > atr_average * volatility_multiplier.
        RANGING: atr_14 below atr_average for N consecutive candles.
        NORMAL: everything else.

        Args:
            df: DataFrame with atr_14, atr_average columns.

        Returns:
            DataFrame with volatility_regime column added.
        """
        atr_arr = df["atr_14"].values
        avg_arr = df["atr_average"].values
        n = len(df)
        regimes = np.empty(n, dtype=object)

        below_count = 0
        for i in range(n):
            atr = atr_arr[i]
            avg = avg_arr[i]

            if np.isnan(atr) or np.isnan(avg):
                regimes[i] = "NORMAL"
                below_count = 0
            elif atr > avg * self.volatility_multiplier:
                regimes[i] = "EXPANDING"
                below_count = 0
            elif atr < avg:
                below_count += 1
                regimes[i] = "RANGING" if below_count >= self.ranging_candle_threshold else "NORMAL"
            else:
                below_count = 0
                regimes[i] = "NORMAL"

        df["volatility_regime"] = regimes
        return df

    def compute_session_tag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag each candle with its trading session based on UTC time.

        OVERLAP takes priority over LONDON and NEW_YORK.

        Args:
            df: DataFrame with a timezone-aware UTC timestamp column.

        Returns:
            DataFrame with session column added.
        """
        sessions = []
        for ts in df["timestamp"]:
            if hasattr(ts, "hour"):
                minutes = ts.hour * 60 + ts.minute
            else:
                t = pd.to_datetime(ts, utc=True)
                minutes = t.hour * 60 + t.minute

            if self.overlap_start <= minutes < self.overlap_end:
                sessions.append("OVERLAP")
            elif self.london_start <= minutes < self.london_end:
                sessions.append("LONDON")
            elif self.ny_start <= minutes < self.ny_end:
                sessions.append("NEW_YORK")
            elif self.asian_start <= minutes < self.asian_end:
                sessions.append("ASIAN")
            else:
                sessions.append("OFF_HOURS")

        df["session"] = sessions
        return df

    def compute_equal_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect equal highs and equal lows from recent swing points.

        Equal highs/lows are swing points within liquidity_threshold_pct of each other.
        Uses a vectorized approach: for each swing point, check if any other swing
        point within the lookback window is within the threshold.

        Args:
            df: DataFrame with swing_high_price, swing_low_price columns.

        Returns:
            DataFrame with near_equal_high, near_equal_low, equal_high_level,
            equal_low_level columns added.
        """
        df["near_equal_high"] = False
        df["near_equal_low"] = False
        df["equal_high_level"] = np.nan
        df["equal_low_level"] = np.nan

        if df.empty:
            return df

        # Get swing point indices and values
        sh_mask = df["swing_high_price"].notna()
        sl_mask = df["swing_low_price"].notna()

        sh_indices = np.where(sh_mask)[0]
        sl_indices = np.where(sl_mask)[0]
        sh_prices = df.loc[sh_mask, "swing_high_price"].values
        sl_prices = df.loc[sl_mask, "swing_low_price"].values

        # For each pair of consecutive swing highs, mark equal levels
        for i in range(1, len(sh_prices)):
            threshold = sh_prices[i] * self.liquidity_threshold_pct
            if abs(sh_prices[i] - sh_prices[i - 1]) <= threshold:
                level = (sh_prices[i] + sh_prices[i - 1]) / 2
                start = sh_indices[i]
                end = min(start + 96, len(df))  # Propagate for ~1 day of M15 candles
                df.iloc[start:end, df.columns.get_loc("near_equal_high")] = True
                df.iloc[start:end, df.columns.get_loc("equal_high_level")] = level

        for i in range(1, len(sl_prices)):
            threshold = sl_prices[i] * self.liquidity_threshold_pct
            if abs(sl_prices[i] - sl_prices[i - 1]) <= threshold:
                level = (sl_prices[i] + sl_prices[i - 1]) / 2
                start = sl_indices[i]
                end = min(start + 96, len(df))
                df.iloc[start:end, df.columns.get_loc("near_equal_low")] = True
                df.iloc[start:end, df.columns.get_loc("equal_low_level")] = level

        return df

    def compute_previous_day_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the previous day's high and low levels per candle.

        Each candle gets the high/low from the calendar day before its own date.

        Args:
            df: DataFrame with timezone-aware UTC timestamp, high, low columns.

        Returns:
            DataFrame with prev_day_high, prev_day_low columns added.
        """
        df["prev_day_high"] = np.nan
        df["prev_day_low"] = np.nan

        if "timestamp" not in df.columns or df.empty:
            return df

        ts = df["timestamp"]
        if not hasattr(ts.iloc[0], "date"):
            ts = pd.to_datetime(ts, utc=True)

        dates = ts.dt.date

        # Compute daily high/low using groupby
        daily_stats = df.groupby(dates).agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
        )

        # Shift to get previous day's values
        daily_stats["prev_high"] = daily_stats["day_high"].shift(1)
        daily_stats["prev_low"] = daily_stats["day_low"].shift(1)

        # Map back to each row
        date_to_prev_high = daily_stats["prev_high"].to_dict()
        date_to_prev_low = daily_stats["prev_low"].to_dict()

        df["prev_day_high"] = dates.map(date_to_prev_high).values
        df["prev_day_low"] = dates.map(date_to_prev_low).values

        return df

    def compute_session_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-day session highs and lows for Asian and London sessions.

        Each candle gets the session levels from its own calendar day.

        Args:
            df: DataFrame with timezone-aware UTC timestamp, high, low columns.

        Returns:
            DataFrame with asian_session_high, asian_session_low,
            london_session_high, london_session_low columns added.
        """
        df["asian_session_high"] = np.nan
        df["asian_session_low"] = np.nan
        df["london_session_high"] = np.nan
        df["london_session_low"] = np.nan

        if "timestamp" not in df.columns or df.empty:
            return df

        if "session" not in df.columns:
            df = self.compute_session_tag(df)

        ts = df["timestamp"]
        if not hasattr(ts.iloc[0], "date"):
            ts = pd.to_datetime(ts, utc=True)
        dates = ts.dt.date

        for session_name, h_col, l_col in [
            ("ASIAN", "asian_session_high", "asian_session_low"),
            ("LONDON", "london_session_high", "london_session_low"),
        ]:
            mask = df["session"] == session_name
            session_df = df.loc[mask]
            if session_df.empty:
                continue

            session_dates = dates[mask]
            day_highs = session_df.groupby(session_dates)["high"].max().to_dict()
            day_lows = session_df.groupby(session_dates)["low"].min().to_dict()

            df[h_col] = dates.map(day_highs).values
            df[l_col] = dates.map(day_lows).values

        return df
