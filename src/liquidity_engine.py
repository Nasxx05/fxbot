"""Liquidity engine module — identifies liquidity pools and sweep zones."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.logger import BotLogger


class LiquidityEngine:
    """Detects liquidity zones and confirms sweep events."""

    def __init__(self, config: dict, logger: BotLogger):
        """Initialize the LiquidityEngine with configuration and logger.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance for structured logging.
        """
        self.config = config
        self.logger = logger
        strategy = config.get("strategy", {})
        self.sweep_wick_body_ratio = strategy.get("sweep_wick_body_ratio", 1.5)
        self.sweep_candle_lookback = strategy.get("sweep_candle_lookback", 3)
        self.liquidity_threshold_pct = strategy.get("liquidity_threshold_pct", 0.02)
        self.atr_invalidation_multiple = 2.0

    def detect_liquidity_zones(self, df: pd.DataFrame) -> list:
        """Scan the DataFrame and return a list of active liquidity zones.

        Each zone is a dict with type, price, strength, created_at, and active keys.
        Uses columns pre-computed by FeatureEngine.

        Args:
            df: DataFrame with FeatureEngine columns computed.

        Returns:
            List of zone dicts.
        """
        zones = []
        if df.empty:
            self.logger.log("DEBUG", "liquidity", "Empty DataFrame, no zones detected")
            return zones

        last = df.iloc[-1]
        ts_now = last.get("timestamp", datetime.now(timezone.utc))

        zone_map = {
            "EQUAL_HIGH": "equal_high_level",
            "EQUAL_LOW": "equal_low_level",
            "PREV_DAY_HIGH": "prev_day_high",
            "PREV_DAY_LOW": "prev_day_low",
            "ASIAN_HIGH": "asian_session_high",
            "ASIAN_LOW": "asian_session_low",
            "LONDON_HIGH": "london_session_high",
            "LONDON_LOW": "london_session_low",
        }

        # Also add recent swing highs and lows as zones
        if "swing_high_price" in df.columns:
            recent = df.tail(20)
            for _, row in recent[recent["swing_high_price"].notna()].iterrows():
                price = float(row["swing_high_price"])
                swept = self._is_swept(df, price, "SWING_HIGH")
                zones.append({
                    "type": "SWING_HIGH",
                    "price": price,
                    "strength": 1,
                    "created_at": ts_now,
                    "active": not swept,
                })
        if "swing_low_price" in df.columns:
            recent = df.tail(20)
            for _, row in recent[recent["swing_low_price"].notna()].iterrows():
                price = float(row["swing_low_price"])
                swept = self._is_swept(df, price, "SWING_LOW")
                zones.append({
                    "type": "SWING_LOW",
                    "price": price,
                    "strength": 1,
                    "created_at": ts_now,
                    "active": not swept,
                })

        for zone_type, col in zone_map.items():
            if col in df.columns:
                price = last.get(col)
                if price is not None and not (isinstance(price, float) and np.isnan(price)):
                    strength = self._count_touches(df, float(price), zone_type)
                    swept = self._is_swept(df, float(price), zone_type)
                    zones.append({
                        "type": zone_type,
                        "price": float(price),
                        "strength": strength,
                        "created_at": ts_now,
                        "active": not swept,
                    })

        self.logger.log("DEBUG", "liquidity", f"Detected {len(zones)} liquidity zones",
                        {"active": sum(1 for z in zones if z["active"])})
        return zones

    def _count_touches(self, df: pd.DataFrame, price: float, zone_type: str) -> int:
        """Count how many candles have tested a price level.

        Args:
            df: OHLC DataFrame.
            price: The price level to check.
            zone_type: The type of zone (HIGH or LOW variant).

        Returns:
            Number of touches.
        """
        threshold = price * self.liquidity_threshold_pct
        if "HIGH" in zone_type:
            touches = ((df["high"] - price).abs() <= threshold).sum()
        else:
            touches = ((df["low"] - price).abs() <= threshold).sum()
        return int(touches)

    def _is_swept(self, df: pd.DataFrame, price: float, zone_type: str) -> bool:
        """Check if a zone has already been swept by a prior candle.

        Args:
            df: OHLC DataFrame.
            price: The zone price level.
            zone_type: The type of zone.

        Returns:
            True if already swept.
        """
        if len(df) < 2:
            return False
        last = df.iloc[-1]
        if "HIGH" in zone_type:
            return bool(last["high"] > price and last["close"] < price)
        else:
            return bool(last["low"] < price and last["close"] > price)

    def detect_sweep(self, df: pd.DataFrame, zone: dict) -> dict or None:
        """Check if recent candles have swept a given liquidity zone.

        A sweep requires:
        1. Wick goes beyond the zone price.
        2. Body closes back inside the zone.
        3. Sweep wick >= sweep_wick_body_ratio * body size.

        Args:
            df: DataFrame with OHLC data.
            zone: A zone dict from detect_liquidity_zones.

        Returns:
            Sweep dict if confirmed, None otherwise.
        """
        if df.empty or not zone.get("active", False):
            return None

        price = zone["price"]
        zone_type = zone["type"]
        is_high_zone = "HIGH" in zone_type

        lookback = min(self.sweep_candle_lookback, len(df))
        tail = df.iloc[-lookback:]

        opens = tail["open"].values
        highs = tail["high"].values
        lows = tail["low"].values
        closes = tail["close"].values
        indices = tail.index.values

        for j in range(len(opens)):
            body_size = abs(closes[j] - opens[j])
            if body_size == 0:
                continue

            if is_high_zone:
                if highs[j] <= price or closes[j] >= price:
                    continue
                sweep_wick = highs[j] - max(closes[j], opens[j])
            else:
                if lows[j] >= price or closes[j] <= price:
                    continue
                sweep_wick = min(closes[j], opens[j]) - lows[j]

            if sweep_wick < self.sweep_wick_body_ratio * body_size:
                continue

            direction = "BULLISH_SWEEP" if not is_high_zone else "BEARISH_SWEEP"
            ts_val = tail["timestamp"].iloc[j] if "timestamp" in tail.columns else datetime.now(timezone.utc)
            return {
                "zone": zone,
                "sweep_candle_index": indices[j],
                "sweep_high": float(highs[j]),
                "sweep_low": float(lows[j]),
                "direction": direction,
                "timestamp": ts_val,
            }

        return None

    def get_active_zones(self, df: pd.DataFrame) -> list:
        """Return only active liquidity zones.

        Args:
            df: DataFrame with FeatureEngine columns computed.

        Returns:
            List of active zone dicts.
        """
        all_zones = self.detect_liquidity_zones(df)
        active = [z for z in all_zones if z["active"]]
        self.logger.log("DEBUG", "liquidity", f"Active zones: {len(active)}")
        return active

    def invalidate_zone(self, zone: dict, df: pd.DataFrame) -> bool:
        """Check if a zone should be deactivated.

        A zone is invalidated if price has moved more than 2 ATR beyond it
        without a sweep rejection.

        Args:
            zone: A zone dict.
            df: DataFrame with atr_14 column.

        Returns:
            True if the zone should be deactivated.
        """
        if df.empty or "atr_14" not in df.columns:
            return False

        atr = df["atr_14"].iloc[-1]
        if pd.isna(atr):
            return False

        price = zone["price"]
        last_close = df["close"].iloc[-1]
        distance = abs(last_close - price)

        if distance > self.atr_invalidation_multiple * atr:
            self.logger.log("DEBUG", "liquidity", "Zone invalidated: price moved beyond 2 ATR",
                            {"zone_type": zone["type"], "price": price, "distance": distance})
            return True

        return False
