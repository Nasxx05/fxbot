"""Structure engine module — analyzes market structure breaks and pullback zones."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.logger import BotLogger


class StructureEngine:
    """Detects Break of Structure (BOS) events and calculates pullback entry zones."""

    def __init__(self, config: dict, logger: BotLogger):
        """Initialize the StructureEngine with configuration and logger.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance for structured logging.
        """
        self.config = config
        self.logger = logger
        strategy = config.get("strategy", {})
        self.pullback_candle_limit = strategy.get("pullback_candle_limit", 5)

    def detect_break_of_structure(self, df: pd.DataFrame, sweep: dict) -> dict or None:
        """Detect a Break of Structure (BOS) after a sweep event.

        Bullish BOS (after bearish sweep of lows): price closes above the most
        recent swing high formed after the sweep candle.
        Bearish BOS (after bullish sweep of highs): price closes below the most
        recent swing low formed after the sweep candle.
        The BOS candle must CLOSE beyond the level — a wick does not count.

        Args:
            df: DataFrame with FeatureEngine columns computed.
            sweep: Sweep dict from LiquidityEngine.detect_sweep.

        Returns:
            BOS dict if confirmed, None otherwise.
        """
        if df.empty or sweep is None:
            self.logger.log("DEBUG", "structure", "No BOS: empty df or no sweep")
            return None

        sweep_index = sweep["sweep_candle_index"]
        direction = sweep["direction"]

        if direction == "BULLISH_SWEEP":
            bos_level = self.get_last_swing_high(df, after_index=sweep_index)
            if bos_level is None:
                self.logger.log("DEBUG", "structure",
                                "No BOS: no swing high found after sweep")
                return None

            after_sweep = df.loc[df.index > sweep_index]
            for idx in after_sweep.index:
                candle = df.loc[idx]
                if candle["close"] > bos_level:
                    result = {
                        "direction": "BULLISH",
                        "bos_level": bos_level,
                        "bos_candle_index": idx,
                        "bos_candle_close": float(candle["close"]),
                        "sweep_reference": sweep,
                        "timestamp": candle.get("timestamp", datetime.now(timezone.utc)),
                    }
                    self.logger.log("DEBUG", "structure", "Bullish BOS confirmed",
                                    {"bos_level": bos_level, "close": float(candle["close"])})
                    return result

        elif direction == "BEARISH_SWEEP":
            bos_level = self.get_last_swing_low(df, after_index=sweep_index)
            if bos_level is None:
                self.logger.log("DEBUG", "structure",
                                "No BOS: no swing low found after sweep")
                return None

            after_sweep = df.loc[df.index > sweep_index]
            for idx in after_sweep.index:
                candle = df.loc[idx]
                if candle["close"] < bos_level:
                    result = {
                        "direction": "BEARISH",
                        "bos_level": bos_level,
                        "bos_candle_index": idx,
                        "bos_candle_close": float(candle["close"]),
                        "sweep_reference": sweep,
                        "timestamp": candle.get("timestamp", datetime.now(timezone.utc)),
                    }
                    self.logger.log("DEBUG", "structure", "Bearish BOS confirmed",
                                    {"bos_level": bos_level, "close": float(candle["close"])})
                    return result

        self.logger.log("DEBUG", "structure", "No BOS detected",
                        {"sweep_direction": direction})
        return None

    def get_last_swing_high(self, df: pd.DataFrame, after_index: int) -> float or None:
        """Return the price of the most recent swing high near the sweep.

        First looks for swing highs after the sweep. If none found, uses the
        most recent swing high before the sweep as the BOS reference level.

        Args:
            df: DataFrame with is_swing_high and swing_high_price columns.
            after_index: The sweep candle index.

        Returns:
            Swing high price or None if not found.
        """
        if "is_swing_high" not in df.columns or "swing_high_price" not in df.columns:
            self.logger.log("DEBUG", "structure", "Missing swing point columns")
            return None

        # First try after sweep
        after = df.loc[df.index > after_index]
        swing_highs = after.loc[after["is_swing_high"] == True, "swing_high_price"]
        if not swing_highs.empty:
            return float(swing_highs.iloc[-1])

        # Fall back to most recent swing high within 20 candles before the sweep
        before = df.loc[(df.index <= after_index) & (df.index >= after_index - 20)]
        swing_highs = before.loc[before["is_swing_high"] == True, "swing_high_price"]
        if not swing_highs.empty:
            return float(swing_highs.iloc[-1])

        return None

    def get_last_swing_low(self, df: pd.DataFrame, after_index: int) -> float or None:
        """Return the price of the most recent swing low near the sweep.

        First looks for swing lows after the sweep. If none found, uses the
        most recent swing low within 20 candles before the sweep.

        Args:
            df: DataFrame with is_swing_low and swing_low_price columns.
            after_index: The sweep candle index.

        Returns:
            Swing low price or None if not found.
        """
        if "is_swing_low" not in df.columns or "swing_low_price" not in df.columns:
            self.logger.log("DEBUG", "structure", "Missing swing point columns")
            return None

        # First try after sweep
        after = df.loc[df.index > after_index]
        swing_lows = after.loc[after["is_swing_low"] == True, "swing_low_price"]
        if not swing_lows.empty:
            return float(swing_lows.iloc[-1])

        # Fall back to most recent swing low within 20 candles before the sweep
        before = df.loc[(df.index <= after_index) & (df.index >= after_index - 20)]
        swing_lows = before.loc[before["is_swing_low"] == True, "swing_low_price"]
        if not swing_lows.empty:
            return float(swing_lows.iloc[-1])

        return None

    def detect_pullback_zone(self, df: pd.DataFrame, sweep: dict, bos: dict) -> dict or None:
        """Calculate the pullback entry zone after a BOS is confirmed.

        Uses two methods:
        1. 50% retrace of the sweep candle range.
        2. Fair Value Gap (FVG) detection on the BOS candle.

        Args:
            df: DataFrame with OHLC data.
            sweep: Sweep dict from LiquidityEngine.
            bos: BOS dict from detect_break_of_structure.

        Returns:
            Pullback zone dict or None if no valid zone can be defined.
        """
        if sweep is None or bos is None:
            self.logger.log("DEBUG", "structure", "No pullback zone: missing sweep or BOS")
            return None

        bos_index = bos["bos_candle_index"]
        direction = bos["direction"]

        fvg = self._detect_fvg(df, bos_index, direction)
        if fvg is not None:
            current_index = df.index[-1] if len(df) > 0 else bos_index
            result = {
                "entry_price": (fvg["zone_high"] + fvg["zone_low"]) / 2,
                "zone_high": fvg["zone_high"],
                "zone_low": fvg["zone_low"],
                "method": "FVG",
                "expiry_candles": current_index + self.pullback_candle_limit,
            }
            self.logger.log("DEBUG", "structure", "Pullback zone via FVG",
                            {"entry": result["entry_price"], "method": "FVG"})
            return result

        sweep_high = sweep["sweep_high"]
        sweep_low = sweep["sweep_low"]
        sweep_range = sweep_high - sweep_low

        if sweep_range <= 0:
            self.logger.log("DEBUG", "structure", "No pullback zone: zero sweep range")
            return None

        if direction == "BEARISH":
            zone_high = sweep_low + (sweep_range * 0.5)
            zone_low = sweep_low + (sweep_range * 0.3)
        else:
            zone_low = sweep_high - (sweep_range * 0.5)
            zone_high = sweep_high - (sweep_range * 0.3)

        current_index = df.index[-1] if len(df) > 0 else bos_index
        result = {
            "entry_price": (zone_high + zone_low) / 2,
            "zone_high": zone_high,
            "zone_low": zone_low,
            "method": "RETRACE_50",
            "expiry_candles": current_index + self.pullback_candle_limit,
        }
        self.logger.log("DEBUG", "structure", "Pullback zone via 50% retrace",
                        {"entry": result["entry_price"], "method": "RETRACE_50"})
        return result

    def _detect_fvg(self, df: pd.DataFrame, bos_index: int, direction: str) -> dict or None:
        """Detect a Fair Value Gap (FVG) around the BOS candle.

        An FVG is a gap between candle N-1 and candle N+1 that candle N bridges.

        Args:
            df: DataFrame with OHLC data.
            bos_index: Index of the BOS candle.
            direction: BULLISH or BEARISH.

        Returns:
            Dict with zone_high and zone_low, or None.
        """
        idx_pos = df.index.get_loc(bos_index)
        if idx_pos < 1 or idx_pos >= len(df) - 1:
            return None

        prev_candle = df.iloc[idx_pos - 1]
        next_candle = df.iloc[idx_pos + 1]

        if direction == "BULLISH":
            if prev_candle["high"] < next_candle["low"]:
                return {
                    "zone_high": float(next_candle["low"]),
                    "zone_low": float(prev_candle["high"]),
                }
        elif direction == "BEARISH":
            if prev_candle["low"] > next_candle["high"]:
                return {
                    "zone_high": float(prev_candle["low"]),
                    "zone_low": float(next_candle["high"]),
                }

        return None

    def check_setup_expired(self, df: pd.DataFrame, setup: dict) -> bool:
        """Check if a pullback setup has expired.

        Expired if: current index past expiry, or price moved > 1 ATR away
        from the pullback zone without entering it.

        Args:
            df: DataFrame with atr_14 column.
            setup: Pullback zone dict from detect_pullback_zone.

        Returns:
            True if the setup has expired.
        """
        if df.empty or setup is None:
            return True

        current_index = df.index[-1]
        if current_index > setup["expiry_candles"]:
            self.logger.log("DEBUG", "structure", "Setup expired: candle limit exceeded",
                            {"current": current_index, "expiry": setup["expiry_candles"]})
            return True

        if "atr_14" in df.columns:
            atr = df["atr_14"].iloc[-1]
            if not pd.isna(atr):
                last_close = df["close"].iloc[-1]
                zone_mid = setup["entry_price"]
                distance = abs(last_close - zone_mid)
                if distance > atr:
                    self.logger.log("DEBUG", "structure",
                                    "Setup expired: price moved > 1 ATR from zone",
                                    {"distance": distance, "atr": atr})
                    return True

        return False
