"""Volatility engine module — tracks volatility regimes and conditions."""

import pandas as pd

from src.logger import BotLogger


class VolatilityEngine:
    """Evaluates volatility conditions for trade entry and exit decisions."""

    def __init__(self, config: dict, logger: BotLogger):
        """Initialize the VolatilityEngine with configuration and logger.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance for structured logging.
        """
        self.config = config
        self.logger = logger
        strategy = config.get("strategy", {})
        trade_mgmt = config.get("trade_management", {})
        self.volatility_multiplier = strategy.get("volatility_multiplier", 1.3)
        self.volatility_collapse_atr_multiple = trade_mgmt.get(
            "volatility_collapse_atr_multiple", 0.7
        )

    def is_volatility_sufficient(self, df: pd.DataFrame, sweep: dict = None) -> bool:
        """Check if volatility conditions are sufficient for trade entry.

        All three conditions must be true:
        1. Current ATR > ATR average * volatility_multiplier.
        2. volatility_regime is EXPANDING or NORMAL (not RANGING).
        3. Sweep candle range >= 1.5 * ATR average.

        Args:
            df: DataFrame with atr_14, atr_average, and volatility_regime columns.
            sweep: Optional sweep dict with sweep_high and sweep_low.

        Returns:
            True if all conditions are met.
        """
        if df.empty:
            self.logger.log("DEBUG", "volatility", "Insufficient: empty DataFrame")
            return False

        current_atr = self.get_atr_value(df)
        atr_avg = self.get_atr_average(df)

        if pd.isna(current_atr) or pd.isna(atr_avg) or atr_avg == 0:
            self.logger.log("DEBUG", "volatility", "Insufficient: ATR values unavailable")
            return False

        if current_atr <= atr_avg * self.volatility_multiplier:
            self.logger.log("DEBUG", "volatility",
                            "Insufficient: ATR below threshold",
                            {"atr": current_atr, "avg": atr_avg,
                             "threshold": atr_avg * self.volatility_multiplier})
            return False

        regime = df["volatility_regime"].iloc[-1] if "volatility_regime" in df.columns else "NORMAL"
        if regime == "RANGING":
            self.logger.log("DEBUG", "volatility",
                            "Insufficient: volatility regime is RANGING")
            return False

        if sweep is not None:
            sweep_range = abs(sweep.get("sweep_high", 0) - sweep.get("sweep_low", 0))
            if sweep_range < 1.5 * atr_avg:
                self.logger.log("DEBUG", "volatility",
                                "Insufficient: sweep candle range too small",
                                {"sweep_range": sweep_range,
                                 "threshold": 1.5 * atr_avg})
                return False

        self.logger.log("DEBUG", "volatility", "Volatility sufficient",
                        {"atr": current_atr, "avg": atr_avg, "regime": regime})
        return True

    def get_volatility_state(self, df: pd.DataFrame) -> dict:
        """Return the current volatility state as a dictionary.

        Args:
            df: DataFrame with atr_14, atr_average, and volatility_regime columns.

        Returns:
            Dict with current_atr, atr_average, regime, is_sufficient, and ratio.
        """
        current_atr = self.get_atr_value(df)
        atr_avg = self.get_atr_average(df)

        regime = "NORMAL"
        if "volatility_regime" in df.columns and not df.empty:
            regime = df["volatility_regime"].iloc[-1]

        ratio = 0.0
        if atr_avg and not pd.isna(atr_avg) and atr_avg > 0:
            ratio = current_atr / atr_avg if not pd.isna(current_atr) else 0.0

        state = {
            "current_atr": current_atr if not pd.isna(current_atr) else 0.0,
            "atr_average": atr_avg if not pd.isna(atr_avg) else 0.0,
            "regime": regime,
            "is_sufficient": self.is_volatility_sufficient(df),
            "ratio": ratio,
        }
        self.logger.log("DEBUG", "volatility", "Volatility state", state)
        return state

    def is_volatility_collapsing(self, df: pd.DataFrame) -> bool:
        """Check if volatility is collapsing (for early trade exit).

        Returns True if ATR has dropped below atr_average * collapse multiple.

        Args:
            df: DataFrame with atr_14 and atr_average columns.

        Returns:
            True if volatility is collapsing.
        """
        current_atr = self.get_atr_value(df)
        atr_avg = self.get_atr_average(df)

        if pd.isna(current_atr) or pd.isna(atr_avg) or atr_avg == 0:
            return False

        collapsing = current_atr < atr_avg * self.volatility_collapse_atr_multiple
        if collapsing:
            self.logger.log("DEBUG", "volatility", "Volatility collapsing",
                            {"atr": current_atr, "avg": atr_avg,
                             "threshold": atr_avg * self.volatility_collapse_atr_multiple})
        return collapsing

    def get_atr_value(self, df: pd.DataFrame) -> float:
        """Return the most recent ATR value.

        Args:
            df: DataFrame with atr_14 column.

        Returns:
            Current ATR as a float.
        """
        if df.empty or "atr_14" not in df.columns:
            return float("nan")
        return float(df["atr_14"].iloc[-1])

    def get_atr_average(self, df: pd.DataFrame) -> float:
        """Return the most recent ATR average value.

        Args:
            df: DataFrame with atr_average column.

        Returns:
            ATR average as a float.
        """
        if df.empty or "atr_average" not in df.columns:
            return float("nan")
        return float(df["atr_average"].iloc[-1])
