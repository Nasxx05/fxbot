"""Spread controller module — monitors spreads, slippage, and partial fills."""

from collections import deque

from src.logger import BotLogger


MAJOR_PAIRS = {"EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD"}


class SpreadController:
    """Validates spreads, detects spikes, checks slippage, and handles partial fills."""

    def __init__(self, config: dict, logger: BotLogger, data_engine):
        """Initialize the SpreadController.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance for structured logging.
            data_engine: DataEngine instance for fetching live spreads.
        """
        self.config = config
        self.logger = logger
        self.data_engine = data_engine

        spread_config = config.get("spread", {})
        self.max_pips = spread_config.get("max_pips", {})
        self.spike_multiplier = spread_config.get("spike_multiplier", 3.0)
        self.rolling_period = spread_config.get("rolling_average_period", 20)

        slippage_config = config.get("slippage", {})
        self.max_slippage = slippage_config.get("max_pips", {})
        self.partial_fill_min_pct = slippage_config.get("partial_fill_min_pct", 0.80)

        self._spread_history = {}

    def _get_history(self, instrument: str) -> deque:
        """Get or create the rolling spread history for an instrument.

        Args:
            instrument: Instrument name (e.g. EUR_USD).

        Returns:
            Deque of recent spread readings.
        """
        if instrument not in self._spread_history:
            self._spread_history[instrument] = deque(maxlen=self.rolling_period)
        return self._spread_history[instrument]

    def check_spread(self, instrument: str) -> tuple:
        """Check if the current spread is acceptable for trading.

        Fetches a fresh spread, adds to rolling history, checks against
        max_pips threshold and spike detection.

        Args:
            instrument: Instrument name (e.g. EUR_USD).

        Returns:
            Tuple of (acceptable: bool, current_spread: float, reason: str or None).
        """
        current_spread = self.data_engine.get_current_spread(instrument)
        history = self._get_history(instrument)
        history.append(current_spread)

        max_allowed = self.max_pips.get(instrument, 999.0)
        if current_spread > max_allowed:
            reason = "SPREAD_TOO_HIGH"
            self.logger.log_spread_check(instrument, current_spread, max_allowed, False)
            return (False, current_spread, reason)

        rolling_avg = self.get_rolling_average_spread(instrument)
        if rolling_avg is not None and current_spread > rolling_avg * self.spike_multiplier:
            reason = "SPREAD_SPIKE"
            self.logger.log_spread_check(instrument, current_spread, max_allowed, False)
            return (False, current_spread, reason)

        self.logger.log_spread_check(instrument, current_spread, max_allowed, True)
        return (True, current_spread, None)

    def get_rolling_average_spread(self, instrument: str) -> float:
        """Return the mean of the last N spread readings.

        Args:
            instrument: Instrument name (e.g. EUR_USD).

        Returns:
            Rolling average spread, or None if fewer than 5 readings.
        """
        history = self._get_history(instrument)
        if len(history) < 5:
            return None
        return sum(history) / len(history)

    def calculate_spread_adjusted_rr(self, entry: float, stop_loss: float,
                                     take_profit: float, instrument: str,
                                     direction: str) -> float:
        """Calculate the risk-reward ratio adjusted for spread.

        Args:
            entry: Intended entry price.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            instrument: Instrument name (e.g. EUR_USD).
            direction: LONG or SHORT.

        Returns:
            Spread-adjusted reward/risk ratio.
        """
        current_spread_pips = self.data_engine.get_current_spread(instrument)
        spread_price = self.pips_to_price(instrument, current_spread_pips)

        if direction.upper() == "LONG":
            effective_entry = entry + spread_price
        else:
            effective_entry = entry

        risk = abs(effective_entry - stop_loss)
        reward = abs(take_profit - effective_entry)

        rr = reward / risk if risk > 0 else 0.0

        original_risk = abs(entry - stop_loss)
        original_reward = abs(take_profit - entry)
        original_rr = original_reward / original_risk if original_risk > 0 else 0.0

        self.logger.log("INFO", "spread_controller",
                        f"RR adjusted for spread: {original_rr:.2f} -> {rr:.2f}",
                        {"instrument": instrument, "spread_pips": current_spread_pips,
                         "original_rr": original_rr, "adjusted_rr": rr})

        return rr

    def check_slippage(self, instrument: str, intended_price: float,
                       actual_fill_price: float, direction: str) -> tuple:
        """Check if slippage is within acceptable limits.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
            intended_price: The price the order was supposed to fill at.
            actual_fill_price: The price the order actually filled at.
            direction: LONG or SHORT.

        Returns:
            Tuple of (within_limit: bool, slippage_pips: float).
        """
        price_diff = abs(actual_fill_price - intended_price)
        slippage_pips = self.price_to_pips(instrument, price_diff)

        if instrument in MAJOR_PAIRS:
            max_slip = self.max_slippage.get("majors", 1.5)
        else:
            max_slip = self.max_slippage.get("exotics", 3.0)

        passed = slippage_pips <= max_slip
        self.logger.log_slippage_check(instrument, intended_price, actual_fill_price, passed)

        return (passed, slippage_pips)

    def handle_partial_fill(self, intended_size: float, actual_fill_size: float) -> tuple:
        """Evaluate a partial fill and decide to accept or reject.

        Args:
            intended_size: The requested order size.
            actual_fill_size: The actual filled size.

        Returns:
            Tuple of (decision: str, fill_pct: float).
            Decision is ACCEPT or REJECT.
        """
        fill_pct = actual_fill_size / intended_size if intended_size > 0 else 0.0

        if fill_pct >= self.partial_fill_min_pct:
            decision = "ACCEPT"
        else:
            decision = "REJECT"

        self.logger.log("INFO", "spread_controller",
                        f"Partial fill {decision}: {fill_pct:.1%}",
                        {"intended": intended_size, "actual": actual_fill_size,
                         "fill_pct": fill_pct, "threshold": self.partial_fill_min_pct})

        return (decision, fill_pct)

    def pips_to_price(self, instrument: str, pips: float) -> float:
        """Convert a pip value to a price difference.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
            pips: Number of pips.

        Returns:
            Price difference as a float.
        """
        if "JPY" in instrument:
            return pips / 100
        elif instrument == "XAU_USD":
            return pips / 10
        else:
            return pips / 10000

    def price_to_pips(self, instrument: str, price_diff: float) -> float:
        """Convert a price difference to pips.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
            price_diff: Absolute price difference.

        Returns:
            Number of pips as a float.
        """
        if "JPY" in instrument:
            return price_diff * 100
        elif instrument == "XAU_USD":
            return price_diff * 10
        else:
            return price_diff * 10000
