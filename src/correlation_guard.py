"""Correlation guard module — prevents correlated pair overexposure."""

from src.logger import BotLogger


class CorrelationGuard:
    """Prevents simultaneous trades on correlated currency pairs."""

    def __init__(self, config: dict, logger: BotLogger):
        """Initialize the CorrelationGuard with configuration and logger.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance for structured logging.
        """
        self.config = config
        self.logger = logger
        self.correlation_groups = config.get("correlation_groups", [])
        self.open_trades_tracker = {}

    def register_open_trade(self, instrument: str):
        """Mark an instrument as having an open trade.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
        """
        self.open_trades_tracker[instrument] = True
        self.logger.log("DEBUG", "correlation_guard",
                        f"Trade registered: {instrument}")

    def register_closed_trade(self, instrument: str):
        """Mark an instrument as no longer having an open trade.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
        """
        self.open_trades_tracker[instrument] = False
        self.logger.log("DEBUG", "correlation_guard",
                        f"Trade closed: {instrument}")

    def is_correlated_trade_open(self, instrument: str) -> tuple:
        """Check if any instrument in the same correlation group has an open trade.

        Args:
            instrument: Instrument name to check (e.g. EUR_USD).

        Returns:
            Tuple of (blocked: bool, blocking_instrument: str or None).
        """
        for group in self.correlation_groups:
            if instrument not in group:
                continue

            for other in group:
                if other == instrument:
                    continue
                if self.open_trades_tracker.get(other, False):
                    self.logger.log("INFO", "correlation_guard",
                                    f"Correlated trade blocks {instrument}: "
                                    f"{other} is open",
                                    {"instrument": instrument, "blocker": other})
                    return (True, other)

        return (False, None)

    def get_best_setup(self, setups: list) -> dict or None:
        """Select the best setup from competing correlated setups.

        Selection criteria:
        1. Highest spread-adjusted risk-reward ratio.
        2. Tie-break: stronger sweep wick ratio.

        Args:
            setups: List of setup dicts with risk_reward_adjusted and
                    sweep_reference keys.

        Returns:
            The best setup dict, or None if list is empty.
        """
        if not setups:
            return None

        def sort_key(s):
            """Sort by RR adjusted descending, then sweep wick ratio descending."""
            rr = s.get("risk_reward_adjusted", 0)
            sweep = s.get("sweep_reference", {})
            sweep_high = sweep.get("sweep_high", 0)
            sweep_low = sweep.get("sweep_low", 0)
            sweep_range = abs(sweep_high - sweep_low)
            return (rr, sweep_range)

        best = max(setups, key=sort_key)

        for s in setups:
            if s is not best:
                self.logger.log("INFO", "correlation_guard",
                                f"Setup rejected in favor of better correlated setup",
                                {"rejected_instrument": s.get("instrument"),
                                 "rejected_rr": s.get("risk_reward_adjusted"),
                                 "selected_instrument": best.get("instrument"),
                                 "selected_rr": best.get("risk_reward_adjusted")})

        return best
