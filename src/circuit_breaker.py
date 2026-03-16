"""Circuit breaker module — pauses trading after consecutive losses."""

import json
import os
from datetime import datetime, timedelta, timezone

from src.logger import BotLogger


class CircuitBreaker:
    """Monitors trading performance and pauses/stops trading when limits are hit."""

    def __init__(self, config: dict, logger: BotLogger):
        """Initialize the CircuitBreaker.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance (can be None for testing).
        """
        self.config = config
        self.logger = logger

        cb = config.get("circuit_breaker", {})
        self.consecutive_losses_pause = cb.get("consecutive_losses_pause", 3)
        self.pause_hours = cb.get("consecutive_losses_pause_hours", 4)
        self.state_file = cb.get("state_file_path", "data/circuit_breaker_state.json")

        os.makedirs(os.path.dirname(self.state_file) if os.path.dirname(self.state_file) else ".", exist_ok=True)

        self.state = {
            "daily_loss_pct": 0.0,
            "weekly_loss_pct": 0.0,
            "monthly_loss_pct": 0.0,
            "consecutive_losses": 0,
            "paused_until": None,
            "status": "ACTIVE",
        }

        self._load_state()

    def _load_state(self):
        """Load state from JSON file if it exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    saved = json.load(f)
                self.state.update(saved)
            except Exception as e:
                if self.logger:
                    self.logger.log_error("circuit_breaker",
                                          f"Failed to load state: {e}")

    def persist_state(self):
        """Save current state to the JSON file. Called after every state change."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            if self.logger:
                self.logger.log_error("circuit_breaker",
                                      f"Failed to persist state: {e}")

    def check_all(self, daily_pnl: float, weekly_pnl: float,
                  monthly_pnl: float, account_balance: float) -> tuple:
        """Run all circuit breaker checks.

        Args:
            daily_pnl: Today's PnL.
            weekly_pnl: This week's PnL.
            monthly_pnl: This month's PnL.
            account_balance: Current account balance.

        Returns:
            Tuple of (allowed: bool, reason: str or None).
        """
        if self.is_paused():
            return (False, "PAUSED")

        if self.state["status"] == "STOPPED":
            return (False, "STOPPED")

        if self.state["status"] == "DAILY_PAUSED":
            return (False, "DAILY_PAUSED")

        if account_balance > 0:
            daily_loss_pct = abs(min(daily_pnl, 0)) / account_balance
            weekly_loss_pct = abs(min(weekly_pnl, 0)) / account_balance
            monthly_loss_pct = abs(min(monthly_pnl, 0)) / account_balance

            self.state["daily_loss_pct"] = daily_loss_pct
            self.state["weekly_loss_pct"] = weekly_loss_pct
            self.state["monthly_loss_pct"] = monthly_loss_pct

        return (True, None)

    def on_trade_closed(self, pnl_r: float):
        """Update state after a trade closes.

        Args:
            pnl_r: PnL in R multiples (negative = loss).
        """
        if pnl_r < 0:
            self.state["consecutive_losses"] += 1
        else:
            self.state["consecutive_losses"] = 0

        if self.state["consecutive_losses"] >= self.consecutive_losses_pause:
            paused_until = datetime.now(timezone.utc) + timedelta(hours=self.pause_hours)
            self.state["paused_until"] = paused_until.isoformat()
            self.state["status"] = "PAUSED"

            # Persist BEFORE alerting
            self.persist_state()

            if self.logger:
                self.logger.log_circuit_breaker("CONSECUTIVE_LOSSES_PAUSE", {
                    "consecutive_losses": self.state["consecutive_losses"],
                    "paused_until": self.state["paused_until"],
                })
            return

        self.persist_state()

    def is_paused(self) -> bool:
        """Check if the circuit breaker is currently paused.

        Auto-resumes if the pause period has elapsed.

        Returns:
            True if still paused.
        """
        paused_until = self.state.get("paused_until")
        if paused_until is None:
            return False

        try:
            pause_end = datetime.fromisoformat(paused_until)
            if pause_end.tzinfo is None:
                pause_end = pause_end.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return False

        now = datetime.now(timezone.utc)
        if now >= pause_end:
            # Auto-resume
            self.state["paused_until"] = None
            self.state["status"] = "ACTIVE"
            self.persist_state()
            if self.logger:
                self.logger.log("INFO", "circuit_breaker", "Auto-resumed after pause period")
            return False

        return True

    def trip_daily(self):
        """Trip the daily circuit breaker."""
        self.state["status"] = "DAILY_PAUSED"
        self.persist_state()
        if self.logger:
            self.logger.log_circuit_breaker("DAILY_LIMIT_TRIPPED", self.state)

    def trip_weekly(self):
        """Trip the weekly circuit breaker. Requires manual reset."""
        self.state["status"] = "STOPPED"
        self.persist_state()
        if self.logger:
            self.logger.log_circuit_breaker("WEEKLY_LIMIT_TRIPPED", self.state)

    def trip_monthly(self):
        """Trip the monthly circuit breaker. Requires manual reset."""
        self.state["status"] = "STOPPED"
        self.persist_state()
        if self.logger:
            self.logger.log_circuit_breaker("MONTHLY_LIMIT_TRIPPED", self.state)

    def manual_reset(self):
        """Manually reset all circuit breaker states to ACTIVE."""
        self.state = {
            "daily_loss_pct": 0.0,
            "weekly_loss_pct": 0.0,
            "monthly_loss_pct": 0.0,
            "consecutive_losses": 0,
            "paused_until": None,
            "status": "ACTIVE",
        }
        self.persist_state()
        if self.logger:
            self.logger.log("INFO", "circuit_breaker",
                            f"Manual reset at {datetime.now(timezone.utc).isoformat()}")
