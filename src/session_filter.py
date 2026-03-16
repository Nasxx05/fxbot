"""Session filter module — enforces trading session restrictions."""

from datetime import datetime, timezone

from src.logger import BotLogger


class SessionFilter:
    """Determines the current trading session and enforces allowed session rules."""

    def __init__(self, config: dict, logger: BotLogger):
        """Initialize the SessionFilter with configuration and logger.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance for structured logging (can be None).
        """
        self.config = config
        self.logger = logger
        sessions = config.get("sessions", {})

        self.asian_start = self._parse_time(sessions.get("asian_start_utc", "00:00"))
        self.asian_end = self._parse_time(sessions.get("asian_end_utc", "07:00"))
        self.london_start = self._parse_time(sessions.get("london_start_utc", "07:00"))
        self.london_end = self._parse_time(sessions.get("london_end_utc", "12:00"))
        self.ny_start = self._parse_time(sessions.get("ny_start_utc", "12:00"))
        self.ny_end = self._parse_time(sessions.get("ny_end_utc", "20:00"))
        self.overlap_start = self._parse_time(sessions.get("overlap_start_utc", "12:00"))
        self.overlap_end = self._parse_time(sessions.get("overlap_end_utc", "16:00"))

        self.allowed_sessions = [s.upper() for s in sessions.get("allowed_sessions", [])]

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

    def get_current_session(self, check_time: datetime = None) -> str:
        """Determine which trading session is active at the given time.

        OVERLAP takes priority over LONDON and NEW_YORK when times overlap.

        Args:
            check_time: UTC datetime to check. Defaults to now.

        Returns:
            Session name: ASIAN, LONDON, NEW_YORK, OVERLAP, or OFF_HOURS.
        """
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        if check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=timezone.utc)

        minutes = check_time.hour * 60 + check_time.minute

        if self.overlap_start <= minutes < self.overlap_end:
            return "OVERLAP"
        if self.london_start <= minutes < self.london_end:
            return "LONDON"
        if self.ny_start <= minutes < self.ny_end:
            return "NEW_YORK"
        if self.asian_start <= minutes < self.asian_end:
            return "ASIAN"

        return "OFF_HOURS"

    def is_trading_allowed(self, check_time: datetime = None) -> tuple:
        """Check if trading is allowed during the current session.

        Args:
            check_time: UTC datetime to check. Defaults to now.

        Returns:
            Tuple of (allowed: bool, session_name: str).
        """
        session = self.get_current_session(check_time)
        allowed = session in self.allowed_sessions

        if not allowed and self.logger:
            self.logger.log("INFO", "session_filter",
                            f"Trading blocked: session {session} not in allowed list",
                            {"session": session, "allowed": self.allowed_sessions})

        return (allowed, session)

    def get_session_schedule(self) -> dict:
        """Return a dict of all session times and time until next allowed session.

        Returns:
            Dict with session times and minutes_to_next_allowed.
        """
        now = datetime.now(timezone.utc)
        current_minutes = now.hour * 60 + now.minute

        schedule = {
            "ASIAN": {"start": self._format_minutes(self.asian_start),
                      "end": self._format_minutes(self.asian_end)},
            "LONDON": {"start": self._format_minutes(self.london_start),
                       "end": self._format_minutes(self.london_end)},
            "NEW_YORK": {"start": self._format_minutes(self.ny_start),
                         "end": self._format_minutes(self.ny_end)},
            "OVERLAP": {"start": self._format_minutes(self.overlap_start),
                        "end": self._format_minutes(self.overlap_end)},
            "current_session": self.get_current_session(now),
            "minutes_to_next_allowed": self._minutes_to_next_allowed(current_minutes),
        }
        return schedule

    def _minutes_to_next_allowed(self, current_minutes: int) -> int:
        """Calculate minutes until the next allowed session starts.

        Args:
            current_minutes: Current time as minutes since midnight.

        Returns:
            Minutes until the next allowed session opens.
        """
        session_starts = {
            "ASIAN": self.asian_start,
            "LONDON": self.london_start,
            "NEW_YORK": self.ny_start,
            "OVERLAP": self.overlap_start,
        }

        min_wait = 1440  # 24 hours max
        for session_name, start in session_starts.items():
            if session_name not in self.allowed_sessions:
                continue
            wait = start - current_minutes
            if wait <= 0:
                wait += 1440
            if wait < min_wait:
                min_wait = wait

        return min_wait

    @staticmethod
    def _format_minutes(minutes: int) -> str:
        """Convert minutes since midnight to HH:MM string.

        Args:
            minutes: Minutes since midnight.

        Returns:
            Formatted time string.
        """
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def is_overlap_session(self, check_time: datetime = None) -> bool:
        """Check if the current time is in the London/NY overlap window.

        Args:
            check_time: UTC datetime to check. Defaults to now.

        Returns:
            True if in the overlap window.
        """
        return self.get_current_session(check_time) == "OVERLAP"
