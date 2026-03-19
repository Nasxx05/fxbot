"""News filter module — blocks trading around high-impact news events."""

import json
import os
from datetime import datetime, timedelta, timezone

import requests

from src.logger import BotLogger


class NewsFilter:
    """Fetches economic calendar data and blocks trading around high-impact events."""

    def __init__(self, config: dict, logger: BotLogger):
        """Initialize the NewsFilter with configuration and logger.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance for structured logging.
        """
        self.config = config
        self.logger = logger
        news_config = config.get("news", {})
        self.block_minutes_before = news_config.get("block_minutes_before", 30)
        self.block_minutes_after = news_config.get("block_minutes_after", 30)
        self.cache_refresh_hours = news_config.get("cache_refresh_hours", 12)
        self.impact_levels = news_config.get("impact_levels_to_block", ["HIGH"])
        self.cache_file = os.path.join("data", "news_cache.json")
        self._events = []

        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self._load_or_refresh()

    def _load_or_refresh(self):
        """Load cached events or refresh the calendar if stale."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)
                cached_at = datetime.fromisoformat(cache.get("cached_at", ""))
                age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
                if age_hours < self.cache_refresh_hours:
                    self._events = cache.get("events", [])
                    return
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        self.refresh_calendar()

    def refresh_calendar(self):
        """Fetch the economic calendar and cache high-impact events.

        Fetches this week and next week calendars from Forex Factory.
        On failure, uses existing cache. If no cache exists, fails open
        with a critical warning.
        """
        urls = [
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
        ]

        all_events = []
        fetch_success = False

        for url in urls:
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                fetch_success = True

                for event in data:
                    impact = event.get("impact", "")
                    if impact.upper() not in [lvl.upper() for lvl in self.impact_levels]:
                        continue

                    date_str = event.get("date", "")
                    try:
                        event_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        if event_dt.tzinfo is None:
                            event_dt = event_dt.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        continue

                    all_events.append({
                        "title": event.get("title", "Unknown"),
                        "currency": event.get("country", ""),
                        "datetime_utc": event_dt.isoformat(),
                        "impact": impact,
                    })

            except Exception as e:
                if self.logger:
                    self.logger.log_error("news_filter", f"Calendar fetch failed: {e}",
                                          {"url": url})

        if fetch_success and all_events:
            self._events = all_events
            cache_data = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "events": all_events,
            }
            try:
                with open(self.cache_file, "w") as f:
                    json.dump(cache_data, f, indent=2)
            except OSError as e:
                if self.logger:
                    self.logger.log_error("news_filter", f"Cache write failed: {e}")
        elif not fetch_success:
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, "r") as f:
                        cache = json.load(f)
                    self._events = cache.get("events", [])
                    if self.logger:
                        self.logger.log("WARNING", "news_filter",
                                        "Using stale cache after fetch failure")
                except (json.JSONDecodeError, ValueError):
                    self._events = []
            else:
                self._events = []
                if self.logger:
                    self.logger.log("CRITICAL", "news_filter",
                                    "No calendar data available — failing open, all trades allowed")

    def get_upcoming_events(self, hours_ahead: int = 24) -> list:
        """Return high-impact events in the next N hours.

        Args:
            hours_ahead: How many hours ahead to look.

        Returns:
            List of event dicts with title, currency, datetime_utc, impact.
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        upcoming = []

        for event in self._events:
            try:
                event_dt = datetime.fromisoformat(event["datetime_utc"])
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
                if now <= event_dt <= cutoff:
                    upcoming.append(event)
            except (ValueError, KeyError):
                continue

        return upcoming

    def is_trading_blocked(self, instrument: str, check_time: datetime = None) -> tuple:
        """Check if trading is blocked for an instrument due to upcoming news.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
            check_time: Time to check against. Defaults to now UTC.

        Returns:
            Tuple of (blocked: bool, reason: str or None).
        """
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        if check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=timezone.utc)

        base_ccy, quote_ccy = self.get_currency_pair(instrument)

        for event in self._events:
            try:
                event_dt = datetime.fromisoformat(event["datetime_utc"])
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
            except (ValueError, KeyError):
                continue

            event_ccy = event.get("currency", "").upper()
            if event_ccy not in (base_ccy.upper(), quote_ccy.upper()):
                continue

            minutes_until = (event_dt - check_time).total_seconds() / 60
            minutes_since = (check_time - event_dt).total_seconds() / 60

            if 0 <= minutes_until <= self.block_minutes_before:
                reason = (f"News block: {event['title']} ({event_ccy}) "
                          f"in {minutes_until:.0f} minutes")
                if self.logger:
                    self.logger.log("INFO", "news_filter", reason,
                                    {"instrument": instrument, "event": event["title"]})
                return (True, reason)

            if 0 <= minutes_since <= self.block_minutes_after:
                reason = (f"News block: {event['title']} ({event_ccy}) "
                          f"occurred {minutes_since:.0f} minutes ago")
                if self.logger:
                    self.logger.log("INFO", "news_filter", reason,
                                    {"instrument": instrument, "event": event["title"]})
                return (True, reason)

        return (False, None)

    def get_next_event(self, instrument: str) -> dict or None:
        """Return the next upcoming high-impact event for this instrument.

        Args:
            instrument: Instrument name (e.g. EUR_USD).

        Returns:
            Event dict or None if no events in the next 24 hours.
        """
        now = datetime.now(timezone.utc)
        base_ccy, quote_ccy = self.get_currency_pair(instrument)
        best = None
        best_dt = None

        for event in self._events:
            try:
                event_dt = datetime.fromisoformat(event["datetime_utc"])
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
            except (ValueError, KeyError):
                continue

            event_ccy = event.get("currency", "").upper()
            if event_ccy not in (base_ccy.upper(), quote_ccy.upper()):
                continue

            if event_dt > now and (event_dt - now).total_seconds() <= 86400:
                if best_dt is None or event_dt < best_dt:
                    best = event
                    best_dt = event_dt

        return best

    def get_currency_pair(self, instrument: str) -> tuple:
        """Parse an instrument string into its two currencies.

        Args:
            instrument: Instrument name (e.g. EUR_USD, XAU_USD).

        Returns:
            Tuple of (base_currency, quote_currency).
        """
        parts = instrument.split("_")
        if len(parts) == 2:
            return (parts[0], parts[1])
        return (instrument, "")
