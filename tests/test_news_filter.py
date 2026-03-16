"""Unit tests for the NewsFilter module."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.news_filter import NewsFilter
from src.logger import BotLogger


def _make_config():
    """Return a minimal config dict for testing."""
    return {
        "news": {
            "block_minutes_before": 30,
            "block_minutes_after": 30,
            "cache_refresh_hours": 12,
            "impact_levels_to_block": ["HIGH"],
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


def _make_events_around_now(minutes_offset):
    """Create a single HIGH-impact event at now + minutes_offset."""
    event_time = datetime.now(timezone.utc) + timedelta(minutes=minutes_offset)
    return [{
        "title": "NFP Release",
        "currency": "USD",
        "datetime_utc": event_time.isoformat(),
        "impact": "High",
    }]


class TestNewsFilter(unittest.TestCase):
    """Tests for NewsFilter class."""

    @patch("src.news_filter.requests.get")
    def setUp(self, mock_get):
        """Set up test fixtures, mocking the calendar fetch on init."""
        mock_get.side_effect = Exception("No network in tests")
        self.config = _make_config()
        self.logger = BotLogger(self.config)
        self.nf = NewsFilter(self.config, self.logger)

    def test_is_trading_blocked_true_event_within_block_before(self):
        """Test is_trading_blocked returns True when event is within block_minutes_before."""
        self.nf._events = _make_events_around_now(minutes_offset=15)
        blocked, reason = self.nf.is_trading_blocked("EUR_USD")
        self.assertTrue(blocked)
        self.assertIn("NFP Release", reason)

    def test_is_trading_blocked_false_event_far_away(self):
        """Test is_trading_blocked returns False when event is more than 30 minutes away."""
        self.nf._events = _make_events_around_now(minutes_offset=60)
        blocked, reason = self.nf.is_trading_blocked("EUR_USD")
        self.assertFalse(blocked)
        self.assertIsNone(reason)

    def test_is_trading_blocked_true_event_within_block_after(self):
        """Test is_trading_blocked returns True when event occurred within block_minutes_after."""
        self.nf._events = _make_events_around_now(minutes_offset=-15)
        blocked, reason = self.nf.is_trading_blocked("EUR_USD")
        self.assertTrue(blocked)
        self.assertIn("ago", reason)

    def test_get_currency_pair_eur_usd(self):
        """Test get_currency_pair correctly parses EUR_USD."""
        base, quote = self.nf.get_currency_pair("EUR_USD")
        self.assertEqual(base, "EUR")
        self.assertEqual(quote, "USD")

    def test_get_currency_pair_xau_usd(self):
        """Test get_currency_pair correctly parses XAU_USD."""
        base, quote = self.nf.get_currency_pair("XAU_USD")
        self.assertEqual(base, "XAU")
        self.assertEqual(quote, "USD")

    def test_get_currency_pair_gbp_jpy(self):
        """Test get_currency_pair correctly parses GBP_JPY."""
        base, quote = self.nf.get_currency_pair("GBP_JPY")
        self.assertEqual(base, "GBP")
        self.assertEqual(quote, "JPY")

    @patch("src.news_filter.requests.get")
    def test_refresh_calendar_handles_network_failure(self, mock_get):
        """Test refresh_calendar handles a network failure without crashing."""
        mock_get.side_effect = Exception("Connection timeout")
        # Should not raise
        self.nf.refresh_calendar()
        # Events should be empty (no cache, no network)
        self.assertIsInstance(self.nf._events, list)

    def test_is_trading_blocked_only_affected_currency(self):
        """Test that events for unrelated currencies do not block."""
        event_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        self.nf._events = [{
            "title": "JPY Rate Decision",
            "currency": "JPY",
            "datetime_utc": event_time.isoformat(),
            "impact": "High",
        }]
        # EUR_USD should not be blocked by JPY event
        blocked, reason = self.nf.is_trading_blocked("EUR_USD")
        self.assertFalse(blocked)

    def test_get_upcoming_events(self):
        """Test get_upcoming_events returns events within the window."""
        self.nf._events = _make_events_around_now(minutes_offset=60)
        events = self.nf.get_upcoming_events(hours_ahead=2)
        self.assertEqual(len(events), 1)

    def test_get_next_event_returns_none_when_empty(self):
        """Test get_next_event returns None when no events exist."""
        self.nf._events = []
        result = self.nf.get_next_event("EUR_USD")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
