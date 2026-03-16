"""Unit tests for the SessionFilter module."""

import unittest
from datetime import datetime, timezone

from src.session_filter import SessionFilter
from src.logger import BotLogger


def _make_config():
    """Return a minimal config dict for testing."""
    return {
        "sessions": {
            "asian_start_utc": "00:00",
            "asian_end_utc": "07:00",
            "london_start_utc": "07:00",
            "london_end_utc": "12:00",
            "ny_start_utc": "12:00",
            "ny_end_utc": "20:00",
            "overlap_start_utc": "12:00",
            "overlap_end_utc": "16:00",
            "allowed_sessions": ["LONDON", "NEW_YORK", "OVERLAP"],
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


class TestSessionFilter(unittest.TestCase):
    """Tests for SessionFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = _make_config()
        self.logger = BotLogger(self.config)
        self.sf = SessionFilter(self.config, self.logger)

    def test_get_current_session_london(self):
        """Test get_current_session returns LONDON for 08:30 UTC."""
        t = datetime(2024, 1, 15, 8, 30, tzinfo=timezone.utc)
        self.assertEqual(self.sf.get_current_session(t), "LONDON")

    def test_get_current_session_overlap(self):
        """Test get_current_session returns OVERLAP for 13:00 UTC."""
        t = datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc)
        self.assertEqual(self.sf.get_current_session(t), "OVERLAP")

    def test_get_current_session_asian(self):
        """Test get_current_session returns ASIAN for 04:00 UTC."""
        t = datetime(2024, 1, 15, 4, 0, tzinfo=timezone.utc)
        self.assertEqual(self.sf.get_current_session(t), "ASIAN")

    def test_get_current_session_off_hours(self):
        """Test get_current_session returns OFF_HOURS for 22:00 UTC."""
        t = datetime(2024, 1, 15, 22, 0, tzinfo=timezone.utc)
        self.assertEqual(self.sf.get_current_session(t), "OFF_HOURS")

    def test_is_trading_allowed_false_for_asian(self):
        """Test is_trading_allowed returns False for ASIAN session."""
        t = datetime(2024, 1, 15, 4, 0, tzinfo=timezone.utc)
        allowed, session = self.sf.is_trading_allowed(t)
        self.assertFalse(allowed)
        self.assertEqual(session, "ASIAN")

    def test_is_trading_allowed_true_for_london(self):
        """Test is_trading_allowed returns True for LONDON session."""
        t = datetime(2024, 1, 15, 8, 30, tzinfo=timezone.utc)
        allowed, session = self.sf.is_trading_allowed(t)
        self.assertTrue(allowed)
        self.assertEqual(session, "LONDON")

    def test_is_trading_allowed_true_for_overlap(self):
        """Test is_trading_allowed returns True for OVERLAP session."""
        t = datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc)
        allowed, session = self.sf.is_trading_allowed(t)
        self.assertTrue(allowed)
        self.assertEqual(session, "OVERLAP")

    def test_is_trading_allowed_true_for_new_york(self):
        """Test is_trading_allowed returns True for NEW_YORK session."""
        t = datetime(2024, 1, 15, 17, 0, tzinfo=timezone.utc)
        allowed, session = self.sf.is_trading_allowed(t)
        self.assertTrue(allowed)
        self.assertEqual(session, "NEW_YORK")

    def test_is_overlap_session_true(self):
        """Test is_overlap_session returns True during overlap."""
        t = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)
        self.assertTrue(self.sf.is_overlap_session(t))

    def test_is_overlap_session_false(self):
        """Test is_overlap_session returns False outside overlap."""
        t = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
        self.assertFalse(self.sf.is_overlap_session(t))

    def test_get_session_schedule_has_keys(self):
        """Test get_session_schedule returns dict with expected keys."""
        schedule = self.sf.get_session_schedule()
        self.assertIn("ASIAN", schedule)
        self.assertIn("LONDON", schedule)
        self.assertIn("NEW_YORK", schedule)
        self.assertIn("OVERLAP", schedule)
        self.assertIn("current_session", schedule)
        self.assertIn("minutes_to_next_allowed", schedule)

    def test_get_current_session_new_york_after_overlap(self):
        """Test get_current_session returns NEW_YORK for 17:00 UTC."""
        t = datetime(2024, 1, 15, 17, 0, tzinfo=timezone.utc)
        self.assertEqual(self.sf.get_current_session(t), "NEW_YORK")


if __name__ == "__main__":
    unittest.main()
