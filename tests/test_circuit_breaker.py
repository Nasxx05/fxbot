"""Unit tests for the CircuitBreaker module."""

import json
import os
import unittest
from datetime import datetime, timedelta, timezone

from src.circuit_breaker import CircuitBreaker
from src.logger import BotLogger


def _make_config(state_file="data/test_cb_state.json"):
    return {
        "circuit_breaker": {
            "consecutive_losses_pause": 3,
            "consecutive_losses_pause_hours": 4,
            "state_file_path": state_file,
        },
        "logging": {"level": "DEBUG", "log_dir": "logs/", "max_file_size_mb": 50, "backup_count": 10},
    }


class TestCircuitBreaker(unittest.TestCase):

    def setUp(self):
        self.state_file = "data/test_cb_state.json"
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        self.config = _make_config(self.state_file)
        self.logger = BotLogger(self.config)

    def tearDown(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    def test_state_persists_to_json(self):
        """State persists correctly to and from JSON file."""
        cb = CircuitBreaker(self.config, self.logger)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-1.0)

        self.assertTrue(os.path.exists(self.state_file))

        # Reload from file
        cb2 = CircuitBreaker(self.config, self.logger)
        self.assertEqual(cb2.state["consecutive_losses"], 2)

    def test_is_paused_false_after_elapsed(self):
        """is_paused returns False after the pause period has elapsed."""
        cb = CircuitBreaker(self.config, self.logger)
        # Set paused_until in the past
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        cb.state["paused_until"] = past.isoformat()
        cb.state["status"] = "PAUSED"
        cb.persist_state()

        self.assertFalse(cb.is_paused())
        self.assertEqual(cb.state["status"], "ACTIVE")

    def test_pause_after_3_consecutive_losses(self):
        """on_trade_closed pauses after 3 consecutive losses."""
        cb = CircuitBreaker(self.config, self.logger)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-0.5)
        cb.on_trade_closed(-2.0)

        self.assertTrue(cb.is_paused())
        self.assertEqual(cb.state["status"], "PAUSED")
        self.assertEqual(cb.state["consecutive_losses"], 3)

    def test_consecutive_losses_reset_on_win(self):
        """on_trade_closed resets consecutive losses after a winning trade."""
        cb = CircuitBreaker(self.config, self.logger)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-1.0)
        self.assertEqual(cb.state["consecutive_losses"], 2)

        cb.on_trade_closed(1.5)
        self.assertEqual(cb.state["consecutive_losses"], 0)

    def test_manual_reset_sets_active(self):
        """manual_reset sets status back to ACTIVE."""
        cb = CircuitBreaker(self.config, self.logger)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-1.0)
        self.assertTrue(cb.is_paused())

        cb.manual_reset()
        self.assertEqual(cb.state["status"], "ACTIVE")
        self.assertEqual(cb.state["consecutive_losses"], 0)
        self.assertFalse(cb.is_paused())

    def test_trip_daily(self):
        """trip_daily sets status to DAILY_PAUSED."""
        cb = CircuitBreaker(self.config, self.logger)
        cb.trip_daily()
        self.assertEqual(cb.state["status"], "DAILY_PAUSED")

    def test_trip_weekly(self):
        """trip_weekly sets status to STOPPED."""
        cb = CircuitBreaker(self.config, self.logger)
        cb.trip_weekly()
        self.assertEqual(cb.state["status"], "STOPPED")

    def test_trip_monthly(self):
        """trip_monthly sets status to STOPPED."""
        cb = CircuitBreaker(self.config, self.logger)
        cb.trip_monthly()
        self.assertEqual(cb.state["status"], "STOPPED")

    def test_check_all_returns_false_when_paused(self):
        """check_all returns False when paused."""
        cb = CircuitBreaker(self.config, self.logger)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-1.0)

        allowed, reason = cb.check_all(0, 0, 0, 10000)
        self.assertFalse(allowed)
        self.assertEqual(reason, "PAUSED")

    def test_state_persisted_before_alert(self):
        """State is persisted BEFORE the alert log is written."""
        cb = CircuitBreaker(self.config, self.logger)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-1.0)

        # State file should exist and contain PAUSED
        with open(self.state_file) as f:
            state = json.load(f)
        self.assertEqual(state["status"], "PAUSED")

    def test_none_logger_does_not_crash(self):
        """CircuitBreaker works with None logger."""
        cb = CircuitBreaker(self.config, None)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-1.0)
        cb.on_trade_closed(-1.0)
        self.assertTrue(cb.is_paused())
        cb.manual_reset()
        self.assertFalse(cb.is_paused())


if __name__ == "__main__":
    unittest.main()
