"""Unit tests for the AlertSystem module."""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.alert_system import AlertSystem
from src.logger import BotLogger


def _make_config(enabled=False):
    return {
        "telegram": {"enabled": enabled},
        "broker": {"environment": "demo"},
        "logging": {"level": "DEBUG", "log_dir": "logs/",
                    "max_file_size_mb": 50, "backup_count": 10},
    }


class TestAlertSystem(unittest.TestCase):

    def setUp(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "test_token"
        os.environ["TELEGRAM_CHAT_ID"] = "12345"

    def test_send_does_not_raise_on_api_error(self):
        """send does not raise an exception when Telegram API returns an error."""
        config = _make_config(enabled=True)
        logger = BotLogger(config)
        alert = AlertSystem(config, logger)

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"

        with patch("src.alert_system.requests.post", return_value=mock_resp):
            # Should not raise
            alert._do_send("Test message")

    def test_alert_trade_opened_format(self):
        """alert_trade_opened formats a message containing instrument and direction."""
        config = _make_config(enabled=False)
        alert = AlertSystem(config, None)

        messages = []
        original_send = alert.send
        alert.send = lambda msg: messages.append(msg)

        trade = {
            "instrument": "EUR_USD",
            "direction": "SHORT",
            "entry_price": 1.0845,
            "stop_loss": 1.0865,
            "take_profit_2": 1.0785,
            "position_size": 0.10,
            "signal_reference": {
                "risk_reward_adjusted": 3.2,
                "session": "LONDON",
                "atr_at_signal": 0.00082,
            },
        }
        alert.alert_trade_opened(trade)

        self.assertEqual(len(messages), 1)
        msg = messages[0]
        self.assertIn("EUR/USD", msg)
        self.assertIn("SHORT", msg)
        self.assertIn("TRADE OPENED", msg)

    def test_all_alerts_noop_when_disabled(self):
        """All alert methods complete without errors when enabled is False."""
        config = _make_config(enabled=False)
        alert = AlertSystem(config, None)

        # Capture sends to verify they don't crash
        sent = []
        alert.send = lambda msg: sent.append(msg)

        trade = {
            "instrument": "EUR_USD", "direction": "LONG",
            "entry_price": 1.20, "stop_loss": 1.19,
            "take_profit_2": 1.23, "position_size": 0.1,
            "signal_reference": {"risk_reward_adjusted": 3.0,
                                 "session": "LONDON", "atr_at_signal": 0.005},
            "pnl_pips": 50,
        }

        # None of these should raise
        alert.alert_trade_opened(trade)
        alert.alert_trade_closed(trade, 1.23, "TP2_HIT", 3.0)
        alert.alert_breakeven_moved(trade)
        alert.alert_partial_close(trade, 42.0)
        alert.alert_daily_limit(2.8)
        alert.alert_circuit_breaker("CONSECUTIVE", "3 losses")
        alert.alert_spread_skip("EUR_USD", 3.2, 1.5)
        alert.alert_news_block("EUR_USD", "NFP", 25)
        alert.alert_slippage_reject("GBP_USD", 4.1)
        alert.alert_bot_online(["EUR_USD"])
        alert.alert_bot_error("Test error", "test")
        alert.alert_daily_summary({"trades_today": 2, "wins_today": 1,
                                   "losses_today": 1, "pnl_usd": 47,
                                   "pnl_pct": 0.47, "drawdown_pct": 0.21,
                                   "upcoming_events": "None"})

        self.assertEqual(len(sent), 12)

    def test_send_retries_on_exception(self):
        """send retries up to 3 times on network failure."""
        config = _make_config(enabled=True)
        alert = AlertSystem(config, None)

        with patch("src.alert_system.requests.post",
                   side_effect=Exception("network error")) as mock_post:
            with patch("src.alert_system.time.sleep"):
                alert._do_send("Test")
            self.assertEqual(mock_post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
