"""Alert system module — sends notifications via Telegram and other channels."""

import os
import threading
import time
from datetime import datetime, timezone
from queue import Queue

import requests

from src.logger import BotLogger


class AlertSystem:
    """Non-blocking Telegram alert system with background message queue."""

    def __init__(self, config: dict, logger: BotLogger):
        """Initialize the AlertSystem.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance (can be None).
        """
        self.config = config
        self.logger = logger

        telegram = config.get("telegram", {})
        self.enabled = telegram.get("enabled", False)

        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

        self._queue = Queue()
        self._worker = threading.Thread(target=self._send_worker, daemon=True)
        self._worker.start()

    def _send_worker(self):
        """Background worker that sends queued messages."""
        while True:
            try:
                message = self._queue.get()
                if message is None:
                    break
                self._do_send(message)
                self._queue.task_done()
            except Exception:
                pass

    def _do_send(self, message: str):
        """Actually send a message to Telegram with retries.

        Args:
            message: Markdown-formatted message string.
        """
        if not self.enabled or not self.bot_token or not self.chat_id:
            self._log("DEBUG", f"Alert (no-op): {message[:80]}")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    return
                self._log("WARNING",
                          f"Telegram API returned {resp.status_code}: {resp.text}")
            except Exception as e:
                self._log("ERROR", f"Telegram send failed (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(5)

    def _log(self, level: str, message: str):
        """Log a message if logger is available."""
        if self.logger:
            self.logger.log(level, "alert_system", message)

    def send(self, message: str):
        """Queue a message for sending (non-blocking).

        Args:
            message: Markdown-formatted message string.
        """
        self._queue.put(message)

    def alert_trade_opened(self, trade: dict):
        """Send trade opened alert.

        Args:
            trade: Confirmed trade dict.
        """
        instrument = trade.get("instrument", "").replace("_", "/")
        direction = trade.get("direction", "")
        entry = trade.get("entry_price", 0)
        sl = trade.get("stop_loss", 0)
        tp2 = trade.get("take_profit_2", 0)
        size = trade.get("position_size", 0)
        rr = trade.get("signal_reference", {}).get("risk_reward_adjusted", 0)
        session = trade.get("signal_reference", {}).get("session", "")
        atr = trade.get("signal_reference", {}).get("atr_at_signal", 0)

        msg = (
            f"*TRADE OPENED*\n"
            f"Pair: {instrument} | {direction}\n"
            f"Entry: {entry:.5f} | SL: {sl:.5f} | TP: {tp2:.5f}\n"
            f"Size: {size:.2f} lots | RR: 1:{rr:.1f}\n"
            f"Session: {session} | ATR: {atr:.5f}"
        )
        self.send(msg)

    def alert_trade_closed(self, trade: dict, close_price: float,
                           reason: str, r_multiple: float):
        """Send trade closed alert.

        Args:
            trade: Trade dict.
            close_price: Price at which trade was closed.
            reason: Exit reason string.
            r_multiple: Final R multiple.
        """
        instrument = trade.get("instrument", "").replace("_", "/")
        direction = trade.get("direction", "")
        entry = trade.get("entry_price", 0)
        result = "WIN" if r_multiple > 0 else "LOSS"
        pnl = trade.get("pnl_pips", 0) * 10  # Approximate USD

        msg = (
            f"*TRADE CLOSED — {result}*\n"
            f"Pair: {instrument} | {direction}\n"
            f"Entry: {entry:.5f} | Exit: {close_price:.5f}\n"
            f"Result: {r_multiple:+.1f}R | ${pnl:+.2f}\n"
            f"Reason: {reason}"
        )
        self.send(msg)

    def alert_breakeven_moved(self, trade: dict):
        """Send breakeven moved alert."""
        instrument = trade.get("instrument", "").replace("_", "/")
        direction = trade.get("direction", "")
        self.send(f"*BE MOVED* — {instrument} {direction} — Now risk-free")

    def alert_partial_close(self, trade: dict, pnl: float):
        """Send partial close alert."""
        instrument = trade.get("instrument", "").replace("_", "/")
        self.send(f"*PARTIAL CLOSE* — {instrument} — +${pnl:.2f} locked in")

    def alert_daily_limit(self, current_loss_pct: float):
        """Send daily limit alert."""
        self.send(
            f"*DAILY LIMIT HIT* — Trading paused for today\n"
            f"Loss: -{current_loss_pct:.1f}% of account"
        )

    def alert_circuit_breaker(self, breaker_type: str, details: str):
        """Send circuit breaker alert."""
        self.send(
            f"*CIRCUIT BREAKER TRIGGERED:* {breaker_type}\n"
            f"{details}\n"
            f"Bot status: PAUSED/STOPPED"
        )

    def alert_spread_skip(self, instrument: str, spread: float,
                          threshold: float):
        """Send spread skip alert."""
        inst = instrument.replace("_", "/")
        self.send(f"SPREAD SKIP — {inst} — {spread:.1f} pips (max: {threshold:.1f})")

    def alert_news_block(self, instrument: str, event_name: str,
                         minutes_away: int):
        """Send news block alert."""
        inst = instrument.replace("_", "/")
        self.send(f"NEWS BLOCK — {inst} — {event_name} in {minutes_away} min")

    def alert_slippage_reject(self, instrument: str, slippage_pips: float):
        """Send slippage reject alert."""
        inst = instrument.replace("_", "/")
        self.send(
            f"*SLIPPAGE REJECT* — {inst} — {slippage_pips:.1f} pips slippage "
            f"— Trade closed"
        )

    def alert_bot_online(self, instruments: list):
        """Send bot online alert."""
        env = self.config.get("broker", {}).get("environment", "DEMO").upper()
        inst_str = ", ".join(i.replace("_", "/") for i in instruments)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.send(
            f"*BOT ONLINE*\n"
            f"Instruments: {inst_str}\n"
            f"Mode: {env}\n"
            f"Time: {now}"
        )

    def alert_bot_error(self, error: str, module: str):
        """Send bot error alert."""
        self.send(
            f"*BOT ERROR* in {module}\n"
            f"{error}\n"
            f"Please check logs."
        )

    def alert_daily_summary(self, summary: dict):
        """Send daily summary alert."""
        now = datetime.now(timezone.utc)
        day_str = now.strftime("%A %d %b")

        trades = summary.get("trades_today", 0)
        wins = summary.get("wins_today", 0)
        losses = summary.get("losses_today", 0)
        pnl = summary.get("pnl_usd", 0)
        pnl_pct = summary.get("pnl_pct", 0)
        dd = summary.get("drawdown_pct", 0)
        events = summary.get("upcoming_events", "None")

        self.send(
            f"*DAILY SUMMARY — {day_str}*\n"
            f"Trades: {trades} | Wins: {wins} | Losses: {losses}\n"
            f"PnL: ${pnl:+.2f} | {pnl_pct:+.2f}% of account\n"
            f"Drawdown today: {dd:.2f}%\n"
            f"Tomorrow's high-impact events: {events}"
        )
