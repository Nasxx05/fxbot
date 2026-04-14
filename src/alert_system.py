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

        # Env vars take precedence; config.yaml values are fallbacks.
        self.bot_token = (os.environ.get("TELEGRAM_BOT_TOKEN", "")
                          or telegram.get("bot_token", ""))
        self.chat_id = (os.environ.get("TELEGRAM_CHAT_ID", "")
                        or str(telegram.get("chat_id", "")))

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

    # ------------------------------------------------------------------
    # Signal-only mode alerts
    # ------------------------------------------------------------------
    def _fmt_price(self, value: float, instrument: str) -> str:
        """Format a price with sensible decimals for the instrument."""
        if value is None:
            return "-"
        if "JPY" in instrument:
            return f"{value:.3f}"
        if instrument == "XAU_USD":
            return f"{value:.2f}"
        return f"{value:.5f}"

    def _price_to_pips(self, instrument: str, diff: float) -> float:
        """Convert an absolute price difference into pips for display."""
        if "JPY" in instrument:
            return diff * 100
        if instrument == "XAU_USD":
            return diff * 10
        return diff * 10000

    def alert_new_signal(self, signal: dict):
        """Send a NEW SIGNAL message for signal-only mode.

        The user receives all parameters needed to enter the trade manually.

        Args:
            signal: Sized signal dict (must contain position_size, entry,
                    SL, TP1, TP2, RR, session, ATR, instrument, direction).
        """
        instrument = signal.get("instrument", "")
        pair = instrument.replace("_", "/")
        direction = signal.get("direction", "")
        arrow = "LONG" if direction == "LONG" else "SHORT"

        entry = signal.get("entry_price", 0)
        sl = signal.get("stop_loss", 0)
        tp1 = signal.get("take_profit_1", 0)
        tp2 = signal.get("take_profit_2", 0)
        size = signal.get("position_size", 0) or 0
        rr = signal.get("risk_reward_adjusted", 0)
        session = signal.get("session", "")
        atr = signal.get("atr_at_signal", 0) or 0
        spread = signal.get("current_spread_pips", 0)
        method = signal.get("entry_method", "")

        risk_pips = self._price_to_pips(instrument, abs(entry - sl))
        tp1_pips = self._price_to_pips(instrument, abs(tp1 - entry))
        tp2_pips = self._price_to_pips(instrument, abs(tp2 - entry))
        atr_pips = self._price_to_pips(instrument, atr)

        msg = (
            f"*NEW SIGNAL — {pair}*\n"
            f"Direction: *{arrow}*  (limit entry)\n"
            f"\n"
            f"Entry: `{self._fmt_price(entry, instrument)}`\n"
            f"Stop Loss: `{self._fmt_price(sl, instrument)}`  "
            f"({risk_pips:.1f} pips)\n"
            f"TP1 (2R): `{self._fmt_price(tp1, instrument)}`  "
            f"({tp1_pips:.1f} pips) — close 50%\n"
            f"TP2 (3R): `{self._fmt_price(tp2, instrument)}`  "
            f"({tp2_pips:.1f} pips) — close rest\n"
            f"\n"
            f"Suggested lot: *{size:.2f}*   (1% account risk)\n"
            f"R:R (spread-adj): *1:{rr:.2f}*\n"
            f"Session: {session}  |  Spread: {spread:.1f}p  |  "
            f"ATR: {atr_pips:.1f}p\n"
            f"Setup: {method}\n"
            f"\n"
            f"_Manage: move SL to BE at +1R, trail by 1×ATR after BE._"
        )
        self.send(msg)

    def alert_signal_entry_hit(self, signal: dict, price: float):
        """Alert that price has reached the entry zone."""
        pair = signal.get("instrument", "").replace("_", "/")
        direction = signal.get("direction", "")
        entry = signal.get("entry_price", 0)
        inst = signal.get("instrument", "")
        self.send(
            f"*ENTRY ZONE HIT — {pair} {direction}*\n"
            f"Price: `{self._fmt_price(price, inst)}`  "
            f"(limit: `{self._fmt_price(entry, inst)}`)\n"
            f"_If you have not placed the order yet, consider entering now._"
        )

    def alert_signal_tp1_hit(self, signal: dict, price: float):
        """Alert that TP1 (2R) has been reached."""
        pair = signal.get("instrument", "").replace("_", "/")
        direction = signal.get("direction", "")
        inst = signal.get("instrument", "")
        tp1 = signal.get("take_profit_1", 0)
        self.send(
            f"*TP1 HIT — {pair} {direction}*  +2R\n"
            f"Price: `{self._fmt_price(price, inst)}`  "
            f"(TP1: `{self._fmt_price(tp1, inst)}`)\n"
            f"_Suggest: close 50% and move SL to breakeven._"
        )

    def alert_signal_tp2_hit(self, signal: dict, price: float):
        """Alert that TP2 (3R) has been reached."""
        pair = signal.get("instrument", "").replace("_", "/")
        direction = signal.get("direction", "")
        inst = signal.get("instrument", "")
        tp2 = signal.get("take_profit_2", 0)
        self.send(
            f"*TP2 HIT — {pair} {direction}*  +3R  WIN\n"
            f"Price: `{self._fmt_price(price, inst)}`  "
            f"(TP2: `{self._fmt_price(tp2, inst)}`)\n"
            f"_Suggest: close remainder._"
        )

    def alert_signal_sl_hit(self, signal: dict, price: float):
        """Alert that the stop loss has been hit."""
        pair = signal.get("instrument", "").replace("_", "/")
        direction = signal.get("direction", "")
        inst = signal.get("instrument", "")
        sl = signal.get("stop_loss", 0)
        self.send(
            f"*SL HIT — {pair} {direction}*  -1R\n"
            f"Price: `{self._fmt_price(price, inst)}`  "
            f"(SL: `{self._fmt_price(sl, inst)}`)\n"
            f"_Trade closed. Next setup will be sent when conditions align._"
        )

    def alert_signal_invalidated(self, signal: dict, reason: str):
        """Alert that a pending signal has been invalidated before entry."""
        pair = signal.get("instrument", "").replace("_", "/")
        direction = signal.get("direction", "")
        self.send(
            f"*SIGNAL INVALIDATED — {pair} {direction}*\n"
            f"Reason: {reason}\n"
            f"_Do not take this entry._"
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
