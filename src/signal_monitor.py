"""Signal monitor — tracks the lifecycle of trade signals in signal_only mode.

In live mode, the ExecutionEngine places orders and the TradeManager watches
positions on MT5. In signal_only mode there are no orders — the user manages
trades manually — so this module stands in to provide the same situational
awareness via Telegram:

    NEW SIGNAL -> ENTRY HIT -> TP1 / TP2 / SL / INVALIDATED

State is persisted to a JSON file so a restart does not drop in-flight
signals. Price checks use the same MT5 bid/ask feed the rest of the bot uses.
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

from src.logger import BotLogger


class SignalMonitor:
    """Tracks pending and active signals; emits Telegram alerts on key events."""

    def __init__(self, config: dict, logger: BotLogger, data_engine,
                 alert_system, correlation_guard=None):
        """Initialize the SignalMonitor.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance.
            data_engine: DataEngine (used for live bid/ask prices).
            alert_system: AlertSystem used to push Telegram messages.
            correlation_guard: Optional CorrelationGuard. When provided,
                the monitor releases the instrument for new signals once
                the current signal closes (win, loss, or invalidation).
        """
        self.config = config
        self.logger = logger
        self.data_engine = data_engine
        self.alert_system = alert_system
        self.correlation_guard = correlation_guard

        sm = config.get("signal_monitor", {})
        self.poll_interval = int(sm.get("poll_interval_seconds", 15))
        self.alert_on_entry = bool(sm.get("alert_on_entry_zone", True))
        self.alert_on_tp_sl = bool(sm.get("alert_on_tp_sl_hit", True))
        self.alert_on_invalidation = bool(sm.get("alert_on_invalidation", True))
        self.pending_ttl = timedelta(
            hours=float(sm.get("pending_signal_ttl_hours", 6)))
        self.filled_ttl = timedelta(
            hours=float(sm.get("filled_signal_ttl_hours", 48)))
        self.state_path = sm.get("state_file_path",
                                 "data/signal_monitor_state.json")
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)

        # signals: list of dicts with keys:
        #   id, signal (original), status ("PENDING"|"FILLED"|"CLOSED"),
        #   tp1_hit, created_at (iso), updated_at (iso)
        self.signals = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = None

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_signal(self, signal: dict) -> str:
        """Register a new signal for monitoring and send the initial alert.

        Args:
            signal: Sized signal dict from StrategyEngine + RiskEngine.

        Returns:
            The signal's monitor id (string).
        """
        sig_id = (f"{signal.get('instrument', 'NA')}-"
                  f"{signal.get('direction', 'NA')}-"
                  f"{int(time.time() * 1000)}")

        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": sig_id,
            "signal": signal,
            "status": "PENDING",
            "tp1_hit": False,
            "entry_hit": False,
            "created_at": now,
            "updated_at": now,
        }

        with self._lock:
            self.signals.append(entry)
            self._save_state()

        self.logger.log("INFO", "signal_monitor",
                        f"Signal registered: {sig_id}",
                        {"instrument": signal.get("instrument"),
                         "direction": signal.get("direction")})

        # Push the NEW SIGNAL alert to Telegram.
        try:
            self.alert_system.alert_new_signal(signal)
        except Exception as e:
            self.logger.log_error("signal_monitor",
                                  f"Failed to send new_signal alert: {e}")

        return sig_id

    def start(self):
        """Start the background polling thread."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="SignalMonitor")
        self._worker.start()
        self.logger.log("INFO", "signal_monitor",
                        f"SignalMonitor started (poll {self.poll_interval}s)")

    def stop(self):
        """Stop the background polling thread."""
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)

    def active_signals(self) -> list:
        """Return a shallow copy of currently tracked signals."""
        with self._lock:
            return list(self.signals)

    def _release_correlation(self, signal: dict):
        """Release the instrument in the correlation guard, if wired."""
        if self.correlation_guard is None:
            return
        try:
            self.correlation_guard.register_closed_trade(
                signal.get("instrument", ""))
        except Exception as e:
            self.logger.log_error("signal_monitor",
                                  f"Correlation release failed: {e}")

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------
    def _run(self):
        """Poll prices and evaluate each tracked signal."""
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as e:
                self.logger.log_error("signal_monitor",
                                      f"Polling error: {e}")
            # Sleep in small increments so stop() is responsive.
            for _ in range(self.poll_interval):
                if self._stop.is_set():
                    return
                time.sleep(1)

    def _poll_once(self):
        """Evaluate every active signal against the latest bid/ask once."""
        with self._lock:
            snapshot = list(self.signals)

        now = datetime.now(timezone.utc)

        for entry in snapshot:
            status = entry.get("status")
            if status == "CLOSED":
                continue

            signal = entry["signal"]
            instrument = signal.get("instrument")
            direction = signal.get("direction")

            bid, ask = self.data_engine.get_bid_ask(instrument)
            if bid <= 0 or ask <= 0:
                continue

            # Use ask for LONG fills / SL / TP checks (what you'd buy at),
            # and bid for SHORT fills (what you'd sell at). This mirrors
            # how a market would actually trigger your orders.
            price_long = ask
            price_short = bid

            if status == "PENDING":
                self._check_pending(entry, direction, price_long, price_short, now)
            elif status == "FILLED":
                self._check_filled(entry, direction, price_long, price_short, now)

        with self._lock:
            self._save_state()

    def _check_pending(self, entry, direction, price_long, price_short, now):
        """Check whether a pending signal should transition to FILLED or expire."""
        signal = entry["signal"]
        entry_price = signal.get("entry_price", 0)

        # Has price reached the limit entry?
        hit = False
        touch_price = None
        if direction == "LONG":
            # A buy-limit fills when ask <= limit price.
            if price_long <= entry_price:
                hit = True
                touch_price = price_long
        else:
            # A sell-limit fills when bid >= limit price.
            if price_short >= entry_price:
                hit = True
                touch_price = price_short

        if hit:
            entry["status"] = "FILLED"
            entry["entry_hit"] = True
            entry["updated_at"] = now.isoformat()
            if self.alert_on_entry:
                try:
                    self.alert_system.alert_signal_entry_hit(signal, touch_price)
                except Exception as e:
                    self.logger.log_error("signal_monitor",
                                          f"entry_hit alert failed: {e}")
            self.logger.log("INFO", "signal_monitor",
                            f"Signal entry hit: {entry['id']}")
            return

        # Expire if older than TTL.
        created = datetime.fromisoformat(entry["created_at"])
        if now - created > self.pending_ttl:
            entry["status"] = "CLOSED"
            entry["updated_at"] = now.isoformat()
            if self.alert_on_invalidation:
                try:
                    self.alert_system.alert_signal_invalidated(
                        signal, "Entry zone not hit within TTL")
                except Exception as e:
                    self.logger.log_error("signal_monitor",
                                          f"invalidated alert failed: {e}")
            self._release_correlation(signal)
            self.logger.log("INFO", "signal_monitor",
                            f"Signal expired unfilled: {entry['id']}")

    def _check_filled(self, entry, direction, price_long, price_short, now):
        """Check TP1, TP2 and SL on a filled (entered) signal."""
        signal = entry["signal"]
        sl = signal.get("stop_loss", 0)
        tp1 = signal.get("take_profit_1", 0)
        tp2 = signal.get("take_profit_2", 0)

        # To close a LONG you sell at bid; to close a SHORT you buy at ask.
        if direction == "LONG":
            exit_price = price_short
            tp1_hit = exit_price >= tp1
            tp2_hit = exit_price >= tp2
            sl_hit = exit_price <= sl
        else:
            exit_price = price_long
            tp1_hit = exit_price <= tp1
            tp2_hit = exit_price <= tp2
            sl_hit = exit_price >= sl

        # TP1 fires once only.
        if tp1_hit and not entry.get("tp1_hit"):
            entry["tp1_hit"] = True
            entry["updated_at"] = now.isoformat()
            if self.alert_on_tp_sl:
                try:
                    self.alert_system.alert_signal_tp1_hit(signal, exit_price)
                except Exception as e:
                    self.logger.log_error("signal_monitor",
                                          f"tp1 alert failed: {e}")

        # TP2 = full win, close the signal.
        if tp2_hit:
            entry["status"] = "CLOSED"
            entry["updated_at"] = now.isoformat()
            if self.alert_on_tp_sl:
                try:
                    self.alert_system.alert_signal_tp2_hit(signal, exit_price)
                except Exception as e:
                    self.logger.log_error("signal_monitor",
                                          f"tp2 alert failed: {e}")
            self._release_correlation(signal)
            self.logger.log("INFO", "signal_monitor",
                            f"Signal reached TP2: {entry['id']}")
            return

        # SL = loss, close the signal.
        if sl_hit:
            entry["status"] = "CLOSED"
            entry["updated_at"] = now.isoformat()
            if self.alert_on_tp_sl:
                try:
                    self.alert_system.alert_signal_sl_hit(signal, exit_price)
                except Exception as e:
                    self.logger.log_error("signal_monitor",
                                          f"sl alert failed: {e}")
            self._release_correlation(signal)
            self.logger.log("INFO", "signal_monitor",
                            f"Signal stopped out: {entry['id']}")
            return

        # Drop from memory if too old even when still open, so we don't
        # track zombie signals forever if SL/TP somehow never triggers.
        created = datetime.fromisoformat(entry["created_at"])
        if now - created > self.filled_ttl:
            entry["status"] = "CLOSED"
            entry["updated_at"] = now.isoformat()
            self._release_correlation(signal)
            self.logger.log("INFO", "signal_monitor",
                            f"Signal aged out: {entry['id']}")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _save_state(self):
        """Persist active signals and drop CLOSED ones older than filled_ttl."""
        now = datetime.now(timezone.utc)
        kept = []
        for entry in self.signals:
            if entry.get("status") == "CLOSED":
                updated = datetime.fromisoformat(
                    entry.get("updated_at", entry["created_at"]))
                # Keep CLOSED signals briefly so the dashboard can show them,
                # then drop.
                if now - updated > timedelta(hours=2):
                    continue
            kept.append(entry)
        self.signals = kept

        try:
            with open(self.state_path, "w") as f:
                json.dump(self.signals, f, indent=2, default=str)
        except Exception as e:
            self.logger.log_error("signal_monitor",
                                  f"Failed to save state: {e}")

    def _load_state(self):
        """Restore signal state from disk on startup."""
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.signals = data
                self.logger.log(
                    "INFO", "signal_monitor",
                    f"Restored {len(self.signals)} signals from state file")
        except Exception as e:
            self.logger.log_error("signal_monitor",
                                  f"Failed to load state: {e}")
            self.signals = []
