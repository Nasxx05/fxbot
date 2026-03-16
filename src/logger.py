"""Bot logger module — structured JSON logging with rotating file handler."""

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON strings."""

    def format(self, record):
        """Format a log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": getattr(record, "bot_module", record.module),
            "message": record.getMessage(),
            "data": getattr(record, "data", None),
        }
        return json.dumps(log_entry)


class BotLogger:
    """Structured JSON logger with rotating file handler for the trading bot."""

    def __init__(self, config: dict):
        """Initialize the logger with configuration.

        Args:
            config: Dictionary containing logging configuration with keys:
                level, log_dir, max_file_size_mb, backup_count.
        """
        log_config = config.get("logging", {})
        log_dir = log_config.get("log_dir", "logs/")
        max_size_mb = log_config.get("max_file_size_mb", 50)
        backup_count = log_config.get("backup_count", 10)
        level = log_config.get("level", "INFO")

        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "bot.log")

        self._logger = logging.getLogger("forex_bot")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers.clear()

        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
        )
        handler.setFormatter(JsonFormatter())
        self._logger.addHandler(handler)

    def log(self, level: str, module: str, message: str, data: dict = None):
        """Core logging method used by all other log methods.

        Args:
            level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            module: Name of the bot module producing the log.
            message: Human-readable log message.
            data: Optional dictionary of structured data.
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        extra = {"bot_module": module, "data": data}
        self._logger.log(log_level, message, extra=extra)

    def log_trade_signal(self, signal: dict):
        """Log a trade signal event.

        Args:
            signal: Dictionary containing trade signal details.
        """
        self.log("INFO", "strategy", "Trade signal generated", signal)

    def log_trade_open(self, trade: dict):
        """Log a trade open event.

        Args:
            trade: Dictionary containing trade open details.
        """
        self.log("INFO", "execution", "Trade opened", trade)

    def log_trade_close(self, trade: dict):
        """Log a trade close event.

        Args:
            trade: Dictionary containing trade close details.
        """
        self.log("INFO", "execution", "Trade closed", trade)

    def log_skipped_signal(self, reason: str, details: dict):
        """Log a skipped trade signal.

        Args:
            reason: Why the signal was skipped.
            details: Additional context about the skipped signal.
        """
        self.log("INFO", "strategy", f"Signal skipped: {reason}", details)

    def log_spread_check(self, instrument: str, spread: float, threshold: float, passed: bool):
        """Log a spread check result.

        Args:
            instrument: The trading instrument.
            spread: The current spread value.
            threshold: The maximum allowed spread.
            passed: Whether the spread check passed.
        """
        self.log(
            "INFO",
            "spread",
            f"Spread check {'passed' if passed else 'failed'}: {instrument}",
            {"instrument": instrument, "spread": spread, "threshold": threshold, "passed": passed},
        )

    def log_slippage_check(self, instrument: str, intended: float, actual: float, passed: bool):
        """Log a slippage check result.

        Args:
            instrument: The trading instrument.
            intended: The intended execution price.
            actual: The actual fill price.
            passed: Whether slippage was within tolerance.
        """
        self.log(
            "INFO",
            "slippage",
            f"Slippage check {'passed' if passed else 'failed'}: {instrument}",
            {"instrument": instrument, "intended": intended, "actual": actual, "passed": passed},
        )

    def log_error(self, module: str, error: str, details: dict = None):
        """Log an error event.

        Args:
            module: The bot module where the error occurred.
            error: Error message string.
            details: Optional additional error context.
        """
        self.log("ERROR", module, error, details)

    def log_circuit_breaker(self, event: str, details: dict):
        """Log a circuit breaker event.

        Args:
            event: The circuit breaker event type.
            details: Event details and context.
        """
        self.log("WARNING", "circuit_breaker", event, details)
