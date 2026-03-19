"""Risk engine module — position sizing and risk management."""

import math
import time

import MetaTrader5 as mt5

from src.logger import BotLogger


class RiskEngine:
    """Calculates position sizes and enforces risk limits."""

    def __init__(self, config: dict, logger: BotLogger, data_engine):
        """Initialize the RiskEngine.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance.
            data_engine: DataEngine instance for MT5 access.
        """
        self.config = config
        self.logger = logger
        self.data_engine = data_engine

        risk = config.get("risk", {})
        self.per_trade_pct = risk.get("per_trade_pct", 0.01)
        self.daily_max_pct = risk.get("daily_max_pct", 0.03)
        self.weekly_max_pct = risk.get("weekly_max_pct", 0.08)
        self.monthly_max_pct = risk.get("monthly_max_pct", 0.15)

        self._balance_cache = None
        self._balance_cache_time = 0
        self._balance_cache_ttl = 60

    def calculate_position_size(self, instrument: str, entry_price: float,
                                stop_loss: float, account_balance: float) -> float:
        """Calculate position size in lots based on risk parameters.

        Args:
            instrument: Config instrument name.
            entry_price: Intended entry price.
            stop_loss: Stop loss price.
            account_balance: Current account balance.

        Returns:
            Position size in lots, rounded down to 2 decimal places.
        """
        account_risk = account_balance * self.per_trade_pct
        stop_distance_price = abs(entry_price - stop_loss)

        if stop_distance_price == 0:
            self.logger.log("ERROR", "risk_engine",
                            "Stop distance is zero, cannot calculate position size")
            return 0.01

        stop_distance_pips = self._price_to_pips(instrument, stop_distance_price)
        pip_value = self._get_pip_value_per_lot(instrument, entry_price)

        if pip_value <= 0 or stop_distance_pips <= 0:
            self.logger.log("ERROR", "risk_engine",
                            "Invalid pip value or stop distance",
                            {"pip_value": pip_value, "stop_pips": stop_distance_pips})
            return 0.01

        position_size = account_risk / (stop_distance_pips * pip_value)

        # Round DOWN to 2 decimal places — never round up
        position_size = math.floor(position_size * 100) / 100

        # Enforce min and max
        position_size = max(0.01, position_size)
        position_size = min(10.0, position_size)

        self.logger.log("INFO", "risk_engine", "Position size calculated", {
            "instrument": instrument,
            "account_balance": account_balance,
            "account_risk": account_risk,
            "stop_distance_pips": stop_distance_pips,
            "pip_value_per_lot": pip_value,
            "position_size": position_size,
        })

        return position_size

    def _price_to_pips(self, instrument: str, price_diff: float) -> float:
        """Convert price difference to pips."""
        if "JPY" in instrument:
            return price_diff * 100
        elif instrument == "XAU_USD":
            return price_diff * 10
        else:
            return price_diff * 10000

    def _get_pip_value_per_lot(self, instrument: str, price: float) -> float:
        """Get pip value in USD per standard lot.

        Args:
            instrument: Config instrument name.
            price: Current price of the instrument.

        Returns:
            Pip value per lot in account currency (USD).
        """
        if instrument == "XAU_USD":
            return 1.0

        parts = instrument.split("_") if "_" in instrument else ["", ""]
        base, quote = parts[0], parts[1] if len(parts) > 1 else ""

        if quote == "USD":
            return 10.0
        elif base == "USD":
            if price > 0:
                return 10.0 / price
            return 10.0
        else:
            return 10.0

    def get_account_balance(self) -> float:
        """Fetch current account balance from MT5, cached for 60 seconds.

        Returns:
            Account balance as a float.
        """
        now = time.time()
        if (self._balance_cache is not None and
                now - self._balance_cache_time < self._balance_cache_ttl):
            return self._balance_cache

        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.log_error("risk_engine",
                                      "Failed to fetch MT5 account info")
                if self._balance_cache is not None:
                    return self._balance_cache
                return 0.0

            balance = account_info.balance
            self._balance_cache = balance
            self._balance_cache_time = now
            self.logger.log("DEBUG", "risk_engine",
                            f"Account balance fetched: {balance}")
            return balance
        except Exception as e:
            self.logger.log_error("risk_engine",
                                  f"Failed to fetch account balance: {e}")
            if self._balance_cache is not None:
                return self._balance_cache
            return 0.0

    def check_daily_limit(self, daily_pnl: float, account_balance: float) -> tuple:
        """Check if daily loss limit has been hit.

        Args:
            daily_pnl: Total PnL for today (negative = loss).
            account_balance: Current account balance.

        Returns:
            Tuple of (allowed: bool, reason: str or None).
        """
        daily_loss = min(daily_pnl, 0)
        if account_balance <= 0:
            return (False, "DAILY_LIMIT_HIT")
        daily_loss_pct = abs(daily_loss) / account_balance
        if daily_loss_pct >= self.daily_max_pct:
            self.logger.log("WARNING", "risk_engine", "Daily loss limit hit",
                            {"pnl": daily_pnl, "pct": daily_loss_pct})
            return (False, "DAILY_LIMIT_HIT")
        return (True, None)

    def check_weekly_limit(self, weekly_pnl: float, account_balance: float) -> tuple:
        """Check if weekly loss limit has been hit.

        Args:
            weekly_pnl: Total PnL for this week.
            account_balance: Current account balance.

        Returns:
            Tuple of (allowed: bool, reason: str or None).
        """
        weekly_loss = min(weekly_pnl, 0)
        if account_balance <= 0:
            return (False, "WEEKLY_LIMIT_HIT")
        weekly_loss_pct = abs(weekly_loss) / account_balance
        if weekly_loss_pct >= self.weekly_max_pct:
            self.logger.log("WARNING", "risk_engine", "Weekly loss limit hit",
                            {"pnl": weekly_pnl, "pct": weekly_loss_pct})
            return (False, "WEEKLY_LIMIT_HIT")
        return (True, None)

    def check_monthly_limit(self, monthly_pnl: float, account_balance: float) -> tuple:
        """Check if monthly loss limit has been hit.

        Args:
            monthly_pnl: Total PnL for this month.
            account_balance: Current account balance.

        Returns:
            Tuple of (allowed: bool, reason: str or None).
        """
        monthly_loss = min(monthly_pnl, 0)
        if account_balance <= 0:
            return (False, "MONTHLY_LIMIT_HIT")
        monthly_loss_pct = abs(monthly_loss) / account_balance
        if monthly_loss_pct >= self.monthly_max_pct:
            self.logger.log("WARNING", "risk_engine", "Monthly loss limit hit",
                            {"pnl": monthly_pnl, "pct": monthly_loss_pct})
            return (False, "MONTHLY_LIMIT_HIT")
        return (True, None)

    def attach_position_size_to_signal(self, signal: dict) -> dict:
        """Attach a fresh position size to a trade signal.

        Args:
            signal: Trade signal dict from StrategyEngine.

        Returns:
            New signal dict with position_size filled, or None on failure.
        """
        try:
            balance = self.get_account_balance()
            if balance <= 0:
                self.logger.log_error("risk_engine",
                                      "Cannot size position: zero balance")
                return None

            size = self.calculate_position_size(
                signal["instrument"],
                signal["entry_price"],
                signal["stop_loss"],
                balance,
            )

            new_signal = dict(signal)
            new_signal["position_size"] = size
            return new_signal
        except Exception as e:
            self.logger.log_error("risk_engine",
                                  f"Position sizing failed: {e}")
            return None
