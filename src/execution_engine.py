"""Execution engine module — order execution and fill management."""

import time
from datetime import datetime, timezone

import requests

from src.logger import BotLogger


class ExecutionError(Exception):
    """Raised when order execution fails after all retries."""


class ExecutionEngine:
    """Handles order submission, monitoring, and fill confirmation with OANDA."""

    def __init__(self, config: dict, logger: BotLogger, data_engine,
                 risk_engine, spread_controller):
        """Initialize the ExecutionEngine.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance.
            data_engine: DataEngine instance for API access.
            risk_engine: RiskEngine instance for position sizing.
            spread_controller: SpreadController instance for spread/slippage checks.
        """
        self.config = config
        self.logger = logger
        self.data_engine = data_engine
        self.risk_engine = risk_engine
        self.spread_controller = spread_controller

        slippage = config.get("slippage", {})
        self.fill_timeout = slippage.get("fill_confirmation_timeout_seconds", 30)

    def execute_signal(self, signal: dict) -> dict or None:
        """Execute a trade signal through the full order lifecycle.

        Steps: re-check spread, re-size position, submit order, confirm fill,
        check slippage, handle partial fill.

        Args:
            signal: Trade signal dict with all required fields.

        Returns:
            Confirmed trade dict, or None if any step fails.
        """
        instrument = signal["instrument"]

        # Step 1: Re-check spread
        spread_ok, current_spread, reason = self.spread_controller.check_spread(instrument)
        if not spread_ok:
            self.logger.log("WARNING", "execution",
                            "EXECUTION_BLOCKED_SPREAD",
                            {"instrument": instrument, "spread": current_spread,
                             "reason": reason})
            return None

        # Step 2: Re-fetch balance and recalculate position size
        sized_signal = self.risk_engine.attach_position_size_to_signal(signal)
        if sized_signal is None:
            self.logger.log("WARNING", "execution",
                            "EXECUTION_BLOCKED_SIZING",
                            {"instrument": instrument})
            return None

        direction = sized_signal["direction"]
        entry_price = sized_signal["entry_price"]
        stop_loss = sized_signal["stop_loss"]
        tp2 = sized_signal["take_profit_2"]
        position_size = sized_signal["position_size"]

        # Step 3: Submit limit order
        try:
            order_id = self.submit_limit_order(
                instrument, direction, entry_price, stop_loss, tp2, position_size
            )
        except ExecutionError as e:
            self.logger.log_error("execution", f"Order submission failed: {e}")
            return None

        # Step 4: Wait for fill confirmation
        fill_info = self._wait_for_fill(order_id)
        if fill_info is None:
            self.cancel_order(order_id)
            self.logger.log("WARNING", "execution", "ENTRY_TIMEOUT",
                            {"order_id": order_id, "instrument": instrument})
            return None

        fill_price = fill_info["fill_price"]
        fill_size = fill_info["fill_size"]
        trade_id = fill_info.get("trade_id", order_id)

        # Step 5: Check slippage
        slip_ok, slip_pips = self.spread_controller.check_slippage(
            instrument, entry_price, fill_price, direction
        )
        if not slip_ok:
            self.close_trade_at_market(trade_id, "SLIPPAGE_REJECT")
            self.logger.log("WARNING", "execution", "SLIPPAGE_REJECT",
                            {"instrument": instrument, "slippage_pips": slip_pips,
                             "intended": entry_price, "actual": fill_price})
            return None

        # Step 6: Handle partial fill
        decision, fill_pct = self.spread_controller.handle_partial_fill(
            position_size, fill_size
        )
        if decision == "REJECT":
            self.close_trade_at_market(trade_id, "PARTIAL_FILL_REJECT")
            self.logger.log("WARNING", "execution", "PARTIAL_FILL_REJECT",
                            {"instrument": instrument, "fill_pct": fill_pct})
            return None

        # Step 7: Build confirmed trade
        confirmed = self.build_confirmed_trade(sized_signal, fill_price, fill_size, trade_id)
        self.logger.log_trade_open(confirmed)
        return confirmed

    def submit_limit_order(self, instrument: str, direction: str, price: float,
                           sl: float, tp: float, size: float) -> str:
        """Submit a limit order to OANDA.

        Args:
            instrument: OANDA instrument name.
            direction: LONG or SHORT.
            price: Limit order price.
            sl: Stop loss price.
            tp: Take profit price.
            size: Position size in units.

        Returns:
            OANDA order ID as a string.

        Raises:
            ExecutionError: If all retries fail.
        """
        url = (f"{self.data_engine.base_url}/v3/accounts/"
               f"{self.data_engine.account_id}/orders")

        units = size if direction == "LONG" else -size

        body = {
            "order": {
                "type": "LIMIT",
                "instrument": instrument,
                "units": str(units),
                "price": str(price),
                "timeInForce": "GTC",
                "stopLossOnFill": {"price": str(sl)},
                "takeProfitOnFill": {"price": str(tp)},
            }
        }

        for attempt in range(3):
            try:
                response = requests.post(url, headers=self.data_engine.headers,
                                         json=body, timeout=15)
                response.raise_for_status()
                data = response.json()
                order_id = data.get("orderCreateTransaction", {}).get("id", "")
                self.logger.log("INFO", "execution", "Limit order submitted",
                                {"order_id": order_id, "instrument": instrument,
                                 "direction": direction, "price": price, "size": size})
                return order_id
            except Exception as e:
                self.logger.log_error("execution",
                                      f"Order submit attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2)

        raise ExecutionError(f"Failed to submit limit order after 3 attempts: {instrument}")

    def submit_market_order(self, instrument: str, direction: str,
                            size: float, reason: str) -> str:
        """Submit a market order to OANDA.

        Args:
            instrument: OANDA instrument name.
            direction: LONG or SHORT.
            size: Position size in units.
            reason: Reason for the market order.

        Returns:
            OANDA order ID as a string.

        Raises:
            ExecutionError: If all retries fail.
        """
        url = (f"{self.data_engine.base_url}/v3/accounts/"
               f"{self.data_engine.account_id}/orders")

        units = size if direction == "LONG" else -size

        body = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
            }
        }

        for attempt in range(3):
            try:
                response = requests.post(url, headers=self.data_engine.headers,
                                         json=body, timeout=15)
                response.raise_for_status()
                data = response.json()
                order_id = data.get("orderCreateTransaction", {}).get("id", "")
                self.logger.log("INFO", "execution", "Market order submitted",
                                {"order_id": order_id, "instrument": instrument,
                                 "reason": reason})
                return order_id
            except Exception as e:
                self.logger.log_error("execution",
                                      f"Market order attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2)

        raise ExecutionError(f"Failed to submit market order after 3 attempts: {instrument}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID.

        Args:
            order_id: OANDA order ID.

        Returns:
            True if cancelled, False if failed.
        """
        url = (f"{self.data_engine.base_url}/v3/accounts/"
               f"{self.data_engine.account_id}/orders/{order_id}/cancel")

        try:
            response = requests.put(url, headers=self.data_engine.headers, timeout=15)
            response.raise_for_status()
            self.logger.log("INFO", "execution", "Order cancelled",
                            {"order_id": order_id})
            return True
        except Exception as e:
            self.logger.log_error("execution", f"Cancel order failed: {e}",
                                  {"order_id": order_id})
            return False

    def close_trade_at_market(self, trade_id: str, reason: str) -> bool:
        """Close an open trade immediately at market.

        Args:
            trade_id: OANDA trade ID.
            reason: Reason for closing.

        Returns:
            True if closed, False if failed.
        """
        url = (f"{self.data_engine.base_url}/v3/accounts/"
               f"{self.data_engine.account_id}/trades/{trade_id}/close")

        try:
            response = requests.put(url, headers=self.data_engine.headers, timeout=15)
            response.raise_for_status()
            self.logger.log("INFO", "execution", f"Trade closed: {reason}",
                            {"trade_id": trade_id, "reason": reason})
            return True
        except Exception as e:
            self.logger.log_error("execution", f"Close trade failed: {e}",
                                  {"trade_id": trade_id, "reason": reason})
            return False

    def get_order_status(self, order_id: str) -> dict:
        """Fetch order details from OANDA.

        Args:
            order_id: OANDA order ID.

        Returns:
            Dict with state, fill_price, fill_size, fill_time.
        """
        url = (f"{self.data_engine.base_url}/v3/accounts/"
               f"{self.data_engine.account_id}/orders/{order_id}")

        try:
            response = requests.get(url, headers=self.data_engine.headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            order = data.get("order", {})

            state = order.get("state", "PENDING")
            fill_price = float(order.get("filledTime", 0) or 0)
            fill_size = float(order.get("fillingTransactionID", 0) or 0)

            # Check transactions for fill info
            if state == "FILLED":
                txn_id = order.get("fillingTransactionID", "")
                fill_price = float(order.get("price", 0))

            return {
                "state": state,
                "fill_price": fill_price,
                "fill_size": fill_size,
                "fill_time": order.get("filledTime"),
                "trade_id": order.get("tradeOpenedID", order_id),
            }
        except Exception as e:
            self.logger.log_error("execution", f"Get order status failed: {e}",
                                  {"order_id": order_id})
            return {"state": "UNKNOWN", "fill_price": 0, "fill_size": 0, "fill_time": None}

    def _wait_for_fill(self, order_id: str) -> dict or None:
        """Poll order status until filled or timeout.

        Args:
            order_id: OANDA order ID.

        Returns:
            Fill info dict or None if timeout.
        """
        elapsed = 0
        poll_interval = 2

        while elapsed < self.fill_timeout:
            status = self.get_order_status(order_id)

            if status["state"] == "FILLED":
                return {
                    "fill_price": status["fill_price"],
                    "fill_size": status["fill_size"],
                    "trade_id": status.get("trade_id", order_id),
                }
            elif status["state"] == "CANCELLED":
                return None

            time.sleep(poll_interval)
            elapsed += poll_interval

        return None

    def build_confirmed_trade(self, signal: dict, fill_price: float,
                              fill_size: float, oanda_trade_id: str) -> dict:
        """Build a confirmed trade dict from signal and fill data.

        Args:
            signal: Original trade signal dict.
            fill_price: Actual fill price.
            fill_size: Actual filled size.
            oanda_trade_id: OANDA trade ID.

        Returns:
            Immutable trade dict for the trade manager.
        """
        return {
            "trade_id": oanda_trade_id,
            "instrument": signal["instrument"],
            "direction": signal["direction"],
            "entry_price": fill_price,
            "intended_entry": signal["entry_price"],
            "stop_loss": signal["stop_loss"],
            "take_profit_1": signal["take_profit_1"],
            "take_profit_2": signal["take_profit_2"],
            "position_size": fill_size,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
            "breakeven_moved": False,
            "partial_closed": False,
            "r_multiple_current": 0.0,
            "signal_reference": signal,
        }
