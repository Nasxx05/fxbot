"""Execution engine module — order execution and fill management via MetaTrader 5."""

import time
from datetime import datetime, timezone

import MetaTrader5 as mt5

from src.logger import BotLogger


class ExecutionError(Exception):
    """Raised when order execution fails after all retries."""


class ExecutionEngine:
    """Handles order submission, monitoring, and fill confirmation with MT5."""

    def __init__(self, config: dict, logger: BotLogger, data_engine,
                 risk_engine, spread_controller):
        """Initialize the ExecutionEngine.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance.
            data_engine: DataEngine instance for MT5 access.
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
        self.default_deviation = 20  # Max price deviation in points

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
            result = self.submit_limit_order(
                instrument, direction, entry_price, stop_loss, tp2, position_size
            )
        except ExecutionError as e:
            self.logger.log_error("execution", f"Order submission failed: {e}")
            return None

        fill_price = result["fill_price"]
        fill_size = result["fill_size"]
        trade_id = str(result["order_id"])

        # Step 4: Check slippage
        slip_ok, slip_pips = self.spread_controller.check_slippage(
            instrument, entry_price, fill_price, direction
        )
        if not slip_ok:
            self.close_trade_at_market(trade_id, instrument, direction, fill_size, "SLIPPAGE_REJECT")
            self.logger.log("WARNING", "execution", "SLIPPAGE_REJECT",
                            {"instrument": instrument, "slippage_pips": slip_pips,
                             "intended": entry_price, "actual": fill_price})
            return None

        # Step 5: Handle partial fill
        decision, fill_pct = self.spread_controller.handle_partial_fill(
            position_size, fill_size
        )
        if decision == "REJECT":
            self.close_trade_at_market(trade_id, instrument, direction, fill_size, "PARTIAL_FILL_REJECT")
            self.logger.log("WARNING", "execution", "PARTIAL_FILL_REJECT",
                            {"instrument": instrument, "fill_pct": fill_pct})
            return None

        # Step 6: Build confirmed trade
        confirmed = self.build_confirmed_trade(sized_signal, fill_price, fill_size, trade_id)
        self.logger.log_trade_open(confirmed)
        return confirmed

    def submit_limit_order(self, instrument: str, direction: str, price: float,
                           sl: float, tp: float, size: float) -> dict:
        """Submit a limit order to MT5.

        Args:
            instrument: Config instrument name.
            direction: LONG or SHORT.
            price: Limit order price.
            sl: Stop loss price.
            tp: Take profit price.
            size: Position size in lots.

        Returns:
            Dict with order_id, fill_price, fill_size.

        Raises:
            ExecutionError: If all retries fail.
        """
        mt5_symbol = self.data_engine.to_mt5_symbol(instrument)

        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "LONG" else mt5.ORDER_TYPE_SELL_LIMIT

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": mt5_symbol,
            "volume": round(size, 2),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.default_deviation,
            "magic": 123456,
            "comment": "fxbot_limit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        for attempt in range(3):
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.log("INFO", "execution", "Limit order submitted",
                                {"order_id": result.order, "instrument": instrument,
                                 "direction": direction, "price": price, "size": size})
                return {
                    "order_id": result.order,
                    "fill_price": result.price if result.price else price,
                    "fill_size": size,
                }

            error_msg = result.comment if result else mt5.last_error()
            self.logger.log_error("execution",
                                  f"Order submit attempt {attempt + 1}/3 failed: {error_msg}")
            if attempt < 2:
                time.sleep(2)

        raise ExecutionError(f"Failed to submit limit order after 3 attempts: {instrument}")

    def submit_market_order(self, instrument: str, direction: str,
                            size: float, sl: float = 0.0, tp: float = 0.0,
                            reason: str = "") -> dict:
        """Submit a market order to MT5.

        Args:
            instrument: Config instrument name.
            direction: LONG or SHORT.
            size: Position size in lots.
            sl: Stop loss price (0 for none).
            tp: Take profit price (0 for none).
            reason: Reason for the market order.

        Returns:
            Dict with order_id, fill_price, fill_size.

        Raises:
            ExecutionError: If all retries fail.
        """
        mt5_symbol = self.data_engine.to_mt5_symbol(instrument)
        order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL

        # Get current price for the order
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None:
            raise ExecutionError(f"Cannot get tick for {mt5_symbol}")

        price = tick.ask if direction == "LONG" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": round(size, 2),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.default_deviation,
            "magic": 123456,
            "comment": f"fxbot_{reason}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        for attempt in range(3):
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.log("INFO", "execution", "Market order submitted",
                                {"order_id": result.order, "instrument": instrument,
                                 "reason": reason})
                return {
                    "order_id": result.order,
                    "fill_price": result.price if result.price else price,
                    "fill_size": size,
                }

            error_msg = result.comment if result else mt5.last_error()
            self.logger.log_error("execution",
                                  f"Market order attempt {attempt + 1}/3 failed: {error_msg}")
            if attempt < 2:
                time.sleep(2)

        raise ExecutionError(f"Failed to submit market order after 3 attempts: {instrument}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ticket.

        Args:
            order_id: MT5 order ticket as string.

        Returns:
            True if cancelled, False if failed.
        """
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
        }

        try:
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.log("INFO", "execution", "Order cancelled",
                                {"order_id": order_id})
                return True
            error_msg = result.comment if result else mt5.last_error()
            self.logger.log_error("execution", f"Cancel order failed: {error_msg}",
                                  {"order_id": order_id})
            return False
        except Exception as e:
            self.logger.log_error("execution", f"Cancel order failed: {e}",
                                  {"order_id": order_id})
            return False

    def close_trade_at_market(self, trade_id: str, instrument: str,
                              direction: str, volume: float, reason: str) -> bool:
        """Close an open trade immediately at market by sending opposite deal.

        Args:
            trade_id: MT5 position ticket as string.
            instrument: Config instrument name.
            direction: Original trade direction (LONG or SHORT).
            volume: Volume to close.
            reason: Reason for closing.

        Returns:
            True if closed, False if failed.
        """
        mt5_symbol = self.data_engine.to_mt5_symbol(instrument)

        # Close by sending opposite direction
        close_type = mt5.ORDER_TYPE_SELL if direction == "LONG" else mt5.ORDER_TYPE_BUY

        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None:
            self.logger.log_error("execution", f"Cannot get tick to close {mt5_symbol}")
            return False

        price = tick.bid if direction == "LONG" else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": round(volume, 2),
            "type": close_type,
            "position": int(trade_id),
            "price": price,
            "deviation": self.default_deviation,
            "magic": 123456,
            "comment": f"fxbot_close_{reason}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        try:
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.log("INFO", "execution", f"Trade closed: {reason}",
                                {"trade_id": trade_id, "reason": reason})
                return True
            error_msg = result.comment if result else mt5.last_error()
            self.logger.log_error("execution", f"Close trade failed: {error_msg}",
                                  {"trade_id": trade_id, "reason": reason})
            return False
        except Exception as e:
            self.logger.log_error("execution", f"Close trade failed: {e}",
                                  {"trade_id": trade_id, "reason": reason})
            return False

    def modify_trade_sl(self, trade_id: str, instrument: str,
                        new_sl: float, current_tp: float) -> bool:
        """Modify the stop loss on an open position via MT5.

        Args:
            trade_id: MT5 position ticket as string.
            instrument: Config instrument name.
            new_sl: New stop loss price.
            current_tp: Current take profit (must be re-sent with SLTP action).

        Returns:
            True if modified, False if failed.
        """
        mt5_symbol = self.data_engine.to_mt5_symbol(instrument)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": mt5_symbol,
            "position": int(trade_id),
            "sl": new_sl,
            "tp": current_tp,
        }

        try:
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.log("INFO", "execution", "SL modified via MT5",
                                {"trade_id": trade_id, "new_sl": new_sl})
                return True
            error_msg = result.comment if result else mt5.last_error()
            self.logger.log_error("execution", f"SL modification failed: {error_msg}",
                                  {"trade_id": trade_id})
            return False
        except Exception as e:
            self.logger.log_error("execution", f"SL modification failed: {e}",
                                  {"trade_id": trade_id})
            return False

    def build_confirmed_trade(self, signal: dict, fill_price: float,
                              fill_size: float, mt5_trade_id: str) -> dict:
        """Build a confirmed trade dict from signal and fill data.

        Args:
            signal: Original trade signal dict.
            fill_price: Actual fill price.
            fill_size: Actual filled size.
            mt5_trade_id: MT5 position ticket as string.

        Returns:
            Immutable trade dict for the trade manager.
        """
        return {
            "trade_id": mt5_trade_id,
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
