"""Strategy engine module — core trading strategy logic and signal generation."""

from datetime import date, datetime, timezone

import pandas as pd

from src.logger import BotLogger


class StrategyEngine:
    """Central coordinator that combines all signal engines into trade decisions."""

    def __init__(self, config, logger, data_engine, feature_engine,
                 liquidity_engine, structure_engine, volatility_engine,
                 news_filter, session_filter, spread_controller,
                 correlation_guard):
        """Initialize the StrategyEngine with all dependencies.

        Args:
            config: Full bot configuration dictionary.
            logger: BotLogger instance.
            data_engine: DataEngine instance.
            feature_engine: FeatureEngine instance.
            liquidity_engine: LiquidityEngine instance.
            structure_engine: StructureEngine instance.
            volatility_engine: VolatilityEngine instance.
            news_filter: NewsFilter instance.
            session_filter: SessionFilter instance.
            spread_controller: SpreadController instance.
            correlation_guard: CorrelationGuard instance.
        """
        self.config = config
        self.logger = logger
        self.data_engine = data_engine
        self.feature_engine = feature_engine
        self.liquidity_engine = liquidity_engine
        self.structure_engine = structure_engine
        self.volatility_engine = volatility_engine
        self.news_filter = news_filter
        self.session_filter = session_filter
        self.spread_controller = spread_controller
        self.correlation_guard = correlation_guard

        risk = config.get("risk", {})
        self.max_trades_per_day = risk.get("max_trades_per_day", 3)
        self.min_rr_after_spread = risk.get("min_risk_reward_after_spread", 2.5)
        self.min_sl_atr_multiple = risk.get("min_sl_atr_multiple", 1.0)
        self.max_sl_atr_multiple = risk.get("max_sl_atr_multiple", 3.0)

        self.active_sweeps = {}
        self.active_setups = {}
        self.daily_trade_count = 0
        self.daily_trade_date = date.today()

    def on_new_candle(self, instrument: str, df: pd.DataFrame) -> dict or None:
        """Process a new completed candle through the full signal pipeline.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
            df: DataFrame with all FeatureEngine columns computed.

        Returns:
            Trade signal dict if valid entry found, None otherwise.
        """
        today = date.today()
        if today != self.daily_trade_date:
            self.reset_daily_count()

        return self.run_signal_pipeline(instrument, df)

    def run_signal_pipeline(self, instrument: str, df: pd.DataFrame) -> dict or None:
        """Run the full signal pipeline through all gates.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
            df: DataFrame with all FeatureEngine columns computed.

        Returns:
            Trade signal dict if all gates pass, None otherwise.
        """
        now = datetime.now(timezone.utc)

        # GATE 1 — Daily trade limit
        if self.daily_trade_count >= self.max_trades_per_day:
            self.logger.log_skipped_signal("SKIP_DAILY_LIMIT",
                                           {"instrument": instrument, "count": self.daily_trade_count,
                                            "timestamp": now.isoformat()})
            return None

        # GATE 2 — Session filter
        allowed, session_name = self.session_filter.is_trading_allowed()
        if not allowed:
            self.logger.log_skipped_signal("SKIP_SESSION",
                                           {"instrument": instrument, "session": session_name,
                                            "timestamp": now.isoformat()})
            return None

        # GATE 3 — News filter
        blocked, news_reason = self.news_filter.is_trading_blocked(instrument)
        if blocked:
            self.logger.log_skipped_signal("SKIP_NEWS",
                                           {"instrument": instrument, "reason": news_reason,
                                            "timestamp": now.isoformat()})
            return None

        # GATE 4 — Volatility regime
        vol_state = self.volatility_engine.get_volatility_state(df)
        if vol_state["regime"] == "RANGING":
            self.logger.log_skipped_signal("SKIP_RANGING",
                                           {"instrument": instrument, "regime": vol_state["regime"],
                                            "timestamp": now.isoformat()})
            return None

        # GATE 5 — Spread check
        spread_ok, current_spread, spread_reason = self.spread_controller.check_spread(instrument)
        if not spread_ok:
            self.logger.log_skipped_signal(f"SKIP_{spread_reason}",
                                           {"instrument": instrument, "spread": current_spread,
                                            "timestamp": now.isoformat()})
            return None

        # GATE 6 — Correlation guard
        corr_blocked, blocker = self.correlation_guard.is_correlated_trade_open(instrument)
        if corr_blocked:
            self.logger.log_skipped_signal("SKIP_CORRELATION",
                                           {"instrument": instrument, "blocker": blocker,
                                            "timestamp": now.isoformat()})
            return None

        # GATE 7 — Liquidity sweep detection
        sweep = self._detect_or_get_sweep(instrument, df)
        if sweep is None:
            self.logger.log_skipped_signal("NO_SWEEP",
                                           {"instrument": instrument, "timestamp": now.isoformat()})
            return None

        # GATE 8 — Market structure break
        bos = self._detect_or_get_bos(instrument, df, sweep)
        if bos is None:
            self.logger.log_skipped_signal("NO_BOS",
                                           {"instrument": instrument, "timestamp": now.isoformat()})
            return None

        # GATE 9 — Volatility confirmation
        if not self.volatility_engine.is_volatility_sufficient(df, sweep=sweep):
            self.logger.log_skipped_signal("SKIP_LOW_VOLATILITY",
                                           {"instrument": instrument, "timestamp": now.isoformat()})
            return None

        # GATE 10 — Pullback zone entry
        pullback = self._check_pullback_entry(instrument, df, sweep, bos)
        if pullback is None:
            return None

        # Determine direction from BOS
        direction = "LONG" if bos["direction"] == "BULLISH" else "SHORT"

        # Calculate SL
        stop_loss = self.calculate_stop_loss(sweep, direction, df)
        if stop_loss is None:
            self.logger.log_skipped_signal("SKIP_SL_TOO_WIDE",
                                           {"instrument": instrument, "timestamp": now.isoformat()})
            return None

        entry_price = pullback["entry_price"]
        tp1, tp2 = self.calculate_take_profits(entry_price, stop_loss, direction)

        # GATE 11 — Final spread-adjusted RR check
        rr_adjusted = self.spread_controller.calculate_spread_adjusted_rr(
            entry_price, stop_loss, tp2, instrument, direction
        )
        if rr_adjusted < self.min_rr_after_spread:
            self.logger.log_skipped_signal("SKIP_RR_TOO_LOW",
                                           {"instrument": instrument,
                                            "rr_adjusted": rr_adjusted,
                                            "min_required": self.min_rr_after_spread,
                                            "timestamp": now.isoformat()})
            return None

        # BUILD SIGNAL
        risk_raw = abs(entry_price - stop_loss)
        reward_raw = abs(tp2 - entry_price)
        rr_raw = reward_raw / risk_raw if risk_raw > 0 else 0.0

        atr_value = self.volatility_engine.get_atr_value(df)

        signal = {
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "position_size": None,
            "risk_reward_raw": rr_raw,
            "risk_reward_adjusted": rr_adjusted,
            "sweep_reference": sweep,
            "bos_reference": bos,
            "entry_method": pullback["method"],
            "signal_time": now.isoformat(),
            "current_spread_pips": current_spread,
            "session": session_name,
            "atr_at_signal": atr_value,
        }

        self.daily_trade_count += 1
        self.logger.log_trade_signal(signal)

        # Clear used setups for this instrument
        self.active_sweeps.pop(instrument, None)
        self.active_setups.pop(instrument, None)

        return signal

    def _detect_or_get_sweep(self, instrument: str, df: pd.DataFrame) -> dict or None:
        """Detect a new sweep or return an existing active sweep.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
            df: DataFrame with FeatureEngine columns.

        Returns:
            Sweep dict or None.
        """
        zones = self.liquidity_engine.get_active_zones(df)
        for zone in zones:
            sweep = self.liquidity_engine.detect_sweep(df, zone)
            if sweep is not None:
                if instrument not in self.active_sweeps:
                    self.active_sweeps[instrument] = []
                self.active_sweeps[instrument].append(sweep)
                self.logger.log("DEBUG", "strategy", f"SWEEP_DETECTED: {instrument}",
                                {"direction": sweep["direction"]})

        sweeps = self.active_sweeps.get(instrument, [])
        if sweeps:
            return sweeps[-1]
        return None

    def _detect_or_get_bos(self, instrument: str, df: pd.DataFrame, sweep: dict) -> dict or None:
        """Detect a BOS or return an existing active BOS for this sweep.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
            df: DataFrame with FeatureEngine columns.
            sweep: Active sweep dict.

        Returns:
            BOS dict or None.
        """
        bos = self.structure_engine.detect_break_of_structure(df, sweep)
        if bos is not None:
            if instrument not in self.active_setups:
                self.active_setups[instrument] = []
            self.active_setups[instrument].append({"sweep": sweep, "bos": bos})
            self.logger.log("DEBUG", "strategy", f"BOS_DETECTED: {instrument}",
                            {"direction": bos["direction"]})
            return bos

        # Check existing setups, prune expired
        setups = self.active_setups.get(instrument, [])
        valid_setups = []
        for setup in setups:
            expired = self.structure_engine.check_setup_expired(
                df, {"entry_price": setup["bos"]["bos_level"],
                     "expiry_candles": setup["bos"]["bos_candle_index"] +
                     self.config.get("strategy", {}).get("pullback_candle_limit", 5)})
            if not expired:
                valid_setups.append(setup)

        self.active_setups[instrument] = valid_setups
        if valid_setups:
            return valid_setups[-1]["bos"]

        return None

    def _check_pullback_entry(self, instrument: str, df: pd.DataFrame,
                              sweep: dict, bos: dict) -> dict or None:
        """Check if price has entered a pullback zone.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
            df: DataFrame with OHLC and feature data.
            sweep: Active sweep dict.
            bos: Active BOS dict.

        Returns:
            Pullback zone dict if price is in zone, None otherwise.
        """
        pullback = self.structure_engine.detect_pullback_zone(df, sweep, bos)
        if pullback is None:
            self.logger.log_skipped_signal("NO_PULLBACK_ZONE",
                                           {"instrument": instrument})
            return None

        if self.structure_engine.check_setup_expired(df, pullback):
            self.invalidate_setup(instrument, "SETUP_EXPIRED")
            return None

        if df.empty:
            return None

        last_close = df["close"].iloc[-1]
        last_low = df["low"].iloc[-1]
        last_high = df["high"].iloc[-1]

        in_zone = (last_low <= pullback["zone_high"] and last_high >= pullback["zone_low"])

        if not in_zone:
            self.logger.log_skipped_signal("PRICE_NOT_IN_PULLBACK",
                                           {"instrument": instrument,
                                            "close": last_close,
                                            "zone_high": pullback["zone_high"],
                                            "zone_low": pullback["zone_low"]})
            return None

        return pullback

    def calculate_stop_loss(self, sweep: dict, direction: str, df: pd.DataFrame) -> float:
        """Calculate the stop loss price for a trade.

        Args:
            sweep: Sweep dict with sweep_high and sweep_low.
            direction: LONG or SHORT.
            df: DataFrame with atr_14 column.

        Returns:
            Stop loss price, or None if SL would exceed max_sl_atr_multiple.
        """
        if df.empty or "atr_14" not in df.columns:
            return None

        atr = df["atr_14"].iloc[-1]
        if pd.isna(atr):
            return None

        instrument = sweep.get("zone", {}).get("type", "")
        spread_pips = 0.0
        try:
            spread_pips = self.spread_controller.data_engine.get_current_spread(
                sweep.get("zone", {}).get("type", "EUR_USD"))
        except Exception:
            pass

        spread_buffer = self.spread_controller.pips_to_price("EUR_USD", spread_pips)

        if direction == "LONG":
            sl = sweep["sweep_low"] - (0.5 * atr) - spread_buffer
            distance = abs(df["close"].iloc[-1] - sl)
        else:
            sl = sweep["sweep_high"] + (0.5 * atr) + spread_buffer
            distance = abs(sl - df["close"].iloc[-1])

        if distance < self.min_sl_atr_multiple * atr:
            if direction == "LONG":
                sl = df["close"].iloc[-1] - (self.min_sl_atr_multiple * atr)
            else:
                sl = df["close"].iloc[-1] + (self.min_sl_atr_multiple * atr)
            distance = self.min_sl_atr_multiple * atr

        if distance > self.max_sl_atr_multiple * atr:
            self.logger.log("DEBUG", "strategy", "SL exceeds max ATR multiple",
                            {"distance": distance, "max": self.max_sl_atr_multiple * atr})
            return None

        return sl

    def calculate_take_profits(self, entry: float, stop_loss: float,
                               direction: str) -> tuple:
        """Calculate 2R and 3R take profit levels.

        Args:
            entry: Entry price.
            stop_loss: Stop loss price.
            direction: LONG or SHORT.

        Returns:
            Tuple of (tp1_2R, tp2_3R).
        """
        risk = abs(entry - stop_loss)

        if direction == "LONG":
            tp1 = entry + (risk * 2)
            tp2 = entry + (risk * 3)
        else:
            tp1 = entry - (risk * 2)
            tp2 = entry - (risk * 3)

        return (tp1, tp2)

    def invalidate_setup(self, instrument: str, reason: str):
        """Remove all active setups for an instrument.

        Args:
            instrument: Instrument name (e.g. EUR_USD).
            reason: Reason for invalidation.
        """
        self.active_sweeps.pop(instrument, None)
        self.active_setups.pop(instrument, None)
        self.logger.log("INFO", "strategy", f"Setup invalidated: {instrument}",
                        {"reason": reason})

    def reset_daily_count(self):
        """Reset the daily trade counter."""
        self.daily_trade_count = 0
        self.daily_trade_date = date.today()
        self.logger.log("INFO", "strategy", "Daily trade count reset")
