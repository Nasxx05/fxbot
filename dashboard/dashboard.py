"""Dashboard module — Streamlit-based monitoring and visualization dashboard."""

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st


def load_json(path: str) -> list | dict:
    """Load a JSON file, returning empty list/dict on failure."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return []


def load_spread_data() -> dict:
    """Load live spread data written by the bot's main loop."""
    path = "data/live_spreads.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_volatility_data() -> dict:
    """Load live volatility data written by the bot's main loop."""
    path = "data/live_volatility.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    """Main dashboard entry point."""
    st.set_page_config(page_title="FX Bot Dashboard", layout="wide")
    st.title("FX Trading Bot Dashboard")
    st.caption("Broker: MetaTrader 5")

    open_trades = load_json("data/open_trades.json")
    if not isinstance(open_trades, list):
        open_trades = list(open_trades.values()) if isinstance(open_trades, dict) else []

    trade_history = load_json("data/trade_history.json")
    if not isinstance(trade_history, list):
        trade_history = []

    cb_state = load_json("data/circuit_breaker_state.json")
    if not isinstance(cb_state, dict):
        cb_state = {}

    spread_data = load_spread_data()
    volatility_data = load_volatility_data()

    # -- Panel 1: Status Bar --
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    status = cb_state.get("status", "ACTIVE")
    status_color = {"ACTIVE": "🟢", "PAUSED": "🟡", "DAILY_PAUSED": "🟡",
                    "STOPPED": "🔴"}.get(status, "⚪")

    with col1:
        st.metric("Bot Status", f"{status_color} {status}")
    with col2:
        now = datetime.now(timezone.utc)
        st.metric("UTC Time", now.strftime("%H:%M:%S"))
    with col3:
        hour = now.hour
        if 7 <= hour < 12:
            session = "LONDON"
        elif 12 <= hour < 16:
            session = "OVERLAP"
        elif 12 <= hour < 20:
            session = "NEW YORK"
        elif 0 <= hour < 7:
            session = "ASIAN"
        else:
            session = "OFF HOURS"
        st.metric("Active Session", session)
    with col4:
        consec = cb_state.get("consecutive_losses", 0)
        st.metric("Consec. Losses", consec)

    # -- Panel 2: Open Trades --
    st.subheader("Open Trades")
    if open_trades:
        rows = []
        for t in open_trades:
            rows.append({
                "Instrument": t.get("instrument", ""),
                "Direction": t.get("direction", ""),
                "Entry": t.get("entry_price", 0),
                "SL": t.get("stop_loss", 0),
                "TP1": t.get("take_profit_1", 0),
                "TP2": t.get("take_profit_2", 0),
                "Size": t.get("position_size", 0),
                "R Multiple": t.get("r_multiple_current", 0),
                "BE Moved": "Yes" if t.get("breakeven_moved") else "No",
                "Entry Time": t.get("entry_time", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No open trades.")

    # -- Panel 3: Today's Performance --
    st.subheader("Today's Performance")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_trades = [t for t in trade_history
                    if str(t.get("close_time", "")).startswith(today)]

    col1, col2, col3 = st.columns(3)
    wins_today = sum(1 for t in today_trades if t.get("final_r_multiple", 0) > 0)
    losses_today = sum(1 for t in today_trades if t.get("final_r_multiple", 0) < 0)
    pnl_today = sum(t.get("pnl_pips", 0) for t in today_trades)

    with col1:
        st.metric("Trades Today", f"{len(today_trades)} / 3")
    with col2:
        st.metric("Win/Loss", f"{wins_today}/{losses_today}")
    with col3:
        st.metric("PnL (pips)", f"{pnl_today:+.1f}")

    # -- Panel 4: Equity Curve --
    st.subheader("Equity Curve")
    if trade_history:
        balance = 10000.0
        equity = [balance]
        dates = [None]
        for t in trade_history:
            pnl = t.get("pnl_pips", 0) * 10  # Approximate USD
            balance += pnl
            equity.append(balance)
            dates.append(t.get("close_time", ""))

        eq_df = pd.DataFrame({"Balance": equity[1:], "Time": dates[1:]})
        eq_df["Time"] = pd.to_datetime(eq_df["Time"], errors="coerce")
        eq_df = eq_df.dropna(subset=["Time"])
        if not eq_df.empty:
            st.line_chart(eq_df.set_index("Time")["Balance"])
        else:
            st.info("No equity data to display.")
    else:
        st.info("No trade history yet.")

    # -- Panel 5: Live Spread Monitor --
    st.subheader("Spread Monitor")
    instruments = ["EUR_USD", "GBP_USD", "GBP_JPY", "XAU_USD"]
    spread_cols = st.columns(len(instruments))
    for i, inst in enumerate(instruments):
        with spread_cols[i]:
            spread_val = spread_data.get(inst)
            if spread_val is not None:
                st.metric(inst.replace("_", "/"), f"{spread_val:.1f} pips")
            else:
                st.metric(inst.replace("_", "/"), "-- pips")

    # -- Panel 6: Volatility Status --
    st.subheader("Volatility Status")
    if volatility_data:
        vol_cols = st.columns(len(instruments))
        for i, inst in enumerate(instruments):
            with vol_cols[i]:
                vol_info = volatility_data.get(inst, {})
                regime = vol_info.get("regime", "N/A")
                atr = vol_info.get("atr", 0)
                color = {"TRENDING": "🟢", "RANGING": "🔴", "EXPANDING": "🟡"}.get(regime, "⚪")
                st.metric(inst.replace("_", "/"), f"{color} {regime}")
                if atr > 0:
                    st.caption(f"ATR: {atr:.5f}")
    else:
        st.info("Volatility data updates when the bot is running.")

    # -- Panel 7: Recent Signals --
    st.subheader("Recent Trade History")
    if trade_history:
        recent = trade_history[-20:]
        rows = []
        for t in recent:
            rows.append({
                "Time": t.get("close_time", ""),
                "Instrument": t.get("instrument", ""),
                "Direction": t.get("direction", ""),
                "Entry": t.get("entry_price", 0),
                "Exit": t.get("close_price", 0),
                "R Multiple": t.get("final_r_multiple", 0),
                "Reason": t.get("close_reason", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No recent signals.")

    # -- Panel 8: Circuit Breaker Status --
    st.subheader("Circuit Breaker Status")
    breaker_cols = st.columns(4)
    breakers = [
        ("Daily", cb_state.get("daily_loss_pct", 0) < 0.03),
        ("Consecutive", cb_state.get("consecutive_losses", 0) < 3),
        ("Weekly", cb_state.get("weekly_loss_pct", 0) < 0.08),
        ("Monthly", cb_state.get("monthly_loss_pct", 0) < 0.15),
    ]
    for i, (name, ok) in enumerate(breakers):
        with breaker_cols[i]:
            indicator = "🟢 Clear" if ok else "🔴 Tripped"
            st.metric(name, indicator)

    # Auto-refresh every 30 seconds
    time.sleep(30)
    st.rerun()


if __name__ == "__main__":
    main()
