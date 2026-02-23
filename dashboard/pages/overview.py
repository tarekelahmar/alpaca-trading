"""Overview page — equity curve, daily P&L, drawdown."""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "strategy-engine"))

st.header("Portfolio Overview")

try:
    from data.store import DataStore
    import pandas as pd

    store = DataStore()
    equity_data = store.get_recent_equity(limit=365)

    if equity_data:
        df = pd.DataFrame(equity_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Equity Curve")
            st.line_chart(df.set_index("timestamp")["equity"])

        with col2:
            st.subheader("Daily P&L")
            st.bar_chart(df.set_index("timestamp")["daily_pnl"])

        st.subheader("Drawdown")
        st.area_chart(df.set_index("timestamp")["drawdown_pct"])

        # Summary stats
        latest = df.iloc[-1]
        st.subheader("Current Status")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Equity", f"${latest['equity']:,.2f}")
        col2.metric("Daily P&L", f"${latest['daily_pnl']:,.2f}")
        col3.metric("Drawdown", f"{latest['drawdown_pct']:.2f}%")
        col4.metric("Positions", int(latest["num_positions"]))
    else:
        st.info("No data yet. Run daily execution to populate.")

    store.close()
except Exception as e:
    st.error(f"Could not load data: {e}")
