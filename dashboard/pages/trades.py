"""Trades page — full trade log with signal details."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "strategy-engine"))

st.header("Trade Log")

try:
    from data.store import DataStore
    import pandas as pd

    store = DataStore()
    trades = store.get_trades(limit=200)

    if trades:
        df = pd.DataFrame(trades)
        df["submitted_at"] = pd.to_datetime(df["submitted_at"])

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            symbols = ["All"] + sorted(df["symbol"].unique().tolist())
            selected_symbol = st.selectbox("Symbol", symbols)
        with col2:
            sides = ["All", "buy", "sell"]
            selected_side = st.selectbox("Side", sides)

        filtered = df
        if selected_symbol != "All":
            filtered = filtered[filtered["symbol"] == selected_symbol]
        if selected_side != "All":
            filtered = filtered[filtered["side"] == selected_side]

        st.dataframe(
            filtered[[
                "submitted_at", "symbol", "side", "qty", "price",
                "filled_price", "status", "regime", "signal_strength",
            ]].sort_values("submitted_at", ascending=False),
            use_container_width=True,
        )

        # P&L summary
        if "filled_price" in filtered.columns and "price" in filtered.columns:
            st.subheader("Trade Statistics")
            st.write(f"Total trades: {len(filtered)}")
    else:
        st.info("No trades recorded yet.")

    store.close()
except Exception as e:
    st.error(f"Could not load trades: {e}")
