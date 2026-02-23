"""Strategies page — per-strategy performance breakdown."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "strategy-engine"))

st.header("Strategy Performance")

try:
    from data.store import DataStore
    import pandas as pd

    store = DataStore()

    # Get strategy list
    conn = store.conn
    cur = conn.cursor()
    cur.execute("SELECT id, name, type, active, params_json FROM strategies ORDER BY name")
    strategies = cur.fetchall()

    if strategies:
        for strat_id, name, stype, active, params in strategies:
            status = "🟢 Active" if active else "🔴 Inactive"
            with st.expander(f"{name} ({stype}) — {status}"):
                st.json(params)

                # Get trades for this strategy
                trades = store.get_trades(strategy_id=strat_id, limit=100)
                if trades:
                    df = pd.DataFrame(trades)
                    st.write(f"Total trades: {len(df)}")

                    if "filled_price" in df.columns and "price" in df.columns:
                        wins = df[df.get("filled_price", 0) > df.get("price", 0)]
                        st.write(f"Win rate: {len(wins)/len(df)*100:.1f}%")
                else:
                    st.info("No trades for this strategy yet.")
    else:
        st.info("No strategies registered yet.")

    store.close()
except Exception as e:
    st.error(f"Could not load strategy data: {e}")
