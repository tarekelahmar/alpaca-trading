"""Streamlit Performance Dashboard.

Launch with:
    streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Add strategy-engine to path
sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))

st.set_page_config(
    page_title="Alpaca Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Alpaca Systematic Trading Dashboard")

# Sidebar
st.sidebar.header("Configuration")
mode = "Paper" if os.environ.get("PAPER_TRADING", "true").lower() == "true" else "LIVE"
st.sidebar.markdown(f"**Mode:** {mode}")

# Try to connect to Alpaca for live data
try:
    from alpaca.trading.client import TradingClient

    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    paper = os.environ.get("PAPER_TRADING", "true").lower() == "true"

    if key and secret:
        client = TradingClient(key, secret, paper=paper)
        account = client.get_account()
        positions = client.get_all_positions()

        # Account metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Equity", f"${float(account.equity):,.2f}")
        col2.metric("Cash", f"${float(account.cash):,.2f}")
        col3.metric("Buying Power", f"${float(account.buying_power):,.2f}")
        col4.metric("Positions", len(positions))

        st.divider()

        # Positions table
        if positions:
            st.subheader("Open Positions")
            import pandas as pd

            pos_data = []
            for pos in positions:
                pos_data.append({
                    "Symbol": pos.symbol,
                    "Qty": float(pos.qty),
                    "Avg Entry": f"${float(pos.avg_entry_price):,.2f}",
                    "Current": f"${float(pos.current_price):,.2f}",
                    "P&L": f"${float(pos.unrealized_pl):,.2f}",
                    "P&L %": f"{float(pos.unrealized_plpc) * 100:.2f}%",
                    "Market Value": f"${float(pos.market_value):,.2f}",
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
        else:
            st.info("No open positions.")

        # Orders
        st.subheader("Recent Orders")
        orders = client.get_orders({"status": "all", "limit": 20})
        if orders:
            import pandas as pd

            order_data = []
            for o in orders:
                order_data.append({
                    "Symbol": o.symbol,
                    "Side": o.side,
                    "Qty": o.qty,
                    "Type": o.type,
                    "Status": o.status,
                    "Submitted": str(o.submitted_at)[:19],
                    "Filled Price": o.filled_avg_price or "—",
                })
            st.dataframe(pd.DataFrame(order_data), use_container_width=True)
        else:
            st.info("No recent orders.")
    else:
        st.warning(
            "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables "
            "to connect to Alpaca."
        )
except ImportError:
    st.error("Install alpaca-py: pip install alpaca-py")
except Exception as e:
    st.error(f"Error connecting to Alpaca: {e}")

# Database section
st.divider()
st.subheader("Database (Postgres)")

try:
    from data.store import DataStore

    store = DataStore()
    equity_data = store.get_recent_equity(limit=90)

    if equity_data:
        import pandas as pd

        df = pd.DataFrame(equity_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        st.line_chart(df.set_index("timestamp")["equity"])

        st.subheader("Drawdown")
        st.area_chart(df.set_index("timestamp")["drawdown_pct"])
    else:
        st.info("No equity curve data yet. Run the daily script to populate.")

    store.close()
except Exception as e:
    st.info(f"Database not connected: {e}. Start Postgres with docker-compose up.")
