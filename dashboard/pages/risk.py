"""Risk page — exposure, concentration, circuit breaker status."""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "strategy-engine"))

st.header("Risk Dashboard")

try:
    from alpaca.trading.client import TradingClient
    import pandas as pd

    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    paper = os.environ.get("PAPER_TRADING", "true").lower() == "true"

    if key and secret:
        client = TradingClient(key, secret, paper=paper)
        account = client.get_account()
        positions = client.get_all_positions()
        equity = float(account.equity)

        # Risk metrics
        st.subheader("Portfolio Exposure")

        if positions:
            total_long = sum(
                float(p.market_value) for p in positions
                if float(p.market_value) > 0
            )
            total_short = sum(
                abs(float(p.market_value)) for p in positions
                if float(p.market_value) < 0
            )
            total_exposure = total_long + total_short
            exposure_pct = (total_exposure / equity * 100) if equity > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Long Exposure", f"${total_long:,.2f}")
            col2.metric("Short Exposure", f"${total_short:,.2f}")
            col3.metric("Total Exposure %", f"{exposure_pct:.1f}%")

            # Position concentration
            st.subheader("Position Concentration")
            pos_data = []
            for p in positions:
                mv = abs(float(p.market_value))
                pct = (mv / equity * 100) if equity > 0 else 0
                pos_data.append({
                    "Symbol": p.symbol,
                    "Market Value": f"${mv:,.2f}",
                    "% of Portfolio": f"{pct:.1f}%",
                    "P&L": f"${float(p.unrealized_pl):,.2f}",
                })

            df = pd.DataFrame(pos_data)
            st.dataframe(df, use_container_width=True)

            # Concentration chart
            chart_data = pd.DataFrame({
                "Symbol": [p.symbol for p in positions],
                "Weight": [abs(float(p.market_value)) / equity * 100 for p in positions],
            })
            st.bar_chart(chart_data.set_index("Symbol"))
        else:
            st.info("No positions. Portfolio is 100% cash.")

        # Risk limits
        st.subheader("Risk Configuration")
        risk_params = {
            "MAX_POSITION_SIZE": os.environ.get("MAX_POSITION_SIZE", "10000"),
            "MAX_DAILY_LOSS": os.environ.get("MAX_DAILY_LOSS", "5000"),
            "POSITION_CONCENTRATION_LIMIT": os.environ.get("POSITION_CONCENTRATION_LIMIT", "0.25"),
            "DRAWDOWN_KILL_SWITCH": os.environ.get("DRAWDOWN_KILL_SWITCH", "0.10"),
            "MAX_OPEN_POSITIONS": os.environ.get("MAX_OPEN_POSITIONS", "20"),
        }
        for key_name, value in risk_params.items():
            st.write(f"**{key_name}:** {value}")
    else:
        st.warning("Set API keys to view risk dashboard.")
except Exception as e:
    st.error(f"Error: {e}")
