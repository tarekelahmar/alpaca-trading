#!/usr/bin/env python3
"""Daily Trading Report Generator.

Generates a markdown report summarizing:
    - Portfolio status
    - Today's trades
    - Strategy performance
    - Regime classification
    - Risk metrics

Output is written to stdout as markdown, suitable for Claude to summarize.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))


def generate_report():
    """Generate daily trading report from database."""
    try:
        from data.store import DataStore
        store = DataStore()
    except Exception:
        # If database not available, generate from API
        store = None

    report = []
    report.append(f"# Daily Trading Report — {datetime.now().strftime('%Y-%m-%d')}\n")

    # Portfolio summary from Alpaca
    try:
        from alpaca.trading.client import TradingClient
        key = os.environ.get("ALPACA_API_KEY_ID")
        secret = os.environ.get("ALPACA_API_SECRET_KEY")
        paper = os.environ.get("PAPER_TRADING", "true").lower() == "true"

        if key and secret:
            client = TradingClient(key, secret, paper=paper)
            account = client.get_account()
            positions = client.get_all_positions()

            report.append("## Portfolio Summary\n")
            report.append(f"- **Equity:** ${float(account.equity):,.2f}")
            report.append(f"- **Cash:** ${float(account.cash):,.2f}")
            report.append(f"- **Buying Power:** ${float(account.buying_power):,.2f}")
            report.append(f"- **Portfolio Value:** ${float(account.portfolio_value):,.2f}")
            report.append(f"- **Open Positions:** {len(positions)}")
            report.append(f"- **Mode:** {'Paper' if paper else 'LIVE'}")
            report.append("")

            if positions:
                report.append("## Open Positions\n")
                report.append("| Symbol | Qty | Entry | Current | P&L | P&L % |")
                report.append("|--------|-----|-------|---------|-----|-------|")
                total_pnl = 0.0
                for pos in positions:
                    pnl = float(pos.unrealized_pl)
                    pnl_pct = float(pos.unrealized_plpc) * 100
                    total_pnl += pnl
                    report.append(
                        f"| {pos.symbol} | {pos.qty} | "
                        f"${float(pos.avg_entry_price):,.2f} | "
                        f"${float(pos.current_price):,.2f} | "
                        f"${pnl:,.2f} | {pnl_pct:.2f}% |"
                    )
                report.append(f"\n**Total Unrealized P&L:** ${total_pnl:,.2f}\n")
    except Exception as e:
        report.append(f"\n*Could not fetch portfolio data: {e}*\n")

    # Recent trades from database
    if store:
        try:
            trades = store.get_trades(limit=20)
            if trades:
                report.append("## Recent Trades\n")
                report.append("| Time | Symbol | Side | Qty | Price | Strategy |")
                report.append("|------|--------|------|-----|-------|----------|")
                for t in trades:
                    report.append(
                        f"| {t['submitted_at']} | {t['symbol']} | "
                        f"{t['side']} | {t['qty']} | "
                        f"${t.get('filled_price', t.get('price', 0)):,.2f} | "
                        f"{t.get('strategy_id', 'N/A')} |"
                    )
                report.append("")

            equity_data = store.get_recent_equity(limit=5)
            if equity_data:
                report.append("## Equity History (Last 5 Days)\n")
                report.append("| Date | Equity | Daily P&L | Drawdown |")
                report.append("|------|--------|-----------|----------|")
                for e in equity_data:
                    report.append(
                        f"| {e['timestamp']} | ${e['equity']:,.2f} | "
                        f"${e['daily_pnl']:,.2f} | {e['drawdown_pct']:.2f}% |"
                    )
                report.append("")

            store.close()
        except Exception as e:
            report.append(f"\n*Database error: {e}*\n")

    report.append("\n---\n*Generated automatically by the trading system.*\n")
    print("\n".join(report))


if __name__ == "__main__":
    generate_report()
