#!/usr/bin/env python3
"""Performance Attribution Report.

Reads the trade log database and generates a comprehensive performance report
broken down by strategy, direction, conviction tier, regime, and exit reason.

Usage:
    python scripts/generate_performance_report.py [--days 30] [--json] [--csv]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))

from portfolio.trade_logger import TradeLogger


def format_pnl(val):
    """Format P&L value with color-like prefix."""
    if val is None:
        return "N/A"
    if val >= 0:
        return f"+${val:,.2f}"
    return f"-${abs(val):,.2f}"


def format_pct(val):
    if val is None:
        return "N/A"
    return f"{val:+.2f}%"


def print_table(headers: list[str], rows: list[list], title: str = ""):
    """Print a simple aligned table."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    if not rows:
        print("  (no data)")
        return

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_line}")
    print(f"  {'-+-'.join('-' * w for w in widths)}")

    # Print rows
    for row in rows:
        line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(f"  {line}")


def generate_report(trade_logger: TradeLogger, days: int = 30, output_json: bool = False):
    """Generate the full performance report."""
    min_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d") if days > 0 else None
    period_label = f"Last {days} days" if days > 0 else "All time"

    # ----- Overall Summary -----
    all_trades = trade_logger.get_closed_trades(min_date=min_date)
    open_positions = trade_logger.get_open_entries()

    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if (t.get("realized_pnl") or 0) > 0)
    losses = total_trades - wins
    total_pnl = sum(t.get("realized_pnl") or 0 for t in all_trades)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    avg_win = 0
    avg_loss = 0
    if wins > 0:
        avg_win = sum(t["realized_pnl"] for t in all_trades if (t.get("realized_pnl") or 0) > 0) / wins
    if losses > 0:
        avg_loss = sum(t["realized_pnl"] for t in all_trades if (t.get("realized_pnl") or 0) <= 0) / losses

    profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float("inf")
    avg_hold = sum(t.get("hold_days") or 0 for t in all_trades) / total_trades if total_trades > 0 else 0

    if output_json:
        report = {
            "period": period_label,
            "generated_at": datetime.now().isoformat(),
            "overall": {
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round(win_rate, 1),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(avg_pnl, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
                "avg_hold_days": round(avg_hold, 1),
                "open_positions": len(open_positions),
            },
            "by_strategy": trade_logger.get_strategy_summary(min_date),
            "by_regime": trade_logger.get_regime_summary(min_date),
            "by_exit_reason": trade_logger.get_exit_reason_summary(min_date),
            "daily_pnl": trade_logger.get_daily_pnl(days if days > 0 else 90),
        }
        print(json.dumps(report, indent=2, default=str))
        return

    # ----- Print Reports -----
    print(f"\n{'#'*60}")
    print(f"  PERFORMANCE REPORT — {period_label}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#'*60}")

    print_table(
        ["Metric", "Value"],
        [
            ["Total Closed Trades", str(total_trades)],
            ["Open Positions", str(len(open_positions))],
            ["Wins / Losses", f"{wins} / {losses}"],
            ["Win Rate", f"{win_rate:.1f}%"],
            ["Total P&L", format_pnl(total_pnl)],
            ["Avg P&L/Trade", format_pnl(avg_pnl)],
            ["Avg Win", format_pnl(avg_win)],
            ["Avg Loss", format_pnl(avg_loss)],
            ["Profit Factor", f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"],
            ["Avg Hold (days)", f"{avg_hold:.1f}"],
        ],
        title="OVERALL SUMMARY",
    )

    # ----- By Strategy -----
    strategy_data = trade_logger.get_strategy_summary(min_date)
    if strategy_data:
        rows = []
        for s in strategy_data:
            w = s["wins"] or 0
            t = s["total_trades"] or 1
            wr = w / t * 100
            rows.append([
                s["strategy"],
                s["direction"],
                f"T{s['conviction_tier']}",
                str(s["total_trades"]),
                f"{wr:.0f}%",
                format_pnl(s["total_pnl"]),
                format_pct(s["avg_pnl_pct"]),
                format_pct(s["avg_win_pct"]),
                format_pct(s["avg_loss_pct"]),
                str(s["avg_hold_days"]),
            ])
        print_table(
            ["Strategy", "Dir", "Tier", "Trades", "Win%", "Total P&L", "Avg%", "AvgWin%", "AvgLoss%", "Hold"],
            rows,
            title="BY STRATEGY / DIRECTION / TIER",
        )

    # ----- By Regime -----
    regime_data = trade_logger.get_regime_summary(min_date)
    if regime_data:
        rows = []
        for r in regime_data:
            w = r["wins"] or 0
            t = r["total_trades"] or 1
            wr = w / t * 100
            rows.append([
                r["entry_regime"],
                str(r["total_trades"]),
                f"{wr:.0f}%",
                format_pnl(r["total_pnl"]),
                format_pct(r["avg_pnl_pct"]),
            ])
        print_table(
            ["Regime", "Trades", "Win%", "Total P&L", "Avg%"],
            rows,
            title="BY ENTRY REGIME",
        )

    # ----- By Exit Reason -----
    exit_data = trade_logger.get_exit_reason_summary(min_date)
    if exit_data:
        rows = []
        for e in exit_data:
            w = e["wins"] or 0
            t = e["total_trades"] or 1
            wr = w / t * 100
            rows.append([
                e["exit_type"],
                str(e["total_trades"]),
                f"{wr:.0f}%",
                format_pnl(e["total_pnl"]),
                format_pct(e["avg_pnl_pct"]),
            ])
        print_table(
            ["Exit Type", "Trades", "Win%", "Total P&L", "Avg%"],
            rows,
            title="BY EXIT REASON",
        )

    # ----- Daily P&L -----
    daily_data = trade_logger.get_daily_pnl(min(days, 30) if days > 0 else 30)
    if daily_data:
        rows = []
        cumulative = 0
        for d in reversed(daily_data):  # chronological order
            cumulative += d["daily_pnl"] or 0
            rows.append([
                d["exit_date"],
                str(d["trades_closed"]),
                f"{d['wins']}/{d['losses']}",
                format_pnl(d["daily_pnl"]),
                format_pnl(cumulative),
            ])
        print_table(
            ["Date", "Trades", "W/L", "Daily P&L", "Cumulative"],
            rows,
            title="DAILY P&L (Last 30 Days)",
        )

    # ----- Open Positions -----
    if open_positions:
        rows = []
        for p in open_positions:
            rows.append([
                p["symbol"],
                p["direction"],
                p["strategy"],
                f"T{p['conviction_tier']}",
                f"${p['entry_price']:.2f}",
                p["entry_date"],
                p.get("entry_regime") or "—",
            ])
        print_table(
            ["Symbol", "Dir", "Strategy", "Tier", "Entry$", "Entry Date", "Regime"],
            rows,
            title="OPEN POSITIONS",
        )

    print(f"\n{'='*60}")
    print(f"  Report complete. DB: {trade_logger.db_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Performance Attribution Report")
    parser.add_argument(
        "--days", type=int, default=30,
        help="Report period in days (0 = all time, default: 30)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of formatted tables",
    )
    args = parser.parse_args()

    logger = TradeLogger()
    generate_report(logger, days=args.days, output_json=args.json)
    logger.close()


if __name__ == "__main__":
    main()
