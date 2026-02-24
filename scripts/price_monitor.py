#!/usr/bin/env python3
"""Real-Time Price Monitor — runs every 30 seconds during market hours.

Lightweight process that:
    1. Checks current prices for all held positions
    2. Triggers instant stop-loss exits if price drops below stop
    3. Triggers take-profit exits
    4. Logs price snapshots for tracking

This does NOT run FinBERT or full strategies — it's a fast price-only
loop that protects capital between full engine runs.

Usage:
    python scripts/price_monitor.py [--paper] [--interval 30]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient


# Stop-loss and take-profit thresholds
STOP_LOSS_ATR_MULT = 2.0      # exit if price drops 2x ATR below entry
TAKE_PROFIT_ATR_MULT = 4.0    # exit if price rises 4x ATR above entry
TRAILING_STOP_PCT = 0.05       # 5% trailing stop from high-water mark
HARD_STOP_PCT = 0.08           # 8% max loss from entry — absolute floor

# Portfolio-level kill switch
MAX_DAILY_LOSS = float(os.environ.get("MAX_DAILY_LOSS", "5000"))
DRAWDOWN_KILL_SWITCH = float(os.environ.get("DRAWDOWN_KILL_SWITCH", "0.10"))


def get_clients(paper: bool):
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print(
            "ERROR: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set",
            file=sys.stderr,
        )
        sys.exit(1)
    data_client = StockHistoricalDataClient(key, secret)
    trading_client = TradingClient(key, secret, paper=paper)
    return data_client, trading_client


def is_market_open(trading_client: TradingClient) -> bool:
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception as e:
        print(f"[Monitor] Error checking market status: {e}", file=sys.stderr)
        return False


def get_positions(trading_client: TradingClient) -> list[dict]:
    """Get all current positions with P&L info."""
    try:
        positions = trading_client.get_all_positions()
        result = []
        for pos in positions:
            result.append({
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "change_today": float(pos.change_today),
            })
        return result
    except Exception as e:
        print(f"[Monitor] Error fetching positions: {e}", file=sys.stderr)
        return []


def check_exits(
    positions: list[dict],
    high_water_marks: dict[str, float],
    trading_client: TradingClient,
    paper: bool,
) -> dict[str, float]:
    """Check all positions for stop-loss or take-profit triggers.

    Returns updated high_water_marks dict.
    """
    for pos in positions:
        symbol = pos["symbol"]
        entry = pos["avg_entry_price"]
        current = pos["current_price"]
        pnl_pct = pos["unrealized_plpc"]

        # Update high-water mark
        prev_high = high_water_marks.get(symbol, entry)
        if current > prev_high:
            high_water_marks[symbol] = current

        hwm = high_water_marks.get(symbol, current)

        # Check triggers
        exit_reason = None

        # 1. Hard stop: absolute max loss from entry
        if pnl_pct <= -HARD_STOP_PCT:
            exit_reason = f"HARD STOP: {pnl_pct:+.1%} loss (limit: -{HARD_STOP_PCT:.0%})"

        # 2. Trailing stop: drop from high-water mark
        elif hwm > entry and current < hwm * (1 - TRAILING_STOP_PCT):
            drop_from_hwm = (current - hwm) / hwm
            exit_reason = (
                f"TRAILING STOP: ${current:.2f} is {drop_from_hwm:+.1%} "
                f"from high of ${hwm:.2f}"
            )

        # 3. Take profit (optional — let winners run with trailing stop)
        # Uncomment if you want hard take-profit:
        # elif pnl_pct >= 0.15:  # 15% gain
        #     exit_reason = f"TAKE PROFIT: {pnl_pct:+.1%} gain"

        if exit_reason:
            print(
                f"\n[Monitor] EXIT {symbol}: {exit_reason}\n"
                f"  Entry: ${entry:.2f} → Current: ${current:.2f} "
                f"(P&L: ${pos['unrealized_pl']:+.2f}, {pnl_pct:+.1%})",
                file=sys.stderr,
            )
            try:
                trading_client.close_position(symbol)
                print(f"  ✓ Position closed.", file=sys.stderr)
                # Remove from high water marks
                high_water_marks.pop(symbol, None)
            except Exception as e:
                print(f"  ✗ Error closing {symbol}: {e}", file=sys.stderr)

    return high_water_marks


def log_snapshot(positions: list[dict], equity: float):
    """Print a compact snapshot to stderr."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    if not positions:
        print(f"[{now}] No positions | Equity: ${equity:,.2f}", file=sys.stderr)
        return

    parts = []
    total_pnl = 0.0
    for pos in positions:
        pnl = pos["unrealized_pl"]
        pnl_pct = pos["unrealized_plpc"]
        total_pnl += pnl
        parts.append(f"{pos['symbol']}:{pnl_pct:+.1%}")

    line = " | ".join(parts)
    print(
        f"[{now}] {line} | Total: ${total_pnl:+.2f} | Equity: ${equity:,.2f}",
        file=sys.stderr,
    )


def check_kill_switch(
    trading_client: TradingClient,
    equity: float,
    starting_equity: float,
) -> bool:
    """Check portfolio-level circuit breakers. Returns True if trading should halt."""
    if starting_equity <= 0 or equity <= 0:
        return False

    # Daily P&L check
    daily_pnl = equity - starting_equity
    if daily_pnl < -MAX_DAILY_LOSS:
        print(
            f"\n[KILL SWITCH] Daily loss ${daily_pnl:+.2f} exceeds "
            f"max ${MAX_DAILY_LOSS}. CLOSING ALL POSITIONS.",
            file=sys.stderr,
        )
        return True

    # Drawdown from starting equity
    drawdown_pct = (equity - starting_equity) / starting_equity
    if drawdown_pct < -DRAWDOWN_KILL_SWITCH:
        print(
            f"\n[KILL SWITCH] Drawdown {drawdown_pct:.1%} exceeds "
            f"max {DRAWDOWN_KILL_SWITCH:.0%}. CLOSING ALL POSITIONS.",
            file=sys.stderr,
        )
        return True

    return False


def close_all_positions(trading_client: TradingClient):
    """Emergency close all positions."""
    try:
        trading_client.close_all_positions(cancel_orders=True)
        print("[KILL SWITCH] All positions closed, all orders cancelled.", file=sys.stderr)
    except Exception as e:
        print(f"[KILL SWITCH] ERROR closing positions: {e}", file=sys.stderr)


def run_loop(trading_client: TradingClient, interval: int, paper: bool):
    """Main monitoring loop."""
    high_water_marks: dict[str, float] = {}
    consecutive_errors = 0
    last_market_check = 0
    market_open = False
    kill_switch_triggered = False
    starting_equity: float = 0.0  # captured at market open each day

    print(
        f"[Monitor] Starting price monitor (interval={interval}s, "
        f"mode={'PAPER' if paper else 'LIVE'})",
        file=sys.stderr,
    )
    print(
        f"[Monitor] Kill switch: max daily loss=${MAX_DAILY_LOSS:,.0f}, "
        f"max drawdown={DRAWDOWN_KILL_SWITCH:.0%}",
        file=sys.stderr,
    )

    while True:
        try:
            now = time.time()

            # Check market status every 60 seconds (not every tick)
            if now - last_market_check > 60:
                was_open = market_open
                market_open = is_market_open(trading_client)
                last_market_check = now

                if not market_open:
                    if was_open:
                        # Market just closed — reset for tomorrow
                        print(
                            f"[Monitor] Market closed. Resetting kill switch.",
                            file=sys.stderr,
                        )
                        kill_switch_triggered = False
                        starting_equity = 0.0
                    print(
                        f"[Monitor] Market closed. Sleeping 5 min...",
                        file=sys.stderr,
                    )
                    time.sleep(300)
                    continue

                # Market just opened — capture starting equity
                if not was_open and market_open:
                    try:
                        account = trading_client.get_account()
                        starting_equity = float(account.equity)
                        print(
                            f"[Monitor] Market open! Starting equity: "
                            f"${starting_equity:,.2f}",
                            file=sys.stderr,
                        )
                    except Exception:
                        pass

            if not market_open:
                time.sleep(60)
                continue

            # If kill switch was triggered, don't trade — just monitor
            if kill_switch_triggered:
                time.sleep(interval)
                continue

            # Get positions and account
            positions = get_positions(trading_client)
            try:
                account = trading_client.get_account()
                equity = float(account.equity)
            except Exception:
                equity = 0.0

            # Check portfolio kill switch
            if starting_equity > 0 and check_kill_switch(
                trading_client, equity, starting_equity
            ):
                close_all_positions(trading_client)
                kill_switch_triggered = True
                continue

            # Check for exits
            high_water_marks = check_exits(
                positions, high_water_marks, trading_client, paper
            )

            # Log snapshot
            log_snapshot(positions, equity)

            consecutive_errors = 0
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[Monitor] Stopped by user.", file=sys.stderr)
            break
        except Exception as e:
            consecutive_errors += 1
            print(
                f"[Monitor] Error (#{consecutive_errors}): {e}",
                file=sys.stderr,
            )
            if consecutive_errors > 10:
                print(
                    "[Monitor] Too many consecutive errors. Sleeping 5 min...",
                    file=sys.stderr,
                )
                time.sleep(300)
                consecutive_errors = 0
            else:
                time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Real-time price monitor")
    parser.add_argument(
        "--paper", action="store_true", default=True,
        help="Use paper trading (default: true)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live trading",
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Check interval in seconds (default: 30)",
    )
    args = parser.parse_args()

    paper = not args.live
    _, trading_client = get_clients(paper)

    run_loop(trading_client, args.interval, paper)


if __name__ == "__main__":
    main()
