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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, TrailingStopOrderRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from alerting import alert, AlertLevel
from portfolio.position_metadata import PositionMeta, load_metadata, save_metadata


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


def count_trading_days(entry_date_str: str) -> int:
    """Count weekdays between entry date and today (approximate trading days)."""
    entry = date.fromisoformat(entry_date_str)
    today = date.today()
    count = 0
    d = entry
    while d < today:
        if d.weekday() < 5:  # Mon-Fri
            count += 1
        d += timedelta(days=1)
    return count


def check_exits(
    positions: list[dict],
    high_water_marks: dict[str, float],
    position_meta: dict[str, PositionMeta],
    trading_client: TradingClient,
    paper: bool,
) -> tuple[dict[str, float], dict[str, PositionMeta]]:
    """Check all positions for exit triggers.

    Checks in order: hard stop, ATR trailing stop, BB middle target,
    first profit target (partial scale-out), time-based exit.

    Returns updated (high_water_marks, position_meta).
    """
    meta_changed = False

    for pos in positions:
        symbol = pos["symbol"]
        entry = pos["avg_entry_price"]
        current = pos["current_price"]
        pnl_pct = pos["unrealized_plpc"]
        qty = pos["qty"]

        # Update high-water mark
        prev_high = high_water_marks.get(symbol, entry)
        if current > prev_high:
            high_water_marks[symbol] = current
        hwm = high_water_marks.get(symbol, current)

        # Get position metadata (may not exist for legacy positions)
        meta = position_meta.get(symbol)

        exit_reason = None
        exit_qty = None  # None = full position, int = partial

        # === 1. Hard stop: absolute max loss from entry (always applies) ===
        if pnl_pct <= -HARD_STOP_PCT:
            exit_reason = (
                f"HARD STOP: {pnl_pct:+.1%} loss "
                f"(limit: -{HARD_STOP_PCT:.0%})"
            )

        # === 2. ATR-based trailing stop (replaces flat 5%) ===
        elif meta and meta.atr_at_entry > 0 and hwm > entry:
            trail_distance = meta.trail_stop_atr_mult * meta.atr_at_entry
            trail_stop_price = hwm - trail_distance
            if current < trail_stop_price:
                exit_reason = (
                    f"ATR TRAILING STOP: ${current:.2f} below "
                    f"${trail_stop_price:.2f} "
                    f"(HWM ${hwm:.2f} - {meta.trail_stop_atr_mult:.1f}x "
                    f"ATR ${meta.atr_at_entry:.2f})"
                )

        # Fallback flat trailing stop for positions without metadata
        elif not meta and hwm > entry and current < hwm * (1 - TRAILING_STOP_PCT):
            drop_from_hwm = (current - hwm) / hwm
            exit_reason = (
                f"TRAILING STOP (flat): ${current:.2f} is "
                f"{drop_from_hwm:+.1%} from high ${hwm:.2f}"
            )

        # === 3. Mean reversion: exit 100% at BB middle ===
        if not exit_reason and meta and meta.exit_at_bb_middle:
            if meta.bb_middle_at_entry and current >= meta.bb_middle_at_entry:
                exit_reason = (
                    f"BB MIDDLE TARGET: ${current:.2f} >= "
                    f"BB middle ${meta.bb_middle_at_entry:.2f}"
                )

        # === 4. First profit target: partial scale-out ===
        if (
            not exit_reason
            and meta
            and not meta.first_target_hit
            and meta.first_target_pct > 0
            and pnl_pct >= meta.first_target_pct
        ):
            sell_qty = int(qty * meta.partial_sell_pct)
            if sell_qty >= 1:
                exit_reason = (
                    f"FIRST TARGET: {pnl_pct:+.1%} >= "
                    f"{meta.first_target_pct:+.0%} target. "
                    f"Scaling out {sell_qty}/{int(qty)} shares."
                )
                exit_qty = sell_qty  # partial exit

        # === 5. Time-based exit (earnings_momentum PEAD exhaustion) ===
        if not exit_reason and meta and meta.time_exit_days:
            days_held = count_trading_days(meta.entry_date)
            if days_held >= meta.time_exit_days:
                exit_reason = (
                    f"TIME EXIT: held {days_held} trading days "
                    f"(limit: {meta.time_exit_days}d for "
                    f"{meta.strategy})"
                )

        # === Execute exit ===
        if exit_reason:
            is_partial = exit_qty is not None
            sell_shares = exit_qty if is_partial else int(qty)

            alert(
                f"{'PARTIAL ' if is_partial else ''}EXIT {symbol}: "
                f"{exit_reason} | "
                f"Entry: ${entry:.2f} -> Current: ${current:.2f} "
                f"(P&L: ${pos['unrealized_pl']:+.2f}, {pnl_pct:+.1%})",
                AlertLevel.WARNING,
            )
            try:
                if is_partial:
                    # Partial sell via market order
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=sell_shares,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    trading_client.submit_order(order_data=order)

                    # Update metadata
                    if meta:
                        meta.first_target_hit = True
                        meta.remaining_qty = int(qty) - sell_shares
                        meta_changed = True

                    # Cancel and replace broker-side trailing stop
                    # with reduced quantity
                    if meta and meta.broker_trailing_stop_order_id:
                        try:
                            trading_client.cancel_order_by_id(
                                meta.broker_trailing_stop_order_id
                            )
                            new_trail_pct = min(
                                12.0,
                                max(
                                    3.0,
                                    (
                                        meta.trail_stop_atr_mult
                                        * meta.atr_at_entry
                                        / entry
                                    )
                                    * 100,
                                ),
                            )
                            new_order = TrailingStopOrderRequest(
                                symbol=symbol,
                                qty=meta.remaining_qty,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.GTC,
                                trail_percent=round(new_trail_pct, 1),
                            )
                            result = trading_client.submit_order(
                                order_data=new_order
                            )
                            meta.broker_trailing_stop_order_id = str(
                                result.id
                            )
                        except Exception as e:
                            alert(
                                f"Failed to replace trailing stop for "
                                f"{symbol}: {e}",
                                AlertLevel.ERROR,
                            )

                    print(
                        f"  Partial close: sold {sell_shares}, "
                        f"remaining {int(qty) - sell_shares}",
                        file=sys.stderr,
                    )
                else:
                    # Full close
                    trading_client.close_position(symbol)
                    high_water_marks.pop(symbol, None)
                    if symbol in position_meta:
                        del position_meta[symbol]
                        meta_changed = True

                    # Cancel broker-side trailing stop if exists
                    if meta and meta.broker_trailing_stop_order_id:
                        try:
                            trading_client.cancel_order_by_id(
                                meta.broker_trailing_stop_order_id
                            )
                        except Exception:
                            pass  # may already be filled/cancelled

                    print(f"  Position closed.", file=sys.stderr)

            except Exception as e:
                alert(f"Failed to close {symbol}: {e}", AlertLevel.ERROR)

    if meta_changed:
        save_metadata(position_meta)

    return high_water_marks, position_meta


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
        alert(
            f"KILL SWITCH: Daily loss ${daily_pnl:+.2f} exceeds "
            f"max ${MAX_DAILY_LOSS}. CLOSING ALL POSITIONS.",
            AlertLevel.CRITICAL,
        )
        return True

    # Drawdown from starting equity
    drawdown_pct = (equity - starting_equity) / starting_equity
    if drawdown_pct < -DRAWDOWN_KILL_SWITCH:
        alert(
            f"KILL SWITCH: Drawdown {drawdown_pct:.1%} exceeds "
            f"max {DRAWDOWN_KILL_SWITCH:.0%}. CLOSING ALL POSITIONS.",
            AlertLevel.CRITICAL,
        )
        return True

    return False


def close_all_positions(trading_client: TradingClient):
    """Emergency close all positions."""
    try:
        trading_client.close_all_positions(cancel_orders=True)
        alert("All positions closed, all orders cancelled.", AlertLevel.CRITICAL)
    except Exception as e:
        alert(f"FAILED to close all positions: {e}", AlertLevel.CRITICAL)


def run_loop(trading_client: TradingClient, interval: int, paper: bool):
    """Main monitoring loop."""
    high_water_marks: dict[str, float] = {}
    position_meta: dict[str, PositionMeta] = {}
    consecutive_errors = 0
    last_market_check = 0
    market_open = False
    kill_switch_triggered = False
    starting_equity: float = 0.0  # captured at market open each day

    # Load position metadata from disk
    try:
        position_meta = load_metadata()
        print(
            f"[Monitor] Loaded metadata for {len(position_meta)} positions",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[Monitor] Could not load position metadata: {e}",
            file=sys.stderr,
        )

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

                # Market just opened — capture starting equity and
                # reload metadata (run_daily.py may have written new entries)
                if not was_open and market_open:
                    try:
                        account = trading_client.get_account()
                        starting_equity = float(account.equity)
                        alert(
                            f"Market open. Equity: ${starting_equity:,.2f}",
                            AlertLevel.INFO,
                        )
                    except Exception:
                        pass
                    try:
                        position_meta = load_metadata()
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

            # Check for exits (ATR trailing, profit targets, time exits)
            high_water_marks, position_meta = check_exits(
                positions, high_water_marks, position_meta,
                trading_client, paper,
            )

            # Clean up orphaned metadata for positions that no longer exist
            held_symbols = {p["symbol"] for p in positions}
            orphans = [s for s in position_meta if s not in held_symbols]
            if orphans:
                for s in orphans:
                    del position_meta[s]
                save_metadata(position_meta)

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
                alert(
                    "Monitor: 10+ consecutive errors. Sleeping 5 min.",
                    AlertLevel.ERROR,
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
