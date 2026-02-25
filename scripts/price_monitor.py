#!/usr/bin/env python3
"""Real-Time Price Monitor — runs every 30 seconds during market hours.

Lightweight process that:
    1. Checks current prices for all held positions
    2. Triggers instant stop-loss exits if price drops below stop
    3. Triggers take-profit exits (2-stage partial scale-out)
    4. Dynamically tightens trailing stops as profit grows
    5. Logs price snapshots for tracking

Water marks (high/low) are persisted in position metadata so
trailing stops survive monitor restarts.

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
from portfolio.drawdown import (
    DrawdownCircuitBreaker, DrawdownLevel,
    get_peak_equity_from_snapshots, get_weakest_positions,
)
from portfolio.position_metadata import PositionMeta, load_metadata, save_metadata
from portfolio.trade_logger import TradeLogger, TradeExit


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
            pos_side = pos.side
            if hasattr(pos_side, "value"):
                pos_side = pos_side.value
            result.append({
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "change_today": float(pos.change_today),
                "side": str(pos_side),
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


def _dynamic_trail_mult(base_mult: float, pnl_pct: float) -> float:
    """Tighten the trailing stop multiplier as profit grows.

    Positions with large unrealized gains get tighter stops to lock in profit.
    The broker-side trailing stop remains as the outer safety net.

    Returns adjusted trail multiplier (always <= base_mult).
    """
    if pnl_pct < 0.05:
        return base_mult           # Not yet profitable enough to tighten
    elif pnl_pct < 0.10:
        return base_mult           # Up 5-10%: keep original
    elif pnl_pct < 0.15:
        return base_mult * 0.85    # Up 10-15%: 15% tighter
    elif pnl_pct < 0.20:
        return base_mult * 0.70    # Up 15-20%: 30% tighter
    else:
        return base_mult * 0.55    # Up 20%+: 45% tighter — lock in most of gain


def _replace_broker_trailing_stop(
    trading_client: TradingClient,
    meta: PositionMeta,
    symbol: str,
    entry: float,
    exit_side: OrderSide,
) -> None:
    """Cancel and replace the broker-side trailing stop with updated qty."""
    if not meta.broker_trailing_stop_order_id:
        return
    try:
        trading_client.cancel_order_by_id(
            meta.broker_trailing_stop_order_id
        )
        new_trail_pct = min(
            12.0,
            max(
                3.0,
                (meta.trail_stop_atr_mult * meta.atr_at_entry / entry) * 100,
            ),
        )
        new_order = TrailingStopOrderRequest(
            symbol=symbol,
            qty=meta.remaining_qty,
            side=exit_side,
            time_in_force=TimeInForce.GTC,
            trail_percent=round(new_trail_pct, 1),
        )
        result = trading_client.submit_order(order_data=new_order)
        meta.broker_trailing_stop_order_id = str(result.id)
    except Exception as e:
        alert(
            f"Failed to replace trailing stop for {symbol}: {e}",
            AlertLevel.ERROR,
        )


def check_exits(
    positions: list[dict],
    position_meta: dict[str, PositionMeta],
    trading_client: TradingClient,
    paper: bool,
    trade_logger: TradeLogger | None = None,
) -> dict[str, PositionMeta]:
    """Check all positions for exit triggers.

    Checks in order: hard stop, ATR trailing stop (dynamically tightened),
    BB middle target, first profit target (partial), second profit target
    (partial), time-based exit.

    Water marks are stored in position metadata and persist across restarts.

    Returns updated position_meta.
    """
    meta_changed = False

    for pos in positions:
        symbol = pos["symbol"]
        entry = pos["avg_entry_price"]
        current = pos["current_price"]
        pnl_pct = pos["unrealized_plpc"]
        qty = abs(pos["qty"])

        # Determine position direction
        meta = position_meta.get(symbol)
        is_short = (meta and meta.position_side == "short") or pos.get("side") == "short"

        # Update water marks in metadata (persisted across restarts)
        if meta:
            if is_short:
                prev_low = meta.low_water_mark if meta.low_water_mark is not None else entry
                if current < prev_low:
                    meta.low_water_mark = current
                    meta_changed = True
                lwm = meta.low_water_mark if meta.low_water_mark is not None else current
            else:
                prev_high = meta.high_water_mark if meta.high_water_mark is not None else entry
                if current > prev_high:
                    meta.high_water_mark = current
                    meta_changed = True
                hwm = meta.high_water_mark if meta.high_water_mark is not None else current
        else:
            # No metadata — use current price as fallback
            hwm = current
            lwm = current

        exit_reason = None
        exit_qty = None  # None = full position, int = partial

        # === 1. Hard stop: absolute max loss from entry (always applies) ===
        # pnl_pct from Alpaca already accounts for direction
        if pnl_pct <= -HARD_STOP_PCT:
            exit_reason = (
                f"HARD STOP: {pnl_pct:+.1%} loss "
                f"(limit: -{HARD_STOP_PCT:.0%})"
            )

        # === 2. ATR-based trailing stop (dynamically tightened) ===
        elif meta and meta.atr_at_entry > 0:
            # Dynamically tighten trail as profit grows
            effective_mult = _dynamic_trail_mult(
                meta.trail_stop_atr_mult, pnl_pct
            )
            trail_distance = effective_mult * meta.atr_at_entry

            if is_short:
                # Short: trail above low-water mark
                if lwm < entry:
                    trail_stop_price = lwm + trail_distance
                    if current > trail_stop_price:
                        exit_reason = (
                            f"ATR TRAILING STOP (short): ${current:.2f} above "
                            f"${trail_stop_price:.2f} "
                            f"(LWM ${lwm:.2f} + {effective_mult:.1f}x "
                            f"ATR ${meta.atr_at_entry:.2f})"
                        )
            else:
                # Long: trail below high-water mark
                if hwm > entry:
                    trail_stop_price = hwm - trail_distance
                    if current < trail_stop_price:
                        exit_reason = (
                            f"ATR TRAILING STOP: ${current:.2f} below "
                            f"${trail_stop_price:.2f} "
                            f"(HWM ${hwm:.2f} - {effective_mult:.1f}x "
                            f"ATR ${meta.atr_at_entry:.2f})"
                        )

        # Fallback flat trailing stop for positions without metadata
        elif not meta:
            if is_short:
                if lwm < entry and current > lwm * (1 + TRAILING_STOP_PCT):
                    rise_from_lwm = (current - lwm) / lwm
                    exit_reason = (
                        f"TRAILING STOP (flat, short): ${current:.2f} is "
                        f"{rise_from_lwm:+.1%} from low ${lwm:.2f}"
                    )
            else:
                if hwm > entry and current < hwm * (1 - TRAILING_STOP_PCT):
                    drop_from_hwm = (current - hwm) / hwm
                    exit_reason = (
                        f"TRAILING STOP (flat): ${current:.2f} is "
                        f"{drop_from_hwm:+.1%} from high ${hwm:.2f}"
                    )

        # === 3. Mean reversion: exit 100% at BB middle ===
        if not exit_reason and meta and meta.exit_at_bb_middle:
            if meta.bb_middle_at_entry:
                if is_short:
                    # Short mean reversion: target is price FALLING to BB middle
                    if current <= meta.bb_middle_at_entry:
                        exit_reason = (
                            f"BB MIDDLE TARGET (short): ${current:.2f} <= "
                            f"BB middle ${meta.bb_middle_at_entry:.2f}"
                        )
                else:
                    if current >= meta.bb_middle_at_entry:
                        exit_reason = (
                            f"BB MIDDLE TARGET: ${current:.2f} >= "
                            f"BB middle ${meta.bb_middle_at_entry:.2f}"
                        )

        # === 4. First profit target: partial scale-out (stage 1) ===
        # pnl_pct from Alpaca is positive for profitable positions (both sides)
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
                    f"Scaling out {sell_qty}/{int(qty)} shares (stage 1)."
                )
                exit_qty = sell_qty  # partial exit

        # === 5. Second profit target: partial scale-out (stage 2) ===
        if (
            not exit_reason
            and meta
            and meta.first_target_hit
            and not meta.second_target_hit
            and meta.second_target_pct > 0
            and pnl_pct >= meta.second_target_pct
        ):
            sell_qty = int(qty * meta.second_sell_pct)
            if sell_qty >= 1:
                exit_reason = (
                    f"SECOND TARGET: {pnl_pct:+.1%} >= "
                    f"{meta.second_target_pct:+.0%} target. "
                    f"Scaling out {sell_qty}/{int(qty)} shares (stage 2)."
                )
                exit_qty = sell_qty  # partial exit

        # === 6. Time-based exit (earnings_momentum PEAD exhaustion) ===
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
            close_shares = exit_qty if is_partial else int(qty)

            # Determine order side: opposite of position direction
            exit_side = OrderSide.BUY if is_short else OrderSide.SELL

            alert(
                f"{'PARTIAL ' if is_partial else ''}EXIT {symbol}: "
                f"{exit_reason} | "
                f"Entry: ${entry:.2f} -> Current: ${current:.2f} "
                f"(P&L: ${pos['unrealized_pl']:+.2f}, {pnl_pct:+.1%})",
                AlertLevel.WARNING,
            )
            try:
                if is_partial:
                    # Partial close via market order
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=close_shares,
                        side=exit_side,
                        time_in_force=TimeInForce.DAY,
                    )
                    result = trading_client.submit_order(order_data=order)

                    # Log partial exit to TradeLogger
                    if trade_logger:
                        try:
                            trade_logger.log_exit(TradeExit(
                                symbol=symbol,
                                exit_price=current,
                                exit_qty=close_shares,
                                exit_reason=exit_reason,
                                exit_order_id=str(result.id),
                            ))
                        except Exception as log_err:
                            print(f"  Warning: failed to log partial exit: {log_err}", file=sys.stderr)

                    # Update metadata — determine which target was hit
                    if meta:
                        if not meta.first_target_hit:
                            meta.first_target_hit = True
                        elif not meta.second_target_hit:
                            meta.second_target_hit = True
                        meta.remaining_qty = int(qty) - close_shares
                        meta_changed = True

                    # Cancel and replace broker-side trailing stop
                    if meta:
                        _replace_broker_trailing_stop(
                            trading_client, meta, symbol, entry, exit_side,
                        )

                    print(
                        f"  Partial close: {close_shares} shares, "
                        f"remaining {int(qty) - close_shares}",
                        file=sys.stderr,
                    )
                else:
                    # Full close — close_position handles direction automatically
                    trading_client.close_position(symbol)

                    # Log full exit to TradeLogger
                    if trade_logger:
                        try:
                            trade_logger.log_exit(TradeExit(
                                symbol=symbol,
                                exit_price=current,
                                exit_qty=int(qty),
                                exit_reason=exit_reason,
                            ))
                        except Exception as log_err:
                            print(f"  Warning: failed to log exit: {log_err}", file=sys.stderr)

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

    return position_meta


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
    position_meta: dict[str, PositionMeta] = {}
    consecutive_errors = 0
    last_market_check = 0
    last_snapshot_time = 0
    last_drawdown_check = 0
    market_open = False
    kill_switch_triggered = False
    starting_equity: float = 0.0  # captured at market open each day

    # Initialize trade logger for exit recording
    trade_logger = TradeLogger()
    print(f"[Monitor] Trade logger initialized: {trade_logger.db_path}", file=sys.stderr)

    # Initialize drawdown circuit breaker
    drawdown_breaker = DrawdownCircuitBreaker()
    peak_equity = get_peak_equity_from_snapshots(trade_logger)
    print(
        f"[Monitor] Drawdown breaker initialized (peak: ${peak_equity:,.2f})",
        file=sys.stderr,
    )

    # Load position metadata from disk (includes persisted water marks)
    try:
        position_meta = load_metadata()
        hwm_count = sum(
            1 for m in position_meta.values() if m.high_water_mark is not None
        )
        lwm_count = sum(
            1 for m in position_meta.values() if m.low_water_mark is not None
        )
        print(
            f"[Monitor] Loaded metadata for {len(position_meta)} positions "
            f"({hwm_count} HWMs, {lwm_count} LWMs restored)",
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

            # Graduated drawdown circuit breaker (check every 2 minutes)
            now_dd = time.time()
            if now_dd - last_drawdown_check > 120:
                last_drawdown_check = now_dd
                # Update peak from snapshots or current equity
                peak_equity = max(peak_equity, equity)
                dd_state = drawdown_breaker.check(equity, peak_equity)

                if dd_state.level >= DrawdownLevel.CAUTION:
                    print(
                        f"[Monitor] Drawdown: {dd_state.description}",
                        file=sys.stderr,
                    )

                if dd_state.unwind_weakest and positions:
                    # Close the 2 weakest positions
                    weakest = get_weakest_positions(positions, n=2)
                    for weak_pos in weakest:
                        sym = weak_pos["symbol"]
                        alert(
                            f"DRAWDOWN UNWIND: Closing {sym} "
                            f"(P&L: {weak_pos.get('unrealized_plpc', 0):+.1%})",
                            AlertLevel.CRITICAL,
                        )
                        try:
                            trading_client.close_position(sym)
                            # Log exit
                            if trade_logger:
                                try:
                                    trade_logger.log_exit(TradeExit(
                                        symbol=sym,
                                        exit_price=weak_pos["current_price"],
                                        exit_qty=int(abs(weak_pos["qty"])),
                                        exit_reason=(
                                            f"DRAWDOWN UNWIND: "
                                            f"{dd_state.drawdown_pct:.1%} drawdown"
                                        ),
                                    ))
                                except Exception:
                                    pass
                            if sym in position_meta:
                                del position_meta[sym]
                        except Exception as e:
                            alert(
                                f"Failed to unwind {sym}: {e}",
                                AlertLevel.ERROR,
                            )
                    save_metadata(position_meta)

            # Check for exits (ATR trailing, profit targets, time exits)
            position_meta = check_exits(
                positions, position_meta, trading_client, paper,
                trade_logger=trade_logger,
            )

            # Log portfolio snapshot every 5 minutes
            now_ts = time.time()
            if now_ts - last_snapshot_time > 300:
                try:
                    num_long = sum(1 for p in positions if p.get("side") != "short")
                    num_short = sum(1 for p in positions if p.get("side") == "short")
                    total_unrealized = sum(p.get("unrealized_pl", 0) for p in positions)
                    trade_logger.log_snapshot(
                        equity=equity,
                        cash=0.0,  # not available from positions endpoint
                        num_positions=len(positions),
                        num_long=num_long,
                        num_short=num_short,
                        total_unrealized_pnl=total_unrealized,
                    )
                    last_snapshot_time = now_ts
                except Exception:
                    pass  # non-critical

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
