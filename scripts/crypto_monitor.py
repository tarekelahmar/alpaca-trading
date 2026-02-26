#!/usr/bin/env python3
"""Crypto Price Monitor — runs 24/7 (no market hours check).

Crypto equivalent of price_monitor.py. Key differences:
    - No market hours check (crypto trades 24/7)
    - Wider hard stop (12% vs 8% for equities — crypto is more volatile)
    - Wider trailing stop default (8% vs 5%)
    - Separate position metadata file (crypto_position_metadata.json)
    - Higher daily loss tolerance (crypto daily swings are bigger)
    - 60-second default interval (vs 30s for equities)

Usage:
    python scripts/crypto_monitor.py [--paper] [--interval 60]
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))

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


# Crypto-specific thresholds (wider than equities)
HARD_STOP_PCT = 0.12          # 12% max loss (vs 8% equities)
TRAILING_STOP_PCT = 0.08      # 8% trailing (vs 5% equities)

# Portfolio-level kill switch
MAX_DAILY_LOSS = float(os.environ.get("CRYPTO_MAX_DAILY_LOSS", "3000"))
DRAWDOWN_KILL_SWITCH = float(os.environ.get("CRYPTO_DRAWDOWN_KILL_SWITCH", "0.15"))

# Separate metadata file
CRYPTO_METADATA_PATH = (
    Path(__file__).parent.parent / "strategy-engine" / "crypto_position_metadata.json"
)


def get_clients(paper: bool):
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print(
            "ERROR: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set",
            file=sys.stderr,
        )
        sys.exit(1)
    trading_client = TradingClient(key, secret, paper=paper)
    return trading_client


def get_crypto_positions(trading_client: TradingClient) -> list[dict]:
    """Get all current crypto positions with P&L info.

    Filters to only include crypto symbols (contain '/' or 'USD' suffix).
    """
    try:
        positions = trading_client.get_all_positions()
        result = []
        for pos in positions:
            symbol = pos.symbol
            # Detect crypto: Alpaca crypto symbols contain USD
            # and are typically formatted as BTCUSD or BTC/USD
            is_crypto = (
                "/" in symbol
                or (symbol.endswith("USD") and not symbol.startswith("$"))
            )
            if not is_crypto:
                continue

            # Normalize symbol
            if "/" not in symbol and "USD" in symbol:
                symbol = symbol.replace("USD", "/USD")

            pos_side = pos.side
            if hasattr(pos_side, "value"):
                pos_side = pos_side.value

            result.append({
                "symbol": symbol,
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
        print(f"[CryptoMonitor] Error fetching positions: {e}", file=sys.stderr)
        return []


def count_calendar_days(entry_date_str: str) -> int:
    """Count calendar days since entry (crypto trades every day)."""
    entry = date.fromisoformat(entry_date_str)
    return (date.today() - entry).days


def _dynamic_trail_mult(base_mult: float, pnl_pct: float) -> float:
    """Tighten trailing stop as profit grows (crypto version — wider thresholds).

    Crypto positions need more room to breathe due to higher volatility.
    Tightening starts at 10% (vs 5% for equities).
    """
    if pnl_pct < 0.10:
        return base_mult           # Not profitable enough to tighten
    elif pnl_pct < 0.20:
        return base_mult * 0.85    # Up 10-20%: 15% tighter
    elif pnl_pct < 0.30:
        return base_mult * 0.70    # Up 20-30%: 30% tighter
    elif pnl_pct < 0.50:
        return base_mult * 0.60    # Up 30-50%: 40% tighter
    else:
        return base_mult * 0.50    # Up 50%+: 50% tighter — lock in big gains


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
        trading_client.cancel_order_by_id(meta.broker_trailing_stop_order_id)
        new_trail_pct = min(
            20.0,  # wider cap for crypto
            max(5.0, (meta.trail_stop_atr_mult * meta.atr_at_entry / entry) * 100),
        )
        # Alpaca may need symbol without / for API calls
        order_symbol = symbol
        new_order = TrailingStopOrderRequest(
            symbol=order_symbol,
            qty=meta.remaining_qty,
            side=exit_side,
            time_in_force=TimeInForce.GTC,
            trail_percent=round(new_trail_pct, 1),
        )
        result = trading_client.submit_order(order_data=new_order)
        meta.broker_trailing_stop_order_id = str(result.id)
    except Exception as e:
        alert(
            f"Failed to replace crypto trailing stop for {symbol}: {e}",
            AlertLevel.ERROR,
        )


def check_exits(
    positions: list[dict],
    position_meta: dict[str, PositionMeta],
    trading_client: TradingClient,
    paper: bool,
    trade_logger: TradeLogger | None = None,
) -> dict[str, PositionMeta]:
    """Check all crypto positions for exit triggers.

    Same logic as equity price_monitor but with wider thresholds
    and calendar days (not trading days) for time exits.
    """
    meta_changed = False

    for pos in positions:
        symbol = pos["symbol"]
        entry = pos["avg_entry_price"]
        current = pos["current_price"]
        pnl_pct = pos["unrealized_plpc"]
        qty = abs(pos["qty"])

        meta = position_meta.get(symbol)
        is_short = (
            (meta and meta.position_side == "short")
            or pos.get("side") == "short"
        )

        # Update water marks
        if meta:
            if is_short:
                prev_low = (
                    meta.low_water_mark if meta.low_water_mark is not None else entry
                )
                if current < prev_low:
                    meta.low_water_mark = current
                    meta_changed = True
                lwm = meta.low_water_mark if meta.low_water_mark is not None else current
            else:
                prev_high = (
                    meta.high_water_mark if meta.high_water_mark is not None else entry
                )
                if current > prev_high:
                    meta.high_water_mark = current
                    meta_changed = True
                hwm = meta.high_water_mark if meta.high_water_mark is not None else current
        else:
            hwm = current
            lwm = current

        exit_reason = None
        exit_qty = None

        # === 1. Hard stop (wider for crypto) ===
        if pnl_pct <= -HARD_STOP_PCT:
            exit_reason = (
                f"HARD STOP: {pnl_pct:+.1%} loss "
                f"(limit: -{HARD_STOP_PCT:.0%})"
            )

        # === 2. ATR-based trailing stop (dynamically tightened, crypto version) ===
        elif meta and meta.atr_at_entry > 0:
            effective_mult = _dynamic_trail_mult(meta.trail_stop_atr_mult, pnl_pct)
            trail_distance = effective_mult * meta.atr_at_entry

            if is_short:
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
                if hwm > entry:
                    trail_stop_price = hwm - trail_distance
                    if current < trail_stop_price:
                        exit_reason = (
                            f"ATR TRAILING STOP: ${current:.2f} below "
                            f"${trail_stop_price:.2f} "
                            f"(HWM ${hwm:.2f} - {effective_mult:.1f}x "
                            f"ATR ${meta.atr_at_entry:.2f})"
                        )

        # Fallback flat trailing stop
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

        # === 3. Mean reversion: exit at BB middle ===
        if not exit_reason and meta and meta.exit_at_bb_middle:
            if meta.bb_middle_at_entry:
                if is_short:
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
                    f"Scaling out {sell_qty}/{int(qty)} (stage 1)."
                )
                exit_qty = sell_qty

        # === 5. Second profit target: partial scale-out ===
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
                    f"Scaling out {sell_qty}/{int(qty)} (stage 2)."
                )
                exit_qty = sell_qty

        # === 6. Time-based exit (calendar days for crypto) ===
        if not exit_reason and meta and meta.time_exit_days:
            days_held = count_calendar_days(meta.entry_date)
            if days_held >= meta.time_exit_days:
                exit_reason = (
                    f"TIME EXIT: held {days_held} calendar days "
                    f"(limit: {meta.time_exit_days}d for {meta.strategy})"
                )

        # === Execute exit ===
        if exit_reason:
            is_partial = exit_qty is not None
            close_shares = exit_qty if is_partial else qty

            exit_side = OrderSide.BUY if is_short else OrderSide.SELL

            alert(
                f"{'PARTIAL ' if is_partial else ''}CRYPTO EXIT {symbol}: "
                f"{exit_reason} | "
                f"Entry: ${entry:.2f} -> Current: ${current:.2f} "
                f"(P&L: ${pos['unrealized_pl']:+.2f}, {pnl_pct:+.1%})",
                AlertLevel.WARNING,
            )
            try:
                if is_partial:
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=close_shares,
                        side=exit_side,
                        time_in_force=TimeInForce.GTC,
                    )
                    result = trading_client.submit_order(order_data=order)

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
                            print(
                                f"  Warning: failed to log partial exit: {log_err}",
                                file=sys.stderr,
                            )

                    if meta:
                        if not meta.first_target_hit:
                            meta.first_target_hit = True
                        elif not meta.second_target_hit:
                            meta.second_target_hit = True
                        meta.remaining_qty = int(qty) - close_shares
                        meta_changed = True

                    if meta:
                        _replace_broker_trailing_stop(
                            trading_client, meta, symbol, entry, exit_side,
                        )

                    print(
                        f"  Partial close: {close_shares}, "
                        f"remaining {int(qty) - close_shares}",
                        file=sys.stderr,
                    )
                else:
                    # Full close
                    # Try both symbol formats for Alpaca
                    try:
                        trading_client.close_position(symbol)
                    except Exception:
                        trading_client.close_position(symbol.replace("/", ""))

                    if trade_logger:
                        try:
                            trade_logger.log_exit(TradeExit(
                                symbol=symbol,
                                exit_price=current,
                                exit_qty=int(qty),
                                exit_reason=exit_reason,
                            ))
                        except Exception as log_err:
                            print(
                                f"  Warning: failed to log exit: {log_err}",
                                file=sys.stderr,
                            )

                    if symbol in position_meta:
                        del position_meta[symbol]
                        meta_changed = True

                    if meta and meta.broker_trailing_stop_order_id:
                        try:
                            trading_client.cancel_order_by_id(
                                meta.broker_trailing_stop_order_id
                            )
                        except Exception:
                            pass

                    print(f"  Position closed.", file=sys.stderr)

            except Exception as e:
                alert(f"Failed to close crypto {symbol}: {e}", AlertLevel.ERROR)

    if meta_changed:
        save_metadata(position_meta, str(CRYPTO_METADATA_PATH))

    return position_meta


def log_snapshot(positions: list[dict], equity: float):
    """Print a compact snapshot to stderr."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    if not positions:
        print(f"[{now}] No crypto positions | Equity: ${equity:,.2f}", file=sys.stderr)
        return

    parts = []
    total_pnl = 0.0
    for pos in positions:
        pnl = pos["unrealized_pl"]
        pnl_pct = pos["unrealized_plpc"]
        total_pnl += pnl
        # Shorten symbol for display
        short_sym = pos["symbol"].replace("/USD", "")
        parts.append(f"{short_sym}:{pnl_pct:+.1%}")

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
    """Check portfolio-level circuit breakers for crypto."""
    if starting_equity <= 0 or equity <= 0:
        return False

    daily_pnl = equity - starting_equity
    if daily_pnl < -MAX_DAILY_LOSS:
        alert(
            f"CRYPTO KILL SWITCH: Daily loss ${daily_pnl:+.2f} exceeds "
            f"max ${MAX_DAILY_LOSS}. CLOSING ALL CRYPTO POSITIONS.",
            AlertLevel.CRITICAL,
        )
        return True

    drawdown_pct = (equity - starting_equity) / starting_equity
    if drawdown_pct < -DRAWDOWN_KILL_SWITCH:
        alert(
            f"CRYPTO KILL SWITCH: Drawdown {drawdown_pct:.1%} exceeds "
            f"max {DRAWDOWN_KILL_SWITCH:.0%}.",
            AlertLevel.CRITICAL,
        )
        return True

    return False


def close_all_crypto_positions(trading_client: TradingClient):
    """Emergency close all crypto positions."""
    try:
        # Get crypto positions and close individually
        positions = trading_client.get_all_positions()
        for pos in positions:
            symbol = pos.symbol
            is_crypto = "/" in symbol or (
                symbol.endswith("USD") and not symbol.startswith("$")
            )
            if is_crypto:
                try:
                    trading_client.close_position(symbol)
                    print(f"  Closed crypto: {symbol}", file=sys.stderr)
                except Exception as e:
                    print(f"  Error closing {symbol}: {e}", file=sys.stderr)
        alert("All crypto positions closed.", AlertLevel.CRITICAL)
    except Exception as e:
        alert(f"FAILED to close crypto positions: {e}", AlertLevel.CRITICAL)


def run_loop(trading_client: TradingClient, interval: int, paper: bool):
    """Main crypto monitoring loop — runs 24/7."""
    position_meta: dict[str, PositionMeta] = {}
    consecutive_errors = 0
    last_snapshot_time = 0
    last_drawdown_check = 0
    last_equity_reset = 0
    kill_switch_triggered = False
    starting_equity: float = 0.0

    # Initialize trade logger
    trade_logger = TradeLogger()
    print(
        f"[CryptoMonitor] Trade logger initialized: {trade_logger.db_path}",
        file=sys.stderr,
    )

    # Initialize drawdown circuit breaker
    drawdown_breaker = DrawdownCircuitBreaker()
    peak_equity = get_peak_equity_from_snapshots(trade_logger)
    print(
        f"[CryptoMonitor] Drawdown breaker initialized (peak: ${peak_equity:,.2f})",
        file=sys.stderr,
    )

    # Load crypto position metadata
    try:
        position_meta = load_metadata(str(CRYPTO_METADATA_PATH))
        hwm_count = sum(
            1 for m in position_meta.values() if m.high_water_mark is not None
        )
        print(
            f"[CryptoMonitor] Loaded metadata for {len(position_meta)} crypto positions "
            f"({hwm_count} HWMs restored)",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[CryptoMonitor] Could not load position metadata: {e}",
            file=sys.stderr,
        )

    # Capture starting equity
    try:
        account = trading_client.get_account()
        starting_equity = float(account.equity)
    except Exception:
        pass

    print(
        f"[CryptoMonitor] Starting crypto price monitor (interval={interval}s, "
        f"mode={'PAPER' if paper else 'LIVE'})",
        file=sys.stderr,
    )
    print(
        f"[CryptoMonitor] 24/7 operation — no market hours check",
        file=sys.stderr,
    )
    print(
        f"[CryptoMonitor] Hard stop: {HARD_STOP_PCT:.0%}, "
        f"Trailing: {TRAILING_STOP_PCT:.0%}, "
        f"Kill switch: ${MAX_DAILY_LOSS:,.0f} / {DRAWDOWN_KILL_SWITCH:.0%}",
        file=sys.stderr,
    )

    while True:
        try:
            now = time.time()

            # Reset daily starting equity every 24 hours
            if now - last_equity_reset > 86400:
                try:
                    account = trading_client.get_account()
                    starting_equity = float(account.equity)
                    kill_switch_triggered = False
                    last_equity_reset = now
                    print(
                        f"[CryptoMonitor] Daily reset: equity=${starting_equity:,.2f}",
                        file=sys.stderr,
                    )
                except Exception:
                    pass

            # If kill switch was triggered, just wait
            if kill_switch_triggered:
                time.sleep(interval)
                continue

            # Get crypto positions and equity
            positions = get_crypto_positions(trading_client)
            try:
                account = trading_client.get_account()
                equity = float(account.equity)
            except Exception:
                equity = 0.0

            # Kill switch check
            if starting_equity > 0 and check_kill_switch(
                trading_client, equity, starting_equity
            ):
                close_all_crypto_positions(trading_client)
                kill_switch_triggered = True
                continue

            # Graduated drawdown check every 2 minutes
            if now - last_drawdown_check > 120:
                last_drawdown_check = now
                peak_equity = max(peak_equity, equity)
                dd_state = drawdown_breaker.check(equity, peak_equity)

                if dd_state.level >= DrawdownLevel.CAUTION:
                    print(
                        f"[CryptoMonitor] Drawdown: {dd_state.description}",
                        file=sys.stderr,
                    )

                if dd_state.unwind_weakest and positions:
                    weakest = get_weakest_positions(positions, n=2)
                    for weak_pos in weakest:
                        sym = weak_pos["symbol"]
                        alert(
                            f"CRYPTO DRAWDOWN UNWIND: Closing {sym} "
                            f"(P&L: {weak_pos.get('unrealized_plpc', 0):+.1%})",
                            AlertLevel.CRITICAL,
                        )
                        try:
                            try:
                                trading_client.close_position(sym)
                            except Exception:
                                trading_client.close_position(sym.replace("/", ""))
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
                                f"Failed to unwind crypto {sym}: {e}",
                                AlertLevel.ERROR,
                            )
                    save_metadata(position_meta, str(CRYPTO_METADATA_PATH))

            # Check for exits
            if positions:
                position_meta = check_exits(
                    positions, position_meta, trading_client, paper,
                    trade_logger=trade_logger,
                )

            # Log portfolio snapshot every 5 minutes
            if now - last_snapshot_time > 300:
                try:
                    num_long = sum(
                        1 for p in positions if p.get("side") != "short"
                    )
                    num_short = sum(
                        1 for p in positions if p.get("side") == "short"
                    )
                    total_unrealized = sum(
                        p.get("unrealized_pl", 0) for p in positions
                    )
                    trade_logger.log_snapshot(
                        equity=equity,
                        cash=0.0,
                        num_positions=len(positions),
                        num_long=num_long,
                        num_short=num_short,
                        total_unrealized_pnl=total_unrealized,
                    )
                    last_snapshot_time = now
                except Exception:
                    pass

            # Clean up orphaned metadata
            held_symbols = {p["symbol"] for p in positions}
            orphans = [s for s in position_meta if s not in held_symbols]
            if orphans:
                for s in orphans:
                    del position_meta[s]
                save_metadata(position_meta, str(CRYPTO_METADATA_PATH))

            # Log snapshot
            log_snapshot(positions, equity)

            consecutive_errors = 0
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[CryptoMonitor] Stopped by user.", file=sys.stderr)
            break
        except Exception as e:
            consecutive_errors += 1
            print(
                f"[CryptoMonitor] Error (#{consecutive_errors}): {e}",
                file=sys.stderr,
            )
            if consecutive_errors > 10:
                alert(
                    "CryptoMonitor: 10+ consecutive errors. Sleeping 5 min.",
                    AlertLevel.ERROR,
                )
                time.sleep(300)
                consecutive_errors = 0
            else:
                time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Crypto price monitor (24/7)")
    parser.add_argument(
        "--paper", action="store_true", default=True,
        help="Use paper trading (default: true)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live trading",
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Check interval in seconds (default: 60)",
    )
    args = parser.parse_args()

    paper = not args.live
    trading_client = get_clients(paper)

    run_loop(trading_client, args.interval, paper)


if __name__ == "__main__":
    main()
