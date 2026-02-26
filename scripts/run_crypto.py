#!/usr/bin/env python3
"""Crypto Execution Script — runs on a schedule (every 4-6 hours).

This is the crypto equivalent of run_daily.py. Key differences:
    - No market hours check (crypto trades 24/7)
    - Uses Alpaca CryptoHistoricalDataClient for data
    - Uses BTC for regime detection (instead of SPY)
    - Runs crypto-adapted strategies with tuned parameters
    - Separate position metadata namespace (crypto_position_metadata.json)
    - Wider trailing stops for crypto volatility

Recommended schedule: every 4 hours via cron/systemd
    0 */4 * * * /path/to/run_crypto.py

Usage:
    python scripts/run_crypto.py [--paper] [--dry-run] [--symbols BTC/USD,ETH/USD,...]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add strategy-engine to path
sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    TrailingStopOrderRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from alerting import alert, AlertLevel
from data.store import DataStore
from crypto.engine import CryptoStrategyEngine
from crypto.universe import CryptoUniverseSelector
from crypto.profit_targets import get_crypto_profit_config, ALT_COIN_TRAIL_MULT
from crypto.universe import TIER3_MID_ALT
from portfolio.drawdown import (
    DrawdownCircuitBreaker, DrawdownLevel,
    get_peak_equity_from_snapshots, get_weakest_positions,
)
from portfolio.position_metadata import PositionMeta, load_metadata, save_metadata
from portfolio.trade_logger import TradeLogger, TradeEntry
from strategies.base import SignalDirection
from portfolio.sizing import PortfolioContext


# Separate metadata file for crypto positions
CRYPTO_METADATA_PATH = Path(__file__).parent.parent / "strategy-engine" / "crypto_position_metadata.json"


def get_clients(paper: bool) -> tuple[CryptoHistoricalDataClient, TradingClient]:
    """Initialize Alpaca clients for crypto trading."""
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print(
            "ERROR: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set",
            file=sys.stderr,
        )
        sys.exit(1)
    # CryptoHistoricalDataClient doesn't need auth keys for free tier
    data_client = CryptoHistoricalDataClient(key, secret)
    trading_client = TradingClient(key, secret, paper=paper)
    return data_client, trading_client


def fetch_crypto_data(
    data_client: CryptoHistoricalDataClient,
    symbols: list[str],
    days: int = 200,
) -> dict[str, pd.DataFrame]:
    """Fetch historical daily bars for crypto symbols.

    Uses CryptoBarsRequest (different from StockBarsRequest).
    Crypto data is available 24/7 with no feed restrictions.
    """
    print(f"Fetching crypto data for {len(symbols)} symbols...", file=sys.stderr)
    start = datetime.now() - timedelta(days=days)

    data: dict[str, pd.DataFrame] = {}
    try:
        params = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
        )
        bars = data_client.get_crypto_bars(params)

        for symbol in symbols:
            if symbol in bars.data and bars[symbol]:
                records = []
                for bar in bars[symbol]:
                    records.append({
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                    })
                df = pd.DataFrame(
                    records,
                    index=[bar.timestamp for bar in bars[symbol]],
                )
                df.index.name = "timestamp"
                data[symbol] = df
    except Exception as e:
        print(f"Error fetching batch crypto data: {e}", file=sys.stderr)
        # Fall back to individual requests
        for symbol in symbols:
            try:
                params = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start,
                )
                bars = data_client.get_crypto_bars(params)
                if symbol in bars.data and bars[symbol]:
                    records = []
                    for bar in bars[symbol]:
                        records.append({
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume),
                        })
                    df = pd.DataFrame(
                        records,
                        index=[b.timestamp for b in bars[symbol]],
                    )
                    df.index.name = "timestamp"
                    data[symbol] = df
            except Exception as e2:
                print(f"Error fetching {symbol}: {e2}", file=sys.stderr)

    print(f"Got crypto data for {len(data)} symbols", file=sys.stderr)
    return data


def get_current_positions(trading_client: TradingClient) -> dict[str, dict]:
    """Get current open crypto positions.

    Filters to only include crypto symbols (contain '/').
    """
    positions = trading_client.get_all_positions()
    result = {}
    for pos in positions:
        symbol = pos.symbol
        # Alpaca may return crypto symbols with or without /
        # Normalize to include /
        if "/" not in symbol and "USD" in symbol:
            symbol = symbol.replace("USD", "/USD")

        result[symbol] = {
            "qty": float(pos.qty),
            "market_value": float(pos.market_value),
            "side": pos.side,
            "avg_entry_price": float(pos.avg_entry_price),
            "unrealized_pl": float(pos.unrealized_pl),
        }
    return result


def get_portfolio_context(trading_client: TradingClient) -> PortfolioContext:
    """Build portfolio context from account info."""
    account = trading_client.get_account()
    positions = trading_client.get_all_positions()
    return PortfolioContext(
        equity=float(account.equity),
        cash=float(account.cash),
        buying_power=float(account.buying_power),
        num_positions=len(positions),
        strategy_allocation_pct=1.0,
    )


def load_crypto_metadata() -> dict[str, PositionMeta]:
    """Load crypto-specific position metadata."""
    try:
        return load_metadata(str(CRYPTO_METADATA_PATH))
    except Exception:
        return {}


def save_crypto_metadata(meta: dict[str, PositionMeta]):
    """Save crypto-specific position metadata."""
    save_metadata(meta, str(CRYPTO_METADATA_PATH))


def main():
    parser = argparse.ArgumentParser(description="Crypto trading execution")
    parser.add_argument(
        "--paper", action="store_true", default=True,
        help="Use paper trading (default: true)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live trading",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate signals but don't execute orders",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated crypto symbols (overrides universe)",
    )
    args = parser.parse_args()

    paper = not args.live
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  CRYPTO TRADING RUN - {datetime.now().isoformat()}", file=sys.stderr)
    print(f"  Mode: {'PAPER' if paper else 'LIVE'}", file=sys.stderr)
    print(f"  Dry run: {args.dry_run}", file=sys.stderr)
    print(f"  Note: Crypto trades 24/7 - no market hours check", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    data_client, trading_client = get_clients(paper)

    # No market hours check for crypto - it trades 24/7!

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        universe = CryptoUniverseSelector()
        symbols = universe.get_symbols()

    # Ensure BTC/USD is included for regime detection
    if "BTC/USD" not in symbols:
        symbols.append("BTC/USD")

    # Fetch crypto data
    all_data = fetch_crypto_data(data_client, symbols)

    if "BTC/USD" not in all_data:
        alert("Crypto run ABORTED: could not fetch BTC/USD data", AlertLevel.CRITICAL)
        sys.exit(1)

    btc_data = all_data["BTC/USD"]
    # Don't pop BTC — strategies need it too

    # Compute crypto volatility proxy (annualized BTC realized vol)
    vol_proxy = None
    try:
        btc_returns = btc_data["close"].pct_change().dropna()
        if len(btc_returns) >= 20:
            realized_vol = btc_returns.tail(20).std() * (365 ** 0.5) * 100
            vol_proxy = float(realized_vol)
    except Exception as e:
        print(f"[Vol] Could not compute crypto volatility proxy: {e}", file=sys.stderr)

    # Get portfolio context
    portfolio = get_portfolio_context(trading_client)
    portfolio.vix_level = vol_proxy  # crypto vol proxy in VIX field
    current_positions = get_current_positions(trading_client)

    vol_str = f", CryptoVol={vol_proxy:.1f}%" if vol_proxy else ""
    print(
        f"\nPortfolio: equity=${portfolio.equity:.2f}, "
        f"cash=${portfolio.cash:.2f}, "
        f"positions={portfolio.num_positions}{vol_str}",
        file=sys.stderr,
    )

    # Initialize loggers
    trade_logger = TradeLogger()

    # Check drawdown circuit breaker
    drawdown_breaker = DrawdownCircuitBreaker()
    peak_equity = get_peak_equity_from_snapshots(trade_logger)
    if peak_equity <= 0:
        peak_equity = portfolio.equity
    drawdown_state = drawdown_breaker.check(portfolio.equity, peak_equity)
    print(f"  Drawdown: {drawdown_state.description}", file=sys.stderr)

    if drawdown_state.level >= DrawdownLevel.HALT:
        alert(
            f"CRYPTO DRAWDOWN CIRCUIT BREAKER: {drawdown_state.description}",
            AlertLevel.CRITICAL,
        )

    # Run crypto engine
    engine = CryptoStrategyEngine()
    orders, regime, allocation = engine.run(
        data=all_data,
        btc_data=btc_data,
        portfolio=portfolio,
        current_positions=current_positions,
        drawdown_state=drawdown_state,
    )

    # Print results
    print(f"\n--- CRYPTO RESULTS ---", file=sys.stderr)
    print(
        f"Regime: {regime.regime.value} (conf: {regime.confidence:.2f})",
        file=sys.stderr,
    )
    print(
        f"Allocation: TF={allocation.trend_following:.0%} "
        f"MR={allocation.mean_reversion:.0%} "
        f"MOM={allocation.momentum:.0%} "
        f"BTC_DOM={allocation.btc_dominance:.0%} "
        f"Cash={allocation.cash:.0%}",
        file=sys.stderr,
    )
    print(f"Orders to execute: {len(orders)}", file=sys.stderr)

    for order in orders:
        print(
            f"  {order.side.upper()} {order.qty} {order.symbol} "
            f"@ {order.order_type} "
            f"(signal: {order.signal.strength:.2f}, "
            f"value: ${order.sizing.dollar_value:.2f})",
            file=sys.stderr,
        )

    if args.dry_run:
        print("\n[DRY RUN] No orders submitted.", file=sys.stderr)
        output = {
            "market": "crypto",
            "regime": regime.regime.value,
            "regime_confidence": regime.confidence,
            "regime_indicators": regime.indicators,
            "allocation": {
                "trend_following": allocation.trend_following,
                "mean_reversion": allocation.mean_reversion,
                "momentum": allocation.momentum,
                "btc_dominance": allocation.btc_dominance,
                "cash": allocation.cash,
            },
            "orders": [
                {
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": o.qty,
                    "type": o.order_type,
                    "signal_strength": o.signal.strength,
                    "dollar_value": o.sizing.dollar_value,
                    "rationale": o.rationale,
                }
                for o in orders
            ],
        }
        print(json.dumps(output, indent=2, default=str))
        trade_logger.close()
        return

    # Dedup check
    already_entered_today = set()
    try:
        already_entered_today = trade_logger.get_todays_entries()
    except Exception as e:
        print(f"[Dedup] Warning: dedup check failed: {e}", file=sys.stderr)

    if already_entered_today:
        print(f"[Dedup] Already entered today: {already_entered_today}", file=sys.stderr)

    # Execute orders
    print(f"\nExecuting {len(orders)} crypto orders...", file=sys.stderr)
    pending_stops: list[tuple] = []
    filled = []
    failed = []
    skipped = []

    for order in orders:
        if order.symbol in already_entered_today:
            print(
                f"  SKIP {order.symbol}: already entered today (dedup)",
                file=sys.stderr,
            )
            skipped.append(order.symbol)
            continue

        try:
            is_entry = order.signal.direction in (
                SignalDirection.LONG, SignalDirection.SHORT
            )

            # Crypto uses market orders (spread is tighter on major pairs,
            # and limit orders on less liquid alts may not fill)
            side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL

            # Alpaca crypto orders use GTC (good till cancelled) for 24/7 markets
            order_req = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.GTC,
            )
            result = trading_client.submit_order(order_data=order_req)
            order_id = str(result.id)

            print(
                f"  {order.side} {order.qty} {order.symbol}: "
                f"order_id={result.id}, status={result.status}",
                file=sys.stderr,
            )

            filled.append(f"{order.side} {order.qty} {order.symbol}")

            # Log entry to TradeLogger
            if is_entry:
                try:
                    direction = (
                        "long" if order.signal.direction == SignalDirection.LONG
                        else "short"
                    )
                    tier = order.sizing.details.get("conviction_tier", 4)
                    trade_logger.log_entry(TradeEntry(
                        symbol=order.symbol,
                        direction=direction,
                        strategy=order.signal.strategy_name,
                        conviction_tier=tier,
                        confluence_count=order.signal.features.get(
                            "confluence_count", 1
                        ),
                        entry_price=order.signal.entry_price,
                        entry_qty=order.qty,
                        entry_order_id=order_id,
                        entry_regime=regime.regime.value,
                        entry_vix=vol_proxy,
                        entry_signal_strength=order.signal.strength,
                        features={
                            k: v
                            for k, v in order.signal.features.items()
                            if isinstance(v, (int, float, str, bool, type(None)))
                        },
                        entry_order_type="market",
                    ))
                except Exception as e:
                    print(
                        f"  Warning: failed to log entry: {e}",
                        file=sys.stderr,
                    )

            # Queue protective trailing stop for entries
            is_long_entry = (
                order.side == "buy"
                and order.signal.direction == SignalDirection.LONG
            )
            is_short_entry = (
                order.side == "sell"
                and order.signal.direction == SignalDirection.SHORT
            )
            if (
                (is_long_entry or is_short_entry)
                and order.stop_loss
                and order.stop_loss > 0
            ):
                pos_side = "short" if is_short_entry else "long"
                pending_stops.append(
                    (order.symbol, order.qty, order.stop_loss, order, pos_side)
                )

        except Exception as e:
            alert(
                f"Crypto order FAILED: {order.side} {order.qty} "
                f"{order.symbol}: {e}",
                AlertLevel.ERROR,
            )
            failed.append(f"{order.side} {order.qty} {order.symbol}")

    # Place protective trailing stops
    if pending_stops:
        print(
            f"\nWaiting 3s for crypto fills before placing "
            f"{len(pending_stops)} protective orders...",
            file=sys.stderr,
        )
        time.sleep(3)

        tier3_set = set(TIER3_MID_ALT)
        metadata = load_crypto_metadata()

        for symbol, qty, stop_price, order_obj, position_side in pending_stops:
            try:
                sig = order_obj.signal
                sizing = order_obj.sizing
                strategy_name = sig.strategy_name
                tier = sizing.details.get("conviction_tier", 4)
                atr = sig.features.get("atr", 0.0)
                is_tier3 = symbol in tier3_set

                # Crypto profit config (wider than equities)
                pc = get_crypto_profit_config(strategy_name, tier)

                # Get actual entry price
                entry_price = sig.entry_price or stop_price / 0.85
                try:
                    pos_info = trading_client.get_open_position(
                        symbol.replace("/", "")  # Alpaca API may need BTCUSD format
                    )
                    entry_price = float(pos_info.avg_entry_price)
                except Exception:
                    try:
                        pos_info = trading_client.get_open_position(symbol)
                        entry_price = float(pos_info.avg_entry_price)
                    except Exception:
                        pass

                # Compute trailing stop %
                trail_atr_mult = pc.trail_stop_atr_mult
                if is_tier3:
                    trail_atr_mult *= ALT_COIN_TRAIL_MULT

                if atr > 0 and entry_price > 0:
                    trail_pct = min(
                        20.0,  # wider cap for crypto (20% vs 12% for equities)
                        max(5.0, (trail_atr_mult * atr / entry_price) * 100),
                    )
                else:
                    trail_pct = 12.0  # crypto default (wider than equity 8%)

                trail_side = (
                    OrderSide.BUY if position_side == "short" else OrderSide.SELL
                )

                trailing_order = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=trail_side,
                    time_in_force=TimeInForce.GTC,
                    trail_percent=round(trail_pct, 1),
                )
                result = trading_client.submit_order(order_data=trailing_order)
                trailing_order_id = str(result.id)

                side_label = "buy" if position_side == "short" else "sell"
                print(
                    f"  TRAILING STOP {symbol}: {side_label} {qty} "
                    f"trail={trail_pct:.1f}% "
                    f"({trail_atr_mult:.1f}x ATR, order_id={result.id})",
                    file=sys.stderr,
                )

                # Save position metadata
                bb_middle = (
                    sig.features.get("bb_middle")
                    if pc.exit_at_bb_middle
                    else None
                )
                metadata[symbol] = PositionMeta(
                    entry_price=entry_price,
                    entry_date=datetime.now().strftime("%Y-%m-%d"),
                    strategy=strategy_name,
                    conviction_tier=tier,
                    atr_at_entry=atr,
                    initial_qty=qty,
                    remaining_qty=qty,
                    first_target_hit=False,
                    first_target_pct=pc.first_target_pct,
                    partial_sell_pct=pc.partial_sell_pct,
                    trail_stop_atr_mult=trail_atr_mult,
                    time_exit_days=pc.time_exit_days,
                    exit_at_bb_middle=pc.exit_at_bb_middle,
                    bb_middle_at_entry=bb_middle,
                    is_smid_cap=is_tier3,
                    broker_trailing_stop_order_id=trailing_order_id,
                    position_side=position_side,
                    second_target_pct=pc.second_target_pct,
                    second_sell_pct=pc.second_sell_pct,
                    second_target_hit=False,
                )

            except Exception as e:
                alert(
                    f"Crypto protective order FAILED for {symbol}: {e}",
                    AlertLevel.ERROR,
                )

        save_crypto_metadata(metadata)

    # Log portfolio snapshot
    try:
        positions_after = get_current_positions(trading_client)
        num_long = sum(
            1 for p in positions_after.values()
            if str(p.get("side", "long")).replace("PositionSide.", "") == "long"
        )
        num_short = sum(
            1 for p in positions_after.values()
            if str(p.get("side", "long")).replace("PositionSide.", "") == "short"
        )
        total_unrealized = sum(
            p.get("unrealized_pl", 0) for p in positions_after.values()
        )
        trade_logger.log_snapshot(
            equity=portfolio.equity,
            cash=portfolio.cash,
            num_positions=len(positions_after),
            num_long=num_long,
            num_short=num_short,
            total_unrealized_pnl=total_unrealized,
            regime=regime.regime.value,
            vix_level=vol_proxy,
        )
    except Exception as e:
        print(
            f"[Snapshot] Warning: failed to log portfolio snapshot: {e}",
            file=sys.stderr,
        )

    # Summary alert
    summary = (
        f"Crypto run complete: {regime.regime.value} regime | "
        f"{len(filled)} orders submitted"
    )
    if skipped:
        summary += f" | {len(skipped)} skipped (dedup): {', '.join(skipped)}"
    if failed:
        summary += f" | {len(failed)} FAILED: {', '.join(failed)}"
    alert(summary, AlertLevel.ERROR if failed else AlertLevel.INFO)

    trade_logger.close()
    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
