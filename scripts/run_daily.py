#!/usr/bin/env python3
"""Daily Execution Script.

This script is the entry point for daily trading operations.
It can be run via cron, manually, or triggered by Claude.

Flow:
    1. Fetch latest market data for the universe
    2. Detect market regime
    3. Run all active strategies → generate signals
    4. Apply position sizing and portfolio optimization
    5. Compare desired positions to current positions → generate orders
    6. Submit orders via Alpaca API (direct, not through MCP)
    7. Log everything to Postgres
    8. Update equity curve

Usage:
    python scripts/run_daily.py [--paper] [--dry-run] [--symbols AAPL,MSFT,...]
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
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
    TrailingStopOrderRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from alerting import alert, AlertLevel
from data.store import DataStore
from engine import StrategyEngine
from portfolio.position_metadata import PositionMeta, load_metadata, save_metadata
from portfolio.profit_targets import get_profit_config, SMID_CAP_TRAIL_MULT
from portfolio.sizing import PortfolioContext
from portfolio.universe import UniverseSelector, SMID_CAP


def get_clients(paper: bool) -> tuple[StockHistoricalDataClient, TradingClient]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print("ERROR: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set", file=sys.stderr)
        sys.exit(1)
    data_client = StockHistoricalDataClient(key, secret)
    trading_client = TradingClient(key, secret, paper=paper)
    return data_client, trading_client


def fetch_data(
    data_client: StockHistoricalDataClient,
    symbols: list[str],
    days: int = 300,
) -> dict[str, pd.DataFrame]:
    """Fetch historical daily bars for all symbols."""
    print(f"Fetching data for {len(symbols)} symbols...", file=sys.stderr)
    start = datetime.now() - timedelta(days=days)

    data: dict[str, pd.DataFrame] = {}
    # Batch request
    try:
        params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(params)

        for symbol in symbols:
            if symbol in bars.data and bars[symbol]:
                records = []
                for bar in bars[symbol]:
                    records.append({
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    })
                df = pd.DataFrame(records, index=[bar.timestamp for bar in bars[symbol]])
                df.index.name = "timestamp"
                data[symbol] = df
    except Exception as e:
        print(f"Error fetching batch data: {e}", file=sys.stderr)
        # Fall back to individual requests
        for symbol in symbols:
            try:
                params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start,
                    feed=DataFeed.IEX,
                )
                bars = data_client.get_stock_bars(params)
                if symbol in bars.data and bars[symbol]:
                    records = []
                    for bar in bars[symbol]:
                        records.append({
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": int(bar.volume),
                        })
                    df = pd.DataFrame(records, index=[b.timestamp for b in bars[symbol]])
                    df.index.name = "timestamp"
                    data[symbol] = df
            except Exception as e2:
                print(f"Error fetching {symbol}: {e2}", file=sys.stderr)

    print(f"Got data for {len(data)} symbols", file=sys.stderr)
    return data


def get_current_positions(trading_client: TradingClient) -> dict[str, dict]:
    """Get current open positions."""
    positions = trading_client.get_all_positions()
    result = {}
    for pos in positions:
        result[pos.symbol] = {
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
        strategy_allocation_pct=1.0,  # Will be overridden by engine
    )


def main():
    parser = argparse.ArgumentParser(description="Daily trading execution")
    parser.add_argument("--paper", action="store_true", default=True,
                       help="Use paper trading (default: true)")
    parser.add_argument("--live", action="store_true",
                       help="Use live trading")
    parser.add_argument("--dry-run", action="store_true",
                       help="Generate signals but don't execute orders")
    parser.add_argument("--symbols", type=str, default=None,
                       help="Comma-separated symbols (overrides universe)")
    args = parser.parse_args()

    paper = not args.live
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  DAILY TRADING RUN — {datetime.now().isoformat()}", file=sys.stderr)
    print(f"  Mode: {'PAPER' if paper else '🔴 LIVE'}", file=sys.stderr)
    print(f"  Dry run: {args.dry_run}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    data_client, trading_client = get_clients(paper)

    # Check if market is open (handles holidays, early closes, etc.)
    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market is closed (holiday or outside hours). Exiting.", file=sys.stderr)
        print(json.dumps({"status": "market_closed"}))
        return

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        universe = UniverseSelector()
        symbols = universe.get_symbols()

    # Include SPY for regime detection
    if "SPY" not in symbols:
        symbols.append("SPY")

    # Fetch data
    all_data = fetch_data(data_client, symbols)

    if "SPY" not in all_data:
        alert("Daily run ABORTED: could not fetch SPY data", AlertLevel.CRITICAL)
        sys.exit(1)

    spy_data = all_data.pop("SPY")

    # Get portfolio context
    portfolio = get_portfolio_context(trading_client)
    current_positions = get_current_positions(trading_client)

    print(f"\nPortfolio: equity=${portfolio.equity:.2f}, "
          f"cash=${portfolio.cash:.2f}, "
          f"positions={portfolio.num_positions}", file=sys.stderr)

    # Run engine
    engine = StrategyEngine()
    orders, regime, allocation = engine.run(
        data=all_data,
        spy_data=spy_data,
        portfolio=portfolio,
        current_positions=current_positions,
    )

    # Initialize trade logger
    store = DataStore()

    # Log regime and equity snapshot
    now = datetime.now()
    try:
        store.log_regime(
            timestamp=now,
            regime_type=regime.regime.value,
            confidence=regime.confidence,
            indicators=regime.indicators,
        )
        store.log_equity(
            timestamp=now,
            equity=portfolio.equity,
            cash=portfolio.cash,
            long_market_value=portfolio.equity - portfolio.cash,
            num_positions=portfolio.num_positions,
            peak_equity=portfolio.equity,
        )
    except Exception as e:
        print(f"[Engine] Warning: failed to log regime/equity: {e}", file=sys.stderr)

    # Print results
    print(f"\n--- RESULTS ---", file=sys.stderr)
    print(f"Regime: {regime.regime.value} (conf: {regime.confidence:.2f})", file=sys.stderr)
    print(f"Allocation: TF={allocation.trend_following:.0%} "
          f"MR={allocation.mean_reversion:.0%} "
          f"MOM={allocation.momentum:.0%} "
          f"SENT={allocation.sentiment:.0%} "
          f"EARN={allocation.earnings_momentum:.0%} "
          f"Cash={allocation.cash:.0%}", file=sys.stderr)
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
        # Output as JSON to stdout for piping
        output = {
            "regime": regime.regime.value,
            "regime_confidence": regime.confidence,
            "regime_indicators": regime.indicators,
            "allocation": {
                "trend_following": allocation.trend_following,
                "mean_reversion": allocation.mean_reversion,
                "momentum": allocation.momentum,
                "sentiment": allocation.sentiment,
                "earnings_momentum": allocation.earnings_momentum,
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
        return

    # Dedup check: skip symbols we already ordered today (crash-restart protection)
    already_ordered = set()
    try:
        already_ordered = store.get_todays_order_symbols(side="buy")
        if already_ordered:
            print(
                f"[Dedup] Already ordered today: {already_ordered}",
                file=sys.stderr,
            )
    except Exception as e:
        print(f"[Dedup] Warning: could not check existing orders: {e}", file=sys.stderr)

    # Execute orders
    print(f"\nExecuting {len(orders)} orders...", file=sys.stderr)
    pending_stops: list[tuple] = []  # (symbol, qty, stop_price) to submit after fills
    filled = []
    failed = []
    skipped = []

    for order in orders:
        # Skip if we already submitted a buy for this symbol today
        if order.side == "buy" and order.symbol in already_ordered:
            print(
                f"  SKIP {order.symbol}: already ordered today (dedup)",
                file=sys.stderr,
            )
            skipped.append(order.symbol)
            continue

        try:
            side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
            if order.limit_price and order.order_type == "limit":
                order_req = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(order.limit_price, 2),
                )
            else:
                order_req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )

            result = trading_client.submit_order(order_data=order_req)
            print(
                f"  {order.side} {order.qty} {order.symbol}: "
                f"order_id={result.id}, status={result.status}",
                file=sys.stderr,
            )
            filled.append(f"{order.side} {order.qty} {order.symbol}")

            # Log trade to database
            try:
                strategy_id = store.get_or_create_strategy(
                    name=order.signal.strategy_name,
                    strategy_type=order.signal.strategy_name,
                    params={},
                )
                tier = order.sizing.details.get("conviction_tier", 4)
                store.log_trade(
                    order_id=str(result.id),
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    price=order.signal.entry_price,
                    filled_price=None,
                    order_type=order.order_type,
                    status=str(result.status),
                    strategy_id=strategy_id,
                    signal_id=None,
                    signal_strength=order.signal.strength,
                    regime=regime.regime.value,
                    features={
                        "conviction_tier": tier,
                        "confluence_count": order.signal.features.get("confluence_count", 1),
                        "dollar_value": order.sizing.dollar_value,
                        "stop_loss": order.stop_loss,
                        "take_profit": order.take_profit,
                        **{k: v for k, v in order.signal.features.items()
                           if isinstance(v, (int, float, str, bool, type(None)))},
                    },
                    rationale=order.rationale,
                    risk_check={
                        "sizing_method": order.sizing.method,
                        "portfolio_risk_pct": order.sizing.portfolio_risk_pct,
                    },
                    submitted_at=datetime.now(),
                )
            except Exception as e:
                print(f"  Warning: failed to log trade: {e}", file=sys.stderr)

            # Queue protective orders for buy orders
            if order.side == "buy" and order.stop_loss and order.stop_loss > 0:
                pending_stops.append((order.symbol, order.qty, order.stop_loss, order))

        except Exception as e:
            alert(
                f"Order FAILED: {order.side} {order.qty} {order.symbol}: {e}",
                AlertLevel.ERROR,
            )
            failed.append(f"{order.side} {order.qty} {order.symbol}")

    # Place protective orders and write position metadata for new buys
    if pending_stops:
        print(
            f"\nWaiting 5s for fills before placing "
            f"{len(pending_stops)} protective orders...",
            file=sys.stderr,
        )
        time.sleep(5)

        smid_set = set(SMID_CAP)
        metadata = load_metadata()

        for symbol, qty, stop_price, order_obj in pending_stops:
            try:
                # Cancel any existing stop/trailing_stop orders for this symbol
                existing_orders = trading_client.get_orders(
                    filter=GetOrdersRequest(
                        status=QueryOrderStatus.OPEN,
                        symbols=[symbol],
                    )
                )
                for existing in existing_orders:
                    if existing.type.value in ("stop", "trailing_stop"):
                        trading_client.cancel_order_by_id(existing.id)
                        print(
                            f"  Cancelled old {existing.type.value} for "
                            f"{symbol} (id={existing.id})",
                            file=sys.stderr,
                        )

                sig = order_obj.signal
                sizing = order_obj.sizing
                strategy_name = sig.strategy_name
                tier = sizing.details.get("conviction_tier", 4)
                atr = sig.features.get("atr", 0.0)
                is_smid = symbol in smid_set

                # Look up profit config for this (strategy, tier)
                pc = get_profit_config(strategy_name, tier)

                # Get actual entry price from filled position
                entry_price = sig.entry_price or stop_price / 0.9  # fallback
                try:
                    pos_info = trading_client.get_open_position(symbol)
                    entry_price = float(pos_info.avg_entry_price)
                except Exception:
                    pass

                # Compute trailing stop % for broker-side order
                trail_atr_mult = pc.trail_stop_atr_mult
                if is_smid:
                    trail_atr_mult *= SMID_CAP_TRAIL_MULT

                if atr > 0 and entry_price > 0:
                    trail_pct = min(12.0, max(3.0,
                        (trail_atr_mult * atr / entry_price) * 100
                    ))
                else:
                    trail_pct = 8.0  # safe default

                # Place broker-side trailing stop (outer safety envelope)
                trailing_order = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    trail_percent=round(trail_pct, 1),
                )
                result = trading_client.submit_order(order_data=trailing_order)
                trailing_order_id = str(result.id)
                print(
                    f"  TRAILING STOP {symbol}: sell {qty} "
                    f"trail={trail_pct:.1f}% "
                    f"({trail_atr_mult:.1f}x ATR, order_id={result.id})",
                    file=sys.stderr,
                )

                # Build and persist position metadata
                bb_middle = (
                    sig.features.get("bb_middle")
                    if pc.exit_at_bb_middle else None
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
                    is_smid_cap=is_smid,
                    broker_trailing_stop_order_id=trailing_order_id,
                )

            except Exception as e:
                alert(
                    f"Protective order FAILED for {symbol}: {e}",
                    AlertLevel.ERROR,
                )

        save_metadata(metadata)

    # Summary alert
    summary = (
        f"Daily run complete: {regime.regime.value} regime | "
        f"{len(filled)} orders submitted"
    )
    if skipped:
        summary += f" | {len(skipped)} skipped (dedup): {', '.join(skipped)}"
    if failed:
        summary += f" | {len(failed)} FAILED: {', '.join(failed)}"
    alert(summary, AlertLevel.ERROR if failed else AlertLevel.INFO)

    store.close()

    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
