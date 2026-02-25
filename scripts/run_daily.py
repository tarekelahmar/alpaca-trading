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
from portfolio.drawdown import (
    DrawdownCircuitBreaker, DrawdownLevel,
    get_peak_equity_from_snapshots, get_weakest_positions,
)
from portfolio.execution import SmartExecutor, OrderTiming
from portfolio.position_metadata import PositionMeta, load_metadata, save_metadata
from portfolio.trade_logger import TradeLogger, TradeEntry
from strategies.base import SignalDirection
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

    # Compute VIX proxy: 20-day annualized realized volatility of SPY
    vix_proxy = None
    try:
        spy_returns = spy_data["close"].pct_change().dropna()
        if len(spy_returns) >= 20:
            realized_vol = spy_returns.tail(20).std() * (252 ** 0.5) * 100
            vix_proxy = float(realized_vol)
    except Exception as e:
        print(f"[VIX] Could not compute VIX proxy: {e}", file=sys.stderr)

    # Get portfolio context
    portfolio = get_portfolio_context(trading_client)
    portfolio.vix_level = vix_proxy
    current_positions = get_current_positions(trading_client)

    vix_str = f", VIX≈{vix_proxy:.1f}" if vix_proxy else ""
    print(f"\nPortfolio: equity=${portfolio.equity:.2f}, "
          f"cash=${portfolio.cash:.2f}, "
          f"positions={portfolio.num_positions}{vix_str}", file=sys.stderr)

    # Initialize loggers
    store = DataStore()
    trade_logger = TradeLogger()

    # Check drawdown circuit breaker
    drawdown_breaker = DrawdownCircuitBreaker()
    peak_equity = get_peak_equity_from_snapshots(trade_logger)
    if peak_equity <= 0:
        peak_equity = portfolio.equity  # first run, no history
    drawdown_state = drawdown_breaker.check(portfolio.equity, peak_equity)
    print(
        f"  Drawdown: {drawdown_state.description}",
        file=sys.stderr,
    )

    if drawdown_state.level >= DrawdownLevel.HALT:
        alert(
            f"DRAWDOWN CIRCUIT BREAKER: {drawdown_state.description}",
            AlertLevel.CRITICAL,
        )

    # Run engine
    engine = StrategyEngine()
    orders, regime, allocation = engine.run(
        data=all_data,
        spy_data=spy_data,
        portfolio=portfolio,
        current_positions=current_positions,
        drawdown_state=drawdown_state,
    )

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
          f"GAP={allocation.gap_trading:.0%} "
          f"SEC={allocation.sector_rotation:.0%} "
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
    # Uses both DataStore (legacy) and TradeLogger (new lifecycle tracking)
    already_ordered_buy = set()
    already_ordered_sell = set()
    already_entered_today = set()
    try:
        already_ordered_buy = store.get_todays_order_symbols(side="buy")
        already_ordered_sell = store.get_todays_order_symbols(side="sell")
    except Exception as e:
        print(f"[Dedup] Warning: DataStore dedup check failed: {e}", file=sys.stderr)
    try:
        already_entered_today = trade_logger.get_todays_entries()
    except Exception as e:
        print(f"[Dedup] Warning: TradeLogger dedup check failed: {e}", file=sys.stderr)

    all_dedup = already_ordered_buy | already_ordered_sell | already_entered_today
    if all_dedup:
        print(
            f"[Dedup] Already ordered today: {all_dedup}",
            file=sys.stderr,
        )

    # Conflict handling: if we hold LONG and get SHORT (or vice versa),
    # exit the existing position but don't enter opposite direction same day.
    conflict_exits = set()
    for order in orders:
        if order.signal.direction in (SignalDirection.LONG, SignalDirection.SHORT):
            if order.symbol in current_positions:
                pos = current_positions[order.symbol]
                pos_side = pos.get("side", "long")
                if hasattr(pos_side, "value"):
                    pos_side = pos_side.value
                pos_side = str(pos_side)
                is_conflict = (
                    (order.signal.direction == SignalDirection.SHORT and pos_side == "long")
                    or (order.signal.direction == SignalDirection.LONG and pos_side == "short")
                )
                if is_conflict:
                    conflict_exits.add(order.symbol)

    if conflict_exits:
        print(
            f"\n[Conflict] Exiting positions before direction flip: "
            f"{conflict_exits}",
            file=sys.stderr,
        )
        for symbol in conflict_exits:
            try:
                trading_client.close_position(symbol)
                print(f"  Closed conflicting position: {symbol}", file=sys.stderr)
            except Exception as e:
                print(f"  Error closing {symbol}: {e}", file=sys.stderr)
        time.sleep(2)  # wait for fills

    # Remove entry orders for conflict symbols (enter on next day)
    orders = [
        o for o in orders
        if o.symbol not in conflict_exits
        or o.signal.direction == SignalDirection.CLOSE
    ]

    # Initialize smart executor for limit orders and timing
    executor = SmartExecutor(trading_client)

    # Group orders by timing window
    immediate_orders = []
    delayed_orders = []  # orders that need to wait for strategy timing window

    for order in orders:
        if order.signal.direction in (SignalDirection.LONG, SignalDirection.SHORT):
            timing = executor.get_order_timing(order.signal.strategy_name)
            if timing == OrderTiming.SKIP_TODAY:
                print(
                    f"  SKIP {order.symbol}: too late in day for "
                    f"{order.signal.strategy_name}",
                    file=sys.stderr,
                )
                continue
            elif timing == OrderTiming.WAIT_FOR_WINDOW:
                wait_min = executor.minutes_until_window(order.signal.strategy_name)
                print(
                    f"  QUEUE {order.symbol}: {order.signal.strategy_name} "
                    f"window opens in {wait_min:.0f}min",
                    file=sys.stderr,
                )
                delayed_orders.append((order, wait_min))
            else:
                immediate_orders.append(order)
        else:
            immediate_orders.append(order)  # exit orders are always immediate

    # Process delayed orders: wait for the longest delay then submit all
    if delayed_orders:
        max_wait = max(wait for _, wait in delayed_orders)
        if max_wait > 0 and max_wait <= 45:  # don't wait more than 45 min
            print(
                f"\n  Waiting {max_wait:.0f}min for strategy timing windows...",
                file=sys.stderr,
            )
            time.sleep(max_wait * 60)
        for order, _ in delayed_orders:
            immediate_orders.append(order)

    # Execute orders
    print(f"\nExecuting {len(immediate_orders)} orders...", file=sys.stderr)
    pending_stops: list[tuple] = []  # (symbol, qty, stop_price, order, position_side)
    filled = []
    failed = []
    skipped = []

    for order in immediate_orders:
        # Skip if we already submitted an order for this symbol/side today
        if order.symbol in already_entered_today:
            print(
                f"  SKIP {order.symbol}: already entered today (trade_logger dedup)",
                file=sys.stderr,
            )
            skipped.append(order.symbol)
            continue
        if order.side == "buy" and order.symbol in already_ordered_buy:
            print(
                f"  SKIP {order.symbol}: already bought today (dedup)",
                file=sys.stderr,
            )
            skipped.append(order.symbol)
            continue
        if order.side == "sell" and order.signal.direction == SignalDirection.SHORT:
            if order.symbol in already_ordered_sell:
                print(
                    f"  SKIP {order.symbol}: already shorted today (dedup)",
                    file=sys.stderr,
                )
                skipped.append(order.symbol)
                continue

        # Shortability check for SHORT entries
        if order.signal.direction == SignalDirection.SHORT:
            try:
                asset = trading_client.get_asset(order.symbol)
                if not (asset.shortable and asset.easy_to_borrow):
                    print(
                        f"  SKIP {order.symbol}: not shortable/easy to borrow",
                        file=sys.stderr,
                    )
                    skipped.append(order.symbol)
                    continue
            except Exception as e:
                print(
                    f"  SKIP {order.symbol}: shortability check failed: {e}",
                    file=sys.stderr,
                )
                skipped.append(order.symbol)
                continue

        try:
            is_entry = order.signal.direction in (
                SignalDirection.LONG, SignalDirection.SHORT
            )
            order_id = None
            order_type_used = "market"
            limit_price_used = None

            if is_entry and order.signal.entry_price and order.signal.entry_price > 0:
                # Use limit order for entries — save on spread
                order_id, limit_price_used = executor.submit_limit_order(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=order.side,
                    signal_price=order.signal.entry_price,
                    strategy_name=order.signal.strategy_name,
                )
                if order_id:
                    order_type_used = "limit"
                else:
                    # Limit failed, fall back to market
                    order_id = executor.submit_market_order(
                        symbol=order.symbol,
                        qty=order.qty,
                        side=order.side,
                        signal_price=order.signal.entry_price,
                        strategy_name=order.signal.strategy_name,
                    )
            else:
                # Exit orders or no entry price — use market
                side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
                order_req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
                result = trading_client.submit_order(order_data=order_req)
                order_id = str(result.id)
                print(
                    f"  {order.side} {order.qty} {order.symbol}: "
                    f"order_id={result.id}, status={result.status}",
                    file=sys.stderr,
                )

            if not order_id:
                raise RuntimeError(f"Order submission returned no order_id")

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
                    order_id=order_id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    price=order.signal.entry_price,
                    filled_price=None,
                    order_type=order_type_used,
                    status="submitted",
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
                        "order_type": order_type_used,
                        "limit_price": limit_price_used,
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

            # Log entry to TradeLogger for lifecycle tracking
            if is_entry:
                try:
                    direction = "long" if order.signal.direction == SignalDirection.LONG else "short"
                    tier = order.sizing.details.get("conviction_tier", 4)
                    trade_logger.log_entry(TradeEntry(
                        symbol=order.symbol,
                        direction=direction,
                        strategy=order.signal.strategy_name,
                        conviction_tier=tier,
                        confluence_count=order.signal.features.get("confluence_count", 1),
                        entry_price=order.signal.entry_price,
                        entry_qty=order.qty,
                        entry_order_id=order_id,
                        entry_regime=regime.regime.value,
                        entry_vix=vix_proxy,
                        entry_signal_strength=order.signal.strength,
                        features={
                            k: v for k, v in order.signal.features.items()
                            if isinstance(v, (int, float, str, bool, type(None)))
                        },
                        entry_order_type=order_type_used,
                    ))
                except Exception as e:
                    print(f"  Warning: failed to log entry to TradeLogger: {e}", file=sys.stderr)

            # Queue protective orders for new positions (both longs and shorts)
            is_long_entry = (
                order.side == "buy"
                and order.signal.direction == SignalDirection.LONG
            )
            is_short_entry = (
                order.side == "sell"
                and order.signal.direction == SignalDirection.SHORT
            )
            if (is_long_entry or is_short_entry) and order.stop_loss and order.stop_loss > 0:
                pos_side = "short" if is_short_entry else "long"
                pending_stops.append(
                    (order.symbol, order.qty, order.stop_loss, order, pos_side)
                )

        except Exception as e:
            alert(
                f"Order FAILED: {order.side} {order.qty} {order.symbol}: {e}",
                AlertLevel.ERROR,
            )
            failed.append(f"{order.side} {order.qty} {order.symbol}")

    # Check pending limit orders and convert timed-out ones to market
    if executor.pending_limits:
        print(
            f"\nWaiting for {len(executor.pending_limits)} pending limit orders "
            f"(timeout: {executor.limit_timeout_minutes}min)...",
            file=sys.stderr,
        )
        # Wait and check periodically
        check_interval = 60  # check every 60 seconds
        max_wait_seconds = executor.limit_timeout_minutes * 60
        waited = 0
        while executor.pending_limits and waited < max_wait_seconds:
            time.sleep(min(check_interval, max_wait_seconds - waited))
            waited += check_interval
            conversions = executor.check_pending_limits()
            if conversions:
                for conv in conversions:
                    print(
                        f"  Converted {conv['symbol']}: limit → market",
                        file=sys.stderr,
                    )
        # Final check for any remaining
        if executor.pending_limits:
            executor.check_pending_limits()

    # Log slippage summary
    slippage_summary = executor.get_slippage_summary()
    if slippage_summary.get("filled", 0) > 0:
        print(
            f"\n--- SLIPPAGE SUMMARY ---\n"
            f"  Orders tracked: {slippage_summary['filled']}\n"
            f"  Avg slippage: {slippage_summary['avg_slippage_pct']:+.4f}%\n"
            f"  Limit orders: {slippage_summary['limit_orders']['count']} "
            f"(avg: {slippage_summary['limit_orders']['avg_slippage_pct']:+.4f}%)\n"
            f"  Market orders: {slippage_summary['market_orders']['count']} "
            f"(avg: {slippage_summary['market_orders']['avg_slippage_pct']:+.4f}%)",
            file=sys.stderr,
        )

    # Place protective orders and write position metadata for new positions
    if pending_stops:
        print(
            f"\nWaiting 5s for fills before placing "
            f"{len(pending_stops)} protective orders...",
            file=sys.stderr,
        )
        time.sleep(5)

        smid_set = set(SMID_CAP)
        metadata = load_metadata()

        for symbol, qty, stop_price, order_obj, position_side in pending_stops:
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

                # Trailing stop side: SELL for longs, BUY for shorts
                trail_side = (
                    OrderSide.BUY if position_side == "short" else OrderSide.SELL
                )

                # Place broker-side trailing stop (outer safety envelope)
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
                    position_side=position_side,
                    second_target_pct=pc.second_target_pct,
                    second_sell_pct=pc.second_sell_pct,
                    second_target_hit=False,
                )

            except Exception as e:
                alert(
                    f"Protective order FAILED for {symbol}: {e}",
                    AlertLevel.ERROR,
                )

        save_metadata(metadata)

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
        total_unrealized = sum(p.get("unrealized_pl", 0) for p in positions_after.values())
        trade_logger.log_snapshot(
            equity=portfolio.equity,
            cash=portfolio.cash,
            num_positions=len(positions_after),
            num_long=num_long,
            num_short=num_short,
            total_unrealized_pnl=total_unrealized,
            regime=regime.regime.value,
            vix_level=vix_proxy,
        )
    except Exception as e:
        print(f"[Snapshot] Warning: failed to log portfolio snapshot: {e}", file=sys.stderr)

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
    trade_logger.close()

    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
