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

from engine import StrategyEngine
from portfolio.sizing import PortfolioContext
from portfolio.universe import UniverseSelector


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
        print("ERROR: Could not fetch SPY data for regime detection", file=sys.stderr)
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

    # Execute orders
    print(f"\nExecuting {len(orders)} orders...", file=sys.stderr)
    for order in orders:
        try:
            order_params = {
                "symbol": order.symbol,
                "qty": order.qty,
                "side": order.side,
                "type": order.order_type,
                "time_in_force": "day",
            }
            if order.limit_price:
                order_params["limit_price"] = order.limit_price
                order_params["type"] = "limit"

            result = trading_client.submit_order(**order_params)
            print(
                f"  ✓ {order.side} {order.qty} {order.symbol}: "
                f"order_id={result.id}, status={result.status}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"  ✗ {order.side} {order.qty} {order.symbol}: {e}",
                file=sys.stderr,
            )

    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
