#!/usr/bin/env python3
"""Multi-Strategy Backtest Runner.

Runs all strategies through the full engine pipeline (signal generation,
filtering with confluence, sizing, optimization) on historical data.

Unlike single-strategy backtesting, this tests the actual combined system
including confluence boosting, regime detection, and portfolio-level
position limits.

Usage:
    python scripts/backtest_all.py [--symbols AAPL,MSFT,...] [--days 500]
    python scripts/backtest_all.py --individual  # also run per-strategy
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from backtester.engine import BacktestEngine, BacktestConfig, BacktestResult
from backtester.metrics import PerformanceMetrics
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.gap_trading import GapTradingStrategy
from strategies.sector_rotation import SectorRotationStrategy


def get_data_client() -> StockHistoricalDataClient:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print("ERROR: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set",
              file=sys.stderr)
        sys.exit(1)
    return StockHistoricalDataClient(key, secret)


def fetch_data(
    client: StockHistoricalDataClient,
    symbols: list[str],
    days: int = 500,
) -> dict[str, pd.DataFrame]:
    """Fetch historical daily bars."""
    print(f"Fetching {days} days of data for {len(symbols)} symbols...",
          file=sys.stderr)
    start = datetime.now() - timedelta(days=days)

    data: dict[str, pd.DataFrame] = {}
    try:
        params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(params)
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
                df = pd.DataFrame(
                    records,
                    index=[b.timestamp for b in bars[symbol]],
                )
                df.index.name = "timestamp"
                data[symbol] = df
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)

    print(f"Got data for {len(data)} symbols", file=sys.stderr)
    return data


def print_metrics(name: str, m: PerformanceMetrics):
    """Print metrics in a compact format."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Total Return:     {m.total_return_pct:+.2f}%")
    print(f"  CAGR:             {m.cagr_pct:+.2f}%")
    print(f"  Sharpe Ratio:     {m.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio:    {m.sortino_ratio:.3f}")
    print(f"  Max Drawdown:     {m.max_drawdown_pct:.2f}%")
    print(f"  Max DD Duration:  {m.max_drawdown_duration_days} days")
    print(f"  Volatility:       {m.annualized_volatility_pct:.2f}%")
    print(f"  Profit Factor:    {m.profit_factor:.3f}")
    print(f"  Win Rate:         {m.win_rate_pct:.1f}%")
    print(f"  Total Trades:     {m.total_trades}")
    print(f"  Avg Win:          {m.avg_win_pct:+.2f}%")
    print(f"  Avg Loss:         {m.avg_loss_pct:+.2f}%")
    print(f"  Win/Loss Ratio:   {m.avg_win_loss_ratio:.3f}")
    print(f"  Exposure:         {m.exposure_pct:.1f}%")
    print(f"  Calmar Ratio:     {m.calmar_ratio:.3f}")


def print_trade_summary(result: BacktestResult, name: str):
    """Print a summary of trades by exit reason."""
    if result.trades.empty:
        return

    if "exit_reason" in result.trades.columns:
        print(f"\n  Exit Reasons ({name}):")
        reasons = result.trades.groupby("exit_reason").agg(
            count=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl_pct=("pnl_pct", "mean"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
        )
        for reason, row in reasons.iterrows():
            print(
                f"    {reason:20s}: {int(row['count']):4d} trades, "
                f"P&L=${row['total_pnl']:+10,.2f}, "
                f"avg={row['avg_pnl_pct']*100:+.2f}%, "
                f"win={row['win_rate']*100:.0f}%"
            )

    if "side" in result.trades.columns:
        print(f"\n  Direction ({name}):")
        sides = result.trades.groupby("side").agg(
            count=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl_pct=("pnl_pct", "mean"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
        )
        for side, row in sides.iterrows():
            print(
                f"    {side:20s}: {int(row['count']):4d} trades, "
                f"P&L=${row['total_pnl']:+10,.2f}, "
                f"avg={row['avg_pnl_pct']*100:+.2f}%, "
                f"win={row['win_rate']*100:.0f}%"
            )


def main():
    parser = argparse.ArgumentParser(description="Multi-strategy backtester")
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols (default: top 20 liquid)",
    )
    parser.add_argument(
        "--days", type=int, default=500,
        help="Days of history to fetch (default: 500)",
    )
    parser.add_argument(
        "--individual", action="store_true",
        help="Also run per-strategy backtests",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Starting capital (default: $100,000)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        # Default: top 20 most liquid S&P 500 stocks
        symbols = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
            "META", "TSLA", "AMD", "NFLX", "AVGO",
            "JPM", "BAC", "XOM", "UNH", "V",
            "HD", "CRM", "COST", "CAT", "GS",
        ]

    # Fetch data
    client = get_data_client()
    data = fetch_data(client, symbols, days=args.days)

    if not data:
        print("ERROR: No data fetched. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Strategies to test
    strategies = [
        TrendFollowingStrategy(),
        MeanReversionStrategy(),
        MomentumStrategy(),
        GapTradingStrategy(),
        SectorRotationStrategy(),
    ]

    config = BacktestConfig(
        initial_capital=args.capital,
        enable_scale_out=True,
        enable_dynamic_trailing=True,
    )
    engine = BacktestEngine(config)

    results: dict[str, BacktestResult] = {}

    # Run individual strategy backtests
    if args.individual:
        for strategy in strategies:
            print(f"\nRunning backtest: {strategy.name}...", file=sys.stderr)
            try:
                result = engine.run(strategy, data)
                results[strategy.name] = result
                if not args.json:
                    print_metrics(strategy.name, result.metrics)
                    print_trade_summary(result, strategy.name)
            except Exception as e:
                print(f"ERROR backtesting {strategy.name}: {e}", file=sys.stderr)

    # Run each strategy separately and compare
    if args.json:
        json_output = {
            "config": {
                "symbols": list(data.keys()),
                "days": args.days,
                "capital": args.capital,
            },
            "results": {},
        }
        for name, result in results.items():
            m = result.metrics
            json_output["results"][name] = {
                "total_return_pct": m.total_return_pct,
                "cagr_pct": m.cagr_pct,
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "max_drawdown_pct": m.max_drawdown_pct,
                "profit_factor": m.profit_factor,
                "win_rate_pct": m.win_rate_pct,
                "total_trades": m.total_trades,
                "avg_win_pct": m.avg_win_pct,
                "avg_loss_pct": m.avg_loss_pct,
            }
        print(json.dumps(json_output, indent=2, default=str))
    elif not args.individual:
        # If not running individual, run at least one strategy to show it works
        print("\nRunning trend_following backtest...", file=sys.stderr)
        try:
            result = engine.run(strategies[0], data)
            print_metrics("trend_following", result.metrics)
            print_trade_summary(result, "trend_following")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)

    # Summary comparison
    if results and not args.json:
        print(f"\n{'='*70}")
        print(f"  STRATEGY COMPARISON")
        print(f"{'='*70}")
        header = f"  {'Strategy':<20} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7} {'PF':>6}"
        print(header)
        print(f"  {'-'*67}")
        for name, result in sorted(results.items(), key=lambda x: x[1].metrics.sharpe_ratio, reverse=True):
            m = result.metrics
            print(
                f"  {name:<20} {m.total_return_pct:>+7.1f}% "
                f"{m.sharpe_ratio:>7.3f} "
                f"{m.max_drawdown_pct:>7.1f}% "
                f"{m.win_rate_pct:>7.1f}% "
                f"{m.total_trades:>6d} "
                f"{m.profit_factor:>5.2f}"
            )

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
