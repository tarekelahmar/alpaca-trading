#!/usr/bin/env python3
"""Backtest CLI — run backtests on strategies with historical data.

Usage:
    python scripts/backtest.py --strategy trend_following --symbols AAPL,MSFT --years 5
    python scripts/backtest.py --strategy all --years 3 --monte-carlo
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

from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from backtester.engine import BacktestEngine, BacktestConfig
from backtester.monte_carlo import run_monte_carlo


STRATEGY_MAP = {
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
}


def fetch_historical_data(
    symbols: list[str], years: int
) -> dict[str, pd.DataFrame]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print("ERROR: Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY", file=sys.stderr)
        sys.exit(1)

    client = StockHistoricalDataClient(key, secret)
    start = datetime.now() - timedelta(days=years * 365)

    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(params)
            if symbol in bars.data and bars[symbol]:
                records = []
                timestamps = []
                for bar in bars[symbol]:
                    records.append({
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    })
                    timestamps.append(bar.timestamp)
                df = pd.DataFrame(records, index=timestamps)
                df.index.name = "timestamp"
                data[symbol] = df
                print(f"  {symbol}: {len(df)} bars", file=sys.stderr)
        except Exception as e:
            print(f"  {symbol}: ERROR — {e}", file=sys.stderr)

    return data


def print_metrics(metrics, strategy_name: str):
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS: {strategy_name}")
    print(f"{'='*60}")
    print(f"  Total Return:       {metrics.total_return_pct:>10.2f}%")
    print(f"  CAGR:               {metrics.cagr_pct:>10.2f}%")
    print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:>10.3f}")
    print(f"  Sortino Ratio:      {metrics.sortino_ratio:>10.3f}")
    print(f"  Calmar Ratio:       {metrics.calmar_ratio:>10.3f}")
    print(f"  Max Drawdown:       {metrics.max_drawdown_pct:>10.2f}%")
    print(f"  Max DD Duration:    {metrics.max_drawdown_duration_days:>10d} days")
    print(f"  Annual Volatility:  {metrics.annualized_volatility_pct:>10.2f}%")
    print(f"  ---")
    print(f"  Total Trades:       {metrics.total_trades:>10d}")
    print(f"  Win Rate:           {metrics.win_rate_pct:>10.2f}%")
    print(f"  Avg Win:            {metrics.avg_win_pct:>10.2f}%")
    print(f"  Avg Loss:           {metrics.avg_loss_pct:>10.2f}%")
    print(f"  Win/Loss Ratio:     {metrics.avg_win_loss_ratio:>10.3f}")
    print(f"  Profit Factor:      {metrics.profit_factor:>10.3f}")
    print(f"  Expectancy/Trade:  ${metrics.expectancy_per_trade:>10.2f}")
    print(f"  Max Consec Wins:    {metrics.max_consecutive_wins:>10d}")
    print(f"  Max Consec Losses:  {metrics.max_consecutive_losses:>10d}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run backtests")
    parser.add_argument("--strategy", type=str, default="trend_following",
                       choices=list(STRATEGY_MAP.keys()) + ["all"],
                       help="Strategy to backtest")
    parser.add_argument("--symbols", type=str,
                       default="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,V,UNH",
                       help="Comma-separated symbols")
    parser.add_argument("--years", type=int, default=5,
                       help="Years of historical data")
    parser.add_argument("--capital", type=float, default=100000,
                       help="Initial capital")
    parser.add_argument("--monte-carlo", action="store_true",
                       help="Run Monte Carlo analysis on results")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print(f"Fetching {args.years} years of data for {len(symbols)} symbols...", file=sys.stderr)
    data = fetch_historical_data(symbols, args.years)

    if not data:
        print("ERROR: No data fetched", file=sys.stderr)
        sys.exit(1)

    strategies_to_test = (
        list(STRATEGY_MAP.keys()) if args.strategy == "all"
        else [args.strategy]
    )

    bt_config = BacktestConfig(initial_capital=args.capital)
    engine = BacktestEngine(bt_config)
    results = {}

    for strat_name in strategies_to_test:
        print(f"\nBacktesting: {strat_name}...", file=sys.stderr)
        strategy = STRATEGY_MAP[strat_name]()
        result = engine.run(strategy, data)
        print_metrics(result.metrics, strat_name)
        results[strat_name] = result

        if args.monte_carlo and len(result.trades) > 0:
            print(f"Running Monte Carlo ({strat_name})...", file=sys.stderr)
            mc = run_monte_carlo(result.trades["pnl_pct"] * 100, num_simulations=10000)
            print(f"  Monte Carlo (10,000 sims):")
            print(f"    Median return:      {mc.median_return_pct:>8.2f}%")
            print(f"    5th percentile:     {mc.p5_return_pct:>8.2f}%")
            print(f"    95th percentile:    {mc.p95_return_pct:>8.2f}%")
            print(f"    95th pctile DD:     {mc.p95_max_drawdown_pct:>8.2f}%")
            print(f"    P(profit):          {mc.probability_of_profit_pct:>8.2f}%")
            print(f"    P(ruin at -30%):    {mc.probability_of_ruin_pct:>8.2f}%")

    # Validation check
    for strat_name, result in results.items():
        m = result.metrics
        passed = m.sharpe_ratio > 1.0 and m.max_drawdown_pct > -20.0
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{status}: {strat_name} — Sharpe={m.sharpe_ratio:.3f}, MaxDD={m.max_drawdown_pct:.2f}%")

    if args.output:
        output_data = {}
        for name, result in results.items():
            output_data[name] = {
                "metrics": result.metrics.__dict__,
                "num_trades": len(result.trades),
                "windows": result.walk_forward_windows,
            }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
