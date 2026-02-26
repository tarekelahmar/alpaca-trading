#!/usr/bin/env python3
"""Crypto Strategy Backtester.

Validates crypto-adapted strategies against historical BTC, ETH, and altcoin
data before deploying live. Uses the same walk-forward framework as equities
but with crypto-tuned parameters.

Tests:
    1. Each strategy individually (momentum, trend, mean reversion, btc_dominance)
    2. Combined system performance
    3. Monte Carlo simulation for robustness

Usage:
    python scripts/backtest_crypto.py
    python scripts/backtest_crypto.py --symbols BTC/USD,ETH/USD --days 365
    python scripts/backtest_crypto.py --individual --monte-carlo
    python scripts/backtest_crypto.py --json > crypto_backtest_results.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from backtester.metrics import PerformanceMetrics
from backtester.monte_carlo import run_monte_carlo
from crypto.backtester import (
    CryptoBacktestEngine, get_crypto_backtest_config, BacktestResult,
)
from crypto.strategy_configs import (
    CRYPTO_MOMENTUM_CONFIG, CRYPTO_TREND_CONFIG, CRYPTO_MEAN_REVERSION_CONFIG,
)
from crypto.btc_dominance import BTCDominanceStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy


def get_data_client() -> CryptoHistoricalDataClient:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print(
            "ERROR: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set",
            file=sys.stderr,
        )
        sys.exit(1)
    return CryptoHistoricalDataClient(key, secret)


def fetch_crypto_data(
    client: CryptoHistoricalDataClient,
    symbols: list[str],
    days: int = 500,
) -> dict[str, pd.DataFrame]:
    """Fetch historical daily bars for crypto symbols."""
    print(
        f"Fetching {days} days of crypto data for {len(symbols)} symbols...",
        file=sys.stderr,
    )
    start = datetime.now() - timedelta(days=days)

    data: dict[str, pd.DataFrame] = {}
    try:
        params = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
        )
        bars = client.get_crypto_bars(params)

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
        print(f"Error fetching crypto data: {e}", file=sys.stderr)
        # Fall back to individual
        for symbol in symbols:
            try:
                params = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start,
                )
                bars = client.get_crypto_bars(params)
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


def print_metrics(name: str, m: PerformanceMetrics):
    """Print metrics in a compact format."""
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
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
    """Print trade summary by exit reason and direction."""
    if result.trades.empty:
        print(f"\n  No trades for {name}")
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

    # Per-symbol breakdown
    if "symbol" in result.trades.columns:
        print(f"\n  Per-Symbol ({name}):")
        symbols = result.trades.groupby("symbol").agg(
            count=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl_pct=("pnl_pct", "mean"),
        )
        for sym, row in symbols.sort_values("total_pnl", ascending=False).iterrows():
            short_sym = sym.replace("/USD", "")
            print(
                f"    {short_sym:8s}: {int(row['count']):3d} trades, "
                f"P&L=${row['total_pnl']:+10,.2f}, "
                f"avg={row['avg_pnl_pct']*100:+.2f}%"
            )


def print_validation(name: str, m: PerformanceMetrics):
    """Print validation checks — does the strategy meet our minimum bar?"""
    print(f"\n  Validation ({name}):")
    checks = [
        ("Sharpe > 0.5", m.sharpe_ratio > 0.5),
        ("Max DD > -30%", m.max_drawdown_pct > -30),
        ("Win Rate > 35%", m.win_rate_pct > 35),
        ("Profit Factor > 1.0", m.profit_factor > 1.0),
        ("Total Trades >= 10", m.total_trades >= 10),
        ("CAGR > 0%", m.cagr_pct > 0),
    ]
    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {label}")

    if all_pass:
        print(f"    >>> ALL CHECKS PASSED - Strategy is viable for paper trading")
    else:
        print(f"    >>> SOME CHECKS FAILED - Review before deploying")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Crypto strategy backtester")
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated crypto symbols (default: BTC,ETH,SOL + alts)",
    )
    parser.add_argument(
        "--days", type=int, default=500,
        help="Days of history (default: 500)",
    )
    parser.add_argument(
        "--individual", action="store_true", default=True,
        help="Run per-strategy backtests (default: true)",
    )
    parser.add_argument(
        "--monte-carlo", action="store_true",
        help="Run Monte Carlo simulation (10,000 resamples)",
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
        # Default: top crypto pairs on Alpaca
        symbols = [
            "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD",
            "LINK/USD", "DOT/USD", "MATIC/USD", "UNI/USD",
            "DOGE/USD", "LTC/USD",
        ]

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  CRYPTO STRATEGY BACKTEST", file=sys.stderr)
    print(f"  {datetime.now().isoformat()}", file=sys.stderr)
    print(f"  Symbols: {', '.join(symbols)}", file=sys.stderr)
    print(f"  History: {args.days} days", file=sys.stderr)
    print(f"  Capital: ${args.capital:,.0f}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # Fetch data
    client = get_data_client()
    data = fetch_crypto_data(client, symbols, days=args.days)

    if not data:
        print("ERROR: No crypto data fetched. Exiting.", file=sys.stderr)
        sys.exit(1)

    if "BTC/USD" not in data:
        print("ERROR: BTC/USD data required. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Create crypto-adapted strategies
    strategies = [
        ("crypto_momentum", MomentumStrategy(CRYPTO_MOMENTUM_CONFIG)),
        ("crypto_trend_following", TrendFollowingStrategy(CRYPTO_TREND_CONFIG)),
        ("crypto_mean_reversion", MeanReversionStrategy(CRYPTO_MEAN_REVERSION_CONFIG)),
        ("btc_dominance", BTCDominanceStrategy()),
    ]

    # Create crypto backtest engine
    config = get_crypto_backtest_config(initial_capital=args.capital)
    engine = CryptoBacktestEngine(config)

    results: dict[str, BacktestResult] = {}
    all_valid = True

    # Run individual strategy backtests
    for name, strategy in strategies:
        print(f"\nRunning crypto backtest: {name}...", file=sys.stderr)
        try:
            result = engine.run(strategy, data)
            results[name] = result

            if not args.json:
                print_metrics(name, result.metrics)
                print_trade_summary(result, name)
                valid = print_validation(name, result.metrics)
                if not valid:
                    all_valid = False

        except Exception as e:
            print(f"ERROR backtesting {name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    # Monte Carlo simulation for the best-performing strategy
    if args.monte_carlo and results:
        best_name = max(
            results.keys(),
            key=lambda k: results[k].metrics.sharpe_ratio,
        )
        best_result = results[best_name]

        if not best_result.trades.empty:
            print(f"\n{'='*55}")
            print(f"  MONTE CARLO SIMULATION: {best_name}")
            print(f"  (10,000 resamples of trade sequence)")
            print(f"{'='*55}")

            try:
                mc = run_monte_carlo(
                    best_result.trades["pnl_pct"],
                    initial_capital=args.capital,
                    num_simulations=10_000,
                    ruin_threshold_pct=-30.0,
                )
                print(f"  Return Distribution:")
                print(f"    5th percentile:  {mc.p5_return_pct:+.2f}%")
                print(f"    25th percentile: {mc.p25_return_pct:+.2f}%")
                print(f"    Median:          {mc.median_return_pct:+.2f}%")
                print(f"    75th percentile: {mc.p75_return_pct:+.2f}%")
                print(f"    95th percentile: {mc.p95_return_pct:+.2f}%")
                print(f"  Probability of profit: {mc.probability_of_profit_pct:.1f}%")
                print(f"  Probability of ruin:   {mc.probability_of_ruin_pct:.1f}%")
                print(f"  Max DD Distribution:")
                print(f"    Median:          {mc.median_max_drawdown_pct:.2f}%")
                print(f"    95th percentile: {mc.p95_max_drawdown_pct:.2f}%")
            except Exception as e:
                print(f"  Monte Carlo error: {e}", file=sys.stderr)

    # Comparison table
    if results and not args.json:
        print(f"\n{'='*80}")
        print(f"  CRYPTO STRATEGY COMPARISON")
        print(f"{'='*80}")
        header = (
            f"  {'Strategy':<25} {'Return':>8} {'Sharpe':>8} "
            f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>7} {'PF':>6}"
        )
        print(header)
        print(f"  {'-'*75}")
        for name, result in sorted(
            results.items(),
            key=lambda x: x[1].metrics.sharpe_ratio,
            reverse=True,
        ):
            m = result.metrics
            print(
                f"  {name:<25} {m.total_return_pct:>+7.1f}% "
                f"{m.sharpe_ratio:>7.3f} "
                f"{m.max_drawdown_pct:>7.1f}% "
                f"{m.win_rate_pct:>7.1f}% "
                f"{m.total_trades:>6d} "
                f"{m.profit_factor:>5.2f}"
            )

        # Overall verdict
        print(f"\n  {'='*30}")
        if all_valid:
            print(f"  ALL STRATEGIES PASSED validation")
            print(f"  Safe to deploy to paper trading")
        else:
            print(f"  SOME STRATEGIES FAILED validation")
            print(f"  Review failing strategies before deploying")
        print(f"  {'='*30}")

    # JSON output
    if args.json:
        json_output = {
            "config": {
                "symbols": list(data.keys()),
                "days": args.days,
                "capital": args.capital,
                "market": "crypto",
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
                "calmar_ratio": m.calmar_ratio,
                "volatility_pct": m.annualized_volatility_pct,
                "exposure_pct": m.exposure_pct,
                "validation": {
                    "sharpe_above_0.5": m.sharpe_ratio > 0.5,
                    "max_dd_above_neg_30": m.max_drawdown_pct > -30,
                    "win_rate_above_35": m.win_rate_pct > 35,
                    "profit_factor_above_1": m.profit_factor > 1.0,
                    "enough_trades": m.total_trades >= 10,
                    "positive_cagr": m.cagr_pct > 0,
                },
            }
        print(json.dumps(json_output, indent=2, default=str))

    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
