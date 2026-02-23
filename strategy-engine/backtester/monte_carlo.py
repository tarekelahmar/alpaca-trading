"""Monte Carlo Resampling for Strategy Validation.

Resamples trade sequences to estimate:
    - Confidence intervals on total return
    - Probability of ruin (drawdown exceeding threshold)
    - Distribution of max drawdowns
    - Expected Sharpe range

This helps assess whether observed performance is robust
or just lucky sequencing of trades.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    num_simulations: int
    median_return_pct: float
    mean_return_pct: float
    p5_return_pct: float  # 5th percentile (worst case)
    p25_return_pct: float
    p75_return_pct: float
    p95_return_pct: float  # 95th percentile (best case)
    median_max_drawdown_pct: float
    p95_max_drawdown_pct: float  # 95th percentile worst drawdown
    probability_of_profit_pct: float
    probability_of_ruin_pct: float  # P(drawdown > ruin_threshold)
    ruin_threshold_pct: float


def run_monte_carlo(
    trade_returns: pd.Series | np.ndarray,
    num_simulations: int = 10_000,
    initial_capital: float = 100_000.0,
    ruin_threshold_pct: float = -30.0,
    seed: int | None = 42,
) -> MonteCarloResult:
    """Run Monte Carlo resampling on trade return sequences.

    Args:
        trade_returns: Series of per-trade percentage returns.
        num_simulations: Number of Monte Carlo paths to simulate.
        initial_capital: Starting capital for each simulation.
        ruin_threshold_pct: Drawdown % that constitutes "ruin" (negative number).
        seed: Random seed for reproducibility.

    Returns:
        MonteCarloResult with distribution statistics.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    returns = np.array(trade_returns, dtype=float)
    n_trades = len(returns)

    if n_trades == 0:
        return _empty_result(num_simulations, ruin_threshold_pct)

    final_returns = np.zeros(num_simulations)
    max_drawdowns = np.zeros(num_simulations)

    for i in range(num_simulations):
        # Resample trades with replacement
        sampled = rng.choice(returns, size=n_trades, replace=True)

        # Build equity curve
        equity = initial_capital
        peak = equity
        max_dd = 0.0

        for r in sampled:
            equity *= (1 + r / 100)
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak
            if dd < max_dd:
                max_dd = dd

        final_returns[i] = ((equity / initial_capital) - 1) * 100
        max_drawdowns[i] = max_dd * 100

    # Statistics
    prob_profit = float(np.mean(final_returns > 0) * 100)
    prob_ruin = float(np.mean(max_drawdowns < ruin_threshold_pct) * 100)

    return MonteCarloResult(
        num_simulations=num_simulations,
        median_return_pct=round(float(np.median(final_returns)), 2),
        mean_return_pct=round(float(np.mean(final_returns)), 2),
        p5_return_pct=round(float(np.percentile(final_returns, 5)), 2),
        p25_return_pct=round(float(np.percentile(final_returns, 25)), 2),
        p75_return_pct=round(float(np.percentile(final_returns, 75)), 2),
        p95_return_pct=round(float(np.percentile(final_returns, 95)), 2),
        median_max_drawdown_pct=round(float(np.median(max_drawdowns)), 2),
        p95_max_drawdown_pct=round(float(np.percentile(max_drawdowns, 5)), 2),
        probability_of_profit_pct=round(prob_profit, 2),
        probability_of_ruin_pct=round(prob_ruin, 2),
        ruin_threshold_pct=ruin_threshold_pct,
    )


def _empty_result(n: int, ruin: float) -> MonteCarloResult:
    return MonteCarloResult(
        num_simulations=n,
        median_return_pct=0, mean_return_pct=0,
        p5_return_pct=0, p25_return_pct=0,
        p75_return_pct=0, p95_return_pct=0,
        median_max_drawdown_pct=0, p95_max_drawdown_pct=0,
        probability_of_profit_pct=0, probability_of_ruin_pct=0,
        ruin_threshold_pct=ruin,
    )
