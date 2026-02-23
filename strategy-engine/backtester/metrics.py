"""Performance Metrics Calculator.

Computes all key trading performance metrics from an equity curve
and trade list. These metrics are used to validate strategies
before live deployment.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    # Returns
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    avg_drawdown_pct: float

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_win_loss_ratio: float
    profit_factor: float
    expectancy_per_trade: float

    # Other
    max_consecutive_wins: int
    max_consecutive_losses: int
    trading_days: int
    exposure_pct: float  # % of time in market


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame | None = None,
    risk_free_rate: float = 0.05,
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics.

    Args:
        equity_curve: Series of portfolio values indexed by date.
        trades: DataFrame with columns: entry_date, exit_date, pnl, pnl_pct, side.
                If None, trade-level stats will be zeroed.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        PerformanceMetrics dataclass.
    """
    # Daily returns
    returns = equity_curve.pct_change().dropna()
    trading_days = len(returns)

    if trading_days < 2:
        return _empty_metrics()

    # Total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # CAGR
    years = trading_days / 252
    if years > 0 and equity_curve.iloc[0] > 0:
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Volatility (annualized)
    vol = float(returns.std() * math.sqrt(252))

    # Sharpe
    excess_daily = returns.mean() - risk_free_rate / 252
    daily_std = returns.std()
    sharpe = float(excess_daily / daily_std * math.sqrt(252)) if daily_std > 0 else 0.0

    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    downside_std = float(downside.std() * math.sqrt(252)) if len(downside) > 0 else 1e-10
    sortino = float((returns.mean() * 252 - risk_free_rate) / downside_std)

    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min())
    avg_dd = float(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0.0

    # Max drawdown duration
    dd_duration = _max_drawdown_duration(drawdown)

    # Calmar
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 0 else 0.0

    # Trade stats
    trade_stats = _compute_trade_stats(trades)

    return PerformanceMetrics(
        total_return_pct=round(total_return * 100, 2),
        cagr_pct=round(cagr * 100, 2),
        annualized_volatility_pct=round(vol * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        calmar_ratio=round(calmar, 3),
        max_drawdown_pct=round(max_dd * 100, 2),
        max_drawdown_duration_days=dd_duration,
        avg_drawdown_pct=round(avg_dd * 100, 2),
        **trade_stats,
    )


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """Calculate the longest drawdown period in trading days."""
    in_drawdown = drawdown < 0
    max_duration = 0
    current = 0
    for is_dd in in_drawdown:
        if is_dd:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return max_duration


def _compute_trade_stats(trades: pd.DataFrame | None) -> dict:
    """Compute trade-level statistics."""
    if trades is None or len(trades) == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "avg_win_loss_ratio": 0.0,
            "profit_factor": 0.0,
            "expectancy_per_trade": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "trading_days": 0,
            "exposure_pct": 0.0,
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]

    total = len(trades)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / total if total > 0 else 0.0

    avg_win = float(wins["pnl_pct"].mean()) if n_wins > 0 else 0.0
    avg_loss = float(losses["pnl_pct"].mean()) if n_losses > 0 else 0.0
    avg_wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    gross_profit = float(wins["pnl"].sum()) if n_wins > 0 else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if n_losses > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    expectancy = float(trades["pnl"].mean()) if total > 0 else 0.0

    # Consecutive wins/losses
    is_win = (trades["pnl"] > 0).astype(int)
    max_consec_wins = _max_consecutive(is_win, 1)
    max_consec_losses = _max_consecutive(is_win, 0)

    # Exposure
    if "entry_date" in trades.columns and "exit_date" in trades.columns:
        total_days_in_market = 0
        for _, trade in trades.iterrows():
            days = (trade["exit_date"] - trade["entry_date"]).days
            total_days_in_market += max(1, days)
        first_date = trades["entry_date"].min()
        last_date = trades["exit_date"].max()
        total_period = max(1, (last_date - first_date).days)
        exposure = min(1.0, total_days_in_market / total_period)
    else:
        exposure = 0.0

    return {
        "total_trades": total,
        "winning_trades": n_wins,
        "losing_trades": n_losses,
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "avg_win_loss_ratio": round(avg_wl_ratio, 3),
        "profit_factor": round(profit_factor, 3),
        "expectancy_per_trade": round(expectancy, 2),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "trading_days": total,
        "exposure_pct": round(exposure * 100, 2),
    }


def _max_consecutive(series: pd.Series, value: int) -> int:
    """Count maximum consecutive occurrences of a value."""
    max_count = 0
    current = 0
    for v in series:
        if v == value:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return_pct=0, cagr_pct=0, annualized_volatility_pct=0,
        sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
        max_drawdown_pct=0, max_drawdown_duration_days=0, avg_drawdown_pct=0,
        total_trades=0, winning_trades=0, losing_trades=0, win_rate_pct=0,
        avg_win_pct=0, avg_loss_pct=0, avg_win_loss_ratio=0, profit_factor=0,
        expectancy_per_trade=0, max_consecutive_wins=0, max_consecutive_losses=0,
        trading_days=0, exposure_pct=0,
    )
