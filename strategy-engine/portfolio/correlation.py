"""Correlation-Aware Position Limits.

Computes rolling pairwise correlations between stocks and uses them
to prevent the portfolio from becoming one big correlated bet.

If a new entry is >0.80 correlated with 2+ existing positions,
the position size is reduced by 50%. If correlated with 3+, it's skipped.

Uses 60-day rolling returns for correlation computation.
"""

import os
import sys

import pandas as pd


# Correlation threshold: above this, two stocks are "highly correlated"
CORRELATION_THRESHOLD = float(os.environ.get("CORRELATION_THRESHOLD", "0.80"))

# How many correlated positions trigger size reduction vs skip
CORR_REDUCE_AT = int(os.environ.get("CORR_REDUCE_AT", "2"))  # reduce size at 2+
CORR_SKIP_AT = int(os.environ.get("CORR_SKIP_AT", "3"))      # skip at 3+

# Size reduction factor when correlation limit is hit (but not skipped)
CORR_SIZE_REDUCTION = float(os.environ.get("CORR_SIZE_REDUCTION", "0.50"))

# Rolling window for correlation computation
CORR_LOOKBACK_DAYS = int(os.environ.get("CORR_LOOKBACK_DAYS", "60"))


def compute_return_correlation(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    lookback: int = CORR_LOOKBACK_DAYS,
) -> float | None:
    """Compute rolling correlation between two stocks' daily returns.

    Args:
        data_a: OHLCV DataFrame for stock A (must have 'close' column).
        data_b: OHLCV DataFrame for stock B (must have 'close' column).
        lookback: Number of trading days to use for correlation.

    Returns:
        Pearson correlation coefficient, or None if insufficient data.
    """
    if data_a is None or data_b is None:
        return None

    if len(data_a) < lookback or len(data_b) < lookback:
        return None

    # Use last `lookback` days of close prices
    returns_a = data_a["close"].pct_change().dropna().tail(lookback)
    returns_b = data_b["close"].pct_change().dropna().tail(lookback)

    if len(returns_a) < lookback * 0.8 or len(returns_b) < lookback * 0.8:
        return None

    # Align on common dates
    combined = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()

    if len(combined) < lookback * 0.5:
        return None

    corr = combined["a"].corr(combined["b"])
    return float(corr) if pd.notna(corr) else None


def check_correlation_limit(
    new_symbol: str,
    existing_symbols: list[str],
    price_data: dict[str, pd.DataFrame],
    threshold: float = CORRELATION_THRESHOLD,
    reduce_at: int = CORR_REDUCE_AT,
    skip_at: int = CORR_SKIP_AT,
    lookback: int = CORR_LOOKBACK_DAYS,
) -> tuple[str, float, list[tuple[str, float]]]:
    """Check if a new entry would be too correlated with existing positions.

    Args:
        new_symbol: Symbol being considered for entry.
        existing_symbols: Symbols currently held in the portfolio.
        price_data: Dict mapping symbol -> OHLCV DataFrame.
        threshold: Correlation threshold (default 0.80).
        reduce_at: Number of correlated positions to trigger size reduction.
        skip_at: Number of correlated positions to skip entry entirely.
        lookback: Days for correlation window.

    Returns:
        Tuple of (action, size_multiplier, correlated_pairs):
            action: "allow" | "reduce" | "skip"
            size_multiplier: 1.0 for allow, CORR_SIZE_REDUCTION for reduce, 0.0 for skip
            correlated_pairs: list of (symbol, correlation) that are highly correlated
    """
    if new_symbol not in price_data:
        return "allow", 1.0, []

    new_data = price_data[new_symbol]
    correlated_pairs: list[tuple[str, float]] = []

    for held_symbol in existing_symbols:
        if held_symbol not in price_data:
            continue

        corr = compute_return_correlation(
            new_data, price_data[held_symbol], lookback=lookback
        )

        if corr is not None and abs(corr) >= threshold:
            correlated_pairs.append((held_symbol, round(corr, 3)))

    num_correlated = len(correlated_pairs)

    if num_correlated >= skip_at:
        return "skip", 0.0, correlated_pairs
    elif num_correlated >= reduce_at:
        return "reduce", CORR_SIZE_REDUCTION, correlated_pairs
    else:
        return "allow", 1.0, correlated_pairs
