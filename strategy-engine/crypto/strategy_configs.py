"""Crypto-Adapted Strategy Configurations.

These configs parameterize the EXISTING equity strategies for crypto markets.
Rather than rewriting strategies from scratch, we use StrategyConfig with
crypto-tuned parameters. Key adaptations:

Momentum:
    - Shorter lookback periods (crypto moves faster)
    - Wider RSI bands (40/80 instead of equity defaults)
    - Lower volume thresholds (crypto volume is different)
    - Fewer top_n (smaller universe)

Trend Following:
    - Faster EMA pair (12/26 instead of 20/50, 100 instead of 200)
    - Wider ATR stops (3x instead of 2x — crypto is more volatile)
    - Lower ADX threshold (20 vs 25 — crypto trends start faster)

Mean Reversion:
    - Only safe in ranging/bull regimes (controlled by allocator)
    - RSI bands 25/75 (more extreme — crypto means harder)
    - Wider BB proximity (1.0% vs 0.5%)
    - Tighter time exits (crypto can gap through stops on trend resumption)
"""

from strategies.base import StrategyConfig


# --- Crypto Momentum Config ---
CRYPTO_MOMENTUM_CONFIG = StrategyConfig(
    name="momentum",
    params={
        "top_n": 5,  # smaller universe → fewer picks
        "exit_rank_threshold_multiplier": 1.5,
        "roc_periods": [7, 14, 30, 90],  # 1w, 2w, 1m, 3m (faster than equity)
        "roc_weights": [0.20, 0.25, 0.30, 0.25],  # more weight on short-term
        "ema_trend_period": 50,  # 50 instead of 100 (crypto cycles are shorter)
        "min_avg_volume": 10_000,  # lower bar (crypto volume denominated differently)
        "volume_lookback": 14,  # 2 weeks
        "min_roc_6m": 5.0,  # lower threshold (3m is our longest, not 6m)
        "min_roc_2w": -15.0,  # allow deeper dips (crypto is more volatile)
    },
)

# --- Crypto Trend Following Config ---
CRYPTO_TREND_CONFIG = StrategyConfig(
    name="trend_following",
    params={
        "fast_ema": 12,  # 12 instead of 20 (faster entry)
        "slow_ema": 26,  # 26 instead of 50 (MACD standard)
        "trend_ema": 100,  # 100 instead of 200 (crypto cycles shorter)
        "adx_period": 14,
        "adx_threshold": 20,  # lower than equity (crypto trends with less ADX)
        "atr_period": 14,
        "atr_stop_multiplier": 3.0,  # wider stops — crypto whips more
        "pullback_atr_distance": 2.5,  # wider pullback zone
        "min_avg_volume": 10_000,
        "volume_lookback": 14,
    },
)

# --- Crypto Mean Reversion Config ---
# WARNING: Mean reversion in crypto is DANGEROUS in trending markets.
# The allocator gives this 0% in trending_bullish and trending_bearish regimes.
CRYPTO_MEAN_REVERSION_CONFIG = StrategyConfig(
    name="mean_reversion",
    params={
        "rsi_period": 14,
        "rsi_oversold": 25,  # more extreme than equity (crypto oversold is deeper)
        "rsi_overbought": 75,  # more extreme
        "rsi_exit": 50,
        "bb_period": 20,
        "bb_std": 2.5,  # wider bands for crypto volatility
        "bb_proximity_pct": 0.01,  # 1% proximity (wider than equity 0.5%)
        "adx_period": 14,
        "adx_max": 30,  # allow slightly trendier markets (crypto is always a bit trendy)
        "atr_period": 14,
        "atr_stop_multiplier": 2.0,  # wider than equity MR (1.5)
        "min_avg_volume": 10_000,
        "volume_lookback": 14,
    },
)


def get_crypto_strategy_configs() -> list[StrategyConfig]:
    """Return all crypto-adapted strategy configs."""
    return [
        CRYPTO_MOMENTUM_CONFIG,
        CRYPTO_TREND_CONFIG,
        CRYPTO_MEAN_REVERSION_CONFIG,
    ]
