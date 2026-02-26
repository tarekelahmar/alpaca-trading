"""Crypto-Adapted Strategy Configurations (Hourly Bars).

These configs parameterize the EXISTING equity strategies for crypto markets
using HOURLY bars instead of daily. Key conversions:
    1 day   = 24 hours
    1 week  = 168 hours
    1 month = 720 hours

Momentum:
    - ROC periods in hours: 24h, 72h (3d), 168h (1w), 720h (1m)
    - Fast EMA for trend filter (120h ≈ 5 days)

Trend Following:
    - Fast EMA: 12h, Slow EMA: 48h, Trend EMA: 168h (1 week)
    - ADX/ATR on 24-bar (1 day) window
    - Wider ATR stops (3x — crypto whips intraday)

Mean Reversion:
    - RSI/BB on 24-bar (1 day) window
    - Only safe in ranging regimes (controlled by allocator)
"""

from strategies.base import StrategyConfig


# --- Crypto Momentum Config (Hourly) ---
CRYPTO_MOMENTUM_CONFIG = StrategyConfig(
    name="momentum",
    params={
        "top_n": 5,
        "exit_rank_threshold_multiplier": 1.5,
        "roc_periods": [24, 72, 168, 720],  # 1d, 3d, 1w, 1m in hours
        "roc_weights": [0.25, 0.30, 0.25, 0.20],  # more weight on short-term
        "ema_trend_period": 120,  # 5-day EMA for trend filter
        "min_avg_volume": 10_000,
        "volume_lookback": 168,  # 1 week of hourly bars
        "min_roc_6m": 5.0,
        "min_roc_2w": -15.0,
    },
)

# --- Crypto Trend Following Config (Hourly) ---
CRYPTO_TREND_CONFIG = StrategyConfig(
    name="trend_following",
    params={
        "fast_ema": 12,      # 12 hours
        "slow_ema": 48,      # 2 days
        "trend_ema": 168,    # 1 week
        "adx_period": 24,    # 1-day ADX
        "adx_threshold": 20,
        "atr_period": 24,    # 1-day ATR
        "atr_stop_multiplier": 3.0,
        "pullback_atr_distance": 2.5,
        "min_avg_volume": 10_000,
        "volume_lookback": 168,  # 1 week
    },
)

# --- Crypto Mean Reversion Config (Hourly) ---
# WARNING: Mean reversion in crypto is DANGEROUS in trending markets.
CRYPTO_MEAN_REVERSION_CONFIG = StrategyConfig(
    name="mean_reversion",
    params={
        "rsi_period": 24,       # 1-day RSI
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "rsi_exit": 50,
        "bb_period": 48,        # 2-day Bollinger Bands
        "bb_std": 2.5,
        "bb_proximity_pct": 0.01,
        "adx_period": 24,       # 1-day ADX
        "adx_max": 30,
        "atr_period": 24,       # 1-day ATR
        "atr_stop_multiplier": 2.0,
        "min_avg_volume": 10_000,
        "volume_lookback": 168,  # 1 week
    },
)


def get_crypto_strategy_configs() -> list[StrategyConfig]:
    """Return all crypto-adapted strategy configs."""
    return [
        CRYPTO_MOMENTUM_CONFIG,
        CRYPTO_TREND_CONFIG,
        CRYPTO_MEAN_REVERSION_CONFIG,
    ]
