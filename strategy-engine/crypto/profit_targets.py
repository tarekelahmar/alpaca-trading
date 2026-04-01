"""Crypto Profit Target Configuration.

Crypto-specific exit parameters — wider targets and stops than equities
because crypto is significantly more volatile.

Key differences from equity profit targets:
    - First target: 15-25% (vs 5-12% equities)
    - Second target: 25-40% (vs 10-20% equities)
    - Trailing stop: 3-4x ATR (vs 1.5-3x equities)
    - Time exits: None (crypto trends can last months)
    - Mean reversion: exit at BB middle (same concept)
"""

from portfolio.profit_targets import ProfitConfig, PROFIT_CONFIGS, SMID_CAP_TRAIL_MULT


# Crypto-specific profit configs — wider ATR targets than equities
# ProfitConfig(first_target_atr, first_target_pct, partial_sell, trail_atr, time, bb,
#              second_target_atr, second_target_pct, second_sell)
CRYPTO_PROFIT_CONFIGS: dict[tuple[str, int], ProfitConfig] = {
    # --- Crypto Trend Following: let winners run, wide stops ---
    ("trend_following", 1): ProfitConfig(5.0, 0.20, 0.33, 4.0, None, False, 8.0, 0.35, 0.50),
    ("trend_following", 2): ProfitConfig(4.0, 0.15, 0.33, 3.5, None, False, 7.0, 0.30, 0.50),
    ("trend_following", 3): ProfitConfig(3.5, 0.12, 0.40, 3.0, None, False, 6.0, 0.25, 0.50),
    ("trend_following", 4): ProfitConfig(3.0, 0.10, 0.40, 2.5, None, False, 5.0, 0.20, 0.50),

    # --- Crypto Mean Reversion: staged BB exits ---
    ("mean_reversion", 1): ProfitConfig(3.0, 0.05, 0.33, 2.5, None, True, 5.0, 0.08, 0.50),
    ("mean_reversion", 2): ProfitConfig(3.0, 0.05, 0.33, 2.5, None, True, 5.0, 0.08, 0.50),
    ("mean_reversion", 3): ProfitConfig(2.5, 0.04, 0.33, 2.0, None, True, 4.0, 0.06, 0.50),
    ("mean_reversion", 4): ProfitConfig(2.5, 0.04, 0.33, 2.0, None, True, 4.0, 0.06, 0.50),

    # --- Crypto Momentum: widest targets, catch parabolic moves ---
    ("momentum", 1): ProfitConfig(6.0, 0.25, 0.33, 4.5, None, False, 10.0, 0.40, 0.50),
    ("momentum", 2): ProfitConfig(5.0, 0.18, 0.33, 4.0, None, False, 8.0, 0.30, 0.50),
    ("momentum", 3): ProfitConfig(4.0, 0.15, 0.40, 3.5, None, False, 7.0, 0.25, 0.50),
    ("momentum", 4): ProfitConfig(3.0, 0.10, 0.40, 3.0, None, False, 5.0, 0.20, 0.50),

    # --- BTC Dominance Rotation: medium targets, rotation-based ---
    ("btc_dominance", 1): ProfitConfig(4.5, 0.18, 0.33, 3.5, None, False, 7.0, 0.30, 0.50),
    ("btc_dominance", 2): ProfitConfig(3.5, 0.12, 0.33, 3.0, None, False, 6.0, 0.25, 0.50),
    ("btc_dominance", 3): ProfitConfig(3.0, 0.10, 0.40, 2.5, None, False, 5.0, 0.20, 0.50),
    ("btc_dominance", 4): ProfitConfig(2.5, 0.08, 0.40, 2.5, None, False, 4.0, 0.15, 0.50),
}

# Crypto fallback config (wider than equity default)
CRYPTO_DEFAULT_CONFIG = ProfitConfig(3.5, 0.12, 0.40, 3.0, None, False, 6.0, 0.25, 0.50)

# Tier 3 altcoins get even wider trailing stops (like SMID_CAP)
ALT_COIN_TRAIL_MULT = 1.3


def get_crypto_profit_config(strategy_name: str, conviction_tier: int) -> ProfitConfig:
    """Look up crypto profit config, falling back to crypto default."""
    return CRYPTO_PROFIT_CONFIGS.get(
        (strategy_name, conviction_tier), CRYPTO_DEFAULT_CONFIG
    )
