"""Profit-taking configuration per strategy and conviction tier.

Consumed by:
  - run_daily.py: to compute position_metadata at entry
  - price_monitor.py: to evaluate exits in real-time
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfitConfig:
    """Exit configuration for a (strategy, tier) combination."""

    first_target_pct: float  # % gain to sell first tranche (0 = no partial)
    partial_sell_pct: float  # fraction of position to sell at first target
    trail_stop_atr_mult: float  # ATR multiplier for trailing stop
    time_exit_days: int | None  # close after N trading days (None = no limit)
    exit_at_bb_middle: bool  # True for mean_reversion: exit 100% at BB middle


# (strategy_name, conviction_tier) -> ProfitConfig
PROFIT_CONFIGS: dict[tuple[str, int], ProfitConfig] = {
    # --- trend_following: let winners run, partial scale at target ---
    ("trend_following", 1): ProfitConfig(0.12, 0.50, 3.0, None, False),
    ("trend_following", 2): ProfitConfig(0.08, 0.50, 2.5, None, False),
    ("trend_following", 3): ProfitConfig(0.06, 0.50, 2.0, None, False),
    ("trend_following", 4): ProfitConfig(0.05, 0.50, 1.5, None, False),
    # --- mean_reversion: exit 100% at BB middle, no partial ---
    ("mean_reversion", 1): ProfitConfig(0.0, 0.0, 1.5, None, True),
    ("mean_reversion", 2): ProfitConfig(0.0, 0.0, 1.5, None, True),
    ("mean_reversion", 3): ProfitConfig(0.0, 0.0, 1.2, None, True),
    ("mean_reversion", 4): ProfitConfig(0.0, 0.0, 1.0, None, True),
    # --- momentum: widest targets, let big movers run ---
    ("momentum", 1): ProfitConfig(0.15, 0.50, 3.5, None, False),
    ("momentum", 2): ProfitConfig(0.10, 0.50, 3.0, None, False),
    ("momentum", 3): ProfitConfig(0.08, 0.50, 2.5, None, False),
    ("momentum", 4): ProfitConfig(0.06, 0.50, 2.0, None, False),
    # --- sentiment: moderate targets ---
    ("sentiment", 1): ProfitConfig(0.10, 0.50, 2.5, None, False),
    ("sentiment", 2): ProfitConfig(0.07, 0.50, 2.0, None, False),
    ("sentiment", 3): ProfitConfig(0.05, 0.50, 1.5, None, False),
    ("sentiment", 4): ProfitConfig(0.04, 0.50, 1.5, None, False),
    # --- earnings_momentum: time-limited (PEAD drift exhaustion) ---
    ("earnings_momentum", 1): ProfitConfig(0.12, 0.50, 3.0, 40, False),
    ("earnings_momentum", 2): ProfitConfig(0.08, 0.50, 2.5, 30, False),
    ("earnings_momentum", 3): ProfitConfig(0.06, 0.50, 2.0, 25, False),
    ("earnings_momentum", 4): ProfitConfig(0.05, 0.50, 1.5, 20, False),
}

# SMID_CAP stocks get wider trailing stops (more volatile)
SMID_CAP_TRAIL_MULT = 1.3

# Fallback for unknown (strategy, tier) combos
DEFAULT_CONFIG = ProfitConfig(0.06, 0.50, 2.0, None, False)


def get_profit_config(strategy_name: str, conviction_tier: int) -> ProfitConfig:
    """Look up profit config, falling back to default."""
    return PROFIT_CONFIGS.get((strategy_name, conviction_tier), DEFAULT_CONFIG)
