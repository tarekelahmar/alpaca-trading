"""Profit-taking configuration per strategy and conviction tier.

Consumed by:
  - run_daily.py: to compute position_metadata at entry
  - price_monitor.py: to evaluate exits in real-time

Scale-out strategy (3-stage):
  - Stage 1: Sell ~33% at first target
  - Stage 2: Sell ~50% of remaining (~33% of original) at second target
  - Stage 3: Let final ~33% ride with dynamically tightened trailing stop

Mean reversion is special: exits 100% at BB middle (no partial scale-out).
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
    second_target_pct: float = 0.0  # % gain to sell second tranche (0 = disabled)
    second_sell_pct: float = 0.0  # fraction of REMAINING to sell at second target


# (strategy_name, conviction_tier) -> ProfitConfig
PROFIT_CONFIGS: dict[tuple[str, int], ProfitConfig] = {
    # --- trend_following: let winners run, scale out in stages ---
    ("trend_following", 1): ProfitConfig(0.12, 0.33, 3.0, None, False, 0.20, 0.50),
    ("trend_following", 2): ProfitConfig(0.08, 0.33, 2.5, None, False, 0.18, 0.50),
    ("trend_following", 3): ProfitConfig(0.06, 0.40, 2.0, None, False, 0.15, 0.50),
    ("trend_following", 4): ProfitConfig(0.05, 0.40, 1.5, None, False, 0.12, 0.50),
    # --- mean_reversion: exit 100% at BB middle, no partial ---
    ("mean_reversion", 1): ProfitConfig(0.0, 0.0, 1.5, None, True),
    ("mean_reversion", 2): ProfitConfig(0.0, 0.0, 1.5, None, True),
    ("mean_reversion", 3): ProfitConfig(0.0, 0.0, 1.2, None, True),
    ("mean_reversion", 4): ProfitConfig(0.0, 0.0, 1.0, None, True),
    # --- momentum: widest targets, let big movers run ---
    ("momentum", 1): ProfitConfig(0.15, 0.33, 3.5, None, False, 0.25, 0.50),
    ("momentum", 2): ProfitConfig(0.10, 0.33, 3.0, None, False, 0.20, 0.50),
    ("momentum", 3): ProfitConfig(0.08, 0.40, 2.5, None, False, 0.18, 0.50),
    ("momentum", 4): ProfitConfig(0.06, 0.40, 2.0, None, False, 0.12, 0.50),
    # --- sentiment: moderate targets ---
    ("sentiment", 1): ProfitConfig(0.10, 0.33, 2.5, None, False, 0.18, 0.50),
    ("sentiment", 2): ProfitConfig(0.07, 0.33, 2.0, None, False, 0.15, 0.50),
    ("sentiment", 3): ProfitConfig(0.05, 0.40, 1.5, None, False, 0.12, 0.50),
    ("sentiment", 4): ProfitConfig(0.04, 0.40, 1.5, None, False, 0.10, 0.50),
    # --- earnings_momentum: time-limited (PEAD drift exhaustion) ---
    ("earnings_momentum", 1): ProfitConfig(0.12, 0.40, 3.0, 40, False, 0.20, 0.50),
    ("earnings_momentum", 2): ProfitConfig(0.08, 0.40, 2.5, 30, False, 0.16, 0.50),
    ("earnings_momentum", 3): ProfitConfig(0.06, 0.40, 2.0, 25, False, 0.14, 0.50),
    ("earnings_momentum", 4): ProfitConfig(0.05, 0.40, 1.5, 20, False, 0.13, 0.50),
    # --- gap_trading: short-hold, tight stops, time-limited (1-5 day trades) ---
    ("gap_trading", 1): ProfitConfig(0.06, 0.50, 1.5, 5, False, 0.10, 0.50),
    ("gap_trading", 2): ProfitConfig(0.05, 0.50, 1.5, 5, False, 0.08, 0.50),
    ("gap_trading", 3): ProfitConfig(0.04, 0.50, 1.2, 4, False, 0.07, 0.50),
    ("gap_trading", 4): ProfitConfig(0.03, 0.50, 1.0, 3, False, 0.06, 0.50),
    # --- sector_rotation: wide targets, longer holds (weeks to months) ---
    ("sector_rotation", 1): ProfitConfig(0.12, 0.33, 3.0, None, False, 0.22, 0.50),
    ("sector_rotation", 2): ProfitConfig(0.08, 0.33, 2.5, None, False, 0.18, 0.50),
    ("sector_rotation", 3): ProfitConfig(0.06, 0.40, 2.0, None, False, 0.15, 0.50),
    ("sector_rotation", 4): ProfitConfig(0.05, 0.40, 2.0, None, False, 0.12, 0.50),
}

# SMID_CAP stocks get wider trailing stops (more volatile)
SMID_CAP_TRAIL_MULT = 1.3

# Fallback for unknown (strategy, tier) combos
DEFAULT_CONFIG = ProfitConfig(0.06, 0.40, 2.0, None, False, 0.14, 0.50)


def get_profit_config(strategy_name: str, conviction_tier: int) -> ProfitConfig:
    """Look up profit config, falling back to default."""
    return PROFIT_CONFIGS.get((strategy_name, conviction_tier), DEFAULT_CONFIG)
