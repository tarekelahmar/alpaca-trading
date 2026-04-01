"""Profit-taking configuration per strategy and conviction tier.

Consumed by:
  - run_daily.py: to compute position_metadata at entry
  - price_monitor.py: to evaluate exits in real-time

Scale-out strategy (3-stage):
  - Stage 1: Sell ~33-40% at first target (ATR-based)
  - Stage 2: Sell ~50% of remaining at second target (ATR-based)
  - Stage 3: Let final ~33% ride with dynamically tightened trailing stop

Targets are expressed as ATR multiples, then converted to % at entry time
using the stock's actual ATR. This adapts to each stock's volatility:
  - Low-vol stock (ATR=1%): T1 at 3x ATR = 3% gain
  - High-vol stock (ATR=4%): T1 at 3x ATR = 12% gain

Mean reversion scales out in thirds (at -1σ, middle, +1σ) instead of
exiting 100% at BB middle — captures overshoot profit.

Gap trading uses wider time limits (7-10 days, was 3-5).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfitConfig:
    """Exit configuration for a (strategy, tier) combination."""

    first_target_atr: float  # ATR multiples for first target (0 = use pct fallback)
    first_target_pct: float  # % gain fallback if ATR unavailable
    partial_sell_pct: float  # fraction of position to sell at first target
    trail_stop_atr_mult: float  # ATR multiplier for trailing stop
    time_exit_days: int | None  # close after N trading days (None = no limit)
    exit_at_bb_middle: bool  # True for mean_reversion: staged BB exits
    second_target_atr: float = 0.0  # ATR multiples for second target
    second_target_pct: float = 0.0  # % gain fallback for second target
    second_sell_pct: float = 0.0  # fraction of REMAINING to sell at second target


def atr_to_pct(atr_mult: float, atr: float, entry_price: float) -> float:
    """Convert ATR multiple target to percentage gain."""
    if entry_price <= 0 or atr <= 0:
        return 0.0
    return (atr_mult * atr) / entry_price


# (strategy_name, conviction_tier) -> ProfitConfig
# first_target_atr, first_target_pct(fallback), partial_sell, trail_atr, time, bb_middle,
#   second_target_atr, second_target_pct(fallback), second_sell
PROFIT_CONFIGS: dict[tuple[str, int], ProfitConfig] = {
    # --- trend_following: let winners run, wide ATR targets ---
    ("trend_following", 1): ProfitConfig(4.0, 0.12, 0.33, 3.0, None, False, 7.0, 0.20, 0.50),
    ("trend_following", 2): ProfitConfig(3.5, 0.08, 0.33, 2.5, None, False, 6.0, 0.18, 0.50),
    ("trend_following", 3): ProfitConfig(3.0, 0.06, 0.40, 2.0, None, False, 5.0, 0.15, 0.50),
    ("trend_following", 4): ProfitConfig(2.5, 0.05, 0.40, 1.5, None, False, 4.0, 0.12, 0.50),
    # --- mean_reversion: staged BB exits (1/3 at T1, 1/3 at middle, 1/3 rides) ---
    ("mean_reversion", 1): ProfitConfig(2.0, 0.03, 0.33, 1.5, None, True, 3.0, 0.05, 0.50),
    ("mean_reversion", 2): ProfitConfig(2.0, 0.03, 0.33, 1.5, None, True, 3.0, 0.05, 0.50),
    ("mean_reversion", 3): ProfitConfig(1.5, 0.02, 0.33, 1.2, None, True, 2.5, 0.04, 0.50),
    ("mean_reversion", 4): ProfitConfig(1.5, 0.02, 0.33, 1.0, None, True, 2.5, 0.04, 0.50),
    # --- momentum: widest ATR targets, let big movers run ---
    ("momentum", 1): ProfitConfig(5.0, 0.15, 0.33, 3.5, None, False, 8.0, 0.25, 0.50),
    ("momentum", 2): ProfitConfig(4.0, 0.10, 0.33, 3.0, None, False, 7.0, 0.20, 0.50),
    ("momentum", 3): ProfitConfig(3.0, 0.08, 0.40, 2.5, None, False, 5.0, 0.18, 0.50),
    ("momentum", 4): ProfitConfig(2.5, 0.06, 0.40, 2.0, None, False, 4.0, 0.12, 0.50),
    # --- sentiment: moderate ATR targets ---
    ("sentiment", 1): ProfitConfig(3.5, 0.10, 0.33, 2.5, None, False, 6.0, 0.18, 0.50),
    ("sentiment", 2): ProfitConfig(3.0, 0.07, 0.33, 2.0, None, False, 5.0, 0.15, 0.50),
    ("sentiment", 3): ProfitConfig(2.5, 0.05, 0.40, 1.5, None, False, 4.0, 0.12, 0.50),
    ("sentiment", 4): ProfitConfig(2.0, 0.04, 0.40, 1.5, None, False, 3.5, 0.10, 0.50),
    # --- earnings_momentum: time-limited with ATR targets ---
    ("earnings_momentum", 1): ProfitConfig(4.0, 0.12, 0.40, 3.0, 40, False, 7.0, 0.20, 0.50),
    ("earnings_momentum", 2): ProfitConfig(3.0, 0.08, 0.40, 2.5, 30, False, 5.0, 0.16, 0.50),
    ("earnings_momentum", 3): ProfitConfig(2.5, 0.06, 0.40, 2.0, 25, False, 4.0, 0.14, 0.50),
    ("earnings_momentum", 4): ProfitConfig(2.0, 0.05, 0.40, 1.5, 20, False, 3.5, 0.13, 0.50),
    # --- gap_trading: wider time limits (7-10 days, was 3-5), moderate ATR ---
    ("gap_trading", 1): ProfitConfig(2.5, 0.06, 0.50, 1.5, 10, False, 4.0, 0.10, 0.50),
    ("gap_trading", 2): ProfitConfig(2.0, 0.05, 0.50, 1.5, 8, False, 3.5, 0.08, 0.50),
    ("gap_trading", 3): ProfitConfig(1.5, 0.04, 0.50, 1.2, 7, False, 3.0, 0.07, 0.50),
    ("gap_trading", 4): ProfitConfig(1.5, 0.03, 0.50, 1.0, 7, False, 2.5, 0.06, 0.50),
    # --- sector_rotation: wide ATR targets, longer holds ---
    ("sector_rotation", 1): ProfitConfig(4.0, 0.12, 0.33, 3.0, None, False, 7.0, 0.22, 0.50),
    ("sector_rotation", 2): ProfitConfig(3.5, 0.08, 0.33, 2.5, None, False, 6.0, 0.18, 0.50),
    ("sector_rotation", 3): ProfitConfig(3.0, 0.06, 0.40, 2.0, None, False, 5.0, 0.15, 0.50),
    ("sector_rotation", 4): ProfitConfig(2.5, 0.05, 0.40, 2.0, None, False, 4.0, 0.12, 0.50),
}

# SMID_CAP stocks get wider trailing stops (more volatile)
SMID_CAP_TRAIL_MULT = 1.3

# Fallback for unknown (strategy, tier) combos
DEFAULT_CONFIG = ProfitConfig(3.0, 0.06, 0.40, 2.0, None, False, 5.0, 0.14, 0.50)


def get_profit_config(strategy_name: str, conviction_tier: int) -> ProfitConfig:
    """Look up profit config, falling back to default."""
    return PROFIT_CONFIGS.get((strategy_name, conviction_tier), DEFAULT_CONFIG)
