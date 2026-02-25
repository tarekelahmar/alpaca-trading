"""Per-position metadata persistence for profit-taking.

Stores entry context (strategy, tier, ATR, targets) as a JSON file.
Used by:
  - run_daily.py: writes metadata at position entry
  - price_monitor.py: reads metadata for exit decisions, updates on partial fills
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path


DEFAULT_METADATA_PATH = os.environ.get(
    "POSITION_METADATA_PATH",
    "/opt/alpaca-trading/data/position_metadata.json",
)


@dataclass
class PositionMeta:
    entry_price: float
    entry_date: str  # ISO date "YYYY-MM-DD"
    strategy: str
    conviction_tier: int
    atr_at_entry: float
    initial_qty: int
    remaining_qty: int
    first_target_hit: bool
    first_target_pct: float
    partial_sell_pct: float
    trail_stop_atr_mult: float
    time_exit_days: int | None
    exit_at_bb_middle: bool
    bb_middle_at_entry: float | None
    is_smid_cap: bool
    broker_trailing_stop_order_id: str | None = None
    position_side: str = "long"  # "long" or "short"
    # Water marks — persisted so trailing stops survive monitor restarts
    high_water_mark: float | None = None  # highest price seen (longs)
    low_water_mark: float | None = None   # lowest price seen (shorts)
    # Second profit target — 3-stage scale-out
    second_target_pct: float = 0.0    # 0 = disabled
    second_sell_pct: float = 0.0      # fraction of remaining to sell
    second_target_hit: bool = False


def load_metadata(path: str = DEFAULT_METADATA_PATH) -> dict[str, PositionMeta]:
    """Load position metadata from JSON file."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for symbol, data in raw.items():
        result[symbol] = PositionMeta(**data)
    return result


def save_metadata(
    metadata: dict[str, PositionMeta],
    path: str = DEFAULT_METADATA_PATH,
) -> None:
    """Atomically write position metadata to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(
            {sym: asdict(meta) for sym, meta in metadata.items()},
            f,
            indent=2,
        )
    os.replace(tmp_path, path)  # atomic on POSIX
