"""Per-position metadata persistence for profit-taking.

Stores entry context (strategy, tier, ATR, targets) as a JSON file.
Used by:
  - run_daily.py: writes metadata at position entry
  - price_monitor.py: reads metadata for exit decisions, updates on partial fills

IMPORTANT: Two processes (daily run + monitor) read/write this file.
All writes use merge-and-save to avoid one process overwriting the other's
changes. A file lock prevents concurrent writes from corrupting data.
"""

import fcntl
import json
import os
from dataclasses import dataclass, asdict, fields
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


def _raw_load(path: str) -> dict[str, dict]:
    """Load raw JSON dict from disk (no PositionMeta conversion)."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _to_meta(data: dict) -> PositionMeta:
    """Convert a raw dict to PositionMeta, ignoring unknown fields."""
    valid_fields = {f.name for f in fields(PositionMeta)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return PositionMeta(**filtered)


def load_metadata(path: str = DEFAULT_METADATA_PATH) -> dict[str, PositionMeta]:
    """Load position metadata from JSON file."""
    raw = _raw_load(path)
    return {sym: _to_meta(data) for sym, data in raw.items()}


def save_metadata(
    metadata: dict[str, PositionMeta],
    path: str = DEFAULT_METADATA_PATH,
) -> None:
    """Merge-and-save position metadata with file locking.

    Instead of blindly overwriting, this:
    1. Acquires an exclusive file lock
    2. Reads the current file from disk
    3. Merges the in-memory changes (in-memory wins for existing keys)
    4. Writes atomically via tmp + rename

    This prevents the daily run from overwriting monitor changes and vice versa.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lock_path = path + ".lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # Read current disk state
            disk_data = _raw_load(path)

            # Build merged dict: start with disk, overlay in-memory changes
            merged = {}
            # Keep disk entries that we didn't touch (e.g., daily run added a new symbol)
            for sym, data in disk_data.items():
                merged[sym] = data
            # Overlay in-memory metadata (these are the authoritative changes)
            for sym, meta in metadata.items():
                merged[sym] = asdict(meta)
            # Remove symbols that are in disk but deleted from in-memory
            # (e.g., monitor closed a position and removed it)
            disk_only = set(disk_data.keys()) - set(metadata.keys())
            # Don't remove disk-only entries — they might be new entries
            # from the daily run that we haven't seen yet. Only the process
            # that deleted them should remove them, and it will have them
            # missing from its metadata dict.

            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(merged, f, indent=2)
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def merge_metadata(
    current: dict[str, PositionMeta],
    from_disk: dict[str, PositionMeta],
) -> dict[str, PositionMeta]:
    """Merge disk metadata into current in-memory metadata.

    Used by price_monitor at market open to pick up new entries from
    run_daily.py without losing in-memory water marks and target states.

    Rules:
    - New symbols from disk are added
    - Existing symbols: preserve in-memory water marks and target hit flags
      (these are more recent than disk)
    - Symbols in memory but not on disk: keep them (monitor may have
      updated them after daily run wrote)
    """
    merged = dict(current)  # start with in-memory

    for sym, disk_meta in from_disk.items():
        if sym not in merged:
            # New entry from daily run — add it
            merged[sym] = disk_meta
        else:
            # Already in memory — update entry-time fields from disk
            # but KEEP runtime fields (water marks, target hits)
            mem = merged[sym]
            # Daily run may have updated broker_trailing_stop_order_id
            if disk_meta.broker_trailing_stop_order_id:
                mem.broker_trailing_stop_order_id = disk_meta.broker_trailing_stop_order_id

    return merged
