"""Local Data Cache.

Caches market data locally to reduce API calls and speed up
backtesting. Uses a simple file-based cache with pickle.
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


CACHE_DIR = Path(__file__).parent.parent / ".cache"


class DataCache:

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, symbol: str, timeframe: str, start: str, end: str) -> str:
        return f"{symbol}_{timeframe}_{start}_{end}.pkl"

    def get(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: str = "",
        end: str = "",
    ) -> pd.DataFrame | None:
        """Get cached data if available and not stale."""
        key = self._cache_key(symbol, timeframe, start, end)
        path = self.cache_dir / key

        if not path.exists():
            return None

        # Check staleness (cache expires after 1 day for daily data)
        mod_time = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mod_time > timedelta(days=1):
            return None

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def put(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1Day",
        start: str = "",
        end: str = "",
    ) -> None:
        """Cache data for a symbol."""
        key = self._cache_key(symbol, timeframe, start, end)
        path = self.cache_dir / key

        with open(path, "wb") as f:
            pickle.dump(df, f)

    def clear(self) -> int:
        """Clear all cached data. Returns number of files removed."""
        count = 0
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
            count += 1
        return count
