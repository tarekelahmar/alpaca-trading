"""Trade Logger — comprehensive entry/exit tracking with P&L attribution.

Extends the base DataStore with:
  - Exit logging: records exits from price_monitor.py with P&L
  - Position lifecycle: matches entries to exits for realized P&L
  - Portfolio snapshots: periodic equity/drawdown tracking
  - Performance queries: per-strategy, per-regime, per-tier breakdowns

Uses the same SQLite DB as DataStore but adds new tables and helpers.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


DEFAULT_TRADE_LOG_PATH = os.environ.get(
    "TRADE_LOG_PATH",
    os.path.join(
        os.environ.get("HOME", "/tmp"),
        "alpaca-trading-data",
        "trade_log.db",
    ),
)

_TRADE_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS position_lifecycle (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    strategy TEXT NOT NULL,
    conviction_tier INTEGER NOT NULL DEFAULT 4,
    confluence_count INTEGER NOT NULL DEFAULT 1,

    -- Entry
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_qty INTEGER NOT NULL,
    entry_order_id TEXT,
    entry_regime TEXT,
    entry_vix REAL,
    entry_signal_strength REAL,

    -- Exit (filled when position closes)
    exit_date TEXT,
    exit_price REAL,
    exit_qty INTEGER,
    exit_reason TEXT,
    exit_order_id TEXT,

    -- P&L (computed on exit)
    realized_pnl REAL,
    realized_pnl_pct REAL,
    hold_days INTEGER,

    -- Metadata
    is_partial_close INTEGER NOT NULL DEFAULT 0,
    parent_lifecycle_id INTEGER,  -- links partial exits to original entry
    features_json TEXT NOT NULL DEFAULT '{}',

    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    num_positions INTEGER NOT NULL DEFAULT 0,
    num_long INTEGER NOT NULL DEFAULT 0,
    num_short INTEGER NOT NULL DEFAULT 0,
    total_unrealized_pnl REAL NOT NULL DEFAULT 0,
    daily_realized_pnl REAL NOT NULL DEFAULT 0,
    regime TEXT,
    vix_level REAL,
    peak_equity REAL NOT NULL DEFAULT 0,
    drawdown_pct REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_lifecycle_symbol ON position_lifecycle(symbol);
CREATE INDEX IF NOT EXISTS idx_lifecycle_strategy ON position_lifecycle(strategy);
CREATE INDEX IF NOT EXISTS idx_lifecycle_entry_date ON position_lifecycle(entry_date);
CREATE INDEX IF NOT EXISTS idx_lifecycle_exit_date ON position_lifecycle(exit_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp);
"""

# Migration: add slippage tracking columns (safe to run on existing DBs)
_SLIPPAGE_MIGRATION = """
-- Slippage tracking: intended vs actual fill price
ALTER TABLE position_lifecycle ADD COLUMN entry_fill_price REAL;
ALTER TABLE position_lifecycle ADD COLUMN entry_slippage_pct REAL;
ALTER TABLE position_lifecycle ADD COLUMN entry_order_type TEXT DEFAULT 'market';
ALTER TABLE position_lifecycle ADD COLUMN exit_fill_price REAL;
ALTER TABLE position_lifecycle ADD COLUMN exit_slippage_pct REAL;
"""


@dataclass
class TradeEntry:
    """Data needed to log a new position entry."""
    symbol: str
    direction: str  # "long" or "short"
    strategy: str
    conviction_tier: int
    confluence_count: int
    entry_price: float  # signal's intended entry price
    entry_qty: int
    entry_order_id: str | None = None
    entry_regime: str | None = None
    entry_vix: float | None = None
    entry_signal_strength: float | None = None
    features: dict | None = None
    # Slippage tracking
    entry_fill_price: float | None = None  # actual fill price from broker
    entry_slippage_pct: float | None = None  # (fill - intended) / intended * 100
    entry_order_type: str = "market"  # "market" or "limit"


@dataclass
class TradeExit:
    """Data needed to log a position exit."""
    symbol: str
    exit_price: float
    exit_qty: int
    exit_reason: str
    exit_order_id: str | None = None


class TradeLogger:

    def __init__(self, db_path: str = DEFAULT_TRADE_LOG_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_schema(self):
        self.conn.executescript(_TRADE_LOG_SCHEMA)
        self.conn.commit()
        # Apply slippage migration (safe to run multiple times)
        for line in _SLIPPAGE_MIGRATION.strip().split("\n"):
            line = line.strip()
            if line.startswith("ALTER"):
                try:
                    self.conn.execute(line)
                except sqlite3.OperationalError:
                    pass  # column already exists
        self.conn.commit()

    # ------------------------------------------------------------------
    # Entry logging
    # ------------------------------------------------------------------

    def log_entry(self, entry: TradeEntry) -> int:
        """Log a new position entry. Returns the lifecycle ID."""
        features_json = json.dumps(entry.features or {})
        cur = self.conn.execute(
            """INSERT INTO position_lifecycle (
                symbol, direction, strategy, conviction_tier, confluence_count,
                entry_date, entry_price, entry_qty, entry_order_id,
                entry_regime, entry_vix, entry_signal_strength, features_json,
                entry_fill_price, entry_slippage_pct, entry_order_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.symbol, entry.direction, entry.strategy,
                entry.conviction_tier, entry.confluence_count,
                datetime.now().strftime("%Y-%m-%d"),
                entry.entry_price, entry.entry_qty,
                entry.entry_order_id,
                entry.entry_regime, entry.entry_vix,
                entry.entry_signal_strength,
                features_json,
                entry.entry_fill_price,
                entry.entry_slippage_pct,
                entry.entry_order_type,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Exit logging
    # ------------------------------------------------------------------

    def log_exit(self, exit_data: TradeExit) -> int | None:
        """Log a position exit and compute P&L.

        Finds the open lifecycle record for this symbol, fills in exit data,
        and computes realized P&L.

        For partial exits, creates a new lifecycle row linked to the parent.

        Returns the lifecycle ID, or None if no matching entry found.
        """
        # Find the open entry for this symbol
        cur = self.conn.execute(
            """SELECT * FROM position_lifecycle
            WHERE symbol = ? AND exit_date IS NULL AND is_partial_close = 0
            ORDER BY entry_date DESC LIMIT 1""",
            (exit_data.symbol,),
        )
        row = cur.fetchone()
        if not row:
            return None

        row_dict = dict(row)
        lifecycle_id = row_dict["id"]
        entry_price = row_dict["entry_price"]
        direction = row_dict["direction"]
        entry_qty = row_dict["entry_qty"]
        entry_date = row_dict["entry_date"]

        # Compute P&L
        if direction == "long":
            pnl_per_share = exit_data.exit_price - entry_price
        else:  # short
            pnl_per_share = entry_price - exit_data.exit_price

        realized_pnl = pnl_per_share * exit_data.exit_qty
        realized_pnl_pct = pnl_per_share / entry_price if entry_price > 0 else 0.0

        # Compute hold days
        try:
            entry_dt = date.fromisoformat(entry_date)
            hold_days = (date.today() - entry_dt).days
        except Exception:
            hold_days = 0

        is_partial = exit_data.exit_qty < entry_qty

        if is_partial:
            # Create a separate partial-exit record linked to parent
            cur2 = self.conn.execute(
                """INSERT INTO position_lifecycle (
                    symbol, direction, strategy, conviction_tier, confluence_count,
                    entry_date, entry_price, entry_qty, entry_order_id,
                    entry_regime, entry_vix, entry_signal_strength,
                    exit_date, exit_price, exit_qty, exit_reason, exit_order_id,
                    realized_pnl, realized_pnl_pct, hold_days,
                    is_partial_close, parent_lifecycle_id, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (
                    row_dict["symbol"], direction, row_dict["strategy"],
                    row_dict["conviction_tier"], row_dict["confluence_count"],
                    entry_date, entry_price, entry_qty,
                    row_dict["entry_order_id"],
                    row_dict["entry_regime"], row_dict["entry_vix"],
                    row_dict["entry_signal_strength"],
                    datetime.now().strftime("%Y-%m-%d"),
                    exit_data.exit_price, exit_data.exit_qty,
                    exit_data.exit_reason, exit_data.exit_order_id,
                    round(realized_pnl, 2), round(realized_pnl_pct, 4),
                    hold_days,
                    lifecycle_id,
                    row_dict["features_json"],
                ),
            )
            self.conn.commit()
            return cur2.lastrowid
        else:
            # Full close — update the original record
            self.conn.execute(
                """UPDATE position_lifecycle SET
                    exit_date = ?, exit_price = ?, exit_qty = ?,
                    exit_reason = ?, exit_order_id = ?,
                    realized_pnl = ?, realized_pnl_pct = ?, hold_days = ?,
                    updated_at = datetime('now')
                WHERE id = ?""",
                (
                    datetime.now().strftime("%Y-%m-%d"),
                    exit_data.exit_price, exit_data.exit_qty,
                    exit_data.exit_reason, exit_data.exit_order_id,
                    round(realized_pnl, 2), round(realized_pnl_pct, 4),
                    hold_days,
                    lifecycle_id,
                ),
            )
            self.conn.commit()
            return lifecycle_id

    # ------------------------------------------------------------------
    # Portfolio snapshots
    # ------------------------------------------------------------------

    def log_snapshot(
        self,
        equity: float,
        cash: float,
        num_positions: int = 0,
        num_long: int = 0,
        num_short: int = 0,
        total_unrealized_pnl: float = 0.0,
        daily_realized_pnl: float = 0.0,
        regime: str | None = None,
        vix_level: float | None = None,
    ) -> None:
        """Log a portfolio snapshot for drawdown tracking."""
        # Get peak equity
        cur = self.conn.execute(
            "SELECT MAX(equity) as peak FROM portfolio_snapshots"
        )
        row = cur.fetchone()
        peak = row["peak"] if row and row["peak"] else equity
        peak = max(peak, equity)

        drawdown_pct = (equity - peak) / peak if peak > 0 else 0.0

        self.conn.execute(
            """INSERT INTO portfolio_snapshots (
                timestamp, equity, cash, num_positions, num_long, num_short,
                total_unrealized_pnl, daily_realized_pnl, regime, vix_level,
                peak_equity, drawdown_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                equity, cash, num_positions, num_long, num_short,
                total_unrealized_pnl, daily_realized_pnl,
                regime, vix_level,
                peak, round(drawdown_pct, 4),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Query helpers for performance reporting
    # ------------------------------------------------------------------

    def get_closed_trades(
        self,
        strategy: str | None = None,
        direction: str | None = None,
        min_date: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Get closed trades with optional filters."""
        query = "SELECT * FROM position_lifecycle WHERE exit_date IS NOT NULL"
        params: list = []

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if direction:
            query += " AND direction = ?"
            params.append(direction)
        if min_date:
            query += " AND exit_date >= ?"
            params.append(min_date)

        query += " ORDER BY exit_date DESC LIMIT ?"
        params.append(limit)

        cur = self.conn.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def get_open_entries(self) -> list[dict]:
        """Get all open positions (entries without exits)."""
        cur = self.conn.execute(
            """SELECT * FROM position_lifecycle
            WHERE exit_date IS NULL AND is_partial_close = 0
            ORDER BY entry_date DESC"""
        )
        return [dict(row) for row in cur.fetchall()]

    def get_strategy_summary(self, min_date: str | None = None) -> list[dict]:
        """Get per-strategy performance summary."""
        where = "WHERE exit_date IS NOT NULL"
        params: list = []
        if min_date:
            where += " AND exit_date >= ?"
            params.append(min_date)

        cur = self.conn.execute(
            f"""SELECT
                strategy,
                direction,
                conviction_tier,
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses,
                ROUND(SUM(realized_pnl), 2) as total_pnl,
                ROUND(AVG(realized_pnl), 2) as avg_pnl,
                ROUND(AVG(realized_pnl_pct) * 100, 2) as avg_pnl_pct,
                ROUND(AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl_pct ELSE NULL END) * 100, 2) as avg_win_pct,
                ROUND(AVG(CASE WHEN realized_pnl <= 0 THEN realized_pnl_pct ELSE NULL END) * 100, 2) as avg_loss_pct,
                ROUND(AVG(hold_days), 1) as avg_hold_days,
                MIN(entry_date) as first_trade,
                MAX(exit_date) as last_trade
            FROM position_lifecycle
            {where}
            GROUP BY strategy, direction, conviction_tier
            ORDER BY strategy, direction, conviction_tier""",
            params,
        )
        return [dict(row) for row in cur.fetchall()]

    def get_regime_summary(self, min_date: str | None = None) -> list[dict]:
        """Get performance by entry regime."""
        where = "WHERE exit_date IS NOT NULL AND entry_regime IS NOT NULL"
        params: list = []
        if min_date:
            where += " AND exit_date >= ?"
            params.append(min_date)

        cur = self.conn.execute(
            f"""SELECT
                entry_regime,
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(realized_pnl), 2) as total_pnl,
                ROUND(AVG(realized_pnl_pct) * 100, 2) as avg_pnl_pct
            FROM position_lifecycle
            {where}
            GROUP BY entry_regime
            ORDER BY total_pnl DESC""",
            params,
        )
        return [dict(row) for row in cur.fetchall()]

    def get_exit_reason_summary(self, min_date: str | None = None) -> list[dict]:
        """Get performance by exit reason."""
        where = "WHERE exit_date IS NOT NULL AND exit_reason IS NOT NULL"
        params: list = []
        if min_date:
            where += " AND exit_date >= ?"
            params.append(min_date)

        cur = self.conn.execute(
            f"""SELECT
                CASE
                    WHEN exit_reason LIKE '%HARD STOP%' THEN 'hard_stop'
                    WHEN exit_reason LIKE '%ATR TRAILING%' THEN 'atr_trailing'
                    WHEN exit_reason LIKE '%TRAILING STOP%' THEN 'trailing_stop'
                    WHEN exit_reason LIKE '%BB MIDDLE%' THEN 'bb_middle'
                    WHEN exit_reason LIKE '%FIRST TARGET%' THEN 'first_target'
                    WHEN exit_reason LIKE '%SECOND TARGET%' THEN 'second_target'
                    WHEN exit_reason LIKE '%TIME EXIT%' THEN 'time_exit'
                    ELSE 'other'
                END as exit_type,
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(realized_pnl), 2) as total_pnl,
                ROUND(AVG(realized_pnl_pct) * 100, 2) as avg_pnl_pct
            FROM position_lifecycle
            {where}
            GROUP BY exit_type
            ORDER BY total_pnl DESC""",
            params,
        )
        return [dict(row) for row in cur.fetchall()]

    def get_daily_pnl(self, days: int = 30) -> list[dict]:
        """Get daily realized P&L for the last N days."""
        cur = self.conn.execute(
            """SELECT
                exit_date,
                COUNT(*) as trades_closed,
                ROUND(SUM(realized_pnl), 2) as daily_pnl,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses
            FROM position_lifecycle
            WHERE exit_date IS NOT NULL
            GROUP BY exit_date
            ORDER BY exit_date DESC
            LIMIT ?""",
            (days,),
        )
        return [dict(row) for row in cur.fetchall()]

    def has_entry_today(self, symbol: str, direction: str) -> bool:
        """Check if we already logged an entry for this symbol/direction today."""
        today = date.today().isoformat()
        cur = self.conn.execute(
            """SELECT 1 FROM position_lifecycle
            WHERE symbol = ? AND direction = ? AND entry_date = ?
            AND is_partial_close = 0
            LIMIT 1""",
            (symbol, direction, today),
        )
        return cur.fetchone() is not None

    def get_slippage_summary(self, min_date: str | None = None) -> list[dict]:
        """Get slippage stats grouped by strategy and order type."""
        where = "WHERE entry_slippage_pct IS NOT NULL"
        params: list = []
        if min_date:
            where += " AND entry_date >= ?"
            params.append(min_date)

        cur = self.conn.execute(
            f"""SELECT
                strategy,
                entry_order_type,
                COUNT(*) as total_trades,
                ROUND(AVG(entry_slippage_pct), 4) as avg_slippage_pct,
                ROUND(MAX(entry_slippage_pct), 4) as max_slippage_pct,
                ROUND(MIN(entry_slippage_pct), 4) as min_slippage_pct,
                ROUND(SUM(CASE WHEN entry_slippage_pct < 0
                    THEN ABS(entry_slippage_pct) * entry_price * entry_qty / 100
                    ELSE 0 END), 2) as total_saved_dollars
            FROM position_lifecycle
            {where}
            GROUP BY strategy, entry_order_type
            ORDER BY strategy, entry_order_type""",
            params,
        )
        return [dict(row) for row in cur.fetchall()]

    def get_todays_entries(self) -> set[str]:
        """Get symbols that had entries logged today (for dedup)."""
        today = date.today().isoformat()
        cur = self.conn.execute(
            """SELECT DISTINCT symbol FROM position_lifecycle
            WHERE entry_date = ? AND is_partial_close = 0""",
            (today,),
        )
        return {row[0] for row in cur.fetchall()}

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
