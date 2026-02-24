"""Data Persistence Layer.

Handles reading/writing to SQLite for trade logging,
equity curves, signals, and regime classifications.

Database file: /opt/alpaca-trading/data/trades.db (configurable via DB_PATH env var)
Tables are auto-created on first connection.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path


DEFAULT_DB_PATH = os.environ.get(
    "DB_PATH",
    "/opt/alpaca-trading/data/trades.db",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL DEFAULT '{}',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS regimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    regime_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    indicators_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short', 'close')),
    strength REAL NOT NULL,
    features_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    qty REAL NOT NULL,
    price REAL,
    filled_price REAL,
    order_type TEXT NOT NULL,
    status TEXT NOT NULL,
    strategy_id INTEGER REFERENCES strategies(id),
    signal_id INTEGER,
    signal_strength REAL,
    regime TEXT,
    features_json TEXT NOT NULL DEFAULT '{}',
    rationale TEXT,
    risk_check_json TEXT NOT NULL DEFAULT '{}',
    submitted_at TEXT NOT NULL,
    filled_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    long_market_value REAL NOT NULL DEFAULT 0,
    short_market_value REAL NOT NULL DEFAULT 0,
    daily_pnl REAL NOT NULL DEFAULT 0,
    cumulative_pnl REAL NOT NULL DEFAULT 0,
    drawdown_pct REAL NOT NULL DEFAULT 0,
    peak_equity REAL NOT NULL,
    num_positions INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    date TEXT NOT NULL,
    num_trades INTEGER NOT NULL DEFAULT 0,
    num_wins INTEGER NOT NULL DEFAULT 0,
    num_losses INTEGER NOT NULL DEFAULT 0,
    gross_pnl REAL NOT NULL DEFAULT 0,
    net_pnl REAL NOT NULL DEFAULT 0,
    max_drawdown REAL NOT NULL DEFAULT 0,
    sharpe_ratio REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (strategy_id, date)
);
"""


class DataStore:

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
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
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def log_trade(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float | None,
        filled_price: float | None,
        order_type: str,
        status: str,
        strategy_id: int | None,
        signal_id: int | None,
        signal_strength: float | None,
        regime: str | None,
        features: dict,
        rationale: str | None,
        risk_check: dict,
        submitted_at: datetime,
        filled_at: datetime | None = None,
    ) -> int:
        """Log a trade to the database. Returns the trade ID."""
        cur = self.conn.execute(
            """INSERT INTO trades (
                order_id, symbol, side, qty, price, filled_price,
                order_type, status, strategy_id, signal_id,
                signal_strength, regime, features_json, rationale,
                risk_check_json, submitted_at, filled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                order_id, symbol, side, qty, price, filled_price,
                order_type, status, strategy_id, signal_id,
                signal_strength, regime, json.dumps(features), rationale,
                json.dumps(risk_check), submitted_at.isoformat(),
                filled_at.isoformat() if filled_at else None,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_signal(
        self,
        timestamp: datetime,
        symbol: str,
        strategy_id: int,
        direction: str,
        strength: float,
        features: dict,
    ) -> int:
        """Log a signal to the database. Returns the signal ID."""
        cur = self.conn.execute(
            """INSERT INTO signals (timestamp, symbol, strategy_id, direction, strength, features_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (timestamp.isoformat(), symbol, strategy_id, direction, strength, json.dumps(features)),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_equity(
        self,
        timestamp: datetime,
        equity: float,
        cash: float,
        long_market_value: float = 0,
        short_market_value: float = 0,
        daily_pnl: float = 0,
        cumulative_pnl: float = 0,
        drawdown_pct: float = 0,
        peak_equity: float = 0,
        num_positions: int = 0,
    ) -> None:
        """Log an equity curve data point."""
        self.conn.execute(
            """INSERT INTO equity_curve (
                timestamp, equity, cash, long_market_value, short_market_value,
                daily_pnl, cumulative_pnl, drawdown_pct, peak_equity, num_positions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp.isoformat(), equity, cash, long_market_value,
                short_market_value, daily_pnl, cumulative_pnl,
                drawdown_pct, peak_equity, num_positions,
            ),
        )
        self.conn.commit()

    def log_regime(
        self,
        timestamp: datetime,
        regime_type: str,
        confidence: float,
        indicators: dict,
    ) -> None:
        """Log a regime classification."""
        self.conn.execute(
            """INSERT INTO regimes (timestamp, regime_type, confidence, indicators_json)
            VALUES (?, ?, ?, ?)""",
            (timestamp.isoformat(), regime_type, confidence, json.dumps(indicators)),
        )
        self.conn.commit()

    def get_or_create_strategy(self, name: str, strategy_type: str, params: dict) -> int:
        """Get strategy ID or create if not exists."""
        cur = self.conn.execute(
            "SELECT id FROM strategies WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur = self.conn.execute(
            """INSERT INTO strategies (name, type, params_json) VALUES (?, ?, ?)""",
            (name, strategy_type, json.dumps(params)),
        )
        self.conn.commit()
        return cur.lastrowid

    def has_order(self, order_id: str) -> bool:
        """Check if an order ID already exists in the database."""
        cur = self.conn.execute(
            "SELECT 1 FROM trades WHERE order_id = ?", (order_id,)
        )
        return cur.fetchone() is not None

    def get_todays_order_symbols(self, side: str = "buy") -> set[str]:
        """Get symbols that already had orders submitted today.

        Used for crash-restart dedup: if we already bought AAPL today,
        don't buy it again on a restart.
        """
        today = date.today().isoformat()
        cur = self.conn.execute(
            """SELECT DISTINCT symbol FROM trades
            WHERE side = ? AND submitted_at >= ?""",
            (side, today),
        )
        return {row[0] for row in cur.fetchall()}

    def get_recent_equity(self, limit: int = 30) -> list[dict]:
        """Get recent equity curve data points."""
        cur = self.conn.execute(
            "SELECT * FROM equity_curve ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_trades(
        self,
        strategy_id: int | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get recent trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if strategy_id is not None:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY submitted_at DESC LIMIT ?"
        params.append(limit)
        cur = self.conn.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
