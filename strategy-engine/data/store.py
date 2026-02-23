"""Data Persistence Layer.

Handles reading/writing to Postgres for trade logging,
equity curves, signals, and regime classifications.
"""

import json
import os
import sys
from datetime import datetime

import psycopg2
import psycopg2.extras


def get_connection():
    """Get a Postgres connection from DATABASE_URL env var."""
    url = os.environ.get("DATABASE_URL", "postgresql://alpaca:alpaca@localhost:5432/alpaca_trading")
    return psycopg2.connect(url)


class DataStore:

    def __init__(self):
        self._conn = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = get_connection()
        return self._conn

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
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO trades (
                order_id, symbol, side, qty, price, filled_price,
                order_type, status, strategy_id, signal_id,
                signal_strength, regime, features_json, rationale,
                risk_check_json, submitted_at, filled_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id""",
            (
                order_id, symbol, side, qty, price, filled_price,
                order_type, status, strategy_id, signal_id,
                signal_strength, regime, json.dumps(features), rationale,
                json.dumps(risk_check), submitted_at, filled_at,
            ),
        )
        trade_id = cur.fetchone()[0]
        self.conn.commit()
        return trade_id

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
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO signals (timestamp, symbol, strategy_id, direction, strength, features_json)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (timestamp, symbol, strategy_id, direction, strength, json.dumps(features)),
        )
        signal_id = cur.fetchone()[0]
        self.conn.commit()
        return signal_id

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
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO equity_curve (
                timestamp, equity, cash, long_market_value, short_market_value,
                daily_pnl, cumulative_pnl, drawdown_pct, peak_equity, num_positions
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                timestamp, equity, cash, long_market_value, short_market_value,
                daily_pnl, cumulative_pnl, drawdown_pct, peak_equity, num_positions,
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
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO regimes (timestamp, regime_type, confidence, indicators_json)
            VALUES (%s, %s, %s, %s)""",
            (timestamp, regime_type, confidence, json.dumps(indicators)),
        )
        self.conn.commit()

    def get_or_create_strategy(self, name: str, strategy_type: str, params: dict) -> int:
        """Get strategy ID or create if not exists."""
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM strategies WHERE name = %s", (name,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            """INSERT INTO strategies (name, type, params_json) VALUES (%s, %s, %s) RETURNING id""",
            (name, strategy_type, json.dumps(params)),
        )
        strategy_id = cur.fetchone()[0]
        self.conn.commit()
        return strategy_id

    def get_recent_equity(self, limit: int = 30) -> list[dict]:
        """Get recent equity curve data points."""
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM equity_curve ORDER BY timestamp DESC LIMIT %s",
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
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if strategy_id is not None:
            query += " AND strategy_id = %s"
            params.append(strategy_id)
        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)
        query += " ORDER BY submitted_at DESC LIMIT %s"
        params.append(limit)
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
