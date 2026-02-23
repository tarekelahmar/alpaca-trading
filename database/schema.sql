CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    params_json JSONB NOT NULL DEFAULT '{}',
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS regimes (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    regime_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    indicators_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_regimes_timestamp ON regimes (timestamp DESC);

CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short', 'close')),
    strength REAL NOT NULL,
    features_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_signals_timestamp ON signals (timestamp DESC);
CREATE INDEX idx_signals_symbol ON signals (symbol);
CREATE INDEX idx_signals_strategy ON signals (strategy_id);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    qty REAL NOT NULL,
    price REAL,
    filled_price REAL,
    order_type TEXT NOT NULL,
    status TEXT NOT NULL,
    strategy_id INTEGER REFERENCES strategies(id),
    signal_id INTEGER REFERENCES signals(id),
    signal_strength REAL,
    regime TEXT,
    features_json JSONB NOT NULL DEFAULT '{}',
    rationale TEXT,
    risk_check_json JSONB NOT NULL DEFAULT '{}',
    submitted_at TIMESTAMPTZ NOT NULL,
    filled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_trades_symbol ON trades (symbol);
CREATE INDEX idx_trades_submitted ON trades (submitted_at DESC);
CREATE INDEX idx_trades_strategy ON trades (strategy_id);

CREATE TABLE IF NOT EXISTS equity_curve (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    long_market_value REAL NOT NULL DEFAULT 0,
    short_market_value REAL NOT NULL DEFAULT 0,
    daily_pnl REAL NOT NULL DEFAULT 0,
    cumulative_pnl REAL NOT NULL DEFAULT 0,
    drawdown_pct REAL NOT NULL DEFAULT 0,
    peak_equity REAL NOT NULL,
    num_positions INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_equity_curve_timestamp ON equity_curve (timestamp DESC);

CREATE TABLE IF NOT EXISTS strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    date DATE NOT NULL,
    num_trades INTEGER NOT NULL DEFAULT 0,
    num_wins INTEGER NOT NULL DEFAULT 0,
    num_losses INTEGER NOT NULL DEFAULT 0,
    gross_pnl REAL NOT NULL DEFAULT 0,
    net_pnl REAL NOT NULL DEFAULT 0,
    max_drawdown REAL NOT NULL DEFAULT 0,
    sharpe_ratio REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (strategy_id, date)
);

CREATE INDEX idx_strat_perf_strategy ON strategy_performance (strategy_id);
CREATE INDEX idx_strat_perf_date ON strategy_performance (date DESC);
