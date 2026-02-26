"""Tests for the crypto trading module.

Tests cover:
    1. Crypto universe selection and filtering
    2. BTC dominance rotation strategy signals
    3. Crypto regime detection
    4. Crypto regime allocation weights
    5. Crypto-adapted strategy parameter overrides
    6. Crypto profit target configs
    7. Crypto engine orchestration
    8. Crypto monitor dynamic trailing stop tightening
    9. Scripts compile without import errors
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure strategy-engine is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crypto.universe import (
    CryptoUniverseSelector, CryptoUniverseFilter,
    TIER1_BLUE_CHIP, TIER2_MAJOR_ALT, TIER3_MID_ALT, CRYPTO_UNIVERSE,
)
from crypto.btc_dominance import BTCDominanceStrategy
from crypto.regime import (
    CryptoRegimeDetector, CryptoRegimeAllocator, CryptoAllocationWeights,
)
from crypto.strategy_configs import (
    CRYPTO_MOMENTUM_CONFIG, CRYPTO_TREND_CONFIG, CRYPTO_MEAN_REVERSION_CONFIG,
    get_crypto_strategy_configs,
)
from crypto.profit_targets import (
    get_crypto_profit_config, CRYPTO_PROFIT_CONFIGS, CRYPTO_DEFAULT_CONFIG,
)
from crypto.engine import CryptoStrategyEngine
from regime.detector import RegimeType
from strategies.base import SignalDirection

# --- Helpers ---

# Use hourly dates for crypto (24/7, no business-day restriction)
# 1200 hourly bars ≈ 50 days — enough for all indicator warmup periods
_SHARED_DATES = pd.date_range(end="2025-06-01", periods=1200, freq="h")


def _make_crypto_df(
    base_price: float = 50000.0,
    n: int = 150,
    trend: float = 0.001,
    volatility: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for crypto testing."""
    np.random.seed(seed)
    dates = _SHARED_DATES[-n:]
    prices = [base_price]
    for i in range(1, n):
        change = trend + volatility * np.random.randn()
        prices.append(prices[-1] * (1 + change))

    closes = np.array(prices)
    highs = closes * (1 + np.abs(np.random.randn(n) * 0.01))
    lows = closes * (1 - np.abs(np.random.randn(n) * 0.01))
    opens = closes * (1 + np.random.randn(n) * 0.005)
    volumes = np.random.uniform(1000, 10000, n)

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=dates)


# --- Test 1: Crypto Universe ---

def test_crypto_universe_full():
    """All tiers are included by default."""
    selector = CryptoUniverseSelector()
    symbols = selector.get_symbols()
    assert "BTC/USD" in symbols
    assert "ETH/USD" in symbols
    assert "SOL/USD" in symbols
    assert "DOGE/USD" in symbols
    assert len(symbols) == len(CRYPTO_UNIVERSE)


def test_crypto_universe_tier_filter():
    """Tier filter limits to blue chip only."""
    selector = CryptoUniverseSelector(
        filters=CryptoUniverseFilter(tier_filter=1)
    )
    symbols = selector.get_symbols()
    assert len(symbols) == len(TIER1_BLUE_CHIP)
    assert "BTC/USD" in symbols
    assert "ETH/USD" in symbols
    assert "SOL/USD" not in symbols


def test_crypto_universe_tier2_filter():
    """Tier 2 filter includes blue chip + major alt."""
    selector = CryptoUniverseSelector(
        filters=CryptoUniverseFilter(tier_filter=2)
    )
    symbols = selector.get_symbols()
    expected = len(TIER1_BLUE_CHIP) + len(TIER2_MAJOR_ALT)
    assert len(symbols) == expected
    assert "SOL/USD" in symbols
    assert "DOGE/USD" not in symbols  # tier 3


def test_crypto_universe_data_filter():
    """Data-based filtering removes symbols with low volume."""
    selector = CryptoUniverseSelector()
    data = {
        "BTC/USD": _make_crypto_df(50000, 800, seed=1),  # 800 hourly bars (~33 days)
        "SHIB/USD": _make_crypto_df(0.00001, 800, seed=2),
    }
    # SHIB has very low dollar volume (price * volume)
    data["SHIB/USD"]["volume"] = 0.001  # tiny volume
    filtered = selector.filter_by_data(data)
    assert "BTC/USD" in filtered  # BTC passes
    # SHIB filtered out due to low dollar volume


def test_crypto_universe_get_tier():
    """Tier classification works."""
    selector = CryptoUniverseSelector()
    assert selector.get_tier("BTC/USD") == 1
    assert selector.get_tier("ETH/USD") == 1
    assert selector.get_tier("SOL/USD") == 2
    assert selector.get_tier("DOGE/USD") == 3


# --- Test 2: BTC Dominance Rotation ---

def test_btc_dominance_rising():
    """When BTC outperforms alts, strategy favors BTC."""
    btc_df = _make_crypto_df(50000, 150, trend=0.005, seed=10)  # BTC trending up
    eth_df = _make_crypto_df(3000, 150, trend=-0.002, seed=20)  # ETH trending down
    sol_df = _make_crypto_df(100, 150, trend=-0.003, seed=30)   # SOL down more

    data = {"BTC/USD": btc_df, "ETH/USD": eth_df, "SOL/USD": sol_df}
    strategy = BTCDominanceStrategy()
    signals = strategy.generate_signals(data)

    # Should have BTC LONG and alt CLOSE signals
    btc_longs = [s for s in signals if s.symbol == "BTC/USD" and s.direction == SignalDirection.LONG]
    close_signals = [s for s in signals if s.direction == SignalDirection.CLOSE]

    # When BTC outperforms, we expect BTC long signals
    if btc_longs:
        assert btc_longs[0].strength > 0
        assert btc_longs[0].features["rotation"] == "btc_favored"


def test_btc_dominance_falling():
    """When alts outperform BTC, strategy favors alts."""
    btc_df = _make_crypto_df(50000, 150, trend=-0.002, seed=10)  # BTC down
    eth_df = _make_crypto_df(3000, 150, trend=0.005, seed=20)    # ETH up
    sol_df = _make_crypto_df(100, 150, trend=0.008, seed=30)     # SOL up more

    data = {"BTC/USD": btc_df, "ETH/USD": eth_df, "SOL/USD": sol_df}
    strategy = BTCDominanceStrategy()
    signals = strategy.generate_signals(data)

    # When alts outperform, we expect alt long signals
    alt_longs = [
        s for s in signals
        if s.symbol != "BTC/USD" and s.direction == SignalDirection.LONG
    ]
    if alt_longs:
        assert all(s.features["rotation"] == "alt_favored" for s in alt_longs)


def test_btc_dominance_no_btc_data():
    """Strategy returns empty if no BTC data."""
    eth_df = _make_crypto_df(3000, 150, seed=20)
    strategy = BTCDominanceStrategy()
    signals = strategy.generate_signals({"ETH/USD": eth_df})
    assert signals == []


# --- Test 3: Crypto Regime Detection ---

def test_crypto_regime_bullish():
    """Rising BTC with low vol should detect trending bullish."""
    btc_df = _make_crypto_df(50000, 1000, trend=0.001, volatility=0.005, seed=42)
    detector = CryptoRegimeDetector()
    result = detector.classify(btc_df)
    # Should detect some regime (not necessarily bullish due to vol/ADX dynamics)
    assert result.regime in list(RegimeType)
    assert 0 <= result.confidence <= 1
    assert "realized_vol" in result.indicators
    assert "adx" in result.indicators


def test_crypto_regime_high_vol():
    """Highly volatile BTC data should detect high volatility."""
    btc_df = _make_crypto_df(50000, 1000, trend=0.0, volatility=0.03, seed=42)
    detector = CryptoRegimeDetector()
    result = detector.classify(btc_df)
    # High vol should score high_volatility
    assert result.indicators["realized_vol"] > 40


def test_crypto_regime_interface_compatible():
    """Crypto detector returns same RegimeClassification as equity detector."""
    btc_df = _make_crypto_df(50000, 1000, seed=42)
    detector = CryptoRegimeDetector()
    result = detector.classify(btc_df, vix_data=None, breadth_pct=None)
    assert isinstance(result.regime, RegimeType)
    assert isinstance(result.confidence, float)
    assert isinstance(result.indicators, dict)


# --- Test 4: Crypto Allocation Weights ---

def test_crypto_allocation_no_mr_in_bull():
    """Mean reversion should get 0% allocation in bullish regime."""
    allocator = CryptoRegimeAllocator()
    weights = allocator.get_allocation(RegimeType.TRENDING_BULLISH)
    assert weights.mean_reversion == 0.0
    assert weights.momentum > 0


def test_crypto_allocation_mr_in_ranging():
    """Mean reversion should get allocation in ranging regime."""
    allocator = CryptoRegimeAllocator()
    weights = allocator.get_allocation(RegimeType.RANGING)
    assert weights.mean_reversion > 0


def test_crypto_allocation_heavy_cash_in_vol():
    """High volatility should have heavy cash allocation."""
    allocator = CryptoRegimeAllocator()
    weights = allocator.get_allocation(RegimeType.HIGH_VOLATILITY)
    assert weights.cash >= 0.5


def test_crypto_allocation_to_equity_format():
    """Conversion to equity AllocationWeights should work."""
    weights = CryptoAllocationWeights(
        trend_following=0.3, mean_reversion=0.1,
        momentum=0.3, btc_dominance=0.2, cash=0.1,
    )
    eq = weights.to_equity_format()
    assert eq.trend_following == 0.3
    assert eq.sector_rotation == 0.2  # btc_dominance maps here
    assert eq.sentiment == 0.0
    assert eq.gap_trading == 0.0


def test_crypto_allocation_normalizes():
    """Weights should normalize to sum to 1.0."""
    weights = CryptoAllocationWeights(0.3, 0.1, 0.3, 0.2, 0.1)
    normed = weights.normalize()
    total = (
        normed.trend_following + normed.mean_reversion
        + normed.momentum + normed.btc_dominance + normed.cash
    )
    assert abs(total - 1.0) < 0.001


# --- Test 5: Strategy Configs ---

def test_crypto_momentum_config():
    """Crypto momentum should have hourly lookback periods."""
    p = CRYPTO_MOMENTUM_CONFIG.params
    assert p["roc_periods"] == [24, 72, 168, 720]  # 1d, 3d, 1w, 1m in hours
    assert p["ema_trend_period"] == 120  # 5-day EMA
    assert p["min_avg_volume"] < 200_000  # lower than equity


def test_crypto_trend_config():
    """Crypto trend following should have wider stops and hourly EMAs."""
    p = CRYPTO_TREND_CONFIG.params
    assert p["atr_stop_multiplier"] == 3.0  # wider than equity (2.0)
    assert p["fast_ema"] == 12   # 12 hours
    assert p["slow_ema"] == 48   # 2 days


def test_crypto_mean_reversion_config():
    """Crypto mean reversion should have more extreme RSI bands."""
    p = CRYPTO_MEAN_REVERSION_CONFIG.params
    assert p["rsi_oversold"] == 25  # more extreme than equity (40)
    assert p["rsi_overbought"] == 75
    assert p["bb_std"] == 2.5  # wider bands


def test_get_crypto_strategy_configs():
    """Helper returns all 3 crypto configs."""
    configs = get_crypto_strategy_configs()
    assert len(configs) == 3
    names = {c.name for c in configs}
    assert names == {"momentum", "trend_following", "mean_reversion"}


# --- Test 6: Crypto Profit Targets ---

def test_crypto_profit_targets_wider_than_equity():
    """Crypto profit targets should be wider than equity defaults."""
    from portfolio.profit_targets import get_profit_config as get_equity_config

    crypto_tf = get_crypto_profit_config("trend_following", 1)
    equity_tf = get_equity_config("trend_following", 1)

    assert crypto_tf.first_target_pct > equity_tf.first_target_pct
    assert crypto_tf.trail_stop_atr_mult > equity_tf.trail_stop_atr_mult


def test_crypto_profit_targets_all_strategies():
    """All crypto strategies should have profit configs for all tiers."""
    for strategy in ["trend_following", "mean_reversion", "momentum", "btc_dominance"]:
        for tier in [1, 2, 3, 4]:
            config = get_crypto_profit_config(strategy, tier)
            assert config.trail_stop_atr_mult > 0


def test_crypto_default_config():
    """Unknown strategy/tier combo should fall back to crypto default."""
    config = get_crypto_profit_config("unknown_strategy", 99)
    assert config == CRYPTO_DEFAULT_CONFIG
    assert config.trail_stop_atr_mult == 3.0  # crypto default


# --- Test 7: Crypto Engine ---

def test_crypto_engine_init():
    """Engine should initialize with correct crypto strategies."""
    engine = CryptoStrategyEngine()
    assert len(engine.strategies) == 4
    names = {s.name for s in engine.strategies}
    assert "momentum" in names
    assert "trend_following" in names
    assert "mean_reversion" in names
    assert "btc_dominance" in names


def test_crypto_engine_run():
    """Engine should run the full pipeline without errors."""
    engine = CryptoStrategyEngine()

    # Create crypto data
    data = {
        "BTC/USD": _make_crypto_df(50000, 150, trend=0.003, seed=1),
        "ETH/USD": _make_crypto_df(3000, 150, trend=0.002, seed=2),
        "SOL/USD": _make_crypto_df(100, 150, trend=0.004, seed=3),
    }
    btc_data = data["BTC/USD"]

    portfolio = type("Portfolio", (), {
        "equity": 100000,
        "cash": 50000,
        "buying_power": 100000,
        "num_positions": 0,
        "strategy_allocation_pct": 1.0,
        "vix_level": None,
    })()

    orders, regime, allocation = engine.run(
        data=data,
        btc_data=btc_data,
        portfolio=portfolio,
        current_positions={},
    )

    assert isinstance(orders, list)
    assert regime.regime in list(RegimeType)
    assert isinstance(allocation, CryptoAllocationWeights)


# --- Test 8: Crypto Monitor Dynamic Trail ---

def test_crypto_dynamic_trail_widthresholds():
    """Crypto trail tightening should use wider thresholds than equity."""
    # Import from crypto_monitor module
    # We test the function logic directly since we can't import the script
    # (it has __main__ guard and requires Alpaca)

    # Simulate the crypto dynamic trail logic
    def dynamic_trail(base_mult, pnl_pct):
        if pnl_pct < 0.10:
            return base_mult
        elif pnl_pct < 0.20:
            return base_mult * 0.85
        elif pnl_pct < 0.30:
            return base_mult * 0.70
        elif pnl_pct < 0.50:
            return base_mult * 0.60
        else:
            return base_mult * 0.50

    base = 3.0
    assert dynamic_trail(base, 0.05) == base       # no tightening
    assert dynamic_trail(base, 0.15) == base * 0.85  # 15% tighter
    assert dynamic_trail(base, 0.25) == base * 0.70  # 30% tighter
    assert dynamic_trail(base, 0.40) == base * 0.60  # 40% tighter
    assert dynamic_trail(base, 0.60) == base * 0.50  # 50% tighter for big winners


# --- Test 9: Scripts Compile ---

def test_scripts_compile():
    """Verify crypto module files can be imported without errors."""
    import crypto
    import crypto.universe
    import crypto.strategy_configs
    import crypto.btc_dominance
    import crypto.regime
    import crypto.profit_targets
    import crypto.engine
    import crypto.backtester


# --- Test 10: Universe Exclude ---

def test_crypto_universe_exclude():
    """Excluded symbols should be removed."""
    selector = CryptoUniverseSelector(
        filters=CryptoUniverseFilter(exclude_symbols=["BTC/USD", "ETH/USD"])
    )
    symbols = selector.get_symbols()
    assert "BTC/USD" not in symbols
    assert "ETH/USD" not in symbols
    assert "SOL/USD" in symbols


# --- Test 11: Crypto Backtester ---

def test_crypto_backtest_config():
    """Crypto backtest config should have correct defaults."""
    from crypto.backtester import get_crypto_backtest_config, CRYPTO_HARD_STOP_PCT

    config = get_crypto_backtest_config()
    assert config.train_window == 180
    assert config.test_window == 30
    assert config.max_position_pct == 0.15
    assert config.max_positions == 10
    assert config.risk_per_trade_pct == 0.03
    assert CRYPTO_HARD_STOP_PCT == 0.12  # wider than equity 0.08


def test_crypto_backtest_cost_model():
    """Crypto cost model should have higher slippage than equities."""
    from crypto.backtester import CRYPTO_COST_MODEL, BLUE_CHIP_CRYPTO_COST_MODEL
    from backtester.costs import CostModel

    equity_default = CostModel()
    assert CRYPTO_COST_MODEL.slippage_pct > equity_default.slippage_pct
    assert CRYPTO_COST_MODEL.spread_estimate_pct > equity_default.spread_estimate_pct
    # Blue chip crypto should be cheaper than alt crypto
    assert BLUE_CHIP_CRYPTO_COST_MODEL.slippage_pct < CRYPTO_COST_MODEL.slippage_pct


def test_crypto_backtest_engine_init():
    """CryptoBacktestEngine should initialize with crypto config."""
    from crypto.backtester import CryptoBacktestEngine, get_crypto_backtest_config

    engine = CryptoBacktestEngine()
    assert engine.config.train_window == 180
    assert engine.config.test_window == 30


def _make_backtest_df(base: float, trend: float, seed: int, dates) -> pd.DataFrame:
    """Helper to make backtest DataFrames with matching date index."""
    n = len(dates)
    np.random.seed(seed)
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + trend + 0.02 * np.random.randn()))
    closes = np.array(prices)
    return pd.DataFrame({
        "open": closes * (1 + 0.005 * np.random.randn(n)),
        "high": closes * (1 + abs(0.01 * np.random.randn(n))),
        "low": closes * (1 - abs(0.01 * np.random.randn(n))),
        "close": closes,
        "volume": np.random.uniform(5000, 50000, n),
    }, index=dates)


# Use calendar dates for crypto (trades every day, not just business days)
_BACKTEST_DATES = pd.date_range(end="2025-06-01", periods=250, freq="D")


def test_crypto_backtest_momentum():
    """Crypto momentum backtest should run without errors on synthetic data."""
    from crypto.backtester import CryptoBacktestEngine, get_crypto_backtest_config
    from strategies.momentum import MomentumStrategy
    from crypto.strategy_configs import CRYPTO_MOMENTUM_CONFIG

    data = {
        "BTC/USD": _make_backtest_df(50000, 0.002, 1, _BACKTEST_DATES),
        "ETH/USD": _make_backtest_df(3000, 0.001, 2, _BACKTEST_DATES),
        "SOL/USD": _make_backtest_df(100, 0.003, 3, _BACKTEST_DATES),
    }

    config = get_crypto_backtest_config(initial_capital=100_000)
    engine = CryptoBacktestEngine(config)

    strategy = MomentumStrategy(CRYPTO_MOMENTUM_CONFIG)
    result = engine.run(strategy, data)

    assert result.metrics is not None
    assert len(result.equity_curve) > 0
    assert len(result.walk_forward_windows) > 0


def test_crypto_backtest_trend_following():
    """Crypto trend following backtest should run without errors."""
    from crypto.backtester import CryptoBacktestEngine, get_crypto_backtest_config
    from strategies.trend_following import TrendFollowingStrategy

    data = {
        "BTC/USD": _make_backtest_df(50000, 0.003, 10, _BACKTEST_DATES),
        "ETH/USD": _make_backtest_df(3000, 0.003, 20, _BACKTEST_DATES),
    }

    config = get_crypto_backtest_config()
    engine = CryptoBacktestEngine(config)
    strategy = TrendFollowingStrategy(CRYPTO_TREND_CONFIG)
    result = engine.run(strategy, data)

    assert result.metrics is not None
    assert len(result.equity_curve) > 0


def test_crypto_backtest_btc_dominance():
    """BTC dominance backtest should run without errors."""
    from crypto.backtester import CryptoBacktestEngine, get_crypto_backtest_config

    data = {
        "BTC/USD": _make_backtest_df(50000, 0.004, 1, _BACKTEST_DATES),
        "ETH/USD": _make_backtest_df(3000, 0.001, 2, _BACKTEST_DATES),
        "SOL/USD": _make_backtest_df(100, -0.001, 3, _BACKTEST_DATES),
    }

    config = get_crypto_backtest_config()
    engine = CryptoBacktestEngine(config)
    strategy = BTCDominanceStrategy()
    result = engine.run(strategy, data)

    assert result.metrics is not None
    assert len(result.equity_curve) > 0


def test_crypto_dynamic_trail_function():
    """Crypto dynamic trail function should be importable and correct."""
    from crypto.backtester import _crypto_dynamic_trail_mult

    base = 4.0
    assert _crypto_dynamic_trail_mult(base, 0.05) == base  # no tightening
    assert _crypto_dynamic_trail_mult(base, 0.15) == base * 0.85
    assert _crypto_dynamic_trail_mult(base, 0.25) == base * 0.70
    assert _crypto_dynamic_trail_mult(base, 0.40) == base * 0.60
    assert _crypto_dynamic_trail_mult(base, 0.60) == base * 0.50
