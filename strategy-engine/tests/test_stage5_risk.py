#!/usr/bin/env python3
"""Tests for Stage 5: Advanced Risk Management.

Tests:
1. Correlation computation between two stocks
2. Correlation limit check (allow/reduce/skip)
3. Drawdown circuit breaker levels (normal → caution → defensive → halt → unwind)
4. Drawdown recovery detection
5. Weakest position identification for unwinding
6. Optimizer integration: correlation reduces position size
7. Optimizer integration: drawdown halt blocks all entries
8. Optimizer integration: drawdown defensive reduces sizes
9. Peak equity from snapshots
10. run_daily.py and price_monitor.py compile
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio.correlation import (
    compute_return_correlation,
    check_correlation_limit,
    CORRELATION_THRESHOLD,
)
from portfolio.drawdown import (
    DrawdownCircuitBreaker,
    DrawdownLevel,
    DrawdownState,
    get_peak_equity_from_snapshots,
    get_weakest_positions,
)
from portfolio.optimizer import PortfolioOptimizer, OrderIntent
from portfolio.sizing import PositionSizeResult, PortfolioContext
from strategies.base import Signal, SignalDirection


# Shared date range for all test DataFrames (alignment is critical for correlation)
# Use 150 dates so we always have enough for any test
_SHARED_DATES = pd.bdate_range(end="2025-06-01", periods=150)


def _make_price_df(closes: list[float], dates=None) -> pd.DataFrame:
    """Helper: create OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    if dates is None:
        dates = _SHARED_DATES[:n]
    else:
        dates = dates[:n]
    return pd.DataFrame({
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
    }, index=dates)


def _make_correlated_prices(
    base_closes: list[float], noise_pct: float = 0.001, seed: int = 42
) -> list[float]:
    """Create prices highly correlated with base by adding small noise."""
    np.random.seed(seed)
    returns = pd.Series(base_closes).pct_change().fillna(0).values
    noise = np.random.normal(0, noise_pct, len(returns))
    corr_returns = returns + noise
    prices = [base_closes[0]]
    for r in corr_returns[1:]:
        prices.append(prices[-1] * (1 + r))
    return prices


def _make_uncorrelated_prices(n: int, seed: int = 99) -> list[float]:
    """Create random uncorrelated prices."""
    np.random.seed(seed)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    return prices


def _make_signal(symbol, direction=SignalDirection.LONG, strength=0.7,
                 entry_price=100.0, stop_loss=95.0, take_profit=110.0,
                 strategy_name="momentum", rationale="test") -> Signal:
    """Helper to create a Signal with required timestamp."""
    return Signal(
        timestamp=datetime.now(),
        symbol=symbol,
        direction=direction,
        strength=strength,
        strategy_name=strategy_name,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        rationale=rationale,
    )


# ============================================================
# TEST 1: Correlation computation
# ============================================================
def test_correlation_computation():
    """Verify correlation between highly correlated and uncorrelated stocks."""
    print("\n=== Test 1: Correlation Computation ===")

    n = 100
    # Create base price series
    np.random.seed(42)
    base_prices = [100.0]
    for _ in range(n - 1):
        base_prices.append(base_prices[-1] * (1 + np.random.normal(0.001, 0.02)))

    # Highly correlated series (same returns + small noise)
    corr_prices = _make_correlated_prices(base_prices, noise_pct=0.001)

    # Uncorrelated series
    uncorr_prices = _make_uncorrelated_prices(n)

    df_base = _make_price_df(base_prices)
    df_corr = _make_price_df(corr_prices)
    df_uncorr = _make_price_df(uncorr_prices)

    # High correlation
    corr_high = compute_return_correlation(df_base, df_corr, lookback=60)
    assert corr_high is not None, "Correlation should not be None"
    assert corr_high > 0.90, f"Expected >0.90 correlation, got {corr_high:.3f}"
    print(f"  Correlated stocks: r = {corr_high:.3f} ✓")

    # Low correlation
    corr_low = compute_return_correlation(df_base, df_uncorr, lookback=60)
    assert corr_low is not None, "Correlation should not be None"
    assert abs(corr_low) < 0.50, f"Expected <0.50 correlation, got {corr_low:.3f}"
    print(f"  Uncorrelated stocks: r = {corr_low:.3f} ✓")

    # Insufficient data
    short_df = _make_price_df(base_prices[:10], dates=_SHARED_DATES[:10])
    corr_short = compute_return_correlation(df_base, short_df, lookback=60)
    assert corr_short is None
    print(f"  Insufficient data: None ✓")

    # None input
    corr_none = compute_return_correlation(None, df_base, lookback=60)
    assert corr_none is None
    print(f"  None input: None ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 2: Correlation limit check
# ============================================================
def test_correlation_limit_check():
    """Verify allow/reduce/skip decisions based on number of correlated positions."""
    print("\n=== Test 2: Correlation Limit Check ===")

    n = 100
    np.random.seed(42)
    base_prices = [100.0]
    for _ in range(n - 1):
        base_prices.append(base_prices[-1] * (1 + np.random.normal(0.001, 0.02)))

    # Create 4 correlated stocks + 1 uncorrelated (all sharing same dates)
    price_data = {
        "BASE": _make_price_df(base_prices),
        "CORR1": _make_price_df(_make_correlated_prices(base_prices, 0.001, seed=10)),
        "CORR2": _make_price_df(_make_correlated_prices(base_prices, 0.002, seed=20)),
        "CORR3": _make_price_df(_make_correlated_prices(base_prices, 0.003, seed=30)),
        "UNCORR": _make_price_df(_make_uncorrelated_prices(n, seed=99)),
        "NEW": _make_price_df(_make_correlated_prices(base_prices, 0.0015, seed=40)),
    }

    # NEW is correlated with BASE, CORR1, CORR2, CORR3 — should be "skip" (3+ correlated)
    action, mult, pairs = check_correlation_limit(
        "NEW", ["BASE", "CORR1", "CORR2", "CORR3", "UNCORR"],
        price_data, threshold=0.80, reduce_at=2, skip_at=3,
    )
    print(f"  4 correlated positions: action={action}, pairs={len(pairs)}")
    assert action == "skip", f"Expected skip with {len(pairs)} correlated, got {action}"
    assert mult == 0.0
    print(f"  ✓ Skip with {len(pairs)} correlated positions")

    # NEW vs only BASE and CORR1 — 2 correlated → "reduce"
    action2, mult2, pairs2 = check_correlation_limit(
        "NEW", ["BASE", "CORR1", "UNCORR"],
        price_data, threshold=0.80, reduce_at=2, skip_at=3,
    )
    print(f"  2 correlated positions: action={action2}, pairs={len(pairs2)}")
    assert action2 == "reduce", f"Expected reduce, got {action2}"
    assert mult2 == 0.50
    print(f"  ✓ Reduce with {len(pairs2)} correlated positions")

    # UNCORR vs BASE — not correlated → "allow"
    action3, mult3, pairs3 = check_correlation_limit(
        "UNCORR", ["BASE", "CORR1"],
        price_data, threshold=0.80, reduce_at=2, skip_at=3,
    )
    print(f"  Uncorrelated: action={action3}, pairs={len(pairs3)}")
    assert action3 == "allow", f"Expected allow, got {action3}"
    assert mult3 == 1.0
    print(f"  ✓ Allow uncorrelated position")

    # Missing symbol in price_data → "allow"
    action4, mult4, _ = check_correlation_limit(
        "MISSING", ["BASE"], price_data,
    )
    assert action4 == "allow"
    print(f"  ✓ Missing symbol → allow")

    print("  ✓ PASSED")


# ============================================================
# TEST 3: Drawdown circuit breaker levels
# ============================================================
def test_drawdown_levels():
    """Verify graduated drawdown response levels."""
    print("\n=== Test 3: Drawdown Circuit Breaker Levels ===")

    breaker = DrawdownCircuitBreaker()
    peak = 100_000.0

    # Normal: 2% drawdown
    state = breaker.check(98_000, peak)
    assert state.level == DrawdownLevel.NORMAL
    assert state.size_multiplier == 1.0
    assert state.allow_new_entries is True
    assert state.unwind_weakest is False
    print(f"  2% drawdown → NORMAL (size=1.0x) ✓")

    # Caution: 6% drawdown
    state = breaker.check(94_000, peak)
    assert state.level == DrawdownLevel.CAUTION
    assert state.size_multiplier == 0.75
    assert state.allow_new_entries is True
    assert state.unwind_weakest is False
    print(f"  6% drawdown → CAUTION (size=0.75x) ✓")

    # Defensive: 8% drawdown
    state = breaker.check(92_000, peak)
    assert state.level == DrawdownLevel.DEFENSIVE
    assert state.size_multiplier == 0.50
    assert state.allow_new_entries is True
    assert state.min_strength_override == 0.45
    print(f"  8% drawdown → DEFENSIVE (size=0.50x, min_strength=0.45) ✓")

    # Halt: 11% drawdown
    state = breaker.check(89_000, peak)
    assert state.level == DrawdownLevel.HALT
    assert state.size_multiplier == 0.0
    assert state.allow_new_entries is False
    assert state.unwind_weakest is False
    print(f"  11% drawdown → HALT (no entries) ✓")

    # Unwind: 13% drawdown
    state = breaker.check(87_000, peak)
    assert state.level == DrawdownLevel.UNWIND
    assert state.size_multiplier == 0.0
    assert state.allow_new_entries is False
    assert state.unwind_weakest is True
    print(f"  13% drawdown → UNWIND (closing weakest) ✓")

    # Edge: very small equity (99%+ drawdown)
    state = breaker.check(100, peak)
    assert state.level == DrawdownLevel.UNWIND
    print(f"  99.9% drawdown → UNWIND ✓")

    # Edge: zero peak → normal
    state = breaker.check(100_000, 0)
    assert state.level == DrawdownLevel.NORMAL
    print(f"  Zero peak → NORMAL ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 4: Drawdown recovery
# ============================================================
def test_drawdown_recovery():
    """Verify recovery detection."""
    print("\n=== Test 4: Drawdown Recovery ===")

    breaker = DrawdownCircuitBreaker()
    peak = 100_000.0

    # Not recovered: 8% drawdown
    assert not breaker.is_recovered(92_000, peak)
    print(f"  8% drawdown: NOT recovered ✓")

    # Recovered: 3% drawdown (below 5% threshold)
    assert breaker.is_recovered(97_000, peak)
    print(f"  3% drawdown: recovered ✓")

    # Edge: exactly at threshold
    assert not breaker.is_recovered(95_000, peak)
    print(f"  5% drawdown: NOT recovered (at threshold) ✓")

    # Edge: above peak (new high)
    assert breaker.is_recovered(105_000, peak)
    print(f"  Above peak: recovered ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 5: Weakest position identification
# ============================================================
def test_weakest_positions():
    """Verify weakest positions are identified correctly."""
    print("\n=== Test 5: Weakest Position Identification ===")

    positions = [
        {"symbol": "AAPL", "unrealized_plpc": 0.05, "current_price": 150, "qty": 100},
        {"symbol": "MSFT", "unrealized_plpc": -0.08, "current_price": 350, "qty": 50},
        {"symbol": "NVDA", "unrealized_plpc": -0.15, "current_price": 800, "qty": 20},
        {"symbol": "GOOGL", "unrealized_plpc": 0.02, "current_price": 140, "qty": 75},
        {"symbol": "TSLA", "unrealized_plpc": -0.03, "current_price": 250, "qty": 40},
    ]

    weakest = get_weakest_positions(positions, n=2)
    assert len(weakest) == 2
    assert weakest[0]["symbol"] == "NVDA", f"Expected NVDA, got {weakest[0]['symbol']}"
    assert weakest[1]["symbol"] == "MSFT", f"Expected MSFT, got {weakest[1]['symbol']}"
    print(f"  Top 2 weakest: {weakest[0]['symbol']} ({weakest[0]['unrealized_plpc']:+.1%}), "
          f"{weakest[1]['symbol']} ({weakest[1]['unrealized_plpc']:+.1%}) ✓")

    # Edge: empty positions
    assert get_weakest_positions([], n=2) == []
    print(f"  Empty positions: [] ✓")

    # Edge: n > len(positions)
    weakest_all = get_weakest_positions(positions, n=10)
    assert len(weakest_all) == 5
    print(f"  n > len: returns all {len(weakest_all)} ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 6: Optimizer — correlation reduces position size
# ============================================================
def test_optimizer_correlation_reduces_size():
    """Verify optimizer reduces position size for correlated entries."""
    print("\n=== Test 6: Optimizer Correlation Reduces Size ===")

    optimizer = PortfolioOptimizer()

    n = 100
    np.random.seed(42)
    base_prices = [100.0]
    for _ in range(n - 1):
        base_prices.append(base_prices[-1] * (1 + np.random.normal(0.001, 0.02)))

    price_data = {
        "HELD1": _make_price_df(base_prices),
        "HELD2": _make_price_df(_make_correlated_prices(base_prices, 0.001, seed=10)),
        "NEW_CORR": _make_price_df(_make_correlated_prices(base_prices, 0.002, seed=20)),
        "NEW_UNCORR": _make_price_df(_make_uncorrelated_prices(n, seed=99)),
    }

    # Existing positions: HELD1, HELD2 (both correlated with each other)
    current_positions = {
        "HELD1": {"qty": 100, "market_value": 10000, "side": "long"},
        "HELD2": {"qty": 50, "market_value": 5000, "side": "long"},
    }

    portfolio = PortfolioContext(
        equity=100_000, cash=50_000, buying_power=50_000,
        num_positions=2, strategy_allocation_pct=0.2,
    )

    # Create signals
    sig_corr = _make_signal("NEW_CORR")
    sizing_corr = PositionSizeResult(
        shares=100, dollar_value=10_000,
        risk_per_share=5.0, portfolio_risk_pct=0.05,
        method="fixed_fractional", details={"conviction_tier": 2},
    )

    sig_uncorr = _make_signal("NEW_UNCORR")
    sizing_uncorr = PositionSizeResult(
        shares=100, dollar_value=10_000,
        risk_per_share=5.0, portfolio_risk_pct=0.05,
        method="fixed_fractional", details={"conviction_tier": 2},
    )

    orders = optimizer.optimize(
        signals_with_sizing=[(sig_corr, sizing_corr), (sig_uncorr, sizing_uncorr)],
        current_positions=current_positions,
        portfolio=portfolio,
        price_data=price_data,
    )

    # Both should be accepted (reduce, not skip, since only 2 correlated)
    corr_order = next((o for o in orders if o.symbol == "NEW_CORR"), None)
    uncorr_order = next((o for o in orders if o.symbol == "NEW_UNCORR"), None)

    assert uncorr_order is not None, "Uncorrelated order should be accepted"
    assert uncorr_order.qty == 100, f"Uncorrelated qty should be 100, got {uncorr_order.qty}"
    print(f"  Uncorrelated: {uncorr_order.qty} shares (unchanged) ✓")

    assert corr_order is not None, "Correlated order should be accepted (reduced, not skipped)"
    assert corr_order.qty < 100, f"Correlated qty should be reduced, got {corr_order.qty}"
    assert corr_order.qty == 50, f"Expected 50 (50% reduction), got {corr_order.qty}"
    print(f"  Correlated: {corr_order.qty} shares (reduced 50%) ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 7: Optimizer — drawdown halt blocks all entries
# ============================================================
def test_optimizer_drawdown_halt():
    """Verify optimizer blocks all entries when drawdown halt is active."""
    print("\n=== Test 7: Optimizer Drawdown Halt ===")

    optimizer = PortfolioOptimizer()

    portfolio = PortfolioContext(
        equity=89_000, cash=40_000, buying_power=40_000,
        num_positions=2, strategy_allocation_pct=0.2,
    )

    sig = _make_signal("AAPL", strength=0.9, entry_price=150.0,
                       stop_loss=140.0, take_profit=170.0)
    sizing = PositionSizeResult(
        shares=50, dollar_value=7500,
        risk_per_share=10.0, portfolio_risk_pct=0.05,
        method="fixed_fractional", details={"conviction_tier": 1},
    )

    # Drawdown halt state
    dd_halt = DrawdownState(
        level=DrawdownLevel.HALT,
        drawdown_pct=0.11,
        peak_equity=100_000,
        current_equity=89_000,
        size_multiplier=0.0,
        min_strength_override=1.0,
        allow_new_entries=False,
        unwind_weakest=False,
        description="HALT: 11% drawdown",
    )

    orders = optimizer.optimize(
        signals_with_sizing=[(sig, sizing)],
        current_positions={},
        portfolio=portfolio,
        drawdown_state=dd_halt,
    )

    assert len(orders) == 0, f"Expected 0 orders during halt, got {len(orders)}"
    print(f"  HALT state: {len(orders)} entries blocked ✓")

    # Exit orders should still go through even during halt
    sig_exit = _make_signal("MSFT", direction=SignalDirection.CLOSE,
                            strength=0.0, entry_price=350.0,
                            stop_loss=None, take_profit=None, rationale="exit")
    sizing_exit = PositionSizeResult(
        shares=50, dollar_value=17500,
        risk_per_share=0, portfolio_risk_pct=0,
        method="fixed_fractional", details={},
    )

    orders_with_exit = optimizer.optimize(
        signals_with_sizing=[(sig, sizing), (sig_exit, sizing_exit)],
        current_positions={"MSFT": {"qty": 50, "market_value": 17500, "side": "long"}},
        portfolio=portfolio,
        drawdown_state=dd_halt,
    )

    assert len(orders_with_exit) == 1, f"Expected 1 exit order, got {len(orders_with_exit)}"
    assert orders_with_exit[0].symbol == "MSFT"
    print(f"  HALT state: exit order for MSFT still goes through ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 8: Optimizer — drawdown defensive reduces sizes
# ============================================================
def test_optimizer_drawdown_defensive():
    """Verify optimizer reduces position sizes during defensive drawdown."""
    print("\n=== Test 8: Optimizer Drawdown Defensive ===")

    optimizer = PortfolioOptimizer()

    portfolio = PortfolioContext(
        equity=92_000, cash=40_000, buying_power=40_000,
        num_positions=1, strategy_allocation_pct=0.2,
    )

    sig = _make_signal("AAPL", entry_price=150.0, stop_loss=140.0, take_profit=170.0)
    sizing = PositionSizeResult(
        shares=100, dollar_value=15000,
        risk_per_share=10.0, portfolio_risk_pct=0.05,
        method="fixed_fractional", details={"conviction_tier": 2},
    )

    # No drawdown — full size
    orders_normal = optimizer.optimize(
        signals_with_sizing=[(sig, sizing)],
        current_positions={},
        portfolio=portfolio,
    )
    assert len(orders_normal) == 1
    normal_qty = orders_normal[0].qty
    assert normal_qty == 100
    print(f"  Normal: {normal_qty} shares ✓")

    # Defensive drawdown — 50% reduction
    dd_defensive = DrawdownState(
        level=DrawdownLevel.DEFENSIVE,
        drawdown_pct=0.08,
        peak_equity=100_000,
        current_equity=92_000,
        size_multiplier=0.50,
        min_strength_override=0.45,
        allow_new_entries=True,
        unwind_weakest=False,
        description="DEFENSIVE: 8% drawdown",
    )

    orders_dd = optimizer.optimize(
        signals_with_sizing=[(sig, sizing)],
        current_positions={},
        portfolio=portfolio,
        drawdown_state=dd_defensive,
    )
    assert len(orders_dd) == 1
    dd_qty = orders_dd[0].qty
    assert dd_qty == 50, f"Expected 50 (50% of 100), got {dd_qty}"
    print(f"  Defensive: {dd_qty} shares (50% reduction) ✓")

    # Caution drawdown — 25% reduction
    dd_caution = DrawdownState(
        level=DrawdownLevel.CAUTION,
        drawdown_pct=0.06,
        peak_equity=100_000,
        current_equity=94_000,
        size_multiplier=0.75,
        min_strength_override=0.35,
        allow_new_entries=True,
        unwind_weakest=False,
        description="CAUTION: 6% drawdown",
    )

    orders_caut = optimizer.optimize(
        signals_with_sizing=[(sig, sizing)],
        current_positions={},
        portfolio=portfolio,
        drawdown_state=dd_caution,
    )
    assert len(orders_caut) == 1
    caut_qty = orders_caut[0].qty
    assert caut_qty == 75, f"Expected 75 (75% of 100), got {caut_qty}"
    print(f"  Caution: {caut_qty} shares (25% reduction) ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 9: Peak equity from snapshots
# ============================================================
def test_peak_equity_from_snapshots():
    """Verify peak equity is read from portfolio snapshots."""
    print("\n=== Test 9: Peak Equity from Snapshots ===")

    from portfolio.trade_logger import TradeLogger

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        logger = TradeLogger(db_path=db_path)

        # No snapshots yet → 0.0
        peak = get_peak_equity_from_snapshots(logger)
        assert peak == 0.0
        print(f"  No snapshots: peak = {peak} ✓")

        # Log some snapshots
        logger.log_snapshot(equity=100_000, cash=50_000)
        logger.log_snapshot(equity=105_000, cash=48_000)
        logger.log_snapshot(equity=102_000, cash=49_000)

        peak = get_peak_equity_from_snapshots(logger)
        assert peak == 105_000.0, f"Expected 105000, got {peak}"
        print(f"  After 3 snapshots: peak = ${peak:,.0f} ✓")

        # New peak
        logger.log_snapshot(equity=110_000, cash=45_000)
        peak = get_peak_equity_from_snapshots(logger)
        assert peak == 110_000.0, f"Expected 110000, got {peak}"
        print(f"  After new high: peak = ${peak:,.0f} ✓")

        logger.close()

    finally:
        os.unlink(db_path)

    print("  ✓ PASSED")


# ============================================================
# TEST 10: Scripts compile with new imports
# ============================================================
def test_scripts_compile():
    """Verify run_daily.py and price_monitor.py compile."""
    print("\n=== Test 10: Scripts Compile ===")
    import py_compile

    scripts_dir = Path(__file__).parent.parent.parent / "scripts"

    py_compile.compile(str(scripts_dir / "run_daily.py"), doraise=True)
    print(f"  run_daily.py compiles ✓")

    py_compile.compile(str(scripts_dir / "price_monitor.py"), doraise=True)
    print(f"  price_monitor.py compiles ✓")

    # Also verify the new modules compile
    py_compile.compile(
        str(Path(__file__).parent.parent / "portfolio" / "correlation.py"),
        doraise=True,
    )
    print(f"  correlation.py compiles ✓")

    py_compile.compile(
        str(Path(__file__).parent.parent / "portfolio" / "drawdown.py"),
        doraise=True,
    )
    print(f"  drawdown.py compiles ✓")

    py_compile.compile(
        str(Path(__file__).parent.parent / "portfolio" / "optimizer.py"),
        doraise=True,
    )
    print(f"  optimizer.py compiles ✓")

    print("  ✓ PASSED")


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    tests = [
        test_correlation_computation,
        test_correlation_limit_check,
        test_drawdown_levels,
        test_drawdown_recovery,
        test_weakest_positions,
        test_optimizer_correlation_reduces_size,
        test_optimizer_drawdown_halt,
        test_optimizer_drawdown_defensive,
        test_peak_equity_from_snapshots,
        test_scripts_compile,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*50}")

    sys.exit(1 if failed > 0 else 0)
