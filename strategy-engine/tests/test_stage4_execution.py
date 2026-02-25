#!/usr/bin/env python3
"""Tests for Stage 4: Execution Improvements.

Tests:
1. SmartExecutor limit price computation
2. Slippage calculation (buy/sell, positive/negative)
3. Order timing logic per strategy
4. TradeLogger slippage columns migration
5. TradeEntry with slippage fields
6. Slippage summary query
7. run_daily.py compiles
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio.execution import (
    SmartExecutor,
    OrderTiming,
    STRATEGY_TIMING,
    SlippageRecord,
)
from portfolio.trade_logger import TradeLogger, TradeEntry


# ============================================================
# TEST 1: Limit price computation
# ============================================================
def test_limit_price_computation():
    """Verify limit prices are set slightly better than signal price."""
    print("\n=== Test 1: Limit Price Computation ===")

    # Create executor without trading client (just test math)
    class MockClient:
        def get_clock(self):
            class Clock:
                is_open = True
            return Clock()

    executor = SmartExecutor.__new__(SmartExecutor)
    executor.limit_offset_pct = 0.10  # 0.10%
    executor.pending_limits = []
    executor.slippage_records = []

    # For buys: limit should be BELOW signal price
    buy_limit = executor.compute_limit_price(100.00, "buy")
    assert buy_limit < 100.00, f"Buy limit ({buy_limit}) should be below signal (100.00)"
    assert buy_limit == 99.90, f"Expected 99.90, got {buy_limit}"
    print(f"  Buy @ 100.00 → limit = ${buy_limit:.2f} ✓")

    # For sells (short entry): limit should be ABOVE signal price
    sell_limit = executor.compute_limit_price(100.00, "sell")
    assert sell_limit > 100.00, f"Sell limit ({sell_limit}) should be above signal (100.00)"
    assert sell_limit == 100.10, f"Expected 100.10, got {sell_limit}"
    print(f"  Sell @ 100.00 → limit = ${sell_limit:.2f} ✓")

    # Test with different offset
    executor.limit_offset_pct = 0.05
    buy_limit = executor.compute_limit_price(200.00, "buy")
    assert buy_limit == 199.90, f"Expected 199.90, got {buy_limit}"
    print(f"  Buy @ 200.00 (0.05% offset) → limit = ${buy_limit:.2f} ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 2: Slippage calculation
# ============================================================
def test_slippage_calculation():
    """Verify slippage is computed correctly for buys and sells."""
    print("\n=== Test 2: Slippage Calculation ===")

    # Positive slippage (bad) — bought higher than intended
    slippage = SmartExecutor._compute_slippage(100.0, 100.10, "buy")
    assert abs(slippage - 0.10) < 0.001, f"Expected +0.10%, got {slippage}"
    print(f"  Buy: intended $100, filled $100.10 → slippage = {slippage:+.3f}% ✓")

    # Negative slippage (good) — bought lower than intended
    slippage = SmartExecutor._compute_slippage(100.0, 99.90, "buy")
    assert abs(slippage - (-0.10)) < 0.001, f"Expected -0.10%, got {slippage}"
    print(f"  Buy: intended $100, filled $99.90 → slippage = {slippage:+.3f}% (saved!) ✓")

    # Positive slippage for sell (bad) — sold lower than intended
    slippage = SmartExecutor._compute_slippage(100.0, 99.90, "sell")
    assert abs(slippage - 0.10) < 0.001, f"Expected +0.10%, got {slippage}"
    print(f"  Sell: intended $100, filled $99.90 → slippage = {slippage:+.3f}% ✓")

    # Negative slippage for sell (good) — sold higher than intended
    slippage = SmartExecutor._compute_slippage(100.0, 100.10, "sell")
    assert abs(slippage - (-0.10)) < 0.001, f"Expected -0.10%, got {slippage}"
    print(f"  Sell: intended $100, filled $100.10 → slippage = {slippage:+.3f}% (saved!) ✓")

    # Zero intended price edge case
    slippage = SmartExecutor._compute_slippage(0.0, 100.0, "buy")
    assert slippage == 0.0, f"Expected 0.0 for zero intended, got {slippage}"
    print(f"  Edge case: zero intended → {slippage} ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 3: Order timing logic
# ============================================================
def test_order_timing():
    """Verify strategies get appropriate timing windows."""
    print("\n=== Test 3: Order Timing ===")

    # Check all strategies have timing entries
    expected = {
        "mean_reversion", "gap_trading", "trend_following",
        "momentum", "sector_rotation", "sentiment", "earnings_momentum",
    }
    actual = set(STRATEGY_TIMING.keys())
    assert expected.issubset(actual), f"Missing: {expected - actual}"
    print(f"  All 7 strategies have timing entries ✓")

    # Mean reversion: should trade at open (0 min delay)
    assert STRATEGY_TIMING["mean_reversion"][0] == 0
    print(f"  mean_reversion: trade at open (0 min) ✓")

    # Trend following: should wait 30 min
    assert STRATEGY_TIMING["trend_following"][0] == 30
    print(f"  trend_following: wait 30 min ✓")

    # Gap trading: trade early (2 min)
    assert STRATEGY_TIMING["gap_trading"][0] == 2
    print(f"  gap_trading: trade at 2 min ✓")

    # Momentum: wait 15 min
    assert STRATEGY_TIMING["momentum"][0] == 15
    print(f"  momentum: wait 15 min ✓")

    # Test timing logic with mock
    executor = SmartExecutor.__new__(SmartExecutor)
    executor.pending_limits = []
    executor.slippage_records = []

    # Simulate 5 minutes after open
    executor.market_open_time = datetime.now() - timedelta(minutes=5)

    # Mean reversion should be immediate (0 min wait)
    assert executor.get_order_timing("mean_reversion") == OrderTiming.IMMEDIATE
    print(f"  At t+5min: mean_reversion = IMMEDIATE ✓")

    # Trend following should wait (needs 30 min)
    assert executor.get_order_timing("trend_following") == OrderTiming.WAIT_FOR_WINDOW
    print(f"  At t+5min: trend_following = WAIT ✓")

    # Simulate 45 minutes after open
    executor.market_open_time = datetime.now() - timedelta(minutes=45)
    assert executor.get_order_timing("trend_following") == OrderTiming.IMMEDIATE
    print(f"  At t+45min: trend_following = IMMEDIATE ✓")

    # Simulate 7 hours after open (3:30+ PM)
    executor.market_open_time = datetime.now() - timedelta(hours=7)
    assert executor.get_order_timing("trend_following") == OrderTiming.SKIP_TODAY
    print(f"  At t+7h: trend_following = SKIP ✓")

    # No market_open_time = always immediate
    executor.market_open_time = None
    assert executor.get_order_timing("trend_following") == OrderTiming.IMMEDIATE
    print(f"  No open time: trend_following = IMMEDIATE ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 4: TradeLogger slippage migration
# ============================================================
def test_trade_logger_slippage_migration():
    """Verify the slippage columns are added to existing DBs."""
    print("\n=== Test 4: TradeLogger Slippage Migration ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Create logger — should apply migration
        logger = TradeLogger(db_path=db_path)

        # Check columns exist by inserting with slippage fields
        entry = TradeEntry(
            symbol="AAPL",
            direction="long",
            strategy="momentum",
            conviction_tier=1,
            confluence_count=1,
            entry_price=150.00,
            entry_qty=100,
            entry_fill_price=150.05,
            entry_slippage_pct=0.033,
            entry_order_type="limit",
        )
        lid = logger.log_entry(entry)
        assert lid > 0
        print(f"  Entry with slippage fields: lifecycle_id={lid} ✓")

        # Read it back
        cur = logger.conn.execute(
            "SELECT entry_fill_price, entry_slippage_pct, entry_order_type "
            "FROM position_lifecycle WHERE id = ?",
            (lid,),
        )
        row = dict(cur.fetchone())
        assert row["entry_fill_price"] == 150.05
        assert abs(row["entry_slippage_pct"] - 0.033) < 0.001
        assert row["entry_order_type"] == "limit"
        print(f"  Read back: fill={row['entry_fill_price']}, "
              f"slip={row['entry_slippage_pct']}, type={row['entry_order_type']} ✓")

        # Insert without slippage (backward compat)
        entry2 = TradeEntry(
            symbol="MSFT",
            direction="long",
            strategy="trend_following",
            conviction_tier=2,
            confluence_count=1,
            entry_price=350.00,
            entry_qty=50,
        )
        lid2 = logger.log_entry(entry2)
        assert lid2 > 0

        cur2 = logger.conn.execute(
            "SELECT entry_fill_price, entry_slippage_pct, entry_order_type "
            "FROM position_lifecycle WHERE id = ?",
            (lid2,),
        )
        row2 = dict(cur2.fetchone())
        assert row2["entry_fill_price"] is None
        assert row2["entry_slippage_pct"] is None
        assert row2["entry_order_type"] == "market"
        print(f"  Entry without slippage (backward compat): ✓")

        # Re-create logger to test idempotent migration
        logger.close()
        logger2 = TradeLogger(db_path=db_path)
        entries = logger2.get_todays_entries()
        assert "AAPL" in entries and "MSFT" in entries
        print(f"  Re-migration idempotent: ✓")

        logger2.close()

    finally:
        os.unlink(db_path)

    print("  ✓ PASSED")


# ============================================================
# TEST 5: Slippage summary query
# ============================================================
def test_slippage_summary_query():
    """Test the slippage summary aggregation query."""
    print("\n=== Test 5: Slippage Summary Query ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        logger = TradeLogger(db_path=db_path)

        # Insert trades with different slippage profiles
        trades = [
            ("AAPL", "momentum", 150.0, 150.05, 0.033, "limit"),
            ("MSFT", "momentum", 350.0, 349.90, -0.029, "limit"),
            ("GOOGL", "trend_following", 140.0, 140.20, 0.143, "market"),
            ("NVDA", "gap_trading", 800.0, 800.50, 0.0625, "limit"),
            ("TSLA", "gap_trading", 250.0, 249.50, -0.200, "market"),
        ]

        for sym, strat, price, fill, slip, otype in trades:
            logger.log_entry(TradeEntry(
                symbol=sym,
                direction="long",
                strategy=strat,
                conviction_tier=1,
                confluence_count=1,
                entry_price=price,
                entry_qty=100,
                entry_fill_price=fill,
                entry_slippage_pct=slip,
                entry_order_type=otype,
            ))

        summary = logger.get_slippage_summary()
        print(f"  Summary rows: {len(summary)}")

        for row in summary:
            print(
                f"    {row['strategy']:20s} ({row['entry_order_type']:6s}): "
                f"{row['total_trades']} trades, "
                f"avg slip={row['avg_slippage_pct']:+.4f}%, "
                f"saved=${row.get('total_saved_dollars', 0):.2f}"
            )

        assert len(summary) > 0, "Should have slippage data"

        # Check momentum limit has 2 trades
        mom_limit = [r for r in summary
                     if r["strategy"] == "momentum" and r["entry_order_type"] == "limit"]
        assert len(mom_limit) == 1
        assert mom_limit[0]["total_trades"] == 2
        print(f"  Momentum limit: 2 trades ✓")

        logger.close()

    finally:
        os.unlink(db_path)

    print("  ✓ PASSED")


# ============================================================
# TEST 6: Slippage summary from SmartExecutor
# ============================================================
def test_executor_slippage_summary():
    """Test SmartExecutor's in-memory slippage tracking."""
    print("\n=== Test 6: Executor Slippage Summary ===")

    executor = SmartExecutor.__new__(SmartExecutor)
    executor.pending_limits = []
    executor.slippage_records = []

    # Record some slippage
    executor.slippage_records = [
        SlippageRecord("AAPL", "momentum", "buy", 150.0, 149.90, 149.95, -0.033, "limit"),
        SlippageRecord("MSFT", "momentum", "buy", 350.0, None, 350.20, 0.057, "market"),
        SlippageRecord("NVDA", "trend_following", "buy", 800.0, 799.80, 799.85, -0.019, "limit"),
        SlippageRecord("TSLA", "gap_trading", "sell", 250.0, 250.10, 250.15, -0.060, "limit"),
    ]

    summary = executor.get_slippage_summary()
    print(f"  Total filled: {summary['filled']}")
    print(f"  Avg slippage: {summary['avg_slippage_pct']:+.4f}%")
    print(f"  Limit orders: {summary['limit_orders']['count']}")
    print(f"  Market orders: {summary['market_orders']['count']}")

    assert summary["filled"] == 4
    assert summary["limit_orders"]["count"] == 3
    assert summary["market_orders"]["count"] == 1
    assert summary["avg_slippage_pct"] < 0, "Should have negative avg (better fills via limits)"

    # Per strategy
    assert "momentum" in summary["by_strategy"]
    assert "gap_trading" in summary["by_strategy"]
    print(f"  By strategy: {list(summary['by_strategy'].keys())} ✓")

    print("  ✓ PASSED")


# ============================================================
# TEST 7: run_daily.py compiles with new imports
# ============================================================
def test_run_daily_compiles():
    """Verify run_daily.py compiles with new executor import."""
    print("\n=== Test 7: run_daily.py compiles ===")
    import py_compile
    py_compile.compile(
        str(Path(__file__).parent.parent.parent / "scripts" / "run_daily.py"),
        doraise=True,
    )
    print("  run_daily.py compiles: ✓")
    print("  ✓ PASSED")


# ============================================================
# TEST 8: Minutes until window
# ============================================================
def test_minutes_until_window():
    """Test minutes_until_window calculation."""
    print("\n=== Test 8: Minutes Until Window ===")

    executor = SmartExecutor.__new__(SmartExecutor)
    executor.pending_limits = []
    executor.slippage_records = []

    # 10 minutes after open
    executor.market_open_time = datetime.now() - timedelta(minutes=10)

    # Trend following needs 30 min → 20 min remaining
    remaining = executor.minutes_until_window("trend_following")
    assert abs(remaining - 20.0) < 1.0, f"Expected ~20 min, got {remaining}"
    print(f"  Trend following at t+10min: {remaining:.0f}min remaining ✓")

    # Mean reversion needs 0 min → 0 remaining
    remaining = executor.minutes_until_window("mean_reversion")
    assert remaining == 0.0, f"Expected 0 min, got {remaining}"
    print(f"  Mean reversion at t+10min: {remaining:.0f}min remaining ✓")

    # 60 minutes after open → all strategies ready
    executor.market_open_time = datetime.now() - timedelta(minutes=60)
    remaining = executor.minutes_until_window("trend_following")
    assert remaining == 0.0, f"Expected 0 min, got {remaining}"
    print(f"  Trend following at t+60min: {remaining:.0f}min remaining ✓")

    print("  ✓ PASSED")


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    tests = [
        test_limit_price_computation,
        test_slippage_calculation,
        test_order_timing,
        test_trade_logger_slippage_migration,
        test_slippage_summary_query,
        test_executor_slippage_summary,
        test_run_daily_compiles,
        test_minutes_until_window,
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
