"""Smart Order Execution — Limit Orders, Timing, and Slippage Tracking.

Provides intelligent order execution that:
1. Submits limit orders instead of market orders to reduce spread costs
2. Times entries based on strategy type (avoid first 15 min for most strategies)
3. Falls back to market orders after a configurable timeout
4. Tracks slippage between intended and actual fill prices

Expected savings: 0.05-0.15% per trade (adds up over hundreds of trades).

Usage:
    executor = SmartExecutor(trading_client)
    result = executor.submit_entry(order, strategy_name)
    # Later:
    executor.check_pending_orders()  # converts timed-out limits to market
"""

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    ReplaceOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus


# Strategy-specific order timing windows
# Key: strategy_name → (earliest_minutes_after_open, latest_minutes_after_open)
# During this window, the strategy's orders can be submitted.
# Outside the window, they queue until the start time.
STRATEGY_TIMING = {
    # Mean reversion: place at open (dips are most extreme at open)
    "mean_reversion": (0, None),
    # Gap trading: place early — gaps move fast
    "gap_trading": (2, None),
    # Trend following: wait 30 min for trend to establish
    "trend_following": (30, None),
    # Momentum: 15 min after open (less sensitive to timing)
    "momentum": (15, None),
    # Sector rotation: any time (longer-term, timing less critical)
    "sector_rotation": (0, None),
    # Sentiment: 15 min after open
    "sentiment": (15, None),
    # Earnings momentum: any time
    "earnings_momentum": (0, None),
}

# How long to wait for a limit order fill before converting to market
LIMIT_TIMEOUT_MINUTES = int(os.environ.get("LIMIT_TIMEOUT_MINUTES", "30"))

# How much better than current price to set the limit
# For BUY: limit = current_price * (1 - LIMIT_OFFSET_PCT/100)
# For SELL (short entry): limit = current_price * (1 + LIMIT_OFFSET_PCT/100)
LIMIT_OFFSET_PCT = float(os.environ.get("LIMIT_OFFSET_PCT", "0.10"))


class OrderTiming(Enum):
    """When to submit an order relative to market open."""
    IMMEDIATE = "immediate"          # submit now
    WAIT_FOR_WINDOW = "wait"        # wait until strategy's time window
    SKIP_TODAY = "skip"             # too late in the day


@dataclass
class SlippageRecord:
    """Records the difference between intended and actual fill price."""
    symbol: str
    strategy: str
    side: str  # "buy" or "sell"
    intended_price: float  # signal's entry_price
    limit_price: float | None  # actual limit price set
    fill_price: float | None  # actual fill price from broker
    slippage_pct: float | None  # (fill - intended) / intended * 100 for buys
    order_type: str  # "limit" or "market"
    submitted_at: datetime | None = None
    filled_at: datetime | None = None


@dataclass
class PendingLimitOrder:
    """Tracks a limit order that might need to be converted to market."""
    order_id: str
    symbol: str
    side: str
    qty: int
    limit_price: float
    intended_price: float  # signal's entry_price
    strategy: str
    submitted_at: datetime
    timeout_at: datetime


class SmartExecutor:
    """Handles intelligent order routing with limit-first and timing logic."""

    def __init__(
        self,
        trading_client: TradingClient,
        limit_timeout_minutes: int = LIMIT_TIMEOUT_MINUTES,
        limit_offset_pct: float = LIMIT_OFFSET_PCT,
        market_open_time: datetime | None = None,
    ):
        self.trading_client = trading_client
        self.limit_timeout_minutes = limit_timeout_minutes
        self.limit_offset_pct = limit_offset_pct
        self.pending_limits: list[PendingLimitOrder] = []
        self.slippage_records: list[SlippageRecord] = []

        # Determine market open time
        if market_open_time:
            self.market_open_time = market_open_time
        else:
            try:
                clock = trading_client.get_clock()
                # next_open is used if market hasn't opened yet
                if clock.is_open:
                    # Market is open — approximate open time as 9:30 ET today
                    self.market_open_time = datetime.now().replace(
                        hour=9, minute=30, second=0, microsecond=0
                    )
                else:
                    self.market_open_time = None
            except Exception:
                self.market_open_time = None

    def get_order_timing(self, strategy_name: str) -> OrderTiming:
        """Determine when to submit an order based on strategy type.

        Returns:
            OrderTiming enum value.
        """
        if self.market_open_time is None:
            return OrderTiming.IMMEDIATE

        timing = STRATEGY_TIMING.get(strategy_name, (15, None))
        earliest_minutes = timing[0]

        now = datetime.now()
        minutes_since_open = (now - self.market_open_time).total_seconds() / 60

        # If it's been less than 15 minutes since open and strategy wants
        # to wait, signal WAIT_FOR_WINDOW
        if minutes_since_open < earliest_minutes:
            return OrderTiming.WAIT_FOR_WINDOW

        # If it's been more than 6 hours since open (after 3:30 PM),
        # skip new entries (too little time for stops to work)
        if minutes_since_open > 360:
            return OrderTiming.SKIP_TODAY

        return OrderTiming.IMMEDIATE

    def minutes_until_window(self, strategy_name: str) -> float:
        """How many minutes until this strategy's trading window opens."""
        if self.market_open_time is None:
            return 0

        timing = STRATEGY_TIMING.get(strategy_name, (15, None))
        earliest_minutes = timing[0]

        now = datetime.now()
        minutes_since_open = (now - self.market_open_time).total_seconds() / 60

        remaining = earliest_minutes - minutes_since_open
        return max(0, remaining)

    def compute_limit_price(
        self, signal_price: float, side: str
    ) -> float:
        """Compute limit price that's slightly better than signal price.

        For buys: limit is slightly below current price (we want to buy cheaper)
        For sells (short entry): limit is slightly above (we want to sell higher)
        """
        offset = self.limit_offset_pct / 100.0
        if side == "buy":
            return round(signal_price * (1 - offset), 2)
        else:
            return round(signal_price * (1 + offset), 2)

    def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        signal_price: float,
        strategy_name: str,
    ) -> tuple[str | None, float]:
        """Submit a limit order and track it for timeout.

        Returns:
            Tuple of (order_id, limit_price) or (None, 0) on failure.
        """
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        limit_price = self.compute_limit_price(signal_price, side)

        try:
            order_req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
            result = self.trading_client.submit_order(order_data=order_req)
            now = datetime.now()

            # Track for timeout conversion
            self.pending_limits.append(PendingLimitOrder(
                order_id=str(result.id),
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=limit_price,
                intended_price=signal_price,
                strategy=strategy_name,
                submitted_at=now,
                timeout_at=now + timedelta(minutes=self.limit_timeout_minutes),
            ))

            print(
                f"  LIMIT {side.upper()} {qty} {symbol} @ ${limit_price:.2f} "
                f"(signal: ${signal_price:.2f}, "
                f"timeout: {self.limit_timeout_minutes}min)",
                file=sys.stderr,
            )
            return str(result.id), limit_price

        except Exception as e:
            print(
                f"  LIMIT order failed for {symbol}, falling back to market: {e}",
                file=sys.stderr,
            )
            return None, 0.0

    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        signal_price: float,
        strategy_name: str,
    ) -> str | None:
        """Submit a market order (fallback or when limit isn't suitable).

        Returns:
            order_id or None on failure.
        """
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        try:
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            result = self.trading_client.submit_order(order_data=order_req)

            # Record slippage placeholder (fill price not yet known)
            self.slippage_records.append(SlippageRecord(
                symbol=symbol,
                strategy=strategy_name,
                side=side,
                intended_price=signal_price,
                limit_price=None,
                fill_price=None,  # will be updated when we check fill
                slippage_pct=None,
                order_type="market",
                submitted_at=datetime.now(),
            ))

            print(
                f"  MARKET {side.upper()} {qty} {symbol} "
                f"(signal: ${signal_price:.2f})",
                file=sys.stderr,
            )
            return str(result.id)

        except Exception as e:
            print(
                f"  MARKET order failed for {symbol}: {e}",
                file=sys.stderr,
            )
            return None

    def check_pending_limits(self) -> list[dict]:
        """Check pending limit orders and convert timed-out ones to market.

        Returns:
            List of conversion results with symbol and new order_id.
        """
        now = datetime.now()
        conversions: list[dict] = []
        still_pending: list[PendingLimitOrder] = []

        for pending in self.pending_limits:
            try:
                # Check if filled
                order = self.trading_client.get_order_by_id(pending.order_id)

                if order.status.value == "filled":
                    # Record slippage
                    fill_price = float(order.filled_avg_price) if order.filled_avg_price else pending.limit_price
                    slippage = self._compute_slippage(
                        pending.intended_price, fill_price, pending.side
                    )
                    self.slippage_records.append(SlippageRecord(
                        symbol=pending.symbol,
                        strategy=pending.strategy,
                        side=pending.side,
                        intended_price=pending.intended_price,
                        limit_price=pending.limit_price,
                        fill_price=fill_price,
                        slippage_pct=slippage,
                        order_type="limit",
                        submitted_at=pending.submitted_at,
                        filled_at=now,
                    ))
                    print(
                        f"  ✓ FILLED {pending.symbol}: limit ${pending.limit_price:.2f} "
                        f"→ fill ${fill_price:.2f} "
                        f"(slippage: {slippage:+.3f}%)",
                        file=sys.stderr,
                    )
                    continue

                if order.status.value in ("canceled", "expired", "rejected"):
                    print(
                        f"  ✗ {pending.symbol}: limit order {order.status.value}",
                        file=sys.stderr,
                    )
                    continue

                # Check timeout
                if now >= pending.timeout_at:
                    # Cancel the limit order
                    try:
                        self.trading_client.cancel_order_by_id(pending.order_id)
                    except Exception:
                        pass

                    time.sleep(1)  # brief wait for cancellation

                    # Resubmit as market
                    new_id = self.submit_market_order(
                        symbol=pending.symbol,
                        qty=pending.qty,
                        side=pending.side,
                        signal_price=pending.intended_price,
                        strategy_name=pending.strategy,
                    )
                    if new_id:
                        conversions.append({
                            "symbol": pending.symbol,
                            "old_order_id": pending.order_id,
                            "new_order_id": new_id,
                            "reason": "limit_timeout",
                        })
                        print(
                            f"  ⟳ CONVERTED {pending.symbol}: "
                            f"limit → market (timeout after {self.limit_timeout_minutes}min)",
                            file=sys.stderr,
                        )
                else:
                    # Still waiting
                    still_pending.append(pending)

            except Exception as e:
                print(
                    f"  Warning: could not check pending order for {pending.symbol}: {e}",
                    file=sys.stderr,
                )
                still_pending.append(pending)

        self.pending_limits = still_pending
        return conversions

    def record_fill_slippage(
        self,
        symbol: str,
        intended_price: float,
        fill_price: float,
        side: str,
        strategy: str,
        order_type: str = "market",
    ) -> SlippageRecord:
        """Record slippage for any order (called after fill confirmation).

        Returns:
            The slippage record.
        """
        slippage = self._compute_slippage(intended_price, fill_price, side)
        record = SlippageRecord(
            symbol=symbol,
            strategy=strategy,
            side=side,
            intended_price=intended_price,
            limit_price=None,
            fill_price=fill_price,
            slippage_pct=slippage,
            order_type=order_type,
            submitted_at=datetime.now(),
            filled_at=datetime.now(),
        )
        self.slippage_records.append(record)
        return record

    def get_slippage_summary(self) -> dict:
        """Summarize slippage across all recorded orders.

        Returns:
            Dict with overall and per-strategy slippage stats.
        """
        if not self.slippage_records:
            return {"total_records": 0}

        filled = [r for r in self.slippage_records if r.slippage_pct is not None]
        if not filled:
            return {"total_records": len(self.slippage_records), "filled": 0}

        all_slippage = [r.slippage_pct for r in filled]
        avg_slippage = sum(all_slippage) / len(all_slippage)

        # Per strategy
        by_strategy: dict[str, list[float]] = {}
        for r in filled:
            by_strategy.setdefault(r.strategy, []).append(r.slippage_pct)

        strategy_summary = {}
        for strat, slippages in by_strategy.items():
            strategy_summary[strat] = {
                "count": len(slippages),
                "avg_slippage_pct": round(sum(slippages) / len(slippages), 4),
                "max_slippage_pct": round(max(slippages), 4),
                "total_saved_vs_market": round(
                    sum(-s for s in slippages if s < 0), 4
                ),
            }

        # Per order type
        limit_fills = [r for r in filled if r.order_type == "limit"]
        market_fills = [r for r in filled if r.order_type == "market"]

        return {
            "total_records": len(self.slippage_records),
            "filled": len(filled),
            "avg_slippage_pct": round(avg_slippage, 4),
            "limit_orders": {
                "count": len(limit_fills),
                "avg_slippage_pct": round(
                    sum(r.slippage_pct for r in limit_fills) / len(limit_fills), 4
                ) if limit_fills else 0,
            },
            "market_orders": {
                "count": len(market_fills),
                "avg_slippage_pct": round(
                    sum(r.slippage_pct for r in market_fills) / len(market_fills), 4
                ) if market_fills else 0,
            },
            "by_strategy": strategy_summary,
        }

    @staticmethod
    def _compute_slippage(
        intended_price: float, fill_price: float, side: str
    ) -> float:
        """Compute slippage percentage.

        Positive slippage = worse fill (paid more / received less).
        Negative slippage = better fill (paid less / received more).

        For buys: slippage = (fill - intended) / intended * 100
        For sells: slippage = (intended - fill) / intended * 100
        """
        if intended_price <= 0:
            return 0.0

        if side == "buy":
            return (fill_price - intended_price) / intended_price * 100
        else:
            return (intended_price - fill_price) / intended_price * 100
