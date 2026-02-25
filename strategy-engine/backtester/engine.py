"""Walk-Forward Backtesting Engine.

Simulates strategy execution on historical data with realistic
trade modeling, position tracking, and slippage.

Walk-forward approach:
    - Train window: 252 days (1 year)
    - Test window: 63 days (1 quarter)
    - Step forward by test window
    - Re-fit parameters at each step (if strategy supports it)

Supports:
    - Long AND short positions
    - ATR-based trailing stops with dynamic tightening
    - 3-stage partial scale-out (first target, second target, trailing remainder)
    - Hard stop loss (8% max loss)
    - Time-based exits (for earnings_momentum)
    - BB middle exits (for mean_reversion)
    - Realistic slippage and transaction costs

This prevents look-ahead bias and gives out-of-sample performance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
import sys

import numpy as np
import pandas as pd
import ta

from strategies.base import Strategy, Signal, SignalDirection
from backtester.costs import TransactionCostCalculator, CostModel
from backtester.metrics import compute_metrics, PerformanceMetrics
from portfolio.profit_targets import get_profit_config, DEFAULT_CONFIG


HARD_STOP_PCT = 0.08  # 8% max loss from entry — absolute floor


@dataclass
class BacktestTrade:
    entry_date: datetime
    exit_date: datetime | None
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float | None
    shares: int
    pnl: float
    pnl_pct: float
    strategy_name: str
    stop_loss: float | None
    exit_reason: str = ""
    is_partial: bool = False
    features: dict = field(default_factory=dict)


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    train_window: int = 252  # trading days
    test_window: int = 63  # trading days
    max_position_pct: float = 0.10  # max 10% per position
    max_positions: int = 30
    max_short_positions: int = 5
    risk_per_trade_pct: float = 0.02  # 2% risk per trade
    enable_scale_out: bool = True  # enable 3-stage profit taking
    enable_dynamic_trailing: bool = True  # tighten trailing stops as profit grows
    cost_model: CostModel = field(default_factory=CostModel)


@dataclass
class BacktestResult:
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    trades: pd.DataFrame
    config: BacktestConfig
    walk_forward_windows: list[dict]


@dataclass
class _Position:
    """Internal position tracking with full exit management state."""
    symbol: str
    shares: int
    entry_price: float
    stop_loss: float | None
    entry_date: datetime
    strategy: str
    side: str  # "long" or "short"
    atr_at_entry: float
    trail_stop_atr_mult: float
    # Water marks
    high_water_mark: float
    low_water_mark: float
    # Scale-out state
    initial_shares: int
    first_target_pct: float
    partial_sell_pct: float
    first_target_hit: bool = False
    second_target_pct: float = 0.0
    second_sell_pct: float = 0.0
    second_target_hit: bool = False
    # BB middle exit
    exit_at_bb_middle: bool = False
    bb_middle_at_entry: float | None = None
    # Time exit
    time_exit_days: int | None = None
    entry_day_idx: int = 0  # index into all_dates for counting hold days


def _dynamic_trail_mult(base_mult: float, pnl_pct: float) -> float:
    """Tighten the trailing stop multiplier as profit grows."""
    if pnl_pct < 0.10:
        return base_mult
    elif pnl_pct < 0.15:
        return base_mult * 0.85
    elif pnl_pct < 0.20:
        return base_mult * 0.70
    else:
        return base_mult * 0.55


class BacktestEngine:

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.cost_calc = TransactionCostCalculator(self.config.cost_model)

    def run(
        self,
        strategy: Strategy,
        data: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            strategy: Strategy instance to test.
            data: Dict mapping symbol -> OHLCV DataFrame with DatetimeIndex.
                  All DataFrames must cover the same date range.

        Returns:
            BacktestResult with metrics, equity curve, and trade log.
        """
        c = self.config

        # Get the common date range
        all_dates = self._get_common_dates(data)
        if len(all_dates) < c.train_window + c.test_window:
            raise ValueError(
                f"Insufficient data: {len(all_dates)} days, "
                f"need at least {c.train_window + c.test_window}"
            )

        # State tracking
        equity = c.initial_capital
        cash = c.initial_capital
        positions: dict[str, _Position] = {}
        all_trades: list[BacktestTrade] = []
        equity_history: list[tuple[datetime, float]] = []
        walk_forward_windows: list[dict] = []

        # Walk-forward loop
        start_idx = c.train_window
        while start_idx + c.test_window <= len(all_dates):
            test_start = start_idx
            test_end = min(start_idx + c.test_window, len(all_dates))

            window_info = {
                "train_start": str(all_dates[start_idx - c.train_window]),
                "train_end": str(all_dates[start_idx - 1]),
                "test_start": str(all_dates[test_start]),
                "test_end": str(all_dates[test_end - 1]),
            }

            # Simulate each day in the test window
            for day_idx in range(test_start, test_end):
                current_date = all_dates[day_idx]

                # Build data slices up to current day (no look-ahead)
                data_slice = {}
                for symbol, df in data.items():
                    mask = df.index <= current_date
                    sliced = df[mask]
                    if len(sliced) >= strategy.required_history_days():
                        data_slice[symbol] = sliced

                if not data_slice:
                    equity_history.append((current_date, equity))
                    continue

                # Check exit conditions on existing positions
                freed_cash = self._check_exits(
                    positions, data, current_date, day_idx,
                    all_trades, c, equity,
                )
                cash += freed_cash

                # Generate signals
                try:
                    signals = strategy.generate_signals(data_slice)
                except Exception as e:
                    print(f"[Backtest] Signal error on {current_date}: {e}", file=sys.stderr)
                    signals = []

                # Process exit signals first
                for sig in signals:
                    if sig.direction == SignalDirection.CLOSE and sig.symbol in positions:
                        pos = positions[sig.symbol]
                        exit_price = self._get_price(data, sig.symbol, current_date)
                        if exit_price is None:
                            continue

                        pnl, pnl_pct, exec_price = self._compute_pnl(
                            pos, exit_price
                        )
                        all_trades.append(BacktestTrade(
                            entry_date=pos.entry_date,
                            exit_date=current_date,
                            symbol=sig.symbol,
                            side=pos.side,
                            entry_price=pos.entry_price,
                            exit_price=exec_price,
                            shares=pos.shares,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            strategy_name=strategy.name,
                            stop_loss=pos.stop_loss,
                            exit_reason="signal_close",
                            features=sig.features,
                        ))
                        cash += self._close_value(pos, exec_price)
                        del positions[sig.symbol]

                # Process entry signals (LONG and SHORT)
                entry_signals = [
                    s for s in signals
                    if s.direction in (SignalDirection.LONG, SignalDirection.SHORT)
                    and s.symbol not in positions
                ]
                entry_signals.sort(key=lambda s: s.strength, reverse=True)

                short_count = sum(
                    1 for p in positions.values() if p.side == "short"
                )

                for sig in entry_signals:
                    if len(positions) >= c.max_positions:
                        break

                    # Short position limit
                    if sig.direction == SignalDirection.SHORT:
                        if short_count >= c.max_short_positions:
                            continue

                    price = self._get_price(data, sig.symbol, current_date)
                    if price is None:
                        continue

                    is_short = sig.direction == SignalDirection.SHORT
                    buy_side = "sell" if is_short else "buy"
                    exec_price = self.cost_calc.calculate_execution_price(
                        price, buy_side
                    )

                    # Position sizing: fixed fractional
                    if sig.stop_loss:
                        if is_short:
                            risk_per_share = sig.stop_loss - exec_price
                        else:
                            risk_per_share = exec_price - sig.stop_loss
                        if risk_per_share <= 0:
                            continue
                        max_risk = equity * c.risk_per_trade_pct
                        shares_by_risk = int(max_risk / risk_per_share)
                    else:
                        shares_by_risk = int(
                            (equity * c.risk_per_trade_pct) / (exec_price * 0.05)
                        )

                    # Position size cap
                    max_shares_by_pct = int(
                        (equity * c.max_position_pct) / exec_price
                    )
                    shares = max(1, min(shares_by_risk, max_shares_by_pct))

                    cost = shares * exec_price
                    if cost > cash:
                        shares = int(cash / exec_price)
                        if shares <= 0:
                            continue

                    # Get ATR for trailing stop
                    atr = self._compute_atr(data, sig.symbol, current_date)

                    # Get profit config for scale-out
                    tier = sig.features.get("conviction_tier", 4)
                    pc = get_profit_config(strategy.name, tier)

                    # Get BB middle for mean reversion
                    bb_middle = sig.features.get("bb_middle")

                    side = "short" if is_short else "long"

                    positions[sig.symbol] = _Position(
                        symbol=sig.symbol,
                        shares=shares,
                        entry_price=exec_price,
                        stop_loss=sig.stop_loss,
                        entry_date=current_date,
                        strategy=strategy.name,
                        side=side,
                        atr_at_entry=atr,
                        trail_stop_atr_mult=pc.trail_stop_atr_mult,
                        high_water_mark=exec_price,
                        low_water_mark=exec_price,
                        initial_shares=shares,
                        first_target_pct=pc.first_target_pct,
                        partial_sell_pct=pc.partial_sell_pct,
                        second_target_pct=pc.second_target_pct,
                        second_sell_pct=pc.second_sell_pct,
                        exit_at_bb_middle=pc.exit_at_bb_middle,
                        bb_middle_at_entry=bb_middle,
                        time_exit_days=pc.time_exit_days,
                        entry_day_idx=day_idx,
                    )
                    cash -= cost

                    if is_short:
                        short_count += 1

                # Mark-to-market
                position_value = 0.0
                for sym, pos in positions.items():
                    p = self._get_price(data, sym, current_date)
                    if p is not None:
                        if pos.side == "short":
                            # Short MTM: initial_value + (entry - current) * shares
                            initial_value = pos.shares * pos.entry_price
                            unrealized = (pos.entry_price - p) * pos.shares
                            position_value += initial_value + unrealized
                        else:
                            position_value += pos.shares * p

                equity = cash + position_value
                equity_history.append((current_date, equity))

            walk_forward_windows.append(window_info)
            start_idx += c.test_window

        # Close any remaining positions at last available price
        last_date = all_dates[-1]
        for sym in list(positions.keys()):
            pos = positions[sym]
            exit_price = self._get_price(data, sym, last_date)
            if exit_price:
                pnl, pnl_pct, exec_price = self._compute_pnl(pos, exit_price)
                all_trades.append(BacktestTrade(
                    entry_date=pos.entry_date,
                    exit_date=last_date,
                    symbol=sym,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exec_price,
                    shares=pos.shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    strategy_name=strategy.name,
                    stop_loss=pos.stop_loss,
                    exit_reason="end_of_backtest",
                ))
                cash += self._close_value(pos, exec_price)

        # Build results
        eq_series = pd.Series(
            {d: v for d, v in equity_history},
            name="equity",
        )

        trades_df = pd.DataFrame([
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "strategy": t.strategy_name,
                "stop_loss": t.stop_loss,
                "exit_reason": t.exit_reason,
                "is_partial": t.is_partial,
            }
            for t in all_trades
        ])

        metrics = compute_metrics(eq_series, trades_df)

        return BacktestResult(
            metrics=metrics,
            equity_curve=eq_series,
            trades=trades_df,
            config=c,
            walk_forward_windows=walk_forward_windows,
        )

    # ------------------------------------------------------------------
    # Exit management
    # ------------------------------------------------------------------

    def _check_exits(
        self,
        positions: dict[str, _Position],
        data: dict[str, pd.DataFrame],
        current_date: datetime,
        day_idx: int,
        trades: list[BacktestTrade],
        config: BacktestConfig,
        equity: float,
    ) -> float:
        """Check all exit conditions: hard stop, trailing stop, profit targets,
        BB middle, time exit.

        Returns cash freed from closed/partially closed positions.
        """
        freed_cash = 0.0
        to_close: list[str] = []

        for sym, pos in positions.items():
            if sym not in data or current_date not in data[sym].index:
                continue

            row = data[sym].loc[current_date]
            current_price = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])

            # Update water marks
            if pos.side == "short":
                pos.low_water_mark = min(pos.low_water_mark, low)
            else:
                pos.high_water_mark = max(pos.high_water_mark, high)

            # Compute P&L %
            if pos.side == "short":
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price
            else:
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price

            exit_reason = None
            exit_price = current_price
            exit_shares = pos.shares  # full by default

            # === 1. Hard stop (8% max loss) ===
            if pnl_pct <= -HARD_STOP_PCT:
                exit_reason = "hard_stop"
                # Use stop price, not close price
                if pos.side == "short":
                    exit_price = pos.entry_price * (1 + HARD_STOP_PCT)
                else:
                    exit_price = pos.entry_price * (1 - HARD_STOP_PCT)

            # === 2. ATR trailing stop (dynamically tightened) ===
            if not exit_reason and pos.atr_at_entry > 0:
                if config.enable_dynamic_trailing:
                    effective_mult = _dynamic_trail_mult(
                        pos.trail_stop_atr_mult, max(0, pnl_pct)
                    )
                else:
                    effective_mult = pos.trail_stop_atr_mult

                trail_distance = effective_mult * pos.atr_at_entry

                if pos.side == "short":
                    if pos.low_water_mark < pos.entry_price:
                        trail_stop = pos.low_water_mark + trail_distance
                        if high >= trail_stop:
                            exit_reason = "atr_trailing_stop"
                            exit_price = trail_stop
                else:
                    if pos.high_water_mark > pos.entry_price:
                        trail_stop = pos.high_water_mark - trail_distance
                        if low <= trail_stop:
                            exit_reason = "atr_trailing_stop"
                            exit_price = trail_stop

            # === 3. Original stop loss (from signal) ===
            if not exit_reason and pos.stop_loss is not None:
                if pos.side == "short":
                    if high >= pos.stop_loss:
                        exit_reason = "stop_loss"
                        exit_price = pos.stop_loss
                else:
                    if low <= pos.stop_loss:
                        exit_reason = "stop_loss"
                        exit_price = pos.stop_loss

            # === 4. BB middle exit (mean reversion, 100% close) ===
            if (
                not exit_reason
                and pos.exit_at_bb_middle
                and pos.bb_middle_at_entry is not None
            ):
                if pos.side == "short":
                    if current_price <= pos.bb_middle_at_entry:
                        exit_reason = "bb_middle"
                else:
                    if current_price >= pos.bb_middle_at_entry:
                        exit_reason = "bb_middle"

            # === 5. First profit target (partial scale-out) ===
            if (
                not exit_reason
                and config.enable_scale_out
                and not pos.first_target_hit
                and pos.first_target_pct > 0
                and pnl_pct >= pos.first_target_pct
            ):
                sell_qty = int(pos.shares * pos.partial_sell_pct)
                if sell_qty >= 1 and sell_qty < pos.shares:
                    # Partial exit
                    pnl, pnl_p, exec_p = self._compute_pnl(
                        pos, current_price, shares_override=sell_qty
                    )
                    trades.append(BacktestTrade(
                        entry_date=pos.entry_date,
                        exit_date=current_date,
                        symbol=sym,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=exec_p,
                        shares=sell_qty,
                        pnl=pnl,
                        pnl_pct=pnl_p,
                        strategy_name=pos.strategy,
                        stop_loss=pos.stop_loss,
                        exit_reason="first_target",
                        is_partial=True,
                    ))
                    freed_cash += self._close_value(
                        pos, exec_p, shares_override=sell_qty
                    )
                    pos.shares -= sell_qty
                    pos.first_target_hit = True
                    continue  # skip further checks this bar

            # === 6. Second profit target (partial scale-out) ===
            if (
                not exit_reason
                and config.enable_scale_out
                and pos.first_target_hit
                and not pos.second_target_hit
                and pos.second_target_pct > 0
                and pnl_pct >= pos.second_target_pct
            ):
                sell_qty = int(pos.shares * pos.second_sell_pct)
                if sell_qty >= 1 and sell_qty < pos.shares:
                    pnl, pnl_p, exec_p = self._compute_pnl(
                        pos, current_price, shares_override=sell_qty
                    )
                    trades.append(BacktestTrade(
                        entry_date=pos.entry_date,
                        exit_date=current_date,
                        symbol=sym,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=exec_p,
                        shares=sell_qty,
                        pnl=pnl,
                        pnl_pct=pnl_p,
                        strategy_name=pos.strategy,
                        stop_loss=pos.stop_loss,
                        exit_reason="second_target",
                        is_partial=True,
                    ))
                    freed_cash += self._close_value(
                        pos, exec_p, shares_override=sell_qty
                    )
                    pos.shares -= sell_qty
                    pos.second_target_hit = True
                    continue

            # === 7. Time-based exit ===
            if not exit_reason and pos.time_exit_days is not None:
                hold_days = day_idx - pos.entry_day_idx
                if hold_days >= pos.time_exit_days:
                    exit_reason = "time_exit"

            # === Execute full close if triggered ===
            if exit_reason:
                pnl, pnl_pct_val, exec_price = self._compute_pnl(
                    pos, exit_price
                )
                trades.append(BacktestTrade(
                    entry_date=pos.entry_date,
                    exit_date=current_date,
                    symbol=sym,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exec_price,
                    shares=pos.shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct_val,
                    strategy_name=pos.strategy,
                    stop_loss=pos.stop_loss,
                    exit_reason=exit_reason,
                ))
                freed_cash += self._close_value(pos, exec_price)
                to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        return freed_cash

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_pnl(
        self,
        pos: _Position,
        exit_price: float,
        shares_override: int | None = None,
    ) -> tuple[float, float, float]:
        """Compute P&L for a position exit.

        Returns (pnl_dollar, pnl_pct, execution_price).
        """
        shares = shares_override if shares_override is not None else pos.shares
        sell_side = "buy" if pos.side == "short" else "sell"
        exec_price = self.cost_calc.calculate_execution_price(
            exit_price, sell_side
        )

        if pos.side == "short":
            pnl = (pos.entry_price - exec_price) * shares
        else:
            pnl = (exec_price - pos.entry_price) * shares

        pnl_pct = pnl / (pos.entry_price * shares) if pos.entry_price > 0 else 0.0
        return pnl, pnl_pct, exec_price

    def _close_value(
        self,
        pos: _Position,
        exec_price: float,
        shares_override: int | None = None,
    ) -> float:
        """Cash returned when closing (part of) a position."""
        shares = shares_override if shares_override is not None else pos.shares
        if pos.side == "short":
            # Short close: return initial margin + P&L
            initial = shares * pos.entry_price
            pnl = (pos.entry_price - exec_price) * shares
            return initial + pnl
        else:
            return shares * exec_price

    def _compute_atr(
        self,
        data: dict[str, pd.DataFrame],
        symbol: str,
        current_date: datetime,
        period: int = 14,
    ) -> float:
        """Compute ATR for a symbol up to current_date."""
        if symbol not in data:
            return 0.0
        df = data[symbol]
        mask = df.index <= current_date
        sliced = df[mask]
        if len(sliced) < period + 1:
            # Fallback: use 5% of price
            return float(sliced.iloc[-1]["close"]) * 0.05 if len(sliced) > 0 else 0.0

        atr_series = ta.volatility.average_true_range(
            sliced["high"], sliced["low"], sliced["close"], window=period
        )
        val = atr_series.iloc[-1]
        return float(val) if not pd.isna(val) else 0.0

    def _get_common_dates(self, data: dict[str, pd.DataFrame]) -> list:
        """Get sorted list of dates present in all DataFrames."""
        if not data:
            return []
        # Use the first symbol's dates as base, then find intersection
        date_sets = [set(df.index) for df in data.values()]
        common = date_sets[0]
        for ds in date_sets[1:]:
            common = common.intersection(ds)
        return sorted(common)

    def _get_price(
        self, data: dict[str, pd.DataFrame], symbol: str, date: datetime
    ) -> float | None:
        """Get closing price for a symbol on a specific date."""
        if symbol not in data:
            return None
        df = data[symbol]
        if date in df.index:
            return float(df.loc[date, "close"])
        # Find nearest previous date
        mask = df.index <= date
        if mask.any():
            return float(df[mask].iloc[-1]["close"])
        return None
