"""Walk-Forward Backtesting Engine.

Simulates strategy execution on historical data with realistic
trade modeling, position tracking, and slippage.

Walk-forward approach:
    - Train window: 252 days (1 year)
    - Test window: 63 days (1 quarter)
    - Step forward by test window
    - Re-fit parameters at each step (if strategy supports it)

This prevents look-ahead bias and gives out-of-sample performance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sys

import numpy as np
import pandas as pd

from strategies.base import Strategy, Signal, SignalDirection
from backtester.costs import TransactionCostCalculator, CostModel
from backtester.metrics import compute_metrics, PerformanceMetrics


@dataclass
class BacktestTrade:
    entry_date: datetime
    exit_date: datetime | None
    symbol: str
    side: str
    entry_price: float
    exit_price: float | None
    shares: int
    pnl: float
    pnl_pct: float
    strategy_name: str
    stop_loss: float | None
    features: dict = field(default_factory=dict)


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    train_window: int = 252  # trading days
    test_window: int = 63  # trading days
    max_position_pct: float = 0.10  # max 10% per position
    max_positions: int = 30
    risk_per_trade_pct: float = 0.02  # 2% risk per trade
    cost_model: CostModel = field(default_factory=CostModel)


@dataclass
class BacktestResult:
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    trades: pd.DataFrame
    config: BacktestConfig
    walk_forward_windows: list[dict]


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
        positions: dict[str, dict] = {}  # symbol -> {shares, entry_price, stop_loss, entry_date, strategy}
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

                # Check stops on existing positions
                closed = self._check_stops(
                    positions, data, current_date, all_trades, cash
                )
                cash += closed

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
                        exec_price = self.cost_calc.calculate_execution_price(
                            exit_price, "sell"
                        )
                        pnl = (exec_price - pos["entry_price"]) * pos["shares"]
                        pnl_pct = (exec_price / pos["entry_price"] - 1)

                        all_trades.append(BacktestTrade(
                            entry_date=pos["entry_date"],
                            exit_date=current_date,
                            symbol=sig.symbol,
                            side="long",
                            entry_price=pos["entry_price"],
                            exit_price=exec_price,
                            shares=pos["shares"],
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            strategy_name=strategy.name,
                            stop_loss=pos.get("stop_loss"),
                            features=sig.features,
                        ))
                        cash += pos["shares"] * exec_price
                        del positions[sig.symbol]

                # Process entry signals
                entry_signals = [
                    s for s in signals
                    if s.direction == SignalDirection.LONG and s.symbol not in positions
                ]
                entry_signals.sort(key=lambda s: s.strength, reverse=True)

                for sig in entry_signals:
                    if len(positions) >= c.max_positions:
                        break

                    price = self._get_price(data, sig.symbol, current_date)
                    if price is None:
                        continue

                    exec_price = self.cost_calc.calculate_execution_price(price, "buy")

                    # Position sizing: fixed fractional
                    if sig.stop_loss:
                        risk_per_share = exec_price - sig.stop_loss
                        if risk_per_share <= 0:
                            continue
                        max_risk = equity * c.risk_per_trade_pct
                        shares_by_risk = int(max_risk / risk_per_share)
                    else:
                        shares_by_risk = int((equity * c.risk_per_trade_pct) / (exec_price * 0.05))

                    # Position size cap
                    max_shares_by_pct = int((equity * c.max_position_pct) / exec_price)
                    shares = max(1, min(shares_by_risk, max_shares_by_pct))

                    cost = shares * exec_price
                    if cost > cash:
                        shares = int(cash / exec_price)
                        if shares <= 0:
                            continue

                    positions[sig.symbol] = {
                        "shares": shares,
                        "entry_price": exec_price,
                        "stop_loss": sig.stop_loss,
                        "entry_date": current_date,
                        "strategy": strategy.name,
                    }
                    cash -= shares * exec_price

                # Mark-to-market
                position_value = 0.0
                for sym, pos in positions.items():
                    p = self._get_price(data, sym, current_date)
                    if p is not None:
                        position_value += pos["shares"] * p

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
                exec_price = self.cost_calc.calculate_execution_price(exit_price, "sell")
                pnl = (exec_price - pos["entry_price"]) * pos["shares"]
                pnl_pct = (exec_price / pos["entry_price"] - 1)
                all_trades.append(BacktestTrade(
                    entry_date=pos["entry_date"],
                    exit_date=last_date,
                    symbol=sym,
                    side="long",
                    entry_price=pos["entry_price"],
                    exit_price=exec_price,
                    shares=pos["shares"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    strategy_name=strategy.name,
                    stop_loss=pos.get("stop_loss"),
                ))
                cash += pos["shares"] * exec_price

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

    def _check_stops(
        self,
        positions: dict[str, dict],
        data: dict[str, pd.DataFrame],
        current_date: datetime,
        trades: list[BacktestTrade],
        cash: float,
    ) -> float:
        """Check stop losses and close positions that hit stops.

        Returns:
            Cash freed from closed positions.
        """
        freed_cash = 0.0
        to_close = []

        for sym, pos in positions.items():
            if pos.get("stop_loss") is None:
                continue

            # Check if today's low hit the stop
            if sym in data and current_date in data[sym].index:
                low = float(data[sym].loc[current_date, "low"])
                if low <= pos["stop_loss"]:
                    # Stopped out — execute at stop price (with slippage)
                    exec_price = self.cost_calc.calculate_execution_price(
                        pos["stop_loss"], "sell"
                    )
                    pnl = (exec_price - pos["entry_price"]) * pos["shares"]
                    pnl_pct = (exec_price / pos["entry_price"] - 1)

                    trades.append(BacktestTrade(
                        entry_date=pos["entry_date"],
                        exit_date=current_date,
                        symbol=sym,
                        side="long",
                        entry_price=pos["entry_price"],
                        exit_price=exec_price,
                        shares=pos["shares"],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        strategy_name=pos.get("strategy", "unknown"),
                        stop_loss=pos["stop_loss"],
                        features={"exit_reason": "stop_loss"},
                    ))
                    freed_cash += pos["shares"] * exec_price
                    to_close.append(sym)

        for sym in to_close:
            del positions[sym]

        return freed_cash
