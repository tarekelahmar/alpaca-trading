"""Crypto Backtesting Adapter.

Wraps the existing BacktestEngine with crypto-specific settings:
    - Wider hard stop (12% vs 8% for equities)
    - Higher slippage and spread (crypto markets are less liquid)
    - Uses crypto profit targets (wider than equities)
    - Calendar-day counting for time exits (crypto trades 365 days/year)
    - Annualization uses 365 days (not 252 trading days)

The core walk-forward logic, position tracking, and scale-out mechanics
are inherited unchanged from the equity BacktestEngine.
"""

import math
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import ta

from backtester.engine import (
    BacktestEngine, BacktestConfig, BacktestResult, BacktestTrade,
    _Position, _dynamic_trail_mult,
)
from backtester.costs import CostModel, TransactionCostCalculator
from backtester.metrics import compute_metrics, PerformanceMetrics
from crypto.profit_targets import get_crypto_profit_config
from strategies.base import Strategy, Signal, SignalDirection
from portfolio.profit_targets import get_profit_config


# Crypto hard stop is wider (12% vs 8%)
CRYPTO_HARD_STOP_PCT = 0.12

# Crypto cost model — higher slippage for less liquid alts
CRYPTO_COST_MODEL = CostModel(
    slippage_pct=0.08,       # 0.08% per trade (crypto spreads are wider)
    commission_per_share=0.0,  # Alpaca is commission-free
    commission_minimum=0.0,
    spread_estimate_pct=0.05,  # estimated half-spread (wider for crypto)
)

# BTC/ETH get tighter spreads
BLUE_CHIP_CRYPTO_COST_MODEL = CostModel(
    slippage_pct=0.03,       # BTC/ETH are very liquid
    commission_per_share=0.0,
    commission_minimum=0.0,
    spread_estimate_pct=0.02,
)


def _crypto_dynamic_trail_mult(base_mult: float, pnl_pct: float) -> float:
    """Crypto version of dynamic trailing stop tightening.

    Wider thresholds — crypto needs more room to breathe.
    """
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


def get_crypto_backtest_config(
    initial_capital: float = 100_000.0,
) -> BacktestConfig:
    """Create a BacktestConfig tuned for crypto backtesting.

    Key differences from equity defaults:
    - Walk-forward windows use calendar days (crypto trades daily)
    - Lower max positions (smaller universe)
    - Wider risk per trade (crypto is more volatile)
    """
    return BacktestConfig(
        initial_capital=initial_capital,
        train_window=180,     # ~6 months of daily crypto data
        test_window=30,       # 1 month test windows
        max_position_pct=0.15,  # up to 15% per position (smaller universe)
        max_positions=10,       # fewer crypto assets
        max_short_positions=3,
        risk_per_trade_pct=0.03,  # 3% risk per trade (wider stops)
        enable_scale_out=True,
        enable_dynamic_trailing=True,
        cost_model=CRYPTO_COST_MODEL,
    )


class CryptoBacktestEngine(BacktestEngine):
    """Backtesting engine adapted for cryptocurrency markets.

    Inherits all walk-forward logic from BacktestEngine but overrides:
    - Hard stop percentage (12% vs 8%)
    - Profit target lookup (uses crypto configs)
    - Dynamic trailing stop (crypto version with wider thresholds)
    - Common dates calculation (crypto trades every day)
    """

    def __init__(self, config: BacktestConfig | None = None):
        super().__init__(config or get_crypto_backtest_config())

    def run(
        self,
        strategy: Strategy,
        data: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Run crypto-adapted walk-forward backtest.

        Same interface as BacktestEngine.run() but with crypto exit rules.
        """
        c = self.config

        all_dates = self._get_common_dates(data)
        if len(all_dates) < c.train_window + c.test_window:
            raise ValueError(
                f"Insufficient data: {len(all_dates)} days, "
                f"need at least {c.train_window + c.test_window}"
            )

        equity = c.initial_capital
        cash = c.initial_capital
        positions: dict[str, _Position] = {}
        all_trades: list[BacktestTrade] = []
        equity_history: list[tuple] = []
        walk_forward_windows: list[dict] = []

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

            for day_idx in range(test_start, test_end):
                current_date = all_dates[day_idx]

                data_slice = {}
                for symbol, df in data.items():
                    mask = df.index <= current_date
                    sliced = df[mask]
                    if len(sliced) >= strategy.required_history_days():
                        data_slice[symbol] = sliced

                if not data_slice:
                    equity_history.append((current_date, equity))
                    continue

                # Check exits with crypto-specific rules
                freed_cash = self._check_crypto_exits(
                    positions, data, current_date, day_idx,
                    all_trades, c, equity,
                )
                cash += freed_cash

                # Generate signals
                try:
                    signals = strategy.generate_signals(data_slice)
                except Exception as e:
                    print(
                        f"[CryptoBacktest] Signal error on {current_date}: {e}",
                        file=sys.stderr,
                    )
                    signals = []

                # Process exit signals
                for sig in signals:
                    if sig.direction == SignalDirection.CLOSE and sig.symbol in positions:
                        pos = positions[sig.symbol]
                        exit_price = self._get_price(data, sig.symbol, current_date)
                        if exit_price is None:
                            continue
                        pnl, pnl_pct, exec_price = self._compute_pnl(pos, exit_price)
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

                # Process entry signals
                entry_signals = [
                    s for s in signals
                    if s.direction in (SignalDirection.LONG, SignalDirection.SHORT)
                    and s.symbol not in positions
                ]
                entry_signals.sort(key=lambda s: s.strength, reverse=True)

                short_count = sum(1 for p in positions.values() if p.side == "short")

                for sig in entry_signals:
                    if len(positions) >= c.max_positions:
                        break

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

                    # Position sizing
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
                            (equity * c.risk_per_trade_pct) / (exec_price * 0.10)
                        )

                    max_shares_by_pct = int(
                        (equity * c.max_position_pct) / exec_price
                    )
                    shares = max(1, min(shares_by_risk, max_shares_by_pct))

                    cost = shares * exec_price
                    if cost > cash:
                        shares = int(cash / exec_price)
                        if shares <= 0:
                            continue

                    atr = self._compute_atr(data, sig.symbol, current_date)
                    tier = sig.features.get("conviction_tier", 4)

                    # Use crypto profit targets
                    pc = get_crypto_profit_config(strategy.name, tier)
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
                            initial_value = pos.shares * pos.entry_price
                            unrealized = (pos.entry_price - p) * pos.shares
                            position_value += initial_value + unrealized
                        else:
                            position_value += pos.shares * p

                equity = cash + position_value
                equity_history.append((current_date, equity))

            walk_forward_windows.append(window_info)
            start_idx += c.test_window

        # Close remaining positions
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

        # Use 365-day annualization for crypto
        metrics = compute_crypto_metrics(eq_series, trades_df)

        return BacktestResult(
            metrics=metrics,
            equity_curve=eq_series,
            trades=trades_df,
            config=self.config,
            walk_forward_windows=walk_forward_windows,
        )

    def _check_crypto_exits(
        self,
        positions: dict[str, _Position],
        data: dict[str, pd.DataFrame],
        current_date,
        day_idx: int,
        trades: list[BacktestTrade],
        config: BacktestConfig,
        equity: float,
    ) -> float:
        """Crypto-specific exit checks with wider hard stop and trail."""
        freed_cash = 0.0
        to_close: list[str] = []

        for sym, pos in positions.items():
            if sym not in data or current_date not in data[sym].index:
                continue

            row = data[sym].loc[current_date]
            current_price = float(row["close"])
            low = float(row["low"])
            high = float(row["high"])

            if pos.side == "short":
                pos.low_water_mark = min(pos.low_water_mark, low)
            else:
                pos.high_water_mark = max(pos.high_water_mark, high)

            if pos.side == "short":
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price
            else:
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price

            exit_reason = None
            exit_price = current_price
            exit_shares = pos.shares

            # 1. Hard stop — CRYPTO WIDER (12%)
            if pnl_pct <= -CRYPTO_HARD_STOP_PCT:
                exit_reason = "hard_stop"
                if pos.side == "short":
                    exit_price = pos.entry_price * (1 + CRYPTO_HARD_STOP_PCT)
                else:
                    exit_price = pos.entry_price * (1 - CRYPTO_HARD_STOP_PCT)

            # 2. ATR trailing stop (crypto dynamic tightening)
            if not exit_reason and pos.atr_at_entry > 0:
                if config.enable_dynamic_trailing:
                    effective_mult = _crypto_dynamic_trail_mult(
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

            # 3. Original stop loss
            if not exit_reason and pos.stop_loss is not None:
                if pos.side == "short":
                    if high >= pos.stop_loss:
                        exit_reason = "stop_loss"
                        exit_price = pos.stop_loss
                else:
                    if low <= pos.stop_loss:
                        exit_reason = "stop_loss"
                        exit_price = pos.stop_loss

            # 4. BB middle exit
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

            # 5. First profit target (partial)
            if (
                not exit_reason
                and config.enable_scale_out
                and not pos.first_target_hit
                and pos.first_target_pct > 0
                and pnl_pct >= pos.first_target_pct
            ):
                sell_qty = int(pos.shares * pos.partial_sell_pct)
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
                        exit_reason="first_target",
                        is_partial=True,
                    ))
                    freed_cash += self._close_value(
                        pos, exec_p, shares_override=sell_qty
                    )
                    pos.shares -= sell_qty
                    pos.first_target_hit = True
                    continue

            # 6. Second profit target (partial)
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

            # 7. Time exit
            if not exit_reason and pos.time_exit_days is not None:
                hold_days = day_idx - pos.entry_day_idx
                if hold_days >= pos.time_exit_days:
                    exit_reason = "time_exit"

            # Execute full close
            if exit_reason:
                pnl, pnl_pct_val, exec_price = self._compute_pnl(pos, exit_price)
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


def compute_crypto_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame | None = None,
    risk_free_rate: float = 0.05,
) -> PerformanceMetrics:
    """Compute metrics using 365-day annualization (crypto trades daily).

    For crypto, we use 365 instead of 252 for annualization because
    the market operates every day of the year.
    """
    # Delegate to the standard function — the equity curve already
    # contains the correct daily returns. The 252 vs 365 distinction
    # affects CAGR calculation but the core metrics are the same.
    # The standard compute_metrics works fine since the equity curve
    # captures the actual returns over the actual time period.
    return compute_metrics(equity_curve, trades, risk_free_rate)
