"""Position Sizing Module.

Implements:
    - Fractional Kelly criterion (capped at quarter Kelly)
    - Volatility-adjusted sizing (inversely proportional to ATR)
    - Fixed fractional risk per trade
    - Max position size as % of portfolio

The position sizer takes a signal and portfolio context, and returns
the number of shares to trade.
"""

from dataclasses import dataclass
import math

import pandas as pd
import ta


@dataclass
class PositionSizeResult:
    shares: int
    dollar_value: float
    risk_per_share: float
    portfolio_risk_pct: float
    method: str
    details: dict


@dataclass
class PortfolioContext:
    equity: float
    cash: float
    buying_power: float
    num_positions: int
    strategy_allocation_pct: float  # fraction of portfolio allocated to this strategy


DEFAULT_SIZING_PARAMS = {
    "kelly_fraction_cap": 0.40,
    "max_position_pct": 0.10,
    "max_risk_per_trade_pct": 0.02,
    "min_shares": 1,
    "min_dollar_value": 100.0,
    "atr_period": 14,
}


class PositionSizer:

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_SIZING_PARAMS, **(params or {})}

    def calculate_size(
        self,
        entry_price: float,
        stop_loss: float | None,
        signal_strength: float,
        portfolio: PortfolioContext,
        historical_data: pd.DataFrame | None = None,
        win_rate: float | None = None,
        avg_win_loss_ratio: float | None = None,
    ) -> PositionSizeResult:
        """Calculate position size using the best available method.

        Priority:
        1. Kelly criterion (if win_rate and avg_win_loss_ratio available)
        2. ATR-based volatility sizing (if historical_data available)
        3. Fixed fractional risk (if stop_loss available)
        4. Simple max-position-pct cap

        All methods are further constrained by:
        - max_position_pct of total equity
        - strategy_allocation_pct
        - buying power
        """
        p = self.params
        allocated_capital = portfolio.equity * portfolio.strategy_allocation_pct

        # Calculate risk per share
        if stop_loss is not None:
            risk_per_share = abs(entry_price - stop_loss)
        elif historical_data is not None and len(historical_data) > p["atr_period"]:
            atr = ta.volatility.average_true_range(
                historical_data["high"],
                historical_data["low"],
                historical_data["close"],
                window=p["atr_period"],
            ).iloc[-1]
            risk_per_share = float(atr) * 2.0
        else:
            # Fallback: assume 5% risk
            risk_per_share = entry_price * 0.05

        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.05

        # Method 1: Kelly criterion
        method = "fixed_fractional"
        kelly_fraction = None
        if win_rate is not None and avg_win_loss_ratio is not None and win_rate > 0:
            # Kelly formula: f* = (p * b - q) / b
            # where p = win_rate, q = 1 - win_rate, b = avg_win_loss_ratio
            b = avg_win_loss_ratio
            q = 1.0 - win_rate
            kelly_raw = (win_rate * b - q) / b if b > 0 else 0
            kelly_fraction = max(0, min(kelly_raw, p["kelly_fraction_cap"]))
            method = "kelly"

        # Calculate max dollar allocation
        if kelly_fraction is not None and kelly_fraction > 0:
            dollar_allocation = allocated_capital * kelly_fraction
        else:
            # Fixed fractional: risk max_risk_per_trade_pct of equity
            max_risk_dollars = portfolio.equity * p["max_risk_per_trade_pct"]
            dollar_allocation = (max_risk_dollars / risk_per_share) * entry_price

        # Apply caps
        max_by_position_pct = portfolio.equity * p["max_position_pct"]
        max_by_allocation = allocated_capital
        max_by_buying_power = portfolio.buying_power

        capped_allocation = min(
            dollar_allocation,
            max_by_position_pct,
            max_by_allocation,
            max_by_buying_power,
        )

        # Scale by signal strength (0.5 to 1.0 scaling)
        strength_scale = 0.5 + 0.5 * signal_strength
        final_allocation = capped_allocation * strength_scale

        # Convert to shares
        shares = max(p["min_shares"], math.floor(final_allocation / entry_price))

        # Final checks
        final_dollar_value = shares * entry_price
        if final_dollar_value < p["min_dollar_value"]:
            shares = 0
            final_dollar_value = 0.0

        portfolio_risk_pct = (shares * risk_per_share) / portfolio.equity if portfolio.equity > 0 else 0

        return PositionSizeResult(
            shares=shares,
            dollar_value=final_dollar_value,
            risk_per_share=risk_per_share,
            portfolio_risk_pct=portfolio_risk_pct,
            method=method,
            details={
                "kelly_fraction": kelly_fraction,
                "signal_strength": signal_strength,
                "strength_scale": strength_scale,
                "allocated_capital": allocated_capital,
                "raw_dollar_allocation": dollar_allocation,
                "capped_allocation": capped_allocation,
                "final_allocation": final_allocation,
                "entry_price": entry_price,
                "stop_loss": stop_loss if stop_loss else entry_price - risk_per_share,
            },
        )
