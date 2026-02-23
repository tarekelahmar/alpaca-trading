"""Transaction Cost and Slippage Modeling.

Models realistic execution costs for backtesting:
    - Slippage (market impact)
    - Commissions (zero for Alpaca)
    - Spread cost estimate
"""

from dataclasses import dataclass


@dataclass
class CostModel:
    slippage_pct: float = 0.05  # 0.05% per trade (conservative for daily)
    commission_per_share: float = 0.0  # Alpaca is commission-free
    commission_minimum: float = 0.0
    spread_estimate_pct: float = 0.02  # estimated half-spread cost


class TransactionCostCalculator:

    def __init__(self, model: CostModel | None = None):
        self.model = model or CostModel()

    def calculate_execution_price(
        self,
        signal_price: float,
        side: str,
        volume: float | None = None,
        shares: int = 0,
    ) -> float:
        """Calculate the estimated execution price after slippage.

        For buys, the execution price is higher than signal price.
        For sells, the execution price is lower.

        Args:
            signal_price: Price at signal generation.
            side: "buy" or "sell".
            volume: Average daily volume (for volume-adjusted impact).
            shares: Number of shares being traded.

        Returns:
            Estimated execution price.
        """
        m = self.model
        total_impact_pct = m.slippage_pct + m.spread_estimate_pct

        # Volume-adjusted market impact
        if volume and volume > 0 and shares > 0:
            participation_rate = shares / volume
            if participation_rate > 0.01:
                # Square root market impact model
                extra_impact = 0.1 * (participation_rate ** 0.5) * 100
                total_impact_pct += extra_impact

        impact_multiplier = total_impact_pct / 100

        if side == "buy":
            return signal_price * (1 + impact_multiplier)
        else:
            return signal_price * (1 - impact_multiplier)

    def calculate_commission(self, shares: int, price: float) -> float:
        """Calculate commission for a trade."""
        m = self.model
        commission = shares * m.commission_per_share
        return max(commission, m.commission_minimum)

    def total_cost(
        self,
        signal_price: float,
        execution_price: float,
        shares: int,
    ) -> float:
        """Total cost of trade including slippage and commission."""
        slippage_cost = abs(execution_price - signal_price) * shares
        commission = self.calculate_commission(shares, execution_price)
        return slippage_cost + commission
