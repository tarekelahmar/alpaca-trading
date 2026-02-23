"""Capital Allocation based on Market Regime.

Maps regime classification to capital allocation weights per strategy type.
"""

from dataclasses import dataclass

from regime.detector import RegimeType


@dataclass
class AllocationWeights:
    trend_following: float
    mean_reversion: float
    momentum: float
    cash: float  # fraction held in cash
    sentiment: float = 0.0  # sentiment strategy allocation
    earnings_momentum: float = 0.0  # earnings momentum strategy allocation

    def normalize(self) -> "AllocationWeights":
        """Ensure weights sum to 1.0."""
        total = (
            self.trend_following + self.mean_reversion
            + self.momentum + self.cash + self.sentiment
            + self.earnings_momentum
        )
        if total == 0:
            return AllocationWeights(0, 0, 0, 1.0, 0, 0)
        return AllocationWeights(
            trend_following=self.trend_following / total,
            mean_reversion=self.mean_reversion / total,
            momentum=self.momentum / total,
            cash=self.cash / total,
            sentiment=self.sentiment / total,
            earnings_momentum=self.earnings_momentum / total,
        )

    def invested_fraction(self) -> float:
        return 1.0 - self.cash


# Default allocation rules per regime
# 6 strategies: TF, MR, MOM, SENT, EARN, Cash
REGIME_ALLOCATIONS: dict[RegimeType, AllocationWeights] = {
    RegimeType.TRENDING_BULLISH: AllocationWeights(
        trend_following=0.25,
        mean_reversion=0.05,
        momentum=0.25,
        cash=0.05,
        sentiment=0.20,
        earnings_momentum=0.20,
    ),
    RegimeType.TRENDING_BEARISH: AllocationWeights(
        trend_following=0.15,
        mean_reversion=0.10,
        momentum=0.10,
        cash=0.40,
        sentiment=0.15,
        earnings_momentum=0.10,
    ),
    RegimeType.RANGING: AllocationWeights(
        trend_following=0.10,
        mean_reversion=0.30,
        momentum=0.10,
        cash=0.10,
        sentiment=0.25,
        earnings_momentum=0.15,
    ),
    RegimeType.HIGH_VOLATILITY: AllocationWeights(
        trend_following=0.10,
        mean_reversion=0.10,
        momentum=0.10,
        cash=0.40,
        sentiment=0.15,
        earnings_momentum=0.15,
    ),
}


class RegimeAllocator:

    def __init__(
        self,
        allocations: dict[RegimeType, AllocationWeights] | None = None,
        max_invested: float = 0.95,
    ):
        self.allocations = allocations or REGIME_ALLOCATIONS
        self.max_invested = max_invested

    def get_allocation(self, regime: RegimeType) -> AllocationWeights:
        """Get capital allocation weights for a given regime.

        Returns normalized weights that sum to 1.0, with the invested
        fraction capped at max_invested.
        """
        weights = self.allocations.get(regime)
        if weights is None:
            # Unknown regime — go mostly cash
            return AllocationWeights(0.05, 0.05, 0.05, 0.75, 0.05, 0.05)

        # Enforce max invested constraint
        invested = weights.invested_fraction()
        if invested > self.max_invested:
            scale = self.max_invested / invested
            weights = AllocationWeights(
                trend_following=weights.trend_following * scale,
                mean_reversion=weights.mean_reversion * scale,
                momentum=weights.momentum * scale,
                cash=1.0 - self.max_invested,
                sentiment=weights.sentiment * scale,
                earnings_momentum=weights.earnings_momentum * scale,
            )

        return weights.normalize()
