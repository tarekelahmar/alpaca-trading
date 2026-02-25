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
    gap_trading: float = 0.0  # gap trading strategy allocation
    sector_rotation: float = 0.0  # sector rotation strategy allocation

    def normalize(self) -> "AllocationWeights":
        """Ensure weights sum to 1.0."""
        total = (
            self.trend_following + self.mean_reversion
            + self.momentum + self.cash + self.sentiment
            + self.earnings_momentum + self.gap_trading
            + self.sector_rotation
        )
        if total == 0:
            return AllocationWeights(0, 0, 0, 1.0, 0, 0, 0, 0)
        return AllocationWeights(
            trend_following=self.trend_following / total,
            mean_reversion=self.mean_reversion / total,
            momentum=self.momentum / total,
            cash=self.cash / total,
            sentiment=self.sentiment / total,
            earnings_momentum=self.earnings_momentum / total,
            gap_trading=self.gap_trading / total,
            sector_rotation=self.sector_rotation / total,
        )

    def invested_fraction(self) -> float:
        return 1.0 - self.cash


# Default allocation rules per regime
# 8 strategies: TF, MR, MOM, SENT, EARN, GAP, SEC_ROT, Cash
REGIME_ALLOCATIONS: dict[RegimeType, AllocationWeights] = {
    RegimeType.TRENDING_BULLISH: AllocationWeights(
        trend_following=0.20,
        mean_reversion=0.05,
        momentum=0.20,
        cash=0.05,
        sentiment=0.15,
        earnings_momentum=0.15,
        gap_trading=0.05,
        sector_rotation=0.15,  # sector rotation thrives in trending markets
    ),
    RegimeType.TRENDING_BEARISH: AllocationWeights(
        trend_following=0.12,
        mean_reversion=0.08,
        momentum=0.08,
        cash=0.40,
        sentiment=0.10,
        earnings_momentum=0.07,
        gap_trading=0.05,  # gap fills still work in bear markets
        sector_rotation=0.10,  # defensive sector rotation (utilities, staples)
    ),
    RegimeType.RANGING: AllocationWeights(
        trend_following=0.08,
        mean_reversion=0.25,
        momentum=0.08,
        cash=0.10,
        sentiment=0.18,
        earnings_momentum=0.12,
        gap_trading=0.10,  # gap fills work well in ranging markets
        sector_rotation=0.09,
    ),
    RegimeType.HIGH_VOLATILITY: AllocationWeights(
        trend_following=0.08,
        mean_reversion=0.08,
        momentum=0.05,
        cash=0.40,
        sentiment=0.10,
        earnings_momentum=0.10,
        gap_trading=0.10,  # gaps are larger and more frequent in high vol
        sector_rotation=0.09,
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
            return AllocationWeights(0.05, 0.05, 0.05, 0.65, 0.05, 0.05, 0.05, 0.05)

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
                gap_trading=weights.gap_trading * scale,
                sector_rotation=weights.sector_rotation * scale,
            )

        return weights.normalize()
