"""Signal Generator — runs all active strategies and collects signals.

This is the main orchestration point for signal generation.
It runs each strategy against the provided data and returns
a unified list of signals, tagged with their source strategy.

Allocation weights are stored in each signal's features dict
so that position sizing can use them later. Signal strength
is NOT scaled by allocation weight here — strength represents
signal quality / conviction, while allocation controls position
sizing downstream.
"""

import sys

import pandas as pd

from strategies.base import Signal, Strategy
from regime.allocator import AllocationWeights


STRATEGY_TYPE_MAP = {
    "trend_following": "trend_following",
    "mean_reversion": "mean_reversion",
    "momentum": "momentum",
    "sentiment": "sentiment",
    "earnings_momentum": "earnings_momentum",
    "btc_dominance": "sector_rotation",  # maps to sector_rotation slot in equity format
}


class SignalGenerator:

    def __init__(self, strategies: list[Strategy]):
        self.strategies = strategies

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        allocation: AllocationWeights | None = None,
    ) -> list[Signal]:
        """Run all strategies and return aggregated signals.

        Args:
            data: Dict mapping symbol -> OHLCV DataFrame.
            allocation: Regime-based allocation weights. If provided,
                       signals from strategies with 0 allocation are skipped.
                       Allocation weight is stored in signal features for
                       downstream position sizing (NOT used to scale strength).

        Returns:
            List of all signals from all strategies.
        """
        all_signals: list[Signal] = []

        for strategy in self.strategies:
            # Check if this strategy type has allocation
            strategy_type = STRATEGY_TYPE_MAP.get(strategy.name, strategy.name)
            weight = 1.0
            if allocation is not None:
                weight = getattr(allocation, strategy_type, 0.0)
                if weight <= 0.01:
                    print(
                        f"[SignalGen] Skipping {strategy.name} — "
                        f"zero allocation in current regime",
                        file=sys.stderr,
                    )
                    continue

            try:
                signals = strategy.generate_signals(data)
                # Store allocation weight in features for position sizing
                # Signal strength stays as-is (represents signal quality)
                for sig in signals:
                    sig.features["allocation_weight"] = weight
                all_signals.extend(signals)
                print(
                    f"[SignalGen] {strategy.name}: {len(signals)} signals "
                    f"(weight={weight:.2f})",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"[SignalGen] ERROR in {strategy.name}: {e}",
                    file=sys.stderr,
                )

        return all_signals
