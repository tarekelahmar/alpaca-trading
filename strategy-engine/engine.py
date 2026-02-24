"""Main Strategy Engine Orchestrator.

This is the top-level coordinator that:
1. Fetches market data
2. Detects the current regime
3. Determines capital allocation
4. Runs all active strategies
5. Filters and ranks signals
6. Sizes positions
7. Generates order intents

It does NOT execute orders — that's the broker server's job.
This engine is called by the daily execution script.
"""

import sys
from datetime import datetime

import pandas as pd

from strategies.base import Strategy, StrategyConfig
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.sentiment import SentimentStrategy
from strategies.earnings_momentum import EarningsMomentumStrategy
from regime.detector import RegimeDetector, RegimeClassification
from regime.allocator import RegimeAllocator, AllocationWeights
from signals.generator import SignalGenerator
from signals.filter import SignalFilter
from portfolio.sizing import PositionSizer, PortfolioContext
from portfolio.optimizer import PortfolioOptimizer, PortfolioOptimizationConfig, OrderIntent
from portfolio.universe import UniverseSelector


class StrategyEngine:
    """Main orchestrator for the trading system."""

    def __init__(
        self,
        strategies: list[Strategy] | None = None,
        universe: UniverseSelector | None = None,
        regime_detector: RegimeDetector | None = None,
        regime_allocator: RegimeAllocator | None = None,
        signal_filter: SignalFilter | None = None,
        position_sizer: PositionSizer | None = None,
        portfolio_optimizer: PortfolioOptimizer | None = None,
    ):
        self.strategies = strategies or [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            MomentumStrategy(),
            SentimentStrategy(),
            EarningsMomentumStrategy(),
        ]
        self.universe = universe or UniverseSelector()
        self.regime_detector = regime_detector or RegimeDetector()
        self.regime_allocator = regime_allocator or RegimeAllocator()
        self.signal_generator = SignalGenerator(self.strategies)
        self.signal_filter = signal_filter or SignalFilter()
        self.position_sizer = position_sizer or PositionSizer()
        self.portfolio_optimizer = portfolio_optimizer or PortfolioOptimizer()

    def run(
        self,
        data: dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
        portfolio: PortfolioContext,
        current_positions: dict[str, dict],
        vix_data: pd.DataFrame | None = None,
        breadth_pct: float | None = None,
    ) -> tuple[list[OrderIntent], RegimeClassification, AllocationWeights]:
        """Run the full strategy pipeline.

        Args:
            data: Dict mapping symbol -> OHLCV DataFrame (already filtered by universe).
            spy_data: SPY OHLCV DataFrame for regime detection.
            portfolio: Current portfolio context.
            current_positions: Dict of current open positions.
            vix_data: Optional VIX data for regime detection.
            breadth_pct: Optional market breadth percentage.

        Returns:
            Tuple of (order_intents, regime_classification, allocation_weights).
        """
        # Step 1: Detect regime
        print("[Engine] Detecting market regime...", file=sys.stderr)
        regime = self.regime_detector.classify(spy_data, vix_data, breadth_pct)
        print(
            f"[Engine] Regime: {regime.regime.value} "
            f"(confidence: {regime.confidence:.2f})",
            file=sys.stderr,
        )

        # Step 2: Get allocation weights
        allocation = self.regime_allocator.get_allocation(regime.regime)
        print(
            f"[Engine] Allocation — TF: {allocation.trend_following:.0%}, "
            f"MR: {allocation.mean_reversion:.0%}, "
            f"MOM: {allocation.momentum:.0%}, "
            f"SENT: {allocation.sentiment:.0%}, "
            f"EARN: {allocation.earnings_momentum:.0%}, "
            f"Cash: {allocation.cash:.0%}",
            file=sys.stderr,
        )

        # Step 3: Filter universe by data quality
        filtered_data = self.universe.filter_by_data(data)
        print(
            f"[Engine] Universe: {len(filtered_data)} symbols pass filters",
            file=sys.stderr,
        )

        # Step 3b: Pre-fetch sentiment data for strategies that need it
        symbols = list(filtered_data.keys())
        for strategy in self.strategies:
            if hasattr(strategy, "pre_fetch"):
                print(
                    f"[Engine] Pre-fetching data for {strategy.name}...",
                    file=sys.stderr,
                )
                try:
                    strategy.pre_fetch(symbols)
                except Exception as e:
                    print(
                        f"[Engine] WARNING: pre_fetch failed for {strategy.name}: {e}",
                        file=sys.stderr,
                    )

        # Step 4: Generate signals (all strategies, weighted by allocation)
        raw_signals = self.signal_generator.generate(filtered_data, allocation)
        print(f"[Engine] Raw signals: {len(raw_signals)}", file=sys.stderr)

        # Step 5: Filter and deduplicate (regime-adaptive strength floor)
        regime_min_strengths = {
            "trending_bullish": 0.10,
            "trending_bearish": 0.25,
            "ranging": 0.10,
            "high_volatility": 0.30,
        }
        self.signal_filter.min_strength = regime_min_strengths.get(
            regime.regime.value, 0.10
        )
        filtered_signals = self.signal_filter.filter(raw_signals)
        print(f"[Engine] Filtered signals: {len(filtered_signals)}", file=sys.stderr)

        # Step 6: Size positions
        sized_signals = []
        for sig in filtered_signals:
            # Skip signals with no valid entry price
            if not sig.entry_price or sig.entry_price <= 0:
                print(
                    f"[Engine] Skipping {sig.symbol} ({sig.direction.value}): "
                    f"no valid entry price",
                    file=sys.stderr,
                )
                continue

            # For confluence signals, sum the allocation weights of all
            # contributing strategies (capped at 60%) so the merged signal
            # isn't bottlenecked by a single strategy's allocation.
            if sig.features.get("confluence") and sig.features.get("confirming_strategies"):
                alloc_pct = min(0.60, sum(
                    getattr(allocation, s, 0.0)
                    for s in sig.features["confirming_strategies"]
                ))
            else:
                alloc_pct = getattr(allocation, sig.strategy_name, 0.1)

            ctx = PortfolioContext(
                equity=portfolio.equity,
                cash=portfolio.cash,
                buying_power=portfolio.buying_power,
                num_positions=portfolio.num_positions,
                strategy_allocation_pct=alloc_pct,
            )

            hist = filtered_data.get(sig.symbol)
            sizing = self.position_sizer.calculate_size(
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                signal_strength=sig.strength,
                portfolio=ctx,
                historical_data=hist,
                features=sig.features,
            )
            tier = sizing.details.get("conviction_tier", "?")
            print(
                f"[Engine] Sized {sig.symbol} ({sig.direction.value}): "
                f"{sizing.shares} shares @ ${sig.entry_price:.2f} "
                f"(${sizing.dollar_value:.2f}, T{tier}, "
                f"strength={sig.strength:.2f}, alloc={alloc_pct:.2f})",
                file=sys.stderr,
            )
            sized_signals.append((sig, sizing))

        # Step 7: Portfolio optimization (regime-adaptive limits)
        orders = self.portfolio_optimizer.optimize(
            sized_signals, current_positions, portfolio,
            regime=regime.regime.value,
        )
        print(f"[Engine] Order intents: {len(orders)}", file=sys.stderr)

        return orders, regime, allocation
