"""Crypto Strategy Engine Orchestrator.

Mirrors the equity engine (strategy-engine/engine.py) but configured for crypto:
    - Uses BTC as the regime benchmark (instead of SPY)
    - Runs crypto-adapted strategies with tuned parameters
    - BTC dominance rotation replaces sector rotation
    - No gap trading, no earnings, no sentiment (Phase 1)
    - Wider stops, different allocation weights

Same 7-step pipeline:
    1. Detect regime (using BTC)
    2. Get allocation weights (crypto-specific)
    3. Filter universe by data quality
    4. Generate signals
    5. Filter and deduplicate
    6. Size positions
    7. Optimize portfolio (with correlation + drawdown)
"""

import sys
from datetime import datetime

import pandas as pd

from strategies.base import Strategy, StrategyConfig
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from signals.generator import SignalGenerator
from signals.filter import SignalFilter
from portfolio.sizing import PositionSizer, PortfolioContext
from portfolio.drawdown import DrawdownState
from portfolio.optimizer import PortfolioOptimizer, PortfolioOptimizationConfig, OrderIntent

from crypto.universe import CryptoUniverseSelector
from crypto.strategy_configs import (
    CRYPTO_MOMENTUM_CONFIG,
    CRYPTO_TREND_CONFIG,
    CRYPTO_MEAN_REVERSION_CONFIG,
)
from crypto.btc_dominance import BTCDominanceStrategy
from crypto.regime import (
    CryptoRegimeDetector,
    CryptoRegimeAllocator,
    CryptoAllocationWeights,
)
from regime.detector import RegimeClassification


class CryptoStrategyEngine:
    """Orchestrator for the crypto trading system.

    Same interface as StrategyEngine but configured for crypto markets.
    """

    def __init__(
        self,
        strategies: list[Strategy] | None = None,
        universe: CryptoUniverseSelector | None = None,
        regime_detector: CryptoRegimeDetector | None = None,
        regime_allocator: CryptoRegimeAllocator | None = None,
        signal_filter: SignalFilter | None = None,
        position_sizer: PositionSizer | None = None,
        portfolio_optimizer: PortfolioOptimizer | None = None,
    ):
        self.strategies = strategies or [
            TrendFollowingStrategy(CRYPTO_TREND_CONFIG),
            MeanReversionStrategy(CRYPTO_MEAN_REVERSION_CONFIG),
            MomentumStrategy(CRYPTO_MOMENTUM_CONFIG),
            BTCDominanceStrategy(),
        ]
        self.universe = universe or CryptoUniverseSelector()
        self.regime_detector = regime_detector or CryptoRegimeDetector()
        self.regime_allocator = regime_allocator or CryptoRegimeAllocator()
        self.signal_generator = SignalGenerator(self.strategies)
        self.signal_filter = signal_filter or SignalFilter()
        self.position_sizer = position_sizer or PositionSizer()
        self.portfolio_optimizer = portfolio_optimizer or PortfolioOptimizer()

    def run(
        self,
        data: dict[str, pd.DataFrame],
        btc_data: pd.DataFrame,
        portfolio: PortfolioContext,
        current_positions: dict[str, dict],
        drawdown_state: DrawdownState | None = None,
    ) -> tuple[list[OrderIntent], RegimeClassification, CryptoAllocationWeights]:
        """Run the full crypto strategy pipeline.

        Args:
            data: Dict mapping crypto symbol -> OHLCV DataFrame.
            btc_data: BTC/USD OHLCV DataFrame for regime detection.
            portfolio: Current portfolio context.
            current_positions: Dict of current open positions.
            drawdown_state: Optional drawdown state for risk controls.

        Returns:
            Tuple of (order_intents, regime_classification, allocation_weights).
        """
        # Step 1: Detect crypto regime (using BTC as benchmark)
        print("[CryptoEngine] Detecting crypto market regime...", file=sys.stderr)
        regime = self.regime_detector.classify(btc_data)
        print(
            f"[CryptoEngine] Regime: {regime.regime.value} "
            f"(confidence: {regime.confidence:.2f})",
            file=sys.stderr,
        )

        # Step 2: Get crypto allocation weights
        allocation = self.regime_allocator.get_allocation(regime.regime)
        print(
            f"[CryptoEngine] Allocation — TF: {allocation.trend_following:.0%}, "
            f"MR: {allocation.mean_reversion:.0%}, "
            f"MOM: {allocation.momentum:.0%}, "
            f"BTC_DOM: {allocation.btc_dominance:.0%}, "
            f"Cash: {allocation.cash:.0%}",
            file=sys.stderr,
        )

        # Convert to equity-compatible format for the signal generator
        equity_allocation = allocation.to_equity_format()

        # Step 3: Filter universe by data quality
        filtered_data = self.universe.filter_by_data(data)
        print(
            f"[CryptoEngine] Universe: {len(filtered_data)} symbols pass filters",
            file=sys.stderr,
        )

        # Step 4: Generate signals
        raw_signals = self.signal_generator.generate(filtered_data, equity_allocation)
        print(f"[CryptoEngine] Raw signals: {len(raw_signals)}", file=sys.stderr)

        # Step 5: Filter and deduplicate (crypto-adaptive strength floor)
        regime_min_strengths = {
            "trending_bullish": 0.10,
            "trending_bearish": 0.15,  # let short signals through
            "ranging": 0.12,
            "high_volatility": 0.25,  # still selective but not paralyzed
        }
        self.signal_filter.min_strength = regime_min_strengths.get(
            regime.regime.value, 0.15
        )
        filtered_signals = self.signal_filter.filter(raw_signals)
        print(
            f"[CryptoEngine] Filtered signals: {len(filtered_signals)}",
            file=sys.stderr,
        )

        # Step 6: Size positions
        sized_signals = []
        for sig in filtered_signals:
            if not sig.entry_price or sig.entry_price <= 0:
                print(
                    f"[CryptoEngine] Skipping {sig.symbol} ({sig.direction.value}): "
                    f"no valid entry price",
                    file=sys.stderr,
                )
                continue

            # Map btc_dominance -> sector_rotation for allocation lookup
            strategy_alloc_name = sig.strategy_name
            if strategy_alloc_name == "btc_dominance":
                strategy_alloc_name = "sector_rotation"

            if sig.features.get("confluence") and sig.features.get("confirming_strategies"):
                alloc_pct = min(0.50, sum(
                    getattr(equity_allocation, s if s != "btc_dominance" else "sector_rotation", 0.0)
                    for s in sig.features["confirming_strategies"]
                ))
            else:
                alloc_pct = getattr(equity_allocation, strategy_alloc_name, 0.1)

            ctx = PortfolioContext(
                equity=portfolio.equity,
                cash=portfolio.cash,
                buying_power=portfolio.buying_power,
                num_positions=portfolio.num_positions,
                strategy_allocation_pct=alloc_pct,
                vix_level=portfolio.vix_level,
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
                f"[CryptoEngine] Sized {sig.symbol} ({sig.direction.value}): "
                f"{sizing.shares} shares @ ${sig.entry_price:.2f} "
                f"(${sizing.dollar_value:.2f}, T{tier}, "
                f"strength={sig.strength:.2f}, alloc={alloc_pct:.2f})",
                file=sys.stderr,
            )
            sized_signals.append((sig, sizing))

        # Step 7: Portfolio optimization
        orders = self.portfolio_optimizer.optimize(
            sized_signals, current_positions, portfolio,
            regime=regime.regime.value,
            price_data=filtered_data,
            drawdown_state=drawdown_state,
        )
        print(f"[CryptoEngine] Order intents: {len(orders)}", file=sys.stderr)

        return orders, regime, allocation
