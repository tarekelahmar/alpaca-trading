"""Portfolio-level Optimization.

Takes a set of sized signals and applies portfolio-level constraints:
    - Maximum number of open positions (regime-adaptive)
    - Total exposure cap (regime-adaptive)
    - Minimum signal strength (regime-adaptive)
"""

from dataclasses import dataclass, field

from strategies.base import Signal, SignalDirection
from portfolio.sizing import PositionSizeResult, PortfolioContext


@dataclass
class OrderIntent:
    """A fully sized order ready for execution."""
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    order_type: str  # "market", "limit"
    limit_price: float | None
    stop_loss: float | None
    take_profit: float | None
    signal: Signal
    sizing: PositionSizeResult
    rationale: str


REGIME_OVERRIDES = {
    "trending_bullish": {
        "max_open_positions": 15,
        "max_new_positions_per_day": 5,
        "max_total_exposure_pct": 0.95,
        "min_signal_strength": 0.25,
    },
    "trending_bearish": {
        "max_open_positions": 10,
        "max_new_positions_per_day": 3,
        "max_total_exposure_pct": 0.60,
        "min_signal_strength": 0.40,
    },
    "ranging": {
        "max_open_positions": 20,
        "max_new_positions_per_day": 5,
        "max_total_exposure_pct": 0.85,
        "min_signal_strength": 0.20,
    },
    "high_volatility": {
        "max_open_positions": 10,
        "max_new_positions_per_day": 2,
        "max_total_exposure_pct": 0.50,
        "min_signal_strength": 0.50,
    },
}


@dataclass
class PortfolioOptimizationConfig:
    max_open_positions: int = 20
    max_new_positions_per_day: int = 5
    max_total_exposure_pct: float = 0.90
    min_signal_strength: float = 0.2


class PortfolioOptimizer:

    def __init__(self, config: PortfolioOptimizationConfig | None = None):
        self.config = config or PortfolioOptimizationConfig()

    def optimize(
        self,
        signals_with_sizing: list[tuple[Signal, PositionSizeResult]],
        current_positions: dict[str, dict],
        portfolio: PortfolioContext,
        regime: str | None = None,
    ) -> list[OrderIntent]:
        """Convert sized signals into order intents, respecting portfolio constraints.

        Args:
            signals_with_sizing: List of (Signal, PositionSizeResult) tuples.
            current_positions: Dict mapping symbol -> position info dict
                               with keys: qty, market_value, side.
            portfolio: Current portfolio context.
            regime: Optional regime string for adaptive limits.

        Returns:
            List of OrderIntent objects, ordered by priority.
        """
        c = self.config

        # Resolve regime-adaptive limits
        max_open = c.max_open_positions
        max_new = c.max_new_positions_per_day
        max_exposure = c.max_total_exposure_pct
        min_strength = c.min_signal_strength

        if regime and regime in REGIME_OVERRIDES:
            ov = REGIME_OVERRIDES[regime]
            max_open = ov.get("max_open_positions", max_open)
            max_new = ov.get("max_new_positions_per_day", max_new)
            max_exposure = ov.get("max_total_exposure_pct", max_exposure)
            min_strength = ov.get("min_signal_strength", min_strength)

        orders: list[OrderIntent] = []

        # Separate exit signals from entry signals
        exits: list[tuple[Signal, PositionSizeResult]] = []
        entries: list[tuple[Signal, PositionSizeResult]] = []

        for sig, sizing in signals_with_sizing:
            if sig.direction == SignalDirection.CLOSE:
                exits.append((sig, sizing))
            else:
                entries.append((sig, sizing))

        # Process exits first (always)
        for sig, sizing in exits:
            if sig.symbol in current_positions:
                pos = current_positions[sig.symbol]
                orders.append(OrderIntent(
                    symbol=sig.symbol,
                    side="sell",
                    qty=abs(int(pos.get("qty", 0))),
                    order_type="market",
                    limit_price=None,
                    stop_loss=None,
                    take_profit=None,
                    signal=sig,
                    sizing=sizing,
                    rationale=sig.rationale,
                ))

        # Count how many positions we'll have after exits
        exiting_symbols = {o.symbol for o in orders}
        remaining_positions = {
            s: p for s, p in current_positions.items()
            if s not in exiting_symbols
        }
        open_count = len(remaining_positions)

        # Process entries — sort by signal strength descending
        entries.sort(key=lambda x: x[0].strength, reverse=True)

        # Filter weak signals (regime-adaptive threshold)
        entries = [
            (s, sz) for s, sz in entries
            if s.strength >= min_strength
        ]

        # Skip symbols we already hold
        entries = [
            (s, sz) for s, sz in entries
            if s.symbol not in remaining_positions
        ]

        new_entries = 0
        total_new_value = 0.0
        max_new_value = portfolio.equity * max_exposure

        for sig, sizing in entries:
            if open_count >= max_open:
                break
            if new_entries >= max_new:
                break
            if sizing.shares <= 0:
                continue

            # Check total exposure
            current_exposure = sum(
                abs(float(p.get("market_value", 0)))
                for p in remaining_positions.values()
            )
            if current_exposure + total_new_value + sizing.dollar_value > max_new_value:
                continue

            side = "buy" if sig.direction == SignalDirection.LONG else "sell"
            orders.append(OrderIntent(
                symbol=sig.symbol,
                side=side,
                qty=sizing.shares,
                order_type="market",
                limit_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                signal=sig,
                sizing=sizing,
                rationale=sig.rationale,
            ))
            open_count += 1
            new_entries += 1
            total_new_value += sizing.dollar_value

        return orders
