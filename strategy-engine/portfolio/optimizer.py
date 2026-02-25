"""Portfolio-level Optimization.

Takes a set of sized signals and applies portfolio-level constraints:
    - Maximum number of open positions (regime-adaptive)
    - Total exposure cap (regime-adaptive)
    - Minimum signal strength (regime-adaptive)
    - Short position limits (count and concentration)
    - Sector exposure limits (max % of portfolio per GICS sector)
    - Correlation-aware position limits (reduce/skip correlated entries)
    - Drawdown circuit breaker (graduated size reduction / halt)
"""

import os
import sys
from dataclasses import dataclass, field

import pandas as pd

from strategies.base import Signal, SignalDirection
from strategies.sector_rotation import get_stock_sector
from portfolio.sizing import PositionSizeResult, PortfolioContext
from portfolio.correlation import check_correlation_limit
from portfolio.drawdown import DrawdownCircuitBreaker, DrawdownState, DrawdownLevel


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


# Short position limits from environment
MAX_SHORT_POSITIONS = int(os.environ.get("MAX_SHORT_POSITIONS", "5"))
MAX_SHORT_CONCENTRATION = float(os.environ.get("MAX_SHORT_CONCENTRATION", "0.15"))

# Sector exposure limit — max fraction of equity in any single sector
MAX_SECTOR_EXPOSURE = float(os.environ.get("MAX_SECTOR_EXPOSURE", "0.30"))


def _normalize_side(side) -> str:
    """Normalize Alpaca position side to string 'long' or 'short'."""
    if hasattr(side, "value"):
        return str(side.value)
    return str(side)


class PortfolioOptimizer:

    def __init__(self, config: PortfolioOptimizationConfig | None = None):
        self.config = config or PortfolioOptimizationConfig()

    def optimize(
        self,
        signals_with_sizing: list[tuple[Signal, PositionSizeResult]],
        current_positions: dict[str, dict],
        portfolio: PortfolioContext,
        regime: str | None = None,
        price_data: dict[str, pd.DataFrame] | None = None,
        drawdown_state: DrawdownState | None = None,
    ) -> list[OrderIntent]:
        """Convert sized signals into order intents, respecting portfolio constraints.

        Args:
            signals_with_sizing: List of (Signal, PositionSizeResult) tuples.
            current_positions: Dict mapping symbol -> position info dict
                               with keys: qty, market_value, side.
            portfolio: Current portfolio context.
            regime: Optional regime string for adaptive limits.
            price_data: Optional dict of symbol -> OHLCV DataFrames for
                        correlation-aware position limits.
            drawdown_state: Optional DrawdownState for circuit breaker controls.

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

        # Apply drawdown circuit breaker overrides
        dd_size_mult = 1.0
        if drawdown_state:
            if not drawdown_state.allow_new_entries:
                print(
                    f"[Optimizer] DRAWDOWN {drawdown_state.level.name}: "
                    f"{drawdown_state.description}",
                    file=sys.stderr,
                )
            dd_size_mult = drawdown_state.size_multiplier
            # Override min strength if drawdown requires tighter filter
            if drawdown_state.min_strength_override > min_strength:
                min_strength = drawdown_state.min_strength_override

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
                # Determine exit side based on position direction
                pos_side = _normalize_side(pos.get("side", "long"))
                exit_side = "buy" if pos_side == "short" else "sell"

                orders.append(OrderIntent(
                    symbol=sig.symbol,
                    side=exit_side,
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

        # Count existing short positions for limit enforcement
        short_count = sum(
            1 for p in remaining_positions.values()
            if _normalize_side(p.get("side", "long")) == "short"
        )
        short_exposure = sum(
            abs(float(p.get("market_value", 0)))
            for p in remaining_positions.values()
            if _normalize_side(p.get("side", "long")) == "short"
        )

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
        new_short_entries = 0
        total_new_value = 0.0
        max_new_value = portfolio.equity * max_exposure

        # Build sector exposure map from existing positions
        sector_exposure: dict[str, float] = {}
        for sym, pos in remaining_positions.items():
            sector = get_stock_sector(sym)
            if sector:
                val = abs(float(pos.get("market_value", 0)))
                sector_exposure[sector] = sector_exposure.get(sector, 0.0) + val

        # Drawdown halt: skip all new entries if circuit breaker is active
        if drawdown_state and not drawdown_state.allow_new_entries:
            # Still allow exit orders (already processed above)
            return orders

        # Build list of held symbols for correlation checking
        held_symbols = list(remaining_positions.keys())

        for sig, sizing in entries:
            if open_count >= max_open:
                break
            if new_entries >= max_new:
                break
            if sizing.shares <= 0:
                continue

            # Short-specific limits
            if sig.direction == SignalDirection.SHORT:
                if short_count + new_short_entries >= MAX_SHORT_POSITIONS:
                    continue
                new_short_value = short_exposure + sizing.dollar_value
                if portfolio.equity > 0 and new_short_value / portfolio.equity > MAX_SHORT_CONCENTRATION:
                    continue

            # Sector exposure limit
            sector = get_stock_sector(sig.symbol)
            if sector and portfolio.equity > 0:
                current_sector_val = sector_exposure.get(sector, 0.0)
                new_sector_val = current_sector_val + sizing.dollar_value
                if new_sector_val / portfolio.equity > MAX_SECTOR_EXPOSURE:
                    print(
                        f"[Optimizer] Skipping {sig.symbol}: would push "
                        f"{sector} exposure to {new_sector_val/portfolio.equity:.0%} "
                        f"(limit: {MAX_SECTOR_EXPOSURE:.0%})",
                        file=sys.stderr,
                    )
                    continue

            # Correlation-aware position limits
            corr_size_mult = 1.0
            if price_data and held_symbols:
                corr_action, corr_mult, corr_pairs = check_correlation_limit(
                    new_symbol=sig.symbol,
                    existing_symbols=held_symbols,
                    price_data=price_data,
                )
                if corr_action == "skip":
                    pairs_str = ", ".join(
                        f"{s}({c:+.2f})" for s, c in corr_pairs
                    )
                    print(
                        f"[Optimizer] Skipping {sig.symbol}: too correlated "
                        f"with {len(corr_pairs)} positions: {pairs_str}",
                        file=sys.stderr,
                    )
                    continue
                elif corr_action == "reduce":
                    corr_size_mult = corr_mult
                    pairs_str = ", ".join(
                        f"{s}({c:+.2f})" for s, c in corr_pairs
                    )
                    print(
                        f"[Optimizer] Reducing {sig.symbol} size by "
                        f"{(1-corr_mult):.0%}: correlated with {pairs_str}",
                        file=sys.stderr,
                    )

            # Apply drawdown and correlation size adjustments
            effective_size_mult = dd_size_mult * corr_size_mult
            adjusted_shares = sizing.shares
            adjusted_dollar_value = sizing.dollar_value

            if effective_size_mult < 1.0:
                adjusted_shares = max(1, int(sizing.shares * effective_size_mult))
                adjusted_dollar_value = adjusted_shares * (
                    sizing.dollar_value / sizing.shares if sizing.shares > 0 else 0
                )

            # Check total exposure
            current_exposure = sum(
                abs(float(p.get("market_value", 0)))
                for p in remaining_positions.values()
            )
            if current_exposure + total_new_value + adjusted_dollar_value > max_new_value:
                continue

            # Create adjusted sizing if needed
            if effective_size_mult < 1.0 and adjusted_shares != sizing.shares:
                from copy import copy
                adjusted_sizing = copy(sizing)
                adjusted_sizing.shares = adjusted_shares
                adjusted_sizing.dollar_value = adjusted_dollar_value
                used_sizing = adjusted_sizing
            else:
                used_sizing = sizing

            side = "buy" if sig.direction == SignalDirection.LONG else "sell"
            orders.append(OrderIntent(
                symbol=sig.symbol,
                side=side,
                qty=used_sizing.shares,
                order_type="market",
                limit_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                signal=sig,
                sizing=used_sizing,
                rationale=sig.rationale,
            ))
            open_count += 1
            new_entries += 1
            total_new_value += used_sizing.dollar_value

            # Track this symbol as held for correlation checks on subsequent entries
            held_symbols.append(sig.symbol)

            # Update sector exposure tracking
            if sector:
                sector_exposure[sector] = sector_exposure.get(sector, 0.0) + used_sizing.dollar_value

            if sig.direction == SignalDirection.SHORT:
                new_short_entries += 1
                short_exposure += used_sizing.dollar_value

        return orders
