"""Enhanced Drawdown Circuit Breaker.

Implements graduated portfolio risk controls based on peak-to-trough drawdown:

    Level 0: Normal operation (drawdown < 5%)
    Level 1: Caution (5-7%) — reduce new position sizes by 25%, raise min strength
    Level 2: Defensive (7-10%) — reduce new position sizes by 50%, raise min strength
    Level 3: Halt (10-12%) — stop all new entries, only allow exits
    Level 4: Unwind (>12%) — start unwinding weakest positions

Auto-recovery: when equity recovers above 5% drawdown from NEW peak,
resume normal operation (level 0).

Uses portfolio snapshots from TradeLogger for persistent drawdown tracking
across restarts.
"""

import os
import sys
from dataclasses import dataclass
from enum import IntEnum


# Drawdown thresholds (configurable via environment)
DRAWDOWN_CAUTION = float(os.environ.get("DRAWDOWN_CAUTION", "0.05"))       # 5%
DRAWDOWN_DEFENSIVE = float(os.environ.get("DRAWDOWN_DEFENSIVE", "0.07"))   # 7%
DRAWDOWN_HALT = float(os.environ.get("DRAWDOWN_HALT", "0.10"))            # 10%
DRAWDOWN_UNWIND = float(os.environ.get("DRAWDOWN_UNWIND", "0.12"))        # 12%
DRAWDOWN_RECOVERY = float(os.environ.get("DRAWDOWN_RECOVERY", "0.05"))    # 5%


class DrawdownLevel(IntEnum):
    """Graduated drawdown response levels."""
    NORMAL = 0       # Business as usual
    CAUTION = 1      # Reduce sizes 25%, tighter filters
    DEFENSIVE = 2    # Reduce sizes 50%, much tighter filters
    HALT = 3         # No new entries
    UNWIND = 4       # Actively close weakest positions


@dataclass
class DrawdownState:
    """Current drawdown state with actionable parameters."""
    level: DrawdownLevel
    drawdown_pct: float              # Current drawdown from peak (negative = in drawdown)
    peak_equity: float               # All-time high equity
    current_equity: float            # Latest equity
    size_multiplier: float           # Multiply new position sizes by this
    min_strength_override: float     # Minimum signal strength to accept entries
    allow_new_entries: bool          # Whether new entries are permitted
    unwind_weakest: bool             # Whether to actively close weak positions
    description: str                 # Human-readable state description


class DrawdownCircuitBreaker:
    """Monitors drawdown and returns graduated risk controls.

    Usage in run_daily.py:
        breaker = DrawdownCircuitBreaker()
        state = breaker.check(equity=100000, peak_equity=110000)
        if not state.allow_new_entries:
            print("HALT: No new entries during drawdown")
        else:
            # Apply size_multiplier to all new entries
            ...

    Usage in price_monitor.py:
        state = breaker.check(equity, peak_equity)
        if state.unwind_weakest:
            # Close weakest 2 positions
            ...
    """

    def __init__(
        self,
        caution_pct: float = DRAWDOWN_CAUTION,
        defensive_pct: float = DRAWDOWN_DEFENSIVE,
        halt_pct: float = DRAWDOWN_HALT,
        unwind_pct: float = DRAWDOWN_UNWIND,
        recovery_pct: float = DRAWDOWN_RECOVERY,
    ):
        self.caution_pct = caution_pct
        self.defensive_pct = defensive_pct
        self.halt_pct = halt_pct
        self.unwind_pct = unwind_pct
        self.recovery_pct = recovery_pct

    def check(
        self,
        equity: float,
        peak_equity: float,
    ) -> DrawdownState:
        """Evaluate current drawdown and return the appropriate response level.

        Args:
            equity: Current portfolio equity.
            peak_equity: All-time peak equity.

        Returns:
            DrawdownState with graduated risk controls.
        """
        if peak_equity <= 0 or equity <= 0:
            return DrawdownState(
                level=DrawdownLevel.NORMAL,
                drawdown_pct=0.0,
                peak_equity=peak_equity,
                current_equity=equity,
                size_multiplier=1.0,
                min_strength_override=0.0,
                allow_new_entries=True,
                unwind_weakest=False,
                description="Normal (no peak data)",
            )

        drawdown_pct = (peak_equity - equity) / peak_equity

        # Level 4: UNWIND — actively close weakest positions
        if drawdown_pct >= self.unwind_pct:
            return DrawdownState(
                level=DrawdownLevel.UNWIND,
                drawdown_pct=drawdown_pct,
                peak_equity=peak_equity,
                current_equity=equity,
                size_multiplier=0.0,
                min_strength_override=1.0,  # effectively blocks all entries
                allow_new_entries=False,
                unwind_weakest=True,
                description=(
                    f"UNWIND: {drawdown_pct:.1%} drawdown from peak "
                    f"(>${peak_equity:,.0f} → ${equity:,.0f}). "
                    f"Closing weakest positions."
                ),
            )

        # Level 3: HALT — stop all new entries
        if drawdown_pct >= self.halt_pct:
            return DrawdownState(
                level=DrawdownLevel.HALT,
                drawdown_pct=drawdown_pct,
                peak_equity=peak_equity,
                current_equity=equity,
                size_multiplier=0.0,
                min_strength_override=1.0,
                allow_new_entries=False,
                unwind_weakest=False,
                description=(
                    f"HALT: {drawdown_pct:.1%} drawdown — no new entries. "
                    f"Only exits allowed."
                ),
            )

        # Level 2: DEFENSIVE — reduce sizes by 50%
        if drawdown_pct >= self.defensive_pct:
            return DrawdownState(
                level=DrawdownLevel.DEFENSIVE,
                drawdown_pct=drawdown_pct,
                peak_equity=peak_equity,
                current_equity=equity,
                size_multiplier=0.50,
                min_strength_override=0.45,
                allow_new_entries=True,
                unwind_weakest=False,
                description=(
                    f"DEFENSIVE: {drawdown_pct:.1%} drawdown — "
                    f"50% size reduction, min strength 0.45."
                ),
            )

        # Level 1: CAUTION — reduce sizes by 25%
        if drawdown_pct >= self.caution_pct:
            return DrawdownState(
                level=DrawdownLevel.CAUTION,
                drawdown_pct=drawdown_pct,
                peak_equity=peak_equity,
                current_equity=equity,
                size_multiplier=0.75,
                min_strength_override=0.35,
                allow_new_entries=True,
                unwind_weakest=False,
                description=(
                    f"CAUTION: {drawdown_pct:.1%} drawdown — "
                    f"25% size reduction, min strength 0.35."
                ),
            )

        # Level 0: NORMAL
        return DrawdownState(
            level=DrawdownLevel.NORMAL,
            drawdown_pct=drawdown_pct,
            peak_equity=peak_equity,
            current_equity=equity,
            size_multiplier=1.0,
            min_strength_override=0.0,
            allow_new_entries=True,
            unwind_weakest=False,
            description=f"Normal operation (drawdown: {drawdown_pct:.1%}).",
        )

    def is_recovered(self, equity: float, peak_equity: float) -> bool:
        """Check if equity has recovered enough to resume normal operation.

        Recovery = drawdown is less than recovery_pct from peak.
        """
        if peak_equity <= 0:
            return True
        drawdown_pct = (peak_equity - equity) / peak_equity
        return drawdown_pct < self.recovery_pct


def get_peak_equity_from_snapshots(trade_logger) -> float:
    """Read peak equity from portfolio snapshots in TradeLogger.

    Args:
        trade_logger: TradeLogger instance with active DB connection.

    Returns:
        Peak equity value, or 0.0 if no snapshots exist.
    """
    try:
        cur = trade_logger.conn.execute(
            "SELECT MAX(peak_equity) as peak FROM portfolio_snapshots"
        )
        row = cur.fetchone()
        if row and row["peak"]:
            return float(row["peak"])
    except Exception:
        pass
    return 0.0


def get_weakest_positions(
    positions: list[dict],
    n: int = 2,
) -> list[dict]:
    """Identify the N weakest positions for unwinding.

    "Weakest" = most negative unrealized P&L percentage.

    Args:
        positions: List of position dicts with 'unrealized_plpc' field.
        n: Number of positions to return.

    Returns:
        List of the N weakest positions, sorted worst-first.
    """
    # Sort by unrealized P&L percentage (ascending = worst first)
    sorted_positions = sorted(
        positions,
        key=lambda p: p.get("unrealized_plpc", 0.0),
    )
    return sorted_positions[:n]
