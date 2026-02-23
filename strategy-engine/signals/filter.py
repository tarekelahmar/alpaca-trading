"""Signal Filter — ranks, deduplicates, and filters signals.

Handles conflicts when multiple strategies produce signals for the
same symbol, and applies final quality checks.

Cross-strategy confluence boosting:
    When 2+ strategies agree on the same symbol and direction, the signal
    gets a significant strength boost. This is one of the strongest
    edges in systematic trading — independent models agreeing is a
    high-conviction indicator.

    2 strategies agree: +0.15 boost
    3 strategies agree: +0.35 boost
    4+ strategies agree: +0.50 boost (max conviction)

    The best entry_price and tightest stop_loss from all agreeing
    signals are preserved, giving the best execution parameters.
"""

import sys

from strategies.base import Signal, SignalDirection


class SignalFilter:

    def __init__(
        self,
        min_strength: float = 0.1,
        max_signals: int = 50,
    ):
        self.min_strength = min_strength
        self.max_signals = max_signals

    def filter(self, signals: list[Signal]) -> list[Signal]:
        """Filter and deduplicate signals.

        1. Remove signals below minimum strength
        2. Resolve conflicts (same symbol, different directions)
        3. Merge reinforcing signals with confluence boosting
        4. Sort by strength descending
        5. Cap at max_signals

        Returns:
            Filtered and deduplicated signal list.
        """
        # Step 1: Min strength filter
        signals = [s for s in signals if s.strength >= self.min_strength]

        # Step 2 & 3: Group by symbol
        by_symbol: dict[str, list[Signal]] = {}
        for sig in signals:
            by_symbol.setdefault(sig.symbol, []).append(sig)

        resolved: list[Signal] = []
        for symbol, sym_signals in by_symbol.items():
            merged = self._resolve_symbol(sym_signals)
            if merged is not None:
                resolved.append(merged)

        # Step 4: Sort by strength
        resolved.sort(key=lambda s: s.strength, reverse=True)

        # Step 5: Cap
        return resolved[: self.max_signals]

    def _resolve_symbol(self, signals: list[Signal]) -> Signal | None:
        """Resolve multiple signals for the same symbol.

        Rules:
        - CLOSE signals always win (safety first)
        - If all directions agree, merge with confluence boosting
        - If directions conflict, the stronger side wins
          but strength is reduced by the opposition
        """
        if not signals:
            return None

        # Check for close signals
        close_signals = [s for s in signals if s.direction == SignalDirection.CLOSE]
        if close_signals:
            # Pick strongest close signal
            best = max(close_signals, key=lambda s: s.strength)
            best.features["merged_from"] = len(close_signals)
            return best

        # All entry signals — check for conflicts
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        short_signals = [s for s in signals if s.direction == SignalDirection.SHORT]

        if long_signals and short_signals:
            # Conflict: net the strengths
            long_strength = sum(s.strength for s in long_signals)
            short_strength = sum(s.strength for s in short_signals)

            if long_strength > short_strength:
                best = max(long_signals, key=lambda s: s.strength)
                net = long_strength - short_strength
                best.strength = net / len(long_signals)
                best.features["conflict_resolved"] = True
                best.features["opposition_strength"] = short_strength
                best.rationale += (
                    f" [Conflict: {len(short_signals)} opposing short signal(s) "
                    f"with total strength {short_strength:.2f}]"
                )
                return best
            elif short_strength > long_strength:
                best = max(short_signals, key=lambda s: s.strength)
                net = short_strength - long_strength
                best.strength = net / len(short_signals)
                best.features["conflict_resolved"] = True
                best.features["opposition_strength"] = long_strength
                return best
            else:
                return None

        # No conflict — merge reinforcing signals with confluence boosting
        all_sigs = long_signals or short_signals
        if not all_sigs:
            return None

        best = max(all_sigs, key=lambda s: s.strength)

        if len(all_sigs) > 1:
            # Confluence boosting — aggressive scaling for multi-strategy agreement
            n = len(all_sigs)
            if n >= 4:
                boost = 0.50  # 4+ strategies = max conviction
            elif n == 3:
                boost = 0.35  # 3 strategies = high conviction
            else:
                boost = 0.15  # 2 strategies = moderate confirmation

            # Average the strengths from all agreeing strategies, then add boost
            avg_strength = sum(s.strength for s in all_sigs) / n
            best.strength = min(1.0, avg_strength + boost)

            # Inherit the best entry price and tightest stop from all signals
            for sig in all_sigs:
                if sig is best:
                    continue
                # Use the entry price if best doesn't have one
                if not best.entry_price and sig.entry_price:
                    best.entry_price = sig.entry_price
                # Use the tightest (highest) stop loss for safety
                if sig.stop_loss and best.stop_loss:
                    best.stop_loss = max(best.stop_loss, sig.stop_loss)
                elif sig.stop_loss:
                    best.stop_loss = sig.stop_loss

            strategies = [s.strategy_name for s in all_sigs]
            best.features["confluence"] = True
            best.features["confluence_count"] = n
            best.features["confluence_boost"] = boost
            best.features["confirming_strategies"] = strategies
            best.features["pre_boost_strength"] = avg_strength
            best.rationale += (
                f" [CONFLUENCE x{n}: {', '.join(strategies)} — "
                f"boost +{boost:.0%}, final strength {best.strength:.2f}]"
            )
            print(
                f"[Filter] CONFLUENCE {best.symbol}: {n} strategies agree "
                f"({', '.join(strategies)}), strength {avg_strength:.2f} → "
                f"{best.strength:.2f}",
                file=sys.stderr,
            )

        return best
