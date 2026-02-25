"""Relative Strength Momentum Strategy.

Ranks stocks by their relative strength (rate of change over multiple periods)
and generates signals for the top-ranked stocks (long) and bottom-ranked (short).

Entry conditions (long):
    - Stock is in top N by composite momentum score
    - 6-month return is positive
    - 2-week return > -10% (allow minor pullbacks)
    - Price is above 100 EMA (intermediate uptrend)

Entry conditions (short):
    - Stock is in bottom N by negative momentum score
    - 6-month return is negative
    - 2-week return < +10% (not rebounding sharply)
    - Price is below 100 EMA (intermediate downtrend)

Exit conditions:
    - Stock drops out of top/bottom 2*N ranking
    - OR 100 EMA broken in opposite direction

This captures the well-documented momentum premium in equities,
now in both directions.
"""

import pandas as pd
import ta

from strategies.base import Signal, SignalDirection, Strategy, StrategyConfig

DEFAULT_PARAMS = {
    "top_n": 8,
    "exit_rank_threshold_multiplier": 1.5,
    "roc_periods": [10, 21, 63, 126],  # 2w, 1m, 3m, 6m
    "roc_weights": [0.15, 0.25, 0.30, 0.30],
    "ema_trend_period": 100,
    "min_avg_volume": 200_000,
    "volume_lookback": 20,
    "min_roc_6m": 10.0,
    "min_roc_2w": -10.0,  # allow deeper recent dips if longer-term momentum is strong
}


class MomentumStrategy(Strategy):

    def __init__(self, config: StrategyConfig | None = None):
        if config is None:
            config = StrategyConfig(name="momentum", params=DEFAULT_PARAMS)
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

    def get_parameters(self) -> dict:
        return self.config.params

    def required_history_days(self) -> int:
        return max(self.config.params["roc_periods"]) + 50

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        p = self.config.params
        scored: list[tuple[str, float, pd.DataFrame]] = []

        for symbol, df in data.items():
            if not self.validate_data(df):
                continue

            score = self._compute_momentum_score(df, p)
            if score is not None:
                scored.append((symbol, score, df))

        # Rank by composite score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        signals: list[Signal] = []
        top_n = p["top_n"]

        for rank, (symbol, score, df) in enumerate(scored, start=1):
            curr = df.iloc[-1]
            timestamp = df.index[-1]
            if hasattr(timestamp, "to_pydatetime"):
                timestamp = timestamp.to_pydatetime()

            if rank <= top_n:
                # Entry signal for top-ranked stocks
                atr = ta.volatility.average_true_range(
                    df["high"], df["low"], df["close"], window=14
                ).iloc[-1]
                stop_loss = float(curr["close"]) - 2.0 * float(atr)

                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    strength=min(1.0, score / 100),
                    strategy_name=self.name,
                    entry_price=float(curr["close"]),
                    stop_loss=stop_loss,
                    features={
                        "momentum_score": score,
                        "rank": rank,
                        "close": float(curr["close"]),
                        "atr": float(atr),
                    },
                    rationale=(
                        f"Momentum rank #{rank}/{len(scored)}, "
                        f"composite score={score:.2f}."
                    ),
                ))
            elif rank > top_n * p["exit_rank_threshold_multiplier"]:
                # Exit signal for stocks that dropped far in ranking
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=SignalDirection.CLOSE,
                    strength=0.6,
                    strategy_name=self.name,
                    features={
                        "momentum_score": score,
                        "rank": rank,
                    },
                    rationale=(
                        f"Momentum rank dropped to #{rank}, "
                        f"below exit threshold of {top_n * p['exit_rank_threshold_multiplier']}."
                    ),
                ))

        # === SHORT signals: bottom-ranked negative-momentum stocks ===
        short_scored: list[tuple[str, float, pd.DataFrame]] = []

        for symbol, df in data.items():
            if not self.validate_data(df):
                continue

            score = self._compute_short_momentum_score(df, p)
            if score is not None:
                short_scored.append((symbol, score, df))

        # Rank by most negative score first (ascending)
        short_scored.sort(key=lambda x: x[1])

        for rank, (symbol, score, df) in enumerate(short_scored[:top_n], start=1):
            curr = df.iloc[-1]
            timestamp = df.index[-1]
            if hasattr(timestamp, "to_pydatetime"):
                timestamp = timestamp.to_pydatetime()

            atr = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"], window=14
            ).iloc[-1]
            stop_loss = float(curr["close"]) + 2.0 * float(atr)

            signals.append(Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=SignalDirection.SHORT,
                strength=min(1.0, abs(score) / 100),
                strategy_name=self.name,
                entry_price=float(curr["close"]),
                stop_loss=stop_loss,
                features={
                    "momentum_score": score,
                    "rank": rank,
                    "close": float(curr["close"]),
                    "atr": float(atr),
                },
                rationale=(
                    f"Negative momentum rank #{rank}/{len(short_scored)}, "
                    f"composite score={score:.2f}. SHORT."
                ),
            ))

        return signals

    def _compute_momentum_score(
        self, df: pd.DataFrame, p: dict
    ) -> float | None:
        curr = df.iloc[-1]

        # Volume filter
        avg_vol = df["volume"].rolling(window=p["volume_lookback"]).mean().iloc[-1]
        if pd.isna(avg_vol) or avg_vol < p["min_avg_volume"]:
            return None

        # EMA trend filter
        ema = ta.trend.ema_indicator(df["close"], window=p["ema_trend_period"]).iloc[-1]
        if pd.isna(ema) or curr["close"] < ema:
            return None

        # Compute rate of change for each period
        rocs = []
        for period in p["roc_periods"]:
            if len(df) <= period:
                return None
            roc = (float(curr["close"]) / float(df.iloc[-(period + 1)]["close"]) - 1) * 100
            rocs.append(roc)

        # Check minimum thresholds
        # 6-month ROC is last in the list
        if rocs[-1] < p["min_roc_6m"]:
            return None
        # 2-week ROC is first
        if rocs[0] < p["min_roc_2w"]:
            return None

        # Weighted composite score
        score = sum(r * w for r, w in zip(rocs, p["roc_weights"]))
        return score

    def _compute_short_momentum_score(
        self, df: pd.DataFrame, p: dict
    ) -> float | None:
        """Score stocks for SHORT candidates — inverse of long momentum."""
        curr = df.iloc[-1]

        # Volume filter (same)
        avg_vol = df["volume"].rolling(window=p["volume_lookback"]).mean().iloc[-1]
        if pd.isna(avg_vol) or avg_vol < p["min_avg_volume"]:
            return None

        # EMA trend filter: price BELOW EMA (downtrend)
        ema = ta.trend.ema_indicator(df["close"], window=p["ema_trend_period"]).iloc[-1]
        if pd.isna(ema) or curr["close"] > ema:
            return None  # only short stocks in downtrends

        # Compute rate of change for each period
        rocs = []
        for period in p["roc_periods"]:
            if len(df) <= period:
                return None
            roc = (float(curr["close"]) / float(df.iloc[-(period + 1)]["close"]) - 1) * 100
            rocs.append(roc)

        # 6-month ROC must be negative (stock in decline)
        if rocs[-1] > 0.0:
            return None
        # 2-week ROC must be < +10% (not rebounding sharply)
        if rocs[0] > 10.0:
            return None

        # Weighted composite score (will be negative for declining stocks)
        score = sum(r * w for r, w in zip(rocs, p["roc_weights"]))
        return score
