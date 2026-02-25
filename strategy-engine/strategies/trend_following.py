"""Trend Following Strategy: EMA Pullback with ATR-based stops.

Entry conditions (long):
    - EMAs aligned bullish: EMA20 > EMA50 > EMA200
    - Price pulled back to within 1 ATR of EMA20 (buy the dip in an uptrend)
    - ADX > 20 (trending market)
    - Price bouncing (today's close > yesterday's close)
    - 20-day average volume above minimum threshold

Entry conditions (short):
    - EMAs aligned bearish: EMA20 < EMA50 < EMA200
    - Price rallied to within 2 ATR of EMA20 (sell the rip in a downtrend)
    - ADX > 20 (trending market)
    - Price declining (today's close < yesterday's close)
    - 20-day average volume above minimum threshold

Exit conditions:
    - EMA20 crosses below EMA50 (trend break — close longs)
    - EMA20 crosses above EMA50 (trend break — close shorts)
    - OR trailing stop hit at 2x ATR from entry
"""

import pandas as pd
import ta

from strategies.base import Signal, SignalDirection, Strategy, StrategyConfig


DEFAULT_PARAMS = {
    "fast_ema": 20,
    "slow_ema": 50,
    "trend_ema": 200,
    "adx_period": 14,
    "adx_threshold": 20,
    "atr_period": 14,
    "atr_stop_multiplier": 2.0,
    "pullback_atr_distance": 2.0,  # enter when price within 2.0 ATR of fast EMA
    "min_avg_volume": 200_000,
    "volume_lookback": 20,
}


class TrendFollowingStrategy(Strategy):

    def __init__(self, config: StrategyConfig | None = None):
        if config is None:
            config = StrategyConfig(name="trend_following", params=DEFAULT_PARAMS)
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

    def get_parameters(self) -> dict:
        return self.config.params

    def required_history_days(self) -> int:
        return self.config.params["trend_ema"] + 50

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        signals: list[Signal] = []
        p = self.config.params

        for symbol, df in data.items():
            if not self.validate_data(df):
                continue

            signals.extend(self._analyze_symbol(symbol, df, p))

        return signals

    def _analyze_symbol(
        self, symbol: str, df: pd.DataFrame, p: dict
    ) -> list[Signal]:
        df = df.copy()
        signals: list[Signal] = []

        # Compute indicators
        df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=p["fast_ema"])
        df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=p["slow_ema"])
        df["ema_trend"] = ta.trend.ema_indicator(df["close"], window=p["trend_ema"])
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=p["adx_period"])
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=p["atr_period"]
        )
        df["avg_volume"] = df["volume"].rolling(window=p["volume_lookback"]).mean()

        if len(df) < 2:
            return signals

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # Skip if indicators not yet computed
        if pd.isna(curr["ema_fast"]) or pd.isna(curr["ema_slow"]) or pd.isna(curr["adx"]) or pd.isna(curr["ema_trend"]):
            return signals

        # Volume filter
        if curr["avg_volume"] < p["min_avg_volume"]:
            return signals

        timestamp = df.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        atr = float(curr["atr"])

        # Check for trend break: EMA fast crosses below slow (close longs)
        cross_down = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]
        if cross_down:
            signals.append(Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=SignalDirection.CLOSE,
                strength=0.8,
                strategy_name=self.name,
                features={
                    "ema_fast": float(curr["ema_fast"]),
                    "ema_slow": float(curr["ema_slow"]),
                    "adx": float(curr["adx"]),
                    "close": float(curr["close"]),
                    "close_side": "long",
                },
                rationale=(
                    f"Trend break: EMA{p['fast_ema']} ({curr['ema_fast']:.2f}) "
                    f"crossed below EMA{p['slow_ema']} ({curr['ema_slow']:.2f}). "
                    f"Close long positions."
                ),
            ))

        # Check for bearish trend break: EMA fast crosses above slow (close shorts)
        cross_up = prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]
        if cross_up:
            signals.append(Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=SignalDirection.CLOSE,
                strength=0.8,
                strategy_name=self.name,
                features={
                    "ema_fast": float(curr["ema_fast"]),
                    "ema_slow": float(curr["ema_slow"]),
                    "adx": float(curr["adx"]),
                    "close": float(curr["close"]),
                    "close_side": "short",
                },
                rationale=(
                    f"Bearish trend break: EMA{p['fast_ema']} ({curr['ema_fast']:.2f}) "
                    f"crossed above EMA{p['slow_ema']} ({curr['ema_slow']:.2f}). "
                    f"Close short positions."
                ),
            ))

        # Price pulled back near the fast EMA (within N * ATR)
        distance_to_fast = float(curr["close"]) - float(curr["ema_fast"])
        near_fast_ema = abs(distance_to_fast) < p["pullback_atr_distance"] * atr
        trending = curr["adx"] > p["adx_threshold"]

        # === LONG entry: pullback in an uptrend ===
        emas_aligned_bull = (
            curr["ema_fast"] > curr["ema_slow"] > curr["ema_trend"]
        )
        bouncing = curr["close"] > prev["close"]
        price_above_fast = curr["close"] >= curr["ema_fast"] * 0.99  # allow 1% slack

        if emas_aligned_bull and trending and near_fast_ema and bouncing and price_above_fast:
            stop_loss = float(curr["ema_slow"])  # stop at the slow EMA
            # Use the tighter of the two stops for better risk control
            atr_stop = float(curr["close"]) - p["atr_stop_multiplier"] * atr
            stop_loss = min(stop_loss, atr_stop)

            strength = min(1.0, (float(curr["adx"]) - p["adx_threshold"]) / 30)

            signals.append(Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=SignalDirection.LONG,
                strength=strength,
                strategy_name=self.name,
                entry_price=float(curr["close"]),
                stop_loss=stop_loss,
                features={
                    "ema_fast": float(curr["ema_fast"]),
                    "ema_slow": float(curr["ema_slow"]),
                    "ema_trend": float(curr["ema_trend"]),
                    "adx": float(curr["adx"]),
                    "atr": atr,
                    "distance_to_ema": distance_to_fast,
                    "close": float(curr["close"]),
                },
                rationale=(
                    f"Trend pullback: EMAs aligned (EMA{p['fast_ema']}={curr['ema_fast']:.2f} > "
                    f"EMA{p['slow_ema']}={curr['ema_slow']:.2f} > "
                    f"EMA{p['trend_ema']}={curr['ema_trend']:.2f}). "
                    f"Price near EMA{p['fast_ema']} (dist={distance_to_fast:.2f}). "
                    f"ADX={curr['adx']:.1f}. "
                    f"Stop at {stop_loss:.2f}."
                ),
            ))

        # === SHORT entry: rally rejection in a downtrend ===
        emas_aligned_bear = (
            curr["ema_fast"] < curr["ema_slow"] < curr["ema_trend"]
        )
        declining = curr["close"] < prev["close"]
        price_below_fast = curr["close"] <= curr["ema_fast"] * 1.01  # allow 1% slack

        if emas_aligned_bear and trending and near_fast_ema and declining and price_below_fast:
            stop_loss = float(curr["ema_slow"])  # stop at the slow EMA (above entry)
            # Ensure stop is at least atr_stop_multiplier * ATR above entry
            max_stop = float(curr["close"]) + p["atr_stop_multiplier"] * atr
            stop_loss = min(stop_loss, max_stop)

            strength = min(1.0, (float(curr["adx"]) - p["adx_threshold"]) / 30)

            signals.append(Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=SignalDirection.SHORT,
                strength=strength,
                strategy_name=self.name,
                entry_price=float(curr["close"]),
                stop_loss=stop_loss,
                features={
                    "ema_fast": float(curr["ema_fast"]),
                    "ema_slow": float(curr["ema_slow"]),
                    "ema_trend": float(curr["ema_trend"]),
                    "adx": float(curr["adx"]),
                    "atr": atr,
                    "distance_to_ema": distance_to_fast,
                    "close": float(curr["close"]),
                },
                rationale=(
                    f"Trend pullback SHORT: EMAs aligned bearish "
                    f"(EMA{p['fast_ema']}={curr['ema_fast']:.2f} < "
                    f"EMA{p['slow_ema']}={curr['ema_slow']:.2f} < "
                    f"EMA{p['trend_ema']}={curr['ema_trend']:.2f}). "
                    f"Price near EMA{p['fast_ema']} (dist={distance_to_fast:.2f}). "
                    f"ADX={curr['adx']:.1f}. "
                    f"Stop at {stop_loss:.2f}."
                ),
            ))

        return signals
