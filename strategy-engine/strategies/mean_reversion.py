"""Mean Reversion Strategy: RSI + Bollinger Bands.

Entry conditions (long):
    - RSI(14) < 35 (oversold)
    - Price within 0.5% of or below lower Bollinger Band (20, 2.0)
    - ADX < 30 (no extreme trend — ranging to mild trend)

Exit conditions:
    - RSI > 50 (mean reverted)
    - OR price touches upper Bollinger Band
    - OR trailing stop at 1.5x ATR

Exit conditions (close longs):
    - RSI(14) > 65 (overbought)
    - Price touches or closes above upper Bollinger Band
    - ADX < 30
"""

import pandas as pd
import ta

from strategies.base import Signal, SignalDirection, Strategy, StrategyConfig

DEFAULT_PARAMS = {
    "rsi_period": 14,
    "rsi_oversold": 40,
    "rsi_overbought": 65,
    "rsi_exit": 50,
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_proximity_pct": 0.015,  # enter within 1.5% of lower BB
    "adx_period": 14,
    "adx_max": 35,
    "atr_period": 14,
    "atr_stop_multiplier": 1.5,
    "min_avg_volume": 500_000,
    "volume_lookback": 20,
}


class MeanReversionStrategy(Strategy):

    def __init__(self, config: StrategyConfig | None = None):
        if config is None:
            config = StrategyConfig(name="mean_reversion", params=DEFAULT_PARAMS)
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

    def get_parameters(self) -> dict:
        return self.config.params

    def required_history_days(self) -> int:
        return max(self.config.params["bb_period"], self.config.params["rsi_period"]) + 50

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        signals: list[Signal] = []
        p = self.config.params

        for symbol, df in data.items():
            if not self.validate_data(df):
                continue

            sig = self._analyze_symbol(symbol, df, p)
            if sig is not None:
                signals.append(sig)

        return signals

    def _analyze_symbol(
        self, symbol: str, df: pd.DataFrame, p: dict
    ) -> Signal | None:
        df = df.copy()

        # Compute indicators
        df["rsi"] = ta.momentum.rsi(df["close"], window=p["rsi_period"])
        bb = ta.volatility.BollingerBands(
            df["close"], window=p["bb_period"], window_dev=p["bb_std"]
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=p["adx_period"])
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=p["atr_period"]
        )
        df["avg_volume"] = df["volume"].rolling(window=p["volume_lookback"]).mean()

        curr = df.iloc[-1]

        if pd.isna(curr["rsi"]) or pd.isna(curr["bb_lower"]) or pd.isna(curr["adx"]):
            return None

        if curr["avg_volume"] < p["min_avg_volume"]:
            return None

        # Only in ranging markets
        if curr["adx"] > p["adx_max"]:
            return None

        timestamp = df.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        atr = float(curr["atr"])

        # Oversold — long entry (price near or below lower BB)
        bb_threshold = float(curr["bb_lower"]) * (1 + p["bb_proximity_pct"])
        if curr["rsi"] < p["rsi_oversold"] and curr["close"] <= bb_threshold:
            stop_loss = float(curr["close"]) - p["atr_stop_multiplier"] * atr
            strength = min(1.0, (p["rsi_oversold"] - float(curr["rsi"])) / 25)

            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=SignalDirection.LONG,
                strength=strength,
                strategy_name=self.name,
                entry_price=float(curr["close"]),
                stop_loss=stop_loss,
                take_profit=float(curr["bb_middle"]),
                features={
                    "rsi": float(curr["rsi"]),
                    "bb_lower": float(curr["bb_lower"]),
                    "bb_middle": float(curr["bb_middle"]),
                    "bb_upper": float(curr["bb_upper"]),
                    "adx": float(curr["adx"]),
                    "atr": atr,
                    "close": float(curr["close"]),
                },
                rationale=(
                    f"Oversold reversal: RSI={curr['rsi']:.1f} below {p['rsi_oversold']}, "
                    f"price {curr['close']:.2f} at/below lower BB {curr['bb_lower']:.2f}. "
                    f"Ranging market (ADX={curr['adx']:.1f}). "
                    f"Target: BB middle {curr['bb_middle']:.2f}."
                ),
            )

        # Overbought — close signal (we only do long-side mean reversion for simplicity)
        if curr["rsi"] > p["rsi_overbought"] and curr["close"] >= curr["bb_upper"]:
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=SignalDirection.CLOSE,
                strength=0.7,
                strategy_name=self.name,
                features={
                    "rsi": float(curr["rsi"]),
                    "bb_upper": float(curr["bb_upper"]),
                    "close": float(curr["close"]),
                },
                rationale=(
                    f"Overbought: RSI={curr['rsi']:.1f} above {p['rsi_overbought']}, "
                    f"price at upper BB {curr['bb_upper']:.2f}. Close long positions."
                ),
            )

        return None
