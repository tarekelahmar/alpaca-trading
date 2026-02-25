"""Gap Trading Strategy: Gap & Go + Gap Fill.

Two sub-strategies driven by overnight price gaps:

Gap & Go (LONG):
    - Stock gaps UP ≥ 3% on above-average volume
    - Prior trend is bullish (EMA20 > EMA50)
    - RSI not already overbought (< 75)
    - Ride the momentum continuation for 1-3 days
    - Tight trailing stop (1.5x ATR) — gaps that fail reverse hard

Gap Fill (LONG):
    - Stock gaps DOWN 2-4% with no major catalyst (mechanical selling)
    - Prior trend is bullish or neutral (price above EMA50)
    - RSI drops but not extreme panic (RSI > 25)
    - High average volume (liquid stock = reliable gap fills)
    - Mean reversion to prior close within 1-5 days
    - Stop below the gap day's low

Gap & Go (SHORT):
    - Stock gaps DOWN ≥ 3% on above-average volume
    - Prior trend is bearish (EMA20 < EMA50)
    - RSI not already oversold (> 25)
    - Momentum continuation to the downside

Gap Fill (SHORT):
    - Stock gaps UP 2-4% without fundamental catalyst
    - Prior trend is bearish or neutral (price below EMA50)
    - RSI elevated but not extreme (< 75)
    - Fade the gap back to prior close

Edge: Gaps on liquid stocks fill ~70% of the time for small gaps (2-4%),
while large gaps (>3%) on strong volume in trending stocks tend to continue.
This strategy captures both patterns with appropriate exit rules.

Holding period: 1-5 days (short-term mean reversion / momentum).
"""

import pandas as pd
import ta

from strategies.base import Signal, SignalDirection, Strategy, StrategyConfig

DEFAULT_PARAMS = {
    # Gap detection
    "gap_go_min_pct": 3.0,       # minimum gap % for Gap & Go
    "gap_fill_min_pct": 2.0,     # minimum gap % for Gap Fill
    "gap_fill_max_pct": 4.5,     # maximum gap % for Gap Fill (avoid huge gaps)
    # Volume
    "min_avg_volume": 500_000,   # higher volume requirement for gap trades
    "volume_lookback": 20,
    "volume_surge_mult": 1.3,    # gap day volume must be > 1.3x average
    # Trend filters
    "fast_ema": 20,
    "slow_ema": 50,
    # RSI filters
    "rsi_period": 14,
    "rsi_overbought_limit": 75,  # don't buy Gap & Go if already overbought
    "rsi_oversold_limit": 25,    # don't short Gap & Go if already oversold
    # ATR for stops
    "atr_period": 14,
    "gap_go_atr_stop_mult": 1.5,     # tight stop for momentum trades
    "gap_fill_atr_stop_mult": 1.0,   # stop below gap day low + buffer
}


class GapTradingStrategy(Strategy):

    def __init__(self, config: StrategyConfig | None = None):
        if config is None:
            config = StrategyConfig(name="gap_trading", params=DEFAULT_PARAMS)
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

    def get_parameters(self) -> dict:
        return self.config.params

    def required_history_days(self) -> int:
        return max(self.config.params["slow_ema"], 50) + 10

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

        # Need at least 2 days to compute gap
        if len(df) < p["slow_ema"] + 10:
            return signals

        # Compute indicators
        df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=p["fast_ema"])
        df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=p["slow_ema"])
        df["rsi"] = ta.momentum.rsi(df["close"], window=p["rsi_period"])
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=p["atr_period"]
        )
        df["avg_volume"] = df["volume"].rolling(window=p["volume_lookback"]).mean()

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # Skip if indicators not ready
        if pd.isna(curr["ema_fast"]) or pd.isna(curr["rsi"]) or pd.isna(curr["atr"]):
            return signals
        if pd.isna(curr["avg_volume"]) or curr["avg_volume"] < p["min_avg_volume"]:
            return signals

        timestamp = df.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        atr = float(curr["atr"])
        prev_close = float(prev["close"])
        curr_open = float(curr["open"])
        curr_close = float(curr["close"])
        curr_low = float(curr["low"])
        curr_high = float(curr["high"])

        # Gap percentage: (today's open - yesterday's close) / yesterday's close
        gap_pct = ((curr_open - prev_close) / prev_close) * 100

        # Volume surge check
        volume_ratio = float(curr["volume"]) / float(curr["avg_volume"]) if curr["avg_volume"] > 0 else 0

        # Trend context
        bullish_trend = curr["ema_fast"] > curr["ema_slow"]
        bearish_trend = curr["ema_fast"] < curr["ema_slow"]

        # ==========================================
        # GAP UP scenarios
        # ==========================================
        if gap_pct >= p["gap_go_min_pct"]:
            # --- Gap & Go LONG ---
            # Strong gap up + bullish trend + volume surge = ride momentum
            if (bullish_trend
                    and volume_ratio >= p["volume_surge_mult"]
                    and curr["rsi"] < p["rsi_overbought_limit"]
                    and curr_close >= curr_open):  # gap held (closed above open)

                stop_loss = curr_close - p["gap_go_atr_stop_mult"] * atr
                # Strength based on gap size and volume surge
                strength = min(1.0, (gap_pct / 10.0) * 0.5 + (volume_ratio / 3.0) * 0.5)

                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=curr_close,
                    stop_loss=stop_loss,
                    features={
                        "gap_type": "gap_and_go",
                        "gap_pct": gap_pct,
                        "volume_ratio": volume_ratio,
                        "rsi": float(curr["rsi"]),
                        "atr": atr,
                        "close": curr_close,
                    },
                    rationale=(
                        f"Gap & Go LONG: {symbol} gapped up {gap_pct:.1f}% on "
                        f"{volume_ratio:.1f}x avg volume. Bullish trend, "
                        f"RSI={curr['rsi']:.0f}. Stop={stop_loss:.2f}."
                    ),
                ))

        elif gap_pct >= p["gap_fill_min_pct"] and gap_pct <= p["gap_fill_max_pct"]:
            # --- Gap Fill SHORT (fade the gap up) ---
            # Moderate gap up + bearish/neutral trend = mean reversion back
            if (not bullish_trend
                    and curr["rsi"] < p["rsi_overbought_limit"]
                    and curr_close < curr_open):  # gap fading (close below open)

                stop_loss = curr_high + p["gap_fill_atr_stop_mult"] * atr
                # Target: fill back to prior close
                strength = min(1.0, gap_pct / 6.0)

                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=curr_close,
                    stop_loss=stop_loss,
                    take_profit=prev_close,  # target: fill the gap
                    features={
                        "gap_type": "gap_fill_short",
                        "gap_pct": gap_pct,
                        "volume_ratio": volume_ratio,
                        "rsi": float(curr["rsi"]),
                        "atr": atr,
                        "close": curr_close,
                        "gap_fill_target": prev_close,
                    },
                    rationale=(
                        f"Gap Fill SHORT: {symbol} gapped up {gap_pct:.1f}% but "
                        f"not in uptrend. Close < open = fading. "
                        f"Target: fill to {prev_close:.2f}. Stop={stop_loss:.2f}."
                    ),
                ))

        # ==========================================
        # GAP DOWN scenarios
        # ==========================================
        if gap_pct <= -p["gap_go_min_pct"]:
            # --- Gap & Go SHORT ---
            # Strong gap down + bearish trend + volume surge = ride downward momentum
            if (bearish_trend
                    and volume_ratio >= p["volume_surge_mult"]
                    and curr["rsi"] > p["rsi_oversold_limit"]
                    and curr_close <= curr_open):  # gap held down (closed below open)

                stop_loss = curr_close + p["gap_go_atr_stop_mult"] * atr
                strength = min(1.0, (abs(gap_pct) / 10.0) * 0.5 + (volume_ratio / 3.0) * 0.5)

                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=curr_close,
                    stop_loss=stop_loss,
                    features={
                        "gap_type": "gap_and_go_short",
                        "gap_pct": gap_pct,
                        "volume_ratio": volume_ratio,
                        "rsi": float(curr["rsi"]),
                        "atr": atr,
                        "close": curr_close,
                    },
                    rationale=(
                        f"Gap & Go SHORT: {symbol} gapped down {abs(gap_pct):.1f}% on "
                        f"{volume_ratio:.1f}x avg volume. Bearish trend, "
                        f"RSI={curr['rsi']:.0f}. Stop={stop_loss:.2f}."
                    ),
                ))

        elif gap_pct <= -p["gap_fill_min_pct"] and gap_pct >= -p["gap_fill_max_pct"]:
            # --- Gap Fill LONG (buy the dip, fade the gap down) ---
            # Moderate gap down + bullish/neutral trend = mean reversion to prior close
            if (not bearish_trend
                    and curr["rsi"] > p["rsi_oversold_limit"]
                    and curr_close > curr_open):  # recovering (close above open)

                stop_loss = curr_low - p["gap_fill_atr_stop_mult"] * atr
                # Target: fill back to prior close
                strength = min(1.0, abs(gap_pct) / 6.0)

                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=curr_close,
                    stop_loss=stop_loss,
                    take_profit=prev_close,  # target: fill the gap
                    features={
                        "gap_type": "gap_fill_long",
                        "gap_pct": gap_pct,
                        "volume_ratio": volume_ratio,
                        "rsi": float(curr["rsi"]),
                        "atr": atr,
                        "close": curr_close,
                        "gap_fill_target": prev_close,
                    },
                    rationale=(
                        f"Gap Fill LONG: {symbol} gapped down {abs(gap_pct):.1f}% but "
                        f"not in downtrend. Close > open = recovering. "
                        f"Target: fill to {prev_close:.2f}. Stop={stop_loss:.2f}."
                    ),
                ))

        return signals
