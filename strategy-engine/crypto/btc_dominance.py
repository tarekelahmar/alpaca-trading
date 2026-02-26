"""BTC Dominance Rotation Strategy.

Replaces sector rotation for crypto markets. The key insight:

When BTC dominance is RISING → capital flows from alts to BTC (risk-off)
    → Go long BTC, reduce/short alts
When BTC dominance is FALLING → capital flows from BTC to alts (risk-on)
    → Go long alts, reduce BTC

BTC dominance is approximated as BTC market cap / total crypto market cap.
Since we can't get direct dominance data from Alpaca, we use the ratio of
BTC returns vs average alt returns as a proxy. When BTC outperforms alts,
dominance is rising.

This strategy generates signals by:
1. Computing 14-day rolling relative performance of BTC vs alt basket
2. When BTC is outperforming: favor BTC, signal CLOSE on weakest alts
3. When alts are outperforming: favor top-performing alts, reduce BTC weight
"""

import sys
from datetime import datetime

import pandas as pd
import ta

from strategies.base import Signal, SignalDirection, Strategy, StrategyConfig

DEFAULT_PARAMS = {
    "dominance_lookback": 14,      # days to compute relative performance
    "trend_lookback": 30,          # days for trend confirmation
    "dominance_threshold": 0.02,   # 2% relative outperformance to trigger rotation
    "top_alts": 3,                 # number of top alts to go long
    "min_history_days": 60,        # minimum data needed
    "atr_period": 14,
    "atr_stop_multiplier": 3.0,    # wide stops for crypto
    "btc_symbol": "BTC/USD",
    "eth_symbol": "ETH/USD",
}


class BTCDominanceStrategy(Strategy):
    """Rotates between BTC and altcoins based on relative performance.

    This is the crypto equivalent of sector rotation. Instead of rotating
    between sectors (XLK, XLF, etc.), we rotate between BTC and altcoins
    based on the flow of capital within the crypto market.
    """

    def __init__(self, config: StrategyConfig | None = None):
        if config is None:
            config = StrategyConfig(name="btc_dominance", params=DEFAULT_PARAMS)
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

    def get_parameters(self) -> dict:
        return self.config.params

    def required_history_days(self) -> int:
        return self.config.params["min_history_days"]

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        """Generate rotation signals based on BTC dominance proxy.

        Args:
            data: Dict mapping crypto symbol -> OHLCV DataFrame.

        Returns:
            List of signals favoring BTC or alts based on dominance trend.
        """
        p = self.config.params
        btc_sym = p["btc_symbol"]

        if btc_sym not in data:
            print(
                f"[BTCDominance] No BTC data found ({btc_sym}), skipping",
                file=sys.stderr,
            )
            return []

        btc_df = data[btc_sym]
        if len(btc_df) < p["min_history_days"]:
            return []

        # Compute BTC returns
        btc_returns = btc_df["close"].pct_change(p["dominance_lookback"]).iloc[-1]
        if pd.isna(btc_returns):
            return []

        # Compute alt returns (exclude BTC)
        alt_returns: dict[str, float] = {}
        alt_data: dict[str, pd.DataFrame] = {}
        for sym, df in data.items():
            if sym == btc_sym or len(df) < p["min_history_days"]:
                continue
            ret = df["close"].pct_change(p["dominance_lookback"]).iloc[-1]
            if not pd.isna(ret):
                alt_returns[sym] = float(ret)
                alt_data[sym] = df

        if not alt_returns:
            return []

        avg_alt_return = sum(alt_returns.values()) / len(alt_returns)
        dominance_delta = float(btc_returns) - avg_alt_return

        # Trend confirmation: is the dominance shift persistent?
        btc_trend = btc_df["close"].pct_change(p["trend_lookback"]).iloc[-1]
        if pd.isna(btc_trend):
            btc_trend = 0.0
        else:
            btc_trend = float(btc_trend)

        signals: list[Signal] = []
        timestamp = btc_df.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        if dominance_delta > p["dominance_threshold"]:
            # BTC dominance RISING → favor BTC, reduce alts
            signals.extend(
                self._btc_dominance_rising(
                    btc_df, alt_returns, alt_data, dominance_delta,
                    btc_trend, timestamp, p
                )
            )
        elif dominance_delta < -p["dominance_threshold"]:
            # BTC dominance FALLING → favor alts, reduce BTC
            signals.extend(
                self._btc_dominance_falling(
                    btc_df, alt_returns, alt_data, dominance_delta,
                    btc_trend, timestamp, p
                )
            )

        return signals

    def _btc_dominance_rising(
        self,
        btc_df: pd.DataFrame,
        alt_returns: dict[str, float],
        alt_data: dict[str, pd.DataFrame],
        dominance_delta: float,
        btc_trend: float,
        timestamp: datetime,
        p: dict,
    ) -> list[Signal]:
        """Generate signals when BTC is outperforming alts."""
        signals: list[Signal] = []

        # Long BTC
        btc_close = float(btc_df.iloc[-1]["close"])
        btc_atr = ta.volatility.average_true_range(
            btc_df["high"], btc_df["low"], btc_df["close"],
            window=p["atr_period"],
        ).iloc[-1]

        if not pd.isna(btc_atr):
            strength = min(1.0, abs(dominance_delta) / 0.10)  # normalize to 10%
            if btc_trend > 0:
                strength = min(1.0, strength + 0.1)  # boost if BTC trending up

            signals.append(Signal(
                timestamp=timestamp,
                symbol=p["btc_symbol"],
                direction=SignalDirection.LONG,
                strength=strength,
                strategy_name=self.name,
                entry_price=btc_close,
                stop_loss=btc_close - p["atr_stop_multiplier"] * float(btc_atr),
                features={
                    "dominance_delta": dominance_delta,
                    "btc_return": float(btc_df["close"].pct_change(p["dominance_lookback"]).iloc[-1]),
                    "avg_alt_return": sum(alt_returns.values()) / len(alt_returns),
                    "rotation": "btc_favored",
                    "atr": float(btc_atr),
                    "close": btc_close,
                },
                rationale=(
                    f"BTC dominance rising ({dominance_delta:+.1%} vs alts). "
                    f"Capital flowing from alts to BTC. "
                    f"BTC 30d trend: {btc_trend:+.1%}."
                ),
            ))

        # Close weakest alts
        sorted_alts = sorted(alt_returns.items(), key=lambda x: x[1])
        for sym, ret in sorted_alts[:2]:  # close 2 weakest
            signals.append(Signal(
                timestamp=timestamp,
                symbol=sym,
                direction=SignalDirection.CLOSE,
                strength=0.6,
                strategy_name=self.name,
                features={
                    "dominance_delta": dominance_delta,
                    "alt_return": ret,
                    "rotation": "alt_unfavored",
                },
                rationale=(
                    f"BTC dominance rising — closing weak alt {sym} "
                    f"({ret:+.1%} vs BTC {dominance_delta:+.1%} outperf)."
                ),
            ))

        return signals

    def _btc_dominance_falling(
        self,
        btc_df: pd.DataFrame,
        alt_returns: dict[str, float],
        alt_data: dict[str, pd.DataFrame],
        dominance_delta: float,
        btc_trend: float,
        timestamp: datetime,
        p: dict,
    ) -> list[Signal]:
        """Generate signals when alts are outperforming BTC."""
        signals: list[Signal] = []

        # Long top-performing alts
        sorted_alts = sorted(alt_returns.items(), key=lambda x: x[1], reverse=True)

        for rank, (sym, ret) in enumerate(sorted_alts[:p["top_alts"]], start=1):
            df = alt_data[sym]
            close = float(df.iloc[-1]["close"])
            atr = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"],
                window=p["atr_period"],
            ).iloc[-1]

            if pd.isna(atr):
                continue

            strength = min(1.0, abs(dominance_delta) / 0.10)
            if ret > 0:
                strength = min(1.0, strength + 0.05 * rank)

            signals.append(Signal(
                timestamp=timestamp,
                symbol=sym,
                direction=SignalDirection.LONG,
                strength=strength,
                strategy_name=self.name,
                entry_price=close,
                stop_loss=close - p["atr_stop_multiplier"] * float(atr),
                features={
                    "dominance_delta": dominance_delta,
                    "alt_return": ret,
                    "rotation": "alt_favored",
                    "rank": rank,
                    "atr": float(atr),
                    "close": close,
                },
                rationale=(
                    f"Alt season — BTC dominance falling ({dominance_delta:+.1%}). "
                    f"{sym} rank #{rank} with {ret:+.1%} return. "
                    f"Capital flowing from BTC to alts."
                ),
            ))

        return signals
