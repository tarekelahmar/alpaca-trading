"""Market Regime Detector.

Classifies the current market regime using:
    - VIX level (volatility)
    - Market breadth (% of S&P above 200 MA)
    - SPY trend strength (ADX)
    - SPY EMA alignment (50 vs 200)

Regimes:
    - trending_bullish: Strong uptrend, low-to-moderate volatility
    - trending_bearish: Strong downtrend, rising volatility
    - ranging: No clear trend, low volatility
    - high_volatility: Elevated VIX, unstable conditions
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd
import ta


class RegimeType(Enum):
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class RegimeClassification:
    regime: RegimeType
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    indicators: dict


DEFAULT_THRESHOLDS = {
    "vix_low": 15,
    "vix_high": 25,
    "vix_extreme": 35,
    "adx_trending": 25,
    "breadth_bullish": 0.60,
    "breadth_bearish": 0.40,
}


class RegimeDetector:

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def classify(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame | None = None,
        breadth_pct: float | None = None,
    ) -> RegimeClassification:
        """Classify current market regime.

        Args:
            spy_data: SPY OHLCV DataFrame with DatetimeIndex.
            vix_data: VIX close prices DataFrame (optional).
                      If not provided, uses SPY volatility as proxy.
            breadth_pct: Fraction of S&P 500 stocks above 200 MA (0.0-1.0).
                         If not provided, this signal is skipped.

        Returns:
            RegimeClassification with regime type, confidence, and indicators.
        """
        t = self.thresholds
        indicators: dict = {}
        scores: dict[RegimeType, float] = {r: 0.0 for r in RegimeType}

        # 1. VIX / Volatility
        if vix_data is not None and len(vix_data) > 0:
            vix_current = float(vix_data.iloc[-1]["close"])
        else:
            # Proxy: annualized rolling 20-day volatility of SPY
            returns = spy_data["close"].pct_change()
            vix_current = float(returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100)

        indicators["vix"] = vix_current

        if vix_current >= t["vix_extreme"]:
            scores[RegimeType.HIGH_VOLATILITY] += 3.0
        elif vix_current >= t["vix_high"]:
            scores[RegimeType.HIGH_VOLATILITY] += 1.5
            scores[RegimeType.TRENDING_BEARISH] += 0.5
        elif vix_current <= t["vix_low"]:
            scores[RegimeType.TRENDING_BULLISH] += 1.0
            scores[RegimeType.RANGING] += 0.5

        # 2. SPY trend via ADX
        adx = ta.trend.adx(
            spy_data["high"], spy_data["low"], spy_data["close"], window=14
        )
        adx_current = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
        indicators["adx"] = adx_current

        if adx_current >= t["adx_trending"]:
            # Strong trend — figure out direction from EMA
            ema50 = ta.trend.ema_indicator(spy_data["close"], window=50).iloc[-1]
            ema200 = ta.trend.ema_indicator(spy_data["close"], window=200).iloc[-1]
            indicators["ema50"] = float(ema50)
            indicators["ema200"] = float(ema200)

            if ema50 > ema200:
                scores[RegimeType.TRENDING_BULLISH] += 2.0
            else:
                scores[RegimeType.TRENDING_BEARISH] += 2.0
        else:
            scores[RegimeType.RANGING] += 2.0

        # 3. EMA alignment (price vs EMAs)
        current_price = float(spy_data.iloc[-1]["close"])
        ema50 = float(ta.trend.ema_indicator(spy_data["close"], window=50).iloc[-1])
        ema200 = float(ta.trend.ema_indicator(spy_data["close"], window=200).iloc[-1])
        indicators["spy_close"] = current_price

        if current_price > ema50 > ema200:
            scores[RegimeType.TRENDING_BULLISH] += 1.0
        elif current_price < ema50 < ema200:
            scores[RegimeType.TRENDING_BEARISH] += 1.0

        # 4. Market breadth
        if breadth_pct is not None:
            indicators["breadth_pct"] = breadth_pct
            if breadth_pct >= t["breadth_bullish"]:
                scores[RegimeType.TRENDING_BULLISH] += 1.5
            elif breadth_pct <= t["breadth_bearish"]:
                scores[RegimeType.TRENDING_BEARISH] += 1.0
                scores[RegimeType.HIGH_VOLATILITY] += 0.5
            else:
                scores[RegimeType.RANGING] += 0.5

        # Determine winner
        total = sum(scores.values()) or 1.0
        best_regime = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best_regime] / total

        timestamp = spy_data.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        return RegimeClassification(
            regime=best_regime,
            confidence=confidence,
            timestamp=timestamp,
            indicators=indicators,
        )
