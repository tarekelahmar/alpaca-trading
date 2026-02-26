"""Crypto Market Regime Detection and Allocation.

Uses BTC as the crypto equivalent of SPY for regime detection.
Crypto-specific adaptations:
    - BTC 20-day realized volatility replaces VIX
    - Higher volatility thresholds (crypto is inherently more volatile)
    - No breadth metric (small universe makes it meaningless)
    - Faster EMA pair (50/100 instead of 50/200)

Allocation weights are tuned for crypto:
    - Mean reversion ONLY in ranging markets (0% in trends)
    - BTC dominance rotation replaces sector rotation
    - Higher cash reserve in high volatility (crypto drawdowns are brutal)
    - No gap trading, no earnings momentum, no sentiment (equity-only)
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd
import ta

from regime.detector import RegimeType, RegimeClassification


# Crypto-specific volatility thresholds (annualized realized vol)
# Crypto normal ≈ 40-60%, equities normal ≈ 12-18%
CRYPTO_THRESHOLDS = {
    "vol_low": 40,       # annualized vol below 40% → calm (rare in crypto)
    "vol_high": 80,      # above 80% → elevated
    "vol_extreme": 120,  # above 120% → panic
    "adx_trending": 20,  # lower bar for crypto (trends emerge faster)
    "ema_fast": 50,
    "ema_slow": 100,
}


class CryptoRegimeDetector:
    """Detect crypto market regime using BTC as the benchmark.

    Uses the same RegimeType enum as equities for compatibility
    with the rest of the system (engine, optimizer, drawdown, etc.).
    """

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = {**CRYPTO_THRESHOLDS, **(thresholds or {})}

    def classify(
        self,
        btc_data: pd.DataFrame,
        vix_data: pd.DataFrame | None = None,
        breadth_pct: float | None = None,
    ) -> RegimeClassification:
        """Classify crypto market regime using BTC data.

        Matches the interface of RegimeDetector.classify() so the engine
        can use either detector interchangeably.

        Args:
            btc_data: BTC/USD OHLCV DataFrame with DatetimeIndex.
            vix_data: Ignored for crypto (we compute our own vol).
            breadth_pct: Ignored for crypto.

        Returns:
            RegimeClassification compatible with equity regime types.
        """
        t = self.thresholds
        indicators: dict = {}
        scores: dict[RegimeType, float] = {r: 0.0 for r in RegimeType}

        # 1. BTC realized volatility (annualized 20-day)
        returns = btc_data["close"].pct_change()
        realized_vol = float(returns.rolling(20).std().iloc[-1] * (365 ** 0.5) * 100)
        indicators["realized_vol"] = realized_vol

        if realized_vol >= t["vol_extreme"]:
            scores[RegimeType.HIGH_VOLATILITY] += 3.0
        elif realized_vol >= t["vol_high"]:
            scores[RegimeType.HIGH_VOLATILITY] += 1.5
            scores[RegimeType.TRENDING_BEARISH] += 0.5
        elif realized_vol <= t["vol_low"]:
            scores[RegimeType.TRENDING_BULLISH] += 1.0
            scores[RegimeType.RANGING] += 0.5

        # 2. BTC trend via ADX
        if len(btc_data) > 20:
            adx = ta.trend.adx(
                btc_data["high"], btc_data["low"], btc_data["close"],
                window=14,
            )
            adx_current = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
        else:
            adx_current = 0.0
        indicators["adx"] = adx_current

        if adx_current >= t["adx_trending"]:
            # Determine trend direction from EMAs
            ema_fast = ta.trend.ema_indicator(
                btc_data["close"], window=t["ema_fast"]
            ).iloc[-1]
            ema_slow = ta.trend.ema_indicator(
                btc_data["close"], window=t["ema_slow"]
            ).iloc[-1]
            indicators["ema_fast"] = float(ema_fast) if not pd.isna(ema_fast) else 0.0
            indicators["ema_slow"] = float(ema_slow) if not pd.isna(ema_slow) else 0.0

            if not pd.isna(ema_fast) and not pd.isna(ema_slow):
                if ema_fast > ema_slow:
                    scores[RegimeType.TRENDING_BULLISH] += 2.0
                else:
                    scores[RegimeType.TRENDING_BEARISH] += 2.0
        else:
            scores[RegimeType.RANGING] += 2.0

        # 3. Price vs EMAs
        current_price = float(btc_data.iloc[-1]["close"])
        ema50 = ta.trend.ema_indicator(btc_data["close"], window=50)
        ema100 = ta.trend.ema_indicator(btc_data["close"], window=100)

        ema50_val = float(ema50.iloc[-1]) if not pd.isna(ema50.iloc[-1]) else current_price
        ema100_val = float(ema100.iloc[-1]) if not pd.isna(ema100.iloc[-1]) else current_price
        indicators["btc_close"] = current_price
        indicators["ema50"] = ema50_val
        indicators["ema100"] = ema100_val

        if current_price > ema50_val > ema100_val:
            scores[RegimeType.TRENDING_BULLISH] += 1.5
        elif current_price < ema50_val < ema100_val:
            scores[RegimeType.TRENDING_BEARISH] += 1.5

        # 4. Recent drawdown check (30-day high vs current)
        high_30d = float(btc_data["high"].tail(30).max())
        drawdown_30d = (high_30d - current_price) / high_30d if high_30d > 0 else 0.0
        indicators["drawdown_30d"] = drawdown_30d

        if drawdown_30d > 0.25:  # 25%+ drop in 30 days
            scores[RegimeType.HIGH_VOLATILITY] += 2.0
            scores[RegimeType.TRENDING_BEARISH] += 1.0
        elif drawdown_30d > 0.15:  # 15%+ drop
            scores[RegimeType.HIGH_VOLATILITY] += 1.0

        # Determine winner
        total = sum(scores.values()) or 1.0
        best_regime = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best_regime] / total

        timestamp = btc_data.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        return RegimeClassification(
            regime=best_regime,
            confidence=confidence,
            timestamp=timestamp,
            indicators=indicators,
        )


# --- Crypto Allocation Weights ---

@dataclass
class CryptoAllocationWeights:
    """Allocation weights for crypto strategies.

    Maps to the AllocationWeights fields:
        trend_following -> crypto_trend_following
        mean_reversion -> crypto_mean_reversion
        momentum -> crypto_momentum
        btc_dominance -> replaces sector_rotation
        cash -> cash reserve
    """
    trend_following: float
    mean_reversion: float
    momentum: float
    btc_dominance: float
    cash: float

    def normalize(self) -> "CryptoAllocationWeights":
        """Ensure weights sum to 1.0."""
        total = (
            self.trend_following + self.mean_reversion
            + self.momentum + self.btc_dominance + self.cash
        )
        if total == 0:
            return CryptoAllocationWeights(0, 0, 0, 0, 1.0)
        return CryptoAllocationWeights(
            trend_following=self.trend_following / total,
            mean_reversion=self.mean_reversion / total,
            momentum=self.momentum / total,
            btc_dominance=self.btc_dominance / total,
            cash=self.cash / total,
        )

    def invested_fraction(self) -> float:
        return 1.0 - self.cash

    def to_equity_format(self):
        """Convert to AllocationWeights for compatibility with the engine.

        Maps crypto strategies to equity AllocationWeights fields:
            btc_dominance -> sector_rotation (same slot)
            Other equity fields (sentiment, earnings, gap) -> 0.0
        """
        from regime.allocator import AllocationWeights
        return AllocationWeights(
            trend_following=self.trend_following,
            mean_reversion=self.mean_reversion,
            momentum=self.momentum,
            cash=self.cash,
            sentiment=0.0,
            earnings_momentum=0.0,
            gap_trading=0.0,
            sector_rotation=self.btc_dominance,  # reuse this slot
        )


# Crypto regime -> allocation mapping
# Key differences from equities:
# - NO mean reversion in trending markets (crypto trends are deadly for MR)
# - BTC dominance rotation is the main diversifier
# - Higher cash in volatile regimes (crypto crashes are 30-50%)
CRYPTO_REGIME_ALLOCATIONS: dict[RegimeType, CryptoAllocationWeights] = {
    RegimeType.TRENDING_BULLISH: CryptoAllocationWeights(
        trend_following=0.30,
        mean_reversion=0.00,  # DANGEROUS in crypto bull trends
        momentum=0.35,
        btc_dominance=0.25,
        cash=0.10,
    ),
    RegimeType.TRENDING_BEARISH: CryptoAllocationWeights(
        trend_following=0.20,  # short-side trend following
        mean_reversion=0.00,  # DANGEROUS in crypto bear trends
        momentum=0.10,
        btc_dominance=0.15,
        cash=0.55,  # heavy cash in bear market
    ),
    RegimeType.RANGING: CryptoAllocationWeights(
        trend_following=0.10,
        mean_reversion=0.30,  # ONLY safe regime for crypto MR
        momentum=0.15,
        btc_dominance=0.25,
        cash=0.20,
    ),
    RegimeType.HIGH_VOLATILITY: CryptoAllocationWeights(
        trend_following=0.10,
        mean_reversion=0.00,  # absolutely not in high vol
        momentum=0.05,
        btc_dominance=0.10,
        cash=0.75,  # heavy cash — survive first
    ),
}


class CryptoRegimeAllocator:
    """Maps crypto regime to allocation weights."""

    def __init__(
        self,
        allocations: dict[RegimeType, CryptoAllocationWeights] | None = None,
        max_invested: float = 0.90,  # more conservative than equities
    ):
        self.allocations = allocations or CRYPTO_REGIME_ALLOCATIONS
        self.max_invested = max_invested

    def get_allocation(self, regime: RegimeType) -> CryptoAllocationWeights:
        """Get crypto allocation weights for a given regime."""
        weights = self.allocations.get(regime)
        if weights is None:
            return CryptoAllocationWeights(0.0, 0.0, 0.0, 0.0, 1.0)

        invested = weights.invested_fraction()
        if invested > self.max_invested:
            scale = self.max_invested / invested
            weights = CryptoAllocationWeights(
                trend_following=weights.trend_following * scale,
                mean_reversion=weights.mean_reversion * scale,
                momentum=weights.momentum * scale,
                btc_dominance=weights.btc_dominance * scale,
                cash=1.0 - self.max_invested,
            )

        return weights.normalize()
