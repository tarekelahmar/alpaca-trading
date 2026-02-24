"""Sentiment Analysis Strategy: News + Social Media Sentiment.

A standalone strategy that generates signals based on aggregate sentiment
from financial news (Alpaca News API + FinBERT) and social media (Finnhub).

Entry conditions (long):
    - Composite sentiment score > 0.2 (bullish)
    - At least 2 news articles in the last 48 hours
    - Price above 20 EMA (price confirmation — won't buy on sentiment alone)
    - Current volume above 20-day average (institutional agreement)
    - 20-day average volume above minimum threshold

Exit conditions:
    - Composite sentiment score < -0.2 (bearish sentiment shift)
    - OR trailing stop at 2x ATR below entry

Signal quality upgrades over basic VADER approach:
    1. FinBERT: Finance-trained BERT model that correctly classifies
       financial idioms ("killed it" = positive, "explosive growth" = positive)
    2. Time-decay: Recent articles weighted exponentially more (6h half-life)
    3. Volume confirmation: Today's volume must exceed 20-day average
       to confirm institutional activity agrees with sentiment

Data sources (all free):
    - Finnhub social sentiment API (50 calls/min free tier)
    - Alpaca News API (free with Alpaca account) + FinBERT scoring
"""

import pandas as pd
import ta

from strategies.base import Signal, SignalDirection, Strategy, StrategyConfig
from strategies.sentiment_fetcher import SentimentFetcher, SentimentData


DEFAULT_PARAMS = {
    # Sentiment thresholds (lowered from 0.3 for more signal generation)
    "sentiment_bull_threshold": 0.2,   # buy when composite > this
    "sentiment_bear_threshold": -0.2,  # close when composite < this
    "min_news_count": 2,               # need at least N articles to act
    "sentiment_lookback_hours": 48,    # look at last 48h of news

    # Price confirmation
    "require_price_confirmation": True,  # price must be above EMA
    "confirmation_ema": 20,             # EMA period for confirmation
    "atr_period": 14,
    "atr_stop_multiplier": 2.0,

    # Volume confirmation
    "require_volume_confirmation": True,   # current vol must beat average
    "volume_surge_threshold": 1.0,         # current vol / avg vol >= this
    "min_avg_volume": 200_000,
    "volume_lookback": 20,

    # Time-decay
    "decay_half_life_hours": 6.0,  # 6h half-life for news weighting

    # Source weighting
    "finnhub_weight": 0.6,
    "news_weight": 0.4,
}


class SentimentStrategy(Strategy):

    def __init__(self, config: StrategyConfig | None = None):
        if config is None:
            config = StrategyConfig(name="sentiment", params=DEFAULT_PARAMS)
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        self._fetcher = SentimentFetcher(
            finnhub_weight=merged["finnhub_weight"],
            news_weight=merged["news_weight"],
            lookback_hours=merged["sentiment_lookback_hours"],
            decay_half_life_hours=merged["decay_half_life_hours"],
        )
        self._sentiment_cache: dict[str, SentimentData] = {}

    def get_parameters(self) -> dict:
        return self.config.params

    def required_history_days(self) -> int:
        return self.config.params["confirmation_ema"] + 50

    def pre_fetch(self, symbols: list[str]) -> None:
        """Pre-fetch sentiment data for all symbols before signal generation.

        This should be called once before generate_signals() to batch
        API calls efficiently. The fetcher handles rate limiting internally.
        """
        results = self._fetcher.fetch_all(symbols)
        self._sentiment_cache = results

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        """Generate signals based on sentiment + price/volume confirmation."""
        signals: list[Signal] = []
        p = self.config.params

        # If pre_fetch wasn't called, fetch now (less efficient but works)
        if not self._sentiment_cache:
            self.pre_fetch(list(data.keys()))

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
        # Get cached sentiment
        sentiment = self._sentiment_cache.get(symbol)
        if sentiment is None:
            return None

        # Need minimum news coverage
        if sentiment.news_count < p["min_news_count"]:
            return None

        df = df.copy()

        # Compute indicators
        df["ema_confirm"] = ta.trend.ema_indicator(
            df["close"], window=p["confirmation_ema"]
        )
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=p["atr_period"]
        )
        df["avg_volume"] = df["volume"].rolling(
            window=p["volume_lookback"]
        ).mean()

        curr = df.iloc[-1]

        if pd.isna(curr["ema_confirm"]) or pd.isna(curr["atr"]):
            return None

        # Minimum average volume filter
        if curr["avg_volume"] < p["min_avg_volume"]:
            return None

        timestamp = df.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        atr = float(curr["atr"])
        score = sentiment.composite_score

        # Compute volume ratio for features/confirmation
        avg_vol = float(curr["avg_volume"])
        current_vol = float(curr["volume"])
        volume_ratio = current_vol / avg_vol if avg_vol > 0 else 0.0

        # CLOSE signal: bearish sentiment shift (no volume requirement)
        if score < p["sentiment_bear_threshold"]:
            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=SignalDirection.CLOSE,
                strength=min(1.0, abs(score - p["sentiment_bear_threshold"]) / 0.5),
                strategy_name=self.name,
                features={
                    "sentiment_score": score,
                    "finnhub_score": sentiment.finnhub_score,
                    "news_finbert_score": sentiment.news_finbert_score,
                    "news_count": sentiment.news_count,
                    "bullish_mentions": sentiment.bullish_mentions,
                    "bearish_mentions": sentiment.bearish_mentions,
                    "volume_ratio": volume_ratio,
                    "sources": sentiment.sources,
                    "close": float(curr["close"]),
                },
                rationale=(
                    f"Bearish sentiment: composite={score:.3f} "
                    f"(threshold={p['sentiment_bear_threshold']}). "
                    f"News count: {sentiment.news_count}. "
                    f"Sources: {', '.join(sentiment.sources)}."
                ),
            )

        # LONG signal: bullish sentiment + price + volume confirmation
        if score > p["sentiment_bull_threshold"]:
            # Price confirmation: must be above EMA
            if p["require_price_confirmation"]:
                if curr["close"] < curr["ema_confirm"]:
                    return None

            # Volume confirmation: current bar volume above average
            if p["require_volume_confirmation"]:
                if volume_ratio < p["volume_surge_threshold"]:
                    return None

            stop_loss = float(curr["close"]) - p["atr_stop_multiplier"] * atr

            # Strength: how far above threshold, scaled to 0–1
            # Boost strength slightly if volume is surging (institutional agreement)
            base_strength = min(1.0, (score - p["sentiment_bull_threshold"]) / 0.5)
            volume_boost = min(0.2, max(0.0, (volume_ratio - 1.0) * 0.1))
            strength = min(1.0, base_strength + volume_boost)

            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=SignalDirection.LONG,
                strength=strength,
                strategy_name=self.name,
                entry_price=float(curr["close"]),
                stop_loss=stop_loss,
                features={
                    "sentiment_score": score,
                    "finnhub_score": sentiment.finnhub_score,
                    "news_finbert_score": sentiment.news_finbert_score,
                    "news_count": sentiment.news_count,
                    "bullish_mentions": sentiment.bullish_mentions,
                    "bearish_mentions": sentiment.bearish_mentions,
                    "volume_ratio": volume_ratio,
                    "sources": sentiment.sources,
                    "ema_confirm": float(curr["ema_confirm"]),
                    "atr": atr,
                    "close": float(curr["close"]),
                },
                rationale=(
                    f"Bullish sentiment: composite={score:.3f} "
                    f"(threshold={p['sentiment_bull_threshold']}). "
                    f"News: {sentiment.news_count} articles (FinBERT scored). "
                    f"Price ${curr['close']:.2f} > EMA{p['confirmation_ema']} "
                    f"${curr['ema_confirm']:.2f}. "
                    f"Volume {volume_ratio:.1f}x avg. "
                    f"Stop: ${stop_loss:.2f}."
                ),
            )

        return None
