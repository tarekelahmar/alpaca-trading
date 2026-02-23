"""Sentiment Data Fetcher — aggregates sentiment from multiple free sources.

Sources (both free, both scored by FinBERT):
    1. Finnhub Company News API (free tier, 50 calls/min)
    2. Alpaca News API (free with Alpaca account)

Both sources provide raw news headlines and summaries. We score
them locally with FinBERT (ProsusAI/finbert), a BERT model fine-tuned
on financial text that correctly handles financial idioms like
"killed it this quarter" = positive, "explosive growth" = positive.

FinBERT runs locally if torch is installed, otherwise falls back
to HuggingFace's free Inference API.

Time-decay weighting: Recent articles are weighted exponentially more
than older ones. Half-life of 6 hours means a headline from 12 hours
ago has 25% the weight of a headline from right now. This captures
the well-documented rapid decay of news-driven price impact.

Returns a composite sentiment score per symbol (-1.0 to +1.0):
    - Positive = bullish sentiment
    - Negative = bearish sentiment
    - 0.0 = neutral or no data

Caches results to avoid redundant API calls within a single run.
"""

import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import finnhub
import requests as http_requests
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest


# FinBERT scoring — tries local model first, falls back to HF API
_finbert_pipeline = None
_finbert_mode = None  # "local" or "api" or None


def _init_finbert():
    """Initialize FinBERT — local if torch is available, else HF API."""
    global _finbert_pipeline, _finbert_mode

    if _finbert_mode is not None:
        return  # already initialized

    # Try local model first (requires torch + transformers)
    try:
        from transformers import pipeline as hf_pipeline
        print("[FinBERT] Loading local model...", file=sys.stderr)
        _finbert_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,
            truncation=True,
            max_length=512,
        )
        _finbert_mode = "local"
        print("[FinBERT] Local model loaded.", file=sys.stderr)
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"[FinBERT] Local model failed: {e}", file=sys.stderr)

    # Fall back to HF Inference API
    _finbert_mode = "api"
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print("[FinBERT] Using HF Inference API (authenticated).", file=sys.stderr)
    else:
        print(
            "[FinBERT] Using HF Inference API (unauthenticated — "
            "set HF_TOKEN for better reliability).",
            file=sys.stderr,
        )


def _score_with_finbert(texts: list[str]) -> list[dict]:
    """Score texts using FinBERT — local model or HF API.

    Args:
        texts: List of texts to classify.

    Returns:
        List of dicts with 'label' and 'score' keys.
        Labels are 'positive', 'negative', or 'neutral'.
    """
    _init_finbert()

    if _finbert_mode == "local":
        return _score_local(texts)
    else:
        return _score_api(texts)


def _score_local(texts: list[str]) -> list[dict]:
    """Score with local FinBERT model (fast, no API calls)."""
    truncated = [t[:512] for t in texts]
    raw_results = _finbert_pipeline(truncated)
    results = []
    for r in raw_results:
        results.append({
            "label": r["label"].lower(),
            "score": r["score"],
        })
    return results


def _score_api(texts: list[str]) -> list[dict]:
    """Score via HuggingFace Inference API."""
    hf_token = os.environ.get("HF_TOKEN", "")
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    # Try both the new router URL and the legacy URL
    urls = [
        "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert",
        "https://api-inference.huggingface.co/models/ProsusAI/finbert",
    ]

    results = []

    for i in range(0, len(texts), 10):
        batch = [t[:512] for t in texts[i:i + 10]]
        scored = False

        for url in urls:
            try:
                resp = http_requests.post(
                    url,
                    headers=headers,
                    json={"inputs": batch, "options": {"wait_for_model": True}},
                    timeout=30,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    for item in data:
                        if isinstance(item, list) and len(item) > 0:
                            top = item[0]
                            results.append({
                                "label": top.get("label", "neutral"),
                                "score": top.get("score", 0.5),
                            })
                        else:
                            results.append({"label": "neutral", "score": 0.5})
                    scored = True
                    break
                elif resp.status_code == 503:
                    print(
                        "[FinBERT] Model loading on HF servers, waiting 20s...",
                        file=sys.stderr,
                    )
                    time.sleep(20)
                    resp2 = http_requests.post(
                        url,
                        headers=headers,
                        json={"inputs": batch, "options": {"wait_for_model": True}},
                        timeout=60,
                    )
                    if resp2.status_code == 200:
                        data = resp2.json()
                        for item in data:
                            if isinstance(item, list) and len(item) > 0:
                                top = item[0]
                                results.append({
                                    "label": top.get("label", "neutral"),
                                    "score": top.get("score", 0.5),
                                })
                            else:
                                results.append({"label": "neutral", "score": 0.5})
                        scored = True
                        break
                # else try next URL

            except Exception:
                continue

        if not scored:
            # All URLs failed — return neutral
            results.extend([{"label": "neutral", "score": 0.5}] * len(batch))

    return results


@dataclass
class SentimentData:
    """Sentiment data for a single symbol."""
    symbol: str
    composite_score: float  # -1.0 to +1.0
    finnhub_score: float | None  # raw Finnhub score
    news_finbert_score: float | None  # FinBERT on Alpaca news headlines
    news_count: int  # number of articles analyzed
    bullish_mentions: int  # Finnhub bullish mention count
    bearish_mentions: int  # Finnhub bearish mention count
    sources: list[str] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FinnhubNewsSentimentClient:
    """Fetches company news from Finnhub free tier and scores with FinBERT.

    Uses the /company-news endpoint (free tier) instead of /social-sentiment
    (premium only). Headlines and summaries are scored locally with FinBERT,
    giving us an independent news source from Alpaca.

    Time-decay weighting is applied identically to the Alpaca news client.
    """

    def __init__(
        self,
        api_key: str | None = None,
        decay_half_life_hours: float = 6.0,
        lookback_hours: int = 48,
    ):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        if not self.api_key:
            print(
                "[SentimentFetcher] WARNING: FINNHUB_API_KEY not set, "
                "Finnhub news disabled",
                file=sys.stderr,
            )
            self.client = None
        else:
            self.client = finnhub.Client(api_key=self.api_key)
        self.decay_half_life_hours = decay_half_life_hours
        self.lookback_hours = lookback_hours

    def fetch_sentiment(
        self, symbol: str
    ) -> tuple[float | None, int, int]:
        """Fetch company news from Finnhub and score with FinBERT.

        Returns:
            Tuple of (sentiment_score, bullish_count, bearish_count).
            sentiment_score is -1.0 to +1.0, or None on failure.
        """
        if self.client is None:
            return None, 0, 0

        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(hours=self.lookback_hours)
            # Finnhub company_news expects YYYY-MM-DD strings
            from_date = start.strftime("%Y-%m-%d")
            to_date = now.strftime("%Y-%m-%d")

            articles = self.client.company_news(symbol, _from=from_date, to=to_date)

            if not articles:
                return None, 0, 0

            # Extract headlines + summaries (limit to 50 most recent)
            texts = []
            timestamps = []
            for article in articles[:50]:
                headline = article.get("headline", "")
                summary = article.get("summary", "")
                text = headline
                if summary and summary.strip():
                    text += " " + summary
                if not text.strip():
                    continue
                texts.append(text)
                # Finnhub timestamps are Unix epoch seconds
                ts_epoch = article.get("datetime", 0)
                if ts_epoch:
                    timestamps.append(
                        datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
                    )
                else:
                    timestamps.append(None)

            if not texts:
                return None, 0, 0

            # Score with FinBERT
            results = _score_with_finbert(texts)

            # Compute time-decay weighted sentiment
            weighted_sum = 0.0
            weight_sum = 0.0
            bullish = 0
            bearish = 0

            for i, result in enumerate(results):
                label = result["label"].lower()
                confidence = result["score"]

                if label == "positive":
                    score = confidence
                    bullish += 1
                elif label == "negative":
                    score = -confidence
                    bearish += 1
                else:
                    score = 0.0

                # Time-decay weight
                age_hours = 0.0
                if timestamps[i] is not None:
                    age_hours = max(
                        0.0, (now - timestamps[i]).total_seconds() / 3600.0
                    )

                decay_lambda = math.log(2) / self.decay_half_life_hours
                time_weight = math.exp(-decay_lambda * age_hours)

                weighted_sum += score * time_weight
                weight_sum += time_weight

            if weight_sum <= 0:
                return None, 0, 0

            final_score = weighted_sum / weight_sum
            return final_score, bullish, bearish

        except Exception as e:
            print(
                f"[SentimentFetcher] Finnhub news error for {symbol}: {e}",
                file=sys.stderr,
            )
            return None, 0, 0


class AlpacaNewsSentimentClient:
    """Fetches news from Alpaca and scores with FinBERT (finance-aware NLP).

    FinBERT advantages over VADER:
        - Trained on 10K+ financial texts (analyst notes, earnings, news)
        - Correctly classifies financial idioms ("killed it" = positive)
        - Returns calibrated probabilities for positive/negative/neutral
        - Significantly more accurate on financial headlines

    Time-decay weighting:
        - Recent articles weighted exponentially more than older ones
        - Half-life of 6 hours (configurable)
        - A 12-hour-old article has 25% the weight of a brand-new one
        - Captures rapid decay of news-driven price impact
    """

    def __init__(self, decay_half_life_hours: float = 6.0):
        key = os.environ.get("ALPACA_API_KEY_ID", "")
        secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
        if key and secret:
            self.client = NewsClient(key, secret)
        else:
            self.client = NewsClient()
        self.decay_half_life_hours = decay_half_life_hours

    def fetch_sentiment(
        self, symbol: str, lookback_hours: int = 48
    ) -> tuple[float | None, int]:
        """Fetch news headlines and compute time-weighted FinBERT sentiment.

        Returns:
            Tuple of (weighted_sentiment_score, num_articles).
            Score is -1.0 to +1.0, or None on failure.
        """
        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(hours=lookback_hours)
            request = NewsRequest(
                symbols=symbol,
                start=start,
                limit=50,
            )
            news_set = self.client.get_news(request)

            # NewsSet stores articles in .data["news"]
            if not news_set or not hasattr(news_set, "data"):
                return None, 0
            articles = (
                news_set.data.get("news", [])
                if isinstance(news_set.data, dict)
                else []
            )
            if not articles:
                return None, 0

            # Extract text and timestamps
            texts = []
            timestamps = []
            for article in articles:
                if isinstance(article, dict):
                    headline = article.get("headline", "")
                    summary = article.get("summary", "")
                    created = article.get("created_at")
                else:
                    headline = getattr(article, "headline", "")
                    summary = getattr(article, "summary", "")
                    created = getattr(article, "created_at", None)

                text = headline
                if summary and summary.strip():
                    text += " " + summary
                if not text.strip():
                    continue

                texts.append(text)
                timestamps.append(created)

            if not texts:
                return None, 0

            # Batch score with FinBERT via HuggingFace Inference API
            results = _score_with_finbert(texts)

            # Convert FinBERT labels to scores and apply time-decay weights
            weighted_sum = 0.0
            weight_sum = 0.0

            for i, result in enumerate(results):
                label = result["label"].lower()
                confidence = result["score"]

                # Map label to signed score
                if label == "positive":
                    score = confidence
                elif label == "negative":
                    score = -confidence
                else:  # neutral
                    score = 0.0

                # Time-decay weight: exp(-lambda * age_hours)
                # lambda = ln(2) / half_life
                age_hours = 0.0
                if timestamps[i] is not None:
                    if isinstance(timestamps[i], datetime):
                        ts = timestamps[i]
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        age_hours = (now - ts).total_seconds() / 3600.0
                    age_hours = max(0.0, age_hours)

                decay_lambda = math.log(2) / self.decay_half_life_hours
                time_weight = math.exp(-decay_lambda * age_hours)

                weighted_sum += score * time_weight
                weight_sum += time_weight

            if weight_sum <= 0:
                return None, 0

            final_score = weighted_sum / weight_sum
            return final_score, len(texts)

        except Exception as e:
            print(
                f"[SentimentFetcher] Alpaca news error for {symbol}: {e}",
                file=sys.stderr,
            )
            return None, 0


class SentimentFetcher:
    """Aggregates sentiment from all sources into a composite score.

    Uses two independent news sources, both scored by FinBERT:
        1. Finnhub company news (free tier /company-news endpoint)
        2. Alpaca News API (free with Alpaca account)

    Both sources are time-decay weighted and combined into a
    weighted average composite score.
    """

    def __init__(
        self,
        finnhub_weight: float = 0.6,
        news_weight: float = 0.4,
        lookback_hours: int = 48,
        decay_half_life_hours: float = 6.0,
        rate_limit_delay: float = 1.25,
    ):
        self.finnhub_client = FinnhubNewsSentimentClient(
            decay_half_life_hours=decay_half_life_hours,
            lookback_hours=lookback_hours,
        )
        self.news_client = AlpacaNewsSentimentClient(
            decay_half_life_hours=decay_half_life_hours,
        )
        self.finnhub_weight = finnhub_weight
        self.news_weight = news_weight
        self.lookback_hours = lookback_hours
        self.rate_limit_delay = rate_limit_delay
        self._cache: dict[str, SentimentData] = {}

    def fetch_all(self, symbols: list[str]) -> dict[str, SentimentData]:
        """Fetch sentiment for all symbols.

        Results are cached for the duration of this object's lifetime.
        """
        results: dict[str, SentimentData] = {}
        to_fetch = [s for s in symbols if s not in self._cache]

        if to_fetch:
            print(
                f"[SentimentFetcher] Fetching sentiment for {len(to_fetch)} symbols...",
                file=sys.stderr,
            )

        for i, symbol in enumerate(to_fetch):
            data = self._fetch_symbol(symbol)
            self._cache[symbol] = data
            results[symbol] = data

            # Rate limit for Finnhub (50 calls/min → 1.25s between calls)
            if self.rate_limit_delay > 0 and i < len(to_fetch) - 1:
                time.sleep(self.rate_limit_delay)

            # Progress every 25 symbols
            if (i + 1) % 25 == 0:
                print(
                    f"[SentimentFetcher] Progress: {i + 1}/{len(to_fetch)}",
                    file=sys.stderr,
                )

        # Include cached results
        for symbol in symbols:
            if symbol in self._cache:
                results[symbol] = self._cache[symbol]

        return results

    def _fetch_symbol(self, symbol: str) -> SentimentData:
        """Fetch and combine sentiment for a single symbol."""
        sources = []

        # Finnhub company news + FinBERT scoring
        fh_score, fh_bullish, fh_bearish = self.finnhub_client.fetch_sentiment(symbol)
        if fh_score is not None:
            sources.append("finnhub_news_finbert")

        # Alpaca News + FinBERT (time-decay weighted)
        news_score, news_count = self.news_client.fetch_sentiment(
            symbol, self.lookback_hours
        )
        if news_score is not None:
            sources.append("alpaca_news_finbert")

        # Total article count from both sources
        total_news = news_count + fh_bullish + fh_bearish

        # Compute composite
        composite = self._compute_composite(fh_score, news_score)

        return SentimentData(
            symbol=symbol,
            composite_score=composite,
            finnhub_score=fh_score,
            news_finbert_score=news_score,
            news_count=total_news,
            bullish_mentions=fh_bullish,
            bearish_mentions=fh_bearish,
            sources=sources,
        )

    def _compute_composite(
        self,
        finnhub_score: float | None,
        news_score: float | None,
    ) -> float:
        """Combine scores from multiple sources into a single composite.

        Uses weighted average of available sources. If only one source
        is available, uses that source's score directly.
        """
        scores = []
        weights = []

        if finnhub_score is not None:
            scores.append(finnhub_score)
            weights.append(self.finnhub_weight)

        if news_score is not None:
            scores.append(news_score)
            weights.append(self.news_weight)

        if not scores:
            return 0.0

        # Weighted average, normalized
        total_weight = sum(weights)
        composite = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, composite))

    def get_cached(self, symbol: str) -> SentimentData | None:
        """Get cached sentiment data for a symbol."""
        return self._cache.get(symbol)

    def clear_cache(self):
        """Clear the sentiment cache."""
        self._cache.clear()
