"""Earnings Momentum Strategy: Post-Earnings Announcement Drift (PEAD).

Exploits the well-documented tendency for stocks to continue drifting
in the direction of their earnings surprise for 30-60 days after the
announcement. This is one of the most persistent anomalies in finance.

How it works:
    1. Fetch recent earnings data from Finnhub (free tier)
    2. Identify stocks that reported in the last 5 trading days
    3. Calculate earnings surprise: (actual - estimate) / |estimate|
    4. BUY if surprise > +5% AND post-earnings price action confirms
    5. CLOSE if we hold a stock that just reported a negative surprise

Why it works:
    - Analysts are slow to revise estimates after surprises
    - Institutional investors take days/weeks to reposition
    - Retail traders underreact to earnings data
    - The drift lasts 30-60 days on average

Data sources (all free):
    - Finnhub earnings calendar API (free tier, /stock/earnings)
    - Price data already available from Alpaca

Entry conditions (long):
    - Earnings surprise > +5% (beat estimates meaningfully)
    - Stock gapped up or moved up on earnings day (price confirmation)
    - Reported within last 5 trading days (fresh catalyst)
    - Price above 20 EMA (uptrend context)
    - Volume surge on earnings day (institutional participation)

Entry conditions (short):
    - Earnings surprise < -5% (missed estimates meaningfully)
    - Stock gapped down on earnings day (price confirmation)
    - Reported within last 5 trading days (fresh catalyst)
    - Price below 20 EMA (downtrend context)
    - Volume surge on earnings day (institutional participation)
    - If not fully confirmed, generates CLOSE instead of SHORT

Exit conditions:
    - Negative earnings surprise closes longs (or opens shorts if confirmed)
    - OR 30 trading days since entry (drift exhausted)
    - OR trailing stop at 2.5x ATR from entry
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import finnhub
import pandas as pd
import ta

from strategies.base import Signal, SignalDirection, Strategy, StrategyConfig


DEFAULT_PARAMS = {
    # Earnings thresholds
    "surprise_threshold_pct": 5.0,      # buy if surprise > +5%
    "negative_surprise_pct": -5.0,      # close if surprise < -5%
    "lookback_days": 5,                 # only look at earnings from last 5 trading days

    # Price confirmation
    "require_price_confirmation": True,  # price must have moved up post-earnings
    "min_earnings_day_return": 0.0,     # minimum return on earnings day (0% = just positive)
    "confirmation_ema": 20,

    # Volume confirmation
    "min_volume_surge": 1.5,            # earnings day volume must be 1.5x avg
    "min_avg_volume": 200_000,
    "volume_lookback": 20,

    # Risk management
    "atr_period": 14,
    "atr_stop_multiplier": 2.5,        # wider stop for earnings plays (gappy)

    # Strength scaling
    "max_surprise_for_scaling": 30.0,   # cap surprise at 30% for strength calc
}


class EarningsMomentumStrategy(Strategy):

    def __init__(self, config: StrategyConfig | None = None):
        if config is None:
            config = StrategyConfig(name="earnings_momentum", params=DEFAULT_PARAMS)
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

        # Initialize Finnhub client for earnings data
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        if api_key:
            self._finnhub = finnhub.Client(api_key=api_key)
        else:
            self._finnhub = None
            print(
                "[EarningsMomentum] WARNING: FINNHUB_API_KEY not set, "
                "earnings strategy disabled",
                file=sys.stderr,
            )

        self._earnings_cache: dict[str, dict] = {}

    def get_parameters(self) -> dict:
        return self.config.params

    def required_history_days(self) -> int:
        return self.config.params["confirmation_ema"] + 50

    def pre_fetch(self, symbols: list[str]) -> None:
        """Pre-fetch earnings data for all symbols."""
        if self._finnhub is None:
            return

        p = self.config.params
        now = datetime.now(timezone.utc)
        # Look back further to catch earnings from last week
        start = now - timedelta(days=p["lookback_days"] + 5)
        from_date = start.strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d")

        # Retry up to 3 times with increasing timeout
        calendar = None
        for attempt in range(3):
            try:
                calendar = self._finnhub.earnings_calendar(
                    _from=from_date,
                    to=to_date,
                    symbol="",  # all symbols
                )
                break  # success
            except Exception as e:
                print(
                    f"[EarningsMomentum] Attempt {attempt + 1}/3 failed: {e}",
                    file=sys.stderr,
                )
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))

        if not calendar or "earningsCalendar" not in calendar:
            print(
                "[EarningsMomentum] No earnings calendar data returned after retries",
                file=sys.stderr,
            )
            return

        # Index by symbol for quick lookup
        # Only keep entries that have actual results (not just estimates)
        symbols_set = set(symbols)
        for entry in calendar["earningsCalendar"]:
            symbol = entry.get("symbol", "")
            actual = entry.get("epsActual")
            estimate = entry.get("epsEstimate")

            if symbol not in symbols_set:
                continue
            if actual is None or estimate is None:
                continue  # no actual results yet

            # Calculate surprise percentage
            if estimate != 0:
                surprise_pct = ((actual - estimate) / abs(estimate)) * 100
            elif actual > 0:
                surprise_pct = 100.0  # beat zero estimate
            elif actual < 0:
                surprise_pct = -100.0
            else:
                surprise_pct = 0.0

            # Store the most recent earnings for each symbol
            report_date = entry.get("date", "")
            existing = self._earnings_cache.get(symbol)
            if existing is None or report_date > existing.get("date", ""):
                self._earnings_cache[symbol] = {
                    "date": report_date,
                    "actual": actual,
                    "estimate": estimate,
                    "surprise_pct": surprise_pct,
                    "revenue_actual": entry.get("revenueActual"),
                    "revenue_estimate": entry.get("revenueEstimate"),
                }

        cached = len(self._earnings_cache)
        print(
            f"[EarningsMomentum] Found {cached} symbols with recent earnings",
            file=sys.stderr,
        )

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        """Generate signals based on earnings surprises + price confirmation."""
        if self._finnhub is None:
            return []

        # If pre_fetch wasn't called, fetch now
        if not self._earnings_cache:
            self.pre_fetch(list(data.keys()))

        signals: list[Signal] = []
        p = self.config.params

        for symbol, df in data.items():
            if not self.validate_data(df):
                continue

            earnings = self._earnings_cache.get(symbol)
            if earnings is None:
                continue

            sig = self._analyze_symbol(symbol, df, earnings, p)
            if sig is not None:
                signals.append(sig)

        return signals

    def _analyze_symbol(
        self, symbol: str, df: pd.DataFrame, earnings: dict, p: dict
    ) -> Signal | None:
        """Analyze a single symbol for earnings momentum signal."""
        surprise_pct = earnings["surprise_pct"]
        report_date = earnings["date"]

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
        avg_vol = float(curr["avg_volume"]) if not pd.isna(curr["avg_volume"]) else 0
        if avg_vol < p["min_avg_volume"]:
            return None

        timestamp = df.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        atr = float(curr["atr"])

        # Try to find the earnings day in price data for confirmation
        earnings_day_return = None
        volume_surge = None
        try:
            # Find the bar on or closest to report_date
            report_dt = pd.Timestamp(report_date, tz="UTC")
            if report_dt in df.index:
                idx = df.index.get_loc(report_dt)
                if idx > 0:
                    prev_close = float(df.iloc[idx - 1]["close"])
                    earn_close = float(df.iloc[idx]["close"])
                    earn_volume = float(df.iloc[idx]["volume"])
                    earnings_day_return = (earn_close - prev_close) / prev_close * 100
                    volume_surge = earn_volume / avg_vol if avg_vol > 0 else 0
            else:
                # Try to find nearest bar after report date
                later_bars = df.index[df.index >= report_dt]
                if len(later_bars) > 0:
                    earn_idx = df.index.get_loc(later_bars[0])
                    if earn_idx > 0:
                        prev_close = float(df.iloc[earn_idx - 1]["close"])
                        earn_close = float(df.iloc[earn_idx]["close"])
                        earn_volume = float(df.iloc[earn_idx]["volume"])
                        earnings_day_return = (earn_close - prev_close) / prev_close * 100
                        volume_surge = earn_volume / avg_vol if avg_vol > 0 else 0
        except Exception:
            pass  # can't find earnings day, skip confirmation

        # Negative earnings surprise: SHORT if confirmed, CLOSE if not
        if surprise_pct < p["negative_surprise_pct"]:
            # Check for short-entry confirmations
            price_confirmed = (
                earnings_day_return is not None and earnings_day_return < 0
            )
            ema_confirmed = curr["close"] < curr["ema_confirm"]
            volume_ok = (
                volume_surge is None or volume_surge >= p["min_volume_surge"]
            )

            if price_confirmed and ema_confirmed and volume_ok:
                # SHORT entry — negative surprise with full confirmation
                stop_loss = float(curr["close"]) + p["atr_stop_multiplier"] * atr

                capped_surprise = min(abs(surprise_pct), p["max_surprise_for_scaling"])
                base_strength = capped_surprise / p["max_surprise_for_scaling"]

                vol_boost = 0.0
                if volume_surge is not None and volume_surge > 2.0:
                    vol_boost = min(0.2, (volume_surge - 2.0) * 0.05)

                strength = min(1.0, base_strength + vol_boost)

                # Revenue miss adds extra conviction
                rev_actual = earnings.get("revenue_actual")
                rev_estimate = earnings.get("revenue_estimate")
                revenue_miss = False
                if rev_actual and rev_estimate and rev_estimate > 0:
                    rev_surprise = (rev_actual - rev_estimate) / rev_estimate * 100
                    if rev_surprise < -2.0:  # revenue also missed by 2%+
                        strength = min(1.0, strength + 0.1)
                        revenue_miss = True

                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=SignalDirection.SHORT,
                    strength=strength,
                    strategy_name=self.name,
                    entry_price=float(curr["close"]),
                    stop_loss=stop_loss,
                    features={
                        "surprise_pct": surprise_pct,
                        "actual_eps": earnings["actual"],
                        "estimate_eps": earnings["estimate"],
                        "report_date": report_date,
                        "earnings_day_return": earnings_day_return,
                        "volume_surge": volume_surge,
                        "revenue_miss": revenue_miss,
                        "ema_confirm": float(curr["ema_confirm"]),
                        "atr": atr,
                        "close": float(curr["close"]),
                    },
                    rationale=(
                        f"Earnings miss SHORT: surprise={surprise_pct:+.1f}% "
                        f"(actual={earnings['actual']:.2f} vs est={earnings['estimate']:.2f}). "
                        f"Reported {report_date}. "
                        f"Gapped down {earnings_day_return:+.1f}%. "
                        f"{'Revenue also missed. ' if revenue_miss else ''}"
                        f"Price ${curr['close']:.2f} < EMA{p['confirmation_ema']} "
                        f"${curr['ema_confirm']:.2f}. "
                        f"Stop: ${stop_loss:.2f}."
                    ),
                )
            else:
                # CLOSE signal — not confirmed for short entry
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=SignalDirection.CLOSE,
                    strength=min(1.0, abs(surprise_pct) / 20.0),
                    strategy_name=self.name,
                    features={
                        "surprise_pct": surprise_pct,
                        "actual_eps": earnings["actual"],
                        "estimate_eps": earnings["estimate"],
                        "report_date": report_date,
                        "earnings_day_return": earnings_day_return,
                        "close": float(curr["close"]),
                    },
                    rationale=(
                        f"Earnings miss: surprise={surprise_pct:+.1f}% "
                        f"(actual={earnings['actual']:.2f} vs est={earnings['estimate']:.2f}). "
                        f"Reported {report_date}."
                    ),
                )

        # LONG signal: positive earnings surprise + confirmation
        if surprise_pct > p["surprise_threshold_pct"]:
            # Price confirmation: must have moved up post-earnings
            if p["require_price_confirmation"] and earnings_day_return is not None:
                if earnings_day_return < p["min_earnings_day_return"]:
                    return None

            # Volume confirmation: earnings day volume spike
            if volume_surge is not None and volume_surge < p["min_volume_surge"]:
                return None

            # EMA confirmation: price in uptrend context
            if curr["close"] < curr["ema_confirm"]:
                return None

            stop_loss = float(curr["close"]) - p["atr_stop_multiplier"] * atr

            # Strength: scaled by surprise magnitude
            capped_surprise = min(surprise_pct, p["max_surprise_for_scaling"])
            base_strength = capped_surprise / p["max_surprise_for_scaling"]

            # Bonus for volume surge (institutional conviction)
            vol_boost = 0.0
            if volume_surge is not None and volume_surge > 2.0:
                vol_boost = min(0.2, (volume_surge - 2.0) * 0.05)

            strength = min(1.0, base_strength + vol_boost)

            # Revenue beat adds extra confidence
            rev_actual = earnings.get("revenue_actual")
            rev_estimate = earnings.get("revenue_estimate")
            revenue_beat = False
            if rev_actual and rev_estimate and rev_estimate > 0:
                rev_surprise = (rev_actual - rev_estimate) / rev_estimate * 100
                if rev_surprise > 2.0:  # revenue also beat by 2%+
                    strength = min(1.0, strength + 0.1)
                    revenue_beat = True

            return Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=SignalDirection.LONG,
                strength=strength,
                strategy_name=self.name,
                entry_price=float(curr["close"]),
                stop_loss=stop_loss,
                features={
                    "surprise_pct": surprise_pct,
                    "actual_eps": earnings["actual"],
                    "estimate_eps": earnings["estimate"],
                    "report_date": report_date,
                    "earnings_day_return": earnings_day_return,
                    "volume_surge": volume_surge,
                    "revenue_beat": revenue_beat,
                    "ema_confirm": float(curr["ema_confirm"]),
                    "atr": atr,
                    "close": float(curr["close"]),
                },
                rationale=(
                    f"Earnings beat: surprise={surprise_pct:+.1f}% "
                    f"(actual={earnings['actual']:.2f} vs est={earnings['estimate']:.2f}). "
                    f"Reported {report_date}. "
                    f"{'Revenue also beat. ' if revenue_beat else ''}"
                    f"Price ${curr['close']:.2f} > EMA{p['confirmation_ema']} "
                    f"${curr['ema_confirm']:.2f}. "
                    f"Stop: ${stop_loss:.2f}."
                ),
            )

        return None
