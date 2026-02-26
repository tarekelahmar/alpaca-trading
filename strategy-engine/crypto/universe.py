"""Crypto Universe for Alpaca Trading.

Defines the tradeable crypto universe on Alpaca, grouped by tier:
    - Tier 1 (Blue Chip): BTC, ETH — highest liquidity, lowest spreads
    - Tier 2 (Major Alt): Large-cap altcoins with good Alpaca liquidity
    - Tier 3 (Mid Alt): Smaller altcoins, wider spreads, more volatile

Alpaca uses the /USD suffix for crypto pairs (e.g., BTC/USD).
All symbols here use the Alpaca format.
"""

from dataclasses import dataclass

import pandas as pd


# Tier 1: Blue Chips — always trade, tightest spreads
TIER1_BLUE_CHIP = [
    "BTC/USD",
    "ETH/USD",
]

# Tier 2: Major Altcoins — available on Alpaca
# Note: Alpaca crypto volume is very thin vs Binance/Coinbase
TIER2_MAJOR_ALT = [
    "SOL/USD",
    "AVAX/USD",
    "LINK/USD",
    "DOT/USD",
    "UNI/USD",
    "AAVE/USD",
    "LTC/USD",
]

# Tier 3: Mid-cap Altcoins — tradeable on Alpaca but thinner
# Removed: MATIC/USD, XLM/USD, ALGO/USD, ATOM/USD, MKR/USD (no Alpaca data)
TIER3_MID_ALT = [
    "DOGE/USD",
    "SHIB/USD",
    "BCH/USD",
    "FIL/USD",
    "GRT/USD",
    "CRV/USD",
    "SUSHI/USD",
    "BAT/USD",
    "XTZ/USD",
]

# Combined universe by tier
CRYPTO_UNIVERSE = TIER1_BLUE_CHIP + TIER2_MAJOR_ALT + TIER3_MID_ALT


@dataclass
class CryptoUniverseFilter:
    """Filters for crypto asset selection."""

    min_price: float = 0.0001  # crypto can be very cheap (SHIB)
    min_avg_dollar_volume: float = 100.0  # $100/hour — Alpaca crypto volume is very thin vs CEXs
    volume_lookback: int = 336  # 14 days of hourly bars (24*14)
    exclude_symbols: list[str] | None = None
    tier_filter: int | None = None  # 1=blue chip only, 2=major+blue, None=all


class CryptoUniverseSelector:
    """Selects and filters the crypto trading universe.

    Crypto-specific considerations:
    - No minimum price floor (many legit coins trade < $1)
    - Dollar volume is the key liquidity metric
    - Tier-based filtering for risk management
    """

    def __init__(
        self,
        base_symbols: list[str] | None = None,
        filters: CryptoUniverseFilter | None = None,
    ):
        self.filters = filters or CryptoUniverseFilter()
        if base_symbols is not None:
            self.base_symbols = base_symbols
        elif self.filters.tier_filter == 1:
            self.base_symbols = TIER1_BLUE_CHIP
        elif self.filters.tier_filter == 2:
            self.base_symbols = TIER1_BLUE_CHIP + TIER2_MAJOR_ALT
        else:
            self.base_symbols = list(CRYPTO_UNIVERSE)

    def get_symbols(self) -> list[str]:
        """Return the base symbol list (before data-based filtering)."""
        symbols = list(self.base_symbols)
        if self.filters.exclude_symbols:
            exclude = set(self.filters.exclude_symbols)
            symbols = [s for s in symbols if s not in exclude]
        return symbols

    def filter_by_data(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Filter symbols based on price and volume data.

        Args:
            data: Dict mapping symbol -> OHLCV DataFrame.

        Returns:
            Filtered dict with only symbols passing all criteria.
        """
        f = self.filters
        filtered: dict[str, pd.DataFrame] = {}

        for symbol, df in data.items():
            if len(df) < f.volume_lookback:
                continue

            current_price = float(df.iloc[-1]["close"])
            if current_price < f.min_price:
                continue

            avg_dollar_vol = (
                df["close"].tail(f.volume_lookback)
                * df["volume"].tail(f.volume_lookback)
            ).mean()
            if avg_dollar_vol < f.min_avg_dollar_volume:
                continue

            filtered[symbol] = df

        return filtered

    def get_tier(self, symbol: str) -> int:
        """Return the tier (1, 2, or 3) for a given symbol."""
        if symbol in TIER1_BLUE_CHIP:
            return 1
        elif symbol in TIER2_MAJOR_ALT:
            return 2
        return 3

    def is_blue_chip(self, symbol: str) -> bool:
        """Check if a symbol is a blue-chip crypto."""
        return symbol in TIER1_BLUE_CHIP
