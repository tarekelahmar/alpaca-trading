"""Stock Universe Selection and Filtering.

Defines the tradeable universe and applies filters:
    - S&P 500 constituents as base
    - Minimum average daily dollar volume
    - Minimum price
    - Exclude stocks with upcoming earnings (optional)
    - Exclude penny stocks and illiquid names
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class UniverseFilter:
    min_price: float = 10.0
    min_avg_dollar_volume: float = 5_000_000.0
    volume_lookback: int = 20
    exclude_symbols: list[str] | None = None


# S&P 500 tickers — a representative subset for initial implementation.
# In production, this would be fetched from an API or maintained list.
# Using top ~100 most liquid S&P 500 names for faster backtesting.
SP500_CORE = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AFL",
    "AIG", "AMAT", "AMD", "AMGN", "AMZN", "ANET", "AVGO", "AXP", "BA", "BAC",
    "BDX", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CDNS",
    "CI", "CL", "CMCSA", "CME", "COF", "COP", "COST", "CRM", "CSCO", "CTAS",
    "CVS", "CVX", "D", "DE", "DHR", "DIS", "DUK", "ECL", "EL", "EMR",
    "EW", "EXC", "F", "FDX", "FISV", "GD", "GE", "GILD", "GM", "GOOG",
    "GOOGL", "GS", "HD", "HON", "IBM", "ICE", "INTC", "INTU", "ISRG", "ITW",
    "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MCHP",
    "MCK", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "MU", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG",
    "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SHW", "SNPS", "SO", "SPG",
    "T", "TGT", "TMO", "TMUS", "TRV", "TSLA", "TXN", "UNH", "UNP", "UPS",
    "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM", "ZTS",
]


class UniverseSelector:

    def __init__(
        self,
        base_symbols: list[str] | None = None,
        filters: UniverseFilter | None = None,
    ):
        self.base_symbols = base_symbols or SP500_CORE
        self.filters = filters or UniverseFilter()

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
                df["close"].tail(f.volume_lookback) * df["volume"].tail(f.volume_lookback)
            ).mean()
            if avg_dollar_vol < f.min_avg_dollar_volume:
                continue

            filtered[symbol] = df

        return filtered
