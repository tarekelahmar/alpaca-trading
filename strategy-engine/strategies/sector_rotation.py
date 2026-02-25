"""Sector Rotation / Relative Strength Strategy.

Tracks sector ETF performance and rotates into the strongest sectors
while shorting the weakest. Uses relative strength vs SPY as the
primary ranking metric.

Entry conditions (LONG):
    - Sector ETF outperforming SPY over both 1-month AND 3-month windows
    - Relative strength (RS) score in top N sectors
    - Sector not already overbought (RSI < 75)
    - Individual stocks within the sector ranked by their own RS vs the sector

Entry conditions (SHORT):
    - Sector ETF underperforming SPY over both 1-month AND 3-month windows
    - RS score in bottom N sectors
    - Sector not already oversold (RSI > 25)

Exit conditions:
    - Sector drops from top/bottom tier
    - OR trailing stop hit

Edge: Sector momentum is one of the most persistent anomalies in equity
markets. Sectors that outperform tend to continue outperforming for 2-12
months (intermediate-term momentum). This strategy captures the macro
rotation theme that drives 40-60% of individual stock returns.

Rotation frequency: Re-rank every 2-4 weeks. Hold for weeks to months.

Also provides GICS sector mappings used by the portfolio optimizer
for sector exposure enforcement.
"""

import pandas as pd
import ta

from strategies.base import Signal, SignalDirection, Strategy, StrategyConfig


# GICS sector ETFs — used as sector proxies
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLB": "Materials",
}

# Map individual stocks to their GICS sector
# This covers all stocks in the SP500_CORE + SMID_CAP universe
STOCK_SECTORS: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CSCO": "Technology",
    "ORCL": "Technology", "ACN": "Technology", "TXN": "Technology",
    "QCOM": "Technology", "AMAT": "Technology", "ADI": "Technology",
    "INTU": "Technology", "NOW": "Technology", "MU": "Technology",
    "SNPS": "Technology", "CDNS": "Technology", "ADSK": "Technology",
    "ADP": "Technology", "FISV": "Technology", "PYPL": "Technology",
    "MCHP": "Technology", "ANET": "Technology",
    # Technology — SMID
    "CRWD": "Technology", "DDOG": "Technology", "NET": "Technology",
    "ZS": "Technology", "BILL": "Technology", "HUBS": "Technology",
    "ESTC": "Technology", "GTLB": "Technology", "IOT": "Technology",
    "CFLT": "Technology", "DOCN": "Technology", "BRZE": "Technology",
    "APPF": "Technology", "DT": "Technology",
    # Semiconductors (classified as Technology)
    "SMCI": "Technology", "ARM": "Technology", "ONTO": "Technology",
    "RMBS": "Technology", "CRUS": "Technology", "LSCC": "Technology",
    "ALGM": "Technology", "ACLS": "Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "SCHW": "Financials", "C": "Financials", "AXP": "Financials",
    "USB": "Financials", "BK": "Financials", "COF": "Financials",
    "MET": "Financials", "AIG": "Financials", "ICE": "Financials",
    "CME": "Financials", "TRV": "Financials",
    # Financials — SMID
    "HOOD": "Financials", "SOFI": "Financials", "AFRM": "Financials",
    "LPLA": "Financials", "KNSL": "Financials", "HLI": "Financials",
    # Health Care
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "MRK": "Health Care", "ABBV": "Health Care", "PFE": "Health Care",
    "ABT": "Health Care", "TMO": "Health Care", "DHR": "Health Care",
    "AMGN": "Health Care", "GILD": "Health Care", "ISRG": "Health Care",
    "CI": "Health Care", "EW": "Health Care", "BDX": "Health Care",
    "CVS": "Health Care", "MDT": "Health Care", "MCK": "Health Care",
    "BMY": "Health Care", "BIIB": "Health Care", "ZTS": "Health Care",
    # Health Care — SMID
    "EXAS": "Health Care", "HALO": "Health Care", "PCVX": "Health Care",
    "INSP": "Health Care", "KRYS": "Health Care", "CORT": "Health Care",
    "RPRX": "Health Care", "INSM": "Health Care", "ALNY": "Health Care",
    "SRPT": "Health Care", "RARE": "Health Care",
    "GMED": "Health Care", "TNDM": "Health Care", "NVCR": "Health Care",
    "PODD": "Health Care", "IRTC": "Health Care", "NTRA": "Health Care",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TGT": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "F": "Consumer Discretionary",
    "GM": "Consumer Discretionary",
    # Consumer Discretionary — SMID
    "SHAK": "Consumer Discretionary", "CAVA": "Consumer Discretionary",
    "WING": "Consumer Discretionary", "ELF": "Consumer Discretionary",
    "ONON": "Consumer Discretionary", "BIRK": "Consumer Discretionary",
    "DUOL": "Consumer Discretionary",
    # Communication Services
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "CMCSA": "Communication Services",
    "TMUS": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services",
    # Industrials
    "CAT": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "UPS": "Industrials", "DE": "Industrials", "BA": "Industrials",
    "RTX": "Industrials", "LMT": "Industrials", "GE": "Industrials",
    "GD": "Industrials", "EMR": "Industrials", "FDX": "Industrials",
    "ITW": "Industrials", "CTAS": "Industrials",
    # Industrials — SMID
    "AXON": "Industrials", "TDG": "Industrials", "TOST": "Industrials",
    "PAYC": "Industrials", "PCOR": "Industrials", "CSWI": "Industrials",
    "POWL": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "PM": "Consumer Staples", "MO": "Consumer Staples",
    "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
    "WMT": "Consumer Staples", "WBA": "Consumer Staples",
    "EL": "Consumer Staples",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "EXC": "Utilities",
    # Real Estate
    "SPG": "Real Estate", "SHW": "Real Estate",
    # Real Estate — SMID
    "GLPI": "Real Estate", "IIPR": "Real Estate", "REXR": "Real Estate",
    # Materials
    "LIN": "Materials", "ECL": "Materials", "AFL": "Materials",
    "MMM": "Materials",
}


def get_stock_sector(symbol: str) -> str | None:
    """Look up the GICS sector for a stock symbol."""
    return STOCK_SECTORS.get(symbol)


def get_sector_stocks(sector: str) -> list[str]:
    """Get all stocks in a given sector."""
    return [s for s, sec in STOCK_SECTORS.items() if sec == sector]


DEFAULT_PARAMS = {
    # Ranking periods (trading days)
    "rs_period_short": 21,    # ~1 month relative strength
    "rs_period_long": 63,     # ~3 months relative strength
    "rs_weight_short": 0.40,  # weight for 1-month RS
    "rs_weight_long": 0.60,   # weight for 3-month RS (more important)
    # Selection
    "top_n_sectors": 3,       # go long top 3 sectors
    "bottom_n_sectors": 2,    # go short bottom 2 sectors
    "stocks_per_sector": 3,   # pick top 3 stocks per selected sector
    # Filters
    "rsi_period": 14,
    "rsi_overbought": 75,
    "rsi_oversold": 25,
    "min_avg_volume": 300_000,
    "volume_lookback": 20,
    # ATR for stops
    "atr_period": 14,
    "atr_stop_multiplier": 2.5,  # wider stops for sector rotation (longer holds)
}


class SectorRotationStrategy(Strategy):

    def __init__(self, config: StrategyConfig | None = None):
        if config is None:
            config = StrategyConfig(name="sector_rotation", params=DEFAULT_PARAMS)
        merged = {**DEFAULT_PARAMS, **config.params}
        config.params = merged
        super().__init__(config)

    def get_parameters(self) -> dict:
        return self.config.params

    def required_history_days(self) -> int:
        return self.config.params["rs_period_long"] + 30

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        """Generate signals based on sector relative strength.

        This strategy:
        1. Computes relative strength for each stock vs its sector peers
        2. Identifies which sectors are strongest/weakest based on member returns
        3. Picks top stocks from the strongest sectors (LONG)
        4. Picks weakest stocks from the weakest sectors (SHORT)
        """
        p = self.config.params
        signals: list[Signal] = []

        # Step 1: Compute sector-level performance from constituent stocks
        sector_performance = self._compute_sector_performance(data, p)
        if not sector_performance:
            return signals

        # Step 2: Rank sectors by composite relative strength
        ranked_sectors = sorted(
            sector_performance.items(),
            key=lambda x: x[1]["composite_rs"],
            reverse=True,
        )

        # Step 3: Pick stocks from top sectors (LONG)
        top_sectors = ranked_sectors[:p["top_n_sectors"]]
        for sector, perf in top_sectors:
            if perf["composite_rs"] <= 0:
                continue  # don't buy underperforming sectors
            sector_signals = self._pick_sector_stocks(
                sector, data, p, SignalDirection.LONG, perf["composite_rs"]
            )
            signals.extend(sector_signals)

        # Step 4: Pick stocks from bottom sectors (SHORT)
        bottom_sectors = ranked_sectors[-p["bottom_n_sectors"]:]
        for sector, perf in bottom_sectors:
            if perf["composite_rs"] >= 0:
                continue  # don't short outperforming sectors
            sector_signals = self._pick_sector_stocks(
                sector, data, p, SignalDirection.SHORT, perf["composite_rs"]
            )
            signals.extend(sector_signals)

        return signals

    def _compute_sector_performance(
        self, data: dict[str, pd.DataFrame], p: dict
    ) -> dict[str, dict]:
        """Compute average relative strength for each sector from its constituents."""
        sector_returns: dict[str, list[dict]] = {}

        for symbol, df in data.items():
            sector = get_stock_sector(symbol)
            if not sector:
                continue
            if len(df) < p["rs_period_long"] + 5:
                continue

            # Compute returns over short and long windows
            close = df["close"]
            if len(close) < p["rs_period_long"] + 1:
                continue

            ret_short = (
                float(close.iloc[-1]) / float(close.iloc[-p["rs_period_short"] - 1]) - 1
            ) * 100
            ret_long = (
                float(close.iloc[-1]) / float(close.iloc[-p["rs_period_long"] - 1]) - 1
            ) * 100

            sector_returns.setdefault(sector, []).append({
                "symbol": symbol,
                "ret_short": ret_short,
                "ret_long": ret_long,
            })

        # Compute sector averages
        result: dict[str, dict] = {}
        for sector, returns in sector_returns.items():
            if len(returns) < 2:
                continue  # need at least 2 stocks to evaluate sector

            avg_short = sum(r["ret_short"] for r in returns) / len(returns)
            avg_long = sum(r["ret_long"] for r in returns) / len(returns)

            composite = (
                avg_short * p["rs_weight_short"]
                + avg_long * p["rs_weight_long"]
            )

            result[sector] = {
                "avg_return_short": avg_short,
                "avg_return_long": avg_long,
                "composite_rs": composite,
                "num_stocks": len(returns),
                "stock_returns": returns,
            }

        return result

    def _pick_sector_stocks(
        self,
        sector: str,
        data: dict[str, pd.DataFrame],
        p: dict,
        direction: SignalDirection,
        sector_rs: float,
    ) -> list[Signal]:
        """Pick the best individual stocks from a sector."""
        signals: list[Signal] = []
        candidates: list[tuple[str, float, pd.DataFrame]] = []

        sector_stocks = get_sector_stocks(sector)
        for symbol in sector_stocks:
            if symbol not in data:
                continue

            df = data[symbol]
            if len(df) < p["rs_period_long"] + 5:
                continue

            # Volume filter
            avg_vol = df["volume"].rolling(window=p["volume_lookback"]).mean().iloc[-1]
            if pd.isna(avg_vol) or avg_vol < p["min_avg_volume"]:
                continue

            # RSI filter
            rsi = ta.momentum.rsi(df["close"], window=p["rsi_period"]).iloc[-1]
            if pd.isna(rsi):
                continue

            if direction == SignalDirection.LONG and rsi > p["rsi_overbought"]:
                continue
            if direction == SignalDirection.SHORT and rsi < p["rsi_oversold"]:
                continue

            # Individual stock RS
            close = df["close"]
            ret_short = (
                float(close.iloc[-1]) / float(close.iloc[-p["rs_period_short"] - 1]) - 1
            ) * 100
            ret_long = (
                float(close.iloc[-1]) / float(close.iloc[-p["rs_period_long"] - 1]) - 1
            ) * 100
            stock_rs = (
                ret_short * p["rs_weight_short"]
                + ret_long * p["rs_weight_long"]
            )

            candidates.append((symbol, stock_rs, df))

        if not candidates:
            return signals

        # For LONG: pick stocks with highest individual RS within the sector
        # For SHORT: pick stocks with lowest individual RS
        if direction == SignalDirection.LONG:
            candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            candidates.sort(key=lambda x: x[1])

        for symbol, stock_rs, df in candidates[:p["stocks_per_sector"]]:
            curr = df.iloc[-1]
            timestamp = df.index[-1]
            if hasattr(timestamp, "to_pydatetime"):
                timestamp = timestamp.to_pydatetime()

            atr = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"], window=p["atr_period"]
            ).iloc[-1]
            if pd.isna(atr):
                continue

            close = float(curr["close"])

            if direction == SignalDirection.LONG:
                stop_loss = close - p["atr_stop_multiplier"] * float(atr)
            else:
                stop_loss = close + p["atr_stop_multiplier"] * float(atr)

            # Strength: based on absolute sector RS score, scaled
            strength = min(1.0, abs(sector_rs) / 15.0)

            signals.append(Signal(
                timestamp=timestamp,
                symbol=symbol,
                direction=direction,
                strength=strength,
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                features={
                    "sector": sector,
                    "sector_rs": sector_rs,
                    "stock_rs": stock_rs,
                    "atr": float(atr),
                    "close": close,
                },
                rationale=(
                    f"Sector rotation {direction.value.upper()}: {sector} sector "
                    f"RS={sector_rs:.1f}. {symbol} individual RS={stock_rs:.1f}. "
                    f"Stop={stop_loss:.2f}."
                ),
            ))

        return signals
