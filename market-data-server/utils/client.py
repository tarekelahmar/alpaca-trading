import os
import sys

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient


def _get_credentials() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print(
            "FATAL: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set",
            file=sys.stderr,
        )
        sys.exit(1)
    return key, secret


_data_client: StockHistoricalDataClient | None = None
_trading_client: TradingClient | None = None


def get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        key, secret = _get_credentials()
        _data_client = StockHistoricalDataClient(key, secret)
    return _data_client


def get_trading_client() -> TradingClient:
    global _trading_client
    if _trading_client is None:
        key, secret = _get_credentials()
        paper = os.environ.get("PAPER_TRADING", "true").lower() == "true"
        _trading_client = TradingClient(key, secret, paper=paper)
    return _trading_client
