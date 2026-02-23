import json
import sys
from datetime import datetime

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from app import mcp
from utils.client import get_data_client

TIMEFRAME_MAP = {
    "1Min": TimeFrame.Minute,
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame.Hour,
    "1Day": TimeFrame.Day,
    "1Week": TimeFrame.Week,
    "1Month": TimeFrame.Month,
}


@mcp.tool()
async def get_stock_bars(
    symbol: str,
    timeframe: str,
    start: str,
    end: str | None = None,
) -> str:
    """Get historical OHLCV bar data for a stock.

    Args:
        symbol: Stock ticker symbol (e.g. AAPL, MSFT)
        timeframe: Bar timeframe - one of: 1Min, 5Min, 15Min, 1Hour, 1Day, 1Week, 1Month
        start: Start date/time in ISO 8601 format (e.g. 2024-01-01)
        end: Optional end date/time in ISO 8601 format. Defaults to now.
    """
    try:
        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            return f"Error: Invalid timeframe '{timeframe}'. Valid options: {list(TIMEFRAME_MAP.keys())}"

        client = get_data_client()
        params = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=tf,
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end) if end else None,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(params)

        sym = symbol.upper()
        if sym not in bars.data or not bars[sym]:
            return f"No bar data found for {sym} in the given range."

        result = []
        for bar in bars[sym]:
            result.append({
                "timestamp": str(bar.timestamp),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
                "vwap": float(bar.vwap) if bar.vwap else None,
                "trade_count": int(bar.trade_count) if bar.trade_count else None,
            })

        return json.dumps(result, indent=2)
    except Exception as e:
        print(f"Error in get_stock_bars: {e}", file=sys.stderr)
        return f"Error fetching bars for {symbol}: {e}"
