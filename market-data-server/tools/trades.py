import json
import sys

from alpaca.data.requests import StockLatestTradeRequest

from app import mcp
from utils.client import get_data_client


@mcp.tool()
async def get_latest_trade(symbols: str) -> str:
    """Get the last trade for one or more stocks.

    Args:
        symbols: Comma-separated stock symbols (e.g. "AAPL,MSFT")
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        client = get_data_client()
        params = StockLatestTradeRequest(symbol_or_symbols=symbol_list)
        trades = client.get_stock_latest_trade(params)

        result = {}
        for sym, trade in trades.items():
            result[sym] = {
                "price": float(trade.price),
                "size": int(trade.size),
                "timestamp": str(trade.timestamp),
                "exchange": trade.exchange,
            }

        return json.dumps(result, indent=2)
    except Exception as e:
        print(f"Error in get_latest_trade: {e}", file=sys.stderr)
        return f"Error fetching trades for {symbols}: {e}"
