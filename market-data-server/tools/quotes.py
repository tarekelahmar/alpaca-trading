import json
import sys

from alpaca.data.requests import StockLatestQuoteRequest

from app import mcp
from utils.client import get_data_client


@mcp.tool()
async def get_latest_quote(symbols: str) -> str:
    """Get the latest bid/ask quote for one or more stocks.

    Args:
        symbols: Comma-separated stock symbols (e.g. "AAPL,MSFT,GOOG")
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        client = get_data_client()
        params = StockLatestQuoteRequest(symbol_or_symbols=symbol_list)
        quotes = client.get_stock_latest_quote(params)

        result = {}
        for sym, quote in quotes.items():
            result[sym] = {
                "bid_price": float(quote.bid_price),
                "ask_price": float(quote.ask_price),
                "bid_size": int(quote.bid_size),
                "ask_size": int(quote.ask_size),
                "timestamp": str(quote.timestamp),
            }

        return json.dumps(result, indent=2)
    except Exception as e:
        print(f"Error in get_latest_quote: {e}", file=sys.stderr)
        return f"Error fetching quotes for {symbols}: {e}"
