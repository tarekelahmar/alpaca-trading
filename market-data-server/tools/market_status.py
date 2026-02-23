import json
import sys

from app import mcp
from utils.client import get_trading_client


@mcp.tool()
async def get_market_status() -> str:
    """Check if the US stock market is currently open, and when it opens/closes next."""
    try:
        client = get_trading_client()
        clock = client.get_clock()

        result = {
            "is_open": clock.is_open,
            "timestamp": str(clock.timestamp),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        print(f"Error in get_market_status: {e}", file=sys.stderr)
        return f"Error fetching market status: {e}"
