import json
import sys

from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from app import mcp
from utils.client import get_trading_client


@mcp.tool()
async def search_assets(query: str) -> str:
    """Search for tradeable stock assets by name or symbol.

    Args:
        query: Search query — partial symbol or company name (e.g. "AAPL" or "Tesla")
    """
    try:
        client = get_trading_client()
        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        all_assets = client.get_all_assets(request)

        query_upper = query.upper()
        matches = []
        for asset in all_assets:
            if (
                query_upper in asset.symbol.upper()
                or (asset.name and query_upper in asset.name.upper())
            ):
                matches.append({
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "exchange": asset.exchange.value if asset.exchange else None,
                    "tradable": asset.tradable,
                    "fractionable": asset.fractionable,
                    "shortable": asset.shortable,
                })
                if len(matches) >= 20:
                    break

        if not matches:
            return f"No tradeable assets found matching '{query}'."

        return json.dumps(matches, indent=2)
    except Exception as e:
        print(f"Error in search_assets: {e}", file=sys.stderr)
        return f"Error searching assets for '{query}': {e}"
