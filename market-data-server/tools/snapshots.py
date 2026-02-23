import json
import sys

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockSnapshotRequest

from app import mcp
from utils.client import get_data_client


@mcp.tool()
async def get_snapshot(symbols: str) -> str:
    """Get a comprehensive snapshot for one or more stocks including latest trade, quote, minute bar, daily bar, and previous daily bar.

    Args:
        symbols: Comma-separated stock symbols (e.g. "AAPL,TSLA")
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        client = get_data_client()
        params = StockSnapshotRequest(symbol_or_symbols=symbol_list, feed=DataFeed.IEX)
        snapshots = client.get_stock_snapshot(params)

        result = {}
        for sym, snap in snapshots.items():
            entry = {}

            if snap.latest_trade:
                entry["latest_trade"] = {
                    "price": float(snap.latest_trade.price),
                    "size": int(snap.latest_trade.size),
                    "timestamp": str(snap.latest_trade.timestamp),
                }

            if snap.latest_quote:
                entry["latest_quote"] = {
                    "bid_price": float(snap.latest_quote.bid_price),
                    "ask_price": float(snap.latest_quote.ask_price),
                    "bid_size": int(snap.latest_quote.bid_size),
                    "ask_size": int(snap.latest_quote.ask_size),
                }

            if snap.minute_bar:
                entry["minute_bar"] = {
                    "open": float(snap.minute_bar.open),
                    "high": float(snap.minute_bar.high),
                    "low": float(snap.minute_bar.low),
                    "close": float(snap.minute_bar.close),
                    "volume": int(snap.minute_bar.volume),
                    "timestamp": str(snap.minute_bar.timestamp),
                }

            if snap.daily_bar:
                entry["daily_bar"] = {
                    "open": float(snap.daily_bar.open),
                    "high": float(snap.daily_bar.high),
                    "low": float(snap.daily_bar.low),
                    "close": float(snap.daily_bar.close),
                    "volume": int(snap.daily_bar.volume),
                    "timestamp": str(snap.daily_bar.timestamp),
                }

            if snap.previous_daily_bar:
                entry["previous_daily_bar"] = {
                    "open": float(snap.previous_daily_bar.open),
                    "high": float(snap.previous_daily_bar.high),
                    "low": float(snap.previous_daily_bar.low),
                    "close": float(snap.previous_daily_bar.close),
                    "volume": int(snap.previous_daily_bar.volume),
                    "timestamp": str(snap.previous_daily_bar.timestamp),
                }

            result[sym] = entry

        return json.dumps(result, indent=2)
    except Exception as e:
        print(f"Error in get_snapshot: {e}", file=sys.stderr)
        return f"Error fetching snapshots for {symbols}: {e}"
