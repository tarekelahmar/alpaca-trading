#!/usr/bin/env python3
"""One-time script to place stop-loss orders for existing positions.

For each position, places a GTC stop order at entry_price - (8% * entry_price)
as a hard floor. The price monitor provides tighter trailing stops on top.

Usage:
    python scripts/place_stops.py [--dry-run]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus


HARD_STOP_PCT = 0.08  # 8% max loss from entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    tc = TradingClient(key, secret, paper=True)

    positions = tc.get_all_positions()
    print(f"Found {len(positions)} positions\n")

    for pos in positions:
        symbol = pos.symbol
        qty = float(pos.qty)
        entry = float(pos.avg_entry_price)
        current = float(pos.current_price)
        stop_price = round(entry * (1 - HARD_STOP_PCT), 2)

        print(f"{symbol}: {qty} shares @ ${entry:.2f} (now ${current:.2f})")
        print(f"  Stop-loss price: ${stop_price:.2f} (-{HARD_STOP_PCT:.0%} from entry)")

        # Check for existing stop orders
        existing = tc.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
        ))
        has_stop = any(o.type.value == "stop" for o in existing)
        if has_stop:
            print(f"  Already has a stop order — skipping")
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would place stop sell {int(qty)} @ ${stop_price}")
        else:
            try:
                order = StopOrderRequest(
                    symbol=symbol,
                    qty=int(qty),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_price,
                )
                result = tc.submit_order(order_data=order)
                print(f"  ✓ Stop placed: sell {int(qty)} @ ${stop_price} (id={result.id})")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        print()


if __name__ == "__main__":
    main()
