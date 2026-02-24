#!/usr/bin/env python3
"""Intraday Scanner — catches intraday opportunities on 15-min bars.

Runs every 30 minutes during market hours (9:45 AM - 3:30 PM ET).
Separate from the daily swing system — uses smaller positions and
exits everything before close.

Strategies:
    1. Dip Buy: Stock drops 2%+ from open, RSI(14) < 40 on 15-min,
       then reclaims VWAP. Quick mean reversion trade.
    2. Breakout: Stock breaks above morning high on 2x average volume.
       Momentum continuation trade.

Risk rules:
    - Max 5% of equity per intraday trade
    - Max 8 concurrent intraday positions
    - Auto-exit all positions at 3:45 PM ET
    - Hard stop at 1.5% loss per trade
    - Take profit at 4%

Usage:
    python scripts/intraday_scanner.py [--paper] [--dry-run]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy-engine"))

import pandas as pd
import ta
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Top 30 most liquid large caps for intraday scanning
INTRADAY_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AMD",
    "NFLX", "AVGO", "CRM", "ORCL", "ADBE", "COST", "PEP",
    "JPM", "BAC", "WFC", "GS", "MS",
    "XOM", "CVX", "UNH", "JNJ", "V",
    "MA", "HD", "CAT", "BA", "DIS",
]

# Intraday strategy parameters
PARAMS = {
    # Dip buy
    "dip_threshold_pct": -2.0,     # min drop from open to trigger (was -3%)
    "dip_rsi_period": 14,
    "dip_rsi_oversold": 40,
    "dip_vwap_reclaim": True,      # price must be back above VWAP

    # Breakout
    "breakout_volume_mult": 2.0,   # current bar volume vs avg
    "breakout_lookback_bars": 8,   # morning session bars to find high

    # Risk
    "max_position_pct": 0.05,      # 5% of equity per trade (was 2%)
    "max_intraday_positions": 8,
    "stop_loss_pct": 0.015,        # 1.5% hard stop
    "take_profit_pct": 0.04,       # 4% take profit (was 2.5%)
    "eod_exit_minute": 945,        # 3:45 PM ET = 15:45 = 945 min from midnight
}


def get_clients(paper: bool):
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        print("ERROR: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set", file=sys.stderr)
        sys.exit(1)
    data_client = StockHistoricalDataClient(key, secret)
    trading_client = TradingClient(key, secret, paper=paper)
    return data_client, trading_client


def is_market_open(trading_client: TradingClient) -> bool:
    clock = trading_client.get_clock()
    return clock.is_open


def is_near_close(trading_client: TradingClient) -> bool:
    """Check if we're within 15 minutes of market close."""
    clock = trading_client.get_clock()
    if clock.next_close:
        minutes_to_close = (clock.next_close - clock.timestamp).total_seconds() / 60
        return minutes_to_close <= 15
    return False


def fetch_intraday_bars(
    data_client: StockHistoricalDataClient, symbols: list[str]
) -> dict[str, pd.DataFrame]:
    """Fetch today's 15-min bars for all symbols."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
    # Go back to start of today's session (14:30 UTC for 9:30 AM ET)
    start = today.replace(hour=13, minute=30)

    data = {}
    # Batch in groups of 10 to avoid overloading
    for i in range(0, len(symbols), 10):
        batch = symbols[i:i + 10]
        try:
            params = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=start,
                feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(params)
            for symbol in batch:
                if symbol in bars.data and bars[symbol]:
                    records = []
                    for bar in bars[symbol]:
                        records.append({
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": int(bar.volume),
                            "vwap": float(bar.vwap) if bar.vwap else None,
                        })
                    df = pd.DataFrame(records, index=[b.timestamp for b in bars[symbol]])
                    df.index.name = "timestamp"
                    if len(df) >= 2:  # need at least 2 bars
                        data[symbol] = df
        except Exception as e:
            print(f"Error fetching intraday data for batch: {e}", file=sys.stderr)

    return data


def scan_dip_buy(symbol: str, df: pd.DataFrame, p: dict) -> dict | None:
    """Look for dip-buy setup: big drop from open + RSI bounce + VWAP reclaim."""
    if len(df) < 4:  # need some bars to compute RSI
        return None

    open_price = df.iloc[0]["open"]
    curr = df.iloc[-1]
    change_from_open = (curr["close"] - open_price) / open_price * 100

    # Must have dropped enough from open
    if change_from_open > p["dip_threshold_pct"]:
        return None

    # RSI on 15-min bars
    rsi_series = ta.momentum.rsi(df["close"], window=min(p["dip_rsi_period"], len(df) - 1))
    rsi = rsi_series.iloc[-1]
    if pd.isna(rsi):
        return None

    # RSI should be oversold OR recovering from oversold
    prev_rsi = rsi_series.iloc[-2] if len(rsi_series) > 1 else rsi
    rsi_bouncing = rsi > prev_rsi and (rsi < 50)  # recovering but not yet overbought

    if not (rsi < p["dip_rsi_oversold"] or rsi_bouncing):
        return None

    # VWAP reclaim check
    if p["dip_vwap_reclaim"] and curr["vwap"] is not None:
        if curr["close"] < curr["vwap"]:
            return None  # not yet reclaimed VWAP

    return {
        "symbol": symbol,
        "strategy": "intraday_dip_buy",
        "side": "buy",
        "price": float(curr["close"]),
        "change_from_open": round(change_from_open, 2),
        "rsi": round(float(rsi), 1),
        "vwap": float(curr["vwap"]) if curr["vwap"] else None,
        "rationale": (
            f"Dip buy: {symbol} down {change_from_open:.1f}% from open, "
            f"RSI={rsi:.1f} bouncing, "
            f"{'above VWAP' if curr['vwap'] and curr['close'] >= curr['vwap'] else 'near VWAP'}"
        ),
    }


def scan_breakout(symbol: str, df: pd.DataFrame, p: dict) -> dict | None:
    """Look for intraday breakout: price above morning high on strong volume."""
    if len(df) < p["breakout_lookback_bars"]:
        return None

    # Morning high (first N bars)
    morning = df.iloc[:p["breakout_lookback_bars"]]
    morning_high = morning["high"].max()

    curr = df.iloc[-1]

    # Price must be breaking above morning high
    if curr["close"] <= morning_high:
        return None

    # Volume must be above average
    avg_volume = df["volume"].mean()
    if avg_volume <= 0:
        return None

    volume_ratio = curr["volume"] / avg_volume
    if volume_ratio < p["breakout_volume_mult"]:
        return None

    return {
        "symbol": symbol,
        "strategy": "intraday_breakout",
        "side": "buy",
        "price": float(curr["close"]),
        "morning_high": float(morning_high),
        "volume_ratio": round(volume_ratio, 1),
        "rationale": (
            f"Breakout: {symbol} above morning high {morning_high:.2f}, "
            f"volume {volume_ratio:.1f}x average"
        ),
    }


def get_intraday_positions(trading_client: TradingClient) -> dict[str, dict]:
    """Get current positions, filtering to intraday universe only."""
    positions = trading_client.get_all_positions()
    result = {}
    for pos in positions:
        if pos.symbol in INTRADAY_UNIVERSE:
            result[pos.symbol] = {
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
            }
    return result


def close_eod_positions(trading_client: TradingClient, positions: dict, dry_run: bool):
    """Close all intraday positions near end of day."""
    for symbol, pos in positions.items():
        print(
            f"  EOD close: {symbol} ({pos['qty']} shares, "
            f"P&L: ${pos['unrealized_pl']:.2f})",
            file=sys.stderr,
        )
        if not dry_run:
            try:
                trading_client.close_position(symbol)
            except Exception as e:
                print(f"  ERROR closing {symbol}: {e}", file=sys.stderr)


def execute_signal(
    signal: dict,
    trading_client: TradingClient,
    equity: float,
    dry_run: bool,
) -> bool:
    """Execute an intraday trade signal."""
    price = signal["price"]
    max_value = equity * PARAMS["max_position_pct"]
    qty = int(max_value / price)

    if qty <= 0:
        return False

    stop_price = round(price * (1 - PARAMS["stop_loss_pct"]), 2)
    tp_price = round(price * (1 + PARAMS["take_profit_pct"]), 2)

    print(
        f"  SIGNAL: {signal['side'].upper()} {qty} {signal['symbol']} "
        f"@ ${price:.2f} "
        f"(stop: ${stop_price}, tp: ${tp_price})",
        file=sys.stderr,
    )
    print(f"    {signal['rationale']}", file=sys.stderr)

    if dry_run:
        print("    [DRY RUN] — not executed", file=sys.stderr)
        return False

    try:
        order = MarketOrderRequest(
            symbol=signal["symbol"],
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        result = trading_client.submit_order(order_data=order)
        print(
            f"    Order submitted: id={result.id}, status={result.status}",
            file=sys.stderr,
        )

        # Submit broker-side stop-loss (DAY duration for intraday trades)
        try:
            import time
            time.sleep(2)  # brief wait for fill
            stop_order = StopOrderRequest(
                symbol=signal["symbol"],
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=stop_price,
            )
            stop_result = trading_client.submit_order(order_data=stop_order)
            print(
                f"    Stop-loss submitted: sell {qty} @ ${stop_price} "
                f"(id={stop_result.id})",
                file=sys.stderr,
            )
        except Exception as se:
            print(f"    WARNING: Stop-loss order failed: {se}", file=sys.stderr)

        return True
    except Exception as e:
        print(f"    ERROR submitting order: {e}", file=sys.stderr)
        return False


def check_stop_and_tp(
    positions: dict,
    data: dict[str, pd.DataFrame],
    trading_client: TradingClient,
    dry_run: bool,
):
    """Check if any positions hit stop loss or take profit."""
    for symbol, pos in positions.items():
        plpc = pos["unrealized_plpc"]

        if plpc <= -PARAMS["stop_loss_pct"]:
            print(
                f"  STOP HIT: {symbol} at {plpc:.2%} loss",
                file=sys.stderr,
            )
            if not dry_run:
                try:
                    trading_client.close_position(symbol)
                except Exception as e:
                    print(f"  ERROR closing {symbol}: {e}", file=sys.stderr)

        elif plpc >= PARAMS["take_profit_pct"]:
            print(
                f"  TAKE PROFIT: {symbol} at {plpc:.2%} gain",
                file=sys.stderr,
            )
            if not dry_run:
                try:
                    trading_client.close_position(symbol)
                except Exception as e:
                    print(f"  ERROR closing {symbol}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Intraday scanner")
    parser.add_argument("--paper", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paper = not args.live

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  INTRADAY SCAN — {datetime.now().isoformat()}", file=sys.stderr)
    print(f"  Mode: {'PAPER' if paper else 'LIVE'} | Dry run: {args.dry_run}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    data_client, trading_client = get_clients(paper)

    # Check if market is open
    if not is_market_open(trading_client):
        print("Market is closed. Skipping scan.", file=sys.stderr)
        print(json.dumps({"status": "market_closed", "signals": []}))
        return

    # Check if near close — exit all positions instead of scanning
    if is_near_close(trading_client):
        print("Near market close — exiting all intraday positions.", file=sys.stderr)
        positions = get_intraday_positions(trading_client)
        if positions:
            close_eod_positions(trading_client, positions, args.dry_run)
        else:
            print("  No intraday positions to close.", file=sys.stderr)
        print(json.dumps({"status": "eod_exit", "positions_closed": len(positions)}))
        return

    # Get account info
    account = trading_client.get_account()
    equity = float(account.equity)
    positions = get_intraday_positions(trading_client)
    num_intraday_positions = len(positions)

    print(f"Equity: ${equity:,.2f}", file=sys.stderr)
    print(f"Intraday positions: {num_intraday_positions}", file=sys.stderr)

    # Check stops and take profits on existing positions
    if positions:
        print("\nChecking stops/TPs on existing positions...", file=sys.stderr)
        check_stop_and_tp(positions, {}, trading_client, args.dry_run)
        # Refresh positions after potential closes
        positions = get_intraday_positions(trading_client)
        num_intraday_positions = len(positions)

    # Scan for new signals if we have room
    signals = []
    if num_intraday_positions < PARAMS["max_intraday_positions"]:
        print(f"\nFetching 15-min bars for {len(INTRADAY_UNIVERSE)} symbols...", file=sys.stderr)
        data = fetch_intraday_bars(data_client, INTRADAY_UNIVERSE)
        print(f"Got data for {len(data)} symbols", file=sys.stderr)

        for symbol, df in data.items():
            # Skip if we already have a position
            if symbol in positions:
                continue

            # Scan dip buy
            dip = scan_dip_buy(symbol, df, PARAMS)
            if dip:
                signals.append(dip)

            # Scan breakout
            brk = scan_breakout(symbol, df, PARAMS)
            if brk:
                signals.append(brk)

        print(f"\nSignals found: {len(signals)}", file=sys.stderr)

        # Execute top signals (sorted by most extreme dip first, then breakout volume)
        signals.sort(key=lambda s: (
            s.get("change_from_open", 0),  # bigger dips first (more negative)
            -s.get("volume_ratio", 0),      # higher volume breakouts first
        ))

        executed = 0
        for signal in signals:
            if num_intraday_positions + executed >= PARAMS["max_intraday_positions"]:
                break
            if execute_signal(signal, trading_client, equity, args.dry_run):
                executed += 1

    # Output summary as JSON
    output = {
        "status": "scanned",
        "timestamp": datetime.now().isoformat(),
        "equity": equity,
        "intraday_positions": num_intraday_positions,
        "signals_found": len(signals),
        "signals": signals,
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
