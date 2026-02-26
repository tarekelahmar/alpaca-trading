"""Crypto trading module for Alpaca.

Adapts existing equity strategies for 24/7 cryptocurrency markets.
Uses the same Alpaca API — no additional broker needed.

Key differences from equities:
    - 24/7 trading (no market hours)
    - Higher volatility → wider stops (3x ATR vs 2x)
    - No gap trading (no market close = no gaps)
    - BTC dominance rotation replaces sector rotation
    - Shorter momentum lookback (1-3 months vs 6 months)
    - Mean reversion only in non-trending regimes (dangerous in crypto trends)
"""
