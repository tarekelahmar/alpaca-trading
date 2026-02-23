from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"


@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    strategy_name: str
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    features: dict = field(default_factory=dict)
    rationale: str = ""


@dataclass
class StrategyConfig:
    name: str
    params: dict = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Strategies must be deterministic: given the same data, they must
    produce the same signals. No randomness, no LLM calls.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        """Generate trading signals from market data.

        Args:
            data: Dict mapping symbol -> DataFrame with OHLCV columns.
                  DataFrame must have columns: open, high, low, close, volume
                  and a DatetimeIndex.

        Returns:
            List of Signal objects for symbols that meet entry/exit criteria.
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return current strategy parameters for logging/serialization."""
        ...

    def required_history_days(self) -> int:
        """Minimum number of trading days of history needed to generate signals.
        Override if the strategy needs more than the default.
        """
        return 252

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Check that a DataFrame has the required columns and sufficient rows."""
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            return False
        if len(df) < self.required_history_days():
            return False
        return True
