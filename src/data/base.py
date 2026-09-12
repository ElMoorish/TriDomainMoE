"""
Abstract Base Classes and Data Contracts for Market Data Loaders.
Supports seamless switching between MT5, Interactive Brokers, and Flat/Parquet files.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd


@dataclass
class SymbolInfo:
    """Canonical representation of symbol properties."""
    symbol: str
    asset_class: str
    base_asset: str
    display_name: str
    point_size: float
    volume_bar_threshold: int
    dollar_bar_threshold: float
    cusum_h_multiplier: float
    digits: int = 2


class BaseDataLoader(ABC):
    """Abstract interface for all financial data adapters."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection with the data provider."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully terminate connection."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if connection is alive and healthy."""
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch broker specifications for a given symbol."""
        pass

    @abstractmethod
    def get_ticks(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        max_ticks: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical millisecond ticks.
        
        Returned DataFrame MUST conform to canonical columns:
        ['timestamp', 'bid', 'ask', 'last', 'volume', 'flags']
        Indexed by pd.DatetimeIndex ('timestamp').
        """
        pass

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        count: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLC bars.
        
        Supported timeframes: 'M1', 'M5', 'M15', 'H1', 'H4', 'D1'.
        Returned DataFrame MUST conform to canonical columns:
        ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']
        Indexed by pd.DatetimeIndex ('timestamp').
        """
        pass

    @abstractmethod
    def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch the most recent tick snapshot for real-time inference."""
        pass

    @abstractmethod
    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current L2 / DOM snapshot (bids and asks with depth).
        Returns None if not supported by the underlying provider.
        """
        pass
