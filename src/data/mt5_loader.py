"""
MetaTrader 5 Data Loader Implementation.
Provides robust, high-throughput extraction of ticks, OHLC rates, and L2 market depth.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
import logging
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

from .base import BaseDataLoader

logger = logging.getLogger(__name__)


class MT5DataLoader(BaseDataLoader):
    """Production-grade MT5 data adapter."""

    TIMEFRAME_MAP = {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M4": 4,
        "M5": 5,
        "M6": 6,
        "M10": 10,
        "M12": 12,
        "M15": 15,
        "M20": 20,
        "M30": 30,
        "H1": 16385,
        "H2": 16386,
        "H3": 16387,
        "H4": 16388,
        "H6": 16390,
        "H8": 16392,
        "H12": 16396,
        "D1": 16408,
        "W1": 32769,
        "MN1": 49153,
    }

    def __init__(self, path: Optional[str] = None, portable: bool = False):
        self.path = path
        self.portable = portable
        self._connected = False

    def connect(self) -> bool:
        """Initialize connection to MetaTrader 5 terminal."""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package is not installed.")
            return False

        if mt5.initialize(path=self.path, portable=self.portable) if self.path else mt5.initialize():
            terminal_info = mt5.terminal_info()
            if terminal_info is not None and terminal_info.connected:
                self._connected = True
                logger.info("MT5 initialized successfully. Terminal connected: %s", terminal_info.name)
                return True
            else:
                logger.warning("MT5 initialized, but terminal reports not connected to trade server.")
                self._connected = True
                return True
        else:
            err = mt5.last_error()
            logger.error("Failed to initialize MT5: %s", err)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Shutdown MT5 terminal connection."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 connection closed.")

    def is_connected(self) -> bool:
        """Check if terminal connection is healthy."""
        if not MT5_AVAILABLE or not self._connected:
            return False
        info = mt5.terminal_info()
        return info is not None and info.connected

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch symbol specification."""
        if not self._connected and not self.connect():
            return None

        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning("Symbol %s not found in MT5.", symbol)
            return None

        return {
            "symbol": info.name,
            "digits": info.digits,
            "point": info.point,
            "spread": info.spread,
            "trade_mode": info.trade_mode,
            "tick_size": info.trade_tick_size,
            "tick_value": info.trade_tick_value,
            "contract_size": info.trade_contract_size,
            "bid": info.bid,
            "ask": info.ask,
        }

    def get_ticks(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        max_ticks: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical ticks with automatic chunked pagination.
        Ensures consistent datetime conversion to UTC.
        """
        if not self._connected and not self.connect():
            raise ConnectionError("Cannot connect to MT5.")

        mt5.symbol_select(symbol, True)
        if end_time is None:
            end_time = datetime.now()

        # MT5 accepts timestamps in UTC or naive local
        logger.info("Fetching ticks for %s from %s to %s", symbol, start_time, end_time)
        
        # If interval is large, paginate day by day to protect memory
        ticks_list = []
        curr_start = start_time
        delta_chunk = timedelta(days=1)

        while curr_start < end_time:
            curr_end = min(curr_start + delta_chunk, end_time)
            raw_ticks = mt5.copy_ticks_range(symbol, curr_start, curr_end, mt5.COPY_TICKS_ALL)
            
            if raw_ticks is not None and len(raw_ticks) > 0:
                ticks_list.append(pd.DataFrame(raw_ticks))
                if max_ticks and sum(len(df) for df in ticks_list) >= max_ticks:
                    break
            
            curr_start = curr_end

        if not ticks_list:
            logger.warning("No ticks retrieved for %s in requested window.", symbol)
            return pd.DataFrame(columns=["timestamp", "bid", "ask", "last", "volume", "flags"])

        df = pd.concat(ticks_list, ignore_index=True)
        if max_ticks and len(df) > max_ticks:
            df = df.iloc[:max_ticks]

        # Convert to canonical schema
        # MT5 tick record contains 'time_msc' for millisecond precision
        if "time_msc" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # Standardize columns
        df = df[["timestamp", "bid", "ask", "last", "volume", "flags"]]
        df.drop_duplicates(subset=["timestamp", "bid", "ask", "volume"], keep="first", inplace=True)
        df.sort_values(by="timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        return df

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "M1",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        count: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLC bars for a given symbol and timeframe.
        """
        if not self._connected and not self.connect():
            raise ConnectionError("Cannot connect to MT5.")

        mt5.symbol_select(symbol, True)
        tf_code = self.TIMEFRAME_MAP.get(timeframe.upper())
        if tf_code is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Choose from {list(self.TIMEFRAME_MAP.keys())}")

        if count is not None:
            rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, count)
        elif start_time is not None and end_time is not None:
            rates = mt5.copy_rates_range(symbol, tf_code, start_time, end_time)
        elif start_time is not None:
            rates = mt5.copy_rates_from(symbol, tf_code, start_time, 10000)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, 1000)

        if rates is None or len(rates) == 0:
            logger.warning("No rates returned for %s (%s).", symbol, timeframe)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "tick_volume", "spread"])

        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "tick_volume", "spread"]]
        df.sort_values(by="timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)
        return df

    def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve the latest live tick snapshot."""
        if not self._connected and not self.connect():
            return None

        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            "timestamp": pd.to_datetime(tick.time_msc, unit="ms", utc=True),
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "flags": tick.flags,
        }

    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve Level 2 Market Depth if enabled by broker."""
        if not self._connected and not self.connect():
            return None

        mt5.symbol_select(symbol, True)
        mt5.market_book_add(symbol)
        book = mt5.market_book_get(symbol)
        mt5.market_book_release(symbol)

        if book is None:
            return None

        bids = []
        asks = []
        for item in book:
            # item type: 1 = BUY (Bid), 2 = SELL (Ask)
            entry = {"price": item.price, "volume": item.volume}
            if item.type == 1:
                bids.append(entry)
            else:
                asks.append(entry)

        return {"bids": bids, "asks": asks}
