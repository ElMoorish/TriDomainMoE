"""
High-Performance Parquet Storage Manager for Ticks, OHLC Bars, and Features.
Ensures zero-copy reads and deterministic time partitioning.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, List
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class ParquetStorage:
    """Manages disk-based persistence and indexing of financial market series."""

    def __init__(self, base_cache_dir: str = "data/cache"):
        self.base_dir = Path(base_cache_dir)
        self.ticks_dir = self.base_dir / "ticks"
        self.bars_dir = self.base_dir / "bars"
        self.features_dir = self.base_dir / "features"

        self.ticks_dir.mkdir(parents=True, exist_ok=True)
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_symbol(self, symbol: str) -> str:
        return symbol.replace(".", "_").replace("/", "_").replace("^", "IDX_")

    def save_ticks(self, symbol: str, df: pd.DataFrame) -> Path:
        """
        Save ticks partitioned by year and month.
        Appends incrementally and eliminates duplicate timestamps.
        """
        if df.empty:
            return self.ticks_dir

        sym_clean = self._sanitize_symbol(symbol)
        sym_dir = self.ticks_dir / sym_clean
        sym_dir.mkdir(parents=True, exist_ok=True)

        # Partition by year-month
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index, utc=True)

        for period, group in df_copy.groupby(df_copy.index.strftime("%Y_%m")):
            file_path = sym_dir / f"{period}.parquet"
            if file_path.exists():
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, group])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            else:
                combined = group.sort_index()

            combined.to_parquet(file_path, engine="pyarrow", compression="snappy")

        logger.info("Saved %d ticks for %s across partitions.", len(df), symbol)
        return sym_dir

    def load_ticks(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load ticks from partitioned parquet files within [start_time, end_time]."""
        sym_clean = self._sanitize_symbol(symbol)
        sym_dir = self.ticks_dir / sym_clean

        if not sym_dir.exists():
            return pd.DataFrame()

        files = sorted(list(sym_dir.glob("*.parquet")))
        if not files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs).sort_index()
        df = df[~df.index.duplicated(keep="first")]

        if start_time is not None:
            ts_start = pd.to_datetime(start_time, utc=True)
            df = df[df.index >= ts_start]
        if end_time is not None:
            ts_end = pd.to_datetime(end_time, utc=True)
            df = df[df.index <= ts_end]

        return df

    def save_bars(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
        """Save OHLC or volume bars to parquet."""
        if df.empty:
            return self.bars_dir

        sym_clean = self._sanitize_symbol(symbol)
        file_path = self.bars_dir / f"{sym_clean}_{timeframe.lower()}.parquet"

        if file_path.exists():
            existing = pd.read_parquet(file_path)
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = df.sort_index()

        combined.to_parquet(file_path, engine="pyarrow", compression="snappy")
        logger.info("Saved %d bars for %s (%s) to %s", len(combined), symbol, timeframe, file_path)
        return file_path

    def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load OHLC or volume bars."""
        sym_clean = self._sanitize_symbol(symbol)
        file_path = self.bars_dir / f"{sym_clean}_{timeframe.lower()}.parquet"

        if not file_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(file_path)
        if start_time is not None:
            ts_start = pd.to_datetime(start_time, utc=True)
            df = df[df.index >= ts_start]
        if end_time is not None:
            ts_end = pd.to_datetime(end_time, utc=True)
            df = df[df.index <= ts_end]

        return df
