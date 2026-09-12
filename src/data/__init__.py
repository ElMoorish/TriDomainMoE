"""Data ingestion and persistence module."""

from .base import BaseDataLoader, SymbolInfo
from .mt5_loader import MT5DataLoader
from .storage import ParquetStorage
from .providers import get_data_loader, register_provider

__all__ = [
    "BaseDataLoader",
    "SymbolInfo",
    "MT5DataLoader",
    "ParquetStorage",
    "get_data_loader",
    "register_provider",
]
