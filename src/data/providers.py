"""
Provider Registry and Loader Factory.
Enables pluggable data backends (MT5, Interactive Brokers, Web APIs).
"""

from typing import Dict, Type
from .base import BaseDataLoader
from .mt5_loader import MT5DataLoader


_REGISTRY: Dict[str, Type[BaseDataLoader]] = {
    "mt5": MT5DataLoader,
}


def register_provider(name: str, loader_cls: Type[BaseDataLoader]) -> None:
    """Register a new provider class."""
    _REGISTRY[name.lower()] = loader_cls


def get_data_loader(provider_name: str = "mt5", **kwargs) -> BaseDataLoader:
    """
    Instantiate and return the requested data loader.
    Raises ValueError if provider is unregistered.
    """
    key = provider_name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Provider '{provider_name}' is not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[key](**kwargs)
