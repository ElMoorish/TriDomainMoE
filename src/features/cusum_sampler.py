"""
Symmetric Cumulative Sum (CUSUM) Event-Driven Sampling Filter.
Extracts observations when structural information arrives rather than arbitrary calendar steps.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


class CUSUMFilter:
    """
    Symmetric CUSUM event filter driven by adaptive volatility thresholds.
    Filters out uninformative noise periods and samples heavily during volatility bursts.
    """

    def __init__(self, h_multiplier: float = 1.5, vol_lookback: int = 100):
        self.h_multiplier = h_multiplier
        self.vol_lookback = vol_lookback

    def filter_events(self, close_prices: pd.Series) -> pd.DatetimeIndex:
        """
        Extract timestamps where cumulative innovations cross adaptive threshold h * sigma_t.
        
        Args:
            close_prices: Price series indexed by DatetimeIndex.
            
        Returns:
            pd.DatetimeIndex containing only event trigger timestamps.
        """
        if len(close_prices) < self.vol_lookback + 2:
            return close_prices.index

        # Compute log returns
        log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0.0)
        
        # Adaptive volatility: rolling standard deviation of log returns
        rolling_vol = log_returns.rolling(window=self.vol_lookback, min_periods=10).std().bfill()
        
        # Expected return baseline (rolling mean or 0 for high-frequency)
        exp_ret = log_returns.rolling(window=self.vol_lookback, min_periods=10).mean().fillna(0.0)

        returns_arr = log_returns.to_numpy()
        vol_arr = rolling_vol.to_numpy()
        exp_arr = exp_ret.to_numpy()
        timestamps = close_prices.index

        s_pos = 0.0
        s_neg = 0.0
        event_indices: List[int] = []

        for t in range(len(returns_arr)):
            r = returns_arr[t]
            e = exp_arr[t]
            sigma = vol_arr[t]
            threshold = self.h_multiplier * sigma

            # If threshold is nearly zero (e.g. at start), safeguard
            if threshold <= 1e-8:
                threshold = 1e-4

            # Update CUSUM accumulators
            s_pos = max(0.0, s_pos + r - e)
            s_neg = min(0.0, s_neg + r - e)

            if max(s_pos, -s_neg) >= threshold:
                event_indices.append(t)
                # Reset accumulators upon trigger
                s_pos = 0.0
                s_neg = 0.0

        if not event_indices:
            # Fallback to all indices if threshold is too strict
            return close_prices.index

        return timestamps[event_indices]
