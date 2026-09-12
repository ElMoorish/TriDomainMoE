"""
Information-Driven Bar Sampling: Volume Bars and Dollar Bars.
Converts irregular calendar-time tick flow into information-homogenous bars.
"""

from typing import Optional
import numpy as np
import pandas as pd


class VolumeBarBuilder:
    """Aggregates tick data into volume-based or dollar-based bars."""

    def __init__(self, volume_threshold: float = 1000.0, dollar_threshold: Optional[float] = None):
        self.volume_threshold = volume_threshold
        self.dollar_threshold = dollar_threshold

    def build_bars(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process tick stream into OHLCV volume bars.
        
        Args:
            ticks_df: DataFrame with ['bid', 'ask', 'volume'] or ['last', 'volume']
                      and DatetimeIndex.
        
        Returns:
            DataFrame of volume bars with ['open', 'high', 'low', 'close', 
                                          'volume', 'vwap', 'ticks', 'spread']
        """
        if ticks_df.empty:
            return pd.DataFrame()

        # Compute mid price or use 'last' if non-zero
        if "last" in ticks_df.columns and (ticks_df["last"] > 0).any():
            prices = np.where(ticks_df["last"] > 0, ticks_df["last"], (ticks_df["bid"] + ticks_df["ask"]) / 2.0)
        else:
            prices = ((ticks_df["bid"] + ticks_df["ask"]) / 2.0).to_numpy()

        volumes = ticks_df["volume"].to_numpy(dtype=float)
        # Handle zero-volume ticks by defaulting to 1 unit
        volumes = np.where(volumes <= 0, 1.0, volumes)

        spreads = (ticks_df["ask"] - ticks_df["bid"]).to_numpy(dtype=float) if "ask" in ticks_df.columns else np.zeros_like(prices)
        timestamps = ticks_df.index.to_numpy()

        bars = []
        n = len(prices)

        cum_vol = 0.0
        cum_dollar = 0.0
        bar_ticks = 0
        spread_sum = 0.0

        bar_open = prices[0]
        bar_high = prices[0]
        bar_low = prices[0]
        bar_close = prices[0]

        buy_vol = 0.0
        sell_vol = 0.0
        prev_price = prices[0]

        for i in range(n):
            p = prices[i]
            v = volumes[i]
            s = spreads[i]
            d = p * v

            if bar_ticks == 0:
                bar_open = p
                bar_high = p
                bar_low = p

            if p > bar_high:
                bar_high = p
            if p < bar_low:
                bar_low = p
            bar_close = p

            # Tick rule for signed volume
            if p > prev_price:
                buy_vol += v
            elif p < prev_price:
                sell_vol += v
            else:
                # Same price: inherit previous direction
                buy_vol += v * 0.5
                sell_vol += v * 0.5
            prev_price = p

            cum_vol += v
            cum_dollar += d
            spread_sum += s
            bar_ticks += 1

            # Trigger threshold condition
            threshold_met = False
            if self.dollar_threshold is not None:
                threshold_met = (cum_dollar >= self.dollar_threshold)
            else:
                threshold_met = (cum_vol >= self.volume_threshold)

            if threshold_met:
                vwap = cum_dollar / cum_vol if cum_vol > 0 else bar_close
                bars.append({
                    "timestamp": timestamps[i],
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                    "volume": cum_vol,
                    "dollar_value": cum_dollar,
                    "vwap": vwap,
                    "ticks": bar_ticks,
                    "spread": spread_sum / bar_ticks if bar_ticks > 0 else 0.0,
                    "buy_volume": buy_vol,
                    "sell_volume": sell_vol,
                })

                # Reset accumulators
                cum_vol = 0.0
                cum_dollar = 0.0
                bar_ticks = 0
                spread_sum = 0.0
                buy_vol = 0.0
                sell_vol = 0.0

        if not bars:
            return pd.DataFrame()

        df_bars = pd.DataFrame(bars)
        df_bars["timestamp"] = pd.to_datetime(df_bars["timestamp"], utc=True)
        df_bars.set_index("timestamp", inplace=True)
        return df_bars
