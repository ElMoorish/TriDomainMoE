"""
Order Flow Imbalance (OFI) and Microstructure Liquidity Dynamics.
Implements the multi-level and top-of-book OFI formulation from Cont et al. / Design Spec.
"""

from typing import Optional, List, Dict
import numpy as np
import pandas as pd


class OrderFlowImbalance:
    """
    Computes Order Flow Imbalance (OFI) across tick updates and aggregates
    over volume bars to capture institutional inventory rebalancing.
    """

    @staticmethod
    def compute_tick_ofi(
        bid_prices: np.ndarray,
        bid_sizes: np.ndarray,
        ask_prices: np.ndarray,
        ask_sizes: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate OFI across consecutive tick states:
        OFI_t = I_{P_t^b >= P_{t-1}^b} q_t^b - I_{P_t^b <= P_{t-1}^b} q_{t-1}^b
               - I_{P_t^a <= P_{t-1}^a} q_t^a + I_{P_t^a >= P_{t-1}^a} q_{t-1}^a
        """
        n = len(bid_prices)
        if n < 2:
            return np.zeros(n, dtype=float)

        ofi = np.zeros(n, dtype=float)

        p_b_prev = bid_prices[:-1]
        p_b_curr = bid_prices[1:]
        q_b_prev = bid_sizes[:-1]
        q_b_curr = bid_sizes[1:]

        p_a_prev = ask_prices[:-1]
        p_a_curr = ask_prices[1:]
        q_a_prev = ask_sizes[:-1]
        q_a_curr = ask_sizes[1:]

        # Bid delta component
        bid_ge = p_b_curr >= p_b_prev
        bid_le = p_b_curr <= p_b_prev
        delta_bid = np.where(bid_ge, q_b_curr, 0.0) - np.where(bid_le, q_b_prev, 0.0)

        # Ask delta component
        ask_le = p_a_curr <= p_a_prev
        ask_ge = p_a_curr >= p_a_prev
        delta_ask = np.where(ask_le, q_a_curr, 0.0) - np.where(ask_ge, q_a_prev, 0.0)

        ofi[1:] = delta_bid - delta_ask
        return ofi

    @classmethod
    def from_ticks_df(cls, ticks_df: pd.DataFrame) -> pd.Series:
        """Calculate tick-level OFI series from standard ticks DataFrame."""
        if ticks_df.empty or len(ticks_df) < 2:
            return pd.Series(index=ticks_df.index, dtype=float)

        bid_p = ticks_df["bid"].to_numpy(dtype=float)
        ask_p = ticks_df["ask"].to_numpy(dtype=float)

        # In MT5 tick feed, 'volume' is tick size or traded volume;
        # If bid_size/ask_size are absent, estimate from tick volume
        bid_v = ticks_df["bid_volume"].to_numpy(dtype=float) if "bid_volume" in ticks_df.columns else ticks_df["volume"].to_numpy(dtype=float)
        ask_v = ticks_df["ask_volume"].to_numpy(dtype=float) if "ask_volume" in ticks_df.columns else ticks_df["volume"].to_numpy(dtype=float)

        ofi_arr = cls.compute_tick_ofi(bid_p, bid_v, ask_p, ask_v)
        return pd.Series(ofi_arr, index=ticks_df.index, name="ofi")

    @classmethod
    def aggregate_to_bars(cls, ticks_df: pd.DataFrame, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Accumulates tick-level OFI into volume bars via asof / interval merge.
        Returns bars_df enriched with 'cum_ofi' and 'normalized_ofi'.
        """
        if ticks_df.empty or bars_df.empty:
            return bars_df

        df_t = ticks_df.copy()
        if "ofi" not in df_t.columns:
            df_t["ofi"] = cls.from_ticks_df(df_t)

        # Group ticks by corresponding bar timestamp using pd.merge_asof
        df_t_sorted = df_t.sort_index()
        bars_sorted = bars_df.sort_index()

        # Assign each tick to the closest subsequent bar close
        merged = pd.merge_asof(
            df_t_sorted[["ofi"]].reset_index(),
            bars_sorted.reset_index()[["timestamp"]].rename(columns={"timestamp": "bar_ts"}),
            left_on="timestamp",
            right_on="bar_ts",
            direction="forward",
        )

        ofi_by_bar = merged.groupby("bar_ts")["ofi"].sum()
        bars_enriched = bars_df.copy()
        bars_enriched["cum_ofi"] = ofi_by_bar.reindex(bars_enriched.index).fillna(0.0)

        # Standardize OFI by volume
        bars_enriched["normalized_ofi"] = np.where(
            bars_enriched["volume"] > 0,
            bars_enriched["cum_ofi"] / bars_enriched["volume"],
            0.0,
        )

        return bars_enriched
