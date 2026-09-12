"""
Triple-Barrier Meta-Labeling and Calibrated Position Sizing.
Implements dynamic upper/lower horizontal volatility barriers, vertical time expirations,
and continuous calibrated sigmoid trade sizing.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


class TripleBarrierLabeler:
    """
    Labels financial time series using dynamic profit-taking, stop-loss,
    and vertical time-decay barriers.
    """

    def __init__(
        self,
        pt_multiplier: float = 2.0,
        sl_multiplier: float = 1.5,
        max_holding_bars: int = 20,
        vol_lookback: int = 50,
    ):
        self.pt_mult = pt_multiplier
        self.sl_mult = sl_multiplier
        self.max_holding_bars = max_holding_bars
        self.vol_lookback = vol_lookback

    def label_events(
        self,
        prices: pd.Series,
        event_timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        Evaluate triple-barrier outcomes for each event timestamp.
        
        Returns DataFrame with:
        ['ret', 'label', 'first_barrier', 'holding_bars', 'p_0', 'volatility']
        where label is +1 (profit-hit), -1 (stop-loss hit), or 0 (vertical expiration).
        """
        if event_timestamps is None:
            event_timestamps = prices.index

        # Compute rolling volatility baseline
        log_ret = np.log(prices / prices.shift(1)).fillna(0.0)
        rolling_vol = log_ret.rolling(self.vol_lookback, min_periods=5).std().bfill()

        price_arr = prices.to_numpy()
        vol_arr = rolling_vol.to_numpy()
        idx_map = {ts: i for i, ts in enumerate(prices.index)}

        records = []
        n_total = len(prices)

        for ts in event_timestamps:
            if ts not in idx_map:
                continue
            t0 = idx_map[ts]
            p0 = price_arr[t0]
            sigma = vol_arr[t0]

            if sigma <= 1e-8:
                sigma = 1e-3

            upper_barrier = p0 * (1.0 + self.pt_mult * sigma)
            lower_barrier = p0 * (1.0 - self.sl_mult * sigma)

            max_idx = min(t0 + self.max_holding_bars, n_total - 1)
            path_prices = price_arr[t0 + 1 : max_idx + 1]

            label = 0
            first_hit = "vertical"
            bars_held = len(path_prices)

            for step_i, pt in enumerate(path_prices):
                if pt >= upper_barrier:
                    label = 1
                    first_hit = "upper"
                    bars_held = step_i + 1
                    break
                elif pt <= lower_barrier:
                    label = -1
                    first_hit = "lower"
                    bars_held = step_i + 1
                    break

            final_p = path_prices[bars_held - 1] if len(path_prices) > 0 else p0
            ret = (final_p - p0) / p0

            records.append({
                "timestamp": ts,
                "p0": p0,
                "volatility": sigma,
                "return": ret,
                "label": label,
                "first_barrier": first_hit,
                "holding_bars": bars_held,
            })

        df_out = pd.DataFrame(records)
        if not df_out.empty:
            df_out.set_index("timestamp", inplace=True)
        return df_out

    @staticmethod
    def calibrated_bet_size(meta_prob: np.ndarray, sigma_cal: float = 0.25) -> np.ndarray:
        """
        Maps secondary meta-label probability p_t in [0, 1] to continuous sizing s_t in [0, 1]:
        s_t = max(0, 2 * (sigmoid((p_t - 0.5) / sigma_cal) - 0.5))
        """
        z = (meta_prob - 0.5) / sigma_cal
        sigmoid_z = 1.0 / (1.0 + np.exp(-z))
        s_t = 2.0 * (sigmoid_z - 0.5)
        return np.clip(s_t, 0.0, 1.0)
