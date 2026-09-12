"""
Fractional Differencing with Augmented Dickey-Fuller (ADF) Optimization.
Preserves long-term financial memory while establishing weak stationarity.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


class FractionalDifferentiator:
    """
    Implements binomial expansion of (1 - B)^d with recursive memory weights
    and grid searches for the minimum stationarity order d* via ADF test.
    """

    @staticmethod
    def get_weights(d: float, size: int, threshold: float = 1e-4) -> np.ndarray:
        """
        Compute memory weights omega_k recursively:
        omega_0 = 1, omega_k = -omega_{k-1} * (d - k + 1) / k
        """
        weights = [1.0]
        k = 1
        while k < size:
            w = -weights[-1] / k * (d - k + 1)
            if abs(w) < threshold:
                break
            weights.append(w)
            k += 1
        return np.array(weights[::-1])  # Reverse for convolution

    @classmethod
    def frac_diff(cls, series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
        """
        Apply fractional differentiation of order d to a univariate series.
        """
        if d == 0.0:
            return series.copy()

        weights = cls.get_weights(d, len(series), threshold)
        res = np.convolve(series.to_numpy(), weights, mode="valid")
        valid_index = series.index[len(weights) - 1 :]
        return pd.Series(res, index=valid_index, name=f"{series.name}_fd_{d:.2f}")

    @classmethod
    def find_min_d(
        cls,
        series: pd.Series,
        d_min: float = 0.0,
        d_max: float = 1.0,
        step: float = 0.05,
        p_val_thresh: float = 0.05,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Search for minimal d* in [d_min, d_max] rejecting non-stationarity null hypothesis (p < 0.05).
        Returns optimal d* and diagnostic table of (d, adf_stat, p_val, 95% critical value).
        """
        clean_series = series.dropna()
        records = []
        best_d = d_max

        d_candidates = np.arange(d_min, d_max + 1e-6, step)
        found = False

        for d in d_candidates:
            diffed = cls.frac_diff(clean_series, d)
            if len(diffed) < 30:
                continue

            try:
                adf_res = adfuller(diffed, maxlag=1, autolag=None)
                stat = adf_res[0]
                pval = adf_res[1]
                crit95 = adf_res[4]["5%"]

                records.append({
                    "d": round(float(d), 2),
                    "adf_stat": stat,
                    "p_value": pval,
                    "crit_95": crit95,
                    "stationary": pval < p_val_thresh,
                })

                if pval < p_val_thresh and not found:
                    best_d = round(float(d), 2)
                    found = True
            except Exception:
                continue

        df_results = pd.DataFrame(records)
        return best_d, df_results
