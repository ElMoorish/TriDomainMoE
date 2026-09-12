"""
Deflated Sharpe Ratio (DSR) and Multiple-Testing Hypothesis Corrections.
Tests whether an observed out-of-sample Sharpe ratio remains statistically significant
after correcting for parameter trials, non-normality (skew/kurtosis), and sample length.
"""

from typing import Dict, Any, Optional
import math
import numpy as np
from scipy.stats import norm, skew, kurtosis


class DeflatedSharpeRatio:
    """
    Computes effective independent trials N_eff and the Deflated Sharpe Ratio (DSR).
    Threshold DSR >= 0.95 rejects the null hypothesis of selection bias.
    """

    EULER_MASCHERONI = 0.5772156649

    @classmethod
    def compute_n_eff(cls, trial_returns_matrix: np.ndarray) -> float:
        """
        Derives N_eff from eigenvalue spectrum of trial correlation matrix C:
        N_eff = K^2 / Tr(C^2)
        
        Args:
            trial_returns_matrix: Array of shape (T, K) for K strategy trials across T periods.
        """
        k = trial_returns_matrix.shape[1]
        if k <= 1:
            return 1.0

        corr_matrix = np.corrcoef(trial_returns_matrix, rowvar=False)
        # Handle potential NaNs in correlation
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        tr_c2 = float(np.sum(corr_matrix ** 2))
        n_eff = (k ** 2) / tr_c2
        return float(np.clip(n_eff, 1.0, float(k)))

    @classmethod
    def expected_max_sharpe(cls, n_trials: float) -> float:
        """
        E[max_n {SR_n}] approx sqrt(2 * ln(N)) * (1 - gamma / (2 * ln(N))) + gamma / sqrt(2 * ln(N))
        """
        if n_trials <= 1.0:
            return 0.0

        ln_n = math.log(n_trials)
        sqrt_2_ln_n = math.sqrt(2.0 * ln_n)
        term1 = sqrt_2_ln_n * (1.0 - cls.EULER_MASCHERONI / (2.0 * ln_n))
        term2 = cls.EULER_MASCHERONI / sqrt_2_ln_n
        return float(term1 + term2)

    @classmethod
    def compute_dsr(
        cls,
        observed_sr: float,
        returns: np.ndarray,
        n_trials: float = 1.0,
        annualization_factor: float = 252.0,
    ) -> Dict[str, Any]:
        """
        Compute Deflated Sharpe Ratio.
        
        Args:
            observed_sr: Annualized Sharpe ratio of best trial.
            returns: Array of period returns (T,).
            n_trials: Effective number of trials N_eff.
            annualization_factor: Number of periods in a year (e.g. 252 for daily, 252*390 for M1).
        """
        t = len(returns)
        if t < 30:
            return {"dsr": 0.0, "sr_benchmark": 0.0, "is_significant": False}

        # Convert annualized SR to period SR for standard error calculation
        sr_period = observed_sr / math.sqrt(annualization_factor)

        # Higher moments of return distribution
        gamma_3 = float(skew(returns))
        gamma_4 = float(kurtosis(returns, fisher=False))  # Pearson kurtosis (normal = 3)

        # Expected maximum Sharpe benchmark under the null hypothesis
        sr_benchmark_period = cls.expected_max_sharpe(n_trials) / math.sqrt(annualization_factor)
        sr_benchmark_annual = sr_benchmark_period * math.sqrt(annualization_factor)

        # Standard error of Sharpe ratio under non-normality (Mertens 2002 / Lo 2002)
        var_sr = 1.0 - gamma_3 * sr_period + ((gamma_4 - 1.0) / 4.0) * (sr_period ** 2)
        se_sr = math.sqrt(max(1e-8, var_sr) / (t - 1.0))

        # DSR = Phi((SR* - SR_0) / SE(SR))
        z_stat = (sr_period - sr_benchmark_period) / se_sr
        dsr_val = float(norm.cdf(z_stat))

        return {
            "observed_annual_sharpe": float(observed_sr),
            "benchmark_annual_sharpe": float(sr_benchmark_annual),
            "effective_trials_n_eff": float(n_trials),
            "sample_size_t": int(t),
            "skewness": gamma_3,
            "kurtosis": gamma_4,
            "dsr": dsr_val,
            "is_significant": bool(dsr_val >= 0.95),
        }
