"""
Advanced Microstructure Feature Engineering.

Implements four information-rich microstructure signals that capture
order-book dynamics destroyed by OHLCV aggregation:

  1. OFI    — Order Flow Imbalance (from ofi.py)
  2. Kyle λ — Price impact per unit signed volume (market depth proxy)
  3. VPIN   — Volume-Synchronized Probability of Informed Trading
  4. RV-of-RV — Realized Volatility of Realized Volatility (vol-of-vol)

These are the inputs that carry genuine informational content beyond what
OHLCV bars can represent. Used as the v3 tech feature set.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Kyle's Lambda Estimator
# ---------------------------------------------------------------------------

class KyleLambdaEstimator:
    """
    Estimates Kyle's λ (price impact coefficient) from OHLCV bars.

    Kyle's λ is defined as:
        λ = Cov(ΔP, Q_signed) / Var(Q_signed)

    where Q_signed = signed volume (positive for buy bars, negative for sell bars).
    We infer trade direction using the Tick Rule: bar is a "buy" if close > open,
    else "sell". This is a bar-level approximation; tick-level is more precise
    but tick data is not always available.

    Higher λ → less market depth, institutional footprint more visible.
    Lower λ  → deep liquidity, harder to detect informed flow.
    """

    @staticmethod
    def infer_signed_volume(bars_df: pd.DataFrame) -> pd.Series:
        """Tick-rule signed volume: positive for uptick bars, negative for downtick bars."""
        sign = np.where(bars_df["close"] >= bars_df["open"], 1.0, -1.0)
        return pd.Series(sign * bars_df["tick_volume"].values, index=bars_df.index, name="signed_vol")

    @classmethod
    def compute(
        cls,
        bars_df: pd.DataFrame,
        window: int = 50,
        min_periods: int = 10,
    ) -> pd.Series:
        """
        Rolling Kyle's λ estimate over a lookback window.

        Args:
            bars_df: OHLCV DataFrame with 'open', 'close', 'tick_volume'.
            window: Rolling lookback in bars.
            min_periods: Minimum observations before producing a value.

        Returns:
            pd.Series of Kyle λ values (price impact per unit volume).
        """
        close = bars_df["close"].astype(float)
        delta_p = close.diff().fillna(0.0)
        q_signed = cls.infer_signed_volume(bars_df)

        # Rolling covariance and variance
        roll_cov = delta_p.rolling(window, min_periods=min_periods).cov(q_signed)
        roll_var = q_signed.rolling(window, min_periods=min_periods).var() + 1e-10

        lambda_est = (roll_cov / roll_var).fillna(0.0)
        return lambda_est.rename("kyle_lambda")


# ---------------------------------------------------------------------------
# 2. VPIN Estimator
# ---------------------------------------------------------------------------

class VPINEstimator:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).

    VPIN measures the imbalance between buy and sell volume across equal-volume
    buckets. High VPIN precedes large adverse price moves (Easley et al., 2012).

    Bar-level approximation: each bar is assigned a buy/sell volume fraction
    using the bulk classification method (Abad & Yagüe, 2012):
        V_buy = V * Z( (close - open) / sigma )
        V_sell = V - V_buy
    where Z is the standard normal CDF.
    """

    @staticmethod
    def _bulk_classify(bars_df: pd.DataFrame, vol_col: str = "tick_volume") -> Tuple[pd.Series, pd.Series]:
        """Returns (V_buy, V_sell) series using bulk volume classification."""
        close = bars_df["close"].astype(float)
        open_ = bars_df["open"].astype(float)
        volume = bars_df[vol_col].astype(float)

        price_move = (close - open_)
        sigma = price_move.rolling(20, min_periods=5).std() + 1e-6

        from scipy.special import ndtr
        z = (price_move / sigma).fillna(0.0)
        frac_buy = pd.Series(ndtr(z), index=bars_df.index)

        v_buy = volume * frac_buy
        v_sell = volume * (1.0 - frac_buy)
        return v_buy, v_sell

    @classmethod
    def compute(
        cls,
        bars_df: pd.DataFrame,
        n_buckets: int = 50,
        vol_col: str = "tick_volume",
    ) -> pd.Series:
        """
        Computes rolling VPIN over a lookback of n_buckets equal-volume buckets.

        Args:
            bars_df: OHLCV DataFrame.
            n_buckets: Number of volume buckets for the rolling VPIN window.
            vol_col: Volume column name.

        Returns:
            pd.Series of VPIN values in [0, 1]; higher = more informed trading.
        """
        try:
            v_buy, v_sell = cls._bulk_classify(bars_df, vol_col)
        except ImportError:
            # Fallback if scipy unavailable: tick-rule approximation
            sign = (bars_df["close"] >= bars_df["open"]).astype(float)
            volume = bars_df[vol_col].astype(float)
            v_buy = volume * sign
            v_sell = volume * (1.0 - sign)

        total_vol = v_buy + v_sell
        imbalance = (v_buy - v_sell).abs()

        # Rolling sum over n_buckets bars
        roll_imbalance = imbalance.rolling(n_buckets, min_periods=max(5, n_buckets // 5)).sum()
        roll_total = total_vol.rolling(n_buckets, min_periods=max(5, n_buckets // 5)).sum() + 1e-8

        vpin = (roll_imbalance / roll_total).clip(0.0, 1.0).fillna(0.5)
        return vpin.rename("vpin")


# ---------------------------------------------------------------------------
# 3. Realized Volatility-of-Volatility Estimator
# ---------------------------------------------------------------------------

class RealizedVolOfVolEstimator:
    """
    Realized Volatility of Realized Volatility (RV-of-RV).

    Captures the second-order uncertainty in market volatility:
      1. Compute 5-bar realized variance RV_t = sum(r_i^2) for i in window.
      2. Compute rolling std of RV_t over a longer outer window.

    High RV-of-RV → regime-transition risk, non-stationary vol regimes.
    Low RV-of-RV  → stable vol environment, models can rely on mean-reversion.
    """

    @classmethod
    def compute(
        cls,
        bars_df: pd.DataFrame,
        inner_window: int = 5,
        outer_window: int = 50,
        min_periods: int = 10,
    ) -> pd.Series:
        """
        Computes realized volatility-of-volatility.

        Args:
            bars_df: OHLCV DataFrame with 'close'.
            inner_window: Window for computing realized variance.
            outer_window: Window for computing std of realized variance.
            min_periods: Minimum observations for outer window.

        Returns:
            pd.Series of RV-of-RV values (non-negative, higher = more unstable vol).
        """
        close = bars_df["close"].astype(float)
        log_ret = np.log(close / close.shift(1)).fillna(0.0)

        # Realized variance: sum of squared returns in rolling inner window
        rv = (log_ret ** 2).rolling(inner_window, min_periods=max(2, inner_window // 2)).sum()

        # Realized vol-of-vol: rolling std of realized variance
        rv_of_rv = rv.rolling(outer_window, min_periods=min_periods).std().fillna(0.0)

        # Normalize to be scale-invariant: divide by rolling mean of RV
        rv_mean = rv.rolling(outer_window, min_periods=min_periods).mean() + 1e-12
        rv_of_rv_normalized = (rv_of_rv / rv_mean).fillna(0.0)

        return rv_of_rv_normalized.rename("rv_of_rv")


# ---------------------------------------------------------------------------
# 4. Unified Microstructure Feature Builder
# ---------------------------------------------------------------------------

class MicrostructureFeatureSet:
    """
    Builds the complete set of 4 advanced microstructure features and
    returns them as an aligned DataFrame suitable for the v3 tech feature tensor.

    Features:
      - normalized_ofi : Order Flow Imbalance (bar-level CLV*VolMom proxy)
      - kyle_lambda    : Price impact coefficient (market depth)
      - vpin           : Probability of Informed Trading
      - rv_of_rv       : Volatility-of-Volatility

    All outputs are rolling-computed from OHLCV bars alone (no L2 order book required).
    """

    def __init__(
        self,
        kyle_window: int = 50,
        vpin_buckets: int = 50,
        rv_inner: int = 5,
        rv_outer: int = 50,
    ):
        self.kyle_window = kyle_window
        self.vpin_buckets = vpin_buckets
        self.rv_inner = rv_inner
        self.rv_outer = rv_outer

    def build(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all 4 microstructure features from OHLCV bars.

        Args:
            bars_df: DataFrame with columns [open, high, low, close, tick_volume].

        Returns:
            DataFrame with columns: [normalized_ofi, kyle_lambda, vpin, rv_of_rv].
            All features are z-score normalized to zero mean and unit variance.
        """
        required = {"open", "high", "low", "close", "tick_volume"}
        missing = required - set(bars_df.columns)
        if missing:
            raise ValueError(f"MicrostructureFeatureSet.build: missing columns {missing}")

        out = pd.DataFrame(index=bars_df.index)

        # 1. OFI proxy: CLV * Volume Momentum (bar-level, no tick data needed)
        high = bars_df["high"].astype(float)
        low = bars_df["low"].astype(float)
        close = bars_df["close"].astype(float)
        volume = bars_df["tick_volume"].astype(float)
        vol_mom = volume / (volume.rolling(20, min_periods=5).mean() + 1e-6)
        hl_range = high - low
        clv = np.where(hl_range > 1e-6, (2.0 * close - low - high) / (hl_range + 1e-6), 0.0)
        out["normalized_ofi"] = clv * vol_mom

        # 2. Kyle's Lambda
        out["kyle_lambda"] = KyleLambdaEstimator.compute(bars_df, window=self.kyle_window)

        # 3. VPIN
        out["vpin"] = VPINEstimator.compute(bars_df, n_buckets=self.vpin_buckets)

        # 4. Realized Vol-of-Vol
        out["rv_of_rv"] = RealizedVolOfVolEstimator.compute(
            bars_df, inner_window=self.rv_inner, outer_window=self.rv_outer
        )

        # Z-score normalize each feature independently
        out = self._zscore_normalize(out)

        return out

    @staticmethod
    def _zscore_normalize(df: pd.DataFrame, window: int = 500, min_periods: int = 50) -> pd.DataFrame:
        """Rolling z-score normalization to keep features stationary."""
        result = df.copy()
        for col in df.columns:
            roll_mean = df[col].rolling(window, min_periods=min_periods).mean().bfill().fillna(0.0)
            roll_std = df[col].rolling(window, min_periods=min_periods).std().bfill().fillna(1.0) + 1e-8
            result[col] = ((df[col] - roll_mean) / roll_std).clip(-5.0, 5.0)
        return result


MICROSTRUCTURE_COLS = ["normalized_ofi", "kyle_lambda", "vpin", "rv_of_rv"]
