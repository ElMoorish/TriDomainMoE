"""
Cross-Asset Feature Engineering and Structural Term Dynamics.
Computes Variance Risk Premium (VRP), Commodity Roll Yields, Synthetic DXY,
and aligns multi-asset feature tensors across Equities, Metals, Energy, and FX.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class CrossAssetFeatureEngine:
    """
    Constructs cross-asset macroeconomic and term structure features
    to power the Macro & Term Structure Expert and Router Regime Vector z_t.
    """

    @staticmethod
    def compute_roll_yield(
        front_month_price: pd.Series,
        second_month_price: pd.Series,
        days_to_expiration_delta: float = 30.0,
    ) -> pd.Series:
        """
        Annualized roll yield: ln(F_1 / F_2) * (365 / Delta t)
        Positive -> Backwardation (inventory scarcity, positive convenience yield).
        Negative -> Contango (inventory surplus, storage costs).
        """
        eps = 1e-6
        ratio = (front_month_price + eps) / (second_month_price + eps)
        annual_factor = 365.0 / max(days_to_expiration_delta, 1.0)
        roll_yield = np.log(ratio) * annual_factor
        return pd.Series(roll_yield, index=front_month_price.index, name="roll_yield")

    @staticmethod
    def compute_variance_risk_premium(
        realized_vol_series: pd.Series,
        implied_vol_series: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Variance Risk Premium (VRP) = Implied Vol - Realized Vol.
        Signals systemic hedging demand and equity crash risk.
        If explicit implied vol series is not supplied, approximates via Parkinson/Garman-Klass or trailing vol z-score.
        """
        if implied_vol_series is not None:
            aligned_iv = implied_vol_series.reindex(realized_vol_series.index).ffill()
            vrp = aligned_iv - realized_vol_series
        else:
            # Synthetic proxy: trailing rolling 90th percentile vs current realized vol
            rolling_high_vol = realized_vol_series.rolling(window=100, min_periods=20).quantile(0.90)
            vrp = rolling_high_vol - realized_vol_series

        return pd.Series(vrp.fillna(0.0), index=realized_vol_series.index, name="vrp")

    @staticmethod
    def compute_synthetic_dxy(
        eurusd: pd.Series,
        gbpusd: Optional[pd.Series] = None,
        usdjpy: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Synthetic US Dollar Index (DXY) proxy.
        EUR/USD comprises 57.6% of the actual DXY basket.
        """
        # Inverse EURUSD is the primary dollar driver: DXY ~ 1 / EURUSD
        dxy_proxy = 1.0 / (eurusd + 1e-8)
        if gbpusd is not None and usdjpy is not None:
            # Weighted geometric basket approximation:
            # DXY ~ 50.14348112 * (EURUSD)^-0.576 * (USDJPY)^0.136 * (GBPUSD)^-0.119
            aligned_gbp = gbpusd.reindex(eurusd.index).ffill()
            aligned_jpy = usdjpy.reindex(eurusd.index).ffill()
            dxy_proxy = (
                (eurusd ** -0.576)
                * (aligned_jpy ** 0.136)
                * (aligned_gbp ** -0.119)
            )

        # Normalize to base 100
        dxy_normalized = (dxy_proxy / dxy_proxy.iloc[0]) * 100.0
        return pd.Series(dxy_normalized, index=eurusd.index, name="synthetic_dxy")

    @classmethod
    def build_cross_asset_matrix(
        cls,
        equity_bars: pd.DataFrame,
        gold_bars: pd.DataFrame,
        oil_bars: pd.DataFrame,
        fx_bars: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Synchronizes timestamps across Equities (NAS100), Metals (XAUUSD),
        Energy (WTI), and FX (EURUSD) into an aligned multi-asset feature table.
        """
        def get_vol_col(df):
            return "volume" if "volume" in df.columns else ("tick_volume" if "tick_volume" in df.columns else df.columns[0])

        dfs = {
            "equity": equity_bars[["close", get_vol_col(equity_bars)]].rename(columns={"close": "equity_close", get_vol_col(equity_bars): "equity_vol"}),
            "gold": gold_bars[["close", get_vol_col(gold_bars)]].rename(columns={"close": "gold_close", get_vol_col(gold_bars): "gold_vol"}),
            "oil": oil_bars[["close", get_vol_col(oil_bars)]].rename(columns={"close": "oil_close", get_vol_col(oil_bars): "oil_vol"}),
            "fx": fx_bars[["close"]].rename(columns={"close": "eurusd_close"}),
        }

        # Outer join on timestamps then forward fill
        aligned = pd.concat(dfs.values(), axis=1).sort_index().ffill().dropna()

        # Compute log returns
        for asset in ["equity", "gold", "oil", "eurusd"]:
            col = f"{asset}_close"
            if col in aligned.columns:
                aligned[f"{asset}_ret"] = np.log(aligned[col] / aligned[col].shift(1)).fillna(0.0)

        # 1. Equity Variance Risk Premium
        eq_vol = aligned["equity_ret"].rolling(30, min_periods=5).std().bfill()
        aligned["equity_vrp"] = cls.compute_variance_risk_premium(eq_vol)

        # 2. Commodity Term / Roll Yield proxy: Gold vs Oil spread dynamics
        # Gold/Oil ratio serves as a classic macro risk & inflation indicator
        gold_oil_ratio = aligned["gold_close"] / (aligned["oil_close"] + 1e-6)
        aligned["gold_oil_ratio"] = (gold_oil_ratio - gold_oil_ratio.mean()) / (gold_oil_ratio.std() + 1e-6)

        # 3. Synthetic Dollar Index
        aligned["dxy"] = cls.compute_synthetic_dxy(aligned["eurusd_close"])
        aligned["dxy_ret"] = np.log(aligned["dxy"] / aligned["dxy"].shift(1)).fillna(0.0)

        # 4. Safe Haven Flow: Gold return minus Equity return
        aligned["safe_haven_flow"] = aligned["gold_ret"] - aligned["equity_ret"]

        return aligned.dropna()
