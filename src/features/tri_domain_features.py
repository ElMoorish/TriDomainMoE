"""
Synchronized Tri-Domain Feature Engineering Pipeline for BTCUSD.
Guarantees strict parity across Training, Backtesting, Continual Learning, and Live Trading.
Supports:
  - v1: 4 Tech, 3 Macro, 6 Fund/Regime
  - v2: 6 Tech (Parkinson High-Low Vol, Bar OFI), 6 Macro (H4/D1 Secular Trend, Vol Term Slope), 8 Fund/Regime (CVD Flow, Basis Proxy)
  - v3: 10 Tech (v2 + 4 Microstructure: Kyle λ, VPIN, RV-of-RV, normalized OFI), same Macro/Fund streams
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd

from src.features.fracdiff import FractionalDifferentiator
from src.features.sentiment_engine import MacroSentimentEngine
from src.features.microstructure_features import MicrostructureFeatureSet, MICROSTRUCTURE_COLS


TECH_COLS_V1 = ["log_ret", "vol_20", "fracdiff", "vol_mom"]
TECH_COLS_V2 = ["log_ret", "vol_20", "fracdiff", "vol_mom", "parkinson_ratio", "bar_ofi"]
# v3: v2 cols + 4 microstructure signals (OFI already in MICROSTRUCTURE_COLS as normalized_ofi)
TECH_COLS_V3 = ["log_ret", "vol_20", "fracdiff", "vol_mom", "parkinson_ratio"] + MICROSTRUCTURE_COLS

MACRO_COLS_V1 = ["h1_ret", "h1_trend_50", "h1_vol_24"]
MACRO_COLS_V2 = ["h1_ret", "h1_trend_50", "h1_vol_24", "h4_trend", "d1_trend", "vol_term_slope"]
MACRO_COLS_V3 = MACRO_COLS_V2  # Macro stream unchanged in v3


def compute_technical_features(df: pd.DataFrame, version: str = "v2") -> pd.DataFrame:
    """Computes technical microstructure features for M5/M1 bars."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    tick_vol = out["tick_volume"]

    # Base features
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    vol_20 = log_ret.rolling(20, min_periods=5).std().bfill().fillna(0.001)
    fracdiff = FractionalDifferentiator.frac_diff(close, d=0.45, threshold=1e-3)
    vol_mom = tick_vol / (tick_vol.rolling(20, min_periods=5).mean() + 1e-6)

    out["log_ret"] = log_ret
    out["vol_20"] = vol_20
    out["fracdiff"] = fracdiff
    out["vol_mom"] = vol_mom

    if version in ("v2", "v3"):
        # Parkinson High-Low Volatility vs Close-to-Close Volatility
        hl_ratio = np.log(np.maximum(high / np.maximum(low, 1e-6), 1.0))
        parkinson_var = (hl_ratio ** 2) / (4.0 * np.log(2.0))
        parkinson_vol = np.sqrt(np.maximum(parkinson_var, 0.0))
        out["parkinson_ratio"] = parkinson_vol / (vol_20 + 1e-6)

        if version == "v2":
            # Bar Order Flow Imbalance (Close Location Value * Volume Momentum)
            hl_range = high - low
            clv = np.where(hl_range > 1e-6, (2.0 * close - low - high) / (hl_range + 1e-6), 0.0)
            out["bar_ofi"] = clv * vol_mom

    if version == "v3":
        # Advanced microstructure signals: Kyle λ, VPIN, RV-of-RV, normalized OFI
        ms = MicrostructureFeatureSet(kyle_window=50, vpin_buckets=50, rv_inner=5, rv_outer=50)
        ms_df = ms.build(out)  # Returns DataFrame with MICROSTRUCTURE_COLS
        for col in MICROSTRUCTURE_COLS:
            out[col] = ms_df[col]

    return out


def compute_macro_features(h1_df: pd.DataFrame, version: str = "v2") -> pd.DataFrame:
    """Computes macroeconomic term structure features for H1 bars."""
    out = h1_df.copy()
    close = out["close"]

    h1_ret = np.log(close / close.shift(1)).fillna(0.0)
    rolling_mean_50 = close.rolling(50, min_periods=10).mean()
    rolling_std_50 = close.rolling(50, min_periods=10).std() + 1e-6
    h1_trend_50 = (close - rolling_mean_50) / rolling_std_50
    h1_vol_24 = h1_ret.rolling(24, min_periods=5).std().bfill().fillna(0.004)

    out["h1_ret"] = h1_ret
    out["h1_trend_50"] = h1_trend_50
    out["h1_vol_24"] = h1_vol_24

    if version in ("v2", "v3"):
        # H4 Swing Trend (200 H1 bars ~ 8.3 days)
        m_200 = close.rolling(200, min_periods=20).mean()
        s_200 = close.rolling(200, min_periods=20).std() + 1e-6
        out["h4_trend"] = (close - m_200) / s_200

        # D1 Secular Trend (600 H1 bars ~ 25 days)
        m_600 = close.rolling(600, min_periods=50).mean()
        s_600 = close.rolling(600, min_periods=50).std() + 1e-6
        out["d1_trend"] = (close - m_600) / s_600

        # Volatility Term Structure Slope (24h vs 168h 7-day volatility)
        vol_168 = h1_ret.rolling(168, min_periods=24).std().bfill().fillna(0.005)
        out["vol_term_slope"] = h1_vol_24 / (vol_168 + 1e-6)

    return out


def build_synchronized_features(
    bars_df: pd.DataFrame,
    h1_bars: pd.DataFrame,
    version: str = "v2",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Merges M5/M1 execution bars with H1 macro bars via backward merge_asof.
    Returns:
      - Synchronized DataFrame with technical and macro features
      - List of technical column names
      - List of macro column names
    """
    tech_df = compute_technical_features(bars_df, version=version)
    macro_df = compute_macro_features(h1_bars, version=version)

    if version == "v3":
        tech_cols = TECH_COLS_V3
        macro_cols = MACRO_COLS_V3
    elif version == "v2":
        tech_cols = TECH_COLS_V2
        macro_cols = MACRO_COLS_V2
    else:
        tech_cols = TECH_COLS_V1
        macro_cols = MACRO_COLS_V1

    bar_reset = tech_df.reset_index().rename(columns={"index": "bar_ts", "timestamp": "bar_ts"})
    h1_reset = macro_df[macro_cols].reset_index().rename(columns={"index": "h1_ts", "timestamp": "h1_ts"})

    merged = pd.merge_asof(
        bar_reset.sort_values("bar_ts"),
        h1_reset.sort_values("h1_ts"),
        left_on="bar_ts",
        right_on="h1_ts",
        direction="backward",
    ).set_index("bar_ts")

    # Add CVD and Basis Proxy for fundamental stream in v2/v3
    if version in ("v2", "v3"):
        # v2: uses bar_ofi; v3: uses normalized_ofi from microstructure module
        ofi_col = "bar_ofi" if version == "v2" else "normalized_ofi"
        if ofi_col in merged.columns:
            merged["cvd_flow"] = merged[ofi_col].rolling(288, min_periods=20).mean().fillna(0.0)
            spread_val = merged["spread"] if "spread" in merged.columns else 6500.0
            hl = merged["high"] - merged["low"]
            merged["spread_basis"] = ((hl / (spread_val * 0.01 + 1e-6)).rolling(20, min_periods=5).mean()).fillna(1.0)

    clean_df = merged.dropna(subset=tech_cols + macro_cols).copy()
    return clean_df, tech_cols, macro_cols
