"""
Combinatorial Purged Cross-Validation (CPCV) Evaluation Utility.
Evaluates out-of-sample performance distribution across combinatorial backtest paths
under strict purging of forward label overlaps and post-test embargoing.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.storage import ParquetStorage
from src.validation.cpcv import CombinatorialPurgedCV
from src.validation.deflated_sharpe import DeflatedSharpeRatio
from src.features.fracdiff import FractionalDifferentiator
from src.features.labeling import TripleBarrierLabeler
from src.models.rg_resmoe import RGResMoE
from src.loss.composite_loss import CompositeLoss
from src.utils.logger import setup_logger

logger = setup_logger("CPCVEvaluation")


def main():
    storage = ParquetStorage()
    bars_df = storage.load_bars("XAUUSD.x", "vol_bars")
    if bars_df.empty or len(bars_df) < 200:
        bars_df = storage.load_bars("XAUUSD.x", "m1")

    if bars_df.empty or len(bars_df) < 100:
        logger.error("No bar data found. Run scripts/download_mt5_data.py first.")
        sys.exit(1)

    logger.info("Initializing CPCV Evaluation over %d bars...", len(bars_df))

    # Basic feature prep
    df = bars_df.copy()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df["vol"] = df["log_ret"].rolling(20, min_periods=5).std().bfill()
    df["fracdiff"] = FractionalDifferentiator.frac_diff(df["close"], d=0.4)
    
    labeler = TripleBarrierLabeler(pt_multiplier=1.5, sl_multiplier=1.5, max_holding_bars=15)
    labeled = labeler.label_events(df["close"])
    df["target_ret"] = labeled["return"]
    df = df.dropna()

    cpcv = CombinatorialPurgedCV(n_blocks=6, k_test_blocks=2, embargo_pct=0.02)
    logger.info("Total Combinatorial Splits: %d, Distinct Backtest Paths: %d", 15, cpcv.num_paths())

    path_sharpes = []
    path_returns = []

    for split_idx, (train_idx, test_idx) in enumerate(cpcv.split(df, label_holding_bars=15)):
        train_sub = df.iloc[train_idx]
        test_sub = df.iloc[test_idx]

        if len(train_sub) < 50 or len(test_sub) < 20:
            continue

        # Target strategy return: simple momentum/trend sign
        # Benchmark model proxy for cross-validation paths
        pred_signal = np.sign(test_sub["log_ret"].shift(1).fillna(0.0))
        strat_ret = pred_signal * test_sub["target_ret"]

        mean_r = strat_ret.mean()
        std_r = strat_ret.std() + 1e-8
        ann_sr = (mean_r / std_r) * np.sqrt(252 * 390)

        path_sharpes.append(ann_sr)
        path_returns.append(strat_ret.to_numpy())

    logger.info("=== CPCV Results across %d Validated Splits ===", len(path_sharpes))
    logger.info("Mean Out-of-Sample Sharpe:   %.2f", float(np.mean(path_sharpes)))
    logger.info("Median Out-of-Sample Sharpe: %.2f", float(np.median(path_sharpes)))
    logger.info("Std Dev of Sharpe across Paths: %.2f", float(np.std(path_sharpes)))
    logger.info("Worst-Path Sharpe:          %.2f", float(np.min(path_sharpes)))
    logger.info("Best-Path Sharpe:           %.2f", float(np.max(path_sharpes)))

    # Compute Deflated Sharpe Ratio
    if path_returns:
        stacked_returns = np.concatenate(path_returns)
        dsr_info = DeflatedSharpeRatio.compute_dsr(
            observed_sr=float(np.max(path_sharpes)),
            returns=stacked_returns,
            n_trials=float(len(path_sharpes)),
            annualization_factor=252 * 390,
        )
        logger.info("Deflated Sharpe Ratio (DSR): %.4f (Significant: %s)", dsr_info["dsr"], dsr_info["is_significant"])


if __name__ == "__main__":
    main()
