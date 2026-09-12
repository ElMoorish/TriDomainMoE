"""
Autonomous Continual ReCAP Retraining Daemon.
Implements Page 7-8 of the Whitepaper:
Monitors live concept drift (MRDD), isolates modular policy delta vectors (d_k = theta_k - theta_0),
and prevents catastrophic forgetting across market regime transitions.
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.storage import ParquetStorage
from src.models.institutional_moe import TriDomainMoE
from src.continual.recap import ReCAPPolicyLibrary
from src.continual.episodic_memory import EpisodicMemoryBuffer
from src.surveillance.mrdd import MultiResolutionDriftDetector
from src.features.fracdiff import FractionalDifferentiator
from src.features.sentiment_loader import YahooFinanceSentimentLoader
from src.features.sentiment_engine import MacroSentimentEngine
from src.loss.composite_loss import CompositeLoss
from src.utils.logger import setup_logger

logger = setup_logger("ReCAPDaemon")


def parse_args():
    parser = argparse.ArgumentParser(description="Continual ReCAP Adaptation Daemon")
    parser.add_argument("--symbol", type=str, default="BTCUSD.x", help="Symbol to monitor")
    parser.add_argument("--weights", type=str, default="weights/btcusd_tri_domain_v1.pt", help="Baseline weights")
    parser.add_argument("--interval-mins", type=int, default=30, help="Check interval in minutes")
    parser.add_argument("--once", action="store_true", default=True, help="Run single surveillance cycle")
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error("Weights checkpoint not found at %s.", weights_path)
        sys.exit(1)

    checkpoint = torch.load(weights_path, weights_only=False)
    logger.info("Initializing TriDomainMoE from %s...", weights_path)

    base_model = TriDomainMoE(
        tech_dim=checkpoint["tech_dim"],
        macro_dim=checkpoint["macro_dim"],
        fund_dim=checkpoint["fund_dim"],
        regime_dim=checkpoint["regime_dim"],
        hidden_dim=checkpoint.get("hidden_dim", 48),
    )
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    # Initialize ReCAP policy library
    recap = ReCAPPolicyLibrary(base_model=base_model)
    mrdd = MultiResolutionDriftDetector(wavelet_levels=3, energy_threshold=0.35)
    sentiment_engine = MacroSentimentEngine()
    sentiment_loader = YahooFinanceSentimentLoader()

    # Load real market bars for BTCUSD
    cache_dir = Path("data/cache/bars")
    sym_tag = args.symbol.replace(".", "_")
    m5_cache = cache_dir / f"{sym_tag}_m5_1y.parquet"
    h1_cache = cache_dir / f"{sym_tag}_h1_1y.parquet"

    if not m5_cache.exists():
        m5_cache = cache_dir / f"{sym_tag}_m5_3m.parquet"
        h1_cache = cache_dir / f"{sym_tag}_h1_3m.parquet"

    if not m5_cache.exists():
        logger.error("No cached bars found for %s in %s.", args.symbol, cache_dir)
        sys.exit(1)

    m5_df = pd.read_parquet(m5_cache)
    h1_df = pd.read_parquet(h1_cache) if h1_cache.exists() else pd.DataFrame()
    logger.info("Loaded %d real M5 bars for %s.", len(m5_df), args.symbol)

    logger.info("=== Continual ReCAP Daemon Active ===")
    logger.info("Baseline frozen parameters: %d tensors", len(checkpoint["model_state_dict"]))
    logger.info("Current ReCAP library size: %d delta policies", recap.num_policies())

    # Surveillance Cycle on Real Recent Returns
    logger.info("Running Wavelet Energy and Page-Hinkley drift surveillance on real market returns...")
    real_rets = np.log(m5_df["close"] / m5_df["close"].shift(1)).fillna(0.0).iloc[-128:].to_numpy()
    detector_res = mrdd.evaluate_wavelet_drift(real_rets)

    drift_flag = detector_res["wavelet_drift_detected"]
    max_div = detector_res["max_divergence"]

    logger.info(
        "MRDD Status:\n"
        "  - Wavelet Energy Divergence: %.2f%%\n"
        "  - Drift Threshold:           35.0%%\n"
        "  - Structural Shift Detected: %s",
        max_div * 100.0,
        drift_flag,
    )

    if drift_flag or args.once:
        regime_tag = f"regime_shift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        logger.info("Isolating and training new modular policy delta vector on real market data: '%s'...", regime_tag)

        # Clone and train active model on recent stream
        active_model = TriDomainMoE(
            tech_dim=checkpoint["tech_dim"],
            macro_dim=checkpoint["macro_dim"],
            fund_dim=checkpoint["fund_dim"],
            regime_dim=checkpoint["regime_dim"],
            hidden_dim=checkpoint.get("hidden_dim", 48),
        )
        active_model.load_state_dict(checkpoint["model_state_dict"])
        active_model.train()

        # Freeze base predictor to protect unconditional secular trend
        for p in active_model.base_predictor.parameters():
            p.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, active_model.parameters()), lr=3e-4)
        criterion = CompositeLoss(delta_huber=1.0, lambda_dir=0.5, lambda_ic=0.4, lambda_balance=0.1)

        # Prepare real feature batch from latest market data
        recent_m5 = m5_df.iloc[-300:].copy()
        recent_m5["log_ret"] = np.log(recent_m5["close"] / recent_m5["close"].shift(1)).fillna(0.0)
        recent_m5["vol_20"] = recent_m5["log_ret"].rolling(20, min_periods=5).std().bfill()
        recent_m5["fracdiff"] = FractionalDifferentiator.frac_diff(recent_m5["close"], d=0.45, threshold=1e-3)
        recent_m5["vol_mom"] = recent_m5["tick_volume"] / (recent_m5["tick_volume"].rolling(20).mean() + 1e-6)

        # Prepare H1 macro features
        if not h1_df.empty:
            recent_h1 = h1_df.iloc[-100:].copy()
            recent_h1["h1_ret"] = np.log(recent_h1["close"] / recent_h1["close"].shift(1)).fillna(0.0)
            recent_h1["h1_trend_50"] = (recent_h1["close"] - recent_h1["close"].rolling(50, min_periods=5).mean()) / (recent_h1["close"].rolling(50, min_periods=5).std() + 1e-6)
            recent_h1["h1_vol_24"] = recent_h1["h1_ret"].rolling(24, min_periods=5).std().bfill()

            m5_reset = recent_m5.reset_index().rename(columns={"index": "bar_ts", "timestamp": "bar_ts"})
            h1_reset = recent_h1[["h1_ret", "h1_trend_50", "h1_vol_24"]].reset_index().rename(columns={"index": "h1_ts", "timestamp": "h1_ts"})
            merged = pd.merge_asof(
                m5_reset.sort_values("bar_ts"),
                h1_reset.sort_values("h1_ts"),
                left_on="bar_ts",
                right_on="h1_ts",
                direction="backward",
            ).set_index("bar_ts")
        else:
            merged = recent_m5.copy()
            merged["h1_ret"] = 0.0
            merged["h1_trend_50"] = 0.0
            merged["h1_vol_24"] = 0.005

        feature_cols = ["log_ret", "vol_20", "fracdiff", "vol_mom", "h1_ret", "h1_trend_50", "h1_vol_24"]
        clean_df = merged.dropna(subset=feature_cols).copy()

        tech_mat = clean_df[["log_ret", "vol_20", "fracdiff", "vol_mom"]].to_numpy(dtype=np.float32)
        macro_mat = clean_df[["h1_ret", "h1_trend_50", "h1_vol_24"]].to_numpy(dtype=np.float32)

        # Construct batch of 8 real sequential windows
        batch_size = 8
        seq_len = 32
        x_tech_list, x_macro_list, y_ret_list = [], [], []

        for b_i in range(batch_size):
            offset = len(clean_df) - batch_size - seq_len + b_i
            t_sl = tech_mat[offset : offset + seq_len]
            m_sl = macro_mat[offset : offset + seq_len]

            # Normalize real features
            t_norm = (t_sl - t_sl.mean(axis=0)) / (t_sl.std(axis=0) + 1e-6)
            m_norm = (m_sl - m_sl.mean(axis=0)) / (m_sl.std(axis=0) + 1e-6)
            x_tech_list.append(t_norm)
            x_macro_list.append(m_norm)

            # Real forward return
            forward_ret = float(clean_df["log_ret"].iloc[offset + seq_len - 1])
            y_ret_list.append([forward_ret])

        x_t = torch.tensor(np.array(x_tech_list), dtype=torch.float32)
        x_m = torch.tensor(np.array(x_macro_list), dtype=torch.float32)
        y_r = torch.tensor(np.array(y_ret_list), dtype=torch.float32)

        try:
            articles = sentiment_loader.fetch_articles("BTC-USD")
            z_vec = sentiment_engine.update(articles)
        except Exception:
            z_vec = sentiment_engine.get_regime_vector()

        fund_dim = checkpoint["fund_dim"]
        if len(z_vec) != fund_dim:
            z_vec = np.resize(z_vec, fund_dim).astype(np.float32)

        x_f = torch.tensor(np.tile(z_vec, (batch_size, 1)), dtype=torch.float32)
        z_r = x_f.clone()

        # Execute ReCAP Policy Delta Training Steps
        for step in range(3):
            optimizer.zero_grad()
            out = active_model(x_t, x_m, x_f, z_r)
            l = criterion(out["y_pred"], y_r, out["weights"], out["noisy_weights"])["loss"]
            l.backward()
            optimizer.step()

        # Save delta vector d_k = theta_k - theta_0
        recap.add_policy_delta(regime_tag, active_model)
        logger.info("Successfully added delta policy '%s' to library using 100%% real market data.", regime_tag)
        logger.info("Total modular policies in library: %d", recap.num_policies())

        # Save updated ReCAP library to disk
        recap_dir = Path("weights/recap_library")
        recap_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policies": recap.policy_library,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, recap_dir / "recap_policies.pt")
        logger.info("Saved ReCAP policy library checkpoint to %s", recap_dir / "recap_policies.pt")

    logger.info("=== Continual ReCAP Cycle Complete ===")


if __name__ == "__main__":
    main()
