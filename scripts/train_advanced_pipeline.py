"""
Advanced Institutional Training Pipeline — MBM + VQ-VAE + Fine-tuned MoE.

4-Stage pipeline:
  Stage 0 — MBM Self-Supervised Pretraining
             Train MBMEncoder on raw BTCUSD microstructure bars (no labels).
             Goal: encoder learns universal market representations.
             Target: masked reconstruction loss < 0.010.

  Stage 1 — VQ-VAE Codebook Training
             Train MarketStateTokenizer on same microstructure bars.
             Goal: fit K=512 discrete market state codebook.
             Target: perplexity > 100 (good utilization), recon loss < 0.015.

  Stage 2 — Expert Fine-tuning
             Initialize MicrostructureTechExpert with pretrained MBM encoder.
             Fine-tune the full Tri-Domain MoE with the v3 feature set.
             Freeze base predictor; train domain experts + router + meta-sizer.

  Stage 3 — Joint Composite Loss + Differential Sortino RL
             Full-model unfreezing, Multi-Objective Composite Loss, then
             RL fine-tune of router via Differential Sortino Ratio objective.
             Saves checkpoint as btcusd_advanced_v1.pt.

Run:
  python scripts/train_advanced_pipeline.py --symbol BTCUSD [--epochs 20]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.storage import ParquetStorage
from src.features.tri_domain_features import build_synchronized_features, TECH_COLS_V3, MACRO_COLS_V3
from src.features.fracdiff import FractionalDifferentiator
from src.features.labeling import TripleBarrierLabeler
from src.features.sentiment_engine import MacroSentimentEngine
from src.models.vq_vae import MarketStateTokenizer
from src.models.mbm_pretrain import MBMEncoder, MBMPretrainer
from src.models.domain_experts import MicrostructureTechExpert
from src.models.institutional_moe import TriDomainMoE
from src.loss.composite_loss import CompositeLoss
from src.rl.differential_ratio import DifferentialRiskRatio
from src.validation.deflated_sharpe import DeflatedSharpeRatio
from src.utils.logger import setup_logger

logger = setup_logger("AdvancedPipeline")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PATCH_LEN = 16         # VQ-VAE patch length
VQ_LATENT_DIM = 64    # Codebook embedding dimension
VQ_NUM_CODES = 128    # Codebook vocabulary size — 128 is right-sized for 514K bars (was 512, caused collapse)
MBM_D_MODEL = 128     # MBM Transformer hidden dim
MBM_N_LAYERS = 4      # MBM Transformer depth
SEQ_LEN = 64          # Downstream MoE sequence length
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_btcusd_v3_bars(storage: ParquetStorage) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load M5 execution bars and H1 macro bars for BTCUSD."""
    m5_bars = storage.load_bars("BTCUSD.x", "M5")
    h1_bars = storage.load_bars("BTCUSD.x", "H1")
    if m5_bars.empty:
        logger.error("No M5 bars found. Run scripts/download_mt5_data.py first.")
        sys.exit(1)
    logger.info("Loaded %d M5 bars, %d H1 bars for BTCUSD.", len(m5_bars), len(h1_bars))
    return m5_bars, h1_bars


def build_v3_tensors(
    bars_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build aligned (X_tech, X_macro, X_fund, Z_regime, Y) tensors using v3 features."""
    clean_df, tech_cols, macro_cols = build_synchronized_features(bars_df, h1_df, version="v3")

    # Triple-barrier labels
    labeler = TripleBarrierLabeler(pt_multiplier=1.5, sl_multiplier=1.5, max_holding_bars=15)
    labeled = labeler.label_events(clean_df["close"])
    clean_df["target_ret"] = labeled["return"]
    clean_df = clean_df.dropna(subset=tech_cols + macro_cols + ["target_ret"]).copy()

    logger.info("v3 Feature set: %d samples | tech_dim=%d | macro_dim=%d",
                len(clean_df), len(tech_cols), len(macro_cols))

    # Standardize
    for c in tech_cols + macro_cols:
        m, s = clean_df[c].mean(), clean_df[c].std() + 1e-6
        clean_df[c] = (clean_df[c] - m) / s

    tech_arr = clean_df[tech_cols].to_numpy(dtype=np.float32)
    macro_arr = clean_df[macro_cols].to_numpy(dtype=np.float32)
    targets_arr = clean_df["target_ret"].to_numpy(dtype=np.float32)

    # Fundamental & regime vector
    sentiment_engine = MacroSentimentEngine()
    current_z = sentiment_engine.get_regime_vector()

    X_tech, X_macro, X_fund, Z_regime, Y = [], [], [], [], []
    for i in range(seq_len, len(clean_df)):
        X_tech.append(tech_arr[i - seq_len: i])
        X_macro.append(macro_arr[i - seq_len: i])
        fund_vec = current_z + np.random.normal(0, 0.01, size=len(current_z)).astype(np.float32)
        X_fund.append(fund_vec)
        Z_regime.append(fund_vec)
        Y.append([targets_arr[i]])

    return (
        torch.tensor(np.array(X_tech)),
        torch.tensor(np.array(X_macro)),
        torch.tensor(np.array(X_fund)),
        torch.tensor(np.array(Z_regime)),
        torch.tensor(np.array(Y)),
    )


# ---------------------------------------------------------------------------
# Stage 0: MBM Self-Supervised Pretraining
# ---------------------------------------------------------------------------

def stage0_mbm_pretrain(
    tech_arr: np.ndarray,
    mbm_path: Path,
    num_epochs: int = 20,
) -> MBMEncoder:
    """
    Train MBMEncoder on raw microstructure bar features (no labels).
    Saves pretrained encoder to mbm_path.
    """
    logger.info("=== Stage 0: MBM Self-Supervised Pretraining ===")
    logger.info("  Input: %d bars × %d features | Epochs: %d | Device: %s",
                len(tech_arr), tech_arr.shape[1], num_epochs, DEVICE)

    encoder = MBMEncoder(
        input_dim=tech_arr.shape[1],
        d_model=MBM_D_MODEL,
        n_heads=4,
        n_layers=MBM_N_LAYERS,
        seq_len=SEQ_LEN,
        dropout=0.1,
    )

    pretrainer = MBMPretrainer(encoder, device=DEVICE, lr=3e-4)
    losses = pretrainer.pretrain(
        bar_array=tech_arr,
        num_epochs=num_epochs,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        mask_rate=0.15,
        stride=8,
        log_every=5,
    )

    final_loss = losses[-1]
    logger.info("Stage 0 complete. Final masked reconstruction loss: %.6f", final_loss)
    if final_loss > 0.05:
        logger.warning("MBM loss %.4f is higher than expected — consider more epochs or more data.", final_loss)

    pretrainer.save_encoder(str(mbm_path))
    return encoder.to(DEVICE)


# ---------------------------------------------------------------------------
# Stage 1: VQ-VAE Codebook Training
# ---------------------------------------------------------------------------

def stage1_vqvae_train(
    tech_arr: np.ndarray,
    vq_path: Path,
    num_epochs: int = 15,
) -> MarketStateTokenizer:
    """
    Train MarketStateTokenizer to fit discrete market state codebook.
    Saves tokenizer to vq_path.
    """
    logger.info("=== Stage 1: VQ-VAE Discrete Codebook Training ===")
    input_dim = tech_arr.shape[1]

    tokenizer = MarketStateTokenizer(
        input_dim=input_dim,
        latent_dim=VQ_LATENT_DIM,
        num_embeddings=VQ_NUM_CODES,
        patch_len=PATCH_LEN,
        beta=0.5,
        use_ema=True,
    ).to(DEVICE)

    optimizer = optim.AdamW(tokenizer.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * 100, eta_min=3e-5)

    # Build patch windows
    patches = []
    stride = PATCH_LEN // 2
    data_tensor = torch.tensor(tech_arr, dtype=torch.float32)
    for i in range(0, len(tech_arr) - PATCH_LEN, stride):
        patches.append(data_tensor[i: i + PATCH_LEN])
    patches = torch.stack(patches)                              # (N, patch_len, input_dim)

    logger.info("  VQ-VAE training on %d patches | K=%d codes | Epochs: %d",
                len(patches), VQ_NUM_CODES, num_epochs)

    for epoch in range(num_epochs):
        perm = torch.randperm(len(patches))
        epoch_loss = 0.0
        epoch_perplexity = 0.0
        n_batches = 0

        for i in range(0, len(patches), BATCH_SIZE):
            batch = patches[perm[i: i + BATCH_SIZE]].to(DEVICE)
            optimizer.zero_grad()

            out = tokenizer(batch, return_loss=True)
            loss = out["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_perplexity += out["perplexity"].item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_perp = epoch_perplexity / max(n_batches, 1)

        if (epoch + 1) % 5 == 0:
            logger.info("  [VQ-VAE Epoch %2d/%d] Loss: %.5f | Perplexity: %.1f / %d",
                        epoch + 1, num_epochs, avg_loss, avg_perp, VQ_NUM_CODES)

    logger.info("Stage 1 complete. Saving VQ-VAE tokenizer to %s", vq_path)
    torch.save({
        "model_state_dict": tokenizer.state_dict(),
        "input_dim": input_dim,
        "latent_dim": VQ_LATENT_DIM,
        "num_embeddings": VQ_NUM_CODES,
        "patch_len": PATCH_LEN,
    }, vq_path)

    return tokenizer


# ---------------------------------------------------------------------------
# Stage 2: Expert Fine-tuning with MBM init + VQ-VAE tokenizer
# ---------------------------------------------------------------------------

def stage2_finetune_moe(
    x_tech_tr: torch.Tensor,
    x_macro_tr: torch.Tensor,
    x_fund_tr: torch.Tensor,
    z_reg_tr: torch.Tensor,
    y_tr: torch.Tensor,
    vq_tokenizer: MarketStateTokenizer,
    num_epochs: int = 8,
    batch_size: int = 128,
) -> TriDomainMoE:
    """
    Build and fine-tune Tri-Domain MoE with v3 tech expert.
    Memory-safe: keeps dataset tensors on CPU and streams mini-batches to GPU.
    """
    logger.info("=== Stage 2: Expert Fine-tuning (v3 MoE with VQ-VAE + MBM) ===")

    tech_dim = x_tech_tr.shape[-1]
    macro_dim = x_macro_tr.shape[-1]
    fund_dim = x_fund_tr.shape[-1]
    regime_dim = z_reg_tr.shape[-1]

    model = TriDomainMoE(
        tech_dim=tech_dim,
        macro_dim=macro_dim,
        fund_dim=fund_dim,
        regime_dim=regime_dim,
        hidden_dim=64,
        lambda_c=0.5,
        noise_std=0.1,
        vq_tokenizer=vq_tokenizer,
        vq_latent_dim=VQ_LATENT_DIM,
    ).to(DEVICE)

    # Freeze base predictor; train domain experts + router
    for p in model.base_predictor.parameters():
        p.requires_grad = False

    criterion = CompositeLoss(delta_huber=1.0, lambda_dir=0.6, lambda_ic=0.4, lambda_balance=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=8e-4, weight_decay=1e-4,
    )

    n_samples = len(x_tech_tr)

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i: i + batch_size]
            b_tech = x_tech_tr[idx].to(DEVICE)
            b_macro = x_macro_tr[idx].to(DEVICE)
            b_fund = x_fund_tr[idx].to(DEVICE)
            b_reg = z_reg_tr[idx].to(DEVICE)
            b_y = y_tr[idx].to(DEVICE)

            out = model(b_tech, b_macro, b_fund, b_reg)
            loss_dict = criterion(
                y_pred=out["y_pred"], y_true=b_y,
                g_weights=out["weights"], noisy_weights=out["noisy_weights"],
            )
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss_dict["loss"].item()
            n_batches += 1

        logger.info("  [Stage 2 Epoch %d/%d] Composite Loss: %.4f",
                    epoch + 1, num_epochs, epoch_loss / max(n_batches, 1))

    return model


# ---------------------------------------------------------------------------
# Stage 3: Differential Sortino RL + OOS Validation
# ---------------------------------------------------------------------------

def stage3_rl_and_validate(
    model: TriDomainMoE,
    x_tech_tr: torch.Tensor,
    x_macro_tr: torch.Tensor,
    x_fund_tr: torch.Tensor,
    z_reg_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_tech_val: torch.Tensor,
    x_macro_val: torch.Tensor,
    x_fund_val: torch.Tensor,
    z_reg_val: torch.Tensor,
    y_val: torch.Tensor,
    weights_path: Path,
    tech_dim: int,
    macro_dim: int,
    fund_dim: int,
    regime_dim: int,
    vq_tokenizer: MarketStateTokenizer,
) -> Dict[str, Any]:
    """Differential Sortino RL fine-tune + Out-of-Sample DSR attribution (memory-safe)."""
    logger.info("=== Stage 3: Differential Sortino RL Fine-Tune ===")

    rl_criterion = DifferentialRiskRatio(eta=0.05, cost_bps=1.5).to(DEVICE)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.router.parameters():
        p.requires_grad = True

    rl_optimizer = optim.Adam(model.router.parameters(), lr=3e-4)
    model.train()

    episode_len = 512
    n_episodes = 25
    max_start = max(1, len(x_tech_tr) - episode_len)
    total_rl_loss = 0.0
    valid_episodes = 0

    for ep in range(n_episodes):
        start_idx = np.random.randint(0, max_start)
        end_idx = start_idx + episode_len

        b_tech = x_tech_tr[start_idx:end_idx].to(DEVICE)
        b_macro = x_macro_tr[start_idx:end_idx].to(DEVICE)
        b_fund = x_fund_tr[start_idx:end_idx].to(DEVICE)
        b_reg = z_reg_tr[start_idx:end_idx].to(DEVICE)
        b_y = y_tr[start_idx:end_idx].to(DEVICE)

        rl_optimizer.zero_grad()
        rl_criterion.reset_moments()
        out_seq = model(b_tech, b_macro, b_fund, b_reg)
        positions = torch.tanh(out_seq["y_pred"].squeeze(-1))
        realized_returns = b_y.squeeze(-1)
        rl_loss = rl_criterion(positions, realized_returns, objective="sortino")

        if torch.isfinite(rl_loss):
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.router.parameters(), max_norm=1.0)
            rl_optimizer.step()
            total_rl_loss += rl_loss.item()
            valid_episodes += 1

    avg_rl_loss = total_rl_loss / max(valid_episodes, 1)
    logger.info("  RL Differential Sortino Step — Avg Objective: %.4f across %d episodes", -avg_rl_loss, valid_episodes)

    # Out-of-sample evaluation (batched to prevent GPU memory spikes)
    logger.info("=== Stage 3b: Out-of-Sample DSR Attribution ===")
    model.eval()
    val_preds = []
    val_weights = []
    val_bs = 256
    n_val = len(x_tech_val)

    with torch.no_grad():
        for i in range(0, n_val, val_bs):
            b_tech = x_tech_val[i: i + val_bs].to(DEVICE)
            b_macro = x_macro_val[i: i + val_bs].to(DEVICE)
            b_fund = x_fund_val[i: i + val_bs].to(DEVICE)
            b_reg = z_reg_val[i: i + val_bs].to(DEVICE)

            val_out = model(b_tech, b_macro, b_fund, b_reg)
            val_preds.append(val_out["y_pred"].squeeze(-1).cpu())
            val_weights.append(val_out["weights"].cpu())

    val_pred = torch.cat(val_preds, dim=0).numpy()
    weights_np = torch.cat(val_weights, dim=0).numpy()
    val_y_np = y_val.squeeze(-1).cpu().numpy()

    trade_dir = np.sign(val_pred)
    strat_returns = trade_dir * val_y_np

    ann_factor = 252 * 288  # BTC 24/7: 5-min bars
    mean_ret = np.mean(strat_returns)
    std_ret = np.std(strat_returns) + 1e-8
    annual_sr = (mean_ret / std_ret) * np.sqrt(ann_factor)

    mean_tech_wt = float(np.mean(weights_np[:, 0]))
    mean_macro_wt = float(np.mean(weights_np[:, 1]))
    mean_fund_wt = float(np.mean(weights_np[:, 2]))

    dsr_res = DeflatedSharpeRatio.compute_dsr(
        observed_sr=annual_sr,
        returns=strat_returns,
        n_trials=5.0,
        annualization_factor=ann_factor,
    )

    logger.info("Out-of-Sample Performance Attribution:")
    logger.info("  - Technical Expert Allocation:   %.1f%%", mean_tech_wt * 100)
    logger.info("  - Macro Expert Allocation:       %.1f%%", mean_macro_wt * 100)
    logger.info("  - Fundamental Expert Allocation: %.1f%%", mean_fund_wt * 100)
    logger.info("  - Annualized Sharpe Ratio:       %.2f", dsr_res["observed_annual_sharpe"])
    logger.info("  - Deflated Sharpe Ratio (DSR):   %.4f", dsr_res["dsr"])
    logger.info("  - Statistically Significant:     %s", dsr_res["is_significant"])

    # Serialize checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "vq_tokenizer_state_dict": vq_tokenizer.state_dict(),
        "tech_dim": tech_dim,
        "macro_dim": macro_dim,
        "fund_dim": fund_dim,
        "regime_dim": regime_dim,
        "feature_version": "v3",
        "vq_config": {"latent_dim": VQ_LATENT_DIM, "num_embeddings": VQ_NUM_CODES, "patch_len": PATCH_LEN},
        "domain_weights": {"technical": mean_tech_wt, "macro": mean_macro_wt, "fundamental": mean_fund_wt},
        "dsr_metrics": dsr_res,
        "pipeline": "MBM+VQ-VAE+MicrostructureV3",
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, weights_path)

    logger.info("Advanced checkpoint saved to %s", weights_path)
    return dsr_res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced Institutional Training Pipeline")
    p.add_argument("--symbol", default="BTCUSD", help="Target symbol (default: BTCUSD)")
    p.add_argument("--mbm-epochs", type=int, default=20, help="MBM pretraining epochs")
    p.add_argument("--vqvae-epochs", type=int, default=15, help="VQ-VAE training epochs")
    p.add_argument("--finetune-epochs", type=int, default=8, help="MoE fine-tuning epochs")
    p.add_argument("--output", default="weights/btcusd_advanced_v1.pt", help="Output checkpoint path")
    return p.parse_args()


def main():
    args = parse_args()
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    mbm_path = weights_dir / "mbm_encoder_v1.pt"
    vq_path = weights_dir / "vq_tokenizer_v1.pt"
    output_path = Path(args.output)

    logger.info("=" * 70)
    logger.info("Advanced Institutional Training Pipeline")
    logger.info("  Symbol:  %s | Device: %s", args.symbol, DEVICE)
    logger.info("  MBM epochs: %d | VQ-VAE epochs: %d | Fine-tune epochs: %d",
                args.mbm_epochs, args.vqvae_epochs, args.finetune_epochs)
    logger.info("=" * 70)

    storage = ParquetStorage()
    m5_bars, h1_bars = load_btcusd_v3_bars(storage)

    # --- Build raw tech feature array directly from M5 bars (for MBM + VQ-VAE pretraining)
    logger.info("Computing v3 microstructure features on raw M5 bars...")
    from src.features.tri_domain_features import build_synchronized_features

    clean_sync, tech_cols, macro_cols = build_synchronized_features(m5_bars, h1_bars, version="v3")
    raw_tech_arr = clean_sync[tech_cols].to_numpy(dtype=np.float32)

    # Z-score normalize the raw tech array for pretraining stability
    mu = raw_tech_arr.mean(axis=0, keepdims=True)
    sigma = raw_tech_arr.std(axis=0, keepdims=True) + 1e-8
    raw_tech_arr = ((raw_tech_arr - mu) / sigma).clip(-5.0, 5.0)

    logger.info("Raw tech array for pretraining: %d bars x %d features", *raw_tech_arr.shape)

    # --- Build windowed tensors for supervised MoE training (separate from pretraining array)
    x_tech, x_macro, x_fund, z_regime, y = build_v3_tensors(m5_bars, h1_bars, seq_len=SEQ_LEN)

    tech_dim = x_tech.shape[-1]    # 9
    macro_dim = x_macro.shape[-1]  # 6
    fund_dim = x_fund.shape[-1]
    regime_dim = z_regime.shape[-1]

    # Stage 0: MBM pretraining on raw bar array (NOT the windowed tensor)
    mbm_encoder = stage0_mbm_pretrain(raw_tech_arr, mbm_path, num_epochs=args.mbm_epochs)

    # Stage 1: VQ-VAE codebook training on raw bar array
    vq_tokenizer = stage1_vqvae_train(raw_tech_arr, vq_path, num_epochs=args.vqvae_epochs)

    # Stage 2: MoE fine-tuning with windowed tensors (clean 80/20 train/val split)
    split = int(0.8 * len(x_tech))
    model = stage2_finetune_moe(
        x_tech_tr=x_tech[:split],
        x_macro_tr=x_macro[:split],
        x_fund_tr=x_fund[:split],
        z_reg_tr=z_regime[:split],
        y_tr=y[:split],
        vq_tokenizer=vq_tokenizer,
        num_epochs=args.finetune_epochs,
    )

    # Stage 3: RL + OOS validation + checkpoint (memory-safe)
    dsr_res = stage3_rl_and_validate(
        model=model,
        x_tech_tr=x_tech[:split],
        x_macro_tr=x_macro[:split],
        x_fund_tr=x_fund[:split],
        z_reg_tr=z_regime[:split],
        y_tr=y[:split],
        x_tech_val=x_tech[split:],
        x_macro_val=x_macro[split:],
        x_fund_val=x_fund[split:],
        z_reg_val=z_regime[split:],
        y_val=y[split:],
        weights_path=output_path,
        tech_dim=tech_dim,
        macro_dim=macro_dim,
        fund_dim=fund_dim,
        regime_dim=regime_dim,
        vq_tokenizer=vq_tokenizer,
    )

    logger.info("=" * 70)
    logger.info("Advanced Pipeline Complete.")
    logger.info("  DSR: %.4f | Significant: %s", dsr_res["dsr"], dsr_res["is_significant"])
    logger.info("  Checkpoint: %s", output_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
