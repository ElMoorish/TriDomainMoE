"""
Institutional Multi-Stage Training Pipeline for Tri-Domain MoE.
Trains specialized Technical, Macro, and Fundamental Experts,
optimizes continuous Softmax CAW routing under Multi-Objective Composite Loss,
fine-tunes via Differential Sortino Direct RL, and saves production weights.
"""

import sys
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
from src.features.fracdiff import FractionalDifferentiator
from src.features.labeling import TripleBarrierLabeler
from src.features.sentiment_engine import MacroSentimentEngine
from src.models.institutional_moe import TriDomainMoE
from src.loss.composite_loss import CompositeLoss
from src.rl.differential_ratio import DifferentialRiskRatio
from src.validation.deflated_sharpe import DeflatedSharpeRatio
from src.utils.logger import setup_logger

logger = setup_logger("InstitutionalTrainer")


def prepare_tri_domain_tensors(
    cross_df: pd.DataFrame,
    target_symbol: str = "equity",
    seq_len: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs aligned tensors for Tech, Macro, Fund, Regime, and Target Labels.
    """
    df = cross_df.copy()

    # 1. Technical domain features for target asset (NAS100)
    df["fracdiff"] = FractionalDifferentiator.frac_diff(df[f"{target_symbol}_close"], d=0.4)
    df["vol_20"] = df[f"{target_symbol}_ret"].rolling(20, min_periods=5).std().bfill()
    df["vol_mom"] = df[f"{target_symbol}_vol"] / (df[f"{target_symbol}_vol"].rolling(20).mean() + 1e-6)

    # 2. Macro domain features
    # df already has 'equity_vrp', 'gold_oil_ratio', 'dxy_ret', 'safe_haven_flow', 'oil_ret'
    
    # 3. Triple-barrier labeling
    labeler = TripleBarrierLabeler(pt_multiplier=1.5, sl_multiplier=1.5, max_holding_bars=15)
    labeled = labeler.label_events(df[f"{target_symbol}_close"])
    df["target_ret"] = labeled["return"]
    df = df.dropna().copy()

    tech_cols = [f"{target_symbol}_ret", "vol_20", "fracdiff", "vol_mom"]
    macro_cols = ["equity_vrp", "gold_oil_ratio", "dxy_ret", "safe_haven_flow", "oil_ret"]

    # Standardize
    for c in tech_cols + macro_cols:
        m, s = df[c].mean(), df[c].std() + 1e-6
        df[c] = (df[c] - m) / s

    tech_arr = df[tech_cols].to_numpy(dtype=np.float32)
    macro_arr = df[macro_cols].to_numpy(dtype=np.float32)
    targets_arr = df["target_ret"].to_numpy(dtype=np.float32)

    # Fundamental & Regime state from MacroSentimentEngine
    sentiment_engine = MacroSentimentEngine()
    current_z = sentiment_engine.get_regime_vector()

    X_tech, X_macro, X_fund, Z_regime, Y = [], [], [], [], []

    for i in range(seq_len, len(df)):
        X_tech.append(tech_arr[i - seq_len : i])
        X_macro.append(macro_arr[i - seq_len : i])

        # Fundamental vector with slight temporal perturbation
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


def main():
    storage = ParquetStorage()
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    cross_df = storage.load_bars("PORTFOLIO", "cross_asset")
    if cross_df.empty or len(cross_df) < 500:
        logger.error("Synchronized multi-asset data insufficient. Run scripts/download_multi_asset.py first.")
        sys.exit(1)

    logger.info("Loaded %d synchronized multi-asset periods.", len(cross_df))

    seq_len = 32
    x_tech, x_macro, x_fund, z_regime, y = prepare_tri_domain_tensors(cross_df, target_symbol="equity", seq_len=seq_len)
    n_samples = len(x_tech)
    split_idx = int(0.8 * n_samples)

    # Train / Val Partition
    x_tech_tr, x_tech_val = x_tech[:split_idx], x_tech[split_idx:]
    x_macro_tr, x_macro_val = x_macro[:split_idx], x_macro[split_idx:]
    x_fund_tr, x_fund_val = x_fund[:split_idx], x_fund[split_idx:]
    z_reg_tr, z_reg_val = z_regime[:split_idx], z_regime[split_idx:]
    y_tr, y_val = y[:split_idx], y[split_idx:]

    logger.info("Dataset: %d Train sequences, %d Out-of-Sample Validation sequences.", len(x_tech_tr), len(x_tech_val))

    tech_dim = x_tech.shape[-1]
    macro_dim = x_macro.shape[-1]
    fund_dim = x_fund.shape[-1]
    regime_dim = z_regime.shape[-1]

    # Instantiate TriDomainMoE
    model = TriDomainMoE(
        tech_dim=tech_dim,
        macro_dim=macro_dim,
        fund_dim=fund_dim,
        regime_dim=regime_dim,
        hidden_dim=48,
        lambda_c=0.5,
        noise_std=0.1,
    )

    criterion = CompositeLoss(delta_huber=1.0, lambda_dir=0.6, lambda_ic=0.4, lambda_balance=0.1)
    rl_criterion = DifferentialRiskRatio(eta=0.05, cost_bps=1.5)

    # Stage 1: Domain-Specialized Pre-training
    logger.info("=== Stage 1: Pre-training Domain Experts independently ===")
    pre_optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()

    for ep in range(3):
        pre_optimizer.zero_grad()
        p_t, _ = model.tech_expert(x_tech_tr)
        p_m, _ = model.macro_expert(x_macro_tr)
        p_f, _ = model.fund_expert(x_fund_tr)

        l_tech = mse(p_t, y_tr)
        l_macro = mse(p_m, y_tr)
        l_fund = mse(p_f, y_tr)

        pre_loss = l_tech + l_macro + l_fund
        pre_loss.backward()
        pre_optimizer.step()
        logger.info("Pretrain Epoch [%d/3] - Tech MSE: %.5f | Macro MSE: %.5f | Fund MSE: %.5f", ep + 1, l_tech.item(), l_macro.item(), l_fund.item())

    # Stage 2: Joint Multi-Objective Optimization
    logger.info("=== Stage 2: Joint Tri-Domain MoE Training under Composite Loss ===")
    joint_optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    batch_size = 32
    num_epochs = 6

    for epoch in range(num_epochs):
        permutation = torch.randperm(len(x_tech_tr))
        epoch_loss = 0.0
        batches = 0

        for i in range(0, len(x_tech_tr), batch_size):
            idx = permutation[i : i + batch_size]
            b_tech = x_tech_tr[idx]
            b_macro = x_macro_tr[idx]
            b_fund = x_fund_tr[idx]
            b_z = z_reg_tr[idx]
            b_y = y_tr[idx]

            joint_optimizer.zero_grad()
            out = model(b_tech, b_macro, b_fund, b_z)

            loss_dict = criterion(
                y_pred=out["y_pred"],
                y_true=b_y,
                g_weights=out["weights"],
                noisy_weights=out["noisy_weights"],
            )

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            joint_optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)
        logger.info("Epoch [%d/%d] - Composite Loss: %.4f", epoch + 1, num_epochs, avg_loss)

    # Stage 3: Continual RL Adaptation on Differential Sortino Ratio
    logger.info("=== Stage 3: Fine-Tuning Router via Differential Sortino RL ===")
    for p in model.base_predictor.parameters():
        p.requires_grad = False

    rl_optimizer = optim.Adam(model.router.parameters(), lr=4e-4)
    model.train()

    with torch.set_grad_enabled(True):
        rl_optimizer.zero_grad()
        out_seq = model(x_tech_tr, x_macro_tr, x_fund_tr, z_reg_tr)
        positions = torch.tanh(out_seq["y_pred"].squeeze(-1))
        realized_returns = y_tr.squeeze(-1)

        rl_loss = rl_criterion(positions, realized_returns, objective="sortino")
        rl_loss.backward()
        rl_optimizer.step()
        logger.info("RL Differential Sortino Step - Objective Value: %.4f", -rl_loss.item())

    # Stage 4: Out-of-Sample Performance Attribution & Deflated Sharpe Testing
    logger.info("=== Stage 4: Out-of-Sample Performance Attribution & Deflated Sharpe ===")
    model.eval()
    with torch.no_grad():
        val_out = model(x_tech_val, x_macro_val, x_fund_val, z_reg_val)
        val_pred = val_out["y_pred"].squeeze(-1).numpy()
        val_y = y_val.squeeze(-1).numpy()
        weights = val_out["weights"].numpy()

        trade_dir = np.sign(val_pred)
        strat_returns = trade_dir * val_y

        ann_factor = 252 * 390
        mean_ret = np.mean(strat_returns)
        std_ret = np.std(strat_returns) + 1e-8
        annual_sr = (mean_ret / std_ret) * np.sqrt(ann_factor)

        # Domain weight allocations
        mean_tech_wt = float(np.mean(weights[:, 0]))
        mean_macro_wt = float(np.mean(weights[:, 1]))
        mean_fund_wt = float(np.mean(weights[:, 2]))

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
        logger.info("  - Benchmark Sharpe (SR_0):        %.2f", dsr_res["benchmark_annual_sharpe"])
        logger.info("  - Deflated Sharpe Ratio (DSR):   %.4f", dsr_res["dsr"])
        logger.info("  - Statistically Significant (p >= 0.95): %s", dsr_res["is_significant"])

    # Stage 5: Serialize Institutional Model Checkpoint
    save_path = weights_dir / "institutional_moe_v1.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "tech_dim": tech_dim,
        "macro_dim": macro_dim,
        "fund_dim": fund_dim,
        "regime_dim": regime_dim,
        "domain_weights": {
            "technical": mean_tech_wt,
            "macro": mean_macro_wt,
            "fundamental": mean_fund_wt,
        },
        "dsr_metrics": dsr_res,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, save_path)

    logger.info("Institutional Tri-Domain MoE weights saved to %s", save_path)
    logger.info("=== Institutional Multi-Stage Training Pipeline Complete ===")


if __name__ == "__main__":
    main()
