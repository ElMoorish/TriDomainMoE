"""
Institutional Performance Attribution and Domain Evaluation Utility.
Analyzes out-of-sample predictive accuracy, profit factor, Sharpe ratio,
and dynamic domain routing allocations (Technical vs Macro vs Fundamental).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.storage import ParquetStorage
from src.models.institutional_moe import TriDomainMoE
from src.validation.deflated_sharpe import DeflatedSharpeRatio
from src.surveillance.risk_controls import RiskControls
from src.utils.logger import setup_logger

logger = setup_logger("InstitutionalEvaluator")


def main():
    weights_path = Path("weights/institutional_moe_v1.pt")
    if not weights_path.exists():
        logger.error("Weights checkpoint not found at %s. Train weights first.", weights_path)
        sys.exit(1)

    checkpoint = torch.load(weights_path, weights_only=False)
    logger.info("Loading TriDomainMoE model from %s...", weights_path)

    model = TriDomainMoE(
        tech_dim=checkpoint["tech_dim"],
        macro_dim=checkpoint["macro_dim"],
        fund_dim=checkpoint["fund_dim"],
        regime_dim=checkpoint["regime_dim"],
        hidden_dim=48,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Stored domain allocations from training:")
    for dom, wt in checkpoint["domain_weights"].items():
        logger.info("  - %s: %.1f%%", dom.capitalize(), wt * 100)

    # Load multi-asset portfolio data
    storage = ParquetStorage()
    cross_df = storage.load_bars("PORTFOLIO", "cross_asset")
    if cross_df.empty:
        logger.error("Cross-asset data not found.")
        sys.exit(1)

    logger.info("Evaluating on %d multi-asset time steps...", len(cross_df))

    from scripts.train_institutional_weights import prepare_tri_domain_tensors

    x_tech, x_macro, x_fund, z_regime, y = prepare_tri_domain_tensors(cross_df, target_symbol="equity", seq_len=32)
    split_idx = int(0.8 * len(x_tech))
    x_tech_val = x_tech[split_idx:]
    x_macro_val = x_macro[split_idx:]
    x_fund_val = x_fund[split_idx:]
    z_reg_val = z_regime[split_idx:]
    y_val = y[split_idx:]

    logger.info("Evaluating on %d real out-of-sample multi-asset steps...", len(x_tech_val))

    wins = 0
    losses = 0
    gross_profits = 0.0
    gross_losses = 0.0
    strategy_returns = []

    with torch.no_grad():
        val_out = model(x_tech_val, x_macro_val, x_fund_val, z_reg_val)

    preds = val_out["y_pred"].squeeze(-1).numpy()
    sizes = val_out["size"].squeeze(-1).numpy()
    actuals = y_val.squeeze(-1).numpy()

    for i in range(len(preds)):
        y_pred = preds[i]
        conviction = sizes[i]
        actual_ret = actuals[i]

        trade_dir = np.sign(y_pred)
        pnl = trade_dir * conviction * actual_ret
        strategy_returns.append(pnl)

        if pnl > 0:
            wins += 1
            gross_profits += pnl
        elif pnl < 0:
            losses += 1
            gross_losses += abs(pnl)

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else 99.0

    strat_arr = np.array(strategy_returns)
    ann_factor = 252 * 390
    ann_sharpe = (np.mean(strat_arr) / (np.std(strat_arr) + 1e-8)) * np.sqrt(ann_factor)

    logger.info("=== Tri-Domain MoE Out-of-Sample Performance ===")
    logger.info("  - Total Evaluated Setups: %d", total_trades)
    logger.info("  - Win Rate:                %.1f%%", win_rate)
    logger.info("  - Profit Factor:           %.2f", profit_factor)
    logger.info("  - Annualized Sharpe:       %.2f", ann_sharpe)
    logger.info("  - Calibrated Conviction Sizing Applied: Active")
    logger.info("=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
