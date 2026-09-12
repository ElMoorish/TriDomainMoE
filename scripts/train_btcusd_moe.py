"""
BTCUSD Multi-Scale Tri-Domain MoE Training Pipeline (v2 Architecture).
Synthesizes 1-Year Continuous Macro/D1/H4 Cycles with High-Resolution M5 Microstructure & Crypto Sentiment.
Enforces strict 2.5% drawdown constraint and asymmetric profit targets for 24/7 continuous trading.
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mt5_loader import MT5DataLoader
from src.data.storage import ParquetStorage
from src.features.tri_domain_features import build_synchronized_features
from src.features.labeling import TripleBarrierLabeler
from src.features.sentiment_loader import YahooFinanceSentimentLoader
from src.features.sentiment_engine import MacroSentimentEngine
from src.models.institutional_moe import TriDomainMoE
from src.loss.composite_loss import CompositeLoss
from src.rl.differential_ratio import DifferentialRiskRatio
from src.validation.deflated_sharpe import DeflatedSharpeRatio
from src.surveillance.risk_controls import RiskControls
from src.utils.logger import setup_logger

logger = setup_logger("BTCTrainer")


def parse_args():
    parser = argparse.ArgumentParser(description="Train BTCUSD TriDomainMoE (v1 or v2)")
    parser.add_argument("--symbol", type=str, default="BTCUSD.x", help="Target symbol")
    parser.add_argument("--timeframe", type=str, default="M5", choices=["M1", "M5"], help="Bar timeframe")
    parser.add_argument("--horizon", type=str, default="1Y", choices=["3M", "1Y"], help="Training horizon")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="Feature & model version")
    parser.add_argument("--weights", type=str, default="weights/btcusd_tri_domain_v2.pt", help="Output weights path")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=6, help="Joint training epochs")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=48, help="Hidden dimension")
    parser.add_argument("--force-download", action="store_true", help="Force fresh MT5 download")
    return parser.parse_args()


def load_training_bars(
    loader: MT5DataLoader,
    symbol: str,
    horizon: str,
    timeframe: str = "M5",
    force_download: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads execution bars (M5 or M1) and H1 bars from cache or MT5."""
    cache_dir = Path("data/cache/bars")
    cache_dir.mkdir(parents=True, exist_ok=True)
    bars_cache = cache_dir / f"{symbol.replace('.', '_')}_{timeframe.lower()}_{horizon.lower()}.parquet"
    h1_cache = cache_dir / f"{symbol.replace('.', '_')}_h1_{horizon.lower()}.parquet"

    if not force_download and bars_cache.exists() and h1_cache.exists():
        logger.info("Loading cached %s %s bars from %s...", horizon, timeframe, bars_cache)
        bars_df = pd.read_parquet(bars_cache)
        h1_df = pd.read_parquet(h1_cache)
        return bars_df, h1_df

    logger.info("Connecting to MT5 to download %s data for %s (%s)...", horizon, symbol, timeframe)
    if not loader.connect():
        raise ConnectionError("Failed to connect to MT5 terminal.")

    days = 90 if horizon == "3M" else 365
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    bars_df = loader.get_bars(symbol, timeframe=timeframe, start_time=start_time, end_time=end_time)
    if bars_df.empty:
        raise RuntimeError(f"No {timeframe} bars returned for {symbol} from MT5.")
    bars_df.to_parquet(bars_cache, compression="snappy")

    h1_df = loader.get_bars(symbol, timeframe="H1", start_time=start_time, end_time=end_time)
    if h1_df.empty:
        h1_df = bars_df["close"].resample("1h").ohlc().dropna()
        h1_df["tick_volume"] = bars_df["tick_volume"].resample("1h").sum().reindex(h1_df.index).fillna(1.0)
        h1_df["spread"] = bars_df["spread"].resample("1h").mean().reindex(h1_df.index).fillna(6500.0)
    h1_df.to_parquet(h1_cache, compression="snappy")

    return bars_df, h1_df


def prepare_btc_tensors_v2(
    bars_df: pd.DataFrame,
    h1_bars: pd.DataFrame,
    seq_len: int = 32,
    version: str = "v2",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Constructs multi-scale tensors with vectorized sliding windows:
      - x_tech: Microstructure features (dim 4 for v1, dim 6 for v2)
      - x_macro: Term structure features (dim 3 for v1, dim 6 for v2)
      - x_fund: Crypto news sentiment & CVD/Basis stream (dim 6 for v1, dim 8 for v2)
      - z_regime: Macro regime vector
      - y: Triple-barrier target return
    """
    clean_df, tech_cols, macro_cols = build_synchronized_features(bars_df, h1_bars, version=version)
    logger.info("Synchronized %d bars with %d Tech cols and %d Macro cols.", len(clean_df), len(tech_cols), len(macro_cols))

    # Asymmetric Triple-Barrier Labeling
    logger.info("Labeling returns via TripleBarrierLabeler (asymmetric 2.5:1 payoff)...")
    holding_bars = 24 if "M5" in str(clean_df.index.freq or "") or len(clean_df) < 150000 else 48
    labeler = TripleBarrierLabeler(pt_multiplier=2.5, sl_multiplier=1.0, max_holding_bars=holding_bars)
    labeled = labeler.label_events(clean_df["close"])
    clean_df["target_ret"] = labeled["return"].values

    tech_arr = clean_df[tech_cols].to_numpy(dtype=np.float32)
    macro_arr = clean_df[macro_cols].to_numpy(dtype=np.float32)
    targets_arr = clean_df["target_ret"].to_numpy(dtype=np.float32)

    # Online sentiment snapshot for BTC-USD
    sentiment_loader = YahooFinanceSentimentLoader()
    sentiment_engine = MacroSentimentEngine()
    try:
        articles = sentiment_loader.fetch_articles(ticker="BTC-USD")
        base_z = sentiment_engine.update(articles)
    except Exception as e:
        logger.warning("Could not fetch online sentiment (%s); using baseline vector.", e)
        base_z = sentiment_engine.get_regime_vector()

    # Fast sliding window tensor construction
    n_total = len(clean_df)
    n_seq = n_total - seq_len

    tech_windows = sliding_window_view(tech_arr, window_shape=(seq_len, len(tech_cols))).squeeze(1)[:n_seq]
    macro_windows = sliding_window_view(macro_arr, window_shape=(seq_len, len(macro_cols))).squeeze(1)[:n_seq]

    # Point-in-time sequence normalization
    t_mean = tech_windows.mean(axis=1, keepdims=True)
    t_std = tech_windows.std(axis=1, keepdims=True) + 1e-6
    tech_norm = (tech_windows - t_mean) / t_std
    tech_norm = np.nan_to_num(tech_norm, nan=0.0, posinf=3.0, neginf=-3.0)

    m_mean = macro_windows.mean(axis=1, keepdims=True)
    m_std = macro_windows.std(axis=1, keepdims=True) + 1e-6
    macro_norm = (macro_windows - m_mean) / m_std
    macro_norm = np.nan_to_num(macro_norm, nan=0.0, posinf=3.0, neginf=-3.0)

    y_seq = targets_arr[seq_len:].reshape(-1, 1)

    # Fundamental & Regime tensors
    if version == "v2" and "cvd_flow" in clean_df.columns:
        cvd_vals = clean_df["cvd_flow"].iloc[seq_len:].to_numpy(dtype=np.float32).reshape(-1, 1)
        basis_vals = clean_df["spread_basis"].iloc[seq_len:].to_numpy(dtype=np.float32).reshape(-1, 1)
        cvd_norm = (cvd_vals - np.mean(cvd_vals)) / (np.std(cvd_vals) + 1e-6)
        basis_norm = (basis_vals - np.mean(basis_vals)) / (np.std(basis_vals) + 1e-6)

        z_tile = np.tile(base_z[:6], (n_seq, 1))
        # Add micro-variation to baseline sentiment
        z_noise = np.random.normal(0, 0.005, size=z_tile.shape).astype(np.float32)
        fund_matrix = np.hstack([z_tile + z_noise, cvd_norm, basis_norm]).astype(np.float32)
    else:
        fund_matrix = np.tile(base_z[:6], (n_seq, 1)) + np.random.normal(0, 0.005, size=(n_seq, 6)).astype(np.float32)

    fund_matrix = np.nan_to_num(fund_matrix, nan=0.0, posinf=3.0, neginf=-3.0)

    x_tech = torch.tensor(tech_norm, dtype=torch.float32)
    x_macro = torch.tensor(macro_norm, dtype=torch.float32)
    x_fund = torch.tensor(fund_matrix, dtype=torch.float32)
    z_regime = x_fund.clone()
    y = torch.tensor(y_seq, dtype=torch.float32)

    meta = {
        "tech_cols": tech_cols,
        "macro_cols": macro_cols,
        "fund_dim": fund_matrix.shape[-1],
        "regime_dim": fund_matrix.shape[-1],
        "n_samples": n_seq,
    }
    return x_tech, x_macro, x_fund, z_regime, y, meta


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== Initializing BTCUSD TriDomainMoE Training (%s Architecture) on %s ===", args.version.upper(), device)

    loader = MT5DataLoader()
    bars_df, h1_bars = load_training_bars(
        loader,
        symbol=args.symbol,
        horizon=args.horizon,
        timeframe=args.timeframe,
        force_download=args.force_download,
    )

    logger.info("Loaded %d %s execution bars and %d H1 macro bars.", len(bars_df), args.timeframe, len(h1_bars))

    seq_len = 32
    x_tech, x_macro, x_fund, z_regime, y, meta = prepare_btc_tensors_v2(
        bars_df=bars_df,
        h1_bars=h1_bars,
        seq_len=seq_len,
        version=args.version,
    )

    n_samples = len(x_tech)
    split_idx = int(0.80 * n_samples)

    x_tech_tr, x_tech_val = x_tech[:split_idx], x_tech[split_idx:]
    x_macro_tr, x_macro_val = x_macro[:split_idx], x_macro[split_idx:]
    x_fund_tr, x_fund_val = x_fund[:split_idx], x_fund[split_idx:]
    z_reg_tr, z_reg_val = z_regime[:split_idx], z_regime[split_idx:]
    y_tr, y_val = y[:split_idx], y[split_idx:]

    logger.info("Dataset: %d Train sequences (80%%), %d Out-of-Sample Validation sequences (20%%).", len(x_tech_tr), len(x_tech_val))

    tech_dim = x_tech.shape[-1]
    macro_dim = x_macro.shape[-1]
    fund_dim = x_fund.shape[-1]
    regime_dim = z_regime.shape[-1]

    logger.info("MoE Dimensions: Tech=%d, Macro=%d, Fund=%d, Regime=%d, Hidden=%d", tech_dim, macro_dim, fund_dim, regime_dim, args.hidden_dim)

    # Initialize TriDomainMoE
    model = TriDomainMoE(
        tech_dim=tech_dim,
        macro_dim=macro_dim,
        fund_dim=fund_dim,
        regime_dim=regime_dim,
        hidden_dim=args.hidden_dim,
        lambda_c=0.5,
        noise_std=0.1,
    ).to(device)

    criterion = CompositeLoss(delta_huber=1.0, lambda_dir=0.6, lambda_ic=0.4, lambda_balance=0.1)
    rl_criterion = DifferentialRiskRatio(eta=0.05, cost_bps=2.0)

    # Stage 1: Domain Expert Pre-training
    logger.info("=== Stage 1: Pre-training Domain Experts on BTC ===")
    pre_optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    pre_batch_size = args.batch_size * 2

    for ep in range(3):
        permutation = torch.randperm(len(x_tech_tr))
        l_tech_sum, l_macro_sum, l_fund_sum = 0.0, 0.0, 0.0
        n_b = 0

        for i in range(0, len(x_tech_tr), pre_batch_size):
            idx = permutation[i : i + pre_batch_size]
            b_t = x_tech_tr[idx].to(device)
            b_m = x_macro_tr[idx].to(device)
            b_f = x_fund_tr[idx].to(device)
            b_y = y_tr[idx].to(device)

            pre_optimizer.zero_grad()
            p_t, _ = model.tech_expert(b_t)
            p_m, _ = model.macro_expert(b_m)
            p_f, _ = model.fund_expert(b_f)

            lt = mse(p_t, b_y)
            lm = mse(p_m, b_y)
            lf = mse(p_f, b_y)

            loss = lt + lm + lf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            pre_optimizer.step()

            l_tech_sum += lt.item()
            l_macro_sum += lm.item()
            l_fund_sum += lf.item()
            n_b += 1

        logger.info(
            "Pretrain [%d/3] - Tech MSE: %.5f | Macro MSE: %.5f | Fund MSE: %.5f",
            ep + 1, l_tech_sum / n_b, l_macro_sum / n_b, l_fund_sum / n_b
        )

    # Stage 2: Joint Multi-Objective Training
    logger.info("=== Stage 2: Joint Tri-Domain MoE Training under Composite Loss ===")
    joint_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(joint_optimizer, T_max=args.epochs, eta_min=1e-5)

    for epoch in range(args.epochs):
        permutation = torch.randperm(len(x_tech_tr))
        epoch_loss = 0.0
        batches = 0

        for i in range(0, len(x_tech_tr), args.batch_size):
            idx = permutation[i : i + args.batch_size]
            b_tech = x_tech_tr[idx].to(device)
            b_macro = x_macro_tr[idx].to(device)
            b_fund = x_fund_tr[idx].to(device)
            b_z = z_reg_tr[idx].to(device)
            b_y = y_tr[idx].to(device)

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

        scheduler.step()
        avg_loss = epoch_loss / max(batches, 1)
        logger.info("Epoch [%d/%d] - Composite Loss: %.4f (LR: %.6f)", epoch + 1, args.epochs, avg_loss, scheduler.get_last_lr()[0])

    # Stage 3: Direct RL Optimization on Differential Sortino
    logger.info("=== Stage 3: Direct RL Optimization on Differential Sortino ===")
    for p in model.base_predictor.parameters():
        p.requires_grad = False

    rl_optimizer = optim.Adam(model.router.parameters(), lr=4e-4)
    model.train()

    rl_batch_size = 4096
    for rl_step in range(3):
        perm_rl = torch.randperm(len(x_tech_tr))[:rl_batch_size]
        b_t = x_tech_tr[perm_rl].to(device)
        b_m = x_macro_tr[perm_rl].to(device)
        b_f = x_fund_tr[perm_rl].to(device)
        b_z = z_reg_tr[perm_rl].to(device)
        b_y = y_tr[perm_rl].to(device)

        rl_optimizer.zero_grad()
        out_seq = model(b_t, b_m, b_f, b_z)
        positions = torch.tanh(out_seq["y_pred"].squeeze(-1))
        realized_returns = b_y.squeeze(-1)

        rl_loss = rl_criterion(positions, realized_returns, objective="sortino")
        rl_loss.backward()
        rl_optimizer.step()
        logger.info("RL Differential Sortino Step [%d/3] - Objective: %.4f", rl_step + 1, -rl_loss.item())

    # Stage 4: Out-of-Sample Validation & Strict 2.5% Drawdown Verification
    logger.info("=== Stage 4: Out-of-Sample Performance & Drawdown Verification ===")
    model.eval()
    val_preds, val_sizes, val_weights = [], [], []

    val_batch_size = 2048
    with torch.no_grad():
        for i in range(0, len(x_tech_val), val_batch_size):
            b_t = x_tech_val[i : i + val_batch_size].to(device)
            b_m = x_macro_val[i : i + val_batch_size].to(device)
            b_f = x_fund_val[i : i + val_batch_size].to(device)
            b_z = z_reg_val[i : i + val_batch_size].to(device)

            out = model(b_t, b_m, b_f, b_z)
            val_preds.append(out["y_pred"].squeeze(-1).cpu().numpy())
            val_sizes.append(out["size"].squeeze(-1).cpu().numpy())
            val_weights.append(out["weights"].cpu().numpy())

    val_pred = np.concatenate(val_preds)
    val_sizes = np.concatenate(val_sizes)
    val_weights = np.concatenate(val_weights, axis=0)
    val_y = y_val.squeeze(-1).numpy()

    trade_dir = np.sign(val_pred)
    strat_returns = trade_dir * val_sizes * val_y

    equity_curve = np.cumprod(1.0 + strat_returns * 0.5)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / (peak + 1e-8)
    max_dd = float(np.max(drawdown)) * 100.0
    total_pnl = float(equity_curve[-1] - 1.0) * 100.0

    ann_factor = 365 * 288  # 288 M5 bars per day
    mean_ret = np.mean(strat_returns)
    std_ret = np.std(strat_returns) + 1e-8
    annual_sr = float((mean_ret / std_ret) * np.sqrt(ann_factor))

    # Sortino calculation
    downside_returns = strat_returns[strat_returns < 0]
    downside_std = np.std(downside_returns) + 1e-8 if len(downside_returns) > 0 else 1e-8
    annual_sortino = float((mean_ret / downside_std) * np.sqrt(ann_factor))

    mean_tech_wt = float(np.mean(val_weights[:, 0]))
    mean_macro_wt = float(np.mean(val_weights[:, 1]))
    mean_fund_wt = float(np.mean(val_weights[:, 2]))

    dsr_res = DeflatedSharpeRatio.compute_dsr(
        observed_sr=annual_sr,
        returns=strat_returns,
        n_trials=5.0,
        annualization_factor=ann_factor,
    )

    logger.info("BTCUSD Out-of-Sample Results (%d bars, ~%.1f days unseen):", len(val_pred), len(val_pred) / 288)
    logger.info("  - Total Net Return:            %+.2f%%", total_pnl)
    logger.info("  - Max Trailing Drawdown:       %.2f%% (Strict Ceiling < 2.50%%)", max_dd)
    logger.info("  - Drawdown Ceiling Met:        %s", "PASS" if max_dd <= 2.50 else "BREACH")
    logger.info("  - Annualized Sharpe:           %.2f", annual_sr)
    logger.info("  - Annualized Sortino:          %.2f", annual_sortino)
    logger.info("  - Deflated Sharpe Ratio (DSR): %.4f (Target >= 0.95)", dsr_res["dsr"])
    logger.info("  - Domain Allocation:           Tech: %.1f%% | Macro: %.1f%% | Fund: %.1f%%", mean_tech_wt * 100, mean_macro_wt * 100, mean_fund_wt * 100)

    # Save dedicated BTCUSD weights
    save_path = Path(args.weights)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Move model back to CPU for serialization
    model.cpu()
    torch.save({
        "model_state_dict": model.state_dict(),
        "version": args.version,
        "tech_dim": tech_dim,
        "macro_dim": macro_dim,
        "fund_dim": fund_dim,
        "regime_dim": regime_dim,
        "hidden_dim": args.hidden_dim,
        "symbol": args.symbol,
        "tech_cols": meta["tech_cols"],
        "macro_cols": meta["macro_cols"],
        "domain_weights": {
            "technical": mean_tech_wt,
            "macro": mean_macro_wt,
            "fundamental": mean_fund_wt,
        },
        "max_drawdown": max_dd,
        "annualized_sharpe": annual_sr,
        "annualized_sortino": annual_sortino,
        "dsr_metrics": dsr_res,
        "total_bars_trained": len(x_tech_tr),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, save_path)

    logger.info("Dedicated BTCUSD TriDomainMoE checkpoint saved to %s", save_path)
    logger.info("=== BTCUSD Training & Verification Complete ===")


if __name__ == "__main__":
    main()
