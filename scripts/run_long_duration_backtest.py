"""
Long-Duration Multi-Horizon Backtesting Pipeline (3 Months & 1 Year).
Executes Tri-Domain MoE across up to 525,000 continuous 24/7 M1 bars directly from MT5.
Features intra-bar execution with real broker floating spreads and slippage.
Generates comprehensive statistical evidence suite and monthly compounding breakdowns.
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mt5_loader import MT5DataLoader
from src.data.storage import ParquetStorage
from src.models.institutional_moe import TriDomainMoE
from src.features.tri_domain_features import build_synchronized_features
from src.features.fracdiff import FractionalDifferentiator
from src.features.sentiment_loader import YahooFinanceSentimentLoader
from src.features.sentiment_engine import MacroSentimentEngine
from src.backtest.bar_engine import IntraBarExecutionEngine
from src.backtest.metrics import StatisticalEvidenceMetrics
from src.utils.logger import setup_logger

logger = setup_logger("LongDurationBacktester")


def parse_args():
    parser = argparse.ArgumentParser(description="Long-Duration MoE Backtest (3M & 1Y)")
    parser.add_argument("--symbol", type=str, default="BTCUSD.x", help="Target symbol")
    parser.add_argument("--horizon", type=str, default="3M", choices=["3M", "1Y", "ALL"], help="Backtest horizon (3M=90d, 1Y=365d)")
    parser.add_argument("--weights", type=str, default="weights/btcusd_tri_domain_v2.pt", help="Path to weights")
    parser.add_argument("--risk-pct", type=float, default=0.0010, help="Risk ceiling per trade (0.0010 = 0.10%)")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Initial capital in USD")
    parser.add_argument("--timeframe", type=str, default="M5", choices=["M1", "M5"], help="Execution bar timeframe (M1 or M5)")
    parser.add_argument("--threshold", type=float, default=0.030, help="Minimum drift threshold to trigger entry")
    parser.add_argument("--min-conviction", type=float, default=0.18, help="Minimum conviction threshold from meta-sizer")
    parser.add_argument("--trend-gate", action=argparse.BooleanOptionalAction, default=True, help="Enable H1 macro trend directional gating to reject counter-trend false breakouts")
    parser.add_argument("--session-filter", action=argparse.BooleanOptionalAction, default=True, help="Enable 24/7 crypto session conditioning (scales TP in institutional overlap, tightens weekend conviction)")
    parser.add_argument("--enhanced", action=argparse.BooleanOptionalAction, default=True, help="Enable Enhanced Live v2 filters (Anti-adverse order flow shield, Entropy gating, Overlap sizing, Breakeven ratchet)")
    parser.add_argument("--breakeven-ratchet", action=argparse.BooleanOptionalAction, default=True, help="Enable dynamic breakeven ratchet in execution engine")
    parser.add_argument("--be-activation-ratio", type=float, default=0.50, help="Activation ratio of TP distance for breakeven ratchet")
    parser.add_argument("--k-sl", type=float, default=2.5, help="Stop loss volatility multiplier (must be >= 2.0 to clear spread)")
    parser.add_argument("--k-tp", type=float, default=7.5, help="Take profit volatility multiplier (asymmetric 3.0:1 payoff)")
    parser.add_argument("--min-sl-dist", type=float, default=500.0, help="Minimum stop-loss distance in USD to prevent spread stop-outs")
    parser.add_argument("--max-holding-bars", type=int, default=1440, help="Maximum holding duration in bars (default 1440 for M1, automatically scaled for M5)")
    parser.add_argument("--force-download", action="store_true", help="Force fresh bar download from MT5")
    return parser.parse_args()


def load_long_duration_bars(
    loader: MT5DataLoader,
    symbol: str,
    horizon: str,
    timeframe: str = "M5",
    force_download: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads execution bars (M5 or M1) and H1 bars from MT5 or cache for 3M (90 days) or 1Y (365 days).
    """
    days = 90 if horizon == "3M" else 365
    cache_dir = Path("data/cache/bars")
    cache_dir.mkdir(parents=True, exist_ok=True)
    bars_cache = cache_dir / f"{symbol.replace('.', '_')}_{timeframe.lower()}_{horizon.lower()}.parquet"
    h1_cache = cache_dir / f"{symbol.replace('.', '_')}_h1_{horizon.lower()}.parquet"

    if not force_download and bars_cache.exists() and h1_cache.exists():
        logger.info("Loading cached %s %s bars from %s...", horizon, timeframe, bars_cache)
        bars_df = pd.read_parquet(bars_cache)
        h1_df = pd.read_parquet(h1_cache)
        return bars_df, h1_df

    logger.info("Connecting to MetaTrader 5 to download %d days of continuous data for %s (%s)...", days, symbol, timeframe)
    if not loader.connect():
        raise ConnectionError("Failed to connect to MetaTrader 5.")

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    logger.info("Downloading %s bars from %s to %s...", timeframe, start_time, end_time)
    bars_df = loader.get_bars(symbol, timeframe=timeframe, start_time=start_time, end_time=end_time)
    if bars_df.empty:
        raise RuntimeError(f"No {timeframe} bars returned for {symbol} from MT5.")
    logger.info("Downloaded %d %s bars for %s.", len(bars_df), timeframe, symbol)
    bars_df.to_parquet(bars_cache, compression="snappy")

    logger.info("Downloading H1 macro bars from %s to %s...", start_time, end_time)
    h1_df = loader.get_bars(symbol, timeframe="H1", start_time=start_time, end_time=end_time)
    if h1_df.empty:
        logger.warning("Could not download H1 bars directly; constructing H1 from %s bars.", timeframe)
        h1_df = bars_df["close"].resample("1h").ohlc().dropna()
        h1_df["tick_volume"] = bars_df["tick_volume"].resample("1h").sum().reindex(h1_df.index).fillna(1.0)
        h1_df["spread"] = bars_df["spread"].resample("1h").mean().reindex(h1_df.index).fillna(6500.0)
    logger.info("Downloaded %d H1 bars for %s.", len(h1_df), symbol)
    h1_df.to_parquet(h1_cache, compression="snappy")

    return bars_df, h1_df


def generate_long_horizon_signals(
    bars_df: pd.DataFrame,
    h1_bars: pd.DataFrame,
    weights_path: Path,
    threshold: float = 0.030,
    min_conviction: float = 0.18,
    trend_gate: bool = True,
    session_filter: bool = True,
    k_sl: float = 2.5,
    k_tp: float = 7.5,
    min_sl_dist: float = 500.0,
    enhanced: bool = True,
) -> pd.DataFrame:
    """
    Computes synchronized technical & macro features and performs batched MoE neural forward passes.
    Supports Enhanced Live v2 anti-adverse gating, entropy filtering, and overlap conditioning.
    """
    logger.info("Computing multi-domain features across %d bars (Enhanced: %s)...", len(bars_df), enhanced)

    # 1. Load Model Checkpoint & Determine Version
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    version = checkpoint.get("feature_version", checkpoint.get("version", "v1" if checkpoint["tech_dim"] == 4 else "v2"))
    hidden_dim = checkpoint.get("hidden_dim", 64 if version == "v3" else 48)

    # 2. Build Synchronized Multi-Domain Features
    clean_df, tech_cols, macro_cols = build_synchronized_features(bars_df, h1_bars, version=version)
    logger.info("Clean synchronized dataset: %d bars (version: %s, tech: %d, macro: %d).", len(clean_df), version, len(tech_cols), len(macro_cols))

    if enhanced and "normalized_ofi" not in clean_df.columns:
        from src.features.microstructure_features import MicrostructureFeatureSet, MICROSTRUCTURE_COLS
        logger.info("Computing microstructure anti-adverse features for Enhanced Live v2...")
        ms = MicrostructureFeatureSet(kyle_window=50, vpin_buckets=50, rv_inner=5, rv_outer=50)
        ms_df = ms.build(clean_df)
        for col in MICROSTRUCTURE_COLS:
            clean_df[col] = ms_df[col]

    vq_tokenizer = None
    vq_latent_dim = 0
    if "vq_tokenizer_state_dict" in checkpoint and "vq_config" in checkpoint:
        from src.models.vq_vae import MarketStateTokenizer
        vq_cfg = checkpoint["vq_config"]
        vq_tokenizer = MarketStateTokenizer(
            input_dim=checkpoint["tech_dim"],
            latent_dim=vq_cfg.get("latent_dim", 64),
            num_embeddings=vq_cfg.get("num_embeddings", 128),
            patch_len=vq_cfg.get("patch_len", 16),
            use_ema=True,
        )
        vq_tokenizer.load_state_dict(checkpoint["vq_tokenizer_state_dict"])
        vq_tokenizer.eval()
        vq_latent_dim = vq_cfg.get("latent_dim", 64)

    model = TriDomainMoE(
        tech_dim=checkpoint["tech_dim"],
        macro_dim=checkpoint["macro_dim"],
        fund_dim=checkpoint["fund_dim"],
        regime_dim=checkpoint["regime_dim"],
        hidden_dim=hidden_dim,
        vq_tokenizer=vq_tokenizer,
        vq_latent_dim=vq_latent_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 3. Batched Point-in-Time Forward Passes
    seq_len = 64 if version == "v3" else 32
    tech_data = clean_df[tech_cols].to_numpy(dtype=np.float32)
    macro_data = clean_df[macro_cols].to_numpy(dtype=np.float32)

    n_samples = len(clean_df) - seq_len
    if n_samples <= 0:
        raise ValueError(f"Insufficient bars ({len(clean_df)}) for sequence length {seq_len}.")

    chunk_size = 2000
    y_preds_list = []
    convictions_list = []
    entropies_list = []

    sentiment_loader = YahooFinanceSentimentLoader()
    sentiment_engine = MacroSentimentEngine()
    try:
        articles = sentiment_loader.fetch_articles("BTC-USD")
        current_z = sentiment_engine.update(articles)
    except Exception as e:
        logger.warning("Could not fetch online sentiment (%s); using baseline regime vector.", e)
        current_z = sentiment_engine.get_regime_vector()

    fund_dim = checkpoint["fund_dim"]
    regime_dim = checkpoint["regime_dim"]
    if fund_dim == 8 and "cvd_flow" in clean_df.columns:
        cvd_vals = clean_df["cvd_flow"].iloc[seq_len:].to_numpy(dtype=np.float32).reshape(-1, 1)
        basis_vals = clean_df["spread_basis"].iloc[seq_len:].to_numpy(dtype=np.float32).reshape(-1, 1)
        cvd_norm = (cvd_vals - np.mean(cvd_vals)) / (np.std(cvd_vals) + 1e-6)
        basis_norm = (basis_vals - np.mean(basis_vals)) / (np.std(basis_vals) + 1e-6)
        z_tile = np.tile(current_z[:6], (n_samples, 1))
        fund_data = np.hstack([z_tile, cvd_norm, basis_norm]).astype(np.float32)
    else:
        current_fund_z = current_z[:fund_dim] if len(current_z) >= fund_dim else np.resize(current_z, fund_dim)
        fund_data = np.tile(current_fund_z, (n_samples, 1)).astype(np.float32)

    current_reg_z = current_z[:regime_dim] if len(current_z) >= regime_dim else np.resize(current_z, regime_dim)
    z_reg_data = np.tile(current_reg_z, (n_samples, 1)).astype(np.float32)

    logger.info("Running chunked MoE neural inference across %d steps on %s...", n_samples, device)
    t0 = time.perf_counter()

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        batch_n = end_idx - start_idx

        x_tech_batch = []
        x_macro_batch = []
        for idx in range(start_idx, end_idx):
            t_slice = tech_data[idx : idx + seq_len]
            m_slice = macro_data[idx : idx + seq_len]
            t_norm = (t_slice - t_slice.mean(axis=0)) / (t_slice.std(axis=0) + 1e-6)
            m_norm = (m_slice - m_slice.mean(axis=0)) / (m_slice.std(axis=0) + 1e-6)
            x_tech_batch.append(t_norm)
            x_macro_batch.append(m_norm)

        x_tech_tensor = torch.tensor(np.array(x_tech_batch), dtype=torch.float32, device=device)
        x_macro_tensor = torch.tensor(np.array(x_macro_batch), dtype=torch.float32, device=device)
        x_fund_tensor = torch.tensor(fund_data[start_idx:end_idx], dtype=torch.float32, device=device)
        z_reg_tensor = torch.tensor(z_reg_data[start_idx:end_idx], dtype=torch.float32, device=device)

        with torch.no_grad():
            out = model(x_tech_tensor, x_macro_tensor, x_fund_tensor, z_reg_tensor)

        y_preds_list.append(out["y_pred"].squeeze(-1).cpu().numpy())
        convictions_list.append(out["size"].squeeze(-1).cpu().numpy())
        if "entropy" in out:
            entropies_list.append(out["entropy"].cpu().numpy())
        elif "weights" in out:
            w = out["weights"].cpu()
            ent = -(w * (w + 1e-10).log()).sum(dim=-1).numpy()
            entropies_list.append(ent)

    inference_time = time.perf_counter() - t0
    logger.info("Inference completed in %.2f seconds (%.0f bars/second).", inference_time, n_samples / max(inference_time, 0.001))

    y_preds = np.concatenate(y_preds_list)
    convictions = np.concatenate(convictions_list)
    entropies = np.concatenate(entropies_list) if entropies_list else np.zeros_like(y_preds)

    signal_indices = clean_df.index[seq_len:]
    sig_df = pd.DataFrame({
        "timestamp": signal_indices,
        "y_pred": y_preds,
        "conviction": convictions,
        "entropy": entropies,
        "h1_trend": clean_df["h1_trend_50"].iloc[seq_len:].values,
        "vol_20": clean_df["vol_20"].iloc[seq_len:].values,
        "close": clean_df["close"].iloc[seq_len:].values,
    }).set_index("timestamp")

    if "normalized_ofi" in clean_df.columns:
        sig_df["normalized_ofi"] = clean_df["normalized_ofi"].iloc[seq_len:].values
        sig_df["vpin"] = clean_df["vpin"].iloc[seq_len:].values

    timestamps = pd.to_datetime(signal_indices)
    hours = timestamps.hour
    weekdays = timestamps.weekday

    # Detect Off-Hours / Weekend: Friday after 21:00 UTC through Sunday 22:00 UTC, and night hours (22:00 - 06:00 UTC)
    is_weekend = (weekdays == 5) | (weekdays == 6) | ((weekdays == 4) & (hours >= 21)) | ((weekdays == 0) & (hours < 2))
    is_night = (hours >= 22) | (hours <= 6)
    is_off_hours = is_weekend | is_night
    is_institutional = (~is_weekend) & (hours >= 12) & (hours <= 19)

    eff_conviction = min_conviction
    if session_filter:
        eff_conviction = np.where(is_off_hours, np.maximum(min_conviction, 0.20), min_conviction)

    if enhanced:
        # Vector 4: London / NY Overlap Conditioning (+15% conviction sizing between 12:00 and 18:00 UTC)
        is_london_ny = (~is_weekend) & (hours >= 12) & (hours <= 18)
        sig_df["conviction"] = np.where(is_london_ny, sig_df["conviction"] * 1.15, sig_df["conviction"])

    buy_mask = (sig_df["y_pred"] > threshold) & (sig_df["conviction"] >= eff_conviction)
    sell_mask = (sig_df["y_pred"] < -threshold) & (sig_df["conviction"] >= eff_conviction)

    if trend_gate:
        # Macro Trend Alignment: prevent counter-trend false breakouts into hostile flows
        buy_mask = buy_mask & (sig_df["h1_trend"] > -0.20)
        sell_mask = sell_mask & (sig_df["h1_trend"] < 0.20)

    if enhanced:
        # Vector 1: Pre-Trade Anti-Adverse Order Book Shield
        if "normalized_ofi" in sig_df.columns:
            buy_mask = buy_mask & ~(sig_df["normalized_ofi"] < -0.20) & ~(sig_df["vpin"] > 0.65)
            sell_mask = sell_mask & ~(sig_df["normalized_ofi"] > 0.20) & ~(sig_df["vpin"] > 0.65)

        # Vector 3: Router Shannon Entropy Filter (require higher conviction if experts are conflicted)
        conflicted_mask = sig_df["entropy"] >= 1.00
        buy_mask = buy_mask & ~(conflicted_mask & (sig_df["y_pred"] < 0.038))
        sell_mask = sell_mask & ~(conflicted_mask & (sig_df["y_pred"] > -0.038))

    sig_df["direction"] = np.where(buy_mask, "BUY", np.where(sell_mask, "SELL", "FLAT"))
    sig_df["sigma"] = np.maximum(sig_df["vol_20"], 0.003) * sig_df["close"]
    sig_df["sl_dist"] = np.maximum(k_sl * sig_df["sigma"], min_sl_dist)

    if session_filter:
        # Dynamic Take-Profit: Expand to 3.0R (7.5 sigma) during institutional overlap, 2.5R (6.25 sigma) during standard/off-hours
        tp_mult = np.where(is_institutional, k_tp, np.where(is_off_hours, 2.0 * k_sl, 2.5 * k_sl))
        sig_df["tp_dist"] = np.maximum(tp_mult * sig_df["sigma"], 2.0 * sig_df["sl_dist"])
    else:
        sig_df["tp_dist"] = np.maximum(k_tp * sig_df["sigma"], 2.0 * sig_df["sl_dist"])

    # Reindex signals to match the exact bars_df index
    full_sig = sig_df[["direction", "conviction", "sl_dist", "tp_dist"]].reindex(bars_df.index).fillna({
        "direction": "FLAT",
        "conviction": 0.0,
        "sl_dist": 500.0,
        "tp_dist": 1000.0,
    })

    actionable = (full_sig["direction"] != "FLAT").sum()
    logger.info("Generated %d actionable signals across %s horizon.", actionable, len(bars_df))
    return full_sig


def compute_monthly_performance(trades_df: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
    """Aggregates trades into calendar months for compounding return analysis."""
    if trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M").astype(str)

    monthly_rows = []
    running_equity = initial_balance

    for month_str, group in df.groupby("month"):
        n_tr = len(group)
        wins = (group["pnl_cash"] > 0).sum()
        losses = (group["pnl_cash"] <= 0).sum()
        win_rate = (wins / n_tr) * 100.0 if n_tr > 0 else 0.0

        gross_win = group[group["pnl_cash"] > 0]["pnl_cash"].sum()
        gross_loss = abs(group[group["pnl_cash"] <= 0]["pnl_cash"].sum())
        pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

        month_pnl = group["pnl_cash"].sum()
        month_return_pct = (month_pnl / running_equity) * 100.0
        running_equity += month_pnl

        monthly_rows.append({
            "Month": month_str,
            "Trades": n_tr,
            "Win Rate (%)": round(win_rate, 2),
            "Profit Factor": round(pf, 2),
            "Net PnL ($)": round(month_pnl, 2),
            "Return (%)": round(month_return_pct, 2),
            "Ending Equity ($)": round(running_equity, 2),
        })

    return pd.DataFrame(monthly_rows)


def print_master_report(
    horizon: str,
    stats: Dict[str, Any],
    monthly_df: pd.DataFrame,
    initial_balance: float,
    risk_pct: float,
    n_bars: int,
    timeframe: str = "M5",
    enhanced: bool = False,
):
    """Formats and prints comprehensive master report."""
    streaks = stats.get("streaks", {})
    runs = stats.get("runs_test", {})
    dd = stats.get("drawdowns", {})
    risk = stats.get("risk_ratios", {})

    sep = "=" * 86
    line = "-" * 86

    print("\n" + sep)
    badge = " [ENHANCED LIVE v2 ACTIVE]" if enhanced else " [STANDARD BASELINE]"
    print(f"       INSTITUTIONAL LONG-DURATION BACKTEST REPORT: {horizon} HORIZON ({timeframe}) (BTCUSD.x){badge}")
    print(f"                  Tri-Domain Mixture of Experts | MT5 Intra-Bar Engine")
    if enhanced:
        print("   Enhancement Vectors: (1) Anti-Adverse OFI/VPIN Shield | (2) Volatility Breakeven Ratchet")
        print("                        (3) Router Shannon Entropy Filter | (4) London/NY Overlap Sizing")
    print(sep)
    print(f"  Dataset Span:             {n_bars:,} continuous {timeframe} bars (Full 24/7 Market Coverage)")
    print(f"  Initial Capital:          ${initial_balance:,.2f} USD")
    print(f"  Final Equity:             ${stats.get('final_equity', 0.0):,.2f} USD")
    print(f"  Net Profit:               ${stats.get('net_profit_cash', 0.0):+,.2f} USD ({stats.get('total_return_pct', 0.0):+.2f}%)")
    print(f"  Risk per Trade Ceiling:   {risk_pct * 100:.2f}% Equity (${initial_balance * risk_pct:.2f} per 1.0 sigma SL)")
    print(line)
    print("  [1] EXECUTION & TRADE OUTCOMES")
    print(f"      - Total Executed Trades:       {stats.get('total_trades', 0):,}")
    print(f"      - Winning Trades:              {stats.get('winning_trades', 0):,} ({stats.get('win_rate_pct', 0.0):.2f}%)")
    print(f"      - Losing Trades:               {stats.get('losing_trades', 0):,} ({stats.get('loss_rate_pct', 0.0):.2f}%)")
    print(f"      - Profit Factor:               {stats.get('profit_factor', 0.0):.2f}")
    print(f"      - Payoff Ratio (Win/Loss):     {stats.get('payoff_ratio', 0.0):.2f}:1")
    print(f"      - Math Expectancy (per trade): ${stats.get('expectancy_cash_per_trade', 0.0):+.2f}")
    print(f"      - Gross Profit:                ${stats.get('gross_profit', 0.0):,.2f}")
    print(f"      - Gross Loss:                  ${stats.get('gross_loss', 0.0):,.2f}")
    print(f"      - Avg Holding Duration:        {stats.get('avg_holding_mins', 0.0):.1f} minutes")
    print(line)
    print("  [2] STREAK ANALYSIS (CONSECUTIVE LOSSES & WINS)")
    print(f"      - MAX CONSECUTIVE LOSSES:      {streaks.get('max_consecutive_losses', 0)} consecutive trades")
    print(f"      - Max Consecutive Wins:        {streaks.get('max_consecutive_wins', 0)} consecutive trades")
    print(f"      - Average Losing Streak:       {streaks.get('avg_consecutive_losses', 0.0):.2f} trades")
    print(f"      - Average Winning Streak:      {streaks.get('avg_consecutive_wins', 0.0):.2f} trades")
    print(f"      - Loss Streak Distribution:    {streaks.get('loss_streak_distribution', {})}")
    print(f"      - Win Streak Distribution:     {streaks.get('win_streak_distribution', {})}")
    print(line)
    print("  [3] CAPITAL PRESERVATION & DRAWDOWN VERIFICATION (GOAL: < 2.50%)")
    print(f"      - MAX TRAILING DRAWDOWN (%):   {dd.get('max_drawdown_pct', 0.0):.4f}%  [Target: < 2.50%]")
    status_dd = "PASS (Well within 2.50% limit)" if dd.get('max_drawdown_pct', 0.0) < 2.50 else "BREACH"
    print(f"      - Drawdown Constraint Status:  {status_dd}")
    print(f"      - Max Drawdown in Cash:        ${dd.get('max_drawdown_cash', 0.0):,.2f}")
    print(f"      - Average Drawdown:            {dd.get('avg_drawdown_pct', 0.0):.4f}%")
    print(f"      - Max Drawdown Duration:       {dd.get('max_drawdown_duration_bars', 0)} bars ({dd.get('max_drawdown_duration_time', 'N/A')})")
    print(line)
    print("  [4] STATISTICAL RUNS TEST (Z-SCORE SEQUENCE INDEPENDENCE)")
    print(f"      - Runs Test Z-Score:           {runs.get('z_score', 0.0):+.4f}")
    print(f"      - P-Value (Two-Tailed):        {runs.get('p_value', 0.0):.4f}")
    print(f"      - Total Observed Runs (R):     {runs.get('total_runs', 0)}")
    print(f"      - Expected Runs (E[R]):        {runs.get('expected_runs', 0.0):.2f}")
    print(f"      - 95% Independence (|Z|<1.96): {runs.get('independent_at_95pct', False)}")
    print(f"      - Statistical Interpretation:  {runs.get('interpretation', 'N/A')}")
    print(line)
    print("  [5] INSTITUTIONAL RISK-ADJUSTED RATIOS")
    print(f"      - Annualized Sharpe Ratio:     {risk.get('sharpe_ratio', 0.0):.2f}")
    print(f"      - Annualized Sortino Ratio:    {risk.get('sortino_ratio', 0.0):.2f}")
    print(f"      - Calmar Ratio:                {risk.get('calmar_ratio', 0.0):.2f}")
    print(f"      - DEFLATED SHARPE RATIO (DSR): {risk.get('deflated_sharpe_ratio', 0.0):.4f}  [Target: >= 0.95]")
    dsr_status = "CONFIRMED (Statistically Significant)" if risk.get('deflated_sharpe_ratio', 0.0) >= 0.95 else "UNVERIFIED"
    print(f"      - DSR Significance Status:     {dsr_status}")
    print(f"      - Trade Return Skewness:       {risk.get('trade_skewness', 0.0):+.3f}")
    print(f"      - Trade Return Kurtosis:       {risk.get('trade_kurtosis', 0.0):.3f}")
    print(line)
    print("  [6] MONTHLY COMPOUNDING BREAKDOWN")
    if not monthly_df.empty:
        print(monthly_df.to_string(index=False))
    else:
        print("      No monthly breakdown available.")
    print(sep + "\n")


def run_single_horizon(horizon: str, args: argparse.Namespace):
    logger.info("=== Commencing %s Horizon Backtest on %s (%s) (Enhanced: %s) ===", horizon, args.symbol, args.timeframe, args.enhanced)
    loader = MT5DataLoader()

    # 1. Load Data
    bars_df, h1_bars = load_long_duration_bars(loader, args.symbol, horizon, timeframe=args.timeframe, force_download=args.force_download)

    # Scale max_holding_bars and cooldown_bars for M5 vs M1
    holding_bars = args.max_holding_bars
    cooldown_bars = 288 if args.timeframe == "M5" else 1440
    if args.timeframe == "M5" and args.max_holding_bars == 1440:
        holding_bars = 288  # 288 M5 bars = 24h holding limit

    # 2. Generate Point-in-Time Signals
    weights_path = Path(args.weights)
    signals_df = generate_long_horizon_signals(
        bars_df=bars_df,
        h1_bars=h1_bars,
        weights_path=weights_path,
        threshold=args.threshold,
        min_conviction=args.min_conviction,
        trend_gate=args.trend_gate,
        session_filter=args.session_filter,
        k_sl=args.k_sl,
        k_tp=args.k_tp,
        min_sl_dist=args.min_sl_dist,
        enhanced=args.enhanced,
    )

    # 3. Initialize & Run Intra-Bar Engine
    engine = IntraBarExecutionEngine(
        symbol=args.symbol,
        initial_balance=args.initial_balance,
        risk_per_trade_pct=args.risk_pct,
        contract_size=1.0,
        min_lot=0.01,
        max_lot=10.0,
        max_holding_bars=holding_bars,
        cooldown_bars=cooldown_bars,
        enable_breakeven_ratchet=args.breakeven_ratchet and args.enhanced,
        be_activation_ratio=args.be_activation_ratio,
        slippage_points=2.0,
        point_size=0.01,
    )

    logger.info("Simulating execution across %d %s bars with real broker floating spreads...", len(bars_df), args.timeframe)
    t0 = time.perf_counter()
    trades_df, equity_series, stats = engine.run_bar_simulation(bars_df, signals_df)
    sim_time = time.perf_counter() - t0
    logger.info("Simulation completed in %.2f seconds (%.0f bars/second).", sim_time, len(bars_df) / max(sim_time, 0.001))

    # 4. Compute Monthly Performance
    monthly_df = compute_monthly_performance(trades_df, args.initial_balance)

    # 5. Output Master Report
    print_master_report(horizon, stats, monthly_df, args.initial_balance, args.risk_pct, len(bars_df), args.timeframe, enhanced=args.enhanced)

    # 6. Save Artifacts
    res_dir = Path("data/backtest")
    res_dir.mkdir(parents=True, exist_ok=True)

    tf_suffix = args.timeframe.lower()
    enh_suffix = "_enhanced" if args.enhanced else ""
    if not trades_df.empty:
        trades_p = res_dir / f"btcusd_{horizon.lower()}_{tf_suffix}{enh_suffix}_trades.parquet"
        trades_c = res_dir / f"btcusd_{horizon.lower()}_{tf_suffix}{enh_suffix}_trades.csv"
        trades_df.to_parquet(trades_p, compression="snappy")
        trades_df.to_csv(trades_c, index=False)
        logger.info("Saved %s trade logs to %s.", horizon, trades_p)

    eq_p = res_dir / f"btcusd_{horizon.lower()}_{tf_suffix}{enh_suffix}_equity.parquet"
    equity_series.to_frame().to_parquet(eq_p, compression="snappy")
    logger.info("Saved %s equity series to %s.", horizon, eq_p)

    return stats, monthly_df


def main():
    args = parse_args()
    if args.horizon == "ALL":
        run_single_horizon("3M", args)
        run_single_horizon("1Y", args)
    else:
        run_single_horizon(args.horizon, args.horizon == "1Y" and args or args)


if __name__ == "__main__":
    main()
