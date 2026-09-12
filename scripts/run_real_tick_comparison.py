"""
Comprehensive Real-Tick Comparative Engine.
Executes the Live Production Model (v2) and the Updated Advanced Model (v3)
across 1,000,000+ real millisecond broker ticks directly from MT5.
Simulates real bid/ask execution, floating spreads, slippage, and bracket orders.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from src.data.storage import ParquetStorage
from src.models.institutional_moe import TriDomainMoE
from src.features.tri_domain_features import build_synchronized_features
from src.models.vq_vae import MarketStateTokenizer
from src.backtest.tick_engine import EventDrivenTickEngine, SimulatedTrade
from src.utils.logger import setup_logger

logger = setup_logger("RealTickComparator")


def generate_model_signals(
    weights_path: Path,
    bars_df: pd.DataFrame,
    h1_bars: pd.DataFrame,
    threshold: float = 0.030,
    min_conviction: float = 0.18,
    k_sl: float = 2.5,
    k_tp: float = 7.5,
    min_sl_dist: float = 500.0,
    anti_adverse_filter: bool = False,
) -> pd.DataFrame:
    """Computes synchronized features and generates point-in-time signals."""
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    version = checkpoint.get("feature_version", checkpoint.get("version", "v1" if checkpoint["tech_dim"] == 4 else "v2"))
    hidden_dim = checkpoint.get("hidden_dim", 64 if version == "v3" else 48)

    clean_df, tech_cols, macro_cols = build_synchronized_features(bars_df, h1_bars, version=version)

    vq_tokenizer = None
    vq_latent_dim = 0
    if "vq_tokenizer_state_dict" in checkpoint and "vq_config" in checkpoint:
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
    model = model.to(device).eval()

    seq_len = 64 if version == "v3" else 32
    tech_data = clean_df[tech_cols].to_numpy(dtype=np.float32)
    macro_data = clean_df[macro_cols].to_numpy(dtype=np.float32)

    n_samples = len(clean_df) - seq_len
    if n_samples <= 0:
        raise ValueError(f"Insufficient bars ({len(clean_df)}) for sequence length {seq_len}.")

    fund_dim = checkpoint["fund_dim"]
    regime_dim = checkpoint["regime_dim"]
    current_z = np.zeros(max(fund_dim, regime_dim), dtype=np.float32)
    current_z[0] = 0.20  # neutral-positive crypto regime

    if fund_dim == 8 and "cvd_flow" in clean_df.columns:
        cvd_vals = clean_df["cvd_flow"].iloc[seq_len:].to_numpy(dtype=np.float32).reshape(-1, 1)
        basis_vals = clean_df["spread_basis"].iloc[seq_len:].to_numpy(dtype=np.float32).reshape(-1, 1)
        cvd_norm = (cvd_vals - np.mean(cvd_vals)) / (np.std(cvd_vals) + 1e-6)
        basis_norm = (basis_vals - np.mean(basis_vals)) / (np.std(basis_vals) + 1e-6)
        z_tile = np.tile(current_z[:6], (n_samples, 1))
        fund_data = np.hstack([z_tile, cvd_norm, basis_norm]).astype(np.float32)
    else:
        fund_data = np.tile(current_z[:fund_dim], (n_samples, 1)).astype(np.float32)

    z_reg_data = np.tile(current_z[:regime_dim], (n_samples, 1)).astype(np.float32)

    chunk_size = 2000
    y_preds_list = []
    convictions_list = []
    entropies_list = []

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
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
        w = out["weights"].cpu()
        ent = -(w * (w + 1e-10).log()).sum(dim=-1).numpy()
        entropies_list.append(ent)

    y_preds = np.concatenate(y_preds_list)
    convictions = np.concatenate(convictions_list)
    entropies = np.concatenate(entropies_list)

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
        sig_df["ofi"] = clean_df["normalized_ofi"].iloc[seq_len:].values
        sig_df["vpin"] = clean_df["vpin"].iloc[seq_len:].values

    # Session conditioning: scale conviction during London/NY overlap (12:00 - 18:00 UTC)
    if anti_adverse_filter:
        hours = pd.to_datetime(sig_df.index).hour
        is_london_ny = (hours >= 12) & (hours <= 18)
        sig_df["conviction"] = np.where(is_london_ny, sig_df["conviction"] * 1.15, sig_df["conviction"])

    # Directional threshold
    buy_mask = (sig_df["y_pred"] > threshold) & (sig_df["conviction"] >= min_conviction) & (sig_df["h1_trend"] > -0.20)
    sell_mask = (sig_df["y_pred"] < -threshold) & (sig_df["conviction"] >= min_conviction) & (sig_df["h1_trend"] < 0.20)

    if anti_adverse_filter:
        # Vector 1: Anti-adverse order book filter
        if "ofi" in sig_df.columns:
            buy_mask = buy_mask & ~(sig_df["ofi"] < -0.20) & ~(sig_df["vpin"] > 0.65)
            sell_mask = sell_mask & ~(sig_df["ofi"] > 0.20) & ~(sig_df["vpin"] > 0.65)

        # Vector 3: Router entropy gate (require higher conviction if experts are conflicted)
        conflicted_mask = sig_df["entropy"] >= 1.00
        buy_mask = buy_mask & ~(conflicted_mask & (sig_df["y_pred"] < 0.038))
        sell_mask = sell_mask & ~(conflicted_mask & (sig_df["y_pred"] > -0.038))

    sig_df["direction"] = np.where(buy_mask, "BUY", np.where(sell_mask, "SELL", "FLAT"))
    sig_df["sigma"] = np.maximum(sig_df["vol_20"], 0.003) * sig_df["close"]
    sig_df["sl_dist"] = np.maximum(k_sl * sig_df["sigma"], min_sl_dist)
    sig_df["tp_dist"] = np.maximum(k_tp * sig_df["sigma"], 2.0 * sig_df["sl_dist"])

    return sig_df


def run_tick_simulation_for_signals(
    ticks_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    symbol: str = "BTCUSD.x",
    initial_balance: float = 10000.0,
    risk_pct: float = 0.0010,
    max_holding_hours: float = 24.0,
    enable_breakeven_ratchet: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Runs tick simulation matching every tick to M5 bar signals."""
    engine = EventDrivenTickEngine(
        symbol=symbol,
        initial_balance=initial_balance,
        risk_per_trade_pct=risk_pct,
        contract_size=1.0,
        min_lot=0.01,
        max_lot=10.0,
        max_holding_seconds=max_holding_hours * 3600.0,
        slippage_points=2.0,
        point_size=0.01,
        enable_breakeven_ratchet=enable_breakeven_ratchet,
        be_activation_ratio=0.50,
    )

    sig_df = signals_df[signals_df["direction"] != "FLAT"].copy()
    sig_idx = pd.to_datetime(sig_df.index)
    if sig_idx.tz is not None:
        sig_idx = sig_idx.tz_convert("UTC").tz_localize(None)
    sig_times = sig_idx.values
    sig_records = {ts: row for ts, row in zip(sig_idx, sig_df.to_dict(orient="records"))}

    ticks_idx = pd.to_datetime(ticks_df.index)
    if ticks_idx.tz is not None:
        ticks_idx = ticks_idx.tz_convert("UTC").tz_localize(None)
        ticks_df = ticks_df.copy()
        ticks_df.index = ticks_idx

    def signal_generator(df_ticks: pd.DataFrame, current_idx: int) -> Optional[Dict[str, Any]]:
        t = df_ticks.index[current_idx]
        pos = np.searchsorted(sig_times, np.datetime64(t), side="right") - 1
        if pos < 0 or pos >= len(sig_times):
            return None
        sig_t = pd.Timestamp(sig_times[pos])
        diff_sec = (t - sig_t).total_seconds()
        if diff_sec > 300.0 or diff_sec < 0:
            return None
        rec = sig_records.get(sig_t)
        if rec:
            return {
                "direction": rec["direction"],
                "conviction": rec["conviction"],
                "sl_dist": rec["sl_dist"],
                "tp_dist": rec["tp_dist"],
            }
        return None

    trades_df, equity_series, stats = engine.run_tick_simulation(
        ticks_df=ticks_df,
        signals_generator=signal_generator,
        bar_interval_ticks=50,
    )
    return trades_df, equity_series, stats


def analyze_trades(trades_df: pd.DataFrame, initial_balance: float) -> Dict[str, Any]:
    if trades_df.empty:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "net_pnl": 0.0, "return_pct": 0.0, "max_drawdown_pct": 0.0,
            "payoff_ratio": 0.0, "expectancy": 0.0, "max_loss_streak": 0,
            "max_win_streak": 0, "exits": {},
        }

    n = len(trades_df)
    wins = (trades_df["pnl_cash"] > 0).sum()
    losses = (trades_df["pnl_cash"] <= 0).sum()
    win_rate = (wins / n) * 100.0 if n > 0 else 0.0

    g_win = trades_df[trades_df["pnl_cash"] > 0]["pnl_cash"].sum()
    g_loss = abs(trades_df[trades_df["pnl_cash"] <= 0]["pnl_cash"].sum())
    pf = g_win / g_loss if g_loss > 0 else (99.0 if g_win > 0 else 0.0)

    avg_win = g_win / wins if wins > 0 else 0.0
    avg_loss = g_loss / losses if losses > 0 else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else 0.0

    net_pnl = trades_df["pnl_cash"].sum()
    ret_pct = (net_pnl / initial_balance) * 100.0
    expectancy = net_pnl / n if n > 0 else 0.0

    cum = trades_df["pnl_cash"].cumsum() + initial_balance
    peak = cum.cummax()
    dd_series = (peak - cum) / peak * 100.0
    mdd = dd_series.max()

    # Streaks
    is_win = (trades_df["pnl_cash"] > 0).astype(int).values
    cur_win, max_win, cur_loss, max_loss = 0, 0, 0, 0
    for w in is_win:
        if w == 1:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)

    exits = trades_df["exit_reason"].value_counts().to_dict()

    return {
        "total_trades": n,
        "win_rate": win_rate,
        "profit_factor": pf,
        "net_pnl": net_pnl,
        "return_pct": ret_pct,
        "max_drawdown_pct": mdd,
        "payoff_ratio": payoff,
        "expectancy": expectancy,
        "max_loss_streak": max_loss,
        "max_win_streak": max_win,
        "exits": exits,
    }


def main():
    print("=" * 88)
    print("        REAL-TICK INSTITUTIONAL COMPARISON: LIVE (v2) vs UPDATED ADVANCED (v3)")
    print("        Executed across Real MT5 Millisecond Broker Ticks (Ask/Bid Spread & Slippage)")
    print("=" * 88)

    # 1. Load Real Millisecond Ticks (3,000,000 Real MT5 Ticks across 14 full days)
    tick_file = WORKSPACE / "data/cache/ticks/BTCUSD_x/ticks_3000000.parquet"
    if not tick_file.exists():
        tick_file = WORKSPACE / "data/cache/ticks/BTCUSD_x/ticks_1000000.parquet"
        print(f"Error: Tick file {tick_file} does not exist.")
        sys.exit(1)

    print(f"Loading real broker ticks from {tick_file}...")
    t0 = time.perf_counter()
    ticks_df = pd.read_parquet(tick_file)
    print(f"Loaded {len(ticks_df):,} real ticks in {time.perf_counter() - t0:.2f}s.")
    print(f"Time Range: {ticks_df.index[0]} to {ticks_df.index[-1]}")

    # 2. Resample Ticks to M5 bars
    print("\nResampling ticks to M5 bars and loading H1 macro context...")
    ohlc = ticks_df["bid"].resample("5min").ohlc().dropna()
    vol = ticks_df["volume"].resample("5min").sum().reindex(ohlc.index).fillna(1.0)
    spread = (ticks_df["ask"] - ticks_df["bid"]).resample("5min").mean().reindex(ohlc.index).fillna(65.0)

    m5_bars = ohlc.copy()
    m5_bars["tick_volume"] = vol
    m5_bars["spread"] = spread

    storage = ParquetStorage(base_cache_dir=str(WORKSPACE / "data/cache"))
    h1_bars = storage.load_bars("BTCUSD.x", "H1")
    # Slice H1 bars around the tick range
    h1_bars = h1_bars.loc[:ticks_df.index[-1]].tail(2000)

    # Load 5-year cached M5 bars to ensure warm lookbacks for fractional differencing
    full_m5 = storage.load_bars("BTCUSD.x", "M5")
    tick_start = ticks_df.index[0]
    warm_m5 = full_m5.loc[:tick_start].tail(500)
    combined_m5 = pd.concat([warm_m5, m5_bars])
    combined_m5 = combined_m5[~combined_m5.index.duplicated(keep="last")].sort_index()

    print(f"Total M5 bars for evaluation (including lookback): {len(combined_m5)}")

    # 3. Generate Signals for all 3 configurations
    w_v2 = WORKSPACE / "weights/btcusd_tri_domain_v2.pt"
    w_v3 = WORKSPACE / "weights/btcusd_advanced_v1.pt"

    print("\n[1/3] Generating signals for Live Production System (v2)...")
    sig_v2 = generate_model_signals(w_v2, combined_m5, h1_bars, threshold=0.030, min_conviction=0.18)

    print("[2/3] Generating signals for Updated Advanced Model (v3 - MBM+VQ-VAE)...")
    sig_v3 = generate_model_signals(w_v3, combined_m5, h1_bars, threshold=0.030, min_conviction=0.18)

    print("[3/3] Generating signals for Live System + Anti-Adverse Gating Filter...")
    sig_v2_gated = generate_model_signals(w_v2, combined_m5, h1_bars, threshold=0.030, min_conviction=0.18, anti_adverse_filter=True)

    # 4. Simulate on Real Millisecond Ticks
    print("\n" + "-" * 88)
    print(f"Executing Real-Tick Simulation on {len(ticks_df):,} Millisecond Ticks...")
    print("-" * 88)

    print("  -> Simulating Live System (v2)...")
    tr_v2, eq_v2, stats_v2 = run_tick_simulation_for_signals(ticks_df, sig_v2, enable_breakeven_ratchet=False)

    print("  -> Simulating Updated Advanced System (v3)...")
    tr_v3, eq_v3, stats_v3 = run_tick_simulation_for_signals(ticks_df, sig_v3, enable_breakeven_ratchet=False)

    print("  -> Simulating Enhanced Live System (v2 + Anti-Adverse + Breakeven Ratchet + Session Sizing)...")
    tr_gated, eq_gated, stats_gated = run_tick_simulation_for_signals(ticks_df, sig_v2_gated, enable_breakeven_ratchet=True)

    # 5. Analyze and Report Everything
    res_v2 = analyze_trades(tr_v2, 10000.0)
    res_v3 = analyze_trades(tr_v3, 10000.0)
    res_gated = analyze_trades(tr_gated, 10000.0)

    sep = "=" * 88
    line = "-" * 88

    print("\n" + sep)
    print("               HEAD-TO-HEAD REAL-TICK BACKTEST COMPARISON REPORT")
    print(f"   Tested across {len(ticks_df):,} Actual MT5 Broker Ticks ({ticks_df.index[0].strftime('%Y-%m-%d %H:%M')} to {ticks_df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    print(sep)
    print(f"{'Performance Metric':<34} | {'Live v2 (Active)':<16} | {'Updated v3 (Adv)':<16} | {'v2 Enhanced (All)':<16}")
    print(line)
    print(f"{'Total Executed Trades':<34} | {res_v2['total_trades']:<16} | {res_v3['total_trades']:<16} | {res_gated['total_trades']:<16}")
    print(f"{'Win Rate (%)':<34} | {res_v2['win_rate']:<15.2f}% | {res_v3['win_rate']:<15.2f}% | {res_gated['win_rate']:<15.2f}%")
    print(f"{'Profit Factor':<34} | {res_v2['profit_factor']:<16.2f} | {res_v3['profit_factor']:<16.2f} | {res_gated['profit_factor']:<16.2f}")
    print(f"{'Payoff Ratio (Avg Win / Loss)':<34} | {res_v2['payoff_ratio']:<14.2f}:1 | {res_v3['payoff_ratio']:<14.2f}:1 | {res_gated['payoff_ratio']:<14.2f}:1")
    print(f"{'Math Expectancy ($/trade)':<34} | ${res_v2['expectancy']:<15.2f} | ${res_v3['expectancy']:<15.2f} | ${res_gated['expectancy']:<15.2f}")
    print(f"{'Net Profit ($ on $10k)':<34} | ${res_v2['net_pnl']:<15.2f} | ${res_v3['net_pnl']:<15.2f} | ${res_gated['net_pnl']:<15.2f}")
    print(f"{'Total Return (%)':<34} | {res_v2['return_pct']:<15.2f}% | {res_v3['return_pct']:<15.2f}% | {res_gated['return_pct']:<15.2f}%")
    print(f"{'Max Trailing Drawdown (%)':<34} | {res_v2['max_drawdown_pct']:<15.4f}% | {res_v3['max_drawdown_pct']:<15.4f}% | {res_gated['max_drawdown_pct']:<15.4f}%")
    print(f"{'Max Consecutive Losses':<34} | {res_v2['max_loss_streak']:<16} | {res_v3['max_loss_streak']:<16} | {res_gated['max_loss_streak']:<16}")
    print(f"{'Max Consecutive Wins':<34} | {res_v2['max_win_streak']:<16} | {res_v3['max_win_streak']:<16} | {res_gated['max_win_streak']:<16}")
    print(line)
    print("EXIT BREAKDOWN:")
    print(f"{'  - Take Profit Exits':<34} | {res_v2['exits'].get('TAKE_PROFIT', 0):<16} | {res_v3['exits'].get('TAKE_PROFIT', 0):<16} | {res_gated['exits'].get('TAKE_PROFIT', 0):<16}")
    print(f"{'  - Breakeven Ratchet Exits':<34} | {res_v2['exits'].get('BREAKEVEN', 0):<16} | {res_v3['exits'].get('BREAKEVEN', 0):<16} | {res_gated['exits'].get('BREAKEVEN', 0):<16}")
    print(f"{'  - Stop Loss Exits':<34} | {res_v2['exits'].get('STOP_LOSS', 0):<16} | {res_v3['exits'].get('STOP_LOSS', 0):<16} | {res_gated['exits'].get('STOP_LOSS', 0):<16}")
    print(f"{'  - Time Barrier Exits':<34} | {res_v2['exits'].get('TIME_BARRIER', 0):<16} | {res_v3['exits'].get('TIME_BARRIER', 0):<16} | {res_gated['exits'].get('TIME_BARRIER', 0):<16}")
    print(sep + "\n")


if __name__ == "__main__":
    main()
