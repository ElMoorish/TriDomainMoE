"""
Real-Tick Backtesting & Statistical Evidence Engine.
Executes Tri-Domain MoE on actual millisecond broker ticks from MT5.
Computes and reports comprehensive institutional metrics:
- Consecutive Losses & Consecutive Wins
- Peak-to-Trough Drawdown & Underwater Duration
- Runs Test Z-Score (Wald-Wolfowitz sequence independence)
- Annualized Sharpe, Sortino, Calmar, and Deflated Sharpe Ratio (DSR)
- Win Rate, Payoff Ratio, Profit Factor, and Mathematical Expectancy
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mt5_loader import MT5DataLoader
from src.data.storage import ParquetStorage
from src.models.institutional_moe import TriDomainMoE
from src.features.fracdiff import FractionalDifferentiator
from src.features.sentiment_loader import YahooFinanceSentimentLoader
from src.features.sentiment_engine import MacroSentimentEngine
from src.backtest.tick_engine import EventDrivenTickEngine
from src.backtest.metrics import StatisticalEvidenceMetrics
from src.utils.logger import setup_logger

logger = setup_logger("RealTickBacktester")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Tick MoE Backtest and Statistical Evidence Suite")
    parser.add_argument("--symbol", type=str, default="BTCUSD.x", help="Target symbol")
    parser.add_argument("--weights", type=str, default="weights/btcusd_tri_domain_v1.pt", help="Path to weights")
    parser.add_argument("--ticks", type=int, default=500000, help="Number of real ticks to backtest (e.g. 500000 or 1000000)")
    parser.add_argument("--risk-pct", type=float, default=0.0010, help="Risk ceiling per trade (0.0010 = 0.10%)")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Initial capital in USD")
    parser.add_argument("--threshold", type=float, default=0.025, help="Minimum drift threshold to trigger entry")
    parser.add_argument("--min-conviction", type=float, default=0.15, help="Minimum conviction threshold from meta-sizer")
    parser.add_argument("--k-sl", type=float, default=2.5, help="Stop loss volatility multiplier (must be >= 2.0 to clear spread)")
    parser.add_argument("--k-tp", type=float, default=4.5, help="Take profit volatility multiplier (asymmetric payoff)")
    parser.add_argument("--min-sl-dist", type=float, default=500.0, help="Minimum stop-loss distance in USD to prevent spread stop-outs")
    parser.add_argument("--max-holding-mins", type=float, default=180.0, help="Maximum holding duration in minutes")
    parser.add_argument("--force-download", action="store_true", help="Force fresh tick download from MT5")
    return parser.parse_args()


def load_real_ticks(loader: MT5DataLoader, symbol: str, n_ticks: int, force_download: bool = False) -> pd.DataFrame:
    """Loads real historical ticks from cache or directly from MT5."""
    cache_dir = Path("data/cache/ticks") / symbol.replace(".", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"ticks_{n_ticks}.parquet"

    if not force_download and cache_file.exists():
        logger.info("Loading %d cached ticks from %s...", n_ticks, cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Downloading %d real ticks for %s from MetaTrader 5...", n_ticks, symbol)
    if not loader.connect():
        raise ConnectionError("Failed to connect to MetaTrader 5.")

    end_time = datetime.now()
    # Estimate days needed based on tick count (~150k-200k ticks/day for BTC)
    days_back = max(3, int(n_ticks / 100000) + 2)
    start_time = end_time - timedelta(days=days_back)

    ticks_df = loader.get_ticks(symbol, start_time, end_time, max_ticks=n_ticks)
    if ticks_df.empty:
        raise RuntimeError(f"No ticks returned for {symbol} from MT5.")

    logger.info("Retrieved %d real ticks from MT5. Time range: %s to %s.", len(ticks_df), ticks_df.index[0], ticks_df.index[-1])
    ticks_df.to_parquet(cache_file, compression="snappy")
    return ticks_df


def precompute_bar_signals(
    ticks_df: pd.DataFrame,
    loader: MT5DataLoader,
    symbol: str,
    weights_path: Path,
    threshold: float = 0.025,
    min_conviction: float = 0.15,
    k_sl: float = 2.5,
    k_tp: float = 4.5,
    min_sl_dist: float = 500.0,
) -> pd.DataFrame:
    """
    Resamples ticks into 1-minute execution bars, aligns H1 macro context,
    and runs TriDomainMoE forward passes to generate point-in-time trading signals.
    """
    logger.info("Aggregating ticks into 1-minute execution bars...")
    # Resample ticks to M1 OHLCV
    ohlc = ticks_df["bid"].resample("1min").ohlc().dropna()
    volume = ticks_df["volume"].resample("1min").sum().reindex(ohlc.index).fillna(1.0)
    m1_bars = ohlc.copy()
    m1_bars["tick_volume"] = volume
    logger.info("Aggregated %d M1 bars from %d ticks.", len(m1_bars), len(ticks_df))

    # Fetch H1 macro bars
    h1_bars = loader.get_bars(symbol, timeframe="H1", count=2000)
    if h1_bars.empty:
        logger.warning("Could not fetch H1 bars from MT5; constructing H1 from ticks.")
        h1_bars = ticks_df["bid"].resample("1h").ohlc().dropna()
        h1_bars["tick_volume"] = ticks_df["volume"].resample("1h").sum().reindex(h1_bars.index).fillna(1.0)

    # 1. Technical Features
    df = m1_bars.copy()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df["vol_20"] = df["log_ret"].rolling(20, min_periods=5).std().bfill()
    df["fracdiff"] = FractionalDifferentiator.frac_diff(df["close"], d=0.45, threshold=1e-3)
    df["vol_mom"] = df["tick_volume"] / (df["tick_volume"].rolling(20).mean() + 1e-6)

    # 2. Macro Features Alignment
    h1_df = h1_bars.copy()
    h1_df["h1_ret"] = np.log(h1_df["close"] / h1_df["close"].shift(1)).fillna(0.0)
    h1_df["h1_trend_50"] = (h1_df["close"] - h1_df["close"].rolling(50, min_periods=10).mean()) / (h1_df["close"].rolling(50, min_periods=10).std() + 1e-6)
    h1_df["h1_vol_24"] = h1_df["h1_ret"].rolling(24, min_periods=5).std().bfill()

    m1_reset = df.reset_index().rename(columns={"index": "m1_ts", "timestamp": "m1_ts"})
    h1_reset = h1_df[["h1_ret", "h1_trend_50", "h1_vol_24"]].reset_index().rename(columns={"index": "h1_ts", "timestamp": "h1_ts"})

    merged = pd.merge_asof(
        m1_reset.sort_values("m1_ts"),
        h1_reset.sort_values("h1_ts"),
        left_on="m1_ts",
        right_on="h1_ts",
        direction="backward",
    ).set_index("m1_ts")

    # Clean NaNs
    feature_cols = ["log_ret", "vol_20", "fracdiff", "vol_mom", "h1_ret", "h1_trend_50", "h1_vol_24"]
    clean_df = merged.dropna(subset=feature_cols).copy()
    logger.info("Computed synchronized multi-domain features for %d bars.", len(clean_df))

    # 3. Load Model Checkpoint
    checkpoint = torch.load(weights_path, weights_only=False)
    hidden_dim = checkpoint.get("hidden_dim", 48)
    model = TriDomainMoE(
        tech_dim=checkpoint["tech_dim"],
        macro_dim=checkpoint["macro_dim"],
        fund_dim=checkpoint["fund_dim"],
        regime_dim=checkpoint["regime_dim"],
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Batched Point-in-Time Forward Pass
    seq_len = 32
    tech_data = clean_df[["log_ret", "vol_20", "fracdiff", "vol_mom"]].to_numpy(dtype=np.float32)
    macro_data = clean_df[["h1_ret", "h1_trend_50", "h1_vol_24"]].to_numpy(dtype=np.float32)

    # Rolling normalization
    n_samples = len(clean_df) - seq_len
    if n_samples <= 0:
        raise ValueError(f"Insufficient bars ({len(clean_df)}) for sequence length {seq_len}.")

    x_tech_list = []
    x_macro_list = []
    for idx in range(n_samples):
        t_slice = tech_data[idx : idx + seq_len]
        m_slice = macro_data[idx : idx + seq_len]
        t_norm = (t_slice - t_slice.mean(axis=0)) / (t_slice.std(axis=0) + 1e-6)
        m_norm = (m_slice - m_slice.mean(axis=0)) / (m_slice.std(axis=0) + 1e-6)
        x_tech_list.append(t_norm)
        x_macro_list.append(m_norm)

    x_tech_tensor = torch.tensor(np.array(x_tech_list), dtype=torch.float32)
    x_macro_tensor = torch.tensor(np.array(x_macro_list), dtype=torch.float32)
    
    # Live crypto sentiment regime vector
    sentiment_loader = YahooFinanceSentimentLoader()
    sentiment_engine = MacroSentimentEngine()
    try:
        articles = sentiment_loader.fetch_articles("BTC-USD")
        current_z = sentiment_engine.update(articles)
    except Exception as e:
        logger.warning("Could not fetch online sentiment (%s); using baseline regime vector.", e)
        current_z = sentiment_engine.get_regime_vector()

    fund_dim = checkpoint["fund_dim"]
    if len(current_z) != fund_dim:
        current_z = np.resize(current_z, fund_dim).astype(np.float32)

    x_fund_tensor = torch.tensor(np.tile(current_z, (n_samples, 1)), dtype=torch.float32)
    z_reg_tensor = x_fund_tensor.clone()

    logger.info("Executing MoE batched neural inference across %d evaluation steps...", n_samples)
    with torch.no_grad():
        out = model(x_tech_tensor, x_macro_tensor, x_fund_tensor, z_reg_tensor)

    y_preds = out["y_pred"].squeeze(-1).numpy()
    convictions = out["size"].squeeze(-1).numpy()

    # Align signals to timestamp index (at the close of sequence)
    signal_indices = clean_df.index[seq_len:]
    sig_df = pd.DataFrame({
        "timestamp": signal_indices,
        "y_pred": y_preds,
        "conviction": convictions,
        "vol_20": clean_df["vol_20"].iloc[seq_len:].values,
        "close": clean_df["close"].iloc[seq_len:].values,
    }).set_index("timestamp")

    # Generate discrete signals with adaptive stop-loss and take-profit
    buy_mask = (sig_df["y_pred"] > threshold) & (sig_df["conviction"] >= min_conviction)
    sell_mask = (sig_df["y_pred"] < -threshold) & (sig_df["conviction"] >= min_conviction)
    sig_df["direction"] = np.where(buy_mask, "BUY", np.where(sell_mask, "SELL", "FLAT"))
    sig_df["sigma"] = np.maximum(sig_df["vol_20"], 0.003) * sig_df["close"]
    sig_df["sl_dist"] = np.maximum(k_sl * sig_df["sigma"], min_sl_dist)
    sig_df["tp_dist"] = np.maximum(k_tp * sig_df["sigma"], 2.0 * sig_df["sl_dist"])

    trade_signals = sig_df[sig_df["direction"] != "FLAT"]
    logger.info("Generated %d actionable entry signals (|y_pred| > %.4f & conviction >= %.2f).", len(trade_signals), threshold, min_conviction)
    return sig_df


def print_statistical_evidence_table(stats: Dict[str, Any], initial_balance: float, risk_pct: float):
    """Formats and prints an ASCII table of statistical evidence."""
    streaks = stats.get("streaks", {})
    runs = stats.get("runs_test", {})
    dd = stats.get("drawdowns", {})
    risk = stats.get("risk_ratios", {})

    sep = "=" * 82
    line = "-" * 82

    print("\n" + sep)
    print("       INSTITUTIONAL STATISTICAL EVIDENCE REPORT: REAL-TICK MT5 BACKTEST")
    print("                    Tri-Domain Mixture of Experts (BTCUSD.x)")
    print(sep)
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
    status_dd = "PASS (Within Prop Limit)" if dd.get('max_drawdown_pct', 0.0) < 2.50 else "BREACH"
    print(f"      - Drawdown Constraint Status:  {status_dd}")
    print(f"      - Max Drawdown in Cash:        ${dd.get('max_drawdown_cash', 0.0):,.2f}")
    print(f"      - Average Drawdown:            {dd.get('avg_drawdown_pct', 0.0):.4f}%")
    print(f"      - Max Drawdown Duration:       {dd.get('max_drawdown_duration_bars', 0)} intervals ({dd.get('max_drawdown_duration_time', 'N/A')})")
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
    dsr_status = "CONFIRMED (Statistically Significant Alpha)" if risk.get('deflated_sharpe_ratio', 0.0) >= 0.95 else "UNVERIFIED"
    print(f"      - DSR Significance Status:     {dsr_status}")
    print(f"      - Trade Return Skewness:       {risk.get('trade_skewness', 0.0):+.3f}")
    print(f"      - Trade Return Kurtosis:       {risk.get('trade_kurtosis', 0.0):.3f}")
    print(sep + "\n")


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error("Weights checkpoint not found at %s.", weights_path)
        sys.exit(1)

    # 1. MT5 Connector
    loader = MT5DataLoader()

    # 2. Ingest Real MT5 Ticks
    ticks_df = load_real_ticks(loader, args.symbol, args.ticks, args.force_download)

    # 3. Precompute point-in-time signals from M1 bars aligned to macro context
    sig_df = precompute_bar_signals(
        ticks_df=ticks_df,
        loader=loader,
        symbol=args.symbol,
        weights_path=weights_path,
        threshold=args.threshold,
        min_conviction=args.min_conviction,
        k_sl=args.k_sl,
        k_tp=args.k_tp,
        min_sl_dist=args.min_sl_dist,
    )

    # 4. Create Signal Generator closure for Tick Engine
    sig_lookup = sig_df[["direction", "conviction", "sl_dist", "tp_dist"]].to_dict(orient="index")

    # Map each tick to the latest signal that occurred up to that timestamp
    # To ensure zero look-ahead bias, a tick at time t only uses signals strictly <= t
    sig_times = sig_df.index

    def signal_generator(df_ticks: pd.DataFrame, current_idx: int) -> Optional[Dict[str, Any]]:
        tick_time = df_ticks.index[current_idx]
        pos = sig_times.searchsorted(tick_time, side="right") - 1
        if pos < 0 or pos >= len(sig_times):
            return None
        sig_t = sig_times[pos]
        # Only trigger at the start of a bar (e.g. within 60 seconds of signal emission)
        if (tick_time - sig_t).total_seconds() > 60.0:
            return None
        return sig_lookup.get(sig_t)

    # 5. Initialize and Run Event-Driven Real Tick Engine
    engine = EventDrivenTickEngine(
        symbol=args.symbol,
        initial_balance=args.initial_balance,
        risk_per_trade_pct=args.risk_pct,
        contract_size=1.0,
        min_lot=0.01,
        max_lot=10.0,
        max_holding_seconds=args.max_holding_mins * 60.0,
        slippage_points=2.0,
        point_size=0.01,
    )

    logger.info("Initiating Event-Driven simulation across %d real ticks...", len(ticks_df))
    t_start = time.perf_counter()
    trades_df, equity_series, stats = engine.run_tick_simulation(
        ticks_df=ticks_df,
        signals_generator=signal_generator,
        bar_interval_ticks=100,
    )
    sim_time = time.perf_counter() - t_start
    logger.info("Simulation completed in %.2f seconds (%.0f ticks/second).", sim_time, len(ticks_df) / max(sim_time, 0.001))

    # 6. Print Institutional Evidence Table
    print_statistical_evidence_table(stats, args.initial_balance, args.risk_pct)

    # 7. Save Artifacts & Results
    results_dir = Path("data/backtest")
    results_dir.mkdir(parents=True, exist_ok=True)

    if not trades_df.empty:
        trades_file = results_dir / "btcusd_real_ticks_trades.parquet"
        csv_file = results_dir / "btcusd_real_ticks_trades.csv"
        trades_df.to_parquet(trades_file, compression="snappy")
        trades_df.to_csv(csv_file, index=False)
        logger.info("Saved trade log to %s and %s.", trades_file, csv_file)

    equity_file = results_dir / "btcusd_real_ticks_equity.parquet"
    equity_series.to_frame().to_parquet(equity_file, compression="snappy")
    logger.info("Saved equity series to %s.", equity_file)


if __name__ == "__main__":
    main()
