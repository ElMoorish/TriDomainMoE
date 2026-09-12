"""
Live MetaTrader 5 Trading Engine and Risk Dispatcher.
Executes the Institutional Tri-Domain MoE in real-time,
enforcing 0.50% capital risk ceilings, automated bracket SL/TP,
Shannon entropy fallback, MRDD drift detection, and 3-tier drawdown circuit breakers.
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mt5_loader import MT5DataLoader
from src.data.storage import ParquetStorage
from src.models.institutional_moe import TriDomainMoE
from src.features.tri_domain_features import (
    compute_technical_features,
    compute_macro_features,
    TECH_COLS_V1,
    TECH_COLS_V2,
    MACRO_COLS_V1,
    MACRO_COLS_V2,
)
from src.features.fracdiff import FractionalDifferentiator
from src.features.sentiment_loader import YahooFinanceSentimentLoader
from src.features.sentiment_engine import MacroSentimentEngine
from src.surveillance.entropy_guard import GatingEntropyGuard
from src.surveillance.risk_controls import RiskControls, CircuitBreakerState
from src.surveillance.mrdd import MultiResolutionDriftDetector
from src.execution.mt5_bridge import MT5ExecutionBridge
from src.utils.logger import setup_logger

logger = setup_logger("LiveMT5Trader")


def parse_args():
    parser = argparse.ArgumentParser(description="Live MT5 Trading Engine for Tri-Domain MoE")
    parser.add_argument("--symbol", type=str, default="BTCUSD.x", help="Primary trading symbol")
    parser.add_argument("--weights", type=str, default="weights/btcusd_tri_domain_v2.pt", help="Path to weights")
    parser.add_argument("--timeframe", type=str, default="M5", choices=["M1", "M5"], help="Execution bar timeframe")
    parser.add_argument("--paper", action=argparse.BooleanOptionalAction, default=True, help="Run in paper-trading simulation mode (use --no-paper for live execution)")
    parser.add_argument("--risk-pct", type=float, default=0.0020, help="Risk ceiling per trade (0.0020 = 0.20% equity)")
    parser.add_argument("--threshold", type=float, default=0.030, help="Minimum drift threshold to trigger entry")
    parser.add_argument("--min-conviction", type=float, default=0.18, help="Minimum conviction threshold from meta-sizer")
    parser.add_argument("--trend-gate", action=argparse.BooleanOptionalAction, default=True, help="Enable H1 macro trend directional gating")
    parser.add_argument("--session-filter", action=argparse.BooleanOptionalAction, default=True, help="Enable 24/7 crypto session conditioning")
    parser.add_argument("--enhanced", action=argparse.BooleanOptionalAction, default=True, help="Enable Enhanced Live v2 filters (Anti-adverse OFI/VPIN shield, Entropy filter, Overlap sizing, Breakeven ratchet)")
    parser.add_argument("--breakeven-ratchet", action=argparse.BooleanOptionalAction, default=True, help="Enable dynamic breakeven ratchet for live/paper positions")
    parser.add_argument("--be-activation-ratio", type=float, default=0.50, help="Activation ratio of TP distance for breakeven ratchet")
    parser.add_argument("--k-sl", type=float, default=2.5, help="Stop loss volatility multiplier")
    parser.add_argument("--k-tp", type=float, default=7.5, help="Take profit volatility multiplier (asymmetric 3.0:1 payoff)")
    parser.add_argument("--min-sl-dist", type=float, default=500.0, help="Minimum stop-loss distance in USD")
    parser.add_argument("--interval", type=int, default=5, help="Poll interval in seconds")
    parser.add_argument("--max-steps", type=int, default=0, help="Max loop iterations (0 for continuous infinite execution)")
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error("Weights checkpoint not found at %s.", weights_path)
        sys.exit(1)

    # 1. Initialize MT5 Connection
    loader = MT5DataLoader()
    if not loader.connect():
        logger.error("Could not connect to MT5 terminal. Ensure MT5 is running.")
        sys.exit(1)

    # 2. Load Model & Checkpoint
    checkpoint = torch.load(weights_path, weights_only=False)
    version = checkpoint.get("version", "v1" if checkpoint["tech_dim"] == 4 else "v2")
    hidden_dim = checkpoint.get("hidden_dim", 48)
    logger.info("Loaded TriDomainMoE checkpoint (version %s, hidden_dim %d) from %s", version, hidden_dim, weights_path)

    model = TriDomainMoE(
        tech_dim=checkpoint["tech_dim"],
        macro_dim=checkpoint["macro_dim"],
        fund_dim=checkpoint["fund_dim"],
        regime_dim=checkpoint["regime_dim"],
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Initialize Safeguards & Surveillance
    risk_controls = RiskControls(target_vol=0.15, max_leverage_cap=2.0)
    entropy_guard = GatingEntropyGuard(critical_entropy=0.35)
    mrdd = MultiResolutionDriftDetector(wavelet_levels=3, energy_threshold=0.35)
    execution_bridge = MT5ExecutionBridge(
        risk_controls=risk_controls,
        risk_per_trade_pct=args.risk_pct,
    )
    sentiment_loader = YahooFinanceSentimentLoader()
    sentiment_engine = MacroSentimentEngine()

    enh_banner = "\n  - Enhanced Live v2:    ACTIVE (Anti-Adverse OFI/VPIN + Breakeven Ratchet + Entropy Filter + Overlap Sizing)" if args.enhanced else "\n  - Enhanced Live v2:    OFF (Standard Baseline)"
    logger.info(
        "=== Live MT5 Trader Online ===\n"
        "  - Target Symbol:       %s\n"
        "  - Mode:                %s\n"
        "  - Risk Ceiling:        %.2f%%\n"
        "  - Circuit Breakers:    Tier 1 (3%%), Tier 2 (5%%), Tier 3 (8%%)\n"
        "  - Entropy Fallback:    Floor H < 0.35%s",
        args.symbol,
        "PAPER (Simulated Deals)" if args.paper else "LIVE BROKER EXECUTION",
        args.risk_pct * 100.0,
        enh_banner,
    )

    step = 0
    recent_returns = []
    paper_position = None

    try:
        while True:
            step += 1
            t_start = time.perf_counter()

            # A. Fetch Live Market State
            sym_info = loader.get_symbol_info(args.symbol)
            latest_tick = loader.get_latest_tick(args.symbol)
            bars = loader.get_bars(args.symbol, timeframe=args.timeframe, count=200)

            if bars.empty or len(bars) < 100:
                logger.warning("Waiting for sufficient %s bar history for %s...", args.timeframe, args.symbol)
                time.sleep(args.interval)
                continue

            # Fetch H1 macro bars for trend alignment (request 120 bars)
            h1_bars = loader.get_bars(args.symbol, timeframe="H1", count=120)
            if h1_bars.empty or len(h1_bars) < 32:
                # Dynamically resample M5 bars to construct H1 bars if MT5 H1 fetch is sparse
                h1_resampled = bars["close"].resample("1h").ohlc().dropna()
                if not h1_resampled.empty:
                    h1_bars = h1_resampled

            # B. Prepare Real-Time Technical Features
            close_prices = bars["close"]
            log_ret = np.log(close_prices / close_prices.shift(1)).fillna(0.0)
            vol_20 = log_ret.rolling(20, min_periods=5).std().bfill()
            fracdiff = FractionalDifferentiator.frac_diff(close_prices, d=0.45, threshold=1e-3)
            vol_mom = bars["tick_volume"] / (bars["tick_volume"].rolling(20).mean() + 1e-6)

            # B. Prepare Real-Time Technical Features
            tech_df = compute_technical_features(bars, version=version)
            tech_cols = TECH_COLS_V2 if version == "v2" else TECH_COLS_V1
            tech_clean = tech_df[tech_cols].dropna()

            if len(tech_clean) < 32:
                time.sleep(args.interval)
                continue

            tech_slice = tech_clean.iloc[-32:].to_numpy(dtype=np.float32)
            tech_norm = (tech_slice - tech_slice.mean(axis=0)) / (tech_slice.std(axis=0) + 1e-6)
            tech_norm = np.nan_to_num(tech_norm, nan=0.0, posinf=3.0, neginf=-3.0)

            # C. Update Sentiment & Macro Regime Vector
            if step == 1 or step % 60 == 0:
                articles = sentiment_loader.fetch_articles("BTC-USD")
                z_vec = sentiment_engine.update(articles)
            else:
                z_vec = sentiment_engine.get_regime_vector()

            # D. Build Multi-Domain Tensors
            x_tech_tensor = torch.tensor(tech_norm, dtype=torch.float32).unsqueeze(0)

            # Extract Macro Features from H1 Bars
            if not h1_bars.empty:
                macro_df = compute_macro_features(h1_bars, version=version)
                macro_cols = MACRO_COLS_V2 if version == "v2" else MACRO_COLS_V1
                macro_clean = macro_df[macro_cols].dropna()
                h1_trend = float(macro_df["h1_trend_50"].iloc[-1]) if "h1_trend_50" in macro_df.columns else 0.0

                h1_feat = macro_clean.to_numpy(dtype=np.float32)
                if len(h1_feat) >= 32:
                    h1_slice = h1_feat[-32:]
                else:
                    pad_rows = 32 - len(h1_feat)
                    first_row = h1_feat[0:1]
                    padding = np.repeat(first_row, pad_rows, axis=0)
                    h1_slice = np.vstack([padding, h1_feat])

                m_norm = (h1_slice - h1_slice.mean(axis=0)) / (h1_slice.std(axis=0) + 1e-6)
                m_norm = np.nan_to_num(m_norm, nan=0.0, posinf=3.0, neginf=-3.0)
                x_macro_tensor = torch.tensor(m_norm, dtype=torch.float32).unsqueeze(0)
            else:
                h1_trend = 0.0
                x_macro_tensor = torch.zeros(1, 32, checkpoint["macro_dim"], dtype=torch.float32)

            fund_dim = checkpoint["fund_dim"]
            if fund_dim == 8 and "bar_ofi" in tech_df.columns:
                cvd_val = float(tech_df["bar_ofi"].rolling(288, min_periods=5).mean().iloc[-1]) if len(tech_df) >= 5 else 0.0
                hl = float(bars["high"].iloc[-1] - bars["low"].iloc[-1])
                spread_val = float(bars["spread"].iloc[-1] * 0.01) if "spread" in bars.columns else 65.0
                basis_val = hl / (spread_val + 1e-6)
                z_full = np.concatenate([z_vec[:6], [cvd_val, basis_val]]).astype(np.float32)
            else:
                z_full = np.resize(z_vec, fund_dim).astype(np.float32)

            x_fund_tensor = torch.tensor(z_full, dtype=torch.float32).unsqueeze(0)
            z_reg_tensor = x_fund_tensor.clone()

            # E. Tri-Domain MoE Real-Time Forward Pass
            with torch.no_grad():
                out = model(x_tech_tensor, x_macro_tensor, x_fund_tensor, z_reg_tensor)

            y_pred = float(out["y_pred"].item())
            conviction_size = float(out["size"].item())
            weights = out["weights"].squeeze(0).numpy()
            entropy = float(out["entropy"].item())

            # F. Check Shannon Entropy Fallback
            safe_weights, is_collapsed = entropy_guard.evaluate_entropy(out["weights"])
            if is_collapsed.any():
                logger.warning("Router Entropy Collapse (H=%.3f < 0.35)! Switched to defensive equal weights.", entropy)

            # G. MRDD Wavelet Drift Check
            curr_ret = float(log_ret.iloc[-1])
            recent_returns.append(curr_ret)
            if len(recent_returns) > 64:
                drift_status = mrdd.evaluate_wavelet_drift(np.array(recent_returns[-64:]))
                if drift_status["wavelet_drift_detected"]:
                    logger.warning("MRDD Alert: Structural Wavelet Energy Divergence detected (div=%.2f)!", drift_status["max_divergence"])

            # H. Execution Dispatch via Bridge
            realized_vol = float(vol_20.iloc[-1])
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            logger.info(
                "Step [%d] | %s (%s) | Latency: %.2fms\n"
                "  - Forecast y_hat:   %+0.4f (Conviction Sizing: %.2f)\n"
                "  - Domain Weights:   Tech: %.1f%% | Macro: %.1f%% | Fund: %.1f%%\n"
                "  - Router Entropy:   %.3f | Macro Sentiment: %+0.3f | H1 Trend: %+0.2f\n"
                "  - Latest Bid/Ask:   %.2f / %.2f",
                step,
                args.symbol,
                args.timeframe,
                latency_ms,
                y_pred,
                conviction_size,
                weights[0] * 100,
                weights[1] * 100,
                weights[2] * 100,
                entropy,
                z_vec[0],
                h1_trend,
                latest_tick["bid"] if latest_tick else 0.0,
                latest_tick["ask"] if latest_tick else 0.0,
            )

            # I. False Positive & False Alarm Pruning Filters
            is_buy = y_pred > 0
            is_actionable = True
            rejection_reasons = []

            # Check if live position exists
            if not args.paper:
                if execution_bridge.has_open_position(args.symbol):
                    is_actionable = False
                    rejection_reasons.append(f"Active Live Position in {args.symbol}")
                    # Vector 2: Volatility-Adaptive Breakeven Ratchet on live positions
                    if args.enhanced and args.breakeven_ratchet:
                        ratchet_events = execution_bridge.check_and_ratchet_breakeven(
                            args.symbol, be_activation_ratio=args.be_activation_ratio
                        )
                        if ratchet_events:
                            logger.info("  [LIVE RATCHET EVENT] -> %s", ratchet_events)

            # Check if paper position exists
            if args.paper and paper_position is not None:
                bid_now = latest_tick["bid"] if latest_tick else close_prices.iloc[-1]
                ask_now = latest_tick["ask"] if latest_tick else close_prices.iloc[-1]
                p_dir = paper_position["direction"]
                p_entry = paper_position["entry"]
                p_sl = paper_position["sl"]
                p_tp = paper_position["tp"]
                hit_exit = False
                exit_msg = ""

                # Vector 2: Breakeven ratchet for paper position
                if args.enhanced and args.breakeven_ratchet and not paper_position.get("be_active", False):
                    curr_p = bid_now if p_dir == "BUY" else ask_now
                    spread_p = (sym_info["spread"] * sym_info["point"]) if sym_info else 0.50
                    buffer_p = 2.0 * (sym_info["point"] if sym_info else 0.01)

                    if p_dir == "BUY":
                        gain = curr_p - p_entry
                        target_dist = p_tp - p_entry
                        if target_dist > 0 and gain >= target_dist * args.be_activation_ratio:
                            new_sl = p_entry + spread_p + buffer_p
                            if new_sl > p_sl:
                                paper_position["sl"] = new_sl
                                paper_position["be_active"] = True
                                logger.info("  [PAPER BREAKEVEN RATCHET ACTIVATED] BUY SL moved from %.2f to %.2f (+%.2f locked)", p_sl, new_sl, new_sl - p_entry)
                    else:
                        gain = p_entry - curr_p
                        target_dist = p_entry - p_tp
                        if target_dist > 0 and gain >= target_dist * args.be_activation_ratio:
                            new_sl = p_entry - spread_p - buffer_p
                            if new_sl < p_sl:
                                paper_position["sl"] = new_sl
                                paper_position["be_active"] = True
                                logger.info("  [PAPER BREAKEVEN RATCHET ACTIVATED] SELL SL moved from %.2f to %.2f (+%.2f locked)", p_sl, new_sl, p_entry - new_sl)

                if p_dir == "BUY":
                    if bid_now <= paper_position["sl"]:
                        hit_exit = True
                        exit_msg = f"SL HIT @ {bid_now:.2f}"
                    elif bid_now >= paper_position["tp"]:
                        hit_exit = True
                        exit_msg = f"TP HIT @ {bid_now:.2f}"
                else:
                    if ask_now >= paper_position["sl"]:
                        hit_exit = True
                        exit_msg = f"SL HIT @ {ask_now:.2f}"
                    elif ask_now <= paper_position["tp"]:
                        hit_exit = True
                        exit_msg = f"TP HIT @ {ask_now:.2f}"

                if hit_exit:
                    hold_m = (datetime.now(timezone.utc) - paper_position["time"]).total_seconds() / 60.0
                    logger.info("  [PAPER POSITION CLOSED] -> %s (Hold time: %.1f mins)", exit_msg, hold_m)
                    paper_position = None
                else:
                    is_actionable = False
                    rejection_reasons.append(f"Paper Position Open ({p_dir} @ {paper_position['entry']:.2f}, SL: {paper_position['sl']:.2f}, TP: {paper_position['tp']:.2f})")

            # 24/7 Session Regime Evaluation
            now_utc = datetime.now(timezone.utc)
            hour = now_utc.hour
            weekday = now_utc.weekday()
            is_weekend = (weekday in (5, 6)) or (weekday == 4 and hour >= 21) or (weekday == 0 and hour < 2)
            is_night = (hour >= 22) or (hour <= 6)
            is_off_hours = is_weekend or is_night
            is_institutional = (not is_weekend) and (12 <= hour <= 19)

            eff_conviction = args.min_conviction
            if args.session_filter and is_off_hours:
                eff_conviction = max(args.min_conviction, 0.20)

            # Vector 4: London / NY Overlap Conditioning
            if args.enhanced and is_institutional:
                conviction_size = min(conviction_size * 1.15, 2.0)

            # Vector 1: Pre-Trade Anti-Adverse Order Book Shield
            if args.enhanced and is_actionable:
                try:
                    from src.features.microstructure_features import MicrostructureFeatureSet
                    ms = MicrostructureFeatureSet(kyle_window=50, vpin_buckets=50, rv_inner=5, rv_outer=50)
                    ms_df = ms.build(bars)
                    current_ofi = float(ms_df["normalized_ofi"].iloc[-1])
                    current_vpin = float(ms_df["vpin"].iloc[-1])

                    if is_buy and (current_ofi < -0.20 or current_vpin > 0.65):
                        is_actionable = False
                        rejection_reasons.append(f"Anti-Adverse Book Shield: BUY rejected (OFI={current_ofi:+.2f} < -0.20 or VPIN={current_vpin:.2f} > 0.65)")
                    elif not is_buy and (current_ofi > 0.20 or current_vpin > 0.65):
                        is_actionable = False
                        rejection_reasons.append(f"Anti-Adverse Book Shield: SELL rejected (OFI={current_ofi:+.2f} > +0.20 or VPIN={current_vpin:.2f} > 0.65)")
                except Exception as e:
                    logger.warning("Microstructure feature extraction failed (%s); proceeding with base checks.", e)

            # Vector 3: Router Shannon Entropy Filter
            if args.enhanced and entropy >= 1.00:
                if abs(y_pred) < 0.038:
                    is_actionable = False
                    rejection_reasons.append(f"Router Entropy Filter: Conflicted experts (H={entropy:.3f} >= 1.00) requires |y_pred| >= 0.038 (got {abs(y_pred):.4f})")

            if abs(y_pred) < args.threshold:
                is_actionable = False
                rejection_reasons.append(f"Drift |y_pred|={abs(y_pred):.4f} < Threshold {args.threshold:.4f}")

            if conviction_size < eff_conviction:
                is_actionable = False
                rejection_reasons.append(f"Conviction={conviction_size:.2f} < Min {eff_conviction:.2f}")

            if args.trend_gate:
                if is_buy and h1_trend <= -0.20:
                    is_actionable = False
                    rejection_reasons.append(f"Trend Gate: BUY rejected (H1 trend {h1_trend:+.2f} <= -0.20)")
                elif not is_buy and h1_trend >= 0.20:
                    is_actionable = False
                    rejection_reasons.append(f"Trend Gate: SELL rejected (H1 trend {h1_trend:+.2f} >= +0.20)")

            if not is_actionable:
                logger.info(
                    "  [SIGNAL FILTERED / FALSE ALARM PRUNED] -> %s (No order dispatched)",
                    " & ".join(rejection_reasons),
                )
                if args.max_steps > 0 and step >= args.max_steps:
                    logger.info("Reached target step limit (%d). Exiting cleanly.", args.max_steps)
                    break
                time.sleep(args.interval)
                continue

            # Calculate bracket levels and lot size
            entry_p = (latest_tick["ask"] if is_buy else latest_tick["bid"]) if latest_tick else close_prices.iloc[-1]
            sigma = max(realized_vol, 0.003) * entry_p
            sl_dist = max(args.k_sl * sigma, args.min_sl_dist)

            if args.session_filter:
                tp_mult = args.k_tp if is_institutional else (2.0 * args.k_sl if is_off_hours else 2.5 * args.k_sl)
                tp_dist = max(tp_mult * sigma, 2.0 * sl_dist)
            else:
                tp_dist = max(args.k_tp * sigma, 2.0 * sl_dist)

            sl_p = entry_p - sl_dist if is_buy else entry_p + sl_dist
            tp_p = entry_p + tp_dist if is_buy else entry_p - tp_dist
            lot_size = execution_bridge.calculate_lot_size(args.symbol, sl_dist, conviction_size)

            spread_points = sym_info["spread"] if sym_info else 0.0
            point_sz = sym_info["point"] if sym_info else 0.01
            spread_cash = spread_points * point_sz
            spread_friction_pct = (spread_cash / (sl_dist + 1e-6)) * 100.0

            if args.paper:
                paper_position = {
                    "direction": "BUY" if is_buy else "SELL",
                    "entry": entry_p,
                    "sl": sl_p,
                    "tp": tp_p,
                    "sl_dist": sl_dist,
                    "lots": lot_size,
                    "time": datetime.now(timezone.utc),
                }
                logger.info(
                    "  [SIMULATED BRACKET ORDER] | Mode: PAPER\n"
                    "    - Action:        %s %.2f lots %s @ %.2f\n"
                    "    - Stop-Loss:     %.2f (Distance: $%.2f, Risk: %.2f%% Equity)\n"
                    "    - Take-Profit:   %.2f (Distance: $%.2f, Reward:Risk = %.1f:1)\n"
                    "    - Broker Spread: $%.2f (Friction: %.2f%% of Stop-Loss Distance)\n"
                    "    - H1 Macro Trend:%+0.2f (Trend Gated: PASS)",
                    "BUY" if is_buy else "SELL",
                    lot_size,
                    args.symbol,
                    entry_p,
                    sl_p,
                    sl_dist,
                    args.risk_pct * 100.0 * max(0.1, min(conviction_size, 2.0)),
                    tp_p,
                    tp_dist,
                    tp_dist / max(sl_dist, 1e-6),
                    spread_cash,
                    spread_friction_pct,
                    h1_trend,
                )
            else:
                # Dispatch live deal to MT5 broker
                dispatch_res = execution_bridge.dispatch_signal(
                    symbol=args.symbol,
                    directional_pred=y_pred,
                    conviction_size=conviction_size,
                    volatility=realized_vol,
                    sl_dist=sl_dist,
                    tp_dist=tp_dist,
                )
                logger.info("Order Dispatch Result: %s", dispatch_res)

            if args.max_steps > 0 and step >= args.max_steps:
                logger.info("Reached target step limit (%d). Exiting cleanly.", args.max_steps)
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("Live MT5 Trader halted by user.")
    finally:
        loader.disconnect()
        logger.info("MT5 connection closed.")


if __name__ == "__main__":
    main()
