"""
High-Resolution Intra-Bar Execution Engine.
Simulates realistic broker executions across hundreds of thousands of M1 bars:
- Enforces real broker floating spreads recorded at every minute
- Evaluates intra-bar High/Low penetration of bracket Stop-Loss and Take-Profit
- Conservative collision resolution (adverse Stop-Loss executed if both barriers touched)
- Dynamic 0.10% risk-per-trade position sizing
- 3-tier drawdown circuit breakers (1.25%, 2.00%, 2.50%)
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

from .metrics import StatisticalEvidenceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BarTrade:
    trade_id: int
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    lots: float
    sl_price: float
    tp_price: float
    entry_spread: float
    exit_spread: float
    pnl_cash: float
    pnl_pct: float
    exit_reason: str  # "TAKE_PROFIT", "STOP_LOSS", "TIME_BARRIER", "CIRCUIT_BREAKER", "END_OF_DATA"
    duration_minutes: float
    equity_after: float


class IntraBarExecutionEngine:
    """High-throughput intra-bar execution engine strictly following broker microstructure rules."""

    def __init__(
        self,
        symbol: str = "BTCUSD.x",
        initial_balance: float = 10000.0,
        risk_per_trade_pct: float = 0.0010,  # 0.10% risk ceiling
        contract_size: float = 1.0,
        min_lot: float = 0.01,
        max_lot: float = 10.0,
        lot_step: float = 0.01,
        max_holding_bars: int = 1440,  # Max holding horizon in bars
        cooldown_bars: Optional[int] = None,  # Cooldown after Tier 2 breach
        trailing_stop: bool = True,
        breakeven_trigger: float = 1.0,  # Move to breakeven once price reaches 1.0R in profit
        enable_breakeven_ratchet: bool = False,
        be_activation_ratio: float = 0.50,
        slippage_points: float = 2.0,
        point_size: float = 0.01,
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.contract_size = contract_size
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.lot_step = lot_step
        self.max_holding_bars = max_holding_bars
        self.cooldown_bars = cooldown_bars if cooldown_bars is not None else max_holding_bars
        self.trailing_stop = trailing_stop
        self.breakeven_trigger = breakeven_trigger
        self.enable_breakeven_ratchet = enable_breakeven_ratchet
        self.be_activation_ratio = be_activation_ratio
        self.slippage_cash = slippage_points * point_size
        self.point_size = point_size

        # Circuit breaker thresholds (Prop firm profile)
        self.tier1_dd_pct = 1.25
        self.tier2_dd_pct = 2.00
        self.tier3_dd_pct = 2.50

    def calculate_lots(self, equity: float, sl_distance: float, conviction: float = 1.0) -> float:
        """
        Dynamically scales lot size adhering to exact 0.10% risk ceiling.
        Lots = (Equity * RiskPct * Conviction) / (SL_Distance * ContractSize)
        """
        if sl_distance <= 0:
            return self.min_lot

        risk_cash = equity * self.risk_per_trade_pct * min(max(conviction, 0.2), 1.0)
        raw_lots = risk_cash / (sl_distance * self.contract_size)

        steps = np.floor(raw_lots / self.lot_step)
        lots = steps * self.lot_step
        return float(np.clip(lots, self.min_lot, self.max_lot))

    def run_bar_simulation(
        self,
        bars_df: pd.DataFrame,
        signals_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Simulates execution across M1 bars with recorded floating spreads.
        
        Args:
            bars_df: DataFrame with DatetimeIndex and columns ['open', 'high', 'low', 'close', 'tick_volume', 'spread']
            signals_df: DataFrame aligned to bars_df.index with columns ['direction', 'conviction', 'sl_dist', 'tp_dist']
        """
        balance = self.initial_balance
        equity = balance
        peak_equity = balance

        open_trade: Optional[Dict[str, Any]] = None
        closed_trades: List[BarTrade] = []
        equity_records = []
        timestamps = []

        n_bars = len(bars_df)
        opens = bars_df["open"].to_numpy(dtype=np.float64)
        highs = bars_df["high"].to_numpy(dtype=np.float64)
        lows = bars_df["low"].to_numpy(dtype=np.float64)
        closes = bars_df["close"].to_numpy(dtype=np.float64)
        spread_points = bars_df["spread"].to_numpy(dtype=np.float64)
        spreads_cash = spread_points * self.point_size
        times = bars_df.index

        # Signal arrays
        directions = signals_df["direction"].to_numpy() if "direction" in signals_df.columns else np.array(["FLAT"] * n_bars)
        convictions = signals_df["conviction"].to_numpy(dtype=np.float64) if "conviction" in signals_df.columns else np.ones(n_bars)
        sl_dists = signals_df["sl_dist"].to_numpy(dtype=np.float64) if "sl_dist" in signals_df.columns else np.full(n_bars, 500.0)
        tp_dists = signals_df["tp_dist"].to_numpy(dtype=np.float64) if "tp_dist" in signals_df.columns else np.full(n_bars, 1000.0)

        trade_counter = 0
        circuit_broken = False
        leverage_penalty = 1.0
        cooldown_until_idx = 0
        cooldown_triggered = False

        for i in range(n_bars):
            t = times[i]
            op = opens[i]
            hi = highs[i]
            lo = lows[i]
            cl = closes[i]
            sp_cash = spreads_cash[i]

            # 1. Update Open Position if exists
            if open_trade is not None:
                is_buy = open_trade["direction"] == "BUY"
                lots = open_trade["lots"]
                entry_p = open_trade["entry_price"]
                sl_p = open_trade["sl_price"]
                tp_p = open_trade["tp_price"]
                entry_idx = open_trade["entry_idx"]

                # Mark to market unrealized PnL at bar close
                # Intra-bar barrier checks for Long/Short
                if is_buy:
                    unrealized_pnl = (cl - entry_p) * lots * self.contract_size
                    hit_sl = lo <= sl_p
                    hit_tp = hi >= tp_p
                else:
                    unrealized_pnl = (entry_p - (cl + sp_cash)) * lots * self.contract_size
                    hit_sl = (hi + sp_cash) >= sl_p
                    hit_tp = (lo + sp_cash) <= tp_p

                # Breakeven ratchet update if trade is active and reaches threshold profit
                if self.enable_breakeven_ratchet and not open_trade.get("be_active", False) and not hit_sl and not hit_tp:
                    if is_buy:
                        gain = hi - entry_p
                        threshold_dist = (tp_p - entry_p) * self.be_activation_ratio
                        if gain >= threshold_dist:
                            new_sl = entry_p + open_trade.get("entry_spread", sp_cash) + self.slippage_cash
                            if new_sl > sl_p:
                                sl_p = new_sl
                                open_trade["sl_price"] = sl_p
                                open_trade["be_active"] = True
                    else:
                        gain = entry_p - lo
                        threshold_dist = (entry_p - tp_p) * self.be_activation_ratio
                        if gain >= threshold_dist:
                            new_sl = entry_p - open_trade.get("entry_spread", sp_cash) - self.slippage_cash
                            if new_sl < sl_p:
                                sl_p = new_sl
                                open_trade["sl_price"] = sl_p
                                open_trade["be_active"] = True

                # Trailing stop & breakeven update if trade is active
                if self.trailing_stop and not hit_sl and not hit_tp:
                    sl_dist = open_trade["sl_dist"]
                    if is_buy:
                        if (hi - entry_p) >= self.breakeven_trigger * sl_dist:
                            be_sl = entry_p + open_trade["entry_spread"] + self.slippage_cash
                            trail_sl = hi - 1.2 * sl_dist
                            sl_p = max(sl_p, be_sl, trail_sl)
                            open_trade["sl_price"] = sl_p
                    else:
                        if (entry_p - lo) >= self.breakeven_trigger * sl_dist:
                            be_sl = entry_p - open_trade["entry_spread"] - self.slippage_cash
                            trail_sl = lo + 1.2 * sl_dist
                            sl_p = min(sl_p, be_sl, trail_sl)
                            open_trade["sl_price"] = sl_p

                current_equity = balance + unrealized_pnl
                if current_equity > peak_equity:
                    peak_equity = current_equity

                hold_bars = i - entry_idx
                hit_time = hold_bars >= self.max_holding_bars

                # Trailing drawdown circuit breaker check
                current_dd_pct = ((peak_equity - current_equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0

                should_exit = False
                exit_reason = ""
                exit_price = 0.0

                # Conservative collision resolution: adverse stop loss triggered if both hit
                if hit_sl and hit_tp:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                    exit_price = (sl_p - self.slippage_cash) if is_buy else (sl_p + self.slippage_cash)
                elif hit_sl:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                    exit_price = (sl_p - self.slippage_cash) if is_buy else (sl_p + self.slippage_cash)
                elif hit_tp:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
                    exit_price = tp_p
                elif current_dd_pct >= self.tier3_dd_pct:
                    should_exit = True
                    exit_reason = "CIRCUIT_BREAKER"
                    exit_price = cl if is_buy else (cl + sp_cash)
                    circuit_broken = True
                elif hit_time:
                    should_exit = True
                    exit_reason = "TIME_BARRIER"
                    exit_price = cl if is_buy else (cl + sp_cash)

                if should_exit:
                    if is_buy:
                        pnl_cash = (exit_price - entry_p) * lots * self.contract_size
                    else:
                        pnl_cash = (entry_p - exit_price) * lots * self.contract_size

                    balance += pnl_cash
                    equity = balance
                    if equity > peak_equity:
                        peak_equity = equity

                    pnl_pct = (pnl_cash / balance) * 100.0
                    hold_mins = (t - open_trade["entry_time"]).total_seconds() / 60.0

                    closed_trades.append(
                        BarTrade(
                            trade_id=open_trade["trade_id"],
                            entry_time=open_trade["entry_time"],
                            exit_time=t,
                            symbol=self.symbol,
                            direction=open_trade["direction"],
                            entry_price=entry_p,
                            exit_price=exit_price,
                            lots=lots,
                            sl_price=sl_p,
                            tp_price=tp_p,
                            entry_spread=open_trade["entry_spread"],
                            exit_spread=sp_cash,
                            pnl_cash=float(pnl_cash),
                            pnl_pct=float(pnl_pct),
                            exit_reason=exit_reason,
                            duration_minutes=float(hold_mins),
                            equity_after=float(balance),
                        )
                    )
                    open_trade = None

            # 2. Check Circuit Breaker Status
            current_dd = ((peak_equity - equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if current_dd >= self.tier3_dd_pct:
                circuit_broken = True
                logger.critical("TIER 3 BREACH: Drawdown %.2f%% exceeded 2.50%% ceiling. Trading halted.", current_dd)
                break
            elif current_dd >= self.tier2_dd_pct:
                if not cooldown_triggered:
                    cooldown_until_idx = i + self.cooldown_bars
                    cooldown_triggered = True
                leverage_penalty = 0.25  # Defensive recovery size
            elif current_dd < self.tier1_dd_pct:
                cooldown_triggered = False
                leverage_penalty = 1.0
            else:
                leverage_penalty = 0.50

            # 3. Evaluate New Signals (only if no open position, not halted, and after cooldown)
            if open_trade is None and not circuit_broken and (i >= 50) and (i >= cooldown_until_idx):
                sig_dir = directions[i]
                if sig_dir in ("BUY", "SELL"):
                    conviction = convictions[i]
                    sl_dist = sl_dists[i]
                    tp_dist = tp_dists[i]

                    trade_lots = self.calculate_lots(equity, sl_dist, conviction * leverage_penalty)

                    if trade_lots >= self.min_lot:
                        trade_counter += 1
                        if sig_dir == "BUY":
                            # Enter Long at Ask (Open + spread)
                            entry_price = op + sp_cash + self.slippage_cash
                            sl_price = entry_price - sl_dist
                            tp_price = entry_price + tp_dist
                        else:
                            # Enter Short at Bid (Open)
                            entry_price = op - self.slippage_cash
                            sl_price = entry_price + sl_dist
                            tp_price = entry_price - tp_dist

                        open_trade = {
                            "trade_id": trade_counter,
                            "entry_time": t,
                            "entry_idx": i,
                            "direction": sig_dir,
                            "entry_price": entry_price,
                            "lots": trade_lots,
                            "sl_price": sl_price,
                            "tp_price": tp_price,
                            "sl_dist": sl_dist,
                            "entry_spread": sp_cash,
                            "be_active": False,
                        }

            # Periodic equity recording (every 5 bars or end)
            if i % 5 == 0 or i == n_bars - 1:
                current_unrealized = 0.0
                if open_trade is not None:
                    if open_trade["direction"] == "BUY":
                        current_unrealized = (cl - open_trade["entry_price"]) * open_trade["lots"] * self.contract_size
                    else:
                        current_unrealized = (open_trade["entry_price"] - (cl + sp_cash)) * open_trade["lots"] * self.contract_size

                equity_records.append(balance + current_unrealized)
                timestamps.append(t)

        # Finalize any remaining open position at end of simulation
        if open_trade is not None:
            last_cl = closes[-1]
            last_sp = spreads_cash[-1]
            last_t = times[-1]
            is_buy = open_trade["direction"] == "BUY"
            exit_price = last_cl if is_buy else (last_cl + last_sp)
            lots = open_trade["lots"]
            entry_p = open_trade["entry_price"]
            hold_mins = (last_t - open_trade["entry_time"]).total_seconds() / 60.0

            if is_buy:
                pnl_cash = (exit_price - entry_p) * lots * self.contract_size
            else:
                pnl_cash = (entry_p - exit_price) * lots * self.contract_size

            balance += pnl_cash
            pnl_pct = (pnl_cash / balance) * 100.0

            closed_trades.append(
                BarTrade(
                    trade_id=open_trade["trade_id"],
                    entry_time=open_trade["entry_time"],
                    exit_time=last_t,
                    symbol=self.symbol,
                    direction=open_trade["direction"],
                    entry_price=entry_p,
                    exit_price=exit_price,
                    lots=lots,
                    sl_price=open_trade["sl_price"],
                    tp_price=open_trade["tp_price"],
                    entry_spread=open_trade["entry_spread"],
                    exit_spread=last_sp,
                    pnl_cash=float(pnl_cash),
                    pnl_pct=float(pnl_pct),
                    exit_reason="END_OF_DATA",
                    duration_minutes=float(hold_mins),
                    equity_after=float(balance),
                )
            )
            open_trade = None

        trades_df = pd.DataFrame([asdict(t) for t in closed_trades]) if closed_trades else pd.DataFrame()
        equity_series = pd.Series(equity_records, index=pd.DatetimeIndex(timestamps), name="equity")

        # Master evaluation
        stats = StatisticalEvidenceMetrics.evaluate_all(trades_df, equity_series, self.initial_balance)
        return trades_df, equity_series, stats
