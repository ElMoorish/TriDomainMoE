"""
Event-Driven Real Tick Execution Engine.
Simulates realistic broker fills with microsecond accuracy using MT5 tick data:
- Fills Buys at Ask, Exits Buys at Bid
- Fills Sells at Bid, Exits Sells at Ask
- Dynamic 0.10% risk-per-trade position sizing
- 3-tier drawdown circuit breakers
- Floating spread & slippage deduction
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
class SimulatedTrade:
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
    exit_reason: str  # "TAKE_PROFIT", "STOP_LOSS", "TIME_BARRIER", "CIRCUIT_BREAKER"
    duration_seconds: float
    equity_after: float


class EventDrivenTickEngine:
    """High-frequency tick execution simulator strictly following broker microstructure rules."""

    def __init__(
        self,
        symbol: str = "BTCUSD.x",
        initial_balance: float = 10000.0,
        risk_per_trade_pct: float = 0.0010,  # 0.10% risk ceiling
        contract_size: float = 1.0,
        min_lot: float = 0.01,
        max_lot: float = 10.0,
        lot_step: float = 0.01,
        max_holding_seconds: float = 7200.0,  # 2-hour max holding horizon
        slippage_points: float = 2.0,
        point_size: float = 0.01,
        enable_breakeven_ratchet: bool = False,
        be_activation_ratio: float = 0.50,
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.contract_size = contract_size
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.lot_step = lot_step
        self.max_holding_seconds = max_holding_seconds
        self.slippage_cash = slippage_points * point_size
        self.point_size = point_size
        self.enable_breakeven_ratchet = enable_breakeven_ratchet
        self.be_activation_ratio = be_activation_ratio

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

        # Round to lot step
        steps = np.floor(raw_lots / self.lot_step)
        lots = steps * self.lot_step
        return float(np.clip(lots, self.min_lot, self.max_lot))

    def run_tick_simulation(
        self,
        ticks_df: pd.DataFrame,
        signals_generator: Callable[[pd.DataFrame, int], Optional[Dict[str, Any]]],
        bar_interval_ticks: int = 100,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Simulates execution across every real tick in `ticks_df`.
        
        Args:
            ticks_df: DataFrame with DatetimeIndex and columns ['bid', 'ask', 'volume']
            signals_generator: Function (ticks_df_history, current_tick_idx) -> Optional[Dict]:
                {'direction': 'BUY'/'SELL', 'conviction': 0.8, 'sl_dist': 350.0, 'tp_dist': 700.0}
            bar_interval_ticks: Frequency of running the signal model (e.g. every 100 ticks)
        """
        balance = self.initial_balance
        equity = balance
        peak_equity = balance

        open_trade: Optional[Dict[str, Any]] = None
        closed_trades: List[SimulatedTrade] = []
        equity_records = []
        timestamps = []

        n_ticks = len(ticks_df)
        bids = ticks_df["bid"].to_numpy(dtype=np.float64)
        asks = ticks_df["ask"].to_numpy(dtype=np.float64)
        times = ticks_df.index

        trade_counter = 0
        circuit_broken = False
        leverage_penalty = 1.0

        for i in range(n_ticks):
            t = times[i]
            bid = bids[i]
            ask = asks[i]
            spread = ask - bid

            # 1. Update Open Position if exists
            if open_trade is not None:
                is_buy = open_trade["direction"] == "BUY"
                lots = open_trade["lots"]
                entry_p = open_trade["entry_price"]
                sl_p = open_trade["sl_price"]
                tp_p = open_trade["tp_price"]
                entry_time = open_trade["entry_time"]

                # Breakeven ratchet: lock in profit when trade reaches activation threshold
                if self.enable_breakeven_ratchet and not open_trade.get("be_active", False):
                    if is_buy:
                        gain = bid - entry_p
                        threshold_dist = (tp_p - entry_p) * self.be_activation_ratio
                        if gain >= threshold_dist:
                            new_sl = entry_p + open_trade.get("entry_spread", spread) + self.slippage_cash
                            if new_sl > sl_p:
                                open_trade["sl_price"] = new_sl
                                open_trade["be_active"] = True
                                sl_p = new_sl
                    else:
                        gain = entry_p - ask
                        threshold_dist = (entry_p - tp_p) * self.be_activation_ratio
                        if gain >= threshold_dist:
                            new_sl = entry_p - open_trade.get("entry_spread", spread) - self.slippage_cash
                            if new_sl < sl_p:
                                open_trade["sl_price"] = new_sl
                                open_trade["be_active"] = True
                                sl_p = new_sl

                # Mark to market unrealized PnL
                if is_buy:
                    unrealized_pnl = (bid - entry_p) * lots * self.contract_size
                    hit_sl = bid <= sl_p
                    hit_tp = bid >= tp_p
                else:
                    unrealized_pnl = (entry_p - ask) * lots * self.contract_size
                    hit_sl = ask >= sl_p
                    hit_tp = ask <= tp_p

                current_equity = balance + unrealized_pnl
                if current_equity > peak_equity:
                    peak_equity = current_equity

                # Check duration
                hold_duration = (t - entry_time).total_seconds()
                hit_time = hold_duration >= self.max_holding_seconds

                # Check trailing drawdown circuit breaker
                current_dd_pct = ((peak_equity - current_equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0

                should_exit = False
                exit_reason = ""
                exit_price = 0.0

                if hit_sl:
                    should_exit = True
                    exit_reason = "BREAKEVEN" if open_trade.get("be_active", False) else "STOP_LOSS"
                    exit_price = (sl_p - self.slippage_cash) if is_buy else (sl_p + self.slippage_cash)
                elif hit_tp:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
                    exit_price = tp_p if is_buy else tp_p
                elif current_dd_pct >= self.tier2_dd_pct:
                    should_exit = True
                    exit_reason = "CIRCUIT_BREAKER"
                    exit_price = bid if is_buy else ask
                    circuit_broken = True
                elif hit_time:
                    should_exit = True
                    exit_reason = "TIME_BARRIER"
                    exit_price = bid if is_buy else ask

                if should_exit:
                    # Finalize trade
                    if is_buy:
                        pnl_cash = (exit_price - entry_p) * lots * self.contract_size
                    else:
                        pnl_cash = (entry_p - exit_price) * lots * self.contract_size

                    balance += pnl_cash
                    equity = balance
                    if equity > peak_equity:
                        peak_equity = equity

                    pnl_pct = (pnl_cash / balance) * 100.0

                    closed_trades.append(
                        SimulatedTrade(
                            trade_id=open_trade["trade_id"],
                            entry_time=entry_time,
                            exit_time=t,
                            symbol=self.symbol,
                            direction=open_trade["direction"],
                            entry_price=entry_p,
                            exit_price=exit_price,
                            lots=lots,
                            sl_price=sl_p,
                            tp_price=tp_p,
                            entry_spread=open_trade["entry_spread"],
                            exit_spread=spread,
                            pnl_cash=float(pnl_cash),
                            pnl_pct=float(pnl_pct),
                            exit_reason=exit_reason,
                            duration_seconds=float(hold_duration),
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
            elif current_dd >= self.tier1_dd_pct:
                leverage_penalty = 0.50  # Cut sizing by 50%
            else:
                leverage_penalty = 1.0

            # 3. Evaluate New Signals (only if no open position and not in circuit break)
            if open_trade is None and not circuit_broken and (i % bar_interval_ticks == 0) and (i >= 200):
                sig = signals_generator(ticks_df, i)
                if sig is not None and sig.get("direction") in ("BUY", "SELL"):
                    direction = sig["direction"]
                    conviction = sig.get("conviction", 1.0)
                    sl_dist = sig.get("sl_dist", 300.0)
                    tp_dist = sig.get("tp_dist", 600.0)

                    # Calculate conservative lot size
                    trade_lots = self.calculate_lots(equity, sl_dist, conviction * leverage_penalty)

                    if trade_lots >= self.min_lot:
                        trade_counter += 1
                        if direction == "BUY":
                            entry_price = ask + self.slippage_cash
                            sl_price = entry_price - sl_dist
                            tp_price = entry_price + tp_dist
                        else:
                            entry_price = bid - self.slippage_cash
                            sl_price = entry_price + sl_dist
                            tp_price = entry_price - tp_dist

                        open_trade = {
                            "trade_id": trade_counter,
                            "entry_time": t,
                            "direction": direction,
                            "entry_price": entry_price,
                            "lots": trade_lots,
                            "sl_price": sl_price,
                            "tp_price": tp_price,
                            "entry_spread": spread,
                        }

            # Periodic equity recording (every 50 ticks to save memory)
            if i % 50 == 0 or i == n_ticks - 1:
                current_unrealized = 0.0
                if open_trade is not None:
                    if open_trade["direction"] == "BUY":
                        current_unrealized = (bid - open_trade["entry_price"]) * open_trade["lots"] * self.contract_size
                    else:
                        current_unrealized = (open_trade["entry_price"] - ask) * open_trade["lots"] * self.contract_size

                equity_records.append(balance + current_unrealized)
                timestamps.append(t)

        # Finalize any remaining open position at end of simulation
        if open_trade is not None:
            last_bid = bids[-1]
            last_ask = asks[-1]
            last_t = times[-1]
            is_buy = open_trade["direction"] == "BUY"
            exit_price = last_bid if is_buy else last_ask
            lots = open_trade["lots"]
            entry_p = open_trade["entry_price"]
            hold_duration = (last_t - open_trade["entry_time"]).total_seconds()
            
            if is_buy:
                pnl_cash = (exit_price - entry_p) * lots * self.contract_size
            else:
                pnl_cash = (entry_p - exit_price) * lots * self.contract_size

            balance += pnl_cash
            pnl_pct = (pnl_cash / balance) * 100.0

            closed_trades.append(
                SimulatedTrade(
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
                    exit_spread=last_ask - last_bid,
                    pnl_cash=float(pnl_cash),
                    pnl_pct=float(pnl_pct),
                    exit_reason="END_OF_DATA",
                    duration_seconds=float(hold_duration),
                    equity_after=float(balance),
                )
            )
            open_trade = None

        # Convert to pandas DataFrames
        trades_df = pd.DataFrame([asdict(t) for t in closed_trades]) if closed_trades else pd.DataFrame()
        equity_series = pd.Series(equity_records, index=pd.DatetimeIndex(timestamps), name="equity")

        # Master evaluation
        stats = StatisticalEvidenceMetrics.evaluate_all(trades_df, equity_series, self.initial_balance)
        return trades_df, equity_series, stats
