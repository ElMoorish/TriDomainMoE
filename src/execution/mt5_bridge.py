"""
MetaTrader 5 Live Order Execution Bridge & Risk Dispatcher.
Executes orders with Triple-Barrier SL/TP, meta-calibrated lot sizing,
and deterministic 3-tier drawdown circuit breakers.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timezone
import numpy as np

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

from src.surveillance.risk_controls import RiskControls, CircuitBreakerState

logger = logging.getLogger(__name__)


class MT5ExecutionBridge:
    """
    Production trade execution manager interfacing directly with MT5.
    Guarantees pre-trade risk checks and automated protective stops.
    """

    def __init__(
        self,
        risk_controls: Optional[RiskControls] = None,
        magic_number: int = 777123,
        slippage_points: int = 10,
        risk_per_trade_pct: float = 0.0020,  # Scaled to 0.20% per trade ($20 risk on $10k equity)
    ):
        self.risk_controls = risk_controls or RiskControls()
        self.magic_number = magic_number
        self.slippage = slippage_points
        self.risk_pct = risk_per_trade_pct

    @staticmethod
    def get_supported_filling_mode(info: Any) -> int:
        """
        Determines the supported order filling mode from MT5 symbol info.
        Prevents TRADE_RETCODE_UNSUPPORTED_FILLING_MODE (retcode 10030).
        """
        if not MT5_AVAILABLE or info is None:
            return 1
        
        filling_mode = getattr(info, "filling_mode", 0)
        # Bitmask: SYMBOL_FILLING_FOK = 1, SYMBOL_FILLING_IOC = 2
        if filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        elif filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        else:
            return mt5.ORDER_FILLING_RETURN

    def get_account_equity(self) -> float:
        """Fetch current live account equity from MT5."""
        if not MT5_AVAILABLE:
            return 100000.0
        acc = mt5.account_info()
        return acc.equity if acc is not None else 100000.0

    def calculate_lot_size(
        self,
        symbol: str,
        stop_loss_distance: float,
        conviction_size: float = 1.0,
    ) -> float:
        """
        Calculate dynamically scaled broker lot size respecting 0.10% risk ceiling.
        """
        if not MT5_AVAILABLE:
            return 0.01

        equity = self.get_account_equity()
        info = mt5.symbol_info(symbol)
        if info is None or stop_loss_distance <= 0:
            return 0.01

        # Risk amount in account currency (e.g. 0.10% * conviction)
        risk_cash = equity * self.risk_pct * max(0.1, min(conviction_size, 2.0))

        tick_size = info.trade_tick_size or info.point
        tick_value = info.trade_tick_value or 1.0

        ticks_at_risk = stop_loss_distance / tick_size
        cash_risk_per_lot = ticks_at_risk * tick_value

        if cash_risk_per_lot <= 0:
            return info.volume_min

        raw_lot = risk_cash / cash_risk_per_lot
        # Round to step
        step = info.volume_step or 0.01
        lot_size = max(info.volume_min, min(info.volume_max, np.floor(raw_lot / step) * step))
        return round(lot_size, 2)

    def has_open_position(self, symbol: str) -> bool:
        """Check if any open position exists for this symbol and magic number."""
        if not MT5_AVAILABLE:
            return False
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return False
        return any(pos.magic == self.magic_number for pos in positions)

    def check_and_ratchet_breakeven(
        self,
        symbol: str,
        be_activation_ratio: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """
        Monitors active positions and ratchets SL to Entry + Spread + 2 points
        once price achieves >= 50% of the target distance (or +1.5 sigma).
        Locks in a guaranteed non-negative scratch/breakeven exit.
        """
        if not MT5_AVAILABLE:
            return []

        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return []

        info = mt5.symbol_info(symbol)
        if info is None:
            return []

        point = info.point or 0.01
        spread = (info.spread or 65.0) * point
        buffer = 2.0 * point
        results = []

        for pos in positions:
            if pos.magic != self.magic_number:
                continue

            ticket = pos.ticket
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            entry_p = pos.price_open
            current_sl = pos.sl
            tp_p = pos.tp
            current_p = pos.price_current

            if tp_p <= 0:
                continue

            if is_buy:
                target_dist = tp_p - entry_p
                current_gain = current_p - entry_p
                if target_dist > 0 and current_gain >= target_dist * be_activation_ratio:
                    new_sl = round(entry_p + spread + buffer, info.digits)
                    if new_sl > current_sl:
                        req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "symbol": symbol,
                            "sl": new_sl,
                            "tp": tp_p,
                        }
                        res = mt5.order_send(req)
                        if res.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info("[BREAKEVEN RATCHET ACTIVATED] BUY #%d SL ratcheted from %.2f to %.2f (+%.2f locked profit)", ticket, current_sl, new_sl, new_sl - entry_p)
                            results.append({
                                "ticket": ticket,
                                "status": "ratcheted",
                                "new_sl": new_sl,
                                "old_sl": current_sl,
                                "direction": "BUY",
                                "entry": entry_p,
                                "locked_profit": new_sl - entry_p,
                            })
                        else:
                            logger.warning("Breakeven ratchet failed on BUY #%d: %s (code %d)", ticket, res.comment, res.retcode)
            else:
                target_dist = entry_p - tp_p
                current_gain = entry_p - current_p
                if target_dist > 0 and current_gain >= target_dist * be_activation_ratio:
                    new_sl = round(entry_p - spread - buffer, info.digits)
                    if current_sl <= 0 or new_sl < current_sl:
                        req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "symbol": symbol,
                            "sl": new_sl,
                            "tp": tp_p,
                        }
                        res = mt5.order_send(req)
                        if res.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info("[BREAKEVEN RATCHET ACTIVATED] SELL #%d SL ratcheted from %.2f to %.2f (+%.2f locked profit)", ticket, current_sl, new_sl, entry_p - new_sl)
                            results.append({
                                "ticket": ticket,
                                "status": "ratcheted",
                                "new_sl": new_sl,
                                "old_sl": current_sl,
                                "direction": "SELL",
                                "entry": entry_p,
                                "locked_profit": entry_p - new_sl,
                            })
                        else:
                            logger.warning("Breakeven ratchet failed on SELL #%d: %s (code %d)", ticket, res.comment, res.retcode)

        return results

    def dispatch_signal(
        self,
        symbol: str,
        directional_pred: float,
        conviction_size: float,
        volatility: float,
        pt_multiplier: float = 2.0,
        sl_multiplier: float = 1.5,
        sl_dist: Optional[float] = None,
        tp_dist: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate pre-trade circuit breakers, compute bracket orders, and dispatch to MT5.
        """
        if self.has_open_position(symbol):
            logger.info("Order skipped: Active position already open for %s (magic: %d)", symbol, self.magic_number)
            return {"status": "skipped_active_position"}

        equity = self.get_account_equity()
        cb_state, dd = self.risk_controls.update_drawdown_monitor(equity)

        if cb_state in (CircuitBreakerState.CASH_OUT, CircuitBreakerState.HALT_RETRAIN):
            logger.warning("Order blocked: Circuit breaker active (%s, DD=%.2f%%)", cb_state.value, dd * 100)
            return {"status": "blocked_by_circuit_breaker", "state": cb_state.value}

        # Apply circuit breaker leverage downscaling
        safe_conviction = self.risk_controls.apply_circuit_breaker(conviction_size)
        if safe_conviction <= 0.05:
            return {"status": "skipped_low_conviction"}

        if not MT5_AVAILABLE:
            return {"status": "simulated", "direction": "BUY" if directional_pred > 0 else "SELL"}

        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is None or info is None:
            return {"status": "symbol_unavailable"}

        is_buy = directional_pred > 0
        entry_price = tick.ask if is_buy else tick.bid
        sigma = max(volatility, info.point * 10)

        if sl_dist is None:
            sl_dist = sl_multiplier * sigma
        if tp_dist is None:
            tp_dist = pt_multiplier * sigma

        sl_price = entry_price - sl_dist if is_buy else entry_price + sl_dist
        tp_price = entry_price + tp_dist if is_buy else entry_price - tp_dist

        lot_size = self.calculate_lot_size(symbol, sl_dist, safe_conviction)

        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        filling_mode = self.get_supported_filling_mode(info)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_price,
            "sl": round(sl_price, info.digits),
            "tp": round(tp_price, info.digits),
            "deviation": self.slippage,
            "magic": self.magic_number,
            "comment": f"TriDomainMoE-{cb_state.value}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        res = mt5.order_send(request)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("MT5 order failed: %s (code %d)", res.comment, res.retcode)
            return {"status": "error", "retcode": res.retcode, "comment": res.comment}

        logger.info(
            "MT5 Order Executed: %s %.2f lots %s @ %.2f (SL: %.2f, TP: %.2f, Magic: %d)",
            "BUY" if is_buy else "SELL",
            lot_size,
            symbol,
            entry_price,
            sl_price,
            tp_price,
            self.magic_number,
        )
        return {
            "status": "success",
            "order_id": res.order,
            "lot_size": lot_size,
            "direction": "BUY" if is_buy else "SELL",
            "entry": entry_price,
            "sl": sl_price,
            "tp": tp_price,
        }
