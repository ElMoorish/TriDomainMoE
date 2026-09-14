"""
src/execution/stealth_sentinel.py
=================================
External Stealth Signal Sentinel for MetaTrader 5 (BTCUSD & NAS100).
Monitors position lifecycle and broadcasts VIP signals for:
  - New Order Entries
  - Breakeven (BE) & SL Modifications
  - SL / TP Proximity Touches
  - Stop-Loss Hits (distinguishing between full loss vs. breakeven scratch)
  - Take-Profit Targets Achieved

Prop Firm Stealth Guarantee:
  - Zero MQL5 WebRequests (terminal logs remain 100% clean)
  - Zero EA/Signal comments on MT5 broker orders
  - External Python IPC read-only memory surveillance
  - Dispatches exclusively via external Python network daemon to VIP & Admin channels
"""

from __future__ import annotations

import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

from src.execution.notifier import TradeNotifier
from src.utils.logger import setup_logger

logger = setup_logger("StealthSentinel")


class StealthSignalSentinel:
    """
    Real-time external position and order surveillance engine.
    Runs completely outside MT5, leaving zero audit footprint for prop firms.
    """

    DEFAULT_SYMBOLS = ["BTCUSD.x", "NAS100.x"]

    # Proximity alert threshold in price units (points/USD)
    PROXIMITY_THRESHOLDS = {
        "BTCUSD": 75.0,     # ~$75 on Bitcoin
        "NAS100": 15.0,     # ~15 index points on Nasdaq
        "DEFAULT": 10.0,
    }

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        notifier: Optional[TradeNotifier] = None,
        enable_proximity_warnings: bool = True,
        auto_notify_initial_positions: bool = False,
    ):
        self.notifier = notifier or TradeNotifier()
        self.enable_proximity_warnings = enable_proximity_warnings
        self.auto_notify_initial_positions = auto_notify_initial_positions

        # Internal state tracking: ticket -> position_dict
        self._positions: Dict[int, Dict[str, Any]] = {}
        self._initialized = False

        # Target symbols to surveil
        self.requested_symbols = symbols or list(self.DEFAULT_SYMBOLS)
        self.resolved_symbols: List[str] = []

    def initialize_mt5(self) -> bool:
        """Connects to active local MT5 terminal via read-only memory IPC."""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 python package not available.")
            return False

        if not mt5.initialize():
            logger.error("Failed to initialize MT5 IPC: %s", mt5.last_error())
            return False

        self._resolve_broker_symbols()
        logger.info("[STEALTH SENTINEL ONLINE] Surveilling active symbols: %s", self.resolved_symbols)
        return True

    def _resolve_broker_symbols(self):
        """Discovers exact broker symbol variants (e.g. BTCUSD.x vs BTCUSD vs BTCUSDm)."""
        all_syms = mt5.symbols_get()
        if not all_syms:
            self.resolved_symbols = list(self.requested_symbols)
            return

        available_names = {s.name for s in all_syms}
        resolved = []

        for req in self.requested_symbols:
            clean_base = req.replace(".x", "").replace("m", "").upper()
            matched = False

            # Direct match
            if req in available_names:
                resolved.append(req)
                mt5.symbol_select(req, True)
                matched = True
            else:
                # Search variants
                for cand in available_names:
                    cand_upper = cand.upper()
                    if clean_base in cand_upper:
                        resolved.append(cand)
                        mt5.symbol_select(cand, True)
                        matched = True
                        break

            if not matched:
                resolved.append(req)

        self.resolved_symbols = list(dict.fromkeys(resolved))

    def _get_proximity_threshold(self, symbol: str) -> float:
        clean = symbol.replace(".x", "").replace("m", "").upper()
        for k, v in self.PROXIMITY_THRESHOLDS.items():
            if k in clean:
                return v
        return self.PROXIMITY_THRESHOLDS["DEFAULT"]

    def scan_once(self):
        """
        Executes a single surveillance cycle:
          1. Detects new positions -> VIP Entry Signal
          2. Detects SL modifications -> VIP Breakeven / SL Moved Signal
          3. Checks tick price proximity to SL/TP -> Proximity Alerts
          4. Detects position closure & inspects history deals -> SL Hit / TP Hit Signal
        """
        if not MT5_AVAILABLE:
            return

        active_tickets: Set[int] = set()

        for sym in self.resolved_symbols:
            positions = mt5.positions_get(symbol=sym)
            if positions is None:
                continue

            tick = mt5.symbol_info_tick(sym)
            info = mt5.symbol_info(sym)
            prox_thresh = self._get_proximity_threshold(sym)

            for pos in positions:
                ticket = pos.ticket
                active_tickets.add(ticket)

                is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
                direction = "BUY" if is_buy else "SELL"
                current_p = tick.bid if (tick and is_buy) else (tick.ask if tick else pos.price_current)

                # --- 1. NEW POSITION DETECTION ---
                if ticket not in self._positions:
                    pos_record = {
                        "ticket": ticket,
                        "symbol": sym,
                        "direction": direction,
                        "is_buy": is_buy,
                        "entry_price": pos.price_open,
                        "initial_sl": pos.sl,
                        "current_sl": pos.sl,
                        "initial_tp": pos.tp,
                        "current_tp": pos.tp,
                        "volume": pos.volume,
                        "time_open": datetime.fromtimestamp(pos.time, tz=timezone.utc),
                        "be_ratcheted": False,
                        "prox_sl_fired": False,
                        "prox_tp_fired": False,
                        "magic": pos.magic,
                    }

                    # Determine if it's already in breakeven state upon boot
                    if pos.sl > 0:
                        if is_buy and pos.sl >= pos.price_open:
                            pos_record["be_ratcheted"] = True
                        elif not is_buy and pos.sl <= pos.price_open:
                            pos_record["be_ratcheted"] = True

                    self._positions[ticket] = pos_record

                    # Fire Entry Signal if not during initial boot scan or if auto_notify enabled
                    if self._initialized or self.auto_notify_initial_positions:
                        logger.info(">>> [SENTINEL ENTRY] #%d %s %s @ %.2f (SL: %.2f, TP: %.2f)",
                                    ticket, sym, direction, pos.price_open, pos.sl, pos.tp)
                        self.notifier.notify_vip_entry(
                            symbol=sym,
                            direction=direction,
                            lots=pos.volume,
                            price=pos.price_open,
                            sl=pos.sl,
                            tp=pos.tp,
                            ticket=ticket,
                        )
                    continue

                # --- 2. STOP-LOSS & BREAKEVEN MODIFICATION DETECTION ---
                rec = self._positions[ticket]
                old_sl = rec["current_sl"]
                new_sl = pos.sl

                if abs(new_sl - old_sl) > 1e-4:
                    rec["current_sl"] = new_sl
                    logger.info(">>> [SENTINEL SL MODIFIED] #%d %s SL: %.2f -> %.2f (Entry: %.2f)",
                                ticket, sym, old_sl, new_sl, rec["entry_price"])

                    # Check if modified into Breakeven / Protected Zone
                    is_now_be = False
                    locked_profit = 0.0

                    if is_buy:
                        if new_sl >= rec["entry_price"] and not rec["be_ratcheted"]:
                            is_now_be = True
                            locked_profit = (new_sl - rec["entry_price"]) * pos.volume * (info.trade_tick_value if info else 1.0)
                    else:
                        if new_sl <= rec["entry_price"] and not rec["be_ratcheted"]:
                            is_now_be = True
                            locked_profit = (rec["entry_price"] - new_sl) * pos.volume * (info.trade_tick_value if info else 1.0)

                    if is_now_be:
                        rec["be_ratcheted"] = True
                        logger.info(">>> [SENTINEL BREAKEVEN ACTIVATED] #%d %s SL locked at %.2f",
                                    ticket, sym, new_sl)
                        self.notifier.notify_vip_breakeven(
                            ticket=ticket,
                            symbol=sym,
                            direction=direction,
                            entry_price=rec["entry_price"],
                            old_sl=old_sl,
                            new_sl=new_sl,
                            locked_profit=max(0.0, locked_profit),
                        )

                # --- 3. PROXIMITY / TOUCH WARNINGS ---
                if self.enable_proximity_warnings and tick:
                    # Check SL proximity
                    if rec["current_sl"] > 0:
                        sl_dist = (current_p - rec["current_sl"]) if is_buy else (rec["current_sl"] - current_p)
                        if 0 <= sl_dist <= prox_thresh and not rec["prox_sl_fired"]:
                            rec["prox_sl_fired"] = True
                            logger.info(">>> [SENTINEL PROXIMITY SL] #%d %s testing SL @ %.2f (Dist: %.2f)",
                                        ticket, sym, rec["current_sl"], sl_dist)
                            self.notifier.notify_vip_touch_warning(
                                ticket=ticket,
                                symbol=sym,
                                direction=direction,
                                target_type="SL",
                                current_price=current_p,
                                target_price=rec["current_sl"],
                                dist_pts=sl_dist,
                            )
                        elif sl_dist > prox_thresh * 2.5:
                            rec["prox_sl_fired"] = False

                    # Check TP proximity
                    if rec["current_tp"] > 0:
                        tp_dist = (rec["current_tp"] - current_p) if is_buy else (current_p - rec["current_tp"])
                        if 0 <= tp_dist <= prox_thresh and not rec["prox_tp_fired"]:
                            rec["prox_tp_fired"] = True
                            logger.info(">>> [SENTINEL PROXIMITY TP] #%d %s testing TP @ %.2f (Dist: %.2f)",
                                        ticket, sym, rec["current_tp"], tp_dist)
                            self.notifier.notify_vip_touch_warning(
                                ticket=ticket,
                                symbol=sym,
                                direction=direction,
                                target_type="TP",
                                current_price=current_p,
                                target_price=rec["current_tp"],
                                dist_pts=tp_dist,
                            )
                        elif tp_dist > prox_thresh * 2.5:
                            rec["prox_tp_fired"] = False

        # --- 4. POSITION CLOSURE & HISTORY DEAL INSPECTION ---
        closed_tickets = set(self._positions.keys()) - active_tickets
        for ticket in closed_tickets:
            rec = self._positions.pop(ticket)
            self._handle_position_closure(ticket, rec)

        if not self._initialized:
            self._initialized = True

    def _handle_position_closure(self, ticket: int, rec: Dict[str, Any]):
        """Inspects MT5 deal history to classify whether exit was SL, TP, or Manual."""
        sym = rec["symbol"]
        direction = rec["direction"]
        entry_p = rec["entry_price"]
        was_be = rec.get("be_ratcheted", False)

        now_utc = datetime.now(timezone.utc)
        hold_time_mins = max(0.1, (now_utc - rec["time_open"]).total_seconds() / 60.0)

        # Look back up to 4 hours for the closing deal
        from_time = datetime.now() - timedelta(hours=4)
        deals = mt5.history_deals_get(from_time, datetime.now(), position=ticket)

        closing_deal = None
        if deals:
            for d in reversed(deals):
                if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT, 1, 2):
                    closing_deal = d
                    break
            if closing_deal is None and len(deals) > 0:
                closing_deal = deals[-1]

        exit_p = closing_deal.price if closing_deal else entry_p
        pnl = (closing_deal.profit + closing_deal.swap + closing_deal.commission) if closing_deal else 0.0
        reason_code = getattr(closing_deal, "reason", -1) if closing_deal else -1

        is_sl = (reason_code == 4) or (reason_code == getattr(mt5, "DEAL_REASON_SL", 4))
        is_tp = (reason_code == 5) or (reason_code == getattr(mt5, "DEAL_REASON_TP", 5))

        # Fallback price comparison if broker didn't tag reason
        if not (is_sl or is_tp):
            if rec["current_tp"] > 0 and abs(exit_p - rec["current_tp"]) <= abs(exit_p - rec["current_sl"]):
                is_tp = True
            elif rec["current_sl"] > 0:
                is_sl = True

        if is_tp:
            target_pts = abs(exit_p - entry_p)
            logger.info(">>> [SENTINEL TP HIT] #%d %s TP filled @ %.2f (PnL: +$%.2f, Hold: %.1fm)",
                        ticket, sym, exit_p, pnl, hold_time_mins)
            self.notifier.notify_vip_tp_hit(
                ticket=ticket,
                symbol=sym,
                direction=direction,
                exit_price=exit_p,
                pnl=pnl,
                hold_time_mins=hold_time_mins,
                target_pts=target_pts,
            )
        elif is_sl:
            logger.info(">>> [SENTINEL SL HIT] #%d %s SL touched @ %.2f (PnL: $%.2f, Breakeven: %s)",
                        ticket, sym, exit_p, pnl, was_be)
            self.notifier.notify_vip_sl_hit(
                ticket=ticket,
                symbol=sym,
                direction=direction,
                exit_price=exit_p,
                pnl=pnl,
                hold_time_mins=hold_time_mins,
                was_breakeven=was_be or (pnl >= -0.50),
            )
        else:
            logger.info(">>> [SENTINEL POSITION CLOSED] #%d %s Exit @ %.2f (PnL: $%.2f, Reason: %d)",
                        ticket, sym, exit_p, pnl, reason_code)
            self.notifier.notify_position_closed(
                ticket=ticket,
                symbol=sym,
                direction=direction,
                pnl=pnl,
                hold_time_mins=hold_time_mins,
                reason="Manual Discretionary Close" if reason_code == 0 else f"Deal Exit (code {reason_code})",
            )
