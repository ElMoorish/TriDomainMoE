"""
Tri-Domain MoE External Trade Notifier & Performance Recap Generator
====================================================================
Dispatches real-time trade execution signals, breakeven ratchet updates,
and automated Daily & Sunday Midnight Weekly Recaps to Telegram and Discord.

Zero-Latency Architecture:
Dispatches run in an asynchronous background queue worker daemon thread,
ensuring 0.0ms delay on MetaTrader 5 order execution and model inference.
"""

from __future__ import annotations

import os
import sys
import json
import time
import queue
import threading
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

from src.utils.logger import setup_logger

logger = setup_logger("TradeNotifier")


def load_notification_credentials() -> Dict[str, str]:
    """
    Resolves notification credentials from local environment, .env file,
    or neighboring FinRL-X-MT5 configuration.
    """
    creds = {
        "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
        "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID", "").strip(),
        "DISCORD_WEBHOOK_URL": os.environ.get("DISCORD_WEBHOOK_URL", "").strip(),
        "ENABLE_NOTIFICATIONS": os.environ.get("ENABLE_NOTIFICATIONS", "true").strip().lower() in ("true", "1", "yes"),
    }

    # Helper to parse key-value lines from a .env file
    def parse_env_file(path: Path):
        if not path.is_file():
            return
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            # Direct match
            if k in creds and not creds[k]:
                creds[k] = v
            # FinRL-X nested key format: NOTIFICATIONS__TELEGRAM_BOT_TOKEN
            elif k == "NOTIFICATIONS__TELEGRAM_BOT_TOKEN" and not creds["TELEGRAM_BOT_TOKEN"]:
                creds["TELEGRAM_BOT_TOKEN"] = v
            elif k == "NOTIFICATIONS__TELEGRAM_CHAT_ID" and not creds["TELEGRAM_CHAT_ID"]:
                creds["TELEGRAM_CHAT_ID"] = v
            elif k == "NOTIFICATIONS__DISCORD_WEBHOOK_URL" and not creds["DISCORD_WEBHOOK_URL"]:
                creds["DISCORD_WEBHOOK_URL"] = v

    # 1. Search in current repository root
    repo_root = Path(__file__).resolve().parent.parent.parent
    parse_env_file(repo_root / ".env")

    # 2. Fallback to FinRL-X-MT5 if credentials are not yet populated
    if not (creds["TELEGRAM_BOT_TOKEN"] and creds["TELEGRAM_CHAT_ID"]):
        finrl_env = repo_root.parent / "FinRL-X-MT5" / ".env"
        parse_env_file(finrl_env)

    return creds


class TradeNotifier:
    """
    Non-blocking notification dispatcher for Telegram and Discord.
    Uses a dedicated background thread with a thread-safe task queue.
    """

    def __init__(self):
        self.creds = load_notification_credentials()
        self.tg_token = self.creds["TELEGRAM_BOT_TOKEN"]
        self.tg_chat_id = self.creds["TELEGRAM_CHAT_ID"]
        self.discord_url = self.creds["DISCORD_WEBHOOK_URL"]
        self.enabled = self.creds["ENABLE_NOTIFICATIONS"]

        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=1000)
        self._worker_thread = threading.Thread(target=self._dispatch_loop, daemon=True, name="NotifierWorker")
        self._worker_thread.start()

        if self.has_telegram:
            logger.info("Telegram Notifier active (Chat ID: %s...)", self.tg_chat_id[:4] if len(self.tg_chat_id) >= 4 else "***")
        if self.has_discord:
            logger.info("Discord Webhook Notifier active.")

    @property
    def has_telegram(self) -> bool:
        return bool(self.tg_token and self.tg_chat_id and self.enabled)

    @property
    def has_discord(self) -> bool:
        return bool(self.discord_url and self.discord_url.startswith("http") and self.enabled)

    def _dispatch_loop(self):
        """Background daemon processing queued notification payloads."""
        while True:
            try:
                task = self._queue.get()
                if task is None:
                    break

                channel = task.get("channel")
                url = task.get("url")
                payload = task.get("payload")

                try:
                    data_bytes = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        url,
                        data=data_bytes,
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "TriDomainMoE-Institutional-Notifier/1.0",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        _ = resp.read()
                except Exception as ex:
                    logger.debug("Dispatch failed to %s: %s", channel, ex)
                finally:
                    self._queue.task_done()
            except Exception as e:
                logger.debug("Error in notifier dispatch loop: %s", e)

    def post_telegram(self, text: str):
        """Enqueue an HTML formatted message to Telegram."""
        if not self.has_telegram:
            return
        tg_url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        payload = {
            "chat_id": self.tg_chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            self._queue.put_nowait({"channel": "telegram", "url": tg_url, "payload": payload})
        except queue.Full:
            logger.warning("Telegram notification queue full; dropping message.")

    def post_discord(self, payload: Dict[str, Any]):
        """Enqueue a rich embed to Discord."""
        if not self.has_discord:
            return
        try:
            self._queue.put_nowait({"channel": "discord", "url": self.discord_url, "payload": payload})
        except queue.Full:
            logger.warning("Discord notification queue full; dropping message.")

    def notify_signal_dispatched(
        self,
        symbol: str,
        direction: str,
        lots: float,
        price: float,
        sl: float,
        tp: float,
        risk_pct: float,
        conviction: float,
        weights: List[float],
        entropy: float,
        h1_trend: float,
        magic: int = 777123,
    ):
        """Dispatch real-time trade entry teaser with full institutional MoE telemetry."""
        is_buy = direction.upper() == "BUY"
        dir_emoji = "🟢 BUY (LONG)" if is_buy else "🔴 SELL (SHORT)"
        sl_dist = abs(price - sl)
        tp_dist = abs(tp - price)
        rr_ratio = (tp_dist / sl_dist) if sl_dist > 0 else 0.0

        w_tech = weights[0] * 100.0 if len(weights) > 0 else 33.3
        w_macro = weights[1] * 100.0 if len(weights) > 1 else 33.3
        w_fund = weights[2] * 100.0 if len(weights) > 2 else 33.3

        tg_text = (
            f"🚀 <b>TriDomainMoE LIVE ORDER DISPATCHED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Symbol:</b> <code>{symbol}</code>\n"
            f"📊 <b>Action:</b> <b>{dir_emoji}</b>\n"
            f"📦 <b>Volume:</b> <code>{lots:.2f} lots</code> @ <b>{price:,.2f}</b>\n"
            f"🛡️ <b>Stop-Loss:</b> <code>{sl:,.2f}</code> (Distance: ${sl_dist:,.2f})\n"
            f"🎯 <b>Take-Profit:</b> <code>{tp:,.2f}</code> (Distance: ${tp_dist:,.2f} | R:R = {rr_ratio:.1f}:1)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧠 <b>MoE Router Weights:</b>\n"
            f"  • Technical Specialist: <code>{w_tech:.1f}%</code>\n"
            f"  • Macro Specialist:     <code>{w_macro:.1f}%</code>\n"
            f"  • Fundamental Specialist: <code>{w_fund:.1f}%</code>\n"
            f"🔮 <b>Conviction Sizing:</b> <code>{conviction:.2f}x</code> (Risk: <code>{risk_pct:.2f}% Equity</code>)\n"
            f"🌐 <b>Router Entropy:</b> <code>{entropy:.3f}</code> | H1 Trend: <code>{h1_trend:+.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏛️ <i>TriDomainMoE Institutional Self-Improving Engine • Magic #{magic}</i>"
        )
        self.post_telegram(tg_text)

        # Discord Embed
        if self.has_discord:
            embed_color = 0x00FF88 if is_buy else 0xFF3366
            discord_payload = {
                "username": "TriDomainMoE Live Execution",
                "avatar_url": "https://raw.githubusercontent.com/ElMoorish/TriDomainMoE/main/assets/tridomain_moe_banner.jpg",
                "embeds": [
                    {
                        "title": f"🚀 Live Order Filled: {direction.upper()} {symbol}",
                        "color": embed_color,
                        "fields": [
                            {"name": "Action", "value": f"**{direction.upper()}** `{lots:.2f} lots` @ `{price:,.2f}`", "inline": True},
                            {"name": "Bracket SL / TP", "value": f"SL: `{sl:,.2f}` | TP: `{tp:,.2f}` (R:R {rr_ratio:.1f}:1)", "inline": True},
                            {"name": "MoE Allocation", "value": f"Tech `{w_tech:.0f}%` | Macro `{w_macro:.0f}%` | Fund `{w_fund:.0f}%`", "inline": False},
                            {"name": "Conviction & Risk", "value": f"Conviction `{conviction:.2f}` | Ceiling `{risk_pct:.2f}%`", "inline": True},
                            {"name": "Gating Entropy", "value": f"`H={entropy:.3f}` | H1 Trend `{h1_trend:+.2f}`", "inline": True},
                        ],
                        "footer": {"text": f"TriDomainMoE Live Desk • Magic #{magic}"},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            }
            self.post_discord(discord_payload)

    def notify_breakeven_ratchet(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        old_sl: float,
        new_sl: float,
        locked_profit: float,
    ):
        """Dispatch Breakeven Ratchet profit lock alert."""
        tg_text = (
            f"🛡️ <b>TriDomainMoE BREAKEVEN RATCHET ACTIVATED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎫 <b>Ticket:</b> <code>#{ticket}</code> ({symbol} {direction.upper()})\n"
            f"📍 <b>Entry Price:</b> <code>{entry_price:,.2f}</code>\n"
            f"🔒 <b>Protective SL Moved:</b> <code>{old_sl:,.2f}</code> ➔ <b>{new_sl:,.2f}</b>\n"
            f"💰 <b>Guaranteed Minimum Profit:</b> <b>+${locked_profit:,.2f}</b> (Spread & Buffer Protected)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ <i>Zero downside trade: Free ride on institutional momentum.</i>"
        )
        self.post_telegram(tg_text)

    def notify_position_closed(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        pnl: float,
        hold_time_mins: float,
        reason: str = "SL/TP Hit",
    ):
        """Dispatch trade closure notification."""
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_emoji = "🟢 WIN" if pnl >= 0 else "🔴 LOSS"
        tg_text = (
            f"🏁 <b>TriDomainMoE POSITION CLOSED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎫 <b>Ticket:</b> <code>#{ticket}</code> ({symbol} {direction.upper()})\n"
            f"📊 <b>Outcome:</b> <b>{pnl_emoji}</b> ({reason})\n"
            f"💰 <b>Realized Net PnL:</b> <b>{pnl_sign}${pnl:,.2f}</b>\n"
            f"⏱️ <b>Holding Duration:</b> <code>{hold_time_mins:.1f} minutes</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏛️ <i>Live capital returned to risk controller for reallocation.</i>"
        )
        self.post_telegram(tg_text)


class TriDomainRecapGenerator:
    """
    Scans MT5 history deals and compiles Daily and Sunday Midnight Weekly Recaps.
    Dispatches to Telegram/Discord and archives markdown audit logs in reports/.
    """

    def __init__(self, notifier: Optional[TradeNotifier] = None):
        self.notifier = notifier or TradeNotifier()
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily_recap(
        self,
        date: Optional[datetime] = None,
        symbol: Optional[str] = "BTCUSD.x",
        magic: Optional[Any] = 777123,
        dispatch: bool = True,
        title_suffix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate End-of-Day recap for BTCUSD (triggered daily at 23:55 UTC).
        """
        now = datetime.now(timezone.utc)
        target_date = date or now
        start_dt = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        if end_dt > now:
            end_dt = now

        sym_tag = f"[{symbol}] " if symbol else ""
        suffix_tag = f" ({title_suffix})" if title_suffix else ""
        title = f"Daily Performance Recap — {sym_tag}{start_dt.strftime('%A, %B %d, %Y')}{suffix_tag}"
        recap_data = self._compute_metrics(start_dt, end_dt, symbol, magic_filter=magic, period_type="daily")

        if dispatch:
            period_label = start_dt.strftime("%Y-%m-%d")
            if symbol:
                period_label += f"_{symbol}"
            self._dispatch_recap(recap_data, title, period_label=period_label)

        return recap_data

    def generate_weekly_recap(
        self,
        weeks_back: int = 0,
        symbol: Optional[str] = "BTCUSD.x",
        magic: Optional[Any] = 777123,
        dispatch: bool = True,
        title_suffix: Optional[str] = "Sunday Midnight 7-Day Crypto Cycle",
    ) -> Dict[str, Any]:
        """
        Generate Weekly recap for BTCUSD (triggered on Sunday Midnight at 23:55 UTC).
        Captures the entire 7-day 24/7 crypto trading week (Monday 00:00 through Sunday 23:55 UTC).
        """
        now = datetime.now(timezone.utc)
        monday = now - timedelta(days=now.weekday() + (7 * weeks_back))
        start_dt = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = start_dt + timedelta(days=6, hours=23, minutes=59, seconds=59)
        if end_dt > now:
            end_dt = now

        sym_tag = f"[{symbol}] " if symbol else ""
        suffix_tag = f" ({title_suffix})" if title_suffix else ""
        title = f"Weekly Performance Audit — {sym_tag}Week {start_dt.strftime('%W')} ({start_dt.strftime('%b %d')} → {end_dt.strftime('%b %d, %Y')}){suffix_tag}"
        recap_data = self._compute_metrics(start_dt, end_dt, symbol, magic_filter=magic, period_type="weekly")

        if dispatch:
            period_label = f"Week_{start_dt.strftime('%Y_W%W')}"
            if symbol:
                period_label += f"_{symbol}"
            self._dispatch_recap(recap_data, title, period_label=period_label)

        return recap_data

    def _compute_metrics(
        self,
        start_dt: datetime,
        end_dt: datetime,
        symbol_filter: Optional[str] = None,
        magic_filter: Optional[Any] = 777123,
        period_type: str = "daily",
    ) -> Dict[str, Any]:
        """Query MT5 closed deals and compute performance analytics."""
        if not MT5_AVAILABLE or not mt5.initialize():
            logger.error("MT5 terminal unavailable for recap metrics computation.")
            return {"error": "MT5 not connected"}

        deals = mt5.history_deals_get(start_dt, end_dt)
        closed_deals = []

        if deals:
            for d in deals:
                if d.entry == mt5.DEAL_ENTRY_OUT:
                    if symbol_filter:
                        sym_base = symbol_filter.split(".")[0]
                        d_base = d.symbol.split(".")[0]
                        if d.symbol != symbol_filter and d_base != sym_base:
                            continue
                    if magic_filter is not None:
                        if isinstance(magic_filter, (list, tuple, set)):
                            if d.magic not in magic_filter:
                                continue
                        elif d.magic != magic_filter:
                            continue
                    closed_deals.append(d)

        acc = mt5.account_info()
        current_equity = acc.equity if acc else 10000.0
        current_balance = acc.balance if acc else 10000.0

        if not closed_deals:
            return {
                "period_type": period_type,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "symbol": symbol_filter or "ALL",
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "current_equity": current_equity,
                "current_balance": current_balance,
                "trades": [],
                "session_breakdown": {},
            }

        pnls = [d.profit + d.swap + d.commission for d in closed_deals]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_trades = len(pnls)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        net_pnl = sum(pnls)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        avg_win = (gross_profit / win_count) if win_count > 0 else 0.0
        avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0.0
        best_trade = max(pnls) if pnls else 0.0
        worst_trade = min(pnls) if pnls else 0.0

        # Session Breakdown
        session_stats = {
            "Asian (00-07 UTC)": {"trades": 0, "pnl": 0.0, "wins": 0},
            "London (07-13.5 UTC)": {"trades": 0, "pnl": 0.0, "wins": 0},
            "New York (13.5-21 UTC)": {"trades": 0, "pnl": 0.0, "wins": 0},
            "Off-Hours (21-24 UTC)": {"trades": 0, "pnl": 0.0, "wins": 0},
        }

        for d in closed_deals:
            dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
            h = dt.hour + dt.minute / 60.0
            sess = (
                "Asian (00-07 UTC)" if 0.0 <= h < 7.0
                else "London (07-13.5 UTC)" if 7.0 <= h < 13.5
                else "New York (13.5-21 UTC)" if 13.5 <= h < 21.0
                else "Off-Hours (21-24 UTC)"
            )
            p = d.profit + d.swap + d.commission
            session_stats[sess]["trades"] += 1
            session_stats[sess]["pnl"] += p
            if p > 0:
                session_stats[sess]["wins"] += 1

        trade_details = []
        for d in closed_deals:
            dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
            pnl = d.profit + d.swap + d.commission
            trade_details.append({
                "ticket": d.ticket,
                "symbol": d.symbol,
                "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "volume": d.volume,
                "pnl": pnl,
                "magic": d.magic,
            })

        return {
            "period_type": period_type,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "symbol": symbol_filter or "ALL",
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": win_rate,
            "net_pnl": net_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "current_equity": current_equity,
            "current_balance": current_balance,
            "trades": trade_details,
            "session_breakdown": session_stats,
        }

    def _dispatch_recap(self, data: Dict[str, Any], title: str, period_label: str):
        """Save markdown audit report and broadcast to Telegram and Discord."""
        if "error" in data:
            logger.warning("Recap generation skipped: %s", data["error"])
            return

        # 1. Save Markdown Report
        report_file = self.reports_dir / f"recap_{data['period_type']}_{period_label}.md"
        md_text = self._build_markdown(data, title)
        report_file.write_text(md_text, encoding="utf-8")
        logger.info("Saved %s recap report -> %s", data["period_type"], report_file)

        # 2. Dispatch to Telegram
        if self.notifier.has_telegram:
            self._send_telegram_recap(data, title)

        # 3. Dispatch to Discord
        if self.notifier.has_discord:
            self._send_discord_recap(data, title)

    def _send_telegram_recap(self, data: Dict[str, Any], title: str):
        net_pnl = data["net_pnl"]
        pnl_sign = "+" if net_pnl >= 0 else ""
        pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"

        sess_lines = []
        for sess_name, s in data["session_breakdown"].items():
            if s["trades"] > 0:
                s_wr = (s["wins"] / s["trades"] * 100.0) if s["trades"] > 0 else 0.0
                sess_lines.append(f"  • <b>{sess_name}:</b> <code>{s['trades']} trds</code> | <code>{s_wr:.0f}% WR</code> | <code>${s['pnl']:+,.2f}</code>")
        session_text = "\n".join(sess_lines) if sess_lines else "  • <i>No closed deals in session</i>"

        tg_text = (
            f"📊 <b>{title}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{pnl_emoji} <b>Net PnL:</b> <b>{pnl_sign}${net_pnl:,.2f}</b>\n"
            f"🎯 <b>Win Rate:</b> <b>{data['win_rate']:.1f}%</b> ({data['wins']}W / {data['losses']}L)\n"
            f"⚖️ <b>Profit Factor:</b> <code>{data['profit_factor']:.2f}</code>\n"
            f"📦 <b>Total Trades:</b> <code>{data['total_trades']}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 <b>Gross Profit:</b> <code>+${data['gross_profit']:,.2f}</code>\n"
            f"📉 <b>Gross Loss:</b> <code>-${data['gross_loss']:,.2f}</code>\n"
            f"🏆 <b>Best Trade:</b> <code>+${data['best_trade']:,.2f}</code>\n"
            f"⚠️ <b>Worst Trade:</b> <code>${data['worst_trade']:,.2f}</code>\n"
            f"📊 <b>Avg Win / Loss:</b> <code>+${data['avg_win']:.2f} / -${data['avg_loss']:.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌍 <b>Session Breakdown:</b>\n"
            f"{session_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏛️ <b>Account Equity:</b> <code>${data['current_equity']:,.2f}</code>\n"
            f"💵 <b>Balance:</b> <code>${data['current_balance']:,.2f}</code>\n"
            f"🛡️ <b>Engine:</b> <code>TriDomainMoE v2 • 0.20% Scaled Ceiling</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>Verified Institutional Multi-Domain Telemetry</i>"
        )
        self.notifier.post_telegram(tg_text)

    def _send_discord_recap(self, data: Dict[str, Any], title: str):
        net_pnl = data["net_pnl"]
        is_profit = net_pnl >= 0
        embed_color = 0x00FF88 if is_profit else 0xFF3366
        pnl_sign = "+" if net_pnl >= 0 else ""

        fields = [
            {"name": "💰 Net PnL", "value": f"**`{pnl_sign}${net_pnl:,.2f}`**", "inline": True},
            {"name": "🎯 Win Rate", "value": f"**`{data['win_rate']:.1f}%`** ({data['wins']}W / {data['losses']}L)", "inline": True},
            {"name": "⚖️ Profit Factor", "value": f"**`{data['profit_factor']:.2f}`**", "inline": True},
            {"name": "📈 Gross Profit", "value": f"`+${data['gross_profit']:,.2f}`", "inline": True},
            {"name": "📉 Gross Loss", "value": f"`-${data['gross_loss']:,.2f}`", "inline": True},
            {"name": "📦 Total Closed", "value": f"`{data['total_trades']}`", "inline": True},
            {"name": "🏆 Best Trade", "value": f"`+${data['best_trade']:,.2f}`", "inline": True},
            {"name": "⚠️ Worst Trade", "value": f"`${data['worst_trade']:,.2f}`", "inline": True},
            {"name": "🏛️ Equity", "value": f"`${data['current_equity']:,.2f}`", "inline": True},
        ]

        payload = {
            "username": "TriDomainMoE Performance Journal",
            "avatar_url": "https://raw.githubusercontent.com/ElMoorish/TriDomainMoE/main/assets/tridomain_moe_banner.jpg",
            "embeds": [
                {
                    "title": f"📊 {title}",
                    "description": "Automated performance audit from **TriDomainMoE Institutional Engine**.",
                    "color": embed_color,
                    "fields": fields,
                    "footer": {"text": "TriDomainMoE • Sunday Midnight & Daily Performance Loop"},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }
        self.notifier.post_discord(payload)

    def _build_markdown(self, data: Dict[str, Any], title: str) -> str:
        pnl_sign = "+" if data["net_pnl"] >= 0 else ""
        lines = [
            f"# {title}",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Period:** {data['start_dt'].strftime('%Y-%m-%d %H:%M')} → {data['end_dt'].strftime('%Y-%m-%d %H:%M')} UTC",
            f"**Symbol Filter:** `{data['symbol']}`",
            "",
            "## Executive Summary",
            "",
            "| Metric | Result |",
            "| :--- | :--- |",
            f"| **Net PnL** | **{pnl_sign}${data['net_pnl']:,.2f}** |",
            f"| **Win Rate** | **{data['win_rate']:.1f}%** ({data['wins']} Wins / {data['losses']} Losses) |",
            f"| **Profit Factor** | **{data['profit_factor']:.2f}** |",
            f"| **Gross Profit** | +${data['gross_profit']:,.2f} |",
            f"| **Gross Loss** | -${data['gross_loss']:,.2f} |",
            f"| **Total Closed Trades** | {data['total_trades']} |",
            f"| **Average Win** | +${data['avg_win']:.2f} |",
            f"| **Average Loss** | -${data['avg_loss']:.2f} |",
            f"| **Best Trade** | +${data['best_trade']:,.2f} |",
            f"| **Worst Trade** | ${data['worst_trade']:,.2f} |",
            f"| **Ending Equity** | ${data['current_equity']:,.2f} |",
            f"| **Ending Balance** | ${data['current_balance']:,.2f} |",
            "",
            "## Session Performance",
            "",
            "| Session | Trades | Win Rate | Net PnL |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for s_name, s in data["session_breakdown"].items():
            wr = (s["wins"] / s["trades"] * 100.0) if s["trades"] > 0 else 0.0
            lines.append(f"| **{s_name}** | {s['trades']} | {wr:.1f}% | ${s['pnl']:+,.2f} |")

        lines.extend([
            "",
            "---",
            "*Report auto-generated by TriDomainMoE Institutional Surveillance Engine.*",
        ])
        return "\n".join(lines)
