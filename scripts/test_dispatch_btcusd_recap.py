"""
Live Test Dispatcher for BTCUSD Daily and Sunday Midnight Weekly Recaps
=======================================================================
Connects to MT5, queries closed deals for BTCUSD.x, compiles the
Daily Recap and Sunday Midnight Weekly Recap, saves markdown reports in
reports/, and dispatches real-time broadcasts to Telegram and Discord.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Ensure workspace is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.execution.notifier import TradeNotifier, TriDomainRecapGenerator
import MetaTrader5 as mt5

def main():
    print("==========================================================================")
    print("      TRIDOMAINMOE — LIVE BTCUSD RECAP DISPATCH TEST                      ")
    print("==========================================================================")

    # 1. Initialize MT5
    if not mt5.initialize():
        print("[ERROR] Could not connect to MetaTrader 5 terminal.")
        sys.exit(1)

    notifier = TradeNotifier()
    recap_gen = TriDomainRecapGenerator(notifier=notifier)

    print(f"Telegram Configured: {notifier.has_telegram}")
    print(f"Discord Configured:  {notifier.has_discord}")
    print(f"Target Symbol:       BTCUSD.x")
    print(f"Magic Number:        All / 777123")

    # 2. Test Daily Recap for BTCUSD
    print("\n--------------------------------------------------------------------------")
    print("1. Compiling & Dispatching BTCUSD Daily Performance Recap...")
    print("--------------------------------------------------------------------------")
    daily_res = recap_gen.generate_daily_recap(
        symbol="BTCUSD.x",
        magic=None,  # query all deals for BTCUSD.x to capture any broker fills
        dispatch=True,
        title_suffix="24/7 Crypto Session",
    )
    print(f"Daily Recap Summary:")
    print(f"  - Period:       {daily_res['start_dt'].strftime('%Y-%m-%d %H:%M')} -> {daily_res['end_dt'].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  - Total Trades: {daily_res['total_trades']}")
    print(f"  - Win Rate:     {daily_res['win_rate']:.1f}% ({daily_res['wins']}W / {daily_res['losses']}L)")
    print(f"  - Net PnL:      ${daily_res['net_pnl']:,.2f}")
    print(f"  - Equity:       ${daily_res['current_equity']:,.2f}")
    print(f"  - Balance:      ${daily_res['current_balance']:,.2f}")

    # 3. Test Sunday Midnight Weekly Recap for BTCUSD
    print("\n--------------------------------------------------------------------------")
    print("2. Compiling & Dispatching BTCUSD Sunday Midnight Weekly Recap...")
    print("--------------------------------------------------------------------------")
    weekly_res = recap_gen.generate_weekly_recap(
        symbol="BTCUSD.x",
        magic=None,
        dispatch=True,
        title_suffix="Sunday Midnight 7-Day Crypto Cycle",
    )
    print(f"Weekly Recap Summary:")
    print(f"  - Period:       {weekly_res['start_dt'].strftime('%Y-%m-%d %H:%M')} -> {weekly_res['end_dt'].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  - Total Trades: {weekly_res['total_trades']}")
    print(f"  - Win Rate:     {weekly_res['win_rate']:.1f}% ({weekly_res['wins']}W / {weekly_res['losses']}L)")
    print(f"  - Net PnL:      ${weekly_res['net_pnl']:,.2f}")
    print(f"  - Profit Factor:{weekly_res['profit_factor']:.2f}")
    print(f"  - Equity:       ${weekly_res['current_equity']:,.2f}")

    # 4. Flush the async queue
    print("\nWaiting for async Telegram & Discord delivery to complete...")
    notifier.flush(timeout=8.0)
    print("[SUCCESS] All notifications flushed and dispatched!")

    # Check generated report files
    report_files = sorted(Path("reports").glob("recap_*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"\nGenerated Audit Reports in reports/:")
    for rf in report_files[:4]:
        print(f"  - {rf} ({rf.stat().st_size} bytes)")

    mt5.shutdown()
    print("\n==========================================================================")
    print("      BTCUSD RECAP TEST COMPLETE — CHECK YOUR TELEGRAM / DISCORD CHANNEL  ")
    print("==========================================================================")

if __name__ == "__main__":
    main()
