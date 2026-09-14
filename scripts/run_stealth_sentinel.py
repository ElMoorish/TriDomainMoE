"""
scripts/run_stealth_sentinel.py
===============================
Standalone External Stealth Signal Sentinel Daemon for MT5 (BTCUSD & NAS100).
Dispatches VIP Entry, Breakeven Ratchet, SL Touched, and TP Targets Achieved
with zero trace inside MetaTrader 5 (Anti-Prop Firm Detection).
"""

import sys
import time
import signal
import argparse
import logging
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.execution.stealth_sentinel import StealthSignalSentinel
from src.execution.notifier import TradeNotifier
from src.utils.logger import setup_logger

logger = setup_logger("StealthSentinelDaemon")


def parse_args():
    parser = argparse.ArgumentParser(description="External Stealth Signal Sentinel for MT5")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSD.x", "NAS100.x"],
        help="Symbols to surveil (e.g. BTCUSD.x NAS100.x)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds (default: 2.0s)",
    )
    parser.add_argument(
        "--proximity-warnings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SL/TP proximity touch warning alerts",
    )
    parser.add_argument(
        "--notify-existing",
        action="store_true",
        default=False,
        help="Broadcast entry signal for existing positions already open on startup",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Run a single scan cycle and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 75)
    print("   Tri-Domain MoE — External Stealth Signal Sentinel (Zero-Trace)")
    print("=" * 75)
    print(f"  [+] Target Assets:          {', '.join(args.symbols)}")
    print(f"  [+] Polling Interval:       {args.interval}s")
    print(f"  [+] Proximity Warnings:     {'ENABLED' if args.proximity_warnings else 'DISABLED'}")
    print("  [+] Prop Firm Stealth:      ACTIVE")
    print("      • Zero MQL5 WebRequests (MT5 terminal logs remain 100% blank)")
    print("      • Zero Signal comments on MT5 broker tickets")
    print("      • Read-only local Windows memory IPC (mt5.positions_get)")
    print("      • External Python network dispatcher to VIP Channel & Admin DM")
    print("=" * 75)

    notifier = TradeNotifier()
    sentinel = StealthSignalSentinel(
        symbols=args.symbols,
        notifier=notifier,
        enable_proximity_warnings=args.proximity_warnings,
        auto_notify_initial_positions=args.notify_existing,
    )

    if not sentinel.initialize_mt5():
        logger.error("Could not connect to MT5 terminal. Please ensure MT5 is running.")
        sys.exit(1)

    # Handle graceful exit
    running = True

    def sig_handler(signum, frame):
        nonlocal running
        logger.info("Termination signal received. Shutting down Stealth Sentinel...")
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    if args.once:
        logger.info("Executing single surveillance scan...")
        sentinel.scan_once()
        notifier.flush(timeout=5.0)
        logger.info("Single scan complete.")
        return

    logger.info("Stealth Sentinel daemon running. Monitoring active positions 24/7 (Press CTRL+C to stop)...")
    scan_count = 0

    while running:
        try:
            sentinel.scan_once()
            scan_count += 1
            if scan_count % 300 == 0:  # Every ~10 minutes at 2s interval
                logger.info("[HEARTBEAT] Stealth Sentinel active. Surveilling %d tracked positions on %s",
                            len(sentinel._positions), sentinel.resolved_symbols)
            time.sleep(args.interval)
        except Exception as e:
            logger.error("Error during sentinel scan cycle: %s", e, exc_info=True)
            time.sleep(args.interval)

    notifier.flush(timeout=5.0)
    logger.info("Stealth Sentinel shutdown complete.")


if __name__ == "__main__":
    main()
