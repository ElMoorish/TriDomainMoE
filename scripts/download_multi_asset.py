"""
Multi-Asset Market Data Ingestion Pipeline.
Downloads synchronized historical feeds from MT5 across:
  - NAS100.x (Equities)
  - XAUUSD.x (Gold)
  - WTI.x (Crude Oil)
  - EURUSD.x (FX Macro)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mt5_loader import MT5DataLoader
from src.data.storage import ParquetStorage
from src.features.volume_bars import VolumeBarBuilder
from src.features.ofi import OrderFlowImbalance
from src.features.sentiment_loader import YahooFinanceSentimentLoader
from src.features.sentiment_engine import MacroSentimentEngine
from src.features.cross_asset import CrossAssetFeatureEngine
from src.utils.logger import setup_logger

logger = setup_logger("MultiAssetDownloader")

CORE_ASSETS = ["NAS100.x", "XAUUSD.x", "WTI.x", "EURUSD.x"]


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Ingestion for Tri-Domain MoE")
    parser.add_argument("--days", type=int, default=5, help="Number of historical days")
    parser.add_argument("--timeframe", type=str, default="M1", help="OHLC timeframe")
    args = parser.parse_args()

    loader = MT5DataLoader()
    storage = ParquetStorage()

    if not loader.connect():
        logger.error("Failed to connect to MT5.")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=args.days)
    logger.info("Ingesting Multi-Asset data from %s to %s...", start_time, now)

    downloaded_bars = {}

    for sym in CORE_ASSETS:
        logger.info("Fetching %s bars for %s...", args.timeframe, sym)
        bars = loader.get_bars(sym, timeframe=args.timeframe, start_time=start_time, end_time=now)
        if not bars.empty:
            storage.save_bars(sym, args.timeframe, bars)
            downloaded_bars[sym] = bars
            logger.info("Retrieved %d %s bars for %s.", len(bars), args.timeframe, sym)
        else:
            logger.warning("Failed to retrieve bars for %s.", sym)

    # If all core assets retrieved, construct aligned cross-asset feature table
    if all(sym in downloaded_bars for sym in CORE_ASSETS):
        logger.info("Building synchronized Cross-Asset matrix...")
        cross_matrix = CrossAssetFeatureEngine.build_cross_asset_matrix(
            equity_bars=downloaded_bars["NAS100.x"],
            gold_bars=downloaded_bars["XAUUSD.x"],
            oil_bars=downloaded_bars["WTI.x"],
            fx_bars=downloaded_bars["EURUSD.x"],
        )
        storage.save_bars("PORTFOLIO", "cross_asset", cross_matrix)
        logger.info("Cross-Asset matrix constructed with %d aligned time periods.", len(cross_matrix))

    # Also harvest fresh Yahoo Finance news
    logger.info("Harvesting macro sentiment articles...")
    sentiment_loader = YahooFinanceSentimentLoader()
    sentiment_engine = MacroSentimentEngine()
    articles = sentiment_loader.fetch_articles()
    z_regime = sentiment_engine.update(articles)
    logger.info("Current Regime Vector z_t: Sentiment=%.3f, Uncertainty=%.3f", z_regime[0], z_regime[1])

    loader.disconnect()
    logger.info("=== Multi-Asset Ingestion Complete ===")


if __name__ == "__main__":
    main()
