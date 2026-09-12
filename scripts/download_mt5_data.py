"""
CLI Utility to Ingest Market Data from MetaTrader 5 into High-Performance Parquet Storage.
Computes Volume Bars, Order Flow Imbalance (OFI), and optimal Fractional Differencing order (d*).
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mt5_loader import MT5DataLoader
from src.data.storage import ParquetStorage
from src.features.volume_bars import VolumeBarBuilder
from src.features.ofi import OrderFlowImbalance
from src.features.fracdiff import FractionalDifferentiator
from src.features.sentiment_loader import YahooFinanceSentimentLoader
from src.features.sentiment_engine import MacroSentimentEngine
from src.utils.logger import setup_logger

logger = setup_logger("MT5DataDownloader")


def parse_args():
    parser = argparse.ArgumentParser(description="Download MT5 data and compute microstructure features.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSD.x"],
        help="Symbols to download from MT5 (e.g. BTCUSD.x ETHUSD.x)",
    )
    parser.add_argument("--days", type=int, default=365, help="Number of historical days to download")
    parser.add_argument("--timeframe", type=str, default="M5", help="OHLC timeframe (e.g. M1, M5, H1)")
    parser.add_argument("--ticks", action="store_true", default=False, help="Also download tick data (slow for long history)")
    parser.add_argument("--volume-threshold", type=float, default=500.0, help="Volume bar aggregation threshold")
    parser.add_argument("--fetch-news", action="store_true", default=False, help="Harvest Yahoo Finance news")
    return parser.parse_args()


def main():
    args = parse_args()
    loader = MT5DataLoader()
    storage = ParquetStorage()

    if not loader.connect():
        logger.error("Failed to connect to MT5 terminal. Ensure MT5 is running.")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=args.days)

    logger.info("Starting ingestion from %s to %s for symbols: %s", start_time, now, args.symbols)

    for symbol in args.symbols:
        logger.info("=== Processing Symbol: %s ===", symbol)
        sym_info = loader.get_symbol_info(symbol)
        if sym_info is None:
            logger.warning("Symbol %s unavailable in MT5. Skipping.", symbol)
            continue

        # 1. Download OHLC Bars
        logger.info("Fetching %s OHLC bars for %s...", args.timeframe, symbol)
        bars_df = loader.get_bars(symbol, timeframe=args.timeframe, start_time=start_time, end_time=now)
        if not bars_df.empty:
            storage.save_bars(symbol, args.timeframe, bars_df)
            logger.info("Saved %d %s bars for %s.", len(bars_df), args.timeframe, symbol)

            # Compute Fractional Differencing d* on Close price
            logger.info("Computing Fractional Differencing d* on %s close prices...", symbol)
            d_star, adf_diag = FractionalDifferentiator.find_min_d(bars_df["close"], step=0.05)
            logger.info("Optimal stationarity order d* for %s = %.2f (preserving memory)", symbol, d_star)
        else:
            logger.warning("No OHLC bars received for %s.", symbol)

        # 2. Download Ticks and Build Volume Bars & OFI
        if args.ticks:
            logger.info("Fetching tick stream for %s...", symbol)
            ticks_df = loader.get_ticks(symbol, start_time=start_time, end_time=now)
            if not ticks_df.empty:
                storage.save_ticks(symbol, ticks_df)
                logger.info("Saved %d raw ticks for %s.", len(ticks_df), symbol)

                # Aggregate into Volume Bars
                logger.info("Building volume bars (threshold: %.1f)...", args.volume_threshold)
                vb_builder = VolumeBarBuilder(volume_threshold=args.volume_threshold)
                vol_bars = vb_builder.build_bars(ticks_df)

                if not vol_bars.empty:
                    # Enrich with OFI
                    enriched_vol_bars = OrderFlowImbalance.aggregate_to_bars(ticks_df, vol_bars)
                    storage.save_bars(symbol, "vol_bars", enriched_vol_bars)
                    logger.info(
                        "Generated %d volume bars with OFI for %s. (Sample OFI mean: %.4f)",
                        len(enriched_vol_bars),
                        symbol,
                        enriched_vol_bars["normalized_ofi"].mean(),
                    )
            else:
                logger.warning("No ticks returned for %s in date range.", symbol)

    # 3. Harvest Yahoo Finance News Sentiment
    if args.fetch_news:
        logger.info("=== Harvesting Yahoo Finance Sentiment & Macro State ===")
        sentiment_loader = YahooFinanceSentimentLoader()
        sentiment_engine = MacroSentimentEngine()

        articles = sentiment_loader.fetch_articles()
        regime_vec = sentiment_engine.update(articles)
        logger.info("Articles fetched: %d", len(articles))
        logger.info(
            "Current Macro Regime Vector z_t:\n"
            "  - Net Sentiment:    %+.3f\n"
            "  - Uncertainty:      %.3f\n"
            "  - Monetary Policy:  %.3f\n"
            "  - Geopolitics:      %.3f\n"
            "  - Corporate Tech:   %.3f\n"
            "  - Commodities:      %.3f",
            regime_vec[0],
            regime_vec[1],
            regime_vec[2],
            regime_vec[3],
            regime_vec[4],
            regime_vec[5],
        )

    loader.disconnect()
    logger.info("=== Ingestion and Microstructure Pipeline Run Complete ===")


if __name__ == "__main__":
    main()
