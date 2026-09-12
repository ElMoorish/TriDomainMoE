"""Unit tests for VolumeBarBuilder."""

import unittest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from src.features.volume_bars import VolumeBarBuilder


class TestVolumeBarBuilder(unittest.TestCase):
    def setUp(self):
        # Create synthetic tick series
        start = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        timestamps = [start + timedelta(seconds=i) for i in range(100)]
        
        # Prices oscillate between 100 and 105
        prices = [100.0 + (i % 5) for i in range(100)]
        volumes = [50.0 for _ in range(100)]  # 50 volume per tick

        self.ticks_df = pd.DataFrame({
            "bid": [p - 0.05 for p in prices],
            "ask": [p + 0.05 for p in prices],
            "last": prices,
            "volume": volumes,
            "flags": [0] * 100,
        }, index=pd.DatetimeIndex(timestamps, name="timestamp"))

    def test_volume_bar_aggregation(self):
        # Threshold 250 -> 5 ticks per bar
        builder = VolumeBarBuilder(volume_threshold=250.0)
        bars = builder.build_bars(self.ticks_df)

        self.assertFalse(bars.empty)
        self.assertEqual(len(bars), 20)  # 100 ticks / 5 ticks per bar = 20 bars
        
        # Verify OHLC invariants
        for _, row in bars.iterrows():
            self.assertGreaterEqual(row["high"], row["low"])
            self.assertGreaterEqual(row["high"], row["open"])
            self.assertGreaterEqual(row["high"], row["close"])
            self.assertLessEqual(row["low"], row["open"])
            self.assertLessEqual(row["low"], row["close"])
            self.assertAlmostEqual(row["volume"], 250.0)
            self.assertEqual(row["ticks"], 5)


if __name__ == "__main__":
    unittest.main()
