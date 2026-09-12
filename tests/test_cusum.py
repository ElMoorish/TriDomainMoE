"""Unit tests for CUSUMFilter and TripleBarrierLabeler."""

import unittest
import numpy as np
import pandas as pd

from src.features.cusum_sampler import CUSUMFilter
from src.features.labeling import TripleBarrierLabeler


class TestCUSUMAndLabeling(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        returns = np.random.normal(0.0001, 0.005, n)
        prices = 100.0 * np.exp(np.cumsum(returns))
        self.price_series = pd.Series(prices, index=dates, name="price")

    def test_cusum_filtering(self):
        filt = CUSUMFilter(h_multiplier=1.0, vol_lookback=50)
        events = filt.filter_events(self.price_series)
        self.assertGreater(len(events), 0)
        self.assertLessEqual(len(events), len(self.price_series))

    def test_triple_barrier_labeling(self):
        labeler = TripleBarrierLabeler(pt_multiplier=1.5, sl_multiplier=1.5, max_holding_bars=10)
        labeled = labeler.label_events(self.price_series)
        self.assertFalse(labeled.empty)
        self.assertTrue(set(labeled["label"].unique()).issubset({-1, 0, 1}))
        self.assertIn("holding_bars", labeled.columns)

    def test_calibrated_sizing(self):
        probs = np.array([0.1, 0.5, 0.6, 0.8, 0.95])
        sizes = TripleBarrierLabeler.calibrated_bet_size(probs, sigma_cal=0.25)
        # At p=0.5, size should be 0.0 (no conviction)
        self.assertAlmostEqual(sizes[1], 0.0, places=5)
        # Sizing should be monotonic for p > 0.5
        self.assertGreater(sizes[3], sizes[2])
        self.assertGreater(sizes[4], sizes[3])


if __name__ == "__main__":
    unittest.main()
