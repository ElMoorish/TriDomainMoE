"""Unit tests for CrossAssetFeatureEngine."""

import unittest
import numpy as np
import pandas as pd

from src.features.cross_asset import CrossAssetFeatureEngine


class TestCrossAssetFeatureEngine(unittest.TestCase):
    def test_roll_yield_calculation(self):
        # Backwardation: F1 > F2 -> positive roll yield
        idx = pd.date_range("2026-01-01", periods=10, freq="1D")
        f1 = pd.Series([100.0] * 10, index=idx)
        f2 = pd.Series([98.0] * 10, index=idx)

        roll_yield = CrossAssetFeatureEngine.compute_roll_yield(f1, f2, days_to_expiration_delta=30.0)
        self.assertTrue((roll_yield > 0).all())

    def test_synthetic_dxy(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1D")
        eurusd = pd.Series([1.10, 1.08, 1.05, 1.07, 1.09], index=idx)
        dxy = CrossAssetFeatureEngine.compute_synthetic_dxy(eurusd)

        # When EURUSD drops, DXY should rise
        self.assertGreater(dxy.iloc[2], dxy.iloc[0])  # EURUSD 1.05 < 1.10 -> DXY higher


if __name__ == "__main__":
    unittest.main()
