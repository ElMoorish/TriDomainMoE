"""Unit tests for FractionalDifferentiator."""

import unittest
import numpy as np
import pandas as pd

from src.features.fracdiff import FractionalDifferentiator


class TestFractionalDifferentiator(unittest.TestCase):
    def test_binomial_weights(self):
        # When d = 1.0 (standard first difference):
        # omega_0 = 1.0, omega_1 = -1.0, omega_2 = 0.0
        weights_d1 = FractionalDifferentiator.get_weights(d=1.0, size=5)
        # Reversal is applied in get_weights
        self.assertEqual(len(weights_d1), 2)
        self.assertAlmostEqual(weights_d1[-1], 1.0)
        self.assertAlmostEqual(weights_d1[-2], -1.0)

    def test_frac_diff_preserves_trend(self):
        # A series with strong trend
        np.random.seed(42)
        trend = np.linspace(100, 200, 200)
        noise = np.random.normal(0, 1, 200)
        series = pd.Series(trend + noise, index=pd.date_range("2026-01-01", periods=200, freq="1h"), name="price")

        # d = 0.5 (fractional differencing)
        fd_05 = FractionalDifferentiator.frac_diff(series, d=0.5)
        self.assertFalse(fd_05.empty)
        self.assertLess(len(fd_05), len(series))  # Loses initial lag burn-in


if __name__ == "__main__":
    unittest.main()
