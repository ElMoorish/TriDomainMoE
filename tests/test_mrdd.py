"""Unit tests for MultiResolutionDriftDetector."""

import unittest
import numpy as np

from src.surveillance.mrdd import MultiResolutionDriftDetector, WaveletFilterBank


class TestMRDD(unittest.TestCase):
    def test_wavelet_decomposition(self):
        signal = np.sin(np.linspace(0, 10, 64))
        coeffs = WaveletFilterBank.dwt_decomposition(signal, levels=3)
        self.assertEqual(len(coeffs), 3)
        self.assertEqual(len(coeffs[0]), 32)
        self.assertEqual(len(coeffs[1]), 16)
        self.assertEqual(len(coeffs[2]), 8)

    def test_mrdd_drift_detection(self):
        np.random.seed(42)
        ref_returns = np.random.normal(0, 0.01, 200)
        # Shifted distribution with high volatility shock
        test_returns = np.random.normal(0, 0.05, 200)

        detector = MultiResolutionDriftDetector(wavelet_levels=3, energy_threshold=0.5)
        detector.set_reference_baseline(ref_returns)
        res = detector.evaluate_wavelet_drift(test_returns)

        self.assertTrue(res["wavelet_drift_detected"])
        self.assertGreater(res["max_divergence"], 0.5)

    def test_page_hinkley_ic_decay(self):
        detector = MultiResolutionDriftDetector(ph_delta=0.01, ph_threshold=0.05)
        # Sequence of decaying IC values
        decaying_ics = [0.15, 0.12, 0.10, 0.05, 0.01, -0.02, -0.05, -0.10]

        alarm_triggered = False
        for ic in decaying_ics:
            alarm, div = detector.update_page_hinkley(ic)
            if alarm:
                alarm_triggered = True
                break

        self.assertTrue(alarm_triggered)


if __name__ == "__main__":
    unittest.main()
