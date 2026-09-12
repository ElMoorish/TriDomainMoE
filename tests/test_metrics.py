"""
Unit Tests for Statistical Evidence Metrics.
"""

import unittest
import pandas as pd
import numpy as np
from src.backtest.metrics import StatisticalEvidenceMetrics


class TestStatisticalMetrics(unittest.TestCase):
    def test_streaks(self):
        # 3 wins, 2 losses, 4 wins, 1 loss
        pnls = [10.0, 20.0, 15.0, -5.0, -10.0, 30.0, 25.0, 40.0, 10.0, -8.0]
        streaks = StatisticalEvidenceMetrics.compute_streaks(pnls)
        self.assertEqual(streaks["max_consecutive_wins"], 4)
        self.assertEqual(streaks["max_consecutive_losses"], 2)

    def test_runs_z_score(self):
        # Balanced alternating sequence
        pnls = [10.0, -10.0] * 50
        runs_res = StatisticalEvidenceMetrics.compute_runs_z_score(pnls)
        self.assertIn("z_score", runs_res)
        self.assertIn("p_value", runs_res)
        self.assertGreater(runs_res["total_runs"], 50)

    def test_drawdowns(self):
        eq = pd.Series([10000, 10500, 10200, 9800, 10600, 10400])
        dd = StatisticalEvidenceMetrics.compute_drawdowns(eq)
        # Peak was 10500, trough was 9800 -> 700 / 10500 = 6.6667%
        self.assertAlmostEqual(dd["max_drawdown_cash"], 700.0, places=1)
        self.assertAlmostEqual(dd["max_drawdown_pct"], 6.6667, places=2)


if __name__ == "__main__":
    unittest.main()
