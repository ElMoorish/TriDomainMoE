"""Unit tests for OrderFlowImbalance."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.features.ofi import OrderFlowImbalance


class TestOrderFlowImbalance(unittest.TestCase):
    def test_ofi_exact_mathematics(self):
        # Case 1: Bid price rises (P_t^b > P_{t-1}^b) -> +q_t^b
        # Ask price rises (P_t^a > P_{t-1}^a) -> +q_{t-1}^a
        bid_prices = np.array([100.0, 101.0])
        bid_sizes = np.array([10.0, 15.0])
        ask_prices = np.array([100.5, 101.5])
        ask_sizes = np.array([8.0, 12.0])

        ofi = OrderFlowImbalance.compute_tick_ofi(bid_prices, bid_sizes, ask_prices, ask_sizes)
        
        # delta_bid = +q_t^b = 15.0 (since 101.0 >= 100.0)
        # delta_ask = +q_{t-1}^a = 8.0 (since 101.5 >= 100.5)
        # ofi[1] = 15.0 - (-8.0) = 23.0
        self.assertEqual(len(ofi), 2)
        self.assertEqual(ofi[0], 0.0)
        self.assertAlmostEqual(ofi[1], 23.0)

    def test_ofi_unchanged_prices(self):
        # Case 2: Prices unchanged -> delta_bid = q_t - q_{t-1}, delta_ask = -(q_t - q_{t-1})
        bid_prices = np.array([100.0, 100.0])
        bid_sizes = np.array([10.0, 14.0])  # +4 bid depth
        ask_prices = np.array([100.5, 100.5])
        ask_sizes = np.array([8.0, 6.0])    # -2 ask depth (ask consumed)

        ofi = OrderFlowImbalance.compute_tick_ofi(bid_prices, bid_sizes, ask_prices, ask_sizes)
        
        # delta_bid = 14.0 - 10.0 = 4.0
        # delta_ask = 6.0 - 8.0 = -2.0
        # ofi[1] = 4.0 - (-2.0) = 6.0 (bullish pressure)
        self.assertAlmostEqual(ofi[1], 6.0)


if __name__ == "__main__":
    unittest.main()
