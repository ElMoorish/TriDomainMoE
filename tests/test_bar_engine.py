"""
Unit Tests for IntraBarExecutionEngine.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtest.bar_engine import IntraBarExecutionEngine


class TestBarEngine(unittest.TestCase):
    def setUp(self):
        self.engine = IntraBarExecutionEngine(
            symbol="BTCUSD.x",
            initial_balance=10000.0,
            risk_per_trade_pct=0.0010,
        )

    def test_intra_bar_tp_hit(self):
        base_time = datetime(2026, 6, 1, 0, 0, 0)
        times = [base_time + timedelta(minutes=i) for i in range(100)]
        
        # 100 flat bars, then at bar 60 a big green bar hits TP
        opens = np.full(100, 70000.0)
        highs = np.full(100, 70100.0)
        lows = np.full(100, 69900.0)
        closes = np.full(100, 70000.0)
        spreads = np.full(100, 6500.0)  # 65 cash spread

        # At bar 60, price pumps to 72000
        highs[60] = 72000.0
        closes[60] = 71800.0

        bars_df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": 100.0,
            "spread": spreads,
        }, index=pd.DatetimeIndex(times))

        signals_df = pd.DataFrame({
            "direction": ["FLAT"] * 100,
            "conviction": [1.0] * 100,
            "sl_dist": [500.0] * 100,
            "tp_dist": [1000.0] * 100,
        }, index=pd.DatetimeIndex(times))

        # Signal BUY at bar 55
        signals_df.loc[times[55], "direction"] = "BUY"

        trades_df, equity_series, stats = self.engine.run_bar_simulation(bars_df, signals_df)
        self.assertFalse(trades_df.empty)
        trade = trades_df.iloc[0]
        self.assertEqual(trade["direction"], "BUY")
        self.assertEqual(trade["exit_reason"], "TAKE_PROFIT")
        self.assertGreater(trade["pnl_cash"], 0.0)


if __name__ == "__main__":
    unittest.main()
