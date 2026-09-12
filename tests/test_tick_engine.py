"""
Unit Tests for EventDrivenTickEngine.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtest.tick_engine import EventDrivenTickEngine


class TestTickEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EventDrivenTickEngine(
            symbol="BTCUSD.x",
            initial_balance=10000.0,
            risk_per_trade_pct=0.0010,  # 0.10%
        )

    def test_lot_calculation(self):
        # Equity = $10,000, Risk = 0.10% ($10.00)
        # SL distance = $500 -> Lots = 10 / 500 = 0.02 lots
        lots = self.engine.calculate_lots(equity=10000.0, sl_distance=500.0, conviction=1.0)
        self.assertEqual(lots, 0.02)

    def test_execution_tp_hit(self):
        # Generate synthetic tick sequence with a rising price hitting TP
        base_time = datetime(2026, 9, 1, 12, 0, 0)
        times = [base_time + timedelta(seconds=i) for i in range(250)]
        
        # Prices starting at 70000 and rising to 71000
        bids = np.linspace(70000.0, 71000.0, 250)
        asks = bids + 50.0  # 50 spread

        df = pd.DataFrame({
            "bid": bids,
            "ask": asks,
            "volume": np.ones(250),
        }, index=pd.DatetimeIndex(times))

        # Signal generator that triggers BUY at index 200
        def signal_gen(ticks_df, idx):
            if idx == 200:
                return {
                    "direction": "BUY",
                    "conviction": 1.0,
                    "sl_dist": 100.0,
                    "tp_dist": 100.0,
                }
            return None

        trades_df, equity_series, stats = self.engine.run_tick_simulation(
            ticks_df=df,
            signals_generator=signal_gen,
            bar_interval_ticks=10,
        )

        self.assertFalse(trades_df.empty)
        self.assertEqual(len(trades_df), 1)
        trade = trades_df.iloc[0]
        self.assertEqual(trade["direction"], "BUY")
        self.assertEqual(trade["exit_reason"], "TAKE_PROFIT")
        self.assertGreater(trade["pnl_cash"], 0.0)


if __name__ == "__main__":
    unittest.main()
