"""Unit tests for MacroSentimentEngine."""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np

from src.features.sentiment_engine import MacroSentimentEngine


class TestSentimentEngine(unittest.TestCase):
    def test_sentiment_scoring_and_decay(self):
        engine = MacroSentimentEngine(decay_halflife_hours=1.0)

        # Bullish article
        articles = [
            {
                "title": "Tech stocks rally as Fed signals rate cut and economic expansion",
                "summary": "Massive surge in Nasdaq earnings boosts market optimism across indices.",
                "published_at": datetime.now(timezone.utc),
            }
        ]

        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        z0 = engine.update(articles, current_time=t0)

        # Net sentiment should be positive
        self.assertGreater(z0[0], 0.0)
        # Monetary policy topic should be detected
        self.assertGreater(z0[2], 0.0)

        # Simulate time passing 1 hour (1 half-life decay)
        t1 = t0 + timedelta(hours=1)
        z1 = engine.update([], current_time=t1)

        # Sentiment should decay approximately by half
        self.assertAlmostEqual(z1[0], z0[0] * 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
