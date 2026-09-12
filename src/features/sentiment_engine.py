"""
Sentiment and Macro Narrative Understanding Engine.
Quantifies sentiment polarity, narrative uncertainty, and macro topic exposure with exponential decay.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
import math
import re
import numpy as np


class MacroSentimentEngine:
    """
    Transforms unstructured financial headlines and summaries into continuous
    regime signals [sentiment, uncertainty, topic_weights] with time decay.
    """

    # Domain-specific financial sentiment lexicons (Loughran-McDonald inspired)
    BULLISH_TERMS = {
        "surge", "rally", "gain", "climb", "jump", "soar", "record", "optimism",
        "beat", "boost", "bullish", "expansion", "growth", "high", "upgrade",
        "recovery", "outperform", "dividend", "revenue", "rate cut", "dovish",
    }
    
    BEARISH_TERMS = {
        "drop", "plunge", "fall", "slump", "slide", "sink", "decline", "tumble",
        "crash", "recession", "loss", "warning", "pessimism", "bearish", "downgrade",
        "inflation", "hike", "hawkish", "debt", "default", "layoff", "selloff",
        "conflict", "war", "sanction", "crisis", "slowdown", "tariffs",
    }

    UNCERTAINTY_TERMS = {
        "uncertain", "volatility", "unclear", "risk", "fear", "hesitant", "pause",
        "caution", "mixed", "speculation", "pending", "turbulent", "unprecedented",
        "headwind", "threat", "fragile", "contagion", "anxiety",
    }

    TOPIC_KEYWORDS = {
        "monetary_policy": {"fed", "powell", "rate", "treasury", "yield", "inflation", "cpi", "fomc", "central bank"},
        "geopolitics": {"war", "sanction", "tariff", "china", "russia", "middle east", "opec", "election", "conflict"},
        "corporate_earnings": {"earnings", "revenue", "profit", "guidance", "ceo", "shares", "buyback", "margin", "tech"},
        "commodities": {"oil", "crude", "brent", "gold", "gas", "energy", "barrel", "opec", "inventory", "metal"},
    }

    def __init__(self, decay_halflife_hours: float = 12.0):
        self.decay_halflife_seconds = decay_halflife_hours * 3600.0
        self.decay_lambda = math.log(2) / self.decay_halflife_seconds
        
        # State variables
        self._last_update_ts: Optional[datetime] = None
        self._current_sentiment: float = 0.0
        self._current_uncertainty: float = 0.0
        self._current_topics: Dict[str, float] = {k: 0.0 for k in self.TOPIC_KEYWORDS}

    def _score_text(self, text: str) -> Tuple[float, float, Dict[str, float]]:
        """Score a single text block for sentiment, uncertainty, and topics."""
        tokens = re.findall(r"\b[a-z\-]+\b", text.lower())
        total_tokens = max(len(tokens), 1)

        bull_count = sum(1 for w in tokens if w in self.BULLISH_TERMS)
        bear_count = sum(1 for w in tokens if w in self.BEARISH_TERMS)
        unc_count = sum(1 for w in tokens if w in self.UNCERTAINTY_TERMS)

        # Net sentiment in [-1.0, 1.0]
        denom = bull_count + bear_count
        sentiment = (bull_count - bear_count) / denom if denom > 0 else 0.0

        # Uncertainty density in [0.0, 1.0]
        uncertainty = min(1.0, (unc_count / total_tokens) * 10.0)

        # Topic detection
        topic_scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            matches = sum(1 for w in tokens if w in keywords)
            topic_scores[topic] = min(1.0, matches / 3.0)

        return sentiment, uncertainty, topic_scores

    def update(self, articles: List[Dict[str, Any]], current_time: Optional[datetime] = None) -> np.ndarray:
        """
        Incorporate new articles and apply exponential time decay.
        Returns the updated regime feature vector z_sentiment:
        [sentiment, uncertainty, topic_monetary, topic_geopolitics, topic_corporate, topic_commodities]
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Apply decay from last update
        if self._last_update_ts is not None:
            dt = max(0.0, (current_time - self._last_update_ts).total_seconds())
            decay_factor = math.exp(-self.decay_lambda * dt)
            self._current_sentiment *= decay_factor
            self._current_uncertainty *= decay_factor
            for k in self._current_topics:
                self._current_topics[k] *= decay_factor

        self._last_update_ts = current_time

        if not articles:
            return self.get_regime_vector()

        # Aggregate new articles
        batch_sentiment = []
        batch_uncertainty = []
        batch_topics = {k: [] for k in self.TOPIC_KEYWORDS}

        for art in articles:
            text = f"{art.get('title', '')}. {art.get('summary', '')}"
            s, u, t_dict = self._score_text(text)
            batch_sentiment.append(s)
            batch_uncertainty.append(u)
            for k, val in t_dict.items():
                batch_topics[k].append(val)

        # Blend new innovations into decaying state (weighting: 0.3 new, 0.7 prior state)
        if batch_sentiment:
            new_s = float(np.mean(batch_sentiment))
            new_u = float(np.mean(batch_uncertainty))
            self._current_sentiment = float(np.clip(0.7 * self._current_sentiment + 0.3 * new_s, -1.0, 1.0))
            self._current_uncertainty = float(np.clip(0.7 * self._current_uncertainty + 0.3 * new_u, 0.0, 1.0))

            for k in self.TOPIC_KEYWORDS:
                new_top = float(np.mean(batch_topics[k]))
                self._current_topics[k] = float(np.clip(0.7 * self._current_topics[k] + 0.3 * new_top, 0.0, 1.0))

        return self.get_regime_vector()

    def get_regime_vector(self) -> np.ndarray:
        """
        Returns structured 6-dimensional array for router regime input z_t:
        [sentiment, uncertainty, monetary_policy, geopolitics, corporate_earnings, commodities]
        """
        vec = [
            self._current_sentiment,
            self._current_uncertainty,
            self._current_topics["monetary_policy"],
            self._current_topics["geopolitics"],
            self._current_topics["corporate_earnings"],
            self._current_topics["commodities"],
        ]
        return np.array(vec, dtype=np.float32)
