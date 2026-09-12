"""
Episodic Memory Buffer and Retrospective Reflection Engine.
Maintains trade histories and adjusts router sensitivity when realized losses exceed 3-sigma thresholds.
"""

from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np


class EpisodicMemoryBuffer:
    """
    Episodic memory store recording executed trades, slippage, realized returns,
    and routing allocations for post-settlement reflection.
    """

    def __init__(self, max_records: int = 5000, anomaly_sigma: float = 3.0):
        self.max_records = max_records
        self.anomaly_sigma = anomaly_sigma
        self.buffer: deque = deque(maxlen=max_records)

    def record_episode(
        self,
        timestamp: Any,
        symbol: str,
        predicted_y: float,
        realized_return: float,
        transaction_cost: float,
        routing_weights: np.ndarray,
        regime_vector: np.ndarray,
    ) -> None:
        """Append a completed execution record."""
        forecast_error = realized_return - predicted_y
        self.buffer.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "predicted_y": predicted_y,
            "realized_return": realized_return,
            "net_return": realized_return - transaction_cost,
            "forecast_error": forecast_error,
            "routing_weights": routing_weights,
            "regime_vector": regime_vector,
        })

    def analyze_reflection(self) -> Dict[str, Any]:
        """
        Scan episodic memory for 3-sigma tail loss anomalies.
        Returns reflection analysis and recommended router sensitivity adjustment.
        """
        if len(self.buffer) < 30:
            return {"status": "insufficient_history", "anomaly_detected": False}

        errors = np.array([r["forecast_error"] for r in self.buffer])
        mean_err = np.mean(errors)
        std_err = np.std(errors)

        latest_record = self.buffer[-1]
        latest_err = latest_record["forecast_error"]

        # 3-sigma anomaly condition
        z_score = abs(latest_err - mean_err) / (std_err + 1e-6)
        is_anomaly = bool(z_score >= self.anomaly_sigma)

        return {
            "status": "ok",
            "sample_size": len(self.buffer),
            "mean_forecast_error": float(mean_err),
            "std_forecast_error": float(std_err),
            "latest_z_score": float(z_score),
            "anomaly_detected": is_anomaly,
            "recommended_action": "dampen_overconfident_expert" if is_anomaly else "maintain_policy",
        }
