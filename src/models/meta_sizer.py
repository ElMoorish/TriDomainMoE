"""
Meta-Labeling Conviction Bet Sizer.
Decouples directional classification from execution sizing via calibrated sigmoids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaSizer(nn.Module):
    """
    Secondary model predicting the probability p_t of hitting the profit-taking
    threshold before the stop-loss, mapped to continuous sizing coefficient s_t in [0, 1].
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, sigma_cal: float = 0.25):
        super().__init__()
        self.sigma_cal = sigma_cal
        # Inputs: asset features + directional prediction signal
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, directional_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch_size, feature_dim).
            directional_pred: Primary directional forecast (batch_size, 1).
        Returns:
            Continuous trade sizing s_t in [0.0, 1.0] of shape (batch_size, 1).
        """
        if features.dim() == 3:
            features = features[:, -1, :]

        x_in = torch.cat([features, directional_pred], dim=-1)
        raw_logit = self.net(x_in)
        p_t = torch.sigmoid(raw_logit)

        # Calibrated transformation: s_t = max(0, 2 * (sigmoid((p - 0.5) / sigma_cal) - 0.5))
        z = (p_t - 0.5) / self.sigma_cal
        cal_sig = torch.sigmoid(z)
        s_t = torch.clamp(2.0 * (cal_sig - 0.5), min=0.0, max=1.0)
        return s_t
