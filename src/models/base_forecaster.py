"""
Base Forecaster Network (f_base).
Models unconditional asset drift and secular trend components to anchor optimization stability.
"""

import torch
import torch.nn as nn


class BaseForecaster(nn.Module):
    """
    Primary baseline predictor modeling secular trend and unconditional drift.
    Ensures that residual experts only learn state-dependent deviations.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim).
        Returns:
            Tensor of shape (batch_size, output_dim).
        """
        if x.dim() == 3:
            # Flatten or take terminal representation
            x = x[:, -1, :]
        return self.net(x)
