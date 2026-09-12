"""
Shannon Gating Entropy Monitor and Fallback Safeguard.
Protects against router collapse during severe market dislocations by falling back to equal weighting.
"""

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class GatingEntropyGuard:
    """
    Monitors Shannon entropy of the soft routing distribution.
    If entropy falls below critical threshold, activates defensive equal-weighted fallback.
    """

    def __init__(self, critical_entropy: float = 0.35):
        self.critical_entropy = critical_entropy

    def evaluate_entropy(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            weights: Tensor of shape (batch_size, num_experts).
        Returns:
            Tuple of:
              - safe_weights: Modified weights with equal fallback applied if collapsed.
              - collapse_mask: Boolean tensor indicating collapsed states.
        """
        eps = 1e-8
        entropy = -torch.sum(weights * torch.log(weights + eps), dim=-1)  # (batch_size,)
        collapse_mask = entropy < self.critical_entropy

        num_experts = weights.shape[-1]
        equal_weights = torch.full_like(weights, 1.0 / float(num_experts))

        # Where collapsed, replace with equal weights
        safe_weights = torch.where(collapse_mask.unsqueeze(-1), equal_weights, weights)
        return safe_weights, collapse_mask
