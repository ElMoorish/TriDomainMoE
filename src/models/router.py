"""
Continuous Softmax Router with Correlation-Aware Weighting (CAW).
Dynamically allocates capital across specialized experts while penalizing redundant representations.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAWRouter(nn.Module):
    """
    Regime-gated continuous softmax router with cosine repulsion regularization (CAW).
    Prevents router collapse and forces functional diversity across nested lookback experts.
    """

    def __init__(
        self,
        regime_dim: int,
        num_experts: int,
        hidden_dim: int = 32,
        lambda_c: float = 0.5,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.regime_dim = regime_dim
        self.num_experts = num_experts
        self.lambda_c = lambda_c
        self.noise_std = noise_std

        # Gating parameterization: maps z_t -> raw gating logits
        self.gate = nn.Sequential(
            nn.Linear(regime_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self,
        z: torch.Tensor,
        expert_hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Macro regime vector of shape (batch_size, regime_dim).
            expert_hidden_states: List of E tensors, each of shape (batch_size, h_dim).
        Returns:
            Tuple of:
              - w_tilde: Final CAW routing weights (batch_size, num_experts)
              - g_raw: Continuous softmax routing without CAW (batch_size, num_experts)
              - entropy: Shannon entropy of routing distribution (batch_size,)
              - noisy_g: Exploration-injected routing distribution for balance loss (batch_size, num_experts)
        """
        logits = self.gate(z)  # (batch_size, num_experts)

        # Base continuous soft routing
        g_raw = F.softmax(logits, dim=-1)

        # Inject exploration noise during training for auxiliary balance regularization
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            noisy_g = F.softmax(logits + noise, dim=-1)
        else:
            noisy_g = g_raw

        # Shannon entropy: H(g) = - sum(g * ln(g))
        eps = 1e-8
        entropy = -torch.sum(g_raw * torch.log(g_raw + eps), dim=-1)

        # Correlation-Aware Weighting (CAW)
        if expert_hidden_states is not None and len(expert_hidden_states) == self.num_experts and self.lambda_c > 0:
            # Stack expert hidden vectors: (batch_size, E, hidden_dim)
            h_stack = torch.stack(expert_hidden_states, dim=1)
            # Normalize along hidden_dim for cosine similarity
            h_norm = F.normalize(h_stack, p=2, dim=-1)
            # Pairwise cosine similarity matrix: (batch_size, E, E)
            cos_sim = torch.bmm(h_norm, h_norm.transpose(1, 2))

            # Mask out self-similarity (diagonal)
            eye_mask = torch.eye(self.num_experts, device=z.device).unsqueeze(0)  # (1, E, E)
            cos_sim_off_diag = cos_sim * (1.0 - eye_mask)

            # Sum of cosine similarities against all other experts: (batch_size, E)
            cos_sum = cos_sim_off_diag.sum(dim=-1)

            # Repulsion term: exp(- lambda_c * sum_{j != i} cos(h_i, h_j))
            repulsion = torch.exp(-self.lambda_c * cos_sum)

            # Adjusted weights
            unnorm_weights = g_raw * repulsion
            w_tilde = unnorm_weights / (unnorm_weights.sum(dim=-1, keepdim=True) + eps)
        else:
            w_tilde = g_raw

        return w_tilde, g_raw, entropy, noisy_g
