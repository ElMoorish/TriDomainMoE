"""
Selective State-Space / Linear Sequence Expert Network.
Specialized for long-horizon lookback contexts (tau_long) with O(L) linear sequence complexity.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRecurrentBlock(nn.Module):
    """
    Input-dependent linear recurrent projection layer inspired by Mamba state spaces.
    Computes recurrence h_t = a_t * h_{t-1} + b_t * x_t in linear time.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Input-dependent decay gate (alpha) and input projection (beta)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self._init_hippo_priors()

    def _init_hippo_priors(self):
        """Initializes decay gate logits with a logarithmic timescale spectrum (0.85 to 0.995)."""
        with torch.no_grad():
            decay_priors = torch.linspace(0.85, 0.995, self.hidden_dim)
            alpha_init = torch.logit(decay_priors)
            self.gate_proj.bias.data[:self.hidden_dim].copy_(alpha_init)
            self.gate_proj.bias.data[self.hidden_dim:].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, hidden_dim)
        Returns:
            Tensor of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, dim = x.shape
        gates = self.gate_proj(x)
        alpha, beta = gates.chunk(2, dim=-1)

        # Decay parameter in (0, 1)
        decay = torch.sigmoid(alpha)
        # Projected input
        cand = torch.tanh(beta)

        # Recurrent scan over sequence
        h_state = torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            h_state = decay[:, t, :] * h_state + (1.0 - decay[:, t, :]) * cand[:, t, :]
            outputs.append(h_state)

        out_stacked = torch.stack(outputs, dim=1)  # (batch, seq_len, dim)
        return self.out_proj(out_stacked)


class SSMExpert(nn.Module):
    """
    Long-lookback expert utilizing linear recurrent projections to capture
    extended macroeconomic momentum and secular cycles.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.input_embed = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            LinearRecurrentBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.head = nn.Linear(hidden_dim, output_dim)
        # Near-zero initialization for residual MoE stability
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            Tuple of:
              - pred: Prediction tensor (batch_size, output_dim)
              - h_terminal: Terminal hidden state (batch_size, hidden_dim)
        """
        batch_size, seq_l, in_dim = x.shape
        if seq_l > self.seq_len:
            x = x[:, -self.seq_len :, :]
        elif seq_l < self.seq_len:
            pad = torch.zeros(batch_size, self.seq_len - seq_l, in_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)

        h = self.input_embed(x)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h))

        h_terminal = h[:, -1, :]
        pred = self.head(h_terminal)
        return pred, h_terminal
