"""
Patched Feed-Forward / Convolutional Expert Network.
Specialized for short-horizon lookback contexts (tau_short) and microstructure order flow.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchExpert(nn.Module):
    """
    Tokenizes localized temporal patches into compact representations.
    Extracts high-frequency microstructure dynamics while dampening Gaussian noise.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 16,
        patch_len: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.hidden_dim = hidden_dim
        self.num_patches = max(1, seq_len // patch_len)

        # Patch projection layer: maps (patch_len * input_dim) to hidden_dim
        self.patch_embed = nn.Linear(patch_len * input_dim, hidden_dim)

        # Inter-patch processing blocks
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Output head
        self.head = nn.Linear(hidden_dim, output_dim)
        # Near-zero initialization for residual MoE stability
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tuple of:
              - pred: Prediction tensor (batch_size, output_dim)
              - h_terminal: Terminal hidden representation (batch_size, hidden_dim) for CAW
        """
        batch_size, seq_l, in_dim = x.shape

        # Crop or pad to seq_len
        if seq_l > self.seq_len:
            x = x[:, -self.seq_len :, :]
        elif seq_l < self.seq_len:
            pad = torch.zeros(batch_size, self.seq_len - seq_l, in_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)

        # Shape into patches: (batch, num_patches, patch_len * in_dim)
        x_patched = x.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        # (batch_size, num_patches, in_dim, patch_len) -> permute & flatten
        x_patched = x_patched.permute(0, 1, 3, 2).contiguous().view(batch_size, self.num_patches, -1)

        h = self.patch_embed(x_patched)  # (batch, num_patches, hidden_dim)
        
        # Conv block across patch dimension
        h_conv = h.permute(0, 2, 1)  # (batch, hidden_dim, num_patches)
        h_conv = F.gelu(self.conv1(h_conv)).permute(0, 2, 1)
        h = self.ln1(h + h_conv)

        # FFN block
        h = self.ln2(h + self.ffn(h))

        # Terminal hidden state (latest patch token)
        h_terminal = h[:, -1, :]
        pred = self.head(h_terminal)

        return pred, h_terminal
