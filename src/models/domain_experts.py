"""
Domain-Specialized Neural Expert Networks.
Contains three orthogonal expert architectures:
  1. MicrostructureTechExpert (Technical & Order Book Flow)
     - v2 mode: dilated causal conv stack over OHLCV+OFI features
     - v3 mode: VQ-VAE token embeddings fed into conv stack, optionally
               initialized from a pretrained MBM encoder backbone
  2. MacroSSMExpert (Macroeconomic Term Structure & Cross-Asset Recurrence)
  3. FundamentalSentimentExpert (News Sentiment & Narrative Uncertainty)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm_expert import LinearRecurrentBlock


class CausalConv1d(nn.Conv1d):
    """Causal 1D Convolution with left-padding to prevent future lookahead leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, **kwargs):
        pad_len = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=0, **kwargs)
        self.pad_len = pad_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_len > 0:
            x = F.pad(x, (self.pad_len, 0))
        return super().forward(x)


class MicrostructureTechExpert(nn.Module):
    """
    Expert 1: Technical & Microstructure Dynamics.

    Two operating modes:

    v2 (legacy / live trader compatible):
        Tokenizes localized temporal patches of volume bars, OFI, and intraday
        price action via dilated causal convolutions.
        vq_tokenizer=None, mbm_encoder_dim=None.

    v3 (advanced pipeline):
        When vq_tokenizer is provided, pre-quantizes each input patch to a
        discrete market-state token, then combines the VQ latent (z_q) with
        the raw feature projection before the conv-FFN stack.
        If mbm_encoder_dim is set, the input projection layer is sized to
        accept a concatenation of raw features + VQ z_q (total dim = input_dim + latent_dim).

    The VQ token entropy can be used to gate signals dynamically,
    replacing the brittle hard-coded 0.030 threshold.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dilation: int = 2,
        vq_tokenizer: Optional[nn.Module] = None,
        vq_latent_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vq_tokenizer = vq_tokenizer
        self.vq_latent_dim = vq_latent_dim if vq_tokenizer is not None else 0

        # If VQ-VAE is present, input to conv = raw features + z_q concatenated
        effective_input_dim = input_dim + self.vq_latent_dim

        self.conv1 = CausalConv1d(effective_input_dim, hidden_dim, kernel_size=3, dilation=1)
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, output_dim)
        # Near-zero initialization
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x_tech: torch.Tensor,
        return_token_ids: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_tech: Tensor of shape (batch, seq_len, input_dim)
            return_token_ids: If True, also returns VQ token IDs for entropy gating.
        Returns:
            Tuple of (pred, h_terminal) or (pred, h_terminal, token_ids) when return_token_ids=True
        """
        # VQ-VAE tokenization path (v3)
        if self.vq_tokenizer is not None:
            with torch.no_grad() if not self.training else torch.enable_grad():
                z_q, token_ids = self.vq_tokenizer.tokenize(x_tech)
            # Expand z_q along the sequence axis and concatenate
            z_q_expanded = z_q.unsqueeze(1).expand(-1, x_tech.shape[1], -1)  # (B, T, latent_dim)
            x_combined = torch.cat([x_tech, z_q_expanded], dim=-1)          # (B, T, input_dim + latent_dim)
        else:
            x_combined = x_tech
            token_ids = None

        # Conv1d operates over (batch, in_channels, seq_len)
        x_perm = x_combined.transpose(1, 2)
        h = F.gelu(self.conv1(x_perm))
        h = F.gelu(self.conv2(h)).transpose(1, 2)  # (batch, seq_len, hidden_dim)

        h = self.ln1(h)
        h = self.ln2(h + self.ffn(h))

        h_terminal = h[:, -1, :]  # (batch, hidden_dim)
        pred = self.head(h_terminal)

        if return_token_ids and token_ids is not None:
            return pred, h_terminal, token_ids
        return pred, h_terminal


class MacroSSMExpert(nn.Module):
    """
    Expert 2: Macroeconomic & Term Structure Dynamics.
    Utilizes linear recurrent state-space projections to capture cross-asset multi-day cycles.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_embed = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            LinearRecurrentBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.head = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_macro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_macro: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tuple of (pred, h_terminal)
        """
        h = self.input_embed(x_macro)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h))

        h_terminal = h[:, -1, :]
        pred = self.head(h_terminal)
        return pred, h_terminal


class FundamentalSentimentExpert(nn.Module):
    """
    Expert 3: Fundamental News & Narrative Sentiment.
    Deep gated residual MLP processing news sentiment polarity, uncertainty, and macro topic exposures.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embed = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_fund: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_fund: Tensor of shape (batch, input_dim) or (batch, seq_len, input_dim)
        Returns:
            Tuple of (pred, h_terminal)
        """
        if x_fund.dim() == 3:
            x_fund = x_fund[:, -1, :]  # Latest fundamental snapshot

        h = F.gelu(self.embed(x_fund))
        g = torch.sigmoid(self.gate(h))
        h_gated = h * g

        h_out = self.norm(h + self.block(h_gated))
        pred = self.head(h_out)
        return pred, h_out
