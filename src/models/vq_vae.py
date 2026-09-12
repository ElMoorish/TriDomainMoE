"""
VQ-VAE Market State Tokenizer.

Compresses local microstructure patch sequences into a finite vocabulary of
discrete "market states" (codebook tokens). This enables:

  1. Stable autoregressive rollout — discrete state space prevents latent drift.
  2. Interpretable market states — codebook entries become named regimes.
  3. Better sequence modeling — token IDs can be used with any sequence model.

Architecture:
    PatchEncoder   → z_e (continuous latent)
    VectorQuantizer→ z_q (nearest codebook entry, straight-through gradient)
    PatchDecoder   → x̂  (reconstructed patch for self-supervised training)

Loss:
    L = L_recon + β * L_commitment + γ * L_codebook
      where β=0.25 (default), γ=1.0
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Causal Conv Block
# ---------------------------------------------------------------------------

class CausalConvBlock(nn.Module):
    """1D causal convolution → LayerNorm → GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                              dilation=dilation, padding=0)
        self.pad = pad
        self.norm = nn.GroupNorm(1, out_ch)  # GroupNorm works well with small batches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return F.gelu(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# Patch Encoder
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """
    Encodes a local window of microstructure features into a continuous latent z_e.

    Input:  (batch, patch_len, input_dim)  — e.g. 16 bars × 10 features
    Output: (batch, latent_dim)            — continuous latent representation
    """

    def __init__(self, input_dim: int, latent_dim: int, patch_len: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.patch_len = patch_len

        hidden = latent_dim * 2

        self.conv1 = CausalConvBlock(input_dim, hidden, kernel_size=3, dilation=1)
        self.conv2 = CausalConvBlock(hidden, hidden, kernel_size=3, dilation=2)
        self.conv3 = CausalConvBlock(hidden, latent_dim, kernel_size=3, dilation=4)

        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, patch_len, input_dim)
        Returns:
            z_e: (batch, latent_dim) — terminal latent state
        """
        h = x.transpose(1, 2)  # (B, input_dim, patch_len)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)       # (B, latent_dim, L')
        z_e = h[:, :, -1]      # Take terminal position
        return self.ln(z_e)     # (B, latent_dim)


# ---------------------------------------------------------------------------
# Vector Quantizer
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """
    Vector Quantization with EMA (Exponential Moving Average) codebook updates.

    EMA updates decouple codebook learning from encoder gradient flow, which
    is the standard fix for codebook collapse (van den Oord et al., 2017;
    Razavi et al., 2019 VQ-VAE-2).

    When use_ema=True (default):
      - Codebook entries are updated via exponential moving averages of encoder
        outputs, NOT via gradient descent. This prevents collapse.
      - Only the commitment loss (encoder → codebook) uses gradients.
      - Dead codes (unused for many batches) are reset to random encoder outputs.

    When use_ema=False:
      - Falls back to original straight-through gradient estimator.
      - Prone to codebook collapse on large datasets.
    """

    def __init__(
        self,
        num_embeddings: int = 128,
        embedding_dim: int = 64,
        beta: float = 0.5,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 10,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        if use_ema:
            # EMA statistics — not model parameters, tracked as buffers
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_weight_sum", self.codebook.weight.data.clone())
            self.register_buffer("usage_count", torch.zeros(num_embeddings, dtype=torch.long))

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: (batch, embedding_dim) — continuous encoder output
        Returns:
            z_q:       (batch, embedding_dim) — quantized latent (straight-through)
            token_ids: (batch,) — integer codebook indices
            vq_loss:   scalar — commitment loss (+ codebook loss if not EMA)
        """
        # Pairwise distances to all codebook entries: (B, K)
        dist = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_e @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1)
        )

        token_ids = dist.argmin(dim=1)   # (B,)
        z_q = self.codebook(token_ids)   # (B, D)

        if self.use_ema and self.training:
            # EMA codebook update (no gradient through codebook weights)
            with torch.no_grad():
                onehot = F.one_hot(token_ids, self.num_embeddings).float()  # (B, K)
                n_i = onehot.sum(0)                                          # (K,) — batch usage counts

                # Update EMA cluster sizes
                self.ema_cluster_size.mul_(self.ema_decay).add_(n_i, alpha=1.0 - self.ema_decay)

                # Update EMA weight sums
                dw = onehot.t() @ z_e                                        # (K, D)
                self.ema_weight_sum.mul_(self.ema_decay).add_(dw, alpha=1.0 - self.ema_decay)

                # Laplace-smoothed codebook update
                n_smooth = self.ema_cluster_size + 1e-5
                self.codebook.weight.data.copy_(self.ema_weight_sum / n_smooth.unsqueeze(1))

                # Track usage and reset dead codes
                self.usage_count.add_(n_i.long())
                dead_mask = self.usage_count < self.dead_code_threshold
                n_dead = dead_mask.sum().item()
                if n_dead > 0:
                    # Reset dead codes to random encoder outputs from this batch
                    n_reset = min(n_dead, z_e.shape[0])
                    reset_idx = torch.randperm(z_e.shape[0], device=z_e.device)[:n_reset]
                    dead_indices = dead_mask.nonzero(as_tuple=True)[0][:n_reset]
                    self.codebook.weight.data[dead_indices] = z_e[reset_idx].detach()
                    self.usage_count[dead_indices] = self.dead_code_threshold  # Give them a fresh start

            # With EMA: only commitment loss (encoder commits to nearest code)
            commit_loss = F.mse_loss(z_e, z_q.detach())
            vq_loss = self.beta * commit_loss
        else:
            # Gradient-only path (use_ema=False fallback)
            codebook_loss = F.mse_loss(z_e.detach(), z_q)
            commit_loss = F.mse_loss(z_e, z_q.detach())
            vq_loss = codebook_loss + self.beta * commit_loss

        # Straight-through: copy gradients from z_q to z_e
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, token_ids, vq_loss

    def perplexity(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Codebook perplexity: measures how evenly the codebook is being used.
        Perfect uniform usage over K codes → perplexity = K.
        Low perplexity (e.g. 7) = codebook collapse.
        """
        onehot = F.one_hot(token_ids, num_classes=self.num_embeddings).float()
        avg_probs = onehot.mean(dim=0)
        entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        return entropy.exp()


# ---------------------------------------------------------------------------
# Patch Decoder
# ---------------------------------------------------------------------------

class PatchDecoder(nn.Module):
    """
    Reconstructs the original microstructure patch from the quantized latent z_q.
    Used only during VQ-VAE self-supervised training — not needed at inference time.

    Input:  (batch, latent_dim)
    Output: (batch, patch_len, input_dim) — reconstructed features
    """

    def __init__(self, latent_dim: int, input_dim: int, patch_len: int = 16):
        super().__init__()
        self.patch_len = patch_len
        self.input_dim = input_dim

        hidden = latent_dim * 2

        self.project = nn.Linear(latent_dim, hidden * patch_len)
        self.hidden = hidden

        self.deconv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, input_dim, kernel_size=3, padding=1),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: (batch, latent_dim)
        Returns:
            x_recon: (batch, patch_len, input_dim)
        """
        B = z_q.shape[0]
        h = self.project(z_q)                               # (B, hidden * patch_len)
        h = h.view(B, self.hidden, self.patch_len)          # (B, hidden, patch_len)
        x_recon = self.deconv(h).transpose(1, 2)            # (B, patch_len, input_dim)
        return x_recon


# ---------------------------------------------------------------------------
# Market State Tokenizer (Top-level module)
# ---------------------------------------------------------------------------

class MarketStateTokenizer(nn.Module):
    """
    Full VQ-VAE tokenizer for microstructure patches.

    Usage modes:
      1. Training mode: encode → quantize → decode → compute VQ loss + recon loss.
      2. Inference mode: encode → quantize → return (z_q, token_ids).
         The z_q latent is passed directly to the MicrostructureTechExpert.

    Args:
        input_dim:      Number of input features per bar (e.g. 10 for v3).
        latent_dim:     Codebook embedding dimension (default 64).
        num_embeddings: Codebook vocabulary size K (default 512).
        patch_len:      Number of bars per patch (default 16).
        beta:           Commitment loss coefficient (default 0.25).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        num_embeddings: int = 128,
        patch_len: int = 16,
        beta: float = 0.5,
        use_ema: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.patch_len = patch_len

        self.encoder = PatchEncoder(input_dim, latent_dim, patch_len)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, beta, use_ema=use_ema)
        self.decoder = PatchDecoder(latent_dim, input_dim, patch_len)

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, patch_len, input_dim) — microstructure patch
            return_loss: If True, also compute reconstruction and VQ losses.

        Returns dict with:
            z_q        : (batch, latent_dim) — quantized latent (straight-through)
            token_ids  : (batch,) — discrete codebook index
            vq_loss    : scalar — codebook + commitment loss (only if return_loss=True)
            recon_loss : scalar — MSE reconstruction loss (only if return_loss=True)
            total_loss : scalar — vq_loss + recon_loss (only if return_loss=True)
            perplexity : scalar — codebook perplexity (batch-level)
        """
        # Validate patch length
        if x.shape[1] < self.patch_len:
            # Pad short sequences
            pad_len = self.patch_len - x.shape[1]
            x = F.pad(x, (0, 0, pad_len, 0))
        elif x.shape[1] > self.patch_len:
            x = x[:, -self.patch_len:, :]

        # Encode
        z_e = self.encoder(x)                           # (B, latent_dim)

        # Quantize
        z_q, token_ids, vq_loss = self.quantizer(z_e)  # (B, latent_dim), (B,), scalar

        out = {
            "z_q": z_q,
            "token_ids": token_ids,
            "perplexity": self.quantizer.perplexity(token_ids),
        }

        if return_loss:
            # Decode
            x_recon = self.decoder(z_q)                 # (B, patch_len, input_dim)
            recon_loss = F.mse_loss(x_recon, x)
            out["vq_loss"] = vq_loss
            out["recon_loss"] = recon_loss
            out["total_loss"] = recon_loss + vq_loss

        return out

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-only: encode patch → return (z_q, token_ids).
        No reconstruction or loss computation.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            z_q:       (batch, latent_dim)
            token_ids: (batch,) — discrete market state index
        """
        self.eval()
        if x.shape[1] < self.patch_len:
            pad_len = self.patch_len - x.shape[1]
            x = F.pad(x, (0, 0, pad_len, 0))
        else:
            x = x[:, -self.patch_len:, :]

        z_e = self.encoder(x)
        z_q, token_ids, _ = self.quantizer(z_e)
        return z_q, token_ids

    def get_codebook(self) -> torch.Tensor:
        """Returns the full codebook embedding matrix (K, latent_dim)."""
        return self.quantizer.codebook.weight.detach()
