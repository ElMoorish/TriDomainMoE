"""
Masked-Bar Modeling (MBM) — BERT-style Self-Supervised Pretraining.

Learns a universal "market representation" from raw microstructure bar sequences
WITHOUT any labels. The pretrained encoder is then transferred to the
MicrostructureTechExpert as its backbone.

Core idea (analogous to BERT's masked token prediction):
  - Take a sequence of microstructure bars (normalized_ofi, kyle_lambda, vpin, rv_of_rv, ...)
  - Randomly mask 15-20% of bars by replacing with a learned [MASK] embedding
  - Train a Transformer encoder to reconstruct the original features at masked positions
  - Loss is computed ONLY on masked positions (like BERT's MLM loss)

After pretraining:
  - The encoder has learned "what market structure looks like" across bar sequences
  - When fine-tuned on Triple-Barrier labels, it starts from a strong representation
    rather than learning structure and targets simultaneously (sample-efficient)
"""

from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Masked-Bar Dataset
# ---------------------------------------------------------------------------

class MaskedBarDataset(Dataset):
    """
    Prepares sequences with randomly masked bars for MBM pretraining.

    Each sample consists of:
      - masked_seq:   (seq_len, input_dim) — sequence with some bars replaced by [MASK]
      - original_seq: (seq_len, input_dim) — original unmasked sequence
      - mask_idx:     (seq_len,) bool — True at masked positions

    Args:
        bar_array:    np.ndarray of shape (N, input_dim) — full bar feature array.
        seq_len:      Context length per sample.
        mask_rate:    Fraction of bars to mask (default 0.15).
        stride:       Step between windows (default 1, use larger for speed).
    """

    def __init__(
        self,
        bar_array: np.ndarray,
        seq_len: int = 64,
        mask_rate: float = 0.15,
        stride: int = 8,
    ):
        self.seq_len = seq_len
        self.mask_rate = mask_rate
        self.input_dim = bar_array.shape[1]
        self.data = torch.tensor(bar_array, dtype=torch.float32)

        # Build window start indices
        self.indices = list(range(0, len(bar_array) - seq_len, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        original_seq = self.data[start: start + self.seq_len].clone()

        # Random mask: 15% of positions
        mask_idx = torch.zeros(self.seq_len, dtype=torch.bool)
        n_mask = max(1, int(self.seq_len * self.mask_rate))
        mask_positions = torch.randperm(self.seq_len)[:n_mask]
        mask_idx[mask_positions] = True

        # Masked sequence: replace masked positions with zeros (learned mask token added in model)
        masked_seq = original_seq.clone()
        masked_seq[mask_idx] = 0.0

        return masked_seq, original_seq, mask_idx


# ---------------------------------------------------------------------------
# MBM Transformer Encoder
# ---------------------------------------------------------------------------

class MBMEncoder(nn.Module):
    """
    Transformer encoder for Masked-Bar Modeling pretraining.

    Architecture:
      - Linear input projection: input_dim → d_model
      - Learned [MASK] token embedding (added at masked positions)
      - Sinusoidal + learned positional encoding
      - N transformer encoder layers (default 4 layers, 4 heads)
      - Linear reconstruction head: d_model → input_dim (for pretraining)

    At fine-tuning time, the reconstruction head is DISCARDED and replaced
    with the domain-expert prediction head.

    Args:
        input_dim:   Number of input features per bar.
        d_model:     Internal transformer dimension (default 128).
        n_heads:     Number of attention heads (default 4).
        n_layers:    Number of transformer encoder layers (default 4).
        seq_len:     Maximum sequence length (default 64).
        dropout:     Dropout probability (default 0.1).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learned [MASK] token embedding (added to masked positions)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (learned)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (batch, seq, dim)
            norm_first=True,   # Pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Reconstruction head (used during pretraining only)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for projection layers."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for module in self.recon_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor, mask_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through projection + positional encoding + transformer.
        At masked positions, replaces projected input with learned mask token.

        Args:
            x:        (batch, seq_len, input_dim) — input bar sequence
            mask_idx: (batch, seq_len) bool — True at masked positions (optional)

        Returns:
            h: (batch, seq_len, d_model) — contextual representation
        """
        B, T, _ = x.shape
        h = self.input_proj(x)      # (B, T, d_model)

        # Add mask token at masked positions
        if mask_idx is not None:
            mask_expanded = mask_idx.unsqueeze(-1).expand_as(h)  # (B, T, d_model)
            mask_tokens = self.mask_token.expand(B, T, -1)
            h = torch.where(mask_expanded, mask_tokens, h)

        # Add positional encoding (truncate/expand if needed)
        pos = self.pos_embed[:, :T, :]
        h = h + pos

        # Transformer encoding
        h = self.transformer(h)     # (B, T, d_model)
        return h

    def forward(
        self,
        masked_seq: torch.Tensor,
        mask_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full pretraining forward pass: encode → reconstruct at masked positions.

        Args:
            masked_seq: (batch, seq_len, input_dim)
            mask_idx:   (batch, seq_len) bool

        Returns:
            recon:     (batch, seq_len, input_dim) — reconstructed sequence
            h_encoded: (batch, seq_len, d_model) — full contextual representations
        """
        h = self.encode(masked_seq, mask_idx)
        recon = self.recon_head(h)
        return recon, h

    def get_terminal_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-time: returns the terminal (last position) hidden state.
        Used as the backbone for MicrostructureTechExpert at fine-tune time.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            h_terminal: (batch, d_model)
        """
        h = self.encode(x, mask_idx=None)
        return h[:, -1, :]   # (B, d_model)


# ---------------------------------------------------------------------------
# MBM Pretrainer
# ---------------------------------------------------------------------------

class MBMPretrainer:
    """
    Manages the Masked-Bar Modeling pretraining loop.

    Trains MBMEncoder on raw microstructure bar sequences without labels.
    Loss is mean-squared error on masked positions only (non-masked bars
    do not contribute to the gradient — consistent with BERT masking).

    Args:
        encoder:     MBMEncoder instance.
        device:      Torch device.
        lr:          Learning rate (default 3e-4).
        weight_decay: AdamW weight decay (default 1e-4).
    """

    def __init__(
        self,
        encoder: MBMEncoder,
        device: torch.device,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        self.encoder = encoder.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=lr, weight_decay=weight_decay,
            betas=(0.9, 0.95),   # Slightly reduced beta2 for non-stationary financial data
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=lr * 0.1,
        )

    def masked_reconstruction_loss(
        self,
        recon: torch.Tensor,
        original: torch.Tensor,
        mask_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss computed ONLY at masked positions.

        Args:
            recon:    (batch, seq_len, input_dim) — reconstructed sequence
            original: (batch, seq_len, input_dim) — original sequence
            mask_idx: (batch, seq_len) bool — True at masked positions

        Returns:
            Scalar loss tensor.
        """
        # Expand mask to feature dimension
        mask_expanded = mask_idx.unsqueeze(-1).expand_as(recon)   # (B, T, D)
        masked_recon = recon[mask_expanded]
        masked_orig = original[mask_expanded]

        if masked_recon.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return F.mse_loss(masked_recon, masked_orig)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run one epoch of MBM pretraining.

        Returns:
            Dict with 'loss' (mean epoch MSE on masked positions).
        """
        self.encoder.train()
        total_loss = 0.0
        n_batches = 0

        for masked_seq, original_seq, mask_idx in dataloader:
            masked_seq = masked_seq.to(self.device)
            original_seq = original_seq.to(self.device)
            mask_idx = mask_idx.to(self.device)

            self.optimizer.zero_grad()
            recon, _ = self.encoder(masked_seq, mask_idx)
            loss = self.masked_reconstruction_loss(recon, original_seq, mask_idx)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        return {"loss": avg_loss}

    def pretrain(
        self,
        bar_array: np.ndarray,
        num_epochs: int = 20,
        seq_len: int = 64,
        batch_size: int = 64,
        mask_rate: float = 0.15,
        stride: int = 8,
        log_every: int = 5,
    ) -> List[float]:
        """
        Full pretraining loop on a numpy bar array.

        Args:
            bar_array:  (N, input_dim) float32 normalized bar features.
            num_epochs: Total training epochs.
            seq_len:    Sequence length per window.
            batch_size: Batch size.
            mask_rate:  Fraction of bars to mask.
            stride:     Stride between windows.
            log_every:  Log loss every N epochs.

        Returns:
            List of per-epoch losses.
        """
        dataset = MaskedBarDataset(bar_array, seq_len=seq_len, mask_rate=mask_rate, stride=stride)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        losses = []
        for epoch in range(num_epochs):
            metrics = self.train_epoch(loader)
            losses.append(metrics["loss"])
            if (epoch + 1) % log_every == 0:
                print(f"  [MBM Epoch {epoch+1:3d}/{num_epochs}] Masked Recon Loss: {metrics['loss']:.6f}")

        return losses

    def save_encoder(self, path: str):
        """Save pretrained encoder weights."""
        torch.save({
            "encoder_state_dict": self.encoder.state_dict(),
            "input_dim": self.encoder.input_dim,
            "d_model": self.encoder.d_model,
            "seq_len": self.encoder.seq_len,
        }, path)
        print(f"MBM encoder saved to {path}")

    @classmethod
    def load_encoder(cls, path: str, device: torch.device) -> MBMEncoder:
        """Load pretrained encoder from checkpoint."""
        ckpt = torch.load(path, map_location=device)
        encoder = MBMEncoder(
            input_dim=ckpt["input_dim"],
            d_model=ckpt["d_model"],
            seq_len=ckpt["seq_len"],
        )
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        encoder = encoder.to(device)
        return encoder
