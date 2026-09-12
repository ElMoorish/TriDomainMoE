"""
Regime-Gated Residual Mixture of Experts (RG-ResMoE) Architecture.
Decouples macro regime routing from localized asset pricing signals with variable-context experts.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .base_forecaster import BaseForecaster
from .patch_expert import PatchExpert
from .ssm_expert import SSMExpert
from .router import CAWRouter
from .meta_sizer import MetaSizer


class RGResMoE(nn.Module):
    """
    End-to-End RG-ResMoE Architecture.
    Combines unconditional base drift with variable-context residual experts,
    gated by continuous regime routing with Correlation-Aware Weighting.
    """

    def __init__(
        self,
        asset_dim: int,
        regime_dim: int,
        hidden_dim: int = 64,
        lookback_horizons: List[int] = [16, 32, 64],
        lambda_c: float = 0.5,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.asset_dim = asset_dim
        self.regime_dim = regime_dim
        self.lookback_horizons = sorted(lookback_horizons)
        self.num_experts = len(self.lookback_horizons)

        # 1. Base unconditional drift predictor
        self.base_predictor = BaseForecaster(input_dim=asset_dim, hidden_dim=hidden_dim, output_dim=1)

        # 2. Bank of specialized variable-context experts (RAVEN)
        self.experts = nn.ModuleList()
        for i, horizon in enumerate(self.lookback_horizons):
            if i == 0:
                # Shortest horizon: compact patched feed-forward expert
                expert = PatchExpert(
                    input_dim=asset_dim,
                    seq_len=horizon,
                    patch_len=max(2, horizon // 4),
                    hidden_dim=hidden_dim,
                    output_dim=1,
                )
            else:
                # Medium and long horizons: selective state space recurrent experts
                expert = SSMExpert(
                    input_dim=asset_dim,
                    seq_len=horizon,
                    hidden_dim=hidden_dim,
                    num_layers=2,
                    output_dim=1,
                )
            self.experts.append(expert)

        # 3. Softmax regime router with Correlation-Aware Weighting
        self.router = CAWRouter(
            regime_dim=regime_dim,
            num_experts=self.num_experts,
            hidden_dim=hidden_dim // 2,
            lambda_c=lambda_c,
            noise_std=noise_std,
        )

        # 4. Meta-label trade sizer
        self.meta_sizer = MetaSizer(feature_dim=asset_dim, hidden_dim=hidden_dim // 2)

    def forward(
        self,
        x_asset: torch.Tensor,
        z_regime: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_asset: Asset feature tensor (batch_size, max_seq_len, asset_dim).
            z_regime: Macro regime vector (batch_size, regime_dim).
        Returns:
            Dict containing:
              - 'y_pred': Combined prediction (batch_size, 1)
              - 'size': Calibrated position size s_t (batch_size, 1)
              - 'base_pred': Unconditional drift forecast (batch_size, 1)
              - 'expert_preds': Individual expert outputs (batch_size, num_experts)
              - 'weights': Final CAW routing weights (batch_size, num_experts)
              - 'raw_weights': Softmax weights without CAW (batch_size, num_experts)
              - 'entropy': Routing Shannon entropy (batch_size,)
              - 'noisy_weights': Exploration-injected weights for balance loss (batch_size, num_experts)
        """
        # 1. Base prediction
        base_out = self.base_predictor(x_asset)  # (batch_size, 1)

        # 2. Evaluate all experts across their respective lookbacks
        expert_preds_list = []
        expert_hidden_list = []

        for horizon, expert in zip(self.lookback_horizons, self.experts):
            # Slice the sequence to the expert's specific lookback horizon tau_i
            x_slice = x_asset[:, -horizon:, :]
            pred_i, h_i = expert(x_slice)
            expert_preds_list.append(pred_i)  # (batch_size, 1)
            expert_hidden_list.append(h_i)    # (batch_size, hidden_dim)

        # Stack expert predictions: (batch_size, num_experts)
        expert_preds = torch.cat(expert_preds_list, dim=-1)

        # 3. Softmax routing over regime vector with CAW cosine repulsion
        weights, raw_weights, entropy, noisy_weights = self.router(
            z=z_regime,
            expert_hidden_states=expert_hidden_list,
        )

        # 4. Weighted residual expert combination
        # sum_{i=1}^E w_i * f_expert_i
        residual_out = torch.sum(weights * expert_preds, dim=-1, keepdim=True)  # (batch_size, 1)

        # 5. Final forecast: y_hat = f_base + residual
        final_pred = base_out + residual_out

        # 6. Meta conviction sizing
        size = self.meta_sizer(x_asset, final_pred)

        return {
            "y_pred": final_pred,
            "size": size,
            "base_pred": base_out,
            "expert_preds": expert_preds,
            "weights": weights,
            "raw_weights": raw_weights,
            "entropy": entropy,
            "noisy_weights": noisy_weights,
        }
