"""
Institutional Tri-Domain Mixture of Experts (TriDomainMoE).
Synthesizes Technical Microstructure, Macroeconomic Term Structures,
and Fundamental Sentiment via Continuous Softmax CAW Routing and Meta-Sizing.

Supports two modes:
  v2 (default): Standard OHLCV+OFI tech features — backward compatible with live trader.
  v3 (advanced): VQ-VAE tokenizer enriches tech expert with discrete market states.
                 Activated by passing vq_tokenizer and vq_latent_dim arguments.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from .base_forecaster import BaseForecaster
from .domain_experts import MicrostructureTechExpert, MacroSSMExpert, FundamentalSentimentExpert
from .router import CAWRouter
from .meta_sizer import MetaSizer


class TriDomainMoE(nn.Module):
    """
    Production Tri-Domain MoE Architecture.
    Achieves alpha by decoupling and synthesizing:
      1. Technical Microstructure (Volume Bars, OFI, Intraday price action)
      2. Macroeconomic Term Structures (Cross-Asset yields, VRP, DXY, Oil/Gold spreads)
      3. Fundamental Sentiment (News polarity, uncertainty, macro narrative topics)
    """

    DOMAIN_NAMES = ["technical", "macro", "fundamental"]

    def __init__(
        self,
        tech_dim: int,
        macro_dim: int,
        fund_dim: int,
        regime_dim: int,
        hidden_dim: int = 64,
        lambda_c: float = 0.5,
        noise_std: float = 0.1,
        vq_tokenizer: Optional[nn.Module] = None,
        vq_latent_dim: int = 64,
    ):
        super().__init__()
        self.tech_dim = tech_dim
        self.macro_dim = macro_dim
        self.fund_dim = fund_dim
        self.regime_dim = regime_dim
        self.num_experts = 3  # [Tech, Macro, Fundamental]

        # 1. Base Unconditional Drift Forecaster
        self.base_predictor = BaseForecaster(input_dim=tech_dim, hidden_dim=hidden_dim, output_dim=1)

        # 2. Specialized Domain Experts
        # v3: MicrostructureTechExpert accepts vq_tokenizer for discrete token enrichment
        self.tech_expert = MicrostructureTechExpert(
            input_dim=tech_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            vq_tokenizer=vq_tokenizer,
            vq_latent_dim=vq_latent_dim,
        )
        self.macro_expert = MacroSSMExpert(input_dim=macro_dim, hidden_dim=hidden_dim, num_layers=2, output_dim=1)
        self.fund_expert = FundamentalSentimentExpert(input_dim=fund_dim, hidden_dim=hidden_dim, output_dim=1)

        # 3. Softmax CAW Router over external regime vector z_t
        self.router = CAWRouter(
            regime_dim=regime_dim,
            num_experts=self.num_experts,
            hidden_dim=hidden_dim // 2,
            lambda_c=lambda_c,
            noise_std=noise_std,
        )

        # 4. Meta-Labeling Trade Sizer
        self.meta_sizer = MetaSizer(feature_dim=tech_dim, hidden_dim=hidden_dim // 2)

    def forward(
        self,
        x_tech: torch.Tensor,
        x_macro: torch.Tensor,
        x_fund: torch.Tensor,
        z_regime: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_tech: Technical feature tensor (batch, seq_len, tech_dim)
            x_macro: Macro cross-asset tensor (batch, seq_len, macro_dim)
            x_fund: Fundamental sentiment tensor (batch, fund_dim) or (batch, seq_len, fund_dim)
            z_regime: Macro regime state vector (batch, regime_dim)
        Returns:
            Dict containing:
              - 'y_pred': Final combined directional forecast (batch, 1)
              - 'size': Calibrated position size s_t (batch, 1)
              - 'base_pred': Unconditional trend forecast (batch, 1)
              - 'expert_preds': Predictions [pred_tech, pred_macro, pred_fund] (batch, 3)
              - 'weights': CAW routing weights [w_tech, w_macro, w_fund] (batch, 3)
              - 'entropy': Routing Shannon entropy (batch,)
              - 'noisy_weights': Exploration weights for balance loss (batch, 3)
        """
        # 1. Base forecast
        base_out = self.base_predictor(x_tech)

        # 2. Domain expert forward passes
        pred_tech, h_tech = self.tech_expert(x_tech)
        pred_macro, h_macro = self.macro_expert(x_macro)
        pred_fund, h_fund = self.fund_expert(x_fund)

        expert_preds = torch.cat([pred_tech, pred_macro, pred_fund], dim=-1)  # (batch, 3)
        hidden_states = [h_tech, h_macro, h_fund]

        # 3. CAW soft routing over regime state z_t
        weights, raw_weights, entropy, noisy_weights = self.router(
            z=z_regime,
            expert_hidden_states=hidden_states,
        )

        # 4. Weighted residual combination: sum_{i=1}^3 w_i * f_domain_i
        residual_out = torch.sum(weights * expert_preds, dim=-1, keepdim=True)  # (batch, 1)
        final_pred = base_out + residual_out

        # 5. Conviction sizing
        size = self.meta_sizer(x_tech, final_pred)

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
