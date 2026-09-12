"""
Multi-Objective Composite Loss for Financial Time Series Forecasting.
Combines Huber regression, directional penalty, cross-sectional Information Coefficient,
and variance-matching router balance regularization.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """
    Composite financial loss balancing magnitude precision, sign accuracy,
    and rank correlation without enforcing uniform routing collapse.
    """

    def __init__(
        self,
        delta_huber: float = 1.0,
        lambda_dir: float = 0.5,
        lambda_ic: float = 0.3,
        lambda_balance: float = 0.1,
    ):
        super().__init__()
        self.delta_huber = delta_huber
        self.lambda_dir = lambda_dir
        self.lambda_ic = lambda_ic
        self.lambda_balance = lambda_balance
        self.huber = nn.HuberLoss(delta=delta_huber)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        g_weights: torch.Tensor,
        noisy_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            y_pred: Model predictions (batch_size, 1).
            y_true: Realized directional returns / targets (batch_size, 1).
            g_weights: Softmax routing allocations (batch_size, num_experts).
            noisy_weights: Exploration-injected routing allocations (batch_size, num_experts).
        Returns:
            Dict containing 'loss', 'loss_huber', 'loss_dir', 'loss_ic', 'loss_balance'.
        """
        eps = 1e-8
        
        # 1. Huber robust regression loss
        l_huber = self.huber(y_pred, y_true)

        # 2. Directional sign penalty: max(0, -sign(y_true) * y_pred)
        sign_mismatch = -torch.sign(y_true) * y_pred
        l_dir = torch.mean(torch.clamp(sign_mismatch, min=0.0))

        # 3. Information Coefficient (Pearson Correlation across batch)
        pred_flat = y_pred.view(-1)
        true_flat = y_true.view(-1)
        if len(pred_flat) > 2:
            pred_centered = pred_flat - pred_flat.mean()
            true_centered = true_flat - true_flat.mean()
            cov = torch.sum(pred_centered * true_centered)
            denom = torch.sqrt(torch.sum(pred_centered**2) + eps) * torch.sqrt(torch.sum(true_centered**2) + eps)
            ic = cov / (denom + eps)
            l_ic = -ic  # Maximize IC
        else:
            l_ic = torch.tensor(0.0, device=y_pred.device)

        # 4. Variance-matching balance regularization
        # P_i = (1 / T) * sum(g_i)
        # m_i = (1 / T) * sum(noisy_g_i)
        # L_balance = E * sum(m_i * P_i)
        num_experts = g_weights.shape[-1]
        p_i = torch.mean(g_weights, dim=0)  # (num_experts,)
        m_i = torch.mean(noisy_weights, dim=0)  # (num_experts,)
        l_balance = float(num_experts) * torch.sum(m_i * p_i)

        total_loss = (
            l_huber
            + self.lambda_dir * l_dir
            + self.lambda_ic * l_ic
            + self.lambda_balance * l_balance
        )

        return {
            "loss": total_loss,
            "loss_huber": l_huber,
            "loss_dir": l_dir,
            "loss_ic": l_ic,
            "loss_balance": l_balance,
        }
