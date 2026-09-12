"""
Counterfactual Expert Credit Assignment and Router Optimization.
Computes first-order Taylor expansion marginal loss reduction for unselected experts.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualCreditAssignment:
    """
    Evaluates what the downstream financial loss would have been if an
    unselected expert had been actively chosen by the router.
    Uses margin-based contrastive loss to update router representations.
    """

    @staticmethod
    def compute_counterfactual_credit(
        grad_loss_wrt_ypred: torch.Tensor,
        active_expert_preds: torch.Tensor,
        all_expert_preds: torch.Tensor,
        margin: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        C_{i,t} approx - grad_L(y_hat)^T (f_i(x_t) - f_a(x_t))
        
        Args:
            grad_loss_wrt_ypred: Gradient of task loss wrt y_hat, shape (batch_size, 1).
            active_expert_preds: Predictions of current highest-weight expert (batch_size, 1).
            all_expert_preds: Predictions of all E experts, shape (batch_size, E).
            margin: Contrastive margin threshold.
        Returns:
            Tuple of:
              - credit_matrix: Counterfactual credit scores C_{i,t} of shape (batch_size, E).
              - contrastive_loss: Router penalty encouraging higher weight for positive credit experts.
        """
        # (batch_size, E) - difference from active expert
        pred_diff = all_expert_preds - active_expert_preds
        # Credit: - grad * diff
        credit_matrix = -grad_loss_wrt_ypred * pred_diff  # (batch_size, E)

        # Contrastive objective: encourage router to upweight experts with positive credit
        positive_mask = (credit_matrix > margin).float()
        contrastive_loss = torch.mean(positive_mask * F.relu(margin - credit_matrix))

        return credit_matrix, contrastive_loss
