"""
Differential Sharpe and Differential Sortino (D3R) Ratio Formulation.
Enables online streaming reinforcement learning of routing policies with transaction friction.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class DifferentialRiskRatio(nn.Module):
    """
    Computes online recursive Differential Sharpe Ratio (DSR) and
    Differential Downside Deviation Ratio (D3R / Differential Sortino).
    """

    def __init__(self, eta: float = 0.05, cost_bps: float = 1.5):
        super().__init__()
        self.eta = eta
        self.cost_coeff = cost_bps * 1e-4  # Convert bps to fraction

        # Register running moment buffers
        self.register_buffer("a_prev", torch.tensor(0.0))
        self.register_buffer("b_prev", torch.tensor(1e-4))
        self.register_buffer("dd_prev", torch.tensor(1e-4))

    def reset_moments(self) -> None:
        """Reset running exponential moments."""
        self.a_prev.fill_(0.0)
        self.b_prev.fill_(1e-4)
        self.dd_prev.fill_(1e-4)

    def forward(
        self,
        positions: torch.Tensor,
        returns: torch.Tensor,
        objective: str = "sortino",
    ) -> torch.Tensor:
        """
        Args:
            positions: Policy target allocations F_t in [-1.0, 1.0] across time (T,).
            returns: Asset period returns r_t across time (T,).
            objective: 'sortino' (default) or 'sharpe'.
        Returns:
            Scalar objective loss to MINIMIZE (- mean differential ratio).
        """
        t_len = len(positions)
        if t_len < 2:
            return torch.tensor(0.0, device=positions.device, requires_grad=True)

        # F_{t-1}
        pos_prev = positions[:-1]
        pos_curr = positions[1:]
        ret_curr = returns[1:]

        # Realized portfolio return with turnover penalty:
        # R_t = F_{t-1} * r_t - delta * |F_t - F_{t-1}|
        turnover = torch.abs(pos_curr - pos_prev)
        r_t = pos_prev * ret_curr - self.cost_coeff * turnover

        d_ratios = []
        a = self.a_prev.clone()
        b = self.b_prev.clone()
        dd = self.dd_prev.clone()
        eps = 1e-6

        for i in range(len(r_t)):
            r = r_t[i]
            delta_a = r - a
            delta_b = r**2 - b

            if objective.lower() == "sharpe":
                # Moody-Saffell Differential Sharpe Ratio
                var = torch.clamp(b - a**2, min=eps)
                d_t = (b * delta_a - 0.5 * a * delta_b) / (var ** 1.5)
            else:
                # Differential Sortino Ratio (D3R)
                downside_sq = torch.clamp(torch.min(r, torch.tensor(0.0, device=r.device)) ** 2, min=0.0)
                dd_val = torch.clamp(torch.sqrt(dd), min=eps)

                # D_t Sortino branching
                term1 = r - 0.5 * a
                if r > 0:
                    d_t = term1 / dd_val
                else:
                    d_t = (dd * term1 - 0.5 * a * (r**2)) / (dd_val**3 + eps)

                # Update downside variance: DD_t^2 = DD_{t-1}^2 + eta * (min(R_t, 0)^2 - DD_{t-1}^2)
                dd = dd + self.eta * (downside_sq - dd)

            # Update running first and second moments
            a = a + self.eta * delta_a
            b = b + self.eta * delta_b
            d_ratios.append(d_t)

        # Update buffers in non-gradient context
        with torch.no_grad():
            self.a_prev.copy_(a)
            self.b_prev.copy_(b)
            self.dd_prev.copy_(dd)

        d_stack = torch.stack(d_ratios)
        # We maximize differential ratio, so return negative mean
        return -torch.mean(d_stack)
