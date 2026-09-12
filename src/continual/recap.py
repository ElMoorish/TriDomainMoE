"""
Regime-Aware Continual Adaptive Portfolio (ReCAP) Framework.
Prevents catastrophic forgetting via modular parameter delta library isolation.
"""

from typing import Dict, List, Optional
import copy
import torch
import torch.nn as nn


class ReCAPPolicyLibrary:
    """
    Maintains a frozen baseline parameter set theta_0 and an expanding
    library of modular parameter delta vectors D = {d_1, ..., d_K}.
    Synthesizes active network parameters dynamically based on regime allocations.
    """

    def __init__(self, base_model: nn.Module):
        # Freeze base parameters
        self.base_model = copy.deepcopy(base_model)
        for param in self.base_model.parameters():
            param.requires_grad = False

        self._base_state_dict = {
            k: v.clone().detach() for k, v in self.base_model.state_dict().items()
        }

        # Library of delta vectors: dict of {regime_name: {param_name: delta_tensor}}
        self.policy_library: Dict[str, Dict[str, torch.Tensor]] = {}

    def add_policy_delta(self, regime_name: str, adapted_model: nn.Module) -> None:
        """
        Compute and store delta vector d_k = theta_k - theta_0.
        """
        delta_dict = {}
        adapted_sd = adapted_model.state_dict()

        for k, v_base in self._base_state_dict.items():
            if k in adapted_sd:
                v_adapted = adapted_sd[k]
                delta_dict[k] = (v_adapted - v_base).clone().detach()

        self.policy_library[regime_name] = delta_dict

    def synthesize_active_parameters(
        self,
        target_model: nn.Module,
        regime_weights: Dict[str, float],
    ) -> None:
        """
        Dynamically synthesize active network parameters:
        theta_active = theta_0 + sum_{k=1}^K w_k * d_k
        """
        new_sd = {}
        for k, v_base in self._base_state_dict.items():
            param_accum = v_base.clone()

            for regime_name, weight in regime_weights.items():
                if regime_name in self.policy_library:
                    delta = self.policy_library[regime_name].get(k)
                    if delta is not None:
                        param_accum += weight * delta

            new_sd[k] = param_accum

        target_model.load_state_dict(new_sd)

    def num_policies(self) -> int:
        return len(self.policy_library)
