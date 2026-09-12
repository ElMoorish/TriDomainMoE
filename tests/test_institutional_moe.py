"""Unit tests for TriDomainMoE and Domain Experts."""

import unittest
import torch

from src.models.domain_experts import MicrostructureTechExpert, MacroSSMExpert, FundamentalSentimentExpert
from src.models.institutional_moe import TriDomainMoE


class TestInstitutionalMoE(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 32
        self.tech_dim = 6
        self.macro_dim = 4
        self.fund_dim = 6
        self.regime_dim = 6

        self.model = TriDomainMoE(
            tech_dim=self.tech_dim,
            macro_dim=self.macro_dim,
            fund_dim=self.fund_dim,
            regime_dim=self.regime_dim,
            hidden_dim=32,
            lambda_c=0.5,
        )

    def test_forward_pass_shapes(self):
        x_tech = torch.randn(self.batch_size, self.seq_len, self.tech_dim)
        x_macro = torch.randn(self.batch_size, self.seq_len, self.macro_dim)
        x_fund = torch.randn(self.batch_size, self.fund_dim)
        z_regime = torch.randn(self.batch_size, self.regime_dim)

        out = self.model(x_tech, x_macro, x_fund, z_regime)

        self.assertEqual(out["y_pred"].shape, (self.batch_size, 1))
        self.assertEqual(out["size"].shape, (self.batch_size, 1))
        self.assertEqual(out["weights"].shape, (self.batch_size, 3))
        self.assertEqual(out["expert_preds"].shape, (self.batch_size, 3))

        # Weights sum to 1.0
        torch.testing.assert_close(out["weights"].sum(dim=-1), torch.ones(self.batch_size), atol=1e-5, rtol=1e-5)

    def test_backward_gradients_all_domains(self):
        x_tech = torch.randn(self.batch_size, self.seq_len, self.tech_dim)
        x_macro = torch.randn(self.batch_size, self.seq_len, self.macro_dim)
        x_fund = torch.randn(self.batch_size, self.fund_dim)
        z_regime = torch.randn(self.batch_size, self.regime_dim)

        out = self.model(x_tech, x_macro, x_fund, z_regime)
        loss = out["y_pred"].sum() + out["size"].sum()
        loss.backward()

        # Check gradients exist on all 3 experts and router
        self.assertIsNotNone(self.model.tech_expert.conv1.weight.grad)
        self.assertIsNotNone(self.model.macro_expert.input_embed.weight.grad)
        self.assertIsNotNone(self.model.fund_expert.embed.weight.grad)
        self.assertIsNotNone(self.model.router.gate[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
