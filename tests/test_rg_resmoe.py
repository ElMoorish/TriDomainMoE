"""Unit tests for RG-ResMoE and neural components."""

import unittest
import torch
from src.models.rg_resmoe import RGResMoE
from src.models.router import CAWRouter


class TestRGResMoE(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 64
        self.asset_dim = 8
        self.regime_dim = 6
        self.model = RGResMoE(
            asset_dim=self.asset_dim,
            regime_dim=self.regime_dim,
            hidden_dim=32,
            lookback_horizons=[16, 32, 64],
            lambda_c=0.5,
        )

    def test_forward_pass_shapes(self):
        x = torch.randn(self.batch_size, self.seq_len, self.asset_dim)
        z = torch.randn(self.batch_size, self.regime_dim)

        out = self.model(x, z)

        self.assertEqual(out["y_pred"].shape, (self.batch_size, 1))
        self.assertEqual(out["size"].shape, (self.batch_size, 1))
        self.assertEqual(out["base_pred"].shape, (self.batch_size, 1))
        self.assertEqual(out["expert_preds"].shape, (self.batch_size, 3))
        self.assertEqual(out["weights"].shape, (self.batch_size, 3))
        self.assertEqual(out["entropy"].shape, (self.batch_size,))

    def test_near_zero_expert_initialization(self):
        # Initialized experts should output 0, so y_pred == base_pred initially
        x = torch.randn(self.batch_size, self.seq_len, self.asset_dim)
        z = torch.randn(self.batch_size, self.regime_dim)

        out = self.model(x, z)
        # Check that residual contribution is practically zero
        torch.testing.assert_close(out["y_pred"], out["base_pred"], atol=1e-5, rtol=1e-5)

    def test_backward_pass_gradients(self):
        x = torch.randn(self.batch_size, self.seq_len, self.asset_dim)
        z = torch.randn(self.batch_size, self.regime_dim)

        out = self.model(x, z)
        loss = out["y_pred"].sum() + out["size"].sum()
        loss.backward()

        # Check gradients exist on router and base predictor
        self.assertIsNotNone(self.model.base_predictor.net[0].weight.grad)
        self.assertIsNotNone(self.model.router.gate[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
