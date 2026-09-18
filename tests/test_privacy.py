"""
Unit Test Suite for Phase 4: Privacy Preservation & Security Hardening
Validates Differential Privacy Engine, Rényi Privacy Accountant, and
Pairwise Masking Secure Aggregation.
"""

import math
import os
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.privacy.dp_engine import DifferentialPrivacyEngine
from src.privacy.accountant import PrivacyAccountant
from src.privacy.secure_aggregation import SecureAggregationSimulator
from src.federated.client import FederatedEdgeClient
from src.federated.server import FederatedServer
from src.federated.strategies import FedProxStrategy


class TestDifferentialPrivacy(unittest.TestCase):
    """Tests for DifferentialPrivacyEngine."""

    def test_clipping_norm_enforcement(self):
        """Verifies that parameter updates are strictly clamped to clip_norm C."""
        clip_c = 1.5
        engine = DifferentialPrivacyEngine(clip_norm=clip_c, noise_multiplier=0.0)

        # Create large weight difference
        global_params = {"weight": torch.zeros(10, 10)}
        local_params = {"weight": torch.ones(10, 10) * 5.0}  # Raw L2 norm = sqrt(100 * 25) = 50.0

        raw_norm = engine.compute_update_norm(local_params, global_params)
        self.assertAlmostEqual(raw_norm, 50.0, places=4)

        priv_params, stats = engine.clip_and_perturb_update(local_params, global_params)
        diff = priv_params["weight"] - global_params["weight"]
        actual_norm = diff.norm(2).item()

        self.assertAlmostEqual(actual_norm, clip_c, places=4)
        self.assertAlmostEqual(stats["post_clip_norm"], clip_c, places=4)

    def test_under_clipping_preservation(self):
        """Verifies that parameter updates below C are preserved without attenuation."""
        clip_c = 10.0
        engine = DifferentialPrivacyEngine(clip_norm=clip_c, noise_multiplier=0.0)

        global_params = {"weight": torch.zeros(4, 4)}
        local_params = {"weight": torch.ones(4, 4) * 0.5}  # Raw L2 norm = sqrt(16 * 0.25) = 2.0

        priv_params, stats = engine.clip_and_perturb_update(local_params, global_params)
        diff = priv_params["weight"] - global_params["weight"]
        actual_norm = diff.norm(2).item()

        self.assertAlmostEqual(actual_norm, 2.0, places=4)
        self.assertAlmostEqual(stats["clip_factor"], 1.0, places=4)

    def test_gaussian_noise_properties(self):
        """Verifies empirical mean and variance of injected perturbation noise."""
        clip_c = 2.0
        sigma = 0.5
        engine = DifferentialPrivacyEngine(clip_norm=clip_c, noise_multiplier=sigma, seed=42)

        # Large tensor for statistical power (100,000 elements)
        global_params = {"weight": torch.zeros(100, 1000)}
        local_params = {"weight": torch.zeros(100, 1000)}  # Zero update -> output is pure noise

        priv_params, _ = engine.clip_and_perturb_update(local_params, global_params)
        noise = priv_params["weight"].numpy()

        expected_std = sigma * clip_c  # 1.0
        self.assertAlmostEqual(float(np.mean(noise)), 0.0, delta=0.02)
        self.assertAlmostEqual(float(np.std(noise)), expected_std, delta=0.02)


class TestPrivacyAccountant(unittest.TestCase):
    """Tests for Rényi Differential Privacy (RDP) Accountant."""

    def test_privacy_accumulation(self):
        """Verifies that epsilon monotonically increases across communication rounds."""
        accountant = PrivacyAccountant(target_delta=1e-5)
        epsilons = []

        for _ in range(5):
            eps, delta = accountant.step(noise_multiplier=0.5)
            epsilons.append(eps)

        self.assertEqual(len(epsilons), 5)
        for i in range(len(epsilons) - 1):
            self.assertGreater(epsilons[i + 1], epsilons[i])

    def test_noise_calibration(self):
        """Verifies that calibrated noise multiplier achieves requested target epsilon."""
        target_eps = 2.5
        target_delta = 1e-5
        rounds = 6

        calibrated_sigma = PrivacyAccountant.calibrate_noise_multiplier(
            target_epsilon=target_eps,
            num_rounds=rounds,
            delta=target_delta,
        )
        self.assertGreater(calibrated_sigma, 0.0)

        # Verify forward computation matches target_eps
        actual_eps = PrivacyAccountant.compute_rdp_epsilon(
            noise_multiplier=calibrated_sigma,
            num_rounds=rounds,
            delta=target_delta,
        )
        self.assertAlmostEqual(actual_eps, target_eps, delta=0.005)


class TestSecureAggregation(unittest.TestCase):
    """Tests for SecureAggregationSimulator."""

    def test_exact_mask_cancellation(self):
        """Verifies that pairwise masks cancel identically to zero for arbitrary client weights."""
        secagg = SecureAggregationSimulator(mask_magnitude=50.0, seed=123)

        # 3 clients with non-trivial model tensors
        client_updates = {
            "client_1": {"fc.weight": torch.full((5, 5), 1.0), "fc.bias": torch.full((5,), 0.5)},
            "client_2": {"fc.weight": torch.full((5, 5), 4.0), "fc.bias": torch.full((5,), 2.0)},
            "client_3": {"fc.weight": torch.full((5, 5), 9.0), "fc.bias": torch.full((5,), 3.5)},
        }
        client_weights = {"client_1": 100.0, "client_2": 200.0, "client_3": 100.0}
        # Normalized weights: 0.25, 0.50, 0.25
        # Expected aggregate weight: 0.25*1 + 0.50*4 + 0.25*9 = 0.25 + 2.0 + 2.25 = 4.5
        # Expected aggregate bias: 0.25*0.5 + 0.50*2.0 + 0.25*3.5 = 0.125 + 1.0 + 0.875 = 2.0

        masked_updates, audit = secagg.mask_client_updates(
            client_updates,
            client_weights=client_weights,
            round_num=1,
        )

        # Check individual client updates are blinded
        self.assertTrue(audit["is_blinded"])

        # Aggregate masked updates
        aggregated_params, _ = secagg.unmask_and_aggregate(masked_updates)

        # Verify exact mathematical match
        expected_weight = torch.full((5, 5), 4.5)
        expected_bias = torch.full((5,), 2.0)

        max_weight_err = (aggregated_params["fc.weight"] - expected_weight).abs().max().item()
        max_bias_err = (aggregated_params["fc.bias"] - expected_bias).abs().max().item()

        self.assertLess(max_weight_err, 1e-5)
        self.assertLess(max_bias_err, 1e-5)


class TestEndToEndPrivacyFederated(unittest.TestCase):
    """End-to-end integration test with FederatedServer."""

    def test_server_with_dp_and_secagg(self):
        """Runs 2 rounds of DP + SecAgg federated training on tiny synthetic data."""
        # Small linear model for rapid test execution
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))

        # Synthetic datasets
        X_dummy = torch.randn(20, 8)
        y_dummy = torch.randint(0, 2, (20, 1)).float()
        ds = TensorDataset(X_dummy, y_dummy)
        loader = DataLoader(ds, batch_size=5)

        clients = [
            FederatedEdgeClient("c1", nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1)), loader, loader, loader),
            FederatedEdgeClient("c2", nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1)), loader, loader, loader),
            FederatedEdgeClient("c3", nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1)), loader, loader, loader),
        ]

        server = FederatedServer(
            global_model=model,
            clients=clients,
            strategy=FedProxStrategy(mu=0.01),
        )

        results = server.fit(
            num_rounds=2,
            local_epochs=1,
            enable_dp=True,
            clip_norm=1.0,
            noise_multiplier=0.05,
            enable_secagg=True,
            verbose=False,
        )

        self.assertEqual(len(results["history"]), 2)
        self.assertTrue(results["history"][0]["dp_enabled"])
        self.assertTrue(results["history"][0]["secagg_enabled"])
        self.assertNotEqual(results["final_epsilon"], float("inf"))


if __name__ == "__main__":
    unittest.main()
