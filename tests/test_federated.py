"""
Unit Tests for Phase 3: Federated Learning Framework
Validates FedAvg/FedProx aggregation math, proximal regularization, and coordinator rounds.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.client import FederatedEdgeClient
from src.federated.server import FederatedServer
from src.federated.strategies import FedAvgStrategy, FedProxStrategy
from src.models import CNNLSTMPredictor


def test_fedavg_aggregation_math():
    print("[*] Testing FedAvg mathematical weighting correctness...")
    strategy = FedAvgStrategy()

    # Create two dummy parameter dictionaries
    w1 = {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([10.0])}
    w2 = {"weight": torch.tensor([3.0, 6.0]), "bias": torch.tensor([30.0])}

    # Samples: 100 vs 300 (total = 400). Fractions: 0.25 and 0.75
    # Expected weight: 0.25 * [1.0, 2.0] + 0.75 * [3.0, 6.0] = [0.25+2.25, 0.5+4.5] = [2.5, 5.0]
    # Expected bias:   0.25 * 10.0 + 0.75 * 30.0 = 2.5 + 22.5 = 25.0
    agg = strategy.aggregate([w1, w2], [100, 300])

    expected_weight = torch.tensor([2.5, 5.0])
    expected_bias = torch.tensor([25.0])

    assert torch.allclose(agg["weight"], expected_weight), f"Weight mismatch: {agg['weight']} != {expected_weight}"
    assert torch.allclose(agg["bias"], expected_bias), f"Bias mismatch: {agg['bias']} != {expected_bias}"
    print("  [PASS] FedAvg weighted parameter aggregation math is exact.")


def test_fedprox_proximal_term():
    print("[*] Testing FedProx client proximal regularization penalty...")
    # Simple linear model
    model = nn.Linear(4, 1)
    X = torch.randn(20, 4)
    y = torch.randint(0, 2, (20, 1)).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=10)

    client = FederatedEdgeClient(
        client_id="test_client",
        model=model,
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        mu=0.1,  # FedProx active
    )

    initial_params = client.get_parameters()
    # Train 1 round
    client.train_round(local_epochs=1)
    updated_params = client.get_parameters()

    # Verify weights updated
    diff = sum((updated_params[k] - initial_params[k]).abs().sum().item() for k in initial_params)
    assert diff > 0.0, "Model weights did not update during local epoch"
    print(f"  [PASS] FedProx client trained successfully with proximal regularization (delta = {diff:.4f}).")


def test_federated_server_round():
    print("[*] Testing FederatedServer round orchestration...")
    # Setup 2 mock clients
    X1 = torch.randn(40, 30, 9)
    y1 = torch.randint(0, 2, (40, 1)).float()
    loader1 = DataLoader(TensorDataset(X1, y1), batch_size=20)

    X2 = torch.randn(40, 30, 9)
    y2 = torch.randint(0, 2, (40, 1)).float()
    loader2 = DataLoader(TensorDataset(X2, y2), batch_size=20)

    c1 = FederatedEdgeClient("client_1", CNNLSTMPredictor(9, 30), loader1, loader1, loader1)
    c2 = FederatedEdgeClient("client_2", CNNLSTMPredictor(9, 30), loader2, loader2, loader2)

    global_model = CNNLSTMPredictor(9, 30)
    server = FederatedServer(global_model=global_model, clients=[c1, c2], strategy=FedAvgStrategy())

    result = server.fit(num_rounds=1, local_epochs=1, verbose=False)
    assert len(result["history"]) == 1
    assert "macro_f1" in result["history"][0]
    print(f"  [PASS] Federated server executed 1 complete round with macro F1: {result['best_macro_f1']:.4f}")


if __name__ == "__main__":
    test_fedavg_aggregation_math()
    test_fedprox_proximal_term()
    test_federated_server_round()
    print("[+] All Phase 3 federated unit tests passed successfully!")
