"""
Unit Tests for Phase 2: Local Edge Predictive Maintenance Models
Validates tensor dimensions, forward pass, gradient backpropagation, and loss functions.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.loss import BinaryFocalLoss, WeightedBCELoss
from src.models import CNNLSTMPredictor, EdgeMLPPredictor
from src.preprocessing import TimeSeriesPreprocessor


def test_preprocessing():
    print("[*] Testing TimeSeriesPreprocessor...")
    import pandas as pd

    # Mock dataset
    num_rows = 500
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-03-01", periods=num_rows, freq="10s"),
        "machine_id": ["TEST_01"] * num_rows,
        "temperature": np.random.uniform(50, 70, size=num_rows),
        "humidity": np.random.uniform(30, 60, size=num_rows),
        "vibration": np.random.uniform(1.0, 2.5, size=num_rows),
        "current": np.random.uniform(15, 25, size=num_rows),
        "rpm": np.random.uniform(3000, 4000, size=num_rows),
        "pressure": np.random.uniform(5.0, 7.0, size=num_rows),
        "failure": [0] * (num_rows - 20) + [1] * 20,
    })

    prep = TimeSeriesPreprocessor(window_size=30, stride=5, lookahead_horizon=12)
    splits = prep.prepare_client_dataset(df)

    assert "X_train" in splits
    assert splits["X_train"].ndim == 3  # (N, W, D)
    assert splits["X_train"].shape[1] == 30
    assert splits["y_train"].ndim == 1
    assert not np.isnan(splits["X_train"]).any()
    print(f"  [PASS] Preprocessor created {len(splits['X_train'])} windows of shape {splits['X_train'].shape[1:]}")


def test_models():
    if not HAS_TORCH:
        print("[!] PyTorch not installed, skipping neural net test.")
        return

    print("[*] Testing CNNLSTMPredictor & EdgeMLPPredictor forward pass...")
    batch_size = 16
    window_size = 30
    num_features = 9

    dummy_input = torch.randn(batch_size, window_size, num_features)
    dummy_target = torch.randint(0, 2, (batch_size, 1)).float()

    # 1. Test CNN-LSTM
    cnn_lstm = CNNLSTMPredictor(num_features=num_features, window_size=window_size)
    logits_cnn = cnn_lstm(dummy_input)
    assert logits_cnn.shape == (batch_size, 1), f"Expected (16, 1), got {logits_cnn.shape}"

    # Test weight extraction & loading for Federated Learning
    weights = cnn_lstm.get_weights()
    assert isinstance(weights, dict)
    cnn_lstm.set_weights(weights)
    print("  [PASS] CNNLSTMPredictor forward pass and weight serialization successful.")

    # 2. Test EdgeMLP
    mlp = EdgeMLPPredictor(num_features=num_features, window_size=window_size)
    logits_mlp = mlp(dummy_input)
    assert logits_mlp.shape == (batch_size, 1)
    print("  [PASS] EdgeMLPPredictor forward pass successful.")

    # 3. Test Loss Functions & Backward Pass
    fresh_logits = cnn_lstm(dummy_input)
    focal_loss = BinaryFocalLoss()
    loss_focal = focal_loss(fresh_logits, dummy_target)
    loss_focal.backward()
    print(f"  [PASS] BinaryFocalLoss backward pass successful (Loss = {loss_focal.item():.4f}).")


if __name__ == "__main__":
    test_preprocessing()
    test_models()
    print("[+] All Phase 2 unit tests passed!")
