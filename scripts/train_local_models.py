#!/usr/bin/env python3
"""
Local Edge Model Training Script
Trains separate predictive maintenance models on client_1, client_2, and client_3 datasets,
simulating isolated on-premise edge factory environments before Federated Learning.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

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
from src.trainer import LocalEdgeTrainer, create_dataloader


def train_clients(
    model_type: str = "cnn_lstm",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    loss_type: str = "focal",
    window_size: int = 30,
    stride: int = 5,
    lookahead_horizon: int = 12,
    data_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
):
    if not HAS_TORCH:
        print("[!] PyTorch is not installed. Please run: pip install torch")
        return

    base_data = Path(data_dir) if data_dir else PROJECT_ROOT / "data"
    chk_dir = Path(checkpoint_dir) if checkpoint_dir else PROJECT_ROOT / "models" / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOCAL EDGE PREDICTIVE MAINTENANCE MODEL TRAINING (PHASE 2)")
    print("Simulating Isolated Edge Factory AI Training")
    print("=" * 80)
    print(f"Model Architecture     : {model_type.upper()}")
    print(f"Loss Function          : {loss_type.upper()}")
    print(f"Window Size (History)  : {window_size} steps (5 mins)")
    print(f"Early Warning Horizon  : {lookahead_horizon} steps (2 mins)")
    print(f"Training Epochs        : {epochs}")
    print(f"Batch Size             : {batch_size}")
    print(f"Checkpoints Path       : {chk_dir}")
    print("-" * 80)

    clients = ["client_1", "client_2", "client_3"]
    results = {}

    for client_id in clients:
        print(f"\n>>> [EDGE CLIENT: {client_id.upper()}] Loading private telemetry...")
        csv_file = base_data / client_id / f"{client_id}_sensor_data.csv"
        df = pd.read_csv(csv_file)

        preprocessor = TimeSeriesPreprocessor(
            window_size=window_size,
            stride=stride,
            lookahead_horizon=lookahead_horizon,
        )
        splits = preprocessor.prepare_client_dataset(df)

        print(
            f"    Sequences: Train={len(splits['X_train']):,}, "
            f"Val={len(splits['X_val']):,}, Test={len(splits['X_test']):,} | "
            f"Input Shape: ({window_size}, {splits['num_features']}) | "
            f"Class Imbalance Weight: {splits['pos_weight']:.2f}"
        )

        train_loader = create_dataloader(splits["X_train"], splits["y_train"], batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(splits["X_val"], splits["y_val"], batch_size=batch_size, shuffle=False)
        test_loader = create_dataloader(splits["X_test"], splits["y_test"], batch_size=batch_size, shuffle=False)

        # Initialize model
        if model_type == "cnn_lstm":
            model = CNNLSTMPredictor(num_features=splits["num_features"], window_size=window_size)
        else:
            model = EdgeMLPPredictor(num_features=splits["num_features"], window_size=window_size)

        # Initialize loss function
        if loss_type == "focal":
            loss_fn = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        else:
            loss_fn = WeightedBCELoss(pos_weight=splits["pos_weight"])

        trainer = LocalEdgeTrainer(model=model, client_id=client_id, loss_fn=loss_fn, lr=lr)

        print("    Training local edge model...")
        trainer.fit(train_loader, val_loader, epochs=epochs, checkpoint_dir=chk_dir, verbose=True)

        print("    Evaluating on held-out test split...")
        test_metrics = trainer.test(test_loader)
        results[client_id] = test_metrics

        print(
            f"    [TEST RESULT] Precision: {test_metrics['precision']:.4f} | "
            f"Recall: {test_metrics['recall']:.4f} | "
            f"F1-Score: {test_metrics['f1']:.4f} | "
            f"ROC-AUC: {test_metrics['roc_auc']:.4f} | "
            f"PR-AUC: {test_metrics['pr_auc']:.4f}"
        )
        print(f"    Confusion Matrix [[TN, FP], [FN, TP]]: {test_metrics['confusion_matrix']}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 2 LOCAL MODEL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Client ID':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'PR-AUC':<10}")
    print("-" * 75)
    for c, m in results.items():
        print(f"{c:<12} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['roc_auc']:<10.4f} | {m['pr_auc']:<10.4f}")
    print("=" * 80)
    print(f"[+] All local models saved to: {chk_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Local Edge Predictive Maintenance Models (Phase 2)")
    parser.add_argument("--model-type", type=str, default="cnn_lstm", choices=["cnn_lstm", "edge_mlp"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "weighted_bce"])
    args = parser.parse_args()

    train_clients(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss,
    )


if __name__ == "__main__":
    main()
