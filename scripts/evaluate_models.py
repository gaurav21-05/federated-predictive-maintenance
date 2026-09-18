#!/usr/bin/env python3
"""
Model Evaluation and Non-IID Domain Divergence Analyzer
Evaluates local edge models and demonstrates cross-client performance drops,
proving the necessity for Federated Learning (Phase 3).
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.loss import BinaryFocalLoss
from src.models import CNNLSTMPredictor
from src.preprocessing import TimeSeriesPreprocessor
from src.trainer import LocalEdgeTrainer, create_dataloader


def evaluate_domain_shift():
    if not HAS_TORCH:
        print("[!] PyTorch is required.")
        return

    data_dir = PROJECT_ROOT / "data"
    chk_dir = PROJECT_ROOT / "models" / "checkpoints"
    clients = ["client_1", "client_2", "client_3"]

    print("=" * 80)
    print("CROSS-CLIENT NON-IID DOMAIN DIVERGENCE EVALUATION")
    print("Evaluating why local factory models struggle on other factories")
    print("=" * 80)

    # Load test sets for all clients
    test_loaders = {}
    for c in clients:
        csv_file = data_dir / c / f"{c}_sensor_data.csv"
        df = pd.read_csv(csv_file)
        prep = TimeSeriesPreprocessor()
        splits = prep.prepare_client_dataset(df)
        test_loaders[c] = create_dataloader(splits["X_test"], splits["y_test"], shuffle=False)

    loss_fn = BinaryFocalLoss()

    print(f"{'Model Trained On':<18} | {'Evaluated On':<18} | {'F1-Score':<10} | {'Recall':<10} | {'ROC-AUC':<10}")
    print("-" * 75)

    for model_client in clients:
        model_path = chk_dir / f"{model_client}_best_model.pt"
        if not model_path.exists():
            print(f"[!] Checkpoint not found: {model_path}. Please run train_local_models.py first.")
            return

        model = CNNLSTMPredictor(num_features=9, window_size=30)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        trainer = LocalEdgeTrainer(model=model, client_id=model_client, loss_fn=loss_fn)

        for eval_client in clients:
            metrics = trainer.evaluate(test_loaders[eval_client])
            tag = "(In-Domain)" if model_client == eval_client else "(Cross-Domain Shift)"
            print(
                f"{model_client:<18} | {eval_client + ' ' + tag:<18} | "
                f"{metrics['f1']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['roc_auc']:<10.4f}"
            )


if __name__ == "__main__":
    evaluate_domain_shift()
