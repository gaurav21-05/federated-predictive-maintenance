#!/usr/bin/env python3
"""
Comparative Benchmark Script: Isolated Local vs. Federated Consensus Model
Demonstrates how Federated Learning eliminates non-IID cross-domain collapse.
"""

import sys
from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loss import BinaryFocalLoss
from src.models import CNNLSTMPredictor
from src.preprocessing import TimeSeriesPreprocessor
from src.trainer import LocalEdgeTrainer, create_dataloader


def run_benchmark():
    data_dir = PROJECT_ROOT / "data"
    chk_dir = PROJECT_ROOT / "models" / "checkpoints"
    clients = ["client_1", "client_2", "client_3"]

    print("=" * 90)
    print("FEDERATED PREDICTIVE MAINTENANCE BENCHMARK STUDY")
    print("Evaluating: (1) Isolated Local Models vs (2) Global Federated Consensus Model")
    print("=" * 90)

    # 1. Prepare test data loaders
    preprocessor = TimeSeriesPreprocessor()
    test_loaders = {}

    for c in clients:
        csv_file = data_dir / c / f"{c}_sensor_data.csv"
        df = pd.read_csv(csv_file)
        splits = preprocessor.prepare_client_dataset(df)
        test_loaders[c] = create_dataloader(splits["X_test"], splits["y_test"], batch_size=64, shuffle=False)

    loss_fn = BinaryFocalLoss()

    # 2. Evaluate Isolated Local Models
    local_evals = {}
    for c in clients:
        model_path = chk_dir / f"{c}_best_model.pt"
        if not model_path.exists():
            print(f"[!] Local checkpoint missing: {model_path}")
            return
        model = CNNLSTMPredictor(num_features=9, window_size=30)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        trainer = LocalEdgeTrainer(model=model, client_id=c, loss_fn=loss_fn)

        c_scores = {}
        for test_c in clients:
            m = trainer.evaluate(test_loaders[test_c])
            c_scores[test_c] = m
        local_evals[c] = c_scores

    # 3. Evaluate Global Federated Model
    global_path = chk_dir / "global_federated_best.pt"
    if not global_path.exists():
        print(f"[!] Global federated checkpoint missing: {global_path}. Run run_federated.py first.")
        return

    global_model = CNNLSTMPredictor(num_features=9, window_size=30)
    global_model.load_state_dict(torch.load(global_path, map_location="cpu"))
    fed_trainer = LocalEdgeTrainer(model=global_model, client_id="global_fed", loss_fn=loss_fn)

    global_scores = {}
    for test_c in clients:
        global_scores[test_c] = fed_trainer.evaluate(test_loaders[test_c])

    # 4. Print Benchmark Summary Table
    print("\n" + "=" * 90)
    print(f"{'Deployment / Model':<28} | {'Client 1 F1':<12} | {'Client 2 F1':<12} | {'Client 3 F1':<12} | {'Macro F1':<10}")
    print("-" * 90)

    for c in clients:
        m1 = local_evals[c]["client_1"]["f1"]
        m2 = local_evals[c]["client_2"]["f1"]
        m3 = local_evals[c]["client_3"]["f1"]
        macro = (m1 + m2 + m3) / 3.0
        print(f"{'Isolated Local (' + c + ')':<28} | {m1:<12.4f} | {m2:<12.4f} | {m3:<12.4f} | {macro:<10.4f}")

    print("-" * 90)
    gf1 = global_scores["client_1"]["f1"]
    gf2 = global_scores["client_2"]["f1"]
    gf3 = global_scores["client_3"]["f1"]
    g_macro = (gf1 + gf2 + gf3) / 3.0
    print(f"{'GLOBAL FEDERATED (FedProx)':<28} | {gf1:<12.4f} | {gf2:<12.4f} | {gf3:<12.4f} | {g_macro:<10.4f}")
    print("=" * 90)

    print("\nDetailed Metric Breakdown for Global Federated Model:")
    for test_c in clients:
        sm = global_scores[test_c]
        print(f"  >>> {test_c.upper()}: Precision={sm['precision']:.4f}, Recall={sm['recall']:.4f}, F1={sm['f1']:.4f}, ROC-AUC={sm['roc_auc']:.4f}")
    print("-" * 90)


if __name__ == "__main__":
    run_benchmark()
