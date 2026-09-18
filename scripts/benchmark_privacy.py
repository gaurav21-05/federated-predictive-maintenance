#!/usr/bin/env python3
"""
Privacy-Utility Trade-Off Benchmark Script
Evaluates the Pareto frontier between Differential Privacy budget (epsilon, delta)
and predictive maintenance diagnostic performance (F1-score, Recall, ROC-AUC)
across all industrial factory nodes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.client import FederatedEdgeClient
from src.federated.server import FederatedServer
from src.federated.strategies import FedProxStrategy
from src.models import CNNLSTMPredictor
from src.preprocessing import TimeSeriesPreprocessor
from src.privacy.accountant import PrivacyAccountant
from src.trainer import create_dataloader


def run_privacy_benchmark(
    rounds: int = 4,
    local_epochs: int = 1,
    mu: float = 0.01,
    clip_norm: float = 1.0,
    target_delta: float = 1e-5,
) -> List[Dict]:
    data_dir = PROJECT_ROOT / "data"
    client_ids = ["client_1", "client_2", "client_3"]
    preprocessor = TimeSeriesPreprocessor(window_size=30, stride=5, lookahead_horizon=12)

    # 1. Load client datasets into memory once
    client_splits = {}
    for cid in client_ids:
        csv_file = data_dir / cid / f"{cid}_sensor_data.csv"
        df = pd.read_csv(csv_file)
        client_splits[cid] = preprocessor.prepare_client_dataset(df)

    num_features = client_splits["client_1"]["num_features"]

    # 2. Define privacy levels to benchmark
    test_levels = [
        {"name": "Non-Private (Baseline)", "sigma": 0.0, "enable_dp": False},
        {"name": "Low Noise (High Utility)", "sigma": 0.01, "enable_dp": True},
        {"name": "Balanced Privacy (Standard)", "sigma": 0.05, "enable_dp": True},
        {"name": "High Privacy (Strict)", "sigma": 0.15, "enable_dp": True},
        {"name": "Extreme Privacy (Stress)", "sigma": 0.50, "enable_dp": True},
    ]

    benchmark_results = []
    baseline_macro_f1 = None

    print("=" * 95)
    print("EMPIRICAL PRIVACY-UTILITY BENCHMARK SUITE (FEDPROX + DP-SGD + SECAGG)")
    print(f"Rounds: {rounds} | Local Epochs: {local_epochs} | Proximal Mu: {mu} | Clip Norm C: {clip_norm} | Delta: {target_delta}")
    print("=" * 95)

    for idx, lvl in enumerate(test_levels, start=1):
        print(f"\n[{idx}/{len(test_levels)}] Running Configuration: {lvl['name']} (sigma = {lvl['sigma']})")

        # Initialize fresh client edge nodes
        clients = []
        for cid in client_ids:
            splits = client_splits[cid]
            train_loader = create_dataloader(splits["X_train"], splits["y_train"], batch_size=64, shuffle=True)
            val_loader = create_dataloader(splits["X_val"], splits["y_val"], batch_size=64, shuffle=False)
            test_loader = create_dataloader(splits["X_test"], splits["y_test"], batch_size=64, shuffle=False)

            model = CNNLSTMPredictor(num_features=num_features, window_size=30)
            client = FederatedEdgeClient(
                client_id=cid,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                lr=0.001,
                mu=mu,
            )
            clients.append(client)

        global_model = CNNLSTMPredictor(num_features=num_features, window_size=30)
        strategy = FedProxStrategy(mu=mu)
        server = FederatedServer(global_model=global_model, clients=clients, strategy=strategy)

        # Run federated training
        fit_results = server.fit(
            num_rounds=rounds,
            local_epochs=local_epochs,
            enable_dp=lvl["enable_dp"],
            clip_norm=clip_norm,
            noise_multiplier=lvl["sigma"],
            target_delta=target_delta,
            enable_secagg=True,
            verbose=False,
        )

        final_eval = server.evaluate_global_model()
        macro = final_eval["macro_average"]
        c1 = final_eval["client_1"]
        c2 = final_eval["client_2"]
        c3 = final_eval["client_3"]

        # Compute cumulative privacy budget
        if lvl["enable_dp"]:
            eps = PrivacyAccountant.compute_rdp_epsilon(
                noise_multiplier=lvl["sigma"],
                num_rounds=rounds,
                delta=target_delta,
            )
            eps_str = f"{eps:.2f}"
        else:
            eps = float("inf")
            eps_str = "Inf"

        if baseline_macro_f1 is None:
            baseline_macro_f1 = macro["f1"]
            relative_drop = 0.0
        else:
            relative_drop = ((baseline_macro_f1 - macro["f1"]) / baseline_macro_f1) * 100.0

        print(
            f"    -> Privacy Spent: eps = {eps_str} (delta={target_delta}) | "
            f"Macro F1: {macro['f1']:.4f} (F1 drop: {relative_drop:+.2f}%) | ROC-AUC: {macro['roc_auc']:.4f}"
        )

        record = {
            "level_name": lvl["name"],
            "noise_multiplier": lvl["sigma"],
            "enable_dp": lvl["enable_dp"],
            "epsilon": float(eps) if eps != float("inf") else "infinity",
            "delta": target_delta,
            "c1_f1": c1["f1"],
            "c2_f1": c2["f1"],
            "c3_f1": c3["f1"],
            "macro_f1": macro["f1"],
            "macro_recall": macro["recall"],
            "macro_precision": macro["precision"],
            "macro_roc_auc": macro["roc_auc"],
            "relative_f1_drop_pct": relative_drop,
        }
        benchmark_results.append(record)

    # 3. Print Final Benchmark Summary Table
    print("\n" + "=" * 105)
    print("FINAL PRIVACY-UTILITY BENCHMARK SUMMARY (PARETO FRONTIER)")
    print("=" * 105)
    header = f"{'Configuration':<28} | {'Noise (s)':<9} | {'Epsilon (e)':<11} | {'C1 F1':<8} | {'C2 F1':<8} | {'C3 F1':<8} | {'Macro F1':<9} | {'ROC-AUC':<8} | {'F1 Drop':<8}"
    print(header)
    print("-" * 105)

    for r in benchmark_results:
        eps_repr = f"{r['epsilon']:.2f}" if isinstance(r['epsilon'], (int, float)) else r['epsilon']
        row = (
            f"{r['level_name']:<28} | "
            f"{r['noise_multiplier']:<9.2f} | "
            f"{eps_repr:<11} | "
            f"{r['c1_f1']:<8.4f} | "
            f"{r['c2_f1']:<8.4f} | "
            f"{r['c3_f1']:<8.4f} | "
            f"{r['macro_f1']:<9.4f} | "
            f"{r['macro_roc_auc']:<8.4f} | "
            f"{r['relative_f1_drop_pct']:<+7.2f}%"
        )
        print(row)
    print("=" * 105)

    # 4. Save results to data/privacy_benchmark_results.json
    out_file = data_dir / "privacy_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"[+] Exported benchmark metrics to {out_file}")

    return benchmark_results


if __name__ == "__main__":
    run_privacy_benchmark()
