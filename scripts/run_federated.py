#!/usr/bin/env python3
"""
Main Federated Learning Execution Script
Coordinates multi-round privacy-preserving collaborative model training across
edge industrial factories (client_1, client_2, client_3) using FedAvg or FedProx,
with optional Differential Privacy (DP-SGD / perturbation) and Secure Aggregation (SecAgg).
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.client import FederatedEdgeClient
from src.federated.server import FederatedServer
from src.federated.strategies import FedAvgStrategy, FedProxStrategy
from src.models import CNNLSTMPredictor
from src.preprocessing import TimeSeriesPreprocessor
from src.trainer import create_dataloader


def run_federated_training(
    rounds: int = 5,
    local_epochs: int = 2,
    strategy_name: str = "fedprox",
    mu: float = 0.01,
    batch_size: int = 64,
    lr: float = 0.001,
    enable_dp: bool = False,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.05,
    delta: float = 1e-5,
    enable_secagg: bool = False,
    secagg_magnitude: float = 10.0,
    window_size: int = 30,
    stride: int = 5,
    lookahead_horizon: int = 12,
    checkpoint_dir: str = "models/checkpoints",
):
    data_dir = PROJECT_ROOT / "data"
    chk_path = PROJECT_ROOT / checkpoint_dir
    chk_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FEDERATED LEARNING ORCHESTRATOR FOR INDUSTRIAL PREDICTIVE MAINTENANCE")
    print("Privacy-Preserving Collaborative Training across 3 Industrial Factories")
    print("=" * 80)
    print(f"Federated Strategy     : {strategy_name.upper()}")
    print(f"Proximal Penalty (mu)  : {mu if strategy_name == 'fedprox' else 0.0}")
    print(f"Federated Rounds       : {rounds}")
    print(f"Local Epochs per Round : {local_epochs}")
    print(f"Differential Privacy   : {'ENABLED' if enable_dp else 'DISABLED'}")
    if enable_dp:
        print(f"  - Clipping Norm (C)  : {clip_norm}")
        print(f"  - Noise Multiplier   : {noise_multiplier}")
        print(f"  - Target Delta       : {delta}")
    print(f"Secure Aggregation     : {'ENABLED' if enable_secagg else 'DISABLED'}")
    print(f"Checkpoint Output      : {chk_path}")
    print("-" * 80)

    # 1. Instantiate edge clients with private data loaders
    clients = []
    client_ids = ["client_1", "client_2", "client_3"]
    preprocessor = TimeSeriesPreprocessor(
        window_size=window_size,
        stride=stride,
        lookahead_horizon=lookahead_horizon,
    )

    print("[*] Initializing isolated edge factory nodes...")
    num_features = 9

    for cid in client_ids:
        csv_file = data_dir / cid / f"{cid}_sensor_data.csv"
        df = pd.read_csv(csv_file)
        splits = preprocessor.prepare_client_dataset(df)
        num_features = splits["num_features"]

        train_loader = create_dataloader(splits["X_train"], splits["y_train"], batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(splits["X_val"], splits["y_val"], batch_size=batch_size, shuffle=False)
        test_loader = create_dataloader(splits["X_test"], splits["y_test"], batch_size=batch_size, shuffle=False)

        client_model = CNNLSTMPredictor(num_features=num_features, window_size=window_size)

        client = FederatedEdgeClient(
            client_id=cid,
            model=client_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=lr,
            mu=mu if strategy_name.lower() == "fedprox" else 0.0,
        )
        clients.append(client)
        print(f"    [+] Registered {cid} (Train sequences: {splits['X_train'].shape[0]:,})")

    # 2. Instantiate global consensus model and strategy
    global_model = CNNLSTMPredictor(num_features=num_features, window_size=window_size)

    if strategy_name.lower() == "fedprox":
        strategy = FedProxStrategy(mu=mu)
    else:
        strategy = FedAvgStrategy()

    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        strategy=strategy,
    )

    # 3. Execute federated rounds
    history = server.fit(
        num_rounds=rounds,
        local_epochs=local_epochs,
        checkpoint_dir=chk_path,
        enable_dp=enable_dp,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        target_delta=delta,
        enable_secagg=enable_secagg,
        secagg_magnitude=secagg_magnitude,
        verbose=True,
    )

    # 4. Final Global Evaluation across all 3 client test partitions
    print("\n" + "=" * 80)
    print("FINAL GLOBAL FEDERATED MODEL EVALUATION (ON HELD-OUT TEST SPLITS)")
    print("=" * 80)
    final_eval = server.evaluate_global_model()
    print(f"{'Client / Factory':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 75)

    for cid in client_ids:
        m = final_eval[cid]
        print(f"{cid:<25} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['roc_auc']:<10.4f}")

    macro = final_eval["macro_average"]
    print("-" * 75)
    print(f"{'MACRO AVERAGE':<25} | {macro['precision']:<10.4f} | {macro['recall']:<10.4f} | {macro['f1']:<10.4f} | {macro['roc_auc']:<10.4f}")
    if enable_dp:
        print(f"{'PRIVACY GUARANTEE':<25} | (epsilon={history['final_epsilon']:.2f}, delta={delta})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Federated Predictive Maintenance Training")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds (default: 5)")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per client per round (default: 2)")
    parser.add_argument("--strategy", type=str, default="fedprox", choices=["fedavg", "fedprox"], help="Aggregation strategy")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal regularization parameter (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="DP clipping norm C (default: 1.0)")
    parser.add_argument("--noise-multiplier", type=float, default=0.05, help="DP noise multiplier sigma (default: 0.05)")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target failure probability delta (default: 1e-5)")
    parser.add_argument("--secagg", action="store_true", help="Enable Secure Aggregation pairwise masking")
    parser.add_argument("--secagg-magnitude", type=float, default=10.0, help="Scale of SecAgg blinding masks")
    args = parser.parse_args()

    run_federated_training(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        strategy_name=args.strategy,
        mu=args.mu,
        batch_size=args.batch_size,
        lr=args.lr,
        enable_dp=args.dp,
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        delta=args.delta,
        enable_secagg=args.secagg,
        secagg_magnitude=args.secagg_magnitude,
    )


if __name__ == "__main__":
    main()
