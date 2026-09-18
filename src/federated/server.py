"""
Federated Server Module
Central orchestrator managing federated communication rounds, weight aggregation,
global consensus model evaluation, convergence tracking, and privacy preservation
(Differential Privacy & Secure Aggregation).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from .client import FederatedEdgeClient
from .strategies import FedAvgStrategy, FederatedStrategy

try:
    from ..privacy import DifferentialPrivacyEngine, PrivacyAccountant, SecureAggregationSimulator
except (ImportError, ValueError):
    from src.privacy import DifferentialPrivacyEngine, PrivacyAccountant, SecureAggregationSimulator


class FederatedServer:
    """
    Coordinates federated learning rounds across distributed edge factory clients.
    Supports Differential Privacy (DP-SGD / Model Perturbation) and Secure Aggregation (SecAgg).
    """

    def __init__(
        self,
        global_model: nn.Module,
        clients: List[FederatedEdgeClient],
        strategy: Optional[FederatedStrategy] = None,
        device: str = "cpu",
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.strategy = strategy or FedAvgStrategy()
        self.device = device
        self.round_history: List[Dict] = []

    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns clone of current global model weights."""
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

    def set_global_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Sets global model weights."""
        self.global_model.load_state_dict(parameters)

    def evaluate_global_model(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluates the current global model on every client's private test partition.
        Returns metrics per client and macro-averaged metrics.
        """
        global_weights = self.get_global_parameters()
        client_metrics = {}

        macro_f1 = []
        macro_recall = []
        macro_precision = []
        macro_roc_auc = []

        for client in self.clients:
            # Temporarily load global weights for evaluation
            client.set_parameters(global_weights)
            metrics = client.evaluate(client.test_loader)
            client_metrics[client.client_id] = metrics

            macro_f1.append(metrics["f1"])
            macro_recall.append(metrics["recall"])
            macro_precision.append(metrics["precision"])
            macro_roc_auc.append(metrics["roc_auc"])

        client_metrics["macro_average"] = {
            "f1": float(np.mean(macro_f1)),
            "recall": float(np.mean(macro_recall)),
            "precision": float(np.mean(macro_precision)),
            "roc_auc": float(np.mean(macro_roc_auc)),
        }
        return client_metrics

    def fit(
        self,
        num_rounds: int = 6,
        local_epochs: int = 2,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        enable_dp: bool = False,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.05,
        target_delta: float = 1e-5,
        enable_secagg: bool = False,
        secagg_magnitude: float = 10.0,
        verbose: bool = True,
    ) -> Dict:
        """
        Executes multi-round federated training with optional Differential Privacy and Secure Aggregation.
        
        Args:
            num_rounds: Number of federated communication rounds.
            local_epochs: Number of local training epochs on each edge client.
            checkpoint_dir: Directory to save best model checkpoint.
            enable_dp: Whether to apply Differential Privacy update clipping and noise.
            clip_norm: L2 clipping bound for Differential Privacy.
            noise_multiplier: Gaussian noise scale relative to clipping bound.
            target_delta: Target failure probability for DP accountant.
            enable_secagg: Whether to simulate Bonawitz pairwise masking Secure Aggregation.
            secagg_magnitude: Noise scale for pairwise blinding masks.
            verbose: Whether to print progress logs.
        """
        chk_path = Path(checkpoint_dir) if checkpoint_dir else None
        if chk_path:
            chk_path.mkdir(parents=True, exist_ok=True)

        best_macro_f1 = -1.0
        best_round = 0

        # Initialize privacy components if enabled
        accountant = None
        if enable_dp:
            accountant = PrivacyAccountant(target_delta=target_delta)
            for client in self.clients:
                client.dp_engine = DifferentialPrivacyEngine(
                    clip_norm=clip_norm,
                    noise_multiplier=noise_multiplier,
                    device=self.device,
                )

        secagg_engine = SecureAggregationSimulator(mask_magnitude=secagg_magnitude) if enable_secagg else None

        if verbose:
            print("=" * 80)
            print(f"STARTING FEDERATED LEARNING: {self.strategy.name}")
            print(f"Participating Edge Clients : {[c.client_id for c in self.clients]}")
            print(f"Total Federated Rounds     : {num_rounds}")
            print(f"Local Epochs per Round     : {local_epochs}")
            print(f"Differential Privacy (DP)  : {'ENABLED' if enable_dp else 'DISABLED'}")
            if enable_dp:
                print(f"  - Clip Norm (C)          : {clip_norm}")
                print(f"  - Noise Multiplier (s)   : {noise_multiplier}")
                print(f"  - Target Delta           : {target_delta}")
            print(f"Secure Aggregation (SecAgg): {'ENABLED' if enable_secagg else 'DISABLED'}")
            print("=" * 80)

        for round_idx in range(1, num_rounds + 1):
            global_params = self.get_global_parameters()

            client_weights = []
            client_samples = []
            client_train_losses = []
            client_updates_dict = {}

            # Step 1 & 2: Local Client Training & Parameter Extraction
            for client in self.clients:
                client.set_parameters(global_params)
                train_metrics = client.train_round(local_epochs=local_epochs)

                if enable_dp:
                    priv_params, _ = client.get_privatized_parameters(global_params)
                    client_weights.append(priv_params)
                    client_updates_dict[client.client_id] = priv_params
                else:
                    raw_params = client.get_parameters()
                    client_weights.append(raw_params)
                    client_updates_dict[client.client_id] = raw_params

                client_samples.append(client.num_samples)
                client_train_losses.append(train_metrics["train_loss"])

            # Step 3: Privacy Accounting (if DP active)
            current_eps = float("inf")
            if enable_dp and accountant is not None:
                current_eps, _ = accountant.step(noise_multiplier)

            # Step 4: Server Aggregation (SecAgg or Strategy)
            secagg_audit = None
            if enable_secagg and secagg_engine is not None:
                weights_map = {c.client_id: float(c.num_samples) for c in self.clients}
                masked_updates, secagg_audit = secagg_engine.mask_client_updates(
                    client_updates_dict,
                    client_weights=weights_map,
                    round_num=round_idx,
                )
                aggregated_params, _ = secagg_engine.unmask_and_aggregate(masked_updates)
            else:
                aggregated_params = self.strategy.aggregate(client_weights, client_samples)

            self.set_global_parameters(aggregated_params)

            # Step 5: Evaluate Global Consensus Model
            eval_results = self.evaluate_global_model()
            macro = eval_results["macro_average"]

            round_record = {
                "round": round_idx,
                "strategy": self.strategy.name,
                "avg_train_loss": float(np.mean(client_train_losses)),
                "macro_f1": macro["f1"],
                "macro_recall": macro["recall"],
                "macro_precision": macro["precision"],
                "macro_roc_auc": macro["roc_auc"],
                "dp_enabled": enable_dp,
                "epsilon": float(current_eps) if current_eps != float("inf") else "inf",
                "delta": target_delta if enable_dp else 0.0,
                "secagg_enabled": enable_secagg,
                "client_metrics": eval_results,
            }
            if secagg_audit:
                round_record["secagg_audit"] = secagg_audit

            self.round_history.append(round_record)

            if verbose:
                c1_f1 = eval_results["client_1"]["f1"]
                c2_f1 = eval_results["client_2"]["f1"]
                c3_f1 = eval_results["client_3"]["f1"]
                dp_info = f" | DP: eps={current_eps:.2f}, delta={target_delta}" if enable_dp else ""
                secagg_info = " | SecAgg: Active" if enable_secagg else ""
                print(
                    f"[Round {round_idx:02d}/{num_rounds:02d}] "
                    f"Macro F1: {macro['f1']:.4f} | "
                    f"C1 F1: {c1_f1:.4f}, C2 F1: {c2_f1:.4f}, C3 F1: {c3_f1:.4f} | "
                    f"Recall: {macro['recall']:.4f} | "
                    f"ROC-AUC: {macro['roc_auc']:.4f}"
                    f"{dp_info}{secagg_info}"
                )

            # Step 6: Save best global checkpoint
            if macro["f1"] > best_macro_f1:
                best_macro_f1 = macro["f1"]
                best_round = round_idx
                if chk_path:
                    save_name = "global_federated_dp_best.pt" if enable_dp else "global_federated_best.pt"
                    save_file = chk_path / save_name
                    torch.save(self.global_model.state_dict(), save_file)

        if verbose:
            print("-" * 80)
        last_round = self.round_history[-1] if self.round_history else {}
        client_metrics = last_round.get("client_metrics", {})
        final_eval = {
            "macro_precision": last_round.get("macro_precision", 0.0),
            "macro_recall": last_round.get("macro_recall", 0.0),
            "macro_f1": last_round.get("macro_f1", 0.0),
            "client_1": client_metrics.get("client_1", {}),
            "client_2": client_metrics.get("client_2", {}),
            "client_3": client_metrics.get("client_3", {}),
        }

        return {
            "best_round": best_round,
            "best_macro_f1": best_macro_f1,
            "final_epsilon": current_eps if enable_dp else float("inf"),
            "history": self.round_history,
            "final_eval": final_eval,
        }
