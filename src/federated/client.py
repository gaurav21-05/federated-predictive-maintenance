"""
Federated Edge Client Module
Represents an on-premise industrial factory node participating in federated learning.
Trains locally on private sensor telemetry without sending raw data to the central server.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..loss import BinaryFocalLoss
from ..trainer import LocalEdgeTrainer


class FederatedEdgeClient:
    """
    Simulates an isolated factory edge node in a Federated Learning ecosystem.
    Holds local training and evaluation data partitions.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: Optional[nn.Module] = None,
        lr: float = 0.001,
        mu: float = 0.0,  # FedProx proximal coefficient (0.0 = FedAvg)
        device: str = "cpu",
        dp_engine: Optional[object] = None,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn or BinaryFocalLoss(alpha=0.75, gamma=2.0)
        self.lr = lr
        self.mu = mu
        self.device = device
        self.dp_engine = dp_engine

        self.num_samples = len(train_loader.dataset)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Receives global consensus model weights and loads them into local model."""
        self.model.load_state_dict(parameters)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns current local model weights to the federated server."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def get_privatized_parameters(
        self,
        global_parameters: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Returns DP-privatized model parameters (clipped + Gaussian noise perturbed).
        If dp_engine is not set, returns raw parameters without noise.
        """
        raw_params = self.get_parameters()
        if self.dp_engine is None:
            return raw_params, {"dp_applied": 0.0}

        return self.dp_engine.clip_and_perturb_update(
            local_params=raw_params,
            global_params=global_parameters,
        )

    def train_round(self, local_epochs: int = 2) -> Dict[str, float]:
        """
        Trains the local model for a specified number of local epochs.
        If mu > 0, applies FedProx proximal regularization penalty:
        Loss_total = Loss_focal + (mu / 2) * sum(||w_local - w_global||^2)
        """
        self.model.train()

        # Cache reference global weights for FedProx proximal regularization
        global_params = None
        if self.mu > 0.0:
            global_params = {k: v.clone().detach().to(self.device) for k, v in self.model.state_dict().items()}

        total_loss = 0.0
        total_batches = 0

        for epoch in range(local_epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.loss_fn(logits, y_batch)

                # FedProx Proximal Regularization Penalty
                if self.mu > 0.0 and global_params is not None:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_params:
                            proximal_term += (param - global_params[name]).pow(2).sum()
                    loss = loss + (self.mu / 2.0) * proximal_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)

        # Evaluate on local validation set
        val_metrics = self.evaluate(self.val_loader)
        val_metrics["train_loss"] = avg_loss
        return val_metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates model performance on local validation or test partition."""
        trainer = LocalEdgeTrainer(
            model=self.model,
            client_id=self.client_id,
            loss_fn=self.loss_fn,
            device=self.device,
        )
        return trainer.evaluate(dataloader)
