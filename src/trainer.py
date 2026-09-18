"""
Local Edge Model Trainer and Evaluator Module
Coordinates local client training loops, metric computations, early stopping,
and checkpoint serialization for edge AI predictive maintenance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class LocalEdgeTrainer:
    """Trains and evaluates predictive maintenance models on isolated client partitions."""

    def __init__(
        self,
        model,
        client_id: str,
        loss_fn=None,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.client_id = client_id
        self.loss_fn = loss_fn
        self.device = device

        if HAS_TORCH:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

    def train_epoch(self, dataloader: "DataLoader") -> float:
        """Trains model for one epoch over the client's local dataset."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.loss_fn(logits, y_batch)
            loss.backward()

            # Gradient clipping to stabilize recurrent and convolutional gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    def evaluate(self, dataloader: "DataLoader") -> Dict[str, float]:
        """Evaluates model performance and computes classification metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = self.loss_fn(logits, y_batch)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                targets = y_batch.view(-1).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_probs.extend(probs)
                all_targets.extend(targets)
                all_preds.extend(preds)

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # Compute metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ROC-AUC & PR-AUC
        try:
            roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        except ValueError:
            roc_auc = 0.5

        try:
            pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        except ValueError:
            pr_auc = 0.0

        return {
            "loss": total_loss / max(1, num_batches),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        }

    def fit(
        self,
        train_loader: "DataLoader",
        val_loader: "DataLoader",
        epochs: int = 10,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Dict:
        """Runs the full training loop with early stopping."""
        best_f1 = -1.0
        best_epoch = 0
        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_recall": []}

        chk_path = Path(checkpoint_dir) if checkpoint_dir else None
        if chk_path:
            chk_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            val_f1 = val_metrics["f1"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_f1)
            history["val_recall"].append(val_metrics["recall"])

            if verbose:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val F1: {val_f1:.4f} | "
                    f"Val Recall: {val_metrics['recall']:.4f} | "
                    f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
                )

            # Checkpoint best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                if chk_path:
                    save_file = chk_path / f"{self.client_id}_best_model.pt"
                    torch.save(self.model.state_dict(), save_file)

        return {
            "best_epoch": best_epoch,
            "best_f1": best_f1,
            "history": history,
        }

    def test(self, test_loader: "DataLoader") -> Dict[str, Union[float, List]]:
        """Evaluates final model on held-out test data and computes confusion matrix."""
        metrics = self.evaluate(test_loader)

        self.model.eval()
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = self.model(X_batch.to(self.device))
                probs = torch.sigmoid(logits).view(-1).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_preds.extend(preds)
                all_targets.extend(y_batch.view(-1).cpu().numpy())

        cm = confusion_matrix(all_targets, all_preds).tolist()
        metrics["confusion_matrix"] = cm
        return metrics


def create_dataloader(
    X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True
) -> "DataLoader":
    """Wraps NumPy sequence arrays in a PyTorch DataLoader."""
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
