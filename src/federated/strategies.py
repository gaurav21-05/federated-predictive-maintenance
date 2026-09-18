"""
Federated Aggregation Strategies Module
Implements FedAvg (McMahan et al.) and FedProx (Li et al.) aggregation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch


class FederatedStrategy(ABC):
    """Abstract base class for federated aggregation algorithms."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_sample_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates model parameter dictionaries from multiple edge clients.
        Args:
            client_weights: List of state_dict dictionaries from participating clients.
            client_sample_sizes: Number of local training sequences per client.
        Returns:
            aggregated_weights: Consolidated global model state_dict.
        """
        pass


class FedAvgStrategy(FederatedStrategy):
    """
    Standard Federated Averaging (FedAvg).
    Computes sample-weighted linear combination of client model weights:
    W_{global} = sum_{k=1}^K (n_k / N) * W_k
    """

    def __init__(self):
        super().__init__(name="FedAvg")

    def aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_sample_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        if not client_weights or not client_sample_sizes:
            raise ValueError("client_weights and client_sample_sizes must not be empty.")

        total_samples = sum(client_sample_sizes)
        weight_keys = client_weights[0].keys()
        aggregated_dict: Dict[str, torch.Tensor] = {}

        for key in weight_keys:
            # Initialize with zeros in the same shape and float dtype
            accum = torch.zeros_like(client_weights[0][key], dtype=torch.float32)
            for weights, n_k in zip(client_weights, client_sample_sizes):
                fraction = float(n_k) / float(total_samples)
                accum += weights[key].to(accum.device).float() * fraction

            # Preserve original parameter dtype (e.g., int64 for batchnorm num_batches_tracked)
            if client_weights[0][key].dtype in (torch.int64, torch.long):
                aggregated_dict[key] = accum.round().to(client_weights[0][key].dtype)
            else:
                aggregated_dict[key] = accum.to(client_weights[0][key].dtype)

        return aggregated_dict


class FedProxStrategy(FederatedStrategy):
    """
    FedProx Aggregation Strategy.
    Designed specifically for heterogeneous, non-IID industrial IoT edge environments.
    Pairs with client-side proximal regularization: (mu / 2) * ||w - w^t||^2.
    """

    def __init__(self, mu: float = 0.01):
        super().__init__(name=f"FedProx(mu={mu})")
        self.mu = mu

    def aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_sample_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        # FedProx aggregates via sample-weighted consensus, while client drift is bounded by mu
        total_samples = sum(client_sample_sizes)
        weight_keys = client_weights[0].keys()
        aggregated_dict: Dict[str, torch.Tensor] = {}

        for key in weight_keys:
            accum = torch.zeros_like(client_weights[0][key], dtype=torch.float32)
            for weights, n_k in zip(client_weights, client_sample_sizes):
                fraction = float(n_k) / float(total_samples)
                accum += weights[key].to(accum.device).float() * fraction

            if client_weights[0][key].dtype in (torch.int64, torch.long):
                aggregated_dict[key] = accum.round().to(client_weights[0][key].dtype)
            else:
                aggregated_dict[key] = accum.to(client_weights[0][key].dtype)

        return aggregated_dict
