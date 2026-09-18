"""
Federated Learning Package
Privacy-preserving distributed training for industrial IoT predictive maintenance.
Provides FedAvg, FedProx, edge client nodes, and central aggregation coordinator.
"""

from .client import FederatedEdgeClient
from .strategies import FedAvgStrategy, FedProxStrategy, FederatedStrategy
from .server import FederatedServer

__all__ = [
    "FederatedEdgeClient",
    "FederatedStrategy",
    "FedAvgStrategy",
    "FedProxStrategy",
    "FederatedServer",
]
