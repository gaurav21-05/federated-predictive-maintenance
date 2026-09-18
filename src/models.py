"""
Local Edge Neural Network Architectures Module
Implements 1D-CNN + Bi-LSTM (Temporal Network) and EdgeMLP with
parameter extraction hooks for Federated Learning aggregation.
"""

from typing import Dict, List, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class CNNLSTMPredictor(nn.Module):
        """
        Temporal Deep Learning Predictive Maintenance Model:
        - 1D-CNN extracts local transient vibration and electrical anomalies.
        - Bidirectional LSTM captures long-term thermal inertia and degradation progression.
        - Dense classification head outputs failure prediction logits.
        """

        def __init__(
            self,
            num_features: int = 9,
            window_size: int = 30,
            conv_filters: int = 32,
            kernel_size: int = 3,
            lstm_hidden: int = 48,
            lstm_layers: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.num_features = num_features
            self.window_size = window_size

            # 1. 1D Convolutional feature extractor
            self.conv1 = nn.Conv1d(
                in_channels=num_features,
                out_channels=conv_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.bn1 = nn.BatchNorm1d(conv_filters)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(kernel_size=2)  # Halves window length

            # 2. Bidirectional LSTM temporal aggregator
            self.lstm = nn.LSTM(
                input_size=conv_filters,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )

            # 3. Fully-Connected classification head
            self.fc1 = nn.Linear(lstm_hidden * 2, 32)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Input sequence tensor of shape (batch_size, window_size, num_features)
            Returns:
                logits: Tensor of shape (batch_size, 1)
            """
            # Conv1d expects (batch_size, in_channels, sequence_length)
            x_conv = x.permute(0, 2, 1)
            h = self.conv1(x_conv)
            h = self.bn1(h)
            h = self.relu(h)
            h = self.pool(h)

            # Permute back for LSTM: (batch_size, pooled_length, conv_filters)
            h_lstm_in = h.permute(0, 2, 1)
            lstm_out, _ = self.lstm(h_lstm_in)

            # Take final time step output from bidirectional LSTM
            last_step = lstm_out[:, -1, :]

            # Classification
            out = self.fc1(last_step)
            out = self.relu(out)
            out = self.dropout(out)
            logits = self.fc2(out)
            return logits

        def get_weights(self) -> Dict[str, np.ndarray]:
            """Extracts model parameters as NumPy arrays for Federated Learning."""
            return {k: v.cpu().detach().numpy() for k, v in self.state_dict().items()}

        def set_weights(self, weights_dict: Dict[str, np.ndarray]):
            """Loads aggregated weights into the model state_dict."""
            state_dict = {k: torch.tensor(v) for k, v in weights_dict.items()}
            self.load_state_dict(state_dict)


    class EdgeMLPPredictor(nn.Module):
        """
        Lightweight feedforward network operating on flattened temporal windows.
        Designed for resource-constrained IIoT edge gateways and embedded microcontrollers.
        """

        def __init__(self, num_features: int = 9, window_size: int = 30, hidden_dim: int = 64, dropout: float = 0.2):
            super().__init__()
            in_features = num_features * window_size

            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            return self.net(x_flat)

        def get_weights(self) -> Dict[str, np.ndarray]:
            return {k: v.cpu().detach().numpy() for k, v in self.state_dict().items()}

        def set_weights(self, weights_dict: Dict[str, np.ndarray]):
            state_dict = {k: torch.tensor(v) for k, v in weights_dict.items()}
            self.load_state_dict(state_dict)

else:
    class CNNLSTMPredictor:
        pass

    class EdgeMLPPredictor:
        pass
