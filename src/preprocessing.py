"""
Time-Series Preprocessing & Feature Engineering Module
Transforms raw industrial IoT sensor streams into sliding-window sequence tensors
for edge AI predictive maintenance models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

SENSOR_FEATURES = [
    "temperature",
    "humidity",
    "vibration",
    "current",
    "rpm",
    "pressure",
]


class TimeSeriesPreprocessor:
    """
    Handles temporal sliding window generation, domain feature engineering,
    and client-isolated standard scaling for edge AI predictive maintenance.
    """

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 5,
        lookahead_horizon: int = 12,
        feature_cols: Optional[List[str]] = None,
    ):
        """
        Args:
            window_size: Number of consecutive time steps per sequence window (default: 30 = 5 min at 10s step).
            stride: Step size between consecutive sliding windows (default: 5).
            lookahead_horizon: Future time steps to monitor for failure (default: 12 = 2 min early warning).
            feature_cols: List of sensor columns to include in input sequences.
        """
        self.window_size = window_size
        self.stride = stride
        self.lookahead_horizon = lookahead_horizon
        self.feature_cols = feature_cols or SENSOR_FEATURES

        # Scaler parameters (mean and standard deviation per feature)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments raw sensor readings with temporal physical derivative features:
        - Thermal gradient (dT/dt)
        - Vibration rate of change (dV/dt)
        - Current rolling RMS (motor electrical strain)
        """
        data = df.copy()

        # 1. Thermal dissipation gradient: dT / dt (spindle friction inception)
        data["temp_gradient"] = data["temperature"].diff().fillna(0.0)

        # 2. Vibration derivative: dV / dt (shock vibration acceleration)
        data["vib_gradient"] = data["vibration"].diff().fillna(0.0)

        # 3. Current rolling moving average (5-step window)
        data["current_rolling_mean"] = data["current"].rolling(window=5, min_periods=1).mean()

        return data

    def create_sliding_windows(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts overlapping sliding windows and constructs lookahead predictive maintenance targets.

        Returns:
            X: Array of shape (num_windows, window_size, num_features)
            y: Binary labels array of shape (num_windows,) where 1 indicates failure or imminent failure
        """
        # Engineer temporal features
        eng_df = self.engineer_features(df)
        all_cols = self.feature_cols + ["temp_gradient", "vib_gradient", "current_rolling_mean"]

        features_data = eng_df[all_cols].values
        failure_data = eng_df["failure"].values

        total_steps = len(df)
        X_list = []
        y_list = []

        # Iterate sliding window
        max_start = total_steps - self.window_size - self.lookahead_horizon
        for start_idx in range(0, max_start, self.stride):
            end_idx = start_idx + self.window_size
            horizon_end = end_idx + self.lookahead_horizon

            # Sequence tensor (window_size, num_features)
            window_features = features_data[start_idx:end_idx]

            # Predictive Maintenance Target:
            # 1 if machine is failing inside the window OR will fail within the lookahead horizon
            window_failure = (failure_data[start_idx:end_idx].max() > 0)
            horizon_failure = (failure_data[end_idx:horizon_end].max() > 0)

            label = 1 if (window_failure or horizon_failure) else 0

            X_list.append(window_features)
            y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        return X, y

    def fit_scaler(self, X_train: np.ndarray):
        """Fit feature normalization strictly on the training partition."""
        # Reshape to (N * W, D) to compute mean and std across time and samples
        flat = X_train.reshape(-1, X_train.shape[-1])
        self.mean_ = np.mean(flat, axis=0)
        self.std_ = np.std(flat, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0.0] = 1.0

    def transform_scaler(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted normalization to a sequence tensor."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit_scaler() first.")
        # Standardize: (X - mean) / std
        return (X - self.mean_) / self.std_

    def prepare_client_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Splits sequential time-series into chronologically ordered Train, Val, Test partitions,
        applies sliding windowing, fits scaler on train, and normalizes sequences.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        # Create windows
        X_train, y_train = self.create_sliding_windows(train_df)
        X_val, y_val = self.create_sliding_windows(val_df)
        X_test, y_test = self.create_sliding_windows(test_df)

        # Fit scaler on training split ONLY
        self.fit_scaler(X_train)

        # Normalize all splits
        X_train = self.transform_scaler(X_train)
        X_val = self.transform_scaler(X_val)
        X_test = self.transform_scaler(X_test)

        # Calculate positive class weight for imbalanced loss
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        pos_weight = float(num_neg / max(1, num_pos))

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "pos_weight": pos_weight,
            "num_features": X_train.shape[-1],
            "window_size": self.window_size,
        }
