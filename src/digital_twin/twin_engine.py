"""
Digital Twin Synchronization Engine Module
Maintains real-time virtual state, physical sub-health indices (vibration, thermal,
electrical, pressure), composite Machine Health Index (MHI), and fault localization.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from ..config import CLIENT_PROFILES, FactoryProfile, OperatingCondition


class MachineDigitalTwin:
    """
    High-fidelity virtual replica of an industrial edge machine.
    Synchronizes with live telemetry streams and tracks physical degradation.
    """

    # Subsystem weights by machine type
    SUBSYSTEM_WEIGHTS = {
        "client_1": {"vibration": 0.35, "thermal": 0.30, "electrical": 0.20, "pressure": 0.15},
        "client_2": {"pressure": 0.40, "vibration": 0.25, "thermal": 0.20, "electrical": 0.15},
        "client_3": {"electrical": 0.35, "pressure": 0.30, "thermal": 0.20, "vibration": 0.15},
    }

    FAULT_DESCRIPTIONS = {
        "vibration": "Spindle Bearing Race Fatigue & Rotor Imbalance",
        "thermal": "Thermal Dissipation Bottleneck & Friction Overheating",
        "pressure": "Hydraulic Surge / Pneumatic Seal Cavitation",
        "electrical": "Motor Stator Overcurrent & Excessive Cutting Load",
    }

    def __init__(
        self,
        client_id: str,
        profile: Optional[FactoryProfile] = None,
        window_size: int = 30,
        model: Optional[nn.Module] = None,
        preprocessor: Optional[object] = None,
        device: str = "cpu",
    ):
        """
        Args:
            client_id: Identifier of the edge factory node (client_1, client_2, client_3).
            profile: Factory physical profile. If None, loaded from CLIENT_PROFILES.
            window_size: Sliding window size for temporal inference (default: 30 ticks = 5 min).
            model: Optional PyTorch predictive maintenance model (e.g. CNNLSTMPredictor).
            preprocessor: Optional TimeSeriesPreprocessor for model input scaling.
            device: Compute device ('cpu' or 'cuda').
        """
        self.client_id = client_id
        self.profile = profile or CLIENT_PROFILES.get(client_id, CLIENT_PROFILES["client_1"])
        self.machine_name = getattr(self.profile, "machine_type", getattr(self.profile, "machine_id", "Industrial Machine"))
        self.window_size = window_size
        self.model = model.to(device) if model is not None else None
        self.preprocessor = preprocessor
        self.device = device

        self.weights = self.SUBSYSTEM_WEIGHTS.get(client_id, self.SUBSYSTEM_WEIGHTS["client_1"])

        # State storage
        self.buffer: Deque[Dict[str, float]] = deque(maxlen=window_size)
        self.total_ticks_processed = 0

        # Health indices (0.0 to 100.0%)
        self.subsystem_health = {
            "vibration": 100.0,
            "thermal": 100.0,
            "electrical": 100.0,
            "pressure": 100.0,
        }
        self.overall_health_index = 100.0
        self.operational_state = OperatingCondition.NORMAL.value
        self.failure_probability = 0.0
        self.primary_fault_subsystem = "None (Healthy)"
        self.latest_telemetry: Dict[str, float] = {}

    def _compute_subsystem_health(self, telemetry: Dict[str, float]) -> Dict[str, float]:
        """
        Computes physical sub-health scores (0-100%) based on deviation from nominal baselines.
        """
        p = self.profile
        temp = float(telemetry.get("temperature", p.temperature.mean))
        vib = float(telemetry.get("vibration", p.vibration.mean))
        current = float(telemetry.get("current", p.current.mean))
        pressure = float(telemetry.get("pressure", p.pressure.mean))

        # 1. Vibration Health (ISO 10816 scale relative to baseline and trip threshold)
        v0 = p.vibration.mean
        vtrip = p.vibration.mean + 6.0 * p.vibration.std
        v_excess = max(0.0, vib - v0)
        v_ratio = min(1.0, v_excess / max(1e-4, vtrip - v0))
        h_vib = max(0.0, min(100.0, 100.0 * (1.0 - v_ratio ** 1.5)))

        # 2. Thermal Health (Heat accumulation vs dissipation)
        t0 = p.temperature.mean
        tmax = p.temperature.mean + 5.5 * p.temperature.std
        t_excess = max(0.0, temp - t0)
        t_ratio = min(1.0, t_excess / max(1e-4, tmax - t0))
        h_therm = max(0.0, min(100.0, 100.0 * (1.0 - t_ratio ** 1.3)))

        # 3. Electrical Health (Current consumption vs nominal full-load amperage)
        i0 = p.current.mean
        imax = p.current.mean + 4.5 * p.current.std
        i_excess = max(0.0, current - i0)
        i_ratio = min(1.0, i_excess / max(1e-4, imax - i0))
        h_elec = max(0.0, min(100.0, 100.0 * (1.0 - i_ratio ** 1.4)))

        # 4. Pressure Health (Tolerance deviation from nominal operating pressure)
        p0 = p.pressure.mean
        p_dev = abs(pressure - p0)
        pmax = 5.0 * p.pressure.std
        p_ratio = min(1.0, p_dev / max(1e-4, pmax))
        h_press = max(0.0, min(100.0, 100.0 * (1.0 - p_ratio ** 1.3)))

        return {
            "vibration": float(h_vib),
            "thermal": float(h_therm),
            "electrical": float(h_elec),
            "pressure": float(h_press),
        }

    def _infer_failure_probability(self) -> float:
        """
        Runs neural model inference on the current sliding window buffer.
        """
        if self.model is None or len(self.buffer) < self.window_size:
            # Fallback heuristic based on Health Index
            return float(np.clip(1.0 - (self.overall_health_index / 100.0), 0.0, 1.0))

        # Extract features from buffer
        feature_keys = ["temperature", "humidity", "vibration", "current", "rpm", "pressure"]
        raw_matrix = np.array([[point[k] for k in feature_keys] for point in self.buffer], dtype=np.float32)

        # Apply scaling if preprocessor exists
        if self.preprocessor is not None and hasattr(self.preprocessor, "scalers") and self.client_id in self.preprocessor.scalers:
            scaler = self.preprocessor.scalers[self.client_id]
            norm_matrix = scaler.transform(raw_matrix)
        else:
            # Simple standard score normalization fallback
            means = raw_matrix.mean(axis=0, keepdims=True)
            stds = raw_matrix.std(axis=0, keepdims=True) + 1e-6
            norm_matrix = (raw_matrix - means) / stds

        # Compute derivative features (gradient of temp, vibration, rolling current)
        temp_grad = np.gradient(norm_matrix[:, 0], axis=0).reshape(-1, 1)
        vib_grad = np.gradient(norm_matrix[:, 2], axis=0).reshape(-1, 1)
        current_roll = pd_roll_mean(norm_matrix[:, 3], window=5).reshape(-1, 1)

        feat_tensor = np.hstack([norm_matrix, temp_grad, vib_grad, current_roll])
        x_in = torch.tensor(feat_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_in)
            prob = torch.sigmoid(logits).item()

        return float(prob)

    def ingest_telemetry_point(self, telemetry: Dict[str, float]) -> Dict:
        """
        Ingests a single time-step telemetry point and updates the Digital Twin virtual state.
        
        Args:
            telemetry: Sensor dictionary containing temperature, vibration, current, rpm, pressure, etc.
            
        Returns:
            Current synchronized state snapshot.
        """
        self.latest_telemetry = telemetry.copy()
        self.buffer.append(telemetry)
        self.total_ticks_processed += 1

        # 1. Update Subsystem Health
        self.subsystem_health = self._compute_subsystem_health(telemetry)

        # 2. Composite Machine Health Index (MHI)
        mhi = sum(self.weights[k] * self.subsystem_health[k] for k in self.weights)
        self.overall_health_index = float(np.clip(mhi, 0.0, 100.0))

        # 3. Model Failure Probability
        self.failure_probability = self._infer_failure_probability()

        # 4. Operational State Transitions
        if telemetry.get("failure", 0) == 1:
            self.operational_state = OperatingCondition.ABNORMAL.value
        elif self.overall_health_index < 50.0 or self.failure_probability > 0.70:
            self.operational_state = "CRITICAL"
        elif self.overall_health_index < 75.0 or self.failure_probability > 0.35:
            self.operational_state = OperatingCondition.WARNING.value
        else:
            self.operational_state = OperatingCondition.NORMAL.value

        # 5. Fault Localization (identify weakest component)
        weakest_subsystem = min(self.subsystem_health, key=self.subsystem_health.get)
        if self.subsystem_health[weakest_subsystem] < 75.0:
            self.primary_fault_subsystem = self.FAULT_DESCRIPTIONS.get(weakest_subsystem, weakest_subsystem)
        else:
            self.primary_fault_subsystem = "None (Healthy Operating Regime)"

        return self.get_state_snapshot()

    def get_state_snapshot(self) -> Dict:
        """Returns a serializable dictionary of the complete Digital Twin state."""
        return {
            "client_id": self.client_id,
            "machine_name": self.machine_name,
            "total_ticks_processed": self.total_ticks_processed,
            "buffer_fill_pct": (len(self.buffer) / self.window_size) * 100.0,
            "overall_health_index": round(self.overall_health_index, 2),
            "subsystem_health": {k: round(v, 2) for k, v in self.subsystem_health.items()},
            "operational_state": self.operational_state,
            "failure_probability": round(self.failure_probability, 4),
            "primary_fault_subsystem": self.primary_fault_subsystem,
            "latest_sensors": {
                k: round(float(v), 2)
                for k, v in self.latest_telemetry.items()
                if k not in ["timestamp", "machine_id"]
            },
        }


def pd_roll_mean(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Helper rolling mean calculation without requiring full pandas DataFrame."""
    res = np.empty_like(arr)
    for i in range(len(arr)):
        start_idx = max(0, i - window + 1)
        res[i] = np.mean(arr[start_idx : i + 1])
    return res
