"""
Sensor and Factory Configuration Module
Defines operational parameters, baseline distributions, and physical bounds
for different industrial IoT clients (factories/machines) to simulate non-IID distributions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


class OperatingCondition(str, Enum):
    """Machine operational states."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    ABNORMAL = "ABNORMAL"


# Target feature columns as specified in requirements
SENSOR_COLUMNS = [
    "timestamp",
    "machine_id",
    "temperature",
    "humidity",
    "vibration",
    "current",
    "rpm",
    "pressure",
    "failure",
]


@dataclass
class SensorBaseline:
    """Nominal mean and Gaussian standard deviation under normal state."""
    mean: float
    std: float
    min_val: float
    max_val: float
    warning_mult: float  # Multiplier or shift during warning state
    abnormal_mult: float  # Multiplier or shift during abnormal state


@dataclass
class FactoryProfile:
    """
    Configuration profile representing an industrial factory/machine client.
    Each client has a distinct physical profile to represent non-IID data distributions.
    """
    client_id: str
    machine_id: str
    machine_type: str
    location: str
    sampling_rate_seconds: int
    temperature: SensorBaseline
    humidity: SensorBaseline
    vibration: SensorBaseline
    current: SensorBaseline
    rpm: SensorBaseline
    pressure: SensorBaseline
    # Failure hazard weights for physics engine
    weight_temp: float = 0.25
    weight_vib: float = 0.30
    weight_current: float = 0.15
    weight_rpm: float = 0.15
    weight_pressure: float = 0.15
    base_degradation_rate: float = 0.00012


# Factory 1: High-Speed CNC Milling Center (High RPM, Thermal & Bearing Vibration critical)
CLIENT_1_PROFILE = FactoryProfile(
    client_id="client_1",
    machine_id="CNC_MILL_01",
    machine_type="5-Axis High-Speed CNC Milling Machine",
    location="Factory-A (Precision Machining Plant)",
    sampling_rate_seconds=10,
    temperature=SensorBaseline(mean=62.0, std=2.5, min_val=30.0, max_val=110.0, warning_mult=1.18, abnormal_mult=1.45),
    humidity=SensorBaseline(mean=45.0, std=3.0, min_val=20.0, max_val=80.0, warning_mult=1.05, abnormal_mult=1.12),
    vibration=SensorBaseline(mean=1.20, std=0.15, min_val=0.2, max_val=9.5, warning_mult=2.2, abnormal_mult=4.5),
    current=SensorBaseline(mean=18.5, std=1.2, min_val=5.0, max_val=48.0, warning_mult=1.35, abnormal_mult=1.9),
    rpm=SensorBaseline(mean=3850.0, std=45.0, min_val=1000.0, max_val=5000.0, warning_mult=0.92, abnormal_mult=0.72),
    pressure=SensorBaseline(mean=6.2, std=0.25, min_val=2.0, max_val=12.0, warning_mult=0.88, abnormal_mult=0.60),
    weight_temp=0.30,
    weight_vib=0.35,
    weight_current=0.15,
    weight_rpm=0.10,
    weight_pressure=0.10,
    base_degradation_rate=0.00010,
)

# Factory 2: Heavy Hydraulic Stamping Press (High Hydraulic Pressure, High Shock Vibration)
CLIENT_2_PROFILE = FactoryProfile(
    client_id="client_2",
    machine_id="HYDR_PRESS_02",
    machine_type="Heavy-Duty Hydraulic Stamping Press",
    location="Factory-B (Metal Forming & Stamping Works)",
    sampling_rate_seconds=10,
    temperature=SensorBaseline(mean=51.5, std=2.0, min_val=25.0, max_val=98.0, warning_mult=1.22, abnormal_mult=1.55),
    humidity=SensorBaseline(mean=38.0, std=2.5, min_val=15.0, max_val=75.0, warning_mult=1.04, abnormal_mult=1.10),
    vibration=SensorBaseline(mean=2.40, std=0.25, min_val=0.5, max_val=12.0, warning_mult=1.9, abnormal_mult=3.8),
    current=SensorBaseline(mean=34.0, std=2.1, min_val=10.0, max_val=75.0, warning_mult=1.40, abnormal_mult=2.1),
    rpm=SensorBaseline(mean=1120.0, std=20.0, min_val=300.0, max_val=1800.0, warning_mult=0.90, abnormal_mult=0.65),
    pressure=SensorBaseline(mean=185.0, std=6.5, min_val=50.0, max_val=280.0, warning_mult=1.18, abnormal_mult=1.38),
    weight_temp=0.15,
    weight_vib=0.25,
    weight_current=0.15,
    weight_rpm=0.10,
    weight_pressure=0.35,
    base_degradation_rate=0.00012,
)

# Factory 3: Industrial Turbine & Compressor (High Humidity sensitivity, Current anomalies, Pneumatic loss)
CLIENT_3_PROFILE = FactoryProfile(
    client_id="client_3",
    machine_id="TURB_COMP_03",
    machine_type="Rotary Screw Air Compressor & Micro-Turbine",
    location="Factory-C (Chemical Processing & Utilities Facility)",
    sampling_rate_seconds=10,
    temperature=SensorBaseline(mean=57.0, std=2.2, min_val=28.0, max_val=105.0, warning_mult=1.20, abnormal_mult=1.50),
    humidity=SensorBaseline(mean=72.0, std=4.5, min_val=35.0, max_val=98.0, warning_mult=1.15, abnormal_mult=1.28),
    vibration=SensorBaseline(mean=1.65, std=0.18, min_val=0.3, max_val=10.0, warning_mult=2.0, abnormal_mult=4.0),
    current=SensorBaseline(mean=27.5, std=1.8, min_val=8.0, max_val=60.0, warning_mult=1.45, abnormal_mult=2.2),
    rpm=SensorBaseline(mean=3150.0, std=35.0, min_val=900.0, max_val=4200.0, warning_mult=0.94, abnormal_mult=0.75),
    pressure=SensorBaseline(mean=8.8, std=0.40, min_val=2.5, max_val=16.0, warning_mult=0.82, abnormal_mult=0.55),
    weight_temp=0.20,
    weight_vib=0.20,
    weight_current=0.30,
    weight_rpm=0.15,
    weight_pressure=0.15,
    base_degradation_rate=0.00011,
)


CLIENT_PROFILES: Dict[str, FactoryProfile] = {
    "client_1": CLIENT_1_PROFILE,
    "client_2": CLIENT_2_PROFILE,
    "client_3": CLIENT_3_PROFILE,
}


class ClientConfig:
    """Helper to access factory profiles."""
    @staticmethod
    def get(client_id: str) -> FactoryProfile:
        if client_id not in CLIENT_PROFILES:
            raise ValueError(f"Unknown client_id: '{client_id}'. Supported clients: {list(CLIENT_PROFILES.keys())}")
        return CLIENT_PROFILES[client_id]

    @staticmethod
    def all() -> Dict[str, FactoryProfile]:
        return CLIENT_PROFILES
