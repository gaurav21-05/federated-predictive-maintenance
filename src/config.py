"""
Physical Hardware Node & Sensor Configuration Module.
Defines operational parameters, baseline bounds, and node configurations
for real Industrial IoT edge devices (ESP32 DevKit V1 + MPU6050 + DHT11 + ACS712).

Pure Python. No synthetic factories or dummy machines.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class OperationalState(str, Enum):
    """Real machine operational states."""
    OPTIMAL = "OPTIMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    OFFLINE = "OFFLINE"


# Real sensor columns matching ThingSpeak 8-channel feed
REAL_SENSOR_COLUMNS = [
    "timestamp",
    "entry_id",
    "temperature_c",
    "humidity_pct",
    "temp_secondary_c",
    "pressure_hpa",
    "vibration_g",
    "current_a",
    "esp32_anomaly",
    "esp32_alert",
    "status",
]


@dataclass
class HardwareNodeProfile:
    """Configuration profile for a physical IoT edge device."""
    node_id: str
    channel_id: str
    device_name: str
    description: str
    microcontroller: str
    sampling_interval_seconds: float
    nominal_vibration_g: float = 0.022
    nominal_temperature_c: float = 24.0
    nominal_humidity_pct: float = 65.0
    nominal_pressure_hpa: float = 994.5
    nominal_current_a: Optional[float] = None


# Configurable registry of physical hardware edge devices.
# Adding more ESP32s later requires only adding entries to this dictionary.
HARDWARE_NODES: Dict[str, HardwareNodeProfile] = {
    "node_1": HardwareNodeProfile(
        node_id="node_1",
        channel_id=os.environ.get("TS_CHANNEL_ID", "3591448"),
        device_name="ESP32 Bench Edge Node 1",
        description="Physical testbed bench with MPU6050, DHT11, and 12V DC motor",
        microcontroller="ESP32 DevKit V1 (ESP-WROOM-32)",
        sampling_interval_seconds=20.0,
        nominal_vibration_g=0.022,
        nominal_temperature_c=24.0,
        nominal_humidity_pct=65.0,
        nominal_pressure_hpa=994.5,
    )
}

# Configurable field mapping for ThingSpeak 8-channel feed
FIELD_MAPPINGS = {
    "field1": {"key": "temperature_c", "name": "Temperature", "unit": "°C", "sensor": "DHT11", "confirmed": True},
    "field2": {"key": "humidity_pct", "name": "Humidity", "unit": "%", "sensor": "DHT11", "confirmed": True},
    "field3": {"key": "temp_secondary_c", "name": "Field 3 (secondary temp?)", "unit": "°C", "sensor": "Unconfirmed", "confirmed": False},
    "field4": {"key": "pressure_hpa", "name": "Field 4 (pressure?)", "unit": "hPa", "sensor": "Unconfirmed (likely BMP/BME)", "confirmed": False},
    "field5": {"key": "vibration_g", "name": "Vibration RMS", "unit": "g", "sensor": "MPU6050", "confirmed": True},
    "field6": {"key": "current_a", "name": "Motor Current", "unit": "A", "sensor": "ACS712 (reserved, currently NA)", "confirmed": True},
    "field7": {"key": "esp32_anomaly", "name": "ESP32 Anomaly Flag", "unit": "binary", "sensor": "Firmware logic", "confirmed": True},
    "field8": {"key": "esp32_alert", "name": "ESP32 Alert/Buzzer", "unit": "binary", "sensor": "LED/Buzzer trigger", "confirmed": True},
}

RUL_CONFIG = {
    "warning_threshold_hi": 60.0,
    "failure_threshold_hi": 30.0,
    "min_samples_for_rul": 30,
    "max_acceptable_vibration_g": 0.25,
    "subsystem_weights": {
        "without_current": {"vibration": 0.55, "thermal": 0.30, "environment": 0.15},
        "with_current": {"vibration": 0.45, "thermal": 0.25, "environment": 0.10, "electrical": 0.20},
    },
}
