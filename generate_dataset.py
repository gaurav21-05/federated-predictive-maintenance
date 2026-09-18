"""
All-in-One Industrial Sensor Dataset Generator for Federated Learning
Project: Privacy-Preserving Predictive Maintenance & Digital Twin Integration in Industrial IoT

Simulates realistic, physics-grounded non-IID sensor telemetry across 3 distinct factories/machines:
- client_1: High-Speed CNC Milling Machine (spindle vibration, bearing overheating, RPM friction drag)
- client_2: Heavy Hydraulic Stamping Press (extreme cyclic hydraulic pressure surges, mechanical shock)
- client_3: Industrial Compressor & Turbine (high humidity sensitivity, motor current overloads, pressure drop)

Outputs:
- data/client_1/client_1_sensor_data.csv (12,000 records)
- data/client_2/client_2_sensor_data.csv (12,000 records)
- data/client_3/client_3_sensor_data.csv (12,000 records)
- data/combined_industrial_sensors.csv (36,000 records total >= 30,000)
"""

import os
import sys
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =====================================================================
# 1. Configuration & Factory Baseline Profiles (Non-IID Modeling)
# =====================================================================

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
    mean: float
    std: float
    min_val: float
    max_val: float
    warning_mult: float
    abnormal_mult: float


@dataclass
class FactoryProfile:
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
    weight_temp: float = 0.25
    weight_vib: float = 0.30
    weight_current: float = 0.15
    weight_rpm: float = 0.15
    weight_pressure: float = 0.15
    base_degradation_rate: float = 0.00012


# Profiles for 3 Heterogeneous Factories
CLIENT_PROFILES: Dict[str, FactoryProfile] = {
    # Factory 1: High-Speed CNC Milling Spindle
    "client_1": FactoryProfile(
        client_id="client_1",
        machine_id="CNC_MILL_01",
        machine_type="5-Axis High-Speed CNC Milling Machine",
        location="Factory-A (Precision Machining)",
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
    ),
    # Factory 2: Heavy Hydraulic Stamping Press
    "client_2": FactoryProfile(
        client_id="client_2",
        machine_id="HYDR_PRESS_02",
        machine_type="Heavy-Duty Hydraulic Stamping Press",
        location="Factory-B (Metal Forming)",
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
    ),
    # Factory 3: Industrial Turbine & Air Compressor
    "client_3": FactoryProfile(
        client_id="client_3",
        machine_id="TURB_COMP_03",
        machine_type="Rotary Screw Air Compressor & Micro-Turbine",
        location="Factory-C (Chemical Processing)",
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
    ),
}


# =====================================================================
# 2. Physics & Degradation Engine (Non-Random Failure Modeling)
# =====================================================================

class IndustrialPhysicsEngine:
    """Simulates physical degradation, cross-coupling, and continuous inertia."""

    def __init__(self, profile: FactoryProfile, seed: int = 42):
        self.profile = profile
        self.rng = random.Random(seed)

        self.current_state = "NORMAL"
        self.state_counter = 0
        self.cycle_length = self.rng.randint(850, 1450)
        self.wear_accumulator = 0.0

        self.active_failure_ticks = 0
        self.maintenance_hold = 0

        self.current_temp = profile.temperature.mean
        self.current_humidity = profile.humidity.mean
        self.current_vibration = profile.vibration.mean
        self.current_current = profile.current.mean
        self.current_rpm = profile.rpm.mean
        self.current_pressure = profile.pressure.mean

    def _update_operational_state(self):
        # 1. Maintenance overhaul state
        if self.maintenance_hold > 0:
            self.maintenance_hold -= 1
            if self.maintenance_hold == 0:
                self.current_state = "NORMAL"
                self.wear_accumulator = self.rng.uniform(0.01, 0.03)
                self.state_counter = 0
                self.cycle_length = self.rng.randint(850, 1450)

                p = self.profile
                self.current_temp = p.temperature.mean + self.rng.gauss(0, p.temperature.std * 0.25)
                self.current_vibration = p.vibration.mean + self.rng.gauss(0, p.vibration.std * 0.2)
                self.current_current = p.current.mean + self.rng.gauss(0, p.current.std * 0.25)
                self.current_rpm = p.rpm.mean + self.rng.gauss(0, p.rpm.std * 0.25)
                self.current_pressure = p.pressure.mean + self.rng.gauss(0, p.pressure.std * 0.25)
            return

        # 2. Active failure sequence
        if self.active_failure_ticks > 0:
            self.active_failure_ticks -= 1
            if self.active_failure_ticks == 0:
                self.maintenance_hold = self.rng.randint(25, 45)
            return

        # 3. Progressive degradation
        self.state_counter += 1
        progress = self.state_counter / max(1, self.cycle_length)

        if progress < 0.82:
            self.current_state = "NORMAL"
            self.wear_accumulator += self.profile.base_degradation_rate * self.rng.uniform(0.8, 1.2)
        elif progress < 0.94:
            self.current_state = "WARNING"
            self.wear_accumulator += self.profile.base_degradation_rate * self.rng.uniform(2.2, 3.8)
        else:
            self.current_state = "ABNORMAL"
            self.wear_accumulator += self.profile.base_degradation_rate * self.rng.uniform(6.0, 9.5)

        self.wear_accumulator = min(1.0, self.wear_accumulator)

    def step(self, step_idx: int) -> Dict[str, float]:
        p = self.profile

        # Case 1: Idle/Maintenance standby
        if self.maintenance_hold > 0:
            self._update_operational_state()
            idle_temp = p.temperature.mean * 0.72 + self.rng.gauss(0, 0.4)
            idle_hum = p.humidity.mean + self.rng.gauss(0, 0.6)
            idle_vib = p.vibration.mean * 0.15 + self.rng.gauss(0, 0.03)
            idle_curr = p.current.mean * 0.12 + self.rng.gauss(0, 0.15)
            idle_rpm = 0.0
            idle_press = p.pressure.mean * 0.50 + self.rng.gauss(0, 0.2)
            return {
                "temperature": round(max(p.temperature.min_val, idle_temp), 2),
                "humidity": round(max(p.humidity.min_val, idle_hum), 2),
                "vibration": round(max(p.vibration.min_val, idle_vib), 3),
                "current": round(max(p.current.min_val, idle_curr), 2),
                "rpm": round(idle_rpm, 1),
                "pressure": round(max(p.pressure.min_val, idle_press), 2),
                "failure": 0,
            }

        # Case 2: Active Failure Breakdown
        if self.active_failure_ticks > 0:
            self._update_operational_state()
            fail_temp = p.temperature.mean * p.temperature.abnormal_mult + self.rng.gauss(0, p.temperature.std * 1.5)
            fail_vib = p.vibration.mean * p.vibration.abnormal_mult + self.rng.gauss(0, p.vibration.std * 1.6)
            fail_curr = p.current.mean * p.current.abnormal_mult + self.rng.gauss(0, p.current.std * 1.4)
            fail_rpm = p.rpm.mean * p.rpm.abnormal_mult + self.rng.gauss(0, p.rpm.std * 2.0)
            fail_press = p.pressure.mean * p.pressure.abnormal_mult + self.rng.gauss(0, p.pressure.std * 1.5)

            self.current_temp = 0.85 * self.current_temp + 0.15 * fail_temp
            self.current_vibration = 0.70 * self.current_vibration + 0.30 * fail_vib
            self.current_current = 0.70 * self.current_current + 0.30 * fail_curr
            self.current_rpm = 0.75 * self.current_rpm + 0.25 * fail_rpm
            self.current_pressure = 0.75 * self.current_pressure + 0.25 * fail_press

            return {
                "temperature": round(min(p.temperature.max_val, max(p.temperature.min_val, self.current_temp)), 2),
                "humidity": round(min(p.humidity.max_val, max(p.humidity.min_val, self.current_humidity)), 2),
                "vibration": round(min(p.vibration.max_val, max(p.vibration.min_val, self.current_vibration)), 3),
                "current": round(min(p.current.max_val, max(p.current.min_val, self.current_current)), 2),
                "rpm": round(min(p.rpm.max_val, max(p.rpm.min_val, self.current_rpm)), 1),
                "pressure": round(min(p.pressure.max_val, max(p.pressure.min_val, self.current_pressure)), 2),
                "failure": 1,
            }

        # Case 3: Operational (NORMAL, WARNING, or ABNORMAL)
        self._update_operational_state()
        state = self.current_state

        if state == "NORMAL":
            target_temp = p.temperature.mean + self.rng.gauss(0, p.temperature.std * 0.4)
            target_vib = p.vibration.mean + self.rng.gauss(0, p.vibration.std * 0.3)
            target_curr = p.current.mean + self.rng.gauss(0, p.current.std * 0.35)
            target_rpm = p.rpm.mean + self.rng.gauss(0, p.rpm.std * 0.3)
            target_press = p.pressure.mean + self.rng.gauss(0, p.pressure.std * 0.3)
        elif state == "WARNING":
            target_temp = p.temperature.mean * p.temperature.warning_mult + self.rng.gauss(0, p.temperature.std * 0.7)
            target_vib = p.vibration.mean * p.vibration.warning_mult + self.rng.gauss(0, p.vibration.std * 0.5)
            target_curr = p.current.mean * p.current.warning_mult + self.rng.gauss(0, p.current.std * 0.5)
            target_rpm = p.rpm.mean * p.rpm.warning_mult + self.rng.gauss(0, p.rpm.std * 0.6)
            target_press = p.pressure.mean * p.pressure.warning_mult + self.rng.gauss(0, p.pressure.std * 0.5)
        else:  # ABNORMAL
            target_temp = p.temperature.mean * (p.temperature.warning_mult + 0.15) + self.rng.gauss(0, p.temperature.std * 1.0)
            target_vib = p.vibration.mean * (p.vibration.warning_mult + 0.8) + self.rng.gauss(0, p.vibration.std * 1.1)
            target_curr = p.current.mean * (p.current.warning_mult + 0.25) + self.rng.gauss(0, p.current.std * 0.9)
            target_rpm = p.rpm.mean * (p.rpm.warning_mult - 0.10) + self.rng.gauss(0, p.rpm.std * 1.2)
            target_press = p.pressure.mean * (p.pressure.warning_mult + 0.10) + self.rng.gauss(0, p.pressure.std * 0.9)

        # Cross-Coupling:
        # High vibration -> friction & thermal buildup
        vib_excess = max(0.0, (self.current_vibration - p.vibration.mean) / max(0.1, p.vibration.std))
        target_temp += vib_excess * 1.5

        # Resistance -> current surge
        curr_load_factor = (vib_excess * 0.7) + (self.wear_accumulator * 1.8)
        target_curr += curr_load_factor

        # Drag -> RPM drop
        target_rpm -= (vib_excess * 28.0) + (curr_load_factor * 14.0)

        # Client-specific interactions:
        if p.client_id == "client_2":
            press_deviation = abs(self.current_pressure - p.pressure.mean) / p.pressure.std
            if press_deviation > 1.6:
                target_vib += press_deviation * 0.5
                target_temp += press_deviation * 0.9
        elif p.client_id == "client_3":
            if self.current_pressure < (p.pressure.mean * 0.88):
                target_curr += 4.0
                target_temp += 2.8

        # Diurnal cycle
        diurnal_phase = (step_idx * p.sampling_rate_seconds) / 86400.0 * 2.0 * math.pi
        target_humidity = p.humidity.mean + 4.5 * math.sin(diurnal_phase) + self.rng.gauss(0, p.humidity.std * 0.3)
        target_temp += 1.4 * math.sin(diurnal_phase - math.pi / 4.0)

        # Inertia smoothing
        self.current_temp = 0.90 * self.current_temp + 0.10 * target_temp
        self.current_humidity = 0.90 * self.current_humidity + 0.10 * target_humidity
        self.current_vibration = 0.78 * self.current_vibration + 0.22 * target_vib
        self.current_current = 0.72 * self.current_current + 0.28 * target_curr
        self.current_rpm = 0.82 * self.current_rpm + 0.18 * target_rpm
        self.current_pressure = 0.80 * self.current_pressure + 0.20 * target_press

        temp_val = max(p.temperature.min_val, min(p.temperature.max_val, self.current_temp))
        hum_val = max(p.humidity.min_val, min(p.humidity.max_val, self.current_humidity))
        vib_val = max(p.vibration.min_val, min(p.vibration.max_val, self.current_vibration))
        curr_val = max(p.current.min_val, min(p.current.max_val, self.current_current))
        rpm_val = max(p.rpm.min_val, min(p.rpm.max_val, self.current_rpm))
        press_val = max(p.pressure.min_val, min(p.pressure.max_val, self.current_pressure))

        # Normalized Machine Stress Index
        norm_temp = max(0.0, (temp_val - (p.temperature.mean + 1.2 * p.temperature.std)) / (p.temperature.std * 2.5))
        norm_vib = max(0.0, (vib_val - (p.vibration.mean + 1.2 * p.vibration.std)) / (p.vibration.std * 2.5))
        norm_curr = max(0.0, (curr_val - (p.current.mean + 1.2 * p.current.std)) / (p.current.std * 2.5))
        norm_rpm = max(0.0, ((p.rpm.mean - 1.2 * p.rpm.std) - rpm_val) / (p.rpm.std * 2.5))
        norm_press = max(0.0, abs(press_val - p.pressure.mean) - 1.2 * p.pressure.std) / (p.pressure.std * 2.5)

        msi = (
            p.weight_temp * min(2.0, norm_temp)
            + p.weight_vib * min(2.0, norm_vib)
            + p.weight_current * min(2.0, norm_curr)
            + p.weight_rpm * min(2.0, norm_rpm)
            + p.weight_pressure * min(2.0, norm_press)
            + 0.25 * self.wear_accumulator
        )

        failure = 0
        if state == "ABNORMAL":
            critical_stress = (
                (norm_vib > 1.6 and norm_temp > 1.4)
                or (norm_vib > 1.9)
                or (norm_temp > 1.9 and norm_curr > 1.5)
                or (p.client_id == "client_2" and norm_press > 1.8 and norm_vib > 1.4)
                or (p.client_id == "client_3" and norm_curr > 1.8 and norm_press > 1.5)
            )
            prob_failure = 0.12 * min(1.0, max(0.0, msi - 0.4) * 2.5)

            if critical_stress or (self.rng.random() < prob_failure):
                failure = 1
                self.active_failure_ticks = self.rng.randint(18, 32)

        return {
            "temperature": round(temp_val, 2),
            "humidity": round(hum_val, 2),
            "vibration": round(vib_val, 3),
            "current": round(curr_val, 2),
            "rpm": round(rpm_val, 1),
            "pressure": round(press_val, 2),
            "failure": int(failure),
        }


# =====================================================================
# 3. Main Generator & CLI Execution
# =====================================================================

def generate_datasets(records_per_client: int = 12000, output_dir: str = "data", seed: int = 42):
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("INDUSTRIAL SENSOR DATASET GENERATOR FOR FEDERATED LEARNING")
    print("Project: Privacy-Preserving Predictive Maintenance in Industrial IoT")
    print("=" * 75)
    print(f"Records per Client : {records_per_client:,}")
    print(f"Total Records      : {records_per_client * 3:,} (Target: >= 30,000)")
    print(f"Output Directory   : {base_path.resolve()}")
    print("-" * 75)

    start_time = datetime(2026, 3, 1, 8, 0, 0)
    all_records: List[Dict] = []

    for idx, (client_id, profile) in enumerate(CLIENT_PROFILES.items()):
        print(f"[*] Simulating {profile.machine_type} ({client_id})...")
        engine = IndustrialPhysicsEngine(profile=profile, seed=seed + idx * 1000)

        client_records = []
        curr_time = start_time
        time_step = timedelta(seconds=profile.sampling_rate_seconds)

        for step_idx in range(records_per_client):
            p_data = engine.step(step_idx)
            row = {
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                "machine_id": profile.machine_id,
                **p_data
            }
            client_records.append(row)
            curr_time += time_step

        all_records.extend(client_records)

        # Write client partition
        client_dir = base_path / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        csv_file = client_dir / f"{client_id}_sensor_data.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SENSOR_COLUMNS)
            writer.writeheader()
            writer.writerows(client_records)
        print(f"    [+] Saved: {csv_file} ({len(client_records):,} rows)")

    # Write combined dataset
    combined_file = base_path / "combined_industrial_sensors.csv"
    with open(combined_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SENSOR_COLUMNS)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"[+] Saved Combined Benchmark: {combined_file} ({len(all_records):,} rows)")

    # Print quick statistics
    if HAS_PANDAS:
        df = pd.DataFrame(all_records)
        print("\n" + "=" * 75)
        print("SUMMARY DATASET VALIDATION")
        print("=" * 75)
        print(f"Total Rows Generated : {len(df):,}")
        print(f"Healthy Records (0)  : {(df['failure'] == 0).sum():,} ({((df['failure'] == 0).mean() * 100):.2f}%)")
        print(f"Failure Records (1)  : {(df['failure'] == 1).sum():,} ({((df['failure'] == 1).mean() * 100):.2f}%)")
        print(f"Missing Values       : {df.isnull().sum().sum()}")
        print("-" * 75)


if __name__ == "__main__":
    generate_datasets()
