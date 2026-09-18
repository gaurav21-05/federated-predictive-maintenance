"""
Synthetic Sensor Dataset Generator Module
Coordinates physics simulation, timestamp progression, schema conformity,
and per-client data partitioning for Federated Learning.
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .config import CLIENT_PROFILES, FactoryProfile, SENSOR_COLUMNS
from .physics_engine import IndustrialPhysicsEngine


class SyntheticSensorGenerator:
    """
    Modular generator for industrial sensor streams across multiple federated clients.
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        start_time: Optional[datetime] = None,
        random_seed: int = 42,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.start_time = start_time or datetime(2026, 3, 1, 8, 0, 0)
        self.random_seed = random_seed

    def generate_client_dataset(
        self,
        client_id: str,
        num_records: int = 12000,
        seed_offset: int = 0,
    ) -> List[Dict[str, Union[str, float, int]]]:
        """
        Generate sensor time-series records for a specific client profile.
        """
        if client_id not in CLIENT_PROFILES:
            raise ValueError(f"Unknown client_id: {client_id}. Available: {list(CLIENT_PROFILES.keys())}")

        profile = CLIENT_PROFILES[client_id]
        engine = IndustrialPhysicsEngine(profile=profile, seed=self.random_seed + seed_offset)

        records: List[Dict[str, Union[str, float, int]]] = []
        current_time = self.start_time
        time_step = timedelta(seconds=profile.sampling_rate_seconds)

        for step_idx in range(num_records):
            phys_data = engine.step(step_idx)

            record = {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "machine_id": profile.machine_id,
                "temperature": phys_data["temperature"],
                "humidity": phys_data["humidity"],
                "vibration": phys_data["vibration"],
                "current": phys_data["current"],
                "rpm": phys_data["rpm"],
                "pressure": phys_data["pressure"],
                "failure": phys_data["failure"],
            }
            records.append(record)
            current_time += time_step

        return records

    def generate_all(
        self,
        records_per_client: int = 12000,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[Dict[str, Union[str, float, int]]]]:
        """
        Generate datasets for all 3 clients and write CSV files.
        Guarantees total records >= 30,000 (default: 3 * 12,000 = 36,000).
        """
        out_path = Path(output_dir) if output_dir else self.base_dir / "data"
        out_path.mkdir(parents=True, exist_ok=True)

        all_client_data: Dict[str, List[Dict[str, Union[str, float, int]]]] = {}
        combined_records: List[Dict[str, Union[str, float, int]]] = []

        for idx, client_id in enumerate(["client_1", "client_2", "client_3"]):
            print(f"[*] Generating {records_per_client:,} synthetic sensor records for {client_id}...")
            client_records = self.generate_client_dataset(
                client_id=client_id,
                num_records=records_per_client,
                seed_offset=idx * 1000,
            )
            all_client_data[client_id] = client_records
            combined_records.extend(client_records)

            # Write client specific dataset
            client_dir = out_path / client_id
            client_dir.mkdir(parents=True, exist_ok=True)
            csv_path = client_dir / f"{client_id}_sensor_data.csv"
            self._write_csv(csv_path, client_records, SENSOR_COLUMNS)
            print(f"    [+] Saved: {csv_path} ({len(client_records):,} rows)")

        # Write combined dataset
        combined_path = out_path / "combined_industrial_sensors.csv"
        self._write_csv(combined_path, combined_records, SENSOR_COLUMNS)
        print(f"[+] Saved Combined Dataset: {combined_path} ({len(combined_records):,} rows)")

        return all_client_data

    @staticmethod
    def _write_csv(filepath: Path, records: List[Dict[str, Union[str, float, int]]], fieldnames: List[str]):
        """Helper to write records to CSV cleanly."""
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    def to_dataframe(self, records: List[Dict[str, Union[str, float, int]]]):
        """Convert records to pandas DataFrame if pandas is installed."""
        if HAS_PANDAS:
            return pd.DataFrame(records)
        return records
