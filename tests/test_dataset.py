"""
Automated Test & Verification Suite for Industrial Sensor Generator
Validates schema conformity, row counts, null value absence, and non-IID properties.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SENSOR_COLUMNS


def test_datasets():
    data_dir = PROJECT_ROOT / "data"
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

    clients = ["client_1", "client_2", "client_3"]
    total_records = 0

    print("[*] Starting Dataset Verification Suite...")

    for client_id in clients:
        csv_file = data_dir / client_id / f"{client_id}_sensor_data.csv"
        assert csv_file.exists(), f"Missing dataset file for {client_id}: {csv_file}"

        df = pd.read_csv(csv_file)

        # 1. Row count verification
        assert len(df) >= 10000, f"{client_id} has insufficient rows: {len(df)}"
        total_records += len(df)

        # 2. Schema check
        assert list(df.columns) == SENSOR_COLUMNS, f"{client_id} columns mismatch: {list(df.columns)}"

        # 3. Missing values check
        null_counts = df.isnull().sum().sum()
        assert null_counts == 0, f"{client_id} contains {null_counts} null/NaN values"

        # 4. Failure column check
        unique_failures = set(df["failure"].unique())
        assert unique_failures.issubset({0, 1}), f"Unexpected failure values: {unique_failures}"
        failure_rate = df["failure"].mean()
        assert 0.01 <= failure_rate <= 0.08, f"Failure rate out of realistic range (1%-8%): {failure_rate:.4f}"

        # 5. Machine ID check
        assert df["machine_id"].nunique() == 1, f"Multiple machine IDs found in {client_id}"

        print(f"  [PASS] {client_id}: {len(df):,} rows, 0 nulls, failure rate = {failure_rate * 100:.2f}%")

    # Combined dataset verification
    combined_file = data_dir / "combined_industrial_sensors.csv"
    assert combined_file.exists(), f"Missing combined dataset: {combined_file}"
    comb_df = pd.read_csv(combined_file)
    assert len(comb_df) == total_records, f"Combined record count mismatch: {len(comb_df)} != {total_records}"
    assert len(comb_df) >= 30000, f"Total records under 30,000 threshold: {len(comb_df)}"
    assert comb_df["machine_id"].nunique() == 3, "Combined dataset does not contain all 3 machines"

    print(f"  [PASS] Combined Dataset: {len(comb_df):,} rows (>= 30,000 threshold satisfied)")
    print("[+] All verification tests passed successfully!")


if __name__ == "__main__":
    test_datasets()
