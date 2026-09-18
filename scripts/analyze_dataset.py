#!/usr/bin/env python3
"""
Statistical Analysis and Sample Record Inspector
Inspects generated datasets, prints sample records, and checks data validation.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistics import DatasetProfiler


def main():
    data_dir = PROJECT_ROOT / "data"
    profiler = DatasetProfiler(data_path=data_dir)

    print("=" * 80)
    print("DATASET VALIDATION AND STATISTICAL SUMMARY")
    print("=" * 80)

    clients = ["client_1", "client_2", "client_3"]
    for c in clients:
        df = profiler.load_client_data(c)
        print(f"\n>>> Client: {c.upper()} ({df['machine_id'].iloc[0]}) | Total Rows: {len(df):,}")
        print("Sample Records (First 5 rows):")
        print(df.head(5).to_string(index=False))

        # Show a sample failure record if exists
        failure_sample = df[df["failure"] == 1]
        if not failure_sample.empty:
            print("\nSample Failure Event (failure == 1):")
            print(failure_sample.head(2).to_string(index=False))
        else:
            print("\nNo failures observed in this segment.")
        print("-" * 80)

    # Master markdown report
    report = profiler.generate_markdown_report()
    print("\n" + report)


if __name__ == "__main__":
    main()
