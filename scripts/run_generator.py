#!/usr/bin/env python3
"""
Main Execution Script: Synthetic Industrial Sensor Dataset Generator
Generates realistic non-IID sensor datasets for client_1, client_2, client_3.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import SyntheticSensorGenerator
from src.statistics import DatasetProfiler


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic industrial sensor datasets for Federated Learning and Digital Twins."
    )
    parser.add_argument(
        "--records-per-client",
        type=int,
        default=12000,
        help="Number of records to generate per client (default: 12000, total = 36000 >= 30000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Directory to save generated CSV files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("INDUSTRIAL SENSOR DATASET GENERATOR FOR FEDERATED LEARNING")
    print("Project: Predictive Maintenance & Digital Twin Integration")
    print("=" * 70)
    print(f"Target Records per Client : {args.records_per_client:,}")
    print(f"Total Target Records      : {args.records_per_client * 3:,}")
    print(f"Random Seed               : {args.seed}")
    print(f"Output Directory          : {args.output_dir}")
    print("-" * 70)

    generator = SyntheticSensorGenerator(base_dir=PROJECT_ROOT, random_seed=args.seed)
    generator.generate_all(
        records_per_client=args.records_per_client,
        output_dir=args.output_dir,
    )

    print("\n[+] Dataset generation complete! Profiling dataset...")
    profiler = DatasetProfiler(data_path=args.output_dir)
    report = profiler.generate_markdown_report()
    print("\n" + report)

    # Save summary report to data directory
    report_file = Path(args.output_dir) / "dataset_summary.md"
    report_file.write_text(report, encoding="utf-8")
    print(f"\n[+] Statistical report saved to: {report_file}")


if __name__ == "__main__":
    main()
