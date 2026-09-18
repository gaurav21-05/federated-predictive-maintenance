#!/usr/bin/env python3
"""
Digital Twin Synchronization & Streaming Simulation Script
Synchronizes virtual replicas for all 3 industrial edge factories (client_1, client_2, client_3),
evaluates live health indices, predicts Remaining Useful Life (RUL), runs What-If stress tests,
dispatches prescriptive maintenance work orders, and dumps live synchronized state to
data/digital_twin_state.json for the interactive dashboard.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLIENT_PROFILES
from src.digital_twin.twin_engine import MachineDigitalTwin
from src.digital_twin.rul_estimator import RULEstimator
from src.digital_twin.what_if_simulator import WhatIfSimulator, SimulationScenario
from src.digital_twin.maintenance_dispatcher import PrescriptiveMaintenanceDispatcher
from src.models import CNNLSTMPredictor
from src.preprocessing import TimeSeriesPreprocessor


def load_federated_model(checkpoint_path: Path) -> CNNLSTMPredictor:
    """Loads the trained global consensus predictive maintenance model."""
    model = CNNLSTMPredictor(num_features=9, window_size=30)
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[+] Loaded Federated Consensus Model from {checkpoint_path.name}")
    else:
        print(f"[-] Warning: Checkpoint {checkpoint_path} not found. Running with baseline initialization.")
    model.eval()
    return model


def run_digital_twin_synchronization(
    num_ticks: int = 120,
    start_tick: int = 800,  # Starts near degradation transition zone for realistic demo
    checkpoint_name: str = "global_federated_best.pt",
    delay_sec: float = 0.0,
):
    data_dir = PROJECT_ROOT / "data"
    chk_dir = PROJECT_ROOT / "models" / "checkpoints"
    chk_file = chk_dir / checkpoint_name

    print("=" * 95)
    print("INDUSTRIAL DIGITAL TWIN SYNCHRONIZATION & RUL ESTIMATION ENGINE")
    print("Real-Time Multi-Factory Virtual Replica Telemetry Ingestion")
    print("=" * 95)

    # 1. Load global predictive maintenance model
    model = load_federated_model(chk_file)
    preprocessor = TimeSeriesPreprocessor(window_size=30, stride=5, lookahead_horizon=12)

    # 2. Load historical telemetry streams for all 3 clients
    client_ids = ["client_1", "client_2", "client_3"]
    telemetry_dfs = {}
    for cid in client_ids:
        csv_file = data_dir / cid / f"{cid}_sensor_data.csv"
        df = pd.read_csv(csv_file)
        telemetry_dfs[cid] = df
        # Fit preprocessor scalers
        _ = preprocessor.prepare_client_dataset(df)

    # 3. Instantiate Digital Twins & Estimators
    twins: Dict[str, MachineDigitalTwin] = {}
    rul_estimators: Dict[str, RULEstimator] = {}

    for cid in client_ids:
        twin = MachineDigitalTwin(
            client_id=cid,
            profile=CLIENT_PROFILES[cid],
            window_size=30,
            model=model,
            preprocessor=preprocessor,
        )
        twins[cid] = twin
        rul_estimators[cid] = RULEstimator(client_id=cid, history_window=25)

    what_if_sim = WhatIfSimulator()
    dispatcher = PrescriptiveMaintenanceDispatcher(dispatch_cooldown_ticks=30)

    print(f"[*] Initialized 3 Digital Twins. Replaying {num_ticks} streaming ticks starting at tick #{start_tick}...")
    print("-" * 95)

    # Header for real-time progress
    print(
        f"{'Tick':<6} | "
        f"{'C1 Health':<10} | {'C1 RUL':<8} | {'C1 State':<8} | "
        f"{'C2 Health':<10} | {'C2 RUL':<8} | {'C2 State':<8} | "
        f"{'C3 Health':<10} | {'C3 RUL':<8} | {'C3 State':<8} | "
        f"{'Open WOs':<8}"
    )
    print("-" * 95)

    final_twin_states = {}

    for t_step in range(start_tick, start_tick + num_ticks):
        current_tick_num = t_step - start_tick + 1

        # Feed each twin its corresponding sensor row
        tick_summary = {}
        for cid in client_ids:
            df = telemetry_dfs[cid]
            row_idx = t_step % len(df)
            row_dict = df.iloc[row_idx].to_dict()

            # Synchronize virtual twin
            state = twins[cid].ingest_telemetry_point(row_dict)

            # Predict Remaining Useful Life
            rul_pred = rul_estimators[cid].predict_rul(
                current_health_index=twins[cid].overall_health_index,
                failure_probability=twins[cid].failure_probability,
            )

            # Evaluate prescriptive maintenance triggers
            new_wo = dispatcher.evaluate_twin(twins[cid], rul_pred, current_tick=t_step)
            if new_wo:
                print(f"    >>> [DISPATCH ALERT] New Work Order: {new_wo.work_order_id} ({new_wo.urgency}) for {new_wo.machine_name}!")

            tick_summary[cid] = {
                "health": state["overall_health_index"],
                "rul": rul_pred.rul_minutes,
                "state": state["operational_state"],
                "rul_pred": rul_pred,
            }

        # Print progress every 15 ticks or on first tick
        if current_tick_num == 1 or current_tick_num % 15 == 0 or current_tick_num == num_ticks:
            c1 = tick_summary["client_1"]
            c2 = tick_summary["client_2"]
            c3 = tick_summary["client_3"]
            open_wos = len(dispatcher.get_active_work_orders())
            print(
                f"#{current_tick_num:<5} | "
                f"{c1['health']:<9.1f}% | {c1['rul']:<7.0f}m | {c1['state']:<8} | "
                f"{c2['health']:<9.1f}% | {c2['rul']:<7.0f}m | {c2['state']:<8} | "
                f"{c3['health']:<9.1f}% | {c3['rul']:<7.0f}m | {c3['state']:<8} | "
                f"{open_wos:<8}"
            )

        if delay_sec > 0:
            time.sleep(delay_sec)

    # 4. Evaluate What-If Stress Scenarios on the latest Twin states
    print("\n" + "=" * 95)
    print("RUNNING WHAT-IF SCENARIO ANALYSIS ON DIGITAL TWINS (CLIENT 1 - CNC MILL)")
    print("=" * 95)
    what_if_results = {}
    for scenario_key, scenario in WhatIfSimulator.PRESETS.items():
        sim_res = what_if_sim.simulate(twins["client_1"], scenario)
        what_if_results[scenario_key] = sim_res
        print(f"[*] Scenario: {sim_res['scenario_name']}")
        print(f"    - Wear Acceleration Factor: {sim_res['wear_acceleration_factor']:.2f}x")
        print(f"    - Baseline RUL: {sim_res['baseline_rul_minutes']:.1f} min -> Projected RUL: {sim_res['projected_rul_minutes']:.1f} min ({sim_res['rul_delta_percent']:+.1f}%)")
        print(f"    - Risk Assessment: [{sim_res['risk_level']}] {sim_res['recommendation']}")
        print("-" * 75)

    # 5. Assemble Complete Digital Twin System State Snapshot
    complete_state = {
        "metadata": {
            "system_name": "Industrial IoT Digital Twin & Federated RUL Engine",
            "last_synced_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_simulation_ticks": num_ticks,
            "federated_model_checkpoint": checkpoint_name,
        },
        "twins": {
            cid: {
                **twins[cid].get_state_snapshot(),
                "rul_prediction": rul_estimators[cid].predict_rul(
                    twins[cid].overall_health_index, twins[cid].failure_probability
                ).to_dict(),
            }
            for cid in client_ids
        },
        "active_work_orders": dispatcher.get_active_work_orders(),
        "what_if_scenarios": what_if_results,
    }

    # 6. Export to data/digital_twin_state.json
    out_file = data_dir / "digital_twin_state.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(complete_state, f, indent=2)

    print(f"\n[+] Digital Twin Synchronized State successfully exported to: {out_file}")
    return complete_state


def main():
    parser = argparse.ArgumentParser(description="Run Digital Twin Telemetry Synchronization")
    parser.add_argument("--ticks", type=int, default=120, help="Number of telemetry replay ticks (default: 120)")
    parser.add_argument("--start-tick", type=int, default=800, help="Starting row index in dataset (default: 800)")
    parser.add_argument("--checkpoint", type=str, default="global_federated_best.pt", help="Federated model checkpoint")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between ticks in seconds (default: 0.0)")
    args = parser.parse_args()

    run_digital_twin_synchronization(
        num_ticks=args.ticks,
        start_tick=args.start_tick,
        checkpoint_name=args.checkpoint,
        delay_sec=args.delay,
    )


if __name__ == "__main__":
    main()
