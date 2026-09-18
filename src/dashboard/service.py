"""
Industrial IoT Predictive Maintenance Dashboard Service Layer
Orchestrates digital twin synchronization, telemetry streaming, RUL forecasting,
what-if stress simulation, prescriptive maintenance dispatching, and federated learning execution.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import CLIENT_PROFILES, SENSOR_COLUMNS, FactoryProfile
from ..digital_twin.twin_engine import MachineDigitalTwin
from ..digital_twin.rul_estimator import RULEstimator, RULPrediction
from ..digital_twin.what_if_simulator import WhatIfSimulator, SimulationScenario
from ..digital_twin.maintenance_dispatcher import PrescriptiveMaintenanceDispatcher, MaintenanceWorkOrder
from ..federated.client import FederatedEdgeClient
from ..federated.server import FederatedServer
from ..federated.strategies import FedAvgStrategy, FedProxStrategy
from ..models import CNNLSTMPredictor, EdgeMLPPredictor
from ..preprocessing import TimeSeriesPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DashboardService:
    """
    Core state management and orchestration engine for the Web Dashboard.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None, device: str = "cpu"):
        self.data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data"
        self.device = device
        self.client_ids = ["client_1", "client_2", "client_3"]

        # 1. Load telemetry datasets
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.current_step: Dict[str, int] = {cid: 0 for cid in self.client_ids}
        self._load_datasets()

        # 2. Digital Twin & RUL Engines
        self.twins: Dict[str, MachineDigitalTwin] = {}
        self.rul_estimators: Dict[str, RULEstimator] = {}
        self.what_if_simulators: Dict[str, WhatIfSimulator] = {}
        self.dispatcher = PrescriptiveMaintenanceDispatcher()

        self._init_digital_twins()

        # 3. Federated Learning state cache
        self.last_fl_results: Optional[Dict[str, Any]] = None
        self._cached_loaders: Optional[Dict[str, DataLoader]] = None

    def _load_datasets(self):
        """Loads client sensor CSVs into memory."""
        for cid in self.client_ids:
            csv_path = self.data_dir / cid / f"{cid}_sensor_data.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                self.datasets[cid] = df
            else:
                # Fallback to empty dataframe
                self.datasets[cid] = pd.DataFrame()

    def _init_digital_twins(self):
        """Initializes MachineDigitalTwin instances for each industrial machine."""
        for cid in self.client_ids:
            profile = CLIENT_PROFILES.get(cid, CLIENT_PROFILES["client_1"])
            self.twins[cid] = MachineDigitalTwin(
                client_id=cid,
                profile=profile,
                window_size=30,
                device=self.device,
            )
            self.rul_estimators[cid] = RULEstimator(client_id=cid)
            self.what_if_simulators[cid] = WhatIfSimulator(self.rul_estimators[cid])

            # Ingest initial 30 nominal points if dataset available
            df = self.datasets.get(cid)
            if df is not None and len(df) >= 30:
                for i in range(30):
                    row = df.iloc[i].to_dict()
                    self.twins[cid].ingest_telemetry_point(row)
                self.current_step[cid] = 30

    def get_overview(self) -> Dict[str, Any]:
        """Provides a high-level operational overview across all factory edge nodes."""
        machines = []
        overall_healths = []
        critical_count = 0
        warning_count = 0

        for cid in self.client_ids:
            twin = self.twins[cid]
            state = twin.get_state_snapshot()
            mhi = state.get("overall_health_index", 100.0)
            overall_healths.append(mhi)

            rul_pred = self.rul_estimators[cid].predict_rul(mhi, twin.failure_probability)
            op_state = state.get("operational_state", "NORMAL")
            if op_state == "CRITICAL":
                critical_count += 1
            elif op_state == "WARNING":
                warning_count += 1

            machines.append({
                "client_id": cid,
                "machine_id": getattr(twin.profile, "machine_id", cid.upper()),
                "machine_name": getattr(twin.profile, "machine_type", "Industrial Unit"),
                "operational_state": op_state,
                "overall_health_index": round(mhi, 1),
                "subsystem_health": {k: round(v, 1) for k, v in state.get("subsystem_health", {}).items()},
                "primary_fault_subsystem": state.get("primary_fault_subsystem", "Healthy"),
                "rul_minutes": round(rul_pred.rul_minutes, 1),
                "urgency_level": rul_pred.urgency_level,
                "current_step": self.current_step[cid],
                "total_records": len(self.datasets.get(cid, [])),
            })

        fleet_health = round(float(np.mean(overall_healths)), 1) if overall_healths else 100.0

        return {
            "timestamp": datetime.now().isoformat(),
            "fleet_health_index": fleet_health,
            "fleet_status": "CRITICAL" if critical_count > 0 else ("WARNING" if warning_count > 0 else "OPTIMAL"),
            "machines": machines,
            "open_work_orders": len([wo for wo in self.dispatcher.work_orders if wo.status == "OPEN"]),
            "total_work_orders": len(self.dispatcher.work_orders),
            "last_fl_macro_f1": round(self.last_fl_results.get("best_macro_f1", 0.0), 4) if self.last_fl_results else None,
        }

    def get_telemetry_window(self, client_id: str, window: int = 50) -> Dict[str, Any]:
        """Returns the recent historical telemetry stream for plotting."""
        if client_id not in self.datasets or self.datasets[client_id].empty:
            return {"client_id": client_id, "data": []}

        df = self.datasets[client_id]
        curr = self.current_step[client_id]
        start_idx = max(0, curr - window)
        end_idx = min(len(df), max(curr, start_idx + 1))

        slice_df = df.iloc[start_idx:end_idx]

        records = []
        for _, row in slice_df.iterrows():
            records.append({
                "timestamp": str(row.get("timestamp", "")),
                "temperature": round(float(row.get("temperature", 0.0)), 2),
                "humidity": round(float(row.get("humidity", 0.0)), 2),
                "vibration": round(float(row.get("vibration", 0.0)), 2),
                "current": round(float(row.get("current", 0.0)), 2),
                "rpm": round(float(row.get("rpm", 0.0)), 1),
                "pressure": round(float(row.get("pressure", 0.0)), 2),
                "failure": int(row.get("failure", 0)),
            })

        twin = self.twins[client_id]
        current_state = twin.get_state_snapshot()
        rul_pred = self.rul_estimators[client_id].predict_rul(
            current_state.get("overall_health_index", 100.0),
            twin.failure_probability,
        )

        return {
            "client_id": client_id,
            "current_step": curr,
            "total_steps": len(df),
            "state_snapshot": current_state,
            "rul_prediction": rul_pred.to_dict(),
            "telemetry": records,
        }

    def step(self, client_id: Optional[str] = None, steps: int = 1) -> Dict[str, Any]:
        """
        Advances telemetry stream for one or all clients and updates Digital Twins.
        """
        targets = [client_id] if client_id and client_id in self.client_ids else self.client_ids
        dispatched_orders = []

        for cid in targets:
            df = self.datasets.get(cid)
            if df is None or df.empty:
                continue

            total_len = len(df)
            twin = self.twins[cid]

            for _ in range(steps):
                curr = self.current_step[cid]
                if curr >= total_len:
                    # Loop around for continuous real-time simulation
                    curr = 0

                row = df.iloc[curr].to_dict()
                state = twin.ingest_telemetry_point(row)
                self.current_step[cid] = curr + 1

                # Check for prescriptive maintenance dispatch
                rul_pred = self.rul_estimators[cid].predict_rul(
                    state.get("overall_health_index", 100.0),
                    twin.failure_probability,
                )
                wo = self.dispatcher.evaluate_twin(twin, rul_pred, current_tick=curr)
                if wo:
                    dispatched_orders.append(wo.to_dict())

        return {
            "status": "success",
            "advanced_steps": steps,
            "current_steps": self.current_step,
            "new_work_orders": dispatched_orders,
            "overview": self.get_overview(),
        }

    def reset_simulation(self, client_id: Optional[str] = None):
        """Resets the streaming step to 0 and reinitializes digital twin."""
        targets = [client_id] if client_id and client_id in self.client_ids else self.client_ids
        for cid in targets:
            profile = CLIENT_PROFILES.get(cid, CLIENT_PROFILES["client_1"])
            self.twins[cid] = MachineDigitalTwin(
                client_id=cid,
                profile=profile,
                window_size=30,
                device=self.device,
            )
            self.rul_estimators[cid] = RULEstimator(client_id=cid)
            self.what_if_simulators[cid] = WhatIfSimulator(self.rul_estimators[cid])
            self.current_step[cid] = 0

            df = self.datasets.get(cid)
            if df is not None and len(df) >= 30:
                for i in range(30):
                    row = df.iloc[i].to_dict()
                    self.twins[cid].ingest_telemetry_point(row)
                self.current_step[cid] = 30

        return {"status": "success", "reset_clients": targets, "current_steps": self.current_step}

    def run_what_if(
        self,
        client_id: str,
        scenario_key: Optional[str] = None,
        rpm_multiplier: float = 1.0,
        load_multiplier: float = 1.0,
        ambient_temp_delta: float = 0.0,
        horizon_minutes: float = 60.0,
    ) -> Dict[str, Any]:
        """Runs a What-If stress simulation on the specified machine."""
        if client_id not in self.twins:
            return {"error": f"Invalid client_id: {client_id}"}

        simulator = self.what_if_simulators[client_id]
        twin = self.twins[client_id]

        if scenario_key and scenario_key in simulator.PRESETS:
            scenario = simulator.PRESETS[scenario_key]
        else:
            scenario = SimulationScenario(
                name=f"Custom Sandbox ({rpm_multiplier:+.0%} RPM, {load_multiplier:+.0%} Load)",
                description=f"User-configured load: {rpm_multiplier}x RPM, {load_multiplier}x torque, +{ambient_temp_delta}°C temp.",
                rpm_multiplier=rpm_multiplier,
                load_multiplier=load_multiplier,
                ambient_temp_delta=ambient_temp_delta,
                horizon_minutes=horizon_minutes,
            )

        result = simulator.simulate(twin, scenario)
        return {
            "client_id": client_id,
            "machine_name": getattr(twin.profile, "machine_type", client_id),
            "simulation": result,
        }

    def get_work_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all dispatched maintenance work orders."""
        orders = self.dispatcher.work_orders
        if status:
            orders = [o for o in orders if o.status.upper() == status.upper()]
        return [o.to_dict() for o in orders]

    def update_work_order_status(self, work_order_id: str, new_status: str) -> Dict[str, Any]:
        """Updates work order state (OPEN, ASSIGNED, RESOLVED)."""
        valid_statuses = {"OPEN", "ASSIGNED", "RESOLVED"}
        if new_status.upper() not in valid_statuses:
            return {"error": f"Invalid status {new_status}. Must be one of {valid_statuses}"}

        for order in self.dispatcher.work_orders:
            if order.work_order_id == work_order_id:
                order.status = new_status.upper()
                return {"status": "success", "work_order": order.to_dict()}

        return {"error": f"Work order {work_order_id} not found"}

    def _prepare_fast_fl_loaders(self, sample_size: int = 400) -> Dict[str, Dict[str, DataLoader]]:
        """
        Creates lightweight sliding-window DataLoaders from client datasets for fast interactive FL runs.
        """
        prep = TimeSeriesPreprocessor(window_size=30, stride=5, lookahead_horizon=12)
        loaders = {}

        for cid in self.client_ids:
            df = self.datasets.get(cid)
            if df is None or df.empty:
                continue

            # Take a representative slice
            sub_df = df.iloc[:sample_size].copy()
            splits = prep.prepare_client_dataset(sub_df, train_ratio=0.70, val_ratio=0.15)

            X_tr = torch.tensor(splits["X_train"], dtype=torch.float32)
            y_tr = torch.tensor(splits["y_train"], dtype=torch.float32).unsqueeze(1)
            X_te = torch.tensor(splits["X_test"], dtype=torch.float32)
            y_te = torch.tensor(splits["y_test"], dtype=torch.float32).unsqueeze(1)

            train_ds = TensorDataset(X_tr, y_tr)
            test_ds = TensorDataset(X_te, y_te)

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

            loaders[cid] = {
                "train": train_loader,
                "val": test_loader,
                "test": test_loader,
                "num_train": len(train_ds),
                "num_test": len(test_ds),
            }

        return loaders

    def run_federated_training(
        self,
        strategy_name: str = "FedProx",
        num_rounds: int = 3,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        enable_dp: bool = True,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.05,
        enable_secagg: bool = True,
        mu: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Executes an on-demand Federated Learning training run and records convergence metrics.
        """
        loaders_map = self._prepare_fast_fl_loaders(sample_size=500)
        if not loaders_map:
            return {"error": "No dataset loaders available for federated training."}

        # Build Global Model
        global_model = CNNLSTMPredictor(num_features=9, window_size=30).to(self.device)

        # Build Client Nodes
        clients = []
        for cid, ld in loaders_map.items():
            local_model = CNNLSTMPredictor(num_features=9, window_size=30).to(self.device)
            local_model.set_weights(global_model.get_weights())

            c_mu = mu if strategy_name.lower() == "fedprox" else 0.0
            client = FederatedEdgeClient(
                client_id=cid,
                model=local_model,
                train_loader=ld["train"],
                val_loader=ld["val"],
                test_loader=ld["test"],
                device=self.device,
                lr=learning_rate,
                mu=c_mu,
            )
            clients.append(client)

        # Strategy Selection
        if strategy_name.lower() == "fedprox":
            strategy = FedProxStrategy(mu=mu)
        else:
            strategy = FedAvgStrategy()

        server = FederatedServer(
            global_model=global_model,
            clients=clients,
            strategy=strategy,
            device=self.device,
        )

        results = server.fit(
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            enable_dp=enable_dp,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            enable_secagg=enable_secagg,
            verbose=False,
        )

        self.last_fl_results = results

        # Format clean response for frontend charts
        convergence_series = []
        for entry in results.get("history", []):
            convergence_series.append({
                "round": entry.get("round"),
                "avg_train_loss": round(float(entry.get("avg_train_loss", 0.0)), 4),
                "macro_f1": round(float(entry.get("macro_f1", 0.0)), 4),
                "macro_recall": round(float(entry.get("macro_recall", 0.0)), 4),
                "macro_precision": round(float(entry.get("macro_precision", 0.0)), 4),
                "epsilon": round(float(entry.get("epsilon", 0.0)), 3) if entry.get("epsilon") != float("inf") else 999.0,
                "delta": float(entry.get("delta", 1e-5)),
            })

        return {
            "status": "success",
            "strategy": strategy_name,
            "rounds_completed": len(convergence_series),
            "best_macro_f1": round(float(results.get("best_macro_f1", 0.0)), 4),
            "best_round": results.get("best_round", 1),
            "final_epsilon": round(float(results.get("final_epsilon", 0.0)), 2) if results.get("final_epsilon") != float("inf") else "Unlimited",
            "dp_enabled": enable_dp,
            "secagg_enabled": enable_secagg,
            "convergence": convergence_series,
            "final_eval": results.get("final_eval", {}),
        }
