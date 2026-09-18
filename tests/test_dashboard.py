"""
Unit & Integration Test Suite for Phase 6: Web Dashboard & Backend Service Layer
Validates RESTful API endpoints, Digital Twin step synchronization, What-If simulation physics,
Prescriptive work orders, and on-demand Federated Learning execution.
"""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.dashboard.app import app, service
from src.dashboard.service import DashboardService


class TestDashboardService(unittest.TestCase):
    """Tests for the DashboardService state and simulation layer."""

    @classmethod
    def setUpClass(cls):
        cls.srv = DashboardService()

    def test_overview_structure(self):
        """Verifies that get_overview returns complete multi-machine metrics."""
        overview = self.srv.get_overview()
        self.assertIn("fleet_health_index", overview)
        self.assertIn("fleet_status", overview)
        self.assertIn("machines", overview)
        self.assertEqual(len(overview["machines"]), 3)

        m0 = overview["machines"][0]
        self.assertIn("overall_health_index", m0)
        self.assertIn("rul_minutes", m0)
        self.assertIn("subsystem_health", m0)
        self.assertIn("operational_state", m0)

    def test_telemetry_stream_window(self):
        """Verifies that telemetry window returns non-empty records."""
        stream = self.srv.get_telemetry_window("client_1", window=30)
        self.assertEqual(stream["client_id"], "client_1")
        self.assertIn("telemetry", stream)
        self.assertIn("state_snapshot", stream)
        self.assertIn("rul_prediction", stream)
        self.assertGreater(len(stream["telemetry"]), 0)

    def test_step_simulation(self):
        """Verifies that step() advances ticks and updates health indices."""
        initial_step = self.srv.current_step["client_1"]
        res = self.srv.step("client_1", steps=5)
        self.assertEqual(res["status"], "success")
        self.assertEqual(self.srv.current_step["client_1"], initial_step + 5)

    def test_what_if_simulation(self):
        """Verifies What-If scenario simulation physics and lifespan impact."""
        res = self.srv.run_what_if("client_1", scenario_key="production_surge")
        self.assertIn("simulation", res)
        sim = res["simulation"]
        self.assertGreater(sim["wear_acceleration_factor"], 1.0)
        self.assertLess(sim["rul_delta_minutes"], 0.0)  # RUL should decrease under surge
        self.assertIn("recommendation", sim)

    def test_work_order_workflow(self):
        """Verifies manual dispatch and status transitions of work orders."""
        # Dispatch dummy order via dispatcher
        from src.digital_twin.maintenance_dispatcher import MaintenanceWorkOrder
        dummy_order = MaintenanceWorkOrder(
            work_order_id="WO-TEST-001",
            machine_id="CNC_MILL_01",
            machine_name="CNC Milling Spindle",
            timestamp="2026-09-17 12:00:00",
            urgency="P2_HIGH",
            affected_subsystem="vibration",
            machine_health_index=55.0,
            estimated_rul_minutes=42.0,
            recommended_action="Inspect spindle bearings",
            estimated_downtime_hours=2.5,
            required_spare_parts=["Bearings #6205"],
            technician_skill_level="Level 3",
            safety_precautions=["Lockout-Tagout"],
            status="OPEN",
        )
        self.srv.dispatcher.work_orders.append(dummy_order)

        # Query orders
        orders = self.srv.get_work_orders(status="OPEN")
        test_wo = [o for o in orders if o["work_order_id"] == "WO-TEST-001"]
        self.assertEqual(len(test_wo), 1)

        # Update status
        upd = self.srv.update_work_order_status("WO-TEST-001", "RESOLVED")
        self.assertEqual(upd["status"], "success")
        self.assertEqual(upd["work_order"]["status"], "RESOLVED")

    def test_fast_federated_training_run(self):
        """Verifies on-demand federated learning execution on sample data."""
        res = self.srv.run_federated_training(
            strategy_name="FedProx",
            num_rounds=1,
            local_epochs=1,
            enable_dp=True,
            clip_norm=1.0,
            noise_multiplier=0.05,
            enable_secagg=True,
            mu=0.01,
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["strategy"], "FedProx")
        self.assertEqual(res["rounds_completed"], 1)
        self.assertIn("best_macro_f1", res)
        self.assertIn("convergence", res)


class TestDashboardAPI(unittest.TestCase):
    """Tests for FastAPI HTTP routes."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_overview_endpoint(self):
        resp = self.client.get("/api/overview")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("fleet_health_index", data)

    def test_get_telemetry_stream_endpoint(self):
        resp = self.client.get("/api/telemetry/stream?client_id=client_1&window=20")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["client_id"], "client_1")
        self.assertIn("telemetry", data)

    def test_post_step_endpoint(self):
        resp = self.client.post("/api/digital-twin/step", json={"client_id": "client_1", "steps": 2})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")

    def test_post_what_if_endpoint(self):
        resp = self.client.post("/api/digital-twin/what-if", json={
            "client_id": "client_1",
            "scenario_key": "summer_heatwave",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("simulation", data)

    def test_get_work_orders_endpoint(self):
        resp = self.client.get("/api/maintenance/orders")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_serve_index_html(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("IIoT Edge Intelligence", resp.text)


if __name__ == "__main__":
    unittest.main()
