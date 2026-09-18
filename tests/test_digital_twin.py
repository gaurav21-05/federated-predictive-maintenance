"""
Unit Test Suite for Phase 5: Digital Twin Synchronization & RUL Estimation
Validates MachineDigitalTwin state sync, health index tracking, RULEstimator forecasts,
What-If stress simulation physics, and Prescriptive Maintenance Dispatching.
"""

import os
import sys
from pathlib import Path
import unittest
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLIENT_PROFILES
from src.digital_twin.twin_engine import MachineDigitalTwin
from src.digital_twin.rul_estimator import RULEstimator
from src.digital_twin.what_if_simulator import WhatIfSimulator, SimulationScenario
from src.digital_twin.maintenance_dispatcher import PrescriptiveMaintenanceDispatcher


class TestMachineDigitalTwin(unittest.TestCase):
    """Tests for MachineDigitalTwin virtual state synchronization."""

    def setUp(self):
        self.profile = CLIENT_PROFILES["client_1"]  # CNC Milling Machine
        self.twin = MachineDigitalTwin(client_id="client_1", profile=self.profile, window_size=30)

    def test_nominal_telemetry_health(self):
        """Verifies that pristine baseline telemetry maintains near 100% health."""
        nominal_point = {
            "temperature": self.profile.temperature.mean,
            "humidity": self.profile.humidity.mean,
            "vibration": self.profile.vibration.mean,
            "current": self.profile.current.mean,
            "rpm": self.profile.rpm.mean,
            "pressure": self.profile.pressure.mean,
            "failure": 0,
        }
        state = self.twin.ingest_telemetry_point(nominal_point)

        self.assertGreater(state["overall_health_index"], 95.0)
        self.assertEqual(state["operational_state"], "NORMAL")
        self.assertIn("Healthy", state["primary_fault_subsystem"])

    def test_vibration_spike_degradation(self):
        """Verifies that severe vibration triggers health degradation and fault localization."""
        spike_point = {
            "temperature": self.profile.temperature.mean,
            "humidity": self.profile.humidity.mean,
            "vibration": self.profile.vibration.mean + 5.0 * self.profile.vibration.std,  # Massive vibration
            "current": self.profile.current.mean,
            "rpm": self.profile.rpm.mean,
            "pressure": self.profile.pressure.mean,
            "failure": 0,
        }
        state = self.twin.ingest_telemetry_point(spike_point)

        self.assertLess(state["subsystem_health"]["vibration"], 60.0)
        self.assertLess(state["overall_health_index"], 85.0)
        self.assertIn("Spindle Bearing", state["primary_fault_subsystem"])

    def test_buffer_sliding_window(self):
        """Verifies that buffer strictly adheres to window_size=30."""
        nominal_point = {
            "temperature": self.profile.temperature.mean,
            "humidity": self.profile.humidity.mean,
            "vibration": self.profile.vibration.mean,
            "current": self.profile.current.mean,
            "rpm": self.profile.rpm.mean,
            "pressure": self.profile.pressure.mean,
            "failure": 0,
        }
        for _ in range(50):
            self.twin.ingest_telemetry_point(nominal_point)

        self.assertEqual(len(self.twin.buffer), 30)
        self.assertEqual(self.twin.total_ticks_processed, 50)


class TestRULEstimator(unittest.TestCase):
    """Tests for RULEstimator remaining life forecasting."""

    def setUp(self):
        self.estimator = RULEstimator(client_id="client_1", history_window=20)

    def test_healthy_rul_prediction(self):
        """Verifies RUL forecast in healthy operational state."""
        # Feed gradual healthy wear
        for h in np.linspace(98.0, 92.0, 15):
            pred = self.estimator.predict_rul(h, failure_probability=0.02)

        self.assertGreater(pred.rul_minutes, 100.0)
        self.assertEqual(pred.urgency_level, "HEALTHY")
        self.assertLessEqual(pred.confidence_lower_minutes, pred.rul_minutes)
        self.assertGreaterEqual(pred.confidence_upper_minutes, pred.rul_minutes)

    def test_critical_rul_prediction(self):
        """Verifies RUL forecast drops sharply when health is near breakdown."""
        # Feed steep abnormal drop
        for h in np.linspace(50.0, 18.0, 15):
            pred = self.estimator.predict_rul(h, failure_probability=0.90)

        self.assertLess(pred.rul_minutes, 15.0)
        self.assertEqual(pred.urgency_level, "CRITICAL")
        self.assertIn("CRITICAL", pred.explanation)


class TestWhatIfSimulator(unittest.TestCase):
    """Tests for What-If stress simulation engine."""

    def setUp(self):
        self.twin = MachineDigitalTwin("client_1")
        # Ingest a point to establish state
        self.twin.ingest_telemetry_point({
            "temperature": 62.0, "humidity": 45.0, "vibration": 1.2,
            "current": 18.5, "rpm": 3850.0, "pressure": 6.2, "failure": 0
        })
        self.simulator = WhatIfSimulator()

    def test_overdrive_accelerates_wear(self):
        """Verifies that +25% RPM overdrive increases wear factor and reduces projected RUL."""
        scenario = SimulationScenario(
            name="Test Overdrive",
            description="Testing speed surge",
            rpm_multiplier=1.25,
            load_multiplier=1.15,
            ambient_temp_delta=5.0,
            horizon_minutes=60.0,
        )
        res = self.simulator.simulate(self.twin, scenario)

        self.assertGreater(res["wear_acceleration_factor"], 1.5)
        self.assertLess(res["projected_rul_minutes"], res["baseline_rul_minutes"])
        self.assertLess(res["rul_delta_minutes"], 0.0)

    def test_derating_extends_life(self):
        """Verifies that -20% speed de-rating decelerates wear and extends projected RUL."""
        scenario = SimulationScenario(
            name="Test Derating",
            description="Testing speed de-rating",
            rpm_multiplier=0.80,
            load_multiplier=0.80,
            ambient_temp_delta=0.0,
            horizon_minutes=60.0,
        )
        res = self.simulator.simulate(self.twin, scenario)

        self.assertLess(res["wear_acceleration_factor"], 0.8)
        self.assertGreater(res["projected_rul_minutes"], res["baseline_rul_minutes"])
        self.assertGreater(res["rul_delta_minutes"], 0.0)
        self.assertEqual(res["risk_level"], "BENEFICIAL")


class TestPrescriptiveMaintenanceDispatcher(unittest.TestCase):
    """Tests for PrescriptiveMaintenanceDispatcher work order creation."""

    def setUp(self):
        self.twin = MachineDigitalTwin("client_1")
        self.dispatcher = PrescriptiveMaintenanceDispatcher(dispatch_cooldown_ticks=20)
        self.estimator = RULEstimator("client_1")

    def test_no_dispatch_when_healthy(self):
        """Verifies that healthy machinery does not trigger work orders."""
        self.twin.ingest_telemetry_point({
            "temperature": 62.0, "humidity": 45.0, "vibration": 1.2,
            "current": 18.5, "rpm": 3850.0, "pressure": 6.2, "failure": 0
        })
        pred = self.estimator.predict_rul(self.twin.overall_health_index)
        wo = self.dispatcher.evaluate_twin(self.twin, pred, current_tick=1)

        self.assertIsNone(wo)
        self.assertEqual(len(self.dispatcher.work_orders), 0)

    def test_dispatch_when_critical(self):
        """Verifies that severe degradation triggers an immediate P1_CRITICAL work order."""
        # Inject abnormal vibration
        self.twin.ingest_telemetry_point({
            "temperature": 85.0, "humidity": 45.0, "vibration": 6.5,
            "current": 32.0, "rpm": 3600.0, "pressure": 5.0, "failure": 0
        })
        pred = self.estimator.predict_rul(self.twin.overall_health_index, failure_probability=0.85)
        wo = self.dispatcher.evaluate_twin(self.twin, pred, current_tick=1)

        self.assertIsNotNone(wo)
        self.assertEqual(wo.urgency, "P1_CRITICAL")
        self.assertIn("WO-CLIENT_1", wo.work_order_id)
        self.assertGreater(len(wo.required_spare_parts), 0)
        self.assertGreater(len(wo.safety_precautions), 0)

    def test_cooldown_suppression(self):
        """Verifies that cooldown prevents duplicate work order flooding on consecutive ticks."""
        self.twin.overall_health_index = 30.0
        self.twin.failure_probability = 0.90
        pred = self.estimator.predict_rul(30.0, failure_probability=0.90)

        wo1 = self.dispatcher.evaluate_twin(self.twin, pred, current_tick=10)
        self.assertIsNotNone(wo1)

        # Immediate next tick should be suppressed
        wo2 = self.dispatcher.evaluate_twin(self.twin, pred, current_tick=11)
        self.assertIsNone(wo2)

        # After cooldown passes (10 + 20 = 30), new work order can fire
        wo3 = self.dispatcher.evaluate_twin(self.twin, pred, current_tick=35)
        self.assertIsNotNone(wo3)


if __name__ == "__main__":
    unittest.main()
