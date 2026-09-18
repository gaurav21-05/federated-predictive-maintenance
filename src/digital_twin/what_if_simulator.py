"""
What-If Stress Simulation Engine Module
Allows plant engineers to simulate operational stress scenarios (overload, heatwaves,
or de-rating) on the Machine Digital Twin to evaluate impacts on Remaining Useful Life (RUL)
without putting physical machinery at risk.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .twin_engine import MachineDigitalTwin
from .rul_estimator import RULEstimator


@dataclass
class SimulationScenario:
    """Defines a hypothetical operational condition to evaluate."""
    name: str
    description: str
    rpm_multiplier: float = 1.0       # e.g. 1.20 for +20% speed
    load_multiplier: float = 1.0      # e.g. 1.15 for +15% torque/cutting force
    ambient_temp_delta: float = 0.0   # e.g. +10.0 deg C ambient rise
    horizon_minutes: float = 60.0     # Forward projection window in minutes


class WhatIfSimulator:
    """
    Simulates forward wear trajectories and RUL impact under hypothetical operating scenarios.
    """

    # Pre-defined industrial scenarios
    PRESETS = {
        "production_surge": SimulationScenario(
            name="Production Surge (+20% RPM, +15% Load)",
            description="Accelerated batch quota: spindle RPM increased by 20% and cutting load by 15%.",
            rpm_multiplier=1.20,
            load_multiplier=1.15,
            ambient_temp_delta=0.0,
            horizon_minutes=60.0,
        ),
        "summer_heatwave": SimulationScenario(
            name="Summer Heatwave (+12°C Ambient)",
            description="HVAC chiller degradation with ambient temperature spiking by 12°C.",
            rpm_multiplier=1.00,
            load_multiplier=1.00,
            ambient_temp_delta=12.0,
            horizon_minutes=60.0,
        ),
        "protective_derating": SimulationScenario(
            name="Protective De-Rating (-20% RPM, -25% Load)",
            description="Throttle machine speed down to safely limp to the weekend maintenance window.",
            rpm_multiplier=0.80,
            load_multiplier=0.75,
            ambient_temp_delta=0.0,
            horizon_minutes=120.0,
        ),
    }

    def __init__(self, rul_estimator: Optional[RULEstimator] = None):
        self.rul_estimator = rul_estimator or RULEstimator()

    def simulate(
        self,
        twin: MachineDigitalTwin,
        scenario: SimulationScenario,
        time_steps: int = 12,
    ) -> Dict:
        """
        Projects future machine health and RUL under baseline vs. scenario stress.
        
        Args:
            twin: Current synchronized Digital Twin state.
            scenario: The What-If simulation parameters.
            time_steps: Number of forward projection intervals.
            
        Returns:
            Dictionary comparing baseline vs scenario trajectory, wear factor, and revised RUL.
        """
        current_mhi = twin.overall_health_index
        current_sensors = twin.latest_telemetry

        # 1. Baseline RUL projection
        baseline_pred = self.rul_estimator.predict_rul(current_mhi, twin.failure_probability)
        baseline_rul_min = baseline_pred.rul_minutes

        # 2. Physics-based Stress Acceleration Factor (kappa)
        # Wear rate scales non-linearly with rotational speed (vibration ~ RPM^1.8) and load
        rpm_factor = (scenario.rpm_multiplier) ** 2.2
        load_factor = (scenario.load_multiplier) ** 1.8
        thermal_factor = 1.0 + max(0.0, scenario.ambient_temp_delta * 0.035)

        wear_acceleration_factor = float(rpm_factor * load_factor * thermal_factor)

        # 3. Projected Scenario RUL
        # Faster wear reduces remaining time inversely:
        scenario_rul_min = max(1.0, baseline_rul_min / max(0.1, wear_acceleration_factor))
        rul_delta_min = scenario_rul_min - baseline_rul_min
        rul_delta_pct = ((scenario_rul_min - baseline_rul_min) / max(1e-4, baseline_rul_min)) * 100.0

        # 4. Forward Health Trajectory Projections
        dt_interval = scenario.horizon_minutes / max(1, time_steps)
        time_axis = [round(i * dt_interval, 1) for i in range(time_steps + 1)]

        # Baseline MHI trajectory
        baseline_slope = - (100.0 - 25.0) / max(1.0, baseline_rul_min)
        baseline_mhi_curve = [
            round(float(np.clip(current_mhi + baseline_slope * t, 0.0, 100.0)), 2)
            for t in time_axis
        ]

        # Scenario MHI trajectory
        scenario_slope = baseline_slope * wear_acceleration_factor
        scenario_mhi_curve = [
            round(float(np.clip(current_mhi + scenario_slope * t, 0.0, 100.0)), 2)
            for t in time_axis
        ]

        # 5. Operational Safety Recommendation
        if scenario_rul_min < 20.0:
            recommendation = (
                f"REJECT SCENARIO: RUL drops dangerously to {scenario_rul_min:.1f} min. "
                f"Breakdown imminent before horizon completes."
            )
            risk_level = "HIGH_RISK"
        elif wear_acceleration_factor > 1.5:
            recommendation = (
                f"CAUTION: Wear accelerates by {wear_acceleration_factor:.2f}x. "
                f"RUL decreases by {abs(rul_delta_min):.1f} min. Limit run to < {scenario_rul_min * 0.6:.0f} min."
            )
            risk_level = "MODERATE_RISK"
        elif wear_acceleration_factor < 0.8:
            recommendation = (
                f"APPROVED (LIFE EXTENSION): De-rating extends machine life by {rul_delta_min:+.1f} min "
                f"({rul_delta_pct:+.1f}%). Safe to postpone maintenance overhaul."
            )
            risk_level = "BENEFICIAL"
        else:
            recommendation = "NEUTRAL: Minor operational impact within acceptable machinery margins."
            risk_level = "LOW_RISK"

        return {
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "wear_acceleration_factor": round(wear_acceleration_factor, 3),
            "baseline_rul_minutes": round(baseline_rul_min, 1),
            "projected_rul_minutes": round(scenario_rul_min, 1),
            "rul_delta_minutes": round(rul_delta_min, 1),
            "rul_delta_percent": round(rul_delta_pct, 1),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "projection_timeline_minutes": time_axis,
            "baseline_mhi_curve": baseline_mhi_curve,
            "scenario_mhi_curve": scenario_mhi_curve,
        }
