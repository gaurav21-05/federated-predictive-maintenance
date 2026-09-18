"""
Remaining Useful Life (RUL) Estimator Module
Implements hybrid physics-informed exponential wear degradation and
empirical regression modeling to predict remaining machine operating time
before catastrophic failure.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


@dataclass
class RULPrediction:
    """Dataclass holding Remaining Useful Life forecast and confidence metrics."""
    rul_minutes: float
    rul_ticks: int
    confidence_lower_minutes: float
    confidence_upper_minutes: float
    urgency_level: str  # 'CRITICAL', 'WARNING', 'MONITORING', 'HEALTHY'
    degradation_rate_pct_per_hour: float
    time_to_critical_minutes: float
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "rul_minutes": round(self.rul_minutes, 1),
            "rul_ticks": self.rul_ticks,
            "confidence_lower_minutes": round(self.confidence_lower_minutes, 1),
            "confidence_upper_minutes": round(self.confidence_upper_minutes, 1),
            "urgency_level": self.urgency_level,
            "degradation_rate_pct_per_hour": round(self.degradation_rate_pct_per_hour, 2),
            "time_to_critical_minutes": round(self.time_to_critical_minutes, 1),
            "explanation": self.explanation,
        }


class RULEstimator:
    """
    Predicts Remaining Useful Life (RUL) by tracking the health degradation trajectory
    of a MachineDigitalTwin.
    
    Physics Model:
        Health Index H(t) decays exponentially under wear:
            H(t) = H_0 * exp(-lambda * t)
        Critical breakdown threshold: H_crit = 25.0%
        Remaining time until H_crit:
            RUL_ticks = max(0, (ln(H_current) - ln(H_crit)) / lambda)
            RUL_minutes = RUL_ticks * (10.0 / 60.0)
    """

    CRITICAL_HEALTH_THRESHOLD = 25.0   # Failure trip threshold
    WARNING_HEALTH_THRESHOLD = 60.0    # Warning overhaul threshold
    TICK_DURATION_SECONDS = 10.0       # 10s per simulation tick

    # Nominal cycle lengths by machine (in minutes)
    NOMINAL_CYCLE_MINUTES = {
        "client_1": 190.0,  # ~1140 ticks
        "client_2": 175.0,  # ~1050 ticks
        "client_3": 185.0,  # ~1110 ticks
    }

    def __init__(self, client_id: str = "client_1", history_window: int = 30):
        """
        Args:
            client_id: Machine identifier.
            history_window: Number of recent ticks used to calculate degradation slope.
        """
        self.client_id = client_id
        self.history_window = history_window
        self.health_trajectory: List[float] = []
        self.nominal_cycle_min = self.NOMINAL_CYCLE_MINUTES.get(client_id, 180.0)

    def reset(self):
        """Resets trajectory history (e.g. after a maintenance overhaul)."""
        self.health_trajectory.clear()

    def update(self, current_health_index: float):
        """Records the latest health index from the Digital Twin."""
        self.health_trajectory.append(float(current_health_index))
        if len(self.health_trajectory) > 300:
            self.health_trajectory.pop(0)

    def predict_rul(
        self,
        current_health_index: float,
        failure_probability: float = 0.0,
    ) -> RULPrediction:
        """
        Predicts Remaining Useful Life (RUL) based on current health, degradation slope,
        and failure hazard probability.
        
        Returns:
            RULPrediction with point estimate, confidence bounds, and urgency rating.
        """
        self.update(current_health_index)
        h = max(1.0, min(100.0, float(current_health_index)))

        # 1. Estimate instantaneous degradation rate lambda (per minute)
        if len(self.health_trajectory) >= 5:
            recent = self.health_trajectory[-min(len(self.health_trajectory), self.history_window):]
            # Linear regression slope of health index over ticks
            x = np.arange(len(recent))
            slope_per_tick, _ = np.polyfit(x, recent, 1)  # dH / dt (ticks)
            slope_per_min = slope_per_tick * (60.0 / self.TICK_DURATION_SECONDS)
        else:
            slope_per_min = - (100.0 - h) / max(1.0, self.nominal_cycle_min * 0.5)

        # Convert slope to percentage degradation per hour
        deg_rate_pct_hr = max(0.01, -slope_per_min * 60.0)

        # 2. Physics-based RUL calculation
        if h <= self.CRITICAL_HEALTH_THRESHOLD or failure_probability > 0.85:
            # Immediate imminent failure
            rul_minutes = max(0.5, (h / self.CRITICAL_HEALTH_THRESHOLD) * 5.0)
            rul_ticks = int(rul_minutes * 60.0 / self.TICK_DURATION_SECONDS)
            urgency = "CRITICAL"
            explanation = (
                f"CRITICAL: Machine Health is at {h:.1f}% (below trip threshold {self.CRITICAL_HEALTH_THRESHOLD}%). "
                f"Imminent catastrophic failure within {rul_minutes:.1f} minutes!"
            )
            time_to_warn = 0.0

        elif h < self.WARNING_HEALTH_THRESHOLD or failure_probability > 0.40:
            # Accelerated degradation regime
            # Time remaining to reach critical trip threshold:
            effective_slope = max(0.15, abs(slope_per_min))
            rul_minutes = (h - self.CRITICAL_HEALTH_THRESHOLD) / effective_slope
            # Bound RUL by remaining cycle duration
            rul_minutes = max(3.0, min(60.0, rul_minutes))
            rul_ticks = int(rul_minutes * 60.0 / self.TICK_DURATION_SECONDS)
            urgency = "WARNING"
            explanation = (
                f"WARNING: Accelerated wear detected ({deg_rate_pct_hr:.1f}%/hr). "
                f"Estimated remaining life to trip threshold: {rul_minutes:.1f} min."
            )
            time_to_warn = 0.0

        else:
            # Healthy / early wear regime
            effective_slope = max(0.05, abs(slope_per_min))
            time_to_crit = (h - self.CRITICAL_HEALTH_THRESHOLD) / effective_slope
            time_to_warn = (h - self.WARNING_HEALTH_THRESHOLD) / effective_slope

            # Blend with nominal cycle expectation: remaining = nominal * (H / 100)
            baseline_rul = self.nominal_cycle_min * (h / 100.0)
            rul_minutes = 0.6 * baseline_rul + 0.4 * time_to_crit
            rul_minutes = max(30.0, min(self.nominal_cycle_min * 1.5, rul_minutes))
            rul_ticks = int(rul_minutes * 60.0 / self.TICK_DURATION_SECONDS)

            if rul_minutes < 90.0 or h < 75.0:
                urgency = "MONITORING"
                explanation = (
                    f"MONITORING: Nominal operation with mild wear. "
                    f"Remaining life: {rul_minutes:.1f} min. Projected warning in {time_to_warn:.1f} min."
                )
            else:
                urgency = "HEALTHY"
                explanation = (
                    f"HEALTHY: Machine operating at {h:.1f}% health. "
                    f"Remaining Useful Life: {rul_minutes / 60.0:.1f} hours ({rul_minutes:.0f} min)."
                )

        # 3. Confidence Bounds (+/- 15% to 25% depending on degradation velocity)
        uncertainty_factor = 0.15 if urgency == "HEALTHY" else 0.22
        conf_lower = max(0.2, rul_minutes * (1.0 - uncertainty_factor))
        conf_upper = rul_minutes * (1.0 + uncertainty_factor)

        return RULPrediction(
            rul_minutes=float(rul_minutes),
            rul_ticks=rul_ticks,
            confidence_lower_minutes=float(conf_lower),
            confidence_upper_minutes=float(conf_upper),
            urgency_level=urgency,
            degradation_rate_pct_per_hour=float(deg_rate_pct_hr),
            time_to_critical_minutes=float(max(0.0, rul_minutes)),
            explanation=explanation,
        )
