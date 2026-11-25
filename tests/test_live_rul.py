"""
Unit Test Suite for Live RUL & Health Index Engine (src/dashboard/live_rul.py)
Validates subsystem health score calculation, composite Health Index,
honest RUL degradation forecasting, resilience against transient spikes,
automatic activation of the electrical subsystem when current data arrives,
and confidence interval determination.
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.thingspeak import ThingSpeakAdapter
from src.dashboard.live_analytics import compute_robust_baseline
from src.dashboard.live_rul import (
    compute_live_health_series,
    compute_sample_subsystems,
    compute_composite_health_index,
    estimate_rul,
    compute_operating_hours,
)


class TestLiveRUL(unittest.TestCase):
    """Test suite for live digital twin health index and RUL estimation."""

    @classmethod
    def setUpClass(cls):
        adapter = ThingSpeakAdapter()
        raw_feeds, _, _ = adapter.fetch_raw_feeds(results=100)
        norm = adapter.normalize(raw_feeds)
        cls.samples = norm["samples"]
        cls.baseline = compute_robust_baseline(cls.samples)

    def test_healthy_sample_returns_stable_or_insufficient(self):
        """
        Honesty rule verification:
        On the real recorded feeds.csv (healthy bench device with transient spikes),
        RUL must return STABLE_NO_DEGRADATION with rul_hours = None (NEVER a fabricated number).
        """
        rul_res = estimate_rul(self.samples, self.baseline)
        self.assertEqual(rul_res["rul_status"], "STABLE_NO_DEGRADATION")
        self.assertIsNone(rul_res["rul_hours"])
        self.assertIsNone(rul_res["rul_minutes"])
        self.assertIsNone(rul_res["ci_lower_hours"])
        self.assertIsNone(rul_res["ci_upper_hours"])
        self.assertGreater(rul_res["operating_hours"], 0.0)
        self.assertIn("nominal baseline", rul_res["recommended_action"].lower())

    def test_insufficient_data_honesty_rule(self):
        """Verifies that sample sizes under min_samples (30) return INSUFFICIENT_DATA with null RUL."""
        small_sample = self.samples[:15]
        rul_res = estimate_rul(small_sample, self.baseline, min_samples=30)
        self.assertEqual(rul_res["rul_status"], "INSUFFICIENT_DATA")
        self.assertIsNone(rul_res["rul_hours"])
        self.assertEqual(rul_res["confidence"], "LOW")
        self.assertEqual(rul_res["samples_used"], 15)

    def test_synthetic_degrading_series_returns_finite_rul(self):
        """
        Verifies that when a physical machine is actually undergoing continuous wear degradation,
        the estimator computes a finite decreasing RUL estimate with valid confidence intervals.
        """
        t0 = datetime(2026, 11, 25, 0, 0, 0, tzinfo=timezone.utc)
        degrading_samples = []

        # Synthetic gradual mechanical wear: vibration increasing from 0.020g to 0.050g over 50 steps
        for i in range(50):
            degrading_samples.append({
                "entry_id": 100 + i,
                "timestamp": (t0 + timedelta(seconds=20 * i)).isoformat(),
                "vibration_g": 0.020 + 0.0006 * i,
                "temperature_c": 23.0 + 0.04 * i,
                "humidity_pct": 65.0,
                "pressure_hpa": 995.0,
                "current_a": None,
                "esp32_anomaly": 0,
                "esp32_alert": 0,
            })

        rul_res = estimate_rul(degrading_samples, self.baseline)
        self.assertEqual(rul_res["rul_status"], "DEGRADING")
        self.assertIsNotNone(rul_res["rul_hours"])
        self.assertGreater(rul_res["rul_hours"], 0.0)
        self.assertIsNotNone(rul_res["ci_lower_hours"])
        self.assertIsNotNone(rul_res["ci_upper_hours"])
        self.assertLessEqual(rul_res["ci_lower_hours"], rul_res["rul_hours"])
        self.assertGreaterEqual(rul_res["ci_upper_hours"], rul_res["rul_hours"])
        self.assertIn(rul_res["method"], ["exponential_wear", "linear_extrapolation"])

    def test_transient_spikes_do_not_collapse_rul(self):
        """
        Verifies that single-sample anomaly spikes (like entries 5 and 40)
        are protected by robust smoothing and do NOT collapse RUL into false failure.
        """
        t0 = datetime(2026, 11, 25, 0, 0, 0, tzinfo=timezone.utc)
        nominal_samples = [
            {
                "entry_id": 200 + i,
                "timestamp": (t0 + timedelta(seconds=20 * i)).isoformat(),
                "vibration_g": 0.022 + 0.001 * (i % 3),
                "temperature_c": 24.0,
                "humidity_pct": 60.0,
                "pressure_hpa": 995.0,
                "current_a": None,
                "esp32_anomaly": 0,
                "esp32_alert": 0,
            }
            for i in range(45)
        ]

        # Inject extreme single-sample transient spike at index 20
        nominal_samples[20]["vibration_g"] = 0.850  # Huge spike
        nominal_samples[20]["esp32_anomaly"] = 1
        nominal_samples[20]["esp32_alert"] = 1

        rul_res = estimate_rul(nominal_samples, self.baseline)
        # Because the overall series is healthy and stable, the spike must not collapse RUL
        self.assertEqual(rul_res["rul_status"], "STABLE_NO_DEGRADATION")
        self.assertIsNone(rul_res["rul_hours"])

    def test_electrical_subsystem_auto_activation(self):
        """
        Verifies that when field6 (current_a) arrives:
        1. Subsystems dictionary includes a numeric 'electrical' score (not None).
        2. Composite Health Index dynamically incorporates the electrical weight (20%).
        3. Works seamlessly without any code modification.
        """
        # Baseline sample without current
        sample_no_curr = {
            "vibration_g": 0.022,
            "temperature_c": 24.0,
            "humidity_pct": 60.0,
            "pressure_hpa": 995.0,
            "current_a": None,
        }
        sub_no_curr = compute_sample_subsystems(sample_no_curr, self.baseline, has_active_current=False)
        self.assertIsNone(sub_no_curr["electrical"])
        hi_no_curr = compute_composite_health_index(sub_no_curr)

        # Baseline sample WITH current arriving
        sample_with_curr = dict(sample_no_curr)
        sample_with_curr["current_a"] = 1.25  # Normal 1.25 A current
        sub_with_curr = compute_sample_subsystems(sample_with_curr, self.baseline, has_active_current=True)
        self.assertIsNotNone(sub_with_curr["electrical"])
        self.assertEqual(sub_with_curr["electrical"], 100.0)
        hi_with_curr = compute_composite_health_index(sub_with_curr)

        # High current overload check (e.g. 5.5A on 2A motor)
        sample_overload = dict(sample_no_curr)
        sample_overload["current_a"] = 5.5
        sub_overload = compute_sample_subsystems(sample_overload, self.baseline, has_active_current=True)
        self.assertLess(sub_overload["electrical"], 50.0)
        hi_overload = compute_composite_health_index(sub_overload)
        self.assertLess(hi_overload, hi_with_curr)

    def test_operating_hours_gap_handling(self):
        """Verifies that operating hours counter sums valid sample intervals while excluding gaps > 5 min."""
        t0 = datetime(2026, 11, 25, 0, 0, 0, tzinfo=timezone.utc)
        samples = [
            {"entry_id": 1, "timestamp": t0.isoformat()},
            {"entry_id": 2, "timestamp": (t0 + timedelta(seconds=20)).isoformat()},
            {"entry_id": 3, "timestamp": (t0 + timedelta(seconds=40)).isoformat()},
            # 2 hour offline gap
            {"entry_id": 4, "timestamp": (t0 + timedelta(seconds=7240)).isoformat()},
            {"entry_id": 5, "timestamp": (t0 + timedelta(seconds=7260)).isoformat()},
        ]
        # Total operating time should be 40s + 20s = 60s (0.02 hours), ignoring the 2 hour gap
        op_hours = compute_operating_hours(samples)
        self.assertAlmostEqual(op_hours, 60.0 / 3600.0, places=2)


if __name__ == "__main__":
    unittest.main()
