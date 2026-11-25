"""
Unit Test Suite for Live Analytics Engine (src/dashboard/live_analytics.py)
Validates descriptive statistics, robust baseline estimation, tri-method anomaly detection,
detection of known hardware spikes (entries 5 & 40), episode clustering, trend significance,
correlation analysis, and data quality diagnostics.
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.thingspeak import ThingSpeakAdapter
from src.dashboard.live_analytics import (
    compute_descriptive_stats,
    compute_robust_baseline,
    detect_anomalies,
    detect_anomaly_episodes,
    analyze_trends,
    compute_correlations,
    generate_data_quality_report,
)


class TestLiveAnalytics(unittest.TestCase):
    """Test suite for live statistical analysis and anomaly detection."""

    @classmethod
    def setUpClass(cls):
        # Load and normalize real recorded feeds.csv
        adapter = ThingSpeakAdapter()
        raw_feeds, _, _ = adapter.fetch_raw_feeds(results=100)
        norm = adapter.normalize(raw_feeds)
        cls.samples = norm["samples"]
        cls.health = norm["health"]
        cls.baseline = compute_robust_baseline(cls.samples)

    def test_na_handling(self):
        """Verifies that missing values and 'NA' strings are handled gracefully without raising errors."""
        stats = compute_descriptive_stats(self.samples)
        # Field 6 (current_a) is completely NA in feeds.csv
        self.assertIn("current_a", stats)
        curr_stat = stats["current_a"]
        self.assertFalse(curr_stat["connected"])
        self.assertEqual(curr_stat["valid_count"], 0)
        self.assertIsNone(curr_stat["mean"])
        self.assertEqual(curr_stat["missing_pct"], 100.0)

        # Vibration and Temperature must be present and valid
        vib_stat = stats["vibration_g"]
        self.assertTrue(vib_stat["connected"])
        self.assertGreater(vib_stat["valid_count"], 0)
        self.assertIsNotNone(vib_stat["mean"])
        self.assertIsNotNone(vib_stat["std"])

    def test_robust_baseline_calculation(self):
        """Verifies robust Median/MAD baseline metrics."""
        self.assertIn("vibration_g", self.baseline)
        vib_b = self.baseline["vibration_g"]
        self.assertAlmostEqual(vib_b["median"], 0.02294, places=3)
        self.assertGreater(vib_b["sigma_mad"], 0.0)
        self.assertGreater(vib_b["upper_3sigma"], vib_b["median"])

    def test_anomaly_detection_catches_known_spikes(self):
        """
        Validates that tri-method anomaly detection successfully catches the known
        physical hardware spikes at entry 5 (0.615g) and entry 40 (0.300g).
        """
        anom_res = detect_anomalies(self.samples, self.baseline)
        anomalies = anom_res["anomalies"]
        flagged_entries = {a["entry_id"]: a for a in anomalies}

        # Check Entry 5
        self.assertIn(5, flagged_entries, "Entry 5 spike (0.615g) was not flagged as anomaly!")
        self.assertEqual(flagged_entries[5]["severity"], "CRITICAL")
        self.assertGreater(flagged_entries[5]["z_score"], 10.0)

        # Check Entry 40
        self.assertIn(40, flagged_entries, "Entry 40 spike (0.300g) was not flagged as anomaly!")
        self.assertEqual(flagged_entries[40]["severity"], "CRITICAL")
        self.assertIn("robust_zscore", flagged_entries[40]["methods_fired"])

        # Check Reconciliation
        recon = anom_res["reconciliation"]
        self.assertIn("agreement_rate_pct", recon)
        self.assertGreater(recon["both_flagged_count"], 0)
        # ESP32 alerts should be fully covered
        self.assertEqual(recon["false_negative_vs_esp32"], 0)

    def test_anomaly_episodes_clustering(self):
        """Verifies that anomalous samples are grouped into contiguous episodes."""
        anom_res = detect_anomalies(self.samples, self.baseline)
        episodes = detect_anomaly_episodes(self.samples, anom_res["anomalies"])
        self.assertIsInstance(episodes, list)
        self.assertGreater(len(episodes), 0)

        # First episode must contain entry 5
        ep1 = episodes[0]
        self.assertEqual(ep1["start_entry"], 5)
        self.assertGreater(ep1["peak_vibration_g"], 0.5)
        self.assertTrue(ep1["esp32_alerted"])

    def test_trend_analysis_and_significance(self):
        """Verifies linear and EWMA trend slope calculation and p-values."""
        trends = analyze_trends(self.samples)
        self.assertIn("vibration_g", trends)
        vib_trend = trends["vibration_g"]
        self.assertIn("slope_per_hour", vib_trend)
        self.assertIn("p_value", vib_trend)
        self.assertIn("is_significant", vib_trend)
        self.assertIn("direction", vib_trend)

    def test_bivariate_correlations(self):
        """Verifies Pearson and Spearman rank correlation computations."""
        corrs = compute_correlations(self.samples)
        self.assertIn("temperature_c__vs__vibration_g", corrs)
        temp_vib = corrs["temperature_c__vs__vibration_g"]
        self.assertEqual(temp_vib["status"], "computed")
        self.assertIn("pearson_r", temp_vib)
        self.assertIn("spearman_rho", temp_vib)
        self.assertGreaterEqual(temp_vib["sample_size"], 50)

    def test_data_quality_report(self):
        """Verifies data quality report including gaps, flatlines, and DHT11 quantisation note."""
        dq = generate_data_quality_report(self.samples, self.health["gaps"])
        self.assertIn("missing_rates_pct", dq)
        self.assertIn("flatline_detected", dq)
        self.assertIn("dht11_quantisation_note", dq)
        self.assertEqual(dq["acs712_status"], "DISCONNECTED_RESERVED")

    def test_offline_detection(self):
        """Verifies that connection health accurately flags offline state when age > threshold."""
        adapter = ThingSpeakAdapter()
        # With current reference time, recorded 2026 data is offline
        norm_now = adapter.normalize(adapter.fetch_raw_feeds()[0])
        self.assertFalse(norm_now["health"]["online"])

        # With reference time set to 10 seconds after latest sample, device is online
        latest_dt = datetime.fromisoformat(self.samples[-1]["timestamp"].replace("Z", "+00:00"))
        norm_fresh = adapter.normalize(adapter.fetch_raw_feeds()[0], reference_time=latest_dt + timedelta(seconds=10))
        self.assertTrue(norm_fresh["health"]["online"])
        self.assertLess(norm_fresh["health"]["age_seconds"], 20.0)


if __name__ == "__main__":
    unittest.main()

