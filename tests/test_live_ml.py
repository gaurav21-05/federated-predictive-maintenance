"""
Unit tests for Pure-NumPy Live ML Inference Engine & Honest RUL Estimation.
Verifies that inference runs without PyTorch imports.
"""

import sys
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.live_ml import LiveMLEngine


class TestLiveMLEngine(unittest.TestCase):
    def setUp(self):
        self.engine = LiveMLEngine()

    def test_feature_extraction(self):
        sample_batch = [
            {
                "temperature_c": 24.5,
                "humidity_pct": 68.0,
                "temp_secondary_c": 25.1,
                "pressure_hpa": 993.8,
                "vibration_g": 0.022,
                "esp32_anomaly": 0,
            },
            {
                "temperature_c": 24.8,
                "humidity_pct": 67.5,
                "temp_secondary_c": 25.3,
                "pressure_hpa": 994.0,
                "vibration_g": 0.024,
                "esp32_anomaly": 0,
            },
        ]
        X, valid = self.engine.extract_features(sample_batch)
        self.assertEqual(len(X), 2)
        self.assertEqual(X.shape[1], 5)
        self.assertEqual(len(valid), 2)

    def test_autoencoder_forward_pure_numpy(self):
        # 5 input features
        X_dummy = np.random.randn(4, 5).astype(np.float32)
        recon = self.engine.autoencoder_forward(X_dummy)
        self.assertEqual(recon.shape, (4, 5))
        # Ensure finite numeric values
        self.assertTrue(np.all(np.isfinite(recon)))

    def test_compute_anomaly_scores(self):
        sample_batch = [
            {
                "entry_id": 1,
                "timestamp": "2026-11-25T10:00:00Z",
                "temperature_c": 24.0,
                "humidity_pct": 65.0,
                "temp_secondary_c": 24.5,
                "pressure_hpa": 994.0,
                "vibration_g": 0.020,
                "esp32_anomaly": 0,
                "esp32_alert": 0,
            },
            {
                "entry_id": 2,
                "timestamp": "2026-11-25T10:00:20Z",
                "temperature_c": 24.2,
                "humidity_pct": 65.2,
                "temp_secondary_c": 24.6,
                "pressure_hpa": 994.1,
                "vibration_g": 0.350,  # Abnormal vibration spike
                "esp32_anomaly": 1,
                "esp32_alert": 1,
            },
        ]
        res = self.engine.compute_anomaly_scores(sample_batch)
        self.assertEqual(res["sample_count"], 2)
        scores = res["scores"]
        self.assertEqual(len(scores), 2)

        normal_sample = scores[0]
        spike_sample = scores[1]

        # Normal sample should have higher health index
        self.assertGreater(normal_sample["health_index"], spike_sample["health_index"])
        # Spike sample should have higher combined anomaly score
        self.assertGreater(spike_sample["combined_anomaly_score"], normal_sample["combined_anomaly_score"])

    def test_honest_rul_stable(self):
        # Machine with normal steady vibration
        samples = [
            {
                "vibration_g": 0.022,
                "temperature_c": 24.0,
                "humidity_pct": 65.0,
                "temp_secondary_c": 24.5,
                "pressure_hpa": 994.0,
            }
            for i in range(25)
        ]
        rul = self.engine.estimate_honest_rul(samples)
        self.assertEqual(rul["rul_state"], "STABLE")
        self.assertEqual(rul["estimated_rul_hours"], "> 1000 h")
        self.assertIsNotNone(rul["confidence_interval_hours"])

    def test_honest_rul_insufficient_data(self):
        samples = [{"vibration_g": 0.02}] * 5
        rul = self.engine.estimate_honest_rul(samples)
        self.assertEqual(rul["rul_state"], "INSUFFICIENT_DATA")
        self.assertIsNone(rul["estimated_rul_hours"])

    def test_retrain_federated_round(self):
        samples = [
            {
                "temperature_c": 24.0 + 0.1 * i,
                "humidity_pct": 65.0 - 0.1 * i,
                "temp_secondary_c": 24.5 + 0.1 * i,
                "pressure_hpa": 994.0,
                "vibration_g": 0.022 + 0.001 * (i % 2),
            }
            for i in range(30)
        ]
        res = self.engine.retrain_federated_round(samples)
        self.assertEqual(res["status"], "success")
        self.assertIn("mse_before", res)
        self.assertIn("mse_after", res)
        self.assertEqual(res["virtual_nodes"], 3)


if __name__ == "__main__":
    unittest.main()
