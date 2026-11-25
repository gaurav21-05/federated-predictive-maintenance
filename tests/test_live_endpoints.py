"""
Unit tests for Real-Data API Endpoints in src/dashboard/app.py.
Verifies response schemas, status codes, and deterministic fallback behaviour.
"""

import sys
from pathlib import Path
import unittest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.app import app


class TestLiveEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["mode"], "REAL_DATA_ONLY")
        self.assertEqual(data["channel_id"], "3591448")

    def test_live_hardware_endpoint(self):
        resp = self.client.get("/api/live/hardware?results=25&window=all")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("series", data)
        self.assertIn("twin", data)
        self.assertIn("stats", data)
        self.assertIn("rul", data)
        self.assertIn("anomalies", data)
        self.assertEqual(data["channel_id"], "3591448")

    def test_live_hardware_history_endpoint(self):
        resp = self.client.get("/api/live/hardware/history?results=20")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("series", data)
        self.assertIn("total_samples", data)

    def test_analysis_stats_endpoint(self):
        resp = self.client.get("/api/analysis/stats?results=25")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("stats", data)
        self.assertIn("baseline", data)
        self.assertIn("correlations", data)
        self.assertIn("data_quality", data)

    def test_analysis_model_endpoint(self):
        resp = self.client.get("/api/analysis/model?results=25")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("ml_inference", data)
        self.assertIn("honest_rul", data)
        self.assertIn("model_card", data)
        self.assertIn("federated_benchmark", data)

    def test_analysis_retrain_endpoint(self):
        resp = self.client.post("/api/analysis/retrain", json={"results": 30})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)

    def test_analysis_insights_fallback_endpoint(self):
        # Even without ANTHROPIC_API_KEY, must never crash or return 500
        resp = self.client.post("/api/analysis/insights", json={"force_refresh": True})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("markdown", data)
        self.assertIn("source", data)

    def test_export_csv_endpoint(self):
        resp = self.client.get("/api/live/export.csv?results=15")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.headers["content-type"])
        self.assertIn("temperature_c", resp.text)
        self.assertIn("vibration_g", resp.text)

    def test_simulated_endpoints_removed(self):
        # Verifies old simulated endpoints are removed
        r1 = self.client.get("/api/telemetry/stream")
        self.assertEqual(r1.status_code, 404)
        r2 = self.client.post("/api/digital-twin/step", json={"steps": 1})
        self.assertEqual(r2.status_code, 404)
        r3 = self.client.post("/api/digital-twin/what-if", json={"client_id": "client_1"})
        self.assertEqual(r3.status_code, 404)


if __name__ == "__main__":
    unittest.main()

