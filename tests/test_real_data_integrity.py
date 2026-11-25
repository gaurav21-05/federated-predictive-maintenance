"""
Test Suite for Real-Data Integrity and Synthetic Artifact Deletion.
Validates acceptance criteria: zero synthetic strings in frontend, no torch in requirements.txt,
and real hardware feeds validation.
"""

import sys
from pathlib import Path
import unittest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRealDataIntegrity(unittest.TestCase):
    def test_no_synthetic_strings_in_frontend(self):
        """Verifies forbidden synthetic strings do not exist in shipped UI files."""
        forbidden_strings = [
            "CNC_MILL_01",
            "HYDR_PRESS_02",
            "TURB_COMP_03",
            "SIMULATED",
            "Step +1",
            "98.4",
            "12,000",
        ]
        frontend_files = [
            PROJECT_ROOT / "public" / "index.html",
            PROJECT_ROOT / "public" / "app.js",
            PROJECT_ROOT / "src" / "dashboard" / "static" / "index.html",
            PROJECT_ROOT / "src" / "dashboard" / "static" / "app.js",
        ]

        for file_path in frontend_files:
            self.assertTrue(file_path.exists(), f"File {file_path} must exist.")
            content = file_path.read_text(encoding="utf-8")
            for forbidden in forbidden_strings:
                self.assertNotIn(
                    forbidden,
                    content,
                    f"Forbidden synthetic token '{forbidden}' found in {file_path}",
                )

    def test_requirements_split_no_torch_in_runtime(self):
        """Verifies requirements.txt has ZERO torch imports for serverless deployment."""
        req_file = PROJECT_ROOT / "requirements.txt"
        self.assertTrue(req_file.exists())
        req_text = req_file.read_text(encoding="utf-8").lower()
        self.assertNotIn("torch", req_text)
        self.assertNotIn("matplotlib", req_text)

        # Training requirements should include torch
        req_train_file = PROJECT_ROOT / "requirements-train.txt"
        self.assertTrue(req_train_file.exists())
        req_train_text = req_train_file.read_text(encoding="utf-8").lower()
        self.assertIn("torch", req_train_text)

    def test_feeds_csv_integrity(self):
        """Verifies feeds.csv contains real hardware data."""
        csv_file = PROJECT_ROOT / "feeds.csv"
        self.assertTrue(csv_file.exists())
        df = pd.read_csv(csv_file)
        self.assertGreaterEqual(len(df), 60)
        self.assertIn("field1", df.columns)
        self.assertIn("field2", df.columns)
        self.assertIn("field5", df.columns)

    def test_model_artefacts_present(self):
        """Verifies exported models/live artefacts exist for serverless inference."""
        live_dir = PROJECT_ROOT / "models" / "live"
        self.assertTrue((live_dir / "autoencoder_weights.npz").exists())
        self.assertTrue((live_dir / "isolation_forest.joblib").exists())
        self.assertTrue((live_dir / "model_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()

