import tempfile
import unittest
from pathlib import Path

from app import audit_repository, inspect_python


class AuditTests(unittest.TestCase):
    def test_detects_metric_risk(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "model.py"
            source.write_text("from sklearn.metrics import accuracy_score\n", encoding="utf-8")
            findings = inspect_python(source)
            self.assertTrue(any(f["control"] == "metrics" for f in findings))

    def test_scans_numbered_projects(self):
        report = audit_repository()
        self.assertGreaterEqual(report["projects_scanned"], 12)
        self.assertIn("disclaimer", report)


if __name__ == "__main__":
    unittest.main()
