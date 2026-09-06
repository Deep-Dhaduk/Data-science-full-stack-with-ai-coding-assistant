import unittest

from app import dashboard, execute_skill, run_skill


class SkillsLabTests(unittest.TestCase):
    def test_profile(self):
        report = execute_skill("profile")
        self.assertEqual(report["rows"], 178)
        self.assertEqual(report["missing"], 0)

    def test_pca_preview(self):
        report = execute_skill("pca")
        self.assertEqual(len(report["preview"][0]), 2)

    def test_classifier(self):
        report = execute_skill("classify")
        self.assertGreater(report["balanced_accuracy"], 0.8)

    def test_dashboard_script_and_api_contract(self):
        self.assertIn("String.fromCharCode(10)", dashboard())
        response = run_skill("profile")
        self.assertEqual(response["title"], "Data profiling")
        self.assertEqual(response["result"]["rows"], 178)


if __name__ == "__main__":
    unittest.main()
