import unittest

from app import lifecycle_report


class CrispDmTests(unittest.TestCase):
    def test_required_techniques_are_present(self):
        report = lifecycle_report()
        for phase in ("clustering", "anomaly_detection", "supervised_learning", "association_rule", "lsh"):
            self.assertIn(phase, report)

    def test_supervised_result_is_holdout_based(self):
        result = lifecycle_report()["supervised_learning"]
        self.assertGreater(result["test_rows"], 0)
        self.assertGreater(result["value"], .7)


if __name__ == "__main__":
    unittest.main()
