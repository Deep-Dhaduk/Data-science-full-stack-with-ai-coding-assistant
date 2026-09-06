import unittest

from app import execute_skill


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


if __name__ == "__main__":
    unittest.main()
