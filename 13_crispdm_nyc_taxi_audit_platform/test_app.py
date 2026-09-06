import unittest

import numpy as np

from app import InferenceRequest, experiment, inference, make_data, population_stability_index


class TaxiAuditTests(unittest.TestCase):
    def test_psi_identity_is_zero(self):
        values = np.arange(100, dtype=float)
        self.assertAlmostEqual(population_stability_index(values, values), 0)

    def test_model_tournament_and_controls(self):
        report = experiment()[1]
        self.assertEqual(len(report["leaderboard"]), 3)
        self.assertGreaterEqual(len(report["audit_controls"]), 5)

    def test_inference(self):
        self.assertGreater(inference(InferenceRequest())["duration_minutes"], 0)


if __name__ == "__main__":
    unittest.main()
