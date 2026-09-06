import unittest

from app import EventRequest, detector, make_telemetry, score_event


class AnomalyTests(unittest.TestCase):
    def test_class_imbalance_is_explicit(self):
        _, y = make_telemetry()
        self.assertAlmostEqual(y.mean(), .05)

    def test_metrics_prioritize_rank_quality(self):
        report = detector()[2]
        self.assertGreater(report["average_precision"], .8)

    def test_extreme_event_is_flagged(self):
        result = score_event(EventRequest(requests_per_second=500, error_rate=.8, latency_ms=3000))
        self.assertTrue(result["flagged"])


if __name__ == "__main__":
    unittest.main()
