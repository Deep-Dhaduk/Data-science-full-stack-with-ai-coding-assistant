import unittest

from app import TripRequest, estimate, haversine_km, make_demo_data, trained_models


class TaxiModelTests(unittest.TestCase):
    def test_haversine_is_symmetric(self):
        a = haversine_km(40.758, -73.9855, 40.6413, -73.7781)
        b = haversine_km(40.6413, -73.7781, 40.758, -73.9855)
        self.assertAlmostEqual(a, b, places=7)
        self.assertGreater(a, 10)

    def test_demo_data_is_deterministic(self):
        x1, y1 = make_demo_data(20, seed=7)
        x2, y2 = make_demo_data(20, seed=7)
        self.assertTrue((x1 == x2).all() and (y1 == y2).all())

    def test_prediction_and_holdout_metrics(self):
        result = estimate(TripRequest())
        metrics = trained_models()[2]
        self.assertGreater(result["duration_minutes"], 0)
        self.assertGreater(result["fare_usd"], 0)
        self.assertGreater(metrics["duration_r2"], 0.7)


if __name__ == "__main__":
    unittest.main()
