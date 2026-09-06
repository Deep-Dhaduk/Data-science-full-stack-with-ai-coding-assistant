import unittest

from app import fitted, forecast, make_series, supervised


class ForecastTests(unittest.TestCase):
    def test_lags_use_only_past_values(self):
        series = make_series(30)
        x, y = supervised(series)
        self.assertEqual(x[0, -1], series[13])
        self.assertEqual(y[0], series[14])

    def test_temporal_report_and_forecast(self):
        report = fitted()[2]
        self.assertIn("no shuffle", report["split"])
        points = forecast(5)
        self.assertEqual(len(points), 5)
        self.assertTrue(all(p["lower_90"] < p["upper_90"] for p in points))


if __name__ == "__main__":
    unittest.main()
