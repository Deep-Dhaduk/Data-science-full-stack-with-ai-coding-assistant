import unittest

import numpy as np

from app import LAGS, PURGE, ScenarioRequest, fitted, lagged_dataset, make_market, scenario


class MarketForecastTests(unittest.TestCase):
    def test_features_are_strictly_lagged(self):
        _, returns = make_market(40)
        x, y = lagged_dataset(returns)
        self.assertEqual(x[0, LAGS - 1], returns[LAGS - 1])
        self.assertEqual(y[0], returns[LAGS])

    def test_purge_and_disclaimer_are_reported(self):
        report = fitted()[1]
        self.assertEqual(report["purged_rows"], PURGE)
        self.assertIn("Not investment advice", report["disclaimer"])

    def test_costly_scenario_executes(self):
        result = scenario(ScenarioRequest(transaction_cost_bps=10))
        self.assertTrue(np.isfinite(result["annualized_sharpe"]))


if __name__ == "__main__":
    unittest.main()
