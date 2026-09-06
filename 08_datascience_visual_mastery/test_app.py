import unittest

from app import GradientRequest, MatrixRequest, bayes_posterior, classification_metrics, descend


class VisualMasteryTests(unittest.TestCase):
    def test_metrics(self):
        result = classification_metrics(MatrixRequest(true_positive=8, false_positive=2, false_negative=2, true_negative=8))
        self.assertEqual(result["precision"], .8)
        self.assertEqual(result["recall"], .8)

    def test_descent_reduces_loss(self):
        path = descend(GradientRequest(start=7, learning_rate=.15, steps=8))
        self.assertLess(path[-1]["loss"], path[0]["loss"])

    def test_bayes_base_rate(self):
        self.assertLess(bayes_posterior(.01, .9, .05), .2)


if __name__ == "__main__":
    unittest.main()
