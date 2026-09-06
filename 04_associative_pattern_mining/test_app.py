import unittest

from app import mine_rules, support


class BasketMiningTests(unittest.TestCase):
    def test_support_bounds(self):
        value = support(frozenset({"bread"}))
        self.assertGreater(value, 0)
        self.assertLessEqual(value, 1)

    def test_rules_obey_thresholds(self):
        rules = mine_rules(0.2, 0.6)
        self.assertTrue(rules)
        self.assertTrue(all(r["support"] >= 0.2 and r["confidence"] >= 0.6 for r in rules))

    def test_higher_support_reduces_rules(self):
        self.assertLessEqual(len(mine_rules(0.4, 0.5)), len(mine_rules(0.2, 0.5)))


if __name__ == "__main__":
    unittest.main()
