import unittest

from app import tournament


class AutoMLTests(unittest.TestCase):
    def test_tournament_is_ranked(self):
        board = tournament()["leaderboard"]
        self.assertEqual(board, sorted(board, key=lambda row: row["roc_auc"], reverse=True))

    def test_every_model_uses_same_metric(self):
        result = tournament()
        self.assertEqual(result["folds"], 5)
        self.assertTrue(all(.5 <= row["roc_auc"] <= 1 for row in result["leaderboard"]))


if __name__ == "__main__":
    unittest.main()
