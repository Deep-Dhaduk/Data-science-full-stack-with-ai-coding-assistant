import unittest

from app import CustomerRequest, assign, choose_k, make_rfm_data, segmentation


class SegmentationTests(unittest.TestCase):
    def test_generation_is_seeded(self):
        self.assertTrue((make_rfm_data(10, 8) == make_rfm_data(10, 8)).all())

    def test_k_selection_finds_structure(self):
        best_k, scores = choose_k(make_rfm_data())
        self.assertIn(best_k, range(3, 6))
        self.assertEqual(len(scores), 5)

    def test_assignment_has_known_persona(self):
        result = assign(CustomerRequest())
        profiles = segmentation()[1]["profiles"]
        self.assertIn(result["persona"], {p["persona"] for p in profiles})


if __name__ == "__main__":
    unittest.main()
