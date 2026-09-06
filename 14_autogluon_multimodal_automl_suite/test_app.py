import unittest

from app import ProductRequest, benchmark, product_row, text_features


class MultimodalTests(unittest.TestCase):
    def test_text_features(self):
        positive, negative, length = text_features("excellent but slow delivery")
        self.assertEqual((positive, negative, length), (1, 1, 4))

    def test_all_modalities_compared(self):
        names = {row["modality"] for row in benchmark()[1]["leaderboard"]}
        self.assertEqual(names, {"tabular", "text", "image_summary", "late_fusion"})

    def test_product_shape(self):
        self.assertEqual(product_row(ProductRequest()).shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
