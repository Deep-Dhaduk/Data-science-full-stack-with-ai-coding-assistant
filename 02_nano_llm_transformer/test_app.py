import unittest

import torch

from app import CHARS, TinyCausalTransformer, decode, encode, new_state, train_steps


class NanoTransformerTests(unittest.TestCase):
    def test_tokenizer_round_trip(self):
        text = "data science"
        self.assertEqual(decode(encode(text)), text)

    def test_forward_shape(self):
        model = TinyCausalTransformer(len(CHARS), width=24, heads=4, layers=1, context=16)
        output = model(torch.tensor([[1, 2, 3, 4]]))
        self.assertEqual(tuple(output.shape), (1, 4, len(CHARS)))

    def test_training_updates_diagnostics(self):
        state = new_state()
        losses = train_steps(state, steps=3, block=24)
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(loss > 0 for loss in losses))


if __name__ == "__main__":
    unittest.main()
