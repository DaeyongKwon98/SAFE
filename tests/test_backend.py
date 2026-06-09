import unittest

import torch

from safe.backend import TransformersBackend


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        return messages[-1]["content"]

    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        width = max(len(text.split()) for text in texts)
        rows = []
        masks = []
        for text in texts:
            tokens = list(range(3, 3 + len(text.split())))
            padding_values = [0] * (width - len(tokens))
            rows.append(padding_values + tokens)
            masks.append([0] * len(padding_values) + [1] * len(tokens))
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor(masks),
        }

    def batch_decode(self, rows, skip_special_tokens):
        return ["decoded-" + str(index) for index, _ in enumerate(rows)]


class FakeEmbeddings:
    weight = torch.zeros(1)


class FakeConfig:
    def to_dict(self):
        return {}

    def get_text_config(self, decoder=False):
        return self


class FakeGenerationConfig:
    pass


class FakeModel:
    device = torch.device("cpu")
    config = FakeConfig()
    generation_config = FakeGenerationConfig()

    def get_input_embeddings(self):
        return FakeEmbeddings()

    def generate(self, input_ids, attention_mask, generation_config, **kwargs):
        suffix = torch.tensor([[9, 2] for _ in range(input_ids.shape[0])])
        return torch.cat([input_ids, suffix], dim=1)


class BackendTests(unittest.TestCase):
    def test_batched_decode_keeps_input_order(self):
        backend = TransformersBackend(
            FakeModel(), FakeTokenizer(), batch_size=2, max_input_tokens=20
        )
        results = backend.generate(
            [
                [{"role": "user", "content": "one two three"}],
                [{"role": "user", "content": "one"}],
            ],
            {"max_new_tokens": 2, "temperature": 0.0, "top_p": 1.0},
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].prompt_tokens, 3)
        self.assertEqual(results[1].prompt_tokens, 1)


if __name__ == "__main__":
    unittest.main()

