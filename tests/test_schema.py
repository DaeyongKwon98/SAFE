import unittest

from safe.schema import normalize_benchmark, normalize_evaluator


class SchemaTests(unittest.TestCase):
    def test_normalizes_legacy_answer_and_passage_strings(self):
        record = normalize_benchmark(
            {
                "id": 1,
                "dataset": "test",
                "question": "Q?",
                "retrieved_passages": "['A', 'B']",
                "answer": "A",
            }
        )
        self.assertEqual(record["id"], "1")
        self.assertEqual(record["retrieved_passages"], ["A", "B"])
        self.assertEqual(record["answers"], ["A"])

    def test_rejects_unknown_error_type(self):
        with self.assertRaises(ValueError):
            normalize_evaluator(
                {
                    "id": "1",
                    "dataset": "test",
                    "question": "Q?",
                    "retrieved_passages": ["A"],
                    "answers": ["A"],
                    "current_step": "Step 1",
                    "error_type": "Other",
                    "diagnosis": "x",
                    "guidance": "y",
                }
            )


if __name__ == "__main__":
    unittest.main()

