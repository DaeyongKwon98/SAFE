import tempfile
import unittest
from pathlib import Path

from safe.config import DEFAULT_CONFIG
from safe.prepare import correct_records, prepare_data


class PrepareTests(unittest.TestCase):
    def test_correct_records_follow_ideal_steps(self):
        records = correct_records(
            [
                {
                    "id": "1",
                    "dataset": "test",
                    "question": "Q?",
                    "retrieved_passages": ["A"],
                    "answers": ["A"],
                    "ideal_steps": ["Step 1: A", "Step 2: ####ANSWER: A"],
                }
            ]
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["previous_steps"], ["Step 1: A"])

    def test_prepare_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared.jsonl"
            prepared = prepare_data(
                "examples/benchmark.jsonl",
                str(output),
                DEFAULT_CONFIG,
                limit=1,
            )
            self.assertTrue(output.exists())
            self.assertEqual(len(prepared), 2)


if __name__ == "__main__":
    unittest.main()

