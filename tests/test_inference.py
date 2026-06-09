import unittest

from safe.backend import GenerationResult
from safe.config import DEFAULT_CONFIG
from safe.inference import run_states


RECORD = {
    "id": "1",
    "dataset": "test",
    "question": "What is the capital of France?",
    "retrieved_passages": ["Paris is the capital of France."],
    "answers": ["Paris"],
}


class QueueBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def generate(self, messages, generation):
        values = self.outputs[: len(messages)]
        del self.outputs[: len(messages)]
        return [
            GenerationResult(text=value, prompt_tokens=10, generated_tokens=4)
            for value in values
        ]


class InferenceTests(unittest.TestCase):
    def config(self):
        config = {
            **DEFAULT_CONFIG,
            "inference": {"max_steps": 3, "max_retries": 2},
        }
        return config

    def test_baseline_finishes_on_answer(self):
        backend = QueueBackend(["Step 1: ####ANSWER: Paris (Final Answer)"])
        result = run_states([RECORD], "baseline", backend, None, self.config())[0]
        self.assertEqual(result["final_answer"], "Paris")
        self.assertEqual(result["status"], "completed")

    def test_safe_retries_then_accepts(self):
        generator = QueueBackend(
            [
                "Step 1: According to Passage 1, Lyon is the capital. (Attribution)",
                "Step 1: According to Passage 1, Paris is the capital. (Attribution)",
                "Step 2: ####ANSWER: Paris (Final Answer)",
            ]
        )
        evaluator = QueueBackend(
            [
                '{"error_type":"Contradictory","diagnosis":"wrong city","guidance":"use Paris"}',
                '{"error_type":"Correct (No Error)","diagnosis":"grounded","guidance":"answer"}',
                '{"error_type":"Correct (No Error)","diagnosis":"correct","guidance":"stop"}',
            ]
        )
        result = run_states(
            [RECORD], "safe", generator, evaluator, self.config()
        )[0]
        self.assertEqual(result["final_answer"], "Paris")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["response"]), 2)
        self.assertEqual(result["attempts"][0]["result"], "rejected")

    def test_self_feedback_uses_shared_backend(self):
        backend = QueueBackend(
            [
                "Step 1: ####ANSWER: Paris (Final Answer)",
                '{"error_type":"Correct (No Error)","diagnosis":"correct","guidance":"stop"}',
            ]
        )
        result = run_states(
            [RECORD], "self-feedback", backend, backend, self.config()
        )[0]
        self.assertEqual(result["final_answer"], "Paris")


if __name__ == "__main__":
    unittest.main()

