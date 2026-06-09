import unittest

from safe.evaluation import normalize_text, score_record, token_f1
from safe.parsing import clean_generation, extract_final_answer, parse_evaluation


class ParsingEvaluationTests(unittest.TestCase):
    def test_cleans_and_extracts_answer(self):
        step = clean_generation("####ANSWER: Paris", 2)
        self.assertEqual(step, "Step 2: ####ANSWER: Paris (Final Answer)")
        self.assertEqual(extract_final_answer([step]), "Paris")

    def test_parses_fenced_evaluator_json(self):
        parsed = parse_evaluation(
            '```json\n{"error_type":"Correct (No Error)",'
            '"diagnosis":"ok","guidance":"finish"}\n```'
        )
        self.assertEqual(parsed["guidance"], "finish")

    def test_metrics_support_aliases(self):
        scored = score_record(
            {
                "id": "1",
                "dataset": "test",
                "answers": ["The City of Paris", "Paris"],
                "final_answer": "Paris",
                "response": [],
                "attempts": [{"result": "rejected"}],
            }
        )
        self.assertEqual(scored["em"], 1.0)
        self.assertEqual(scored["retries"], 1)
        self.assertEqual(normalize_text("The Paris!"), "paris")
        self.assertEqual(token_f1("new york", "new york"), 1.0)


if __name__ == "__main__":
    unittest.main()

