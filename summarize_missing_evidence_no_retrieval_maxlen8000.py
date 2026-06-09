import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Callable


RESULT_DIR = Path(
    "/workspace/daeyong/inference_results/"
    "dev_kg_correct_missing_evidence_100sample_no_retrieval_maxlen8000_10_3_"
    "qwen3_8b_missing_evidence_training_data"
)
SUMMARY_PATH = RESULT_DIR / "missing_evidence_no_retrieval_maxlen8000_metrics_summary.csv"
MODELS = ("gemma12b", "qwen8b", "llama8b")
DATASETS = ("2wiki", "hotpotqa", "musique")


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    value = value.lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split()).strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction == ground_truth else 0.0


def max_over_ground_truths(metric_fn: Callable[[str, str], float], prediction: str, ground_truths: list[str]) -> float:
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, ground_truth) for ground_truth in ground_truths)


def parse_ground_truths(row: dict[str, Any], dataset: str) -> list[str]:
    ground_truth = normalize_text(row.get("ground_truth"))
    if dataset != "musique":
        return [ground_truth]

    if ground_truth == normalize_text("Cannot Answer"):
        return [ground_truth]

    aliases = row.get("answer_list_norm")
    if not isinstance(aliases, list):
        aliases = []
    normalized_aliases = [normalize_text(alias) for alias in aliases]
    normalized_aliases = [alias for alias in normalized_aliases if alias]
    return normalized_aliases or [ground_truth]


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON: {path}")
    return rows


def summarize_one(model: str, dataset: str) -> dict[str, Any]:
    final_rows = load_rows(RESULT_DIR / f"{model}_{dataset}_final_answer.json")
    judge_rows = load_rows(RESULT_DIR / f"{model}_{dataset}_llm_judge.json")
    if len(final_rows) != len(judge_rows):
        raise ValueError(f"Row count mismatch for {model}/{dataset}: final={len(final_rows)} judge={len(judge_rows)}")

    em_scores = []
    f1_scores = []
    for row in final_rows:
        prediction = normalize_text(row.get("final_answer"))
        ground_truths = parse_ground_truths(row, dataset)
        em_scores.append(max_over_ground_truths(exact_match, prediction, ground_truths))
        f1_scores.append(max_over_ground_truths(f1_score, prediction, ground_truths))

    judge_labels = [normalize_text(row.get("is_correct")) for row in judge_rows]
    correct = sum(label == "correct" for label in judge_labels)
    wrong = sum(label == "wrong" for label in judge_labels)
    error = len(judge_labels) - correct - wrong
    n = len(final_rows)

    return {
        "model": model,
        "dataset": dataset,
        "n": n,
        "exact_match": round(100 * sum(em_scores) / n, 2) if n else 0.0,
        "f1": round(100 * sum(f1_scores) / n, 2) if n else 0.0,
        "llm_judge": round(100 * correct / n, 2) if n else 0.0,
        "llm_judge_correct": correct,
        "llm_judge_wrong": wrong,
        "llm_judge_error": error,
    }


def main() -> None:
    rows = [summarize_one(model, dataset) for model in MODELS for dataset in DATASETS]
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {SUMMARY_PATH}")
    print("| Model | Dataset | N | EM | F1 | LLM judge | Judge correct/wrong/error |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['dataset']} | {row['n']} | "
            f"{row['exact_match']:.2f} | {row['f1']:.2f} | {row['llm_judge']:.2f} | "
            f"{row['llm_judge_correct']}/{row['llm_judge_wrong']}/{row['llm_judge_error']} |"
        )


if __name__ == "__main__":
    main()
