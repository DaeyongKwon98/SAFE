#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


MODELS: Sequence[str] = ("qwen4b", "qwen8b", "qwen14b", "llama8b", "gemma12b")
DATASETS: Sequence[str] = ("2wiki", "hotpotqa", "musique")

NO_FEEDBACK_DIR_BY_MODEL: Dict[str, str] = {
    "qwen4b": "no_feedback_Qwen3-4B-Instruct-2507",
    "qwen8b": "no_feedback_Qwen3-8B",
    "qwen14b": "no_feedback_Qwen2.5-14B-Instruct",
    "llama8b": "no_feedback_Meta-Llama-3.1-8B-Instruct",
    "gemma12b": "no_feedback_gemma-3-12b-it",
}


def self_feedback_log_path(base_dir: Path, model: str, dataset: str) -> Path:
    if model in {"qwen4b", "qwen8b", "qwen14b"}:
        return base_dir / "self_feedback_kg_correct_1k_sample" / f"{model}_{dataset}_logs.json"
    if model == "llama8b":
        return base_dir / "self_feedback_Meta-Llama-3.1-8B-Instruct" / f"{dataset}_logs.json"
    if model == "gemma12b":
        return base_dir / "self_feedback_gemma-3-12b-it" / f"{dataset}_logs.json"
    raise ValueError(f"Unsupported model key: {model}")


def load_json_list(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"JSON is not list: {path}")
    return data


def calc_no_feedback_avg_steps(path: Path) -> Tuple[int, float]:
    records = load_json_list(path)
    if not records:
        return 0, 0.0

    total_steps = 0
    valid_count = 0
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            raise TypeError(f"Row is not dict at index={idx}: {path}")
        generated_steps = row.get("generated_steps")
        if not isinstance(generated_steps, list):
            raise TypeError(f"generated_steps is not list at index={idx}: {path}")
        total_steps += len(generated_steps)
        valid_count += 1

    avg_steps = (total_steps / valid_count) if valid_count > 0 else 0.0
    return valid_count, avg_steps


def calc_self_feedback_avg_step_retry(path: Path) -> Tuple[int, float, float]:
    records = load_json_list(path)
    if not records:
        return 0, 0.0, 0.0

    total_steps = 0
    total_retries = 0
    valid_count = 0

    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            raise TypeError(f"Row is not dict at index={idx}: {path}")

        steps_history = row.get("steps_history")
        if not isinstance(steps_history, list):
            raise TypeError(f"steps_history is not list at index={idx}: {path}")

        total_steps += len(steps_history)
        for step in steps_history:
            if not isinstance(step, dict):
                continue
            attempts = step.get("attempts", [])
            if not isinstance(attempts, list):
                continue
            total_retries += max(0, len(attempts) - 1)

        valid_count += 1

    avg_steps = (total_steps / valid_count) if valid_count > 0 else 0.0
    avg_retries = (total_retries / valid_count) if valid_count > 0 else 0.0
    return valid_count, avg_steps, avg_retries


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def fmt_line(values: Sequence[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(fmt_line(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_line(row))


def append_no_feedback_overall_row(rows: List[Dict]) -> List[Dict]:
    total_questions = sum(int(row["questions"]) for row in rows)
    weighted_step_sum = sum(
        int(row["questions"]) * float(row["avg_steps_per_question"]) for row in rows
    )
    overall_avg_steps = (weighted_step_sum / total_questions) if total_questions > 0 else 0.0
    overall_row = {
        "model": "OVERALL",
        "dataset": "ALL",
        "questions": total_questions,
        "avg_steps_per_question": f"{overall_avg_steps:.6f}",
    }
    return [*rows, overall_row]


def append_self_feedback_overall_row(rows: List[Dict]) -> List[Dict]:
    total_questions = sum(int(row["questions"]) for row in rows)
    weighted_step_sum = sum(
        int(row["questions"]) * float(row["avg_steps_per_question"]) for row in rows
    )
    weighted_retry_sum = sum(
        int(row["questions"]) * float(row["avg_retries_per_question"]) for row in rows
    )
    overall_avg_steps = (weighted_step_sum / total_questions) if total_questions > 0 else 0.0
    overall_avg_retries = (
        weighted_retry_sum / total_questions if total_questions > 0 else 0.0
    )
    overall_row = {
        "model": "OVERALL",
        "dataset": "ALL",
        "questions": total_questions,
        "avg_steps_per_question": f"{overall_avg_steps:.6f}",
        "avg_retries_per_question": f"{overall_avg_retries:.6f}",
    }
    return [*rows, overall_row]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute avg step/retry metrics for no-feedback and self-feedback runs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/workspace/daeyong/inference_results"),
        help="Base inference_results directory.",
    )
    parser.add_argument(
        "--no-feedback-csv",
        type=Path,
        default=Path("/workspace/daeyong/inference_results/no_feedback_avg_steps_5models_3datasets.csv"),
        help="Output CSV for no-feedback stats.",
    )
    parser.add_argument(
        "--self-feedback-csv",
        type=Path,
        default=Path("/workspace/daeyong/inference_results/self_feedback_avg_steps_retries_5models_3datasets.csv"),
        help="Output CSV for self-feedback stats.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    no_feedback_rows: List[Dict] = []
    self_feedback_rows: List[Dict] = []

    for model in MODELS:
        for dataset in DATASETS:
            no_feedback_path = (
                args.base_dir / NO_FEEDBACK_DIR_BY_MODEL[model] / f"{dataset}_results.json"
            )
            q_count, avg_steps_no = calc_no_feedback_avg_steps(no_feedback_path)
            no_feedback_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "questions": q_count,
                    "avg_steps_per_question": f"{avg_steps_no:.6f}",
                }
            )

            self_feedback_path = self_feedback_log_path(args.base_dir, model, dataset)
            q_count_self, avg_steps_self, avg_retries_self = calc_self_feedback_avg_step_retry(
                self_feedback_path
            )
            self_feedback_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "questions": q_count_self,
                    "avg_steps_per_question": f"{avg_steps_self:.6f}",
                    "avg_retries_per_question": f"{avg_retries_self:.6f}",
                }
            )

    no_feedback_rows.sort(key=lambda r: (r["model"], r["dataset"]))
    self_feedback_rows.sort(key=lambda r: (r["model"], r["dataset"]))
    no_feedback_rows = append_no_feedback_overall_row(no_feedback_rows)
    self_feedback_rows = append_self_feedback_overall_row(self_feedback_rows)

    write_csv(
        path=args.no_feedback_csv,
        fieldnames=["model", "dataset", "questions", "avg_steps_per_question"],
        rows=no_feedback_rows,
    )
    write_csv(
        path=args.self_feedback_csv,
        fieldnames=[
            "model",
            "dataset",
            "questions",
            "avg_steps_per_question",
            "avg_retries_per_question",
        ],
        rows=self_feedback_rows,
    )

    print("[NO FEEDBACK] avg step per question")
    print_table(
        headers=("model", "dataset", "questions", "avg_steps_per_question"),
        rows=[
            (
                row["model"],
                row["dataset"],
                str(row["questions"]),
                row["avg_steps_per_question"],
            )
            for row in no_feedback_rows
        ],
    )
    print()

    print("[SELF-FEEDBACK] avg step/retry per question")
    print_table(
        headers=(
            "model",
            "dataset",
            "questions",
            "avg_steps_per_question",
            "avg_retries_per_question",
        ),
        rows=[
            (
                row["model"],
                row["dataset"],
                str(row["questions"]),
                row["avg_steps_per_question"],
                row["avg_retries_per_question"],
            )
            for row in self_feedback_rows
        ],
    )
    print()
    print(f"Saved: {args.no_feedback_csv}")
    print(f"Saved: {args.self_feedback_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
