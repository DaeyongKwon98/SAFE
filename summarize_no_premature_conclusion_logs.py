#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CORRECT_ERROR_TYPE = "Correct (No Error)"
CORRECT_LABEL = "correct"
MISSING_ERROR_TYPE = "<MISSING_ERROR_TYPE>"
FOLDER_PATTERN_PREFIX = "dev_kg_correct_1ksample_no_premature_conclusion_"
FOLDER_PATTERN_SUFFIX = "_qwen3_8b_no_premature_conclusion"
KNOWN_DATASETS = ("2wiki", "hotpotqa", "musique")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate no_premature_conclusion *_logs.json files by folder and export "
            "summary/error_type count CSV files."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/workspace/daeyong/inference_results"),
        help="Base directory containing no_premature_conclusion run folders.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="10",
        help="Comma-separated max_steps filter (default: 7,10).",
    )
    parser.add_argument(
        "--retries",
        type=str,
        default="3",
        help="Comma-separated max_retries filter (default: 1,2,3).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/workspace/daeyong/inference_results/no_premature_conclusion_summary.csv"),
        help="Output CSV path for per-folder summary.",
    )
    parser.add_argument(
        "--error-csv",
        type=Path,
        default=Path("/workspace/daeyong/inference_results/no_premature_conclusion_error_type_counts.csv"),
        help="Output CSV path for per-folder error_type counts.",
    )
    parser.add_argument(
        "--accuracy-csv",
        type=Path,
        default=Path("/workspace/daeyong/inference_results/no_premature_conclusion_accuracy_by_model_dataset.csv"),
        help="Output CSV path for per-folder model/dataset accuracy rows.",
    )
    parser.add_argument(
        "--error-step-csv",
        type=Path,
        default=Path("/workspace/daeyong/inference_results/no_premature_conclusion_error_type_step_distribution.csv"),
        help="Output CSV path for per-folder error_type-by-step distribution rows.",
    )
    parser.add_argument(
        "--include-correct-error-type",
        action="store_true",
        help="Include 'Correct (No Error)' in error_type count output.",
    )
    return parser.parse_args()


def parse_int_tokens(value: str) -> List[int]:
    out: List[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer token: {token}") from exc
    if not out:
        raise ValueError(f"No valid integers parsed from: {value}")
    return sorted(set(out))


def parse_folder_name(folder_name: str) -> Tuple[int, int]:
    if not (folder_name.startswith(FOLDER_PATTERN_PREFIX) and folder_name.endswith(FOLDER_PATTERN_SUFFIX)):
        raise ValueError(f"Folder name does not match expected pattern: {folder_name}")

    middle = folder_name[len(FOLDER_PATTERN_PREFIX) : -len(FOLDER_PATTERN_SUFFIX)]
    tokens = middle.split("_")
    if len(tokens) != 2:
        raise ValueError(f"Unable to parse max_steps/max_retries from: {folder_name}")
    try:
        max_steps = int(tokens[0])
        max_retries = int(tokens[1])
    except ValueError as exc:
        raise ValueError(f"Invalid numeric values in folder name: {folder_name}") from exc
    return max_steps, max_retries


def discover_target_dirs(base_dir: Path, steps: Sequence[int], retries: Sequence[int]) -> List[Tuple[Path, int, int]]:
    targets: List[Tuple[Path, int, int]] = []
    step_set = set(steps)
    retry_set = set(retries)

    for path in sorted(base_dir.glob(f"{FOLDER_PATTERN_PREFIX}*{FOLDER_PATTERN_SUFFIX}")):
        if not path.is_dir():
            continue
        try:
            max_steps, max_retries = parse_folder_name(path.name)
        except ValueError:
            continue
        if max_steps in step_set and max_retries in retry_set:
            targets.append((path, max_steps, max_retries))

    return sorted(targets, key=lambda x: (x[1], x[2], x[0].name))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_csv_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_csv_value(row.get(k)) for k in fieldnames})


def parse_model_dataset_from_judge_path(judge_path: Path) -> Tuple[str, str]:
    stem = judge_path.name[: -len("_llm_judge.json")]
    for dataset in KNOWN_DATASETS:
        suffix = f"_{dataset}"
        if stem.endswith(suffix):
            model = stem[: -len(suffix)]
            if model:
                return model, dataset
    if "_" not in stem:
        return stem, "<UNKNOWN_DATASET>"
    model, dataset = stem.rsplit("_", 1)
    return model, dataset


def parse_is_correct(value) -> bool:
    return str(value).strip().lower() == CORRECT_LABEL


def safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def aggregate_folder(
    folder_path: Path,
    include_correct_error_type: bool,
) -> Tuple[Dict, Counter, Counter, int]:
    log_paths = sorted(folder_path.glob("*_logs.json"))

    total_questions = 0
    total_steps = 0
    total_retries = 0
    skipped_non_list_steps_history = 0
    error_counter: Counter = Counter()
    error_step_counter: Counter = Counter()

    for log_path in log_paths:
        logs = load_json(log_path)
        if not isinstance(logs, list):
            continue

        for record in logs:
            if not isinstance(record, dict):
                continue
            steps_history = record.get("steps_history")
            if not isinstance(steps_history, list):
                skipped_non_list_steps_history += 1
                continue

            total_questions += 1
            total_steps += len(steps_history)

            for step_idx, step in enumerate(steps_history, start=1):
                if not isinstance(step, dict):
                    continue
                attempts = step.get("attempts", [])
                if not isinstance(attempts, list):
                    continue
                step_num = safe_int(step.get("step_num"))
                if step_num is None:
                    step_num = step_idx

                total_retries += max(0, len(attempts) - 1)

                for attempt in attempts:
                    if not isinstance(attempt, dict):
                        continue
                    evaluation = attempt.get("evaluation")
                    if isinstance(evaluation, dict):
                        error_type = str(evaluation.get("error_type", "")).strip()
                    else:
                        error_type = ""
                    if not error_type:
                        error_type = MISSING_ERROR_TYPE
                    if (not include_correct_error_type) and error_type == CORRECT_ERROR_TYPE:
                        continue
                    error_counter[error_type] += 1
                    error_step_counter[(error_type, step_num)] += 1

    avg_steps_per_question = (total_steps / total_questions) if total_questions > 0 else 0.0
    avg_retries_per_question = (total_retries / total_questions) if total_questions > 0 else 0.0

    summary = {
        "folder_name": folder_path.name,
        "log_file_count": len(log_paths),
        "total_questions": total_questions,
        "avg_steps_per_question": avg_steps_per_question,
        "avg_retries_per_question": avg_retries_per_question,
    }
    return summary, error_counter, error_step_counter, skipped_non_list_steps_history


def aggregate_accuracy_folder(folder_path: Path) -> Tuple[float, List[Dict], int]:
    judge_paths = sorted(folder_path.glob("*_llm_judge.json"))

    folder_total_rows = 0
    folder_correct_rows = 0
    skipped_non_list_judges = 0
    detail_rows: List[Dict] = []

    for judge_path in judge_paths:
        model, dataset = parse_model_dataset_from_judge_path(judge_path)
        judges = load_json(judge_path)
        if not isinstance(judges, list):
            skipped_non_list_judges += 1
            continue

        total_rows = 0
        correct_rows = 0
        for row in judges:
            if not isinstance(row, dict):
                continue
            total_rows += 1
            if parse_is_correct(row.get("is_correct", "")):
                correct_rows += 1

        accuracy = (correct_rows / total_rows) if total_rows > 0 else 0.0
        folder_total_rows += total_rows
        folder_correct_rows += correct_rows
        detail_rows.append(
            {
                "folder_name": folder_path.name,
                "model": model,
                "dataset": dataset,
                "judge_file": judge_path.name,
                "total_samples": total_rows,
                "correct_samples": correct_rows,
                "accuracy": accuracy,
            }
        )

    total_accuracy = (folder_correct_rows / folder_total_rows) if folder_total_rows > 0 else 0.0
    return total_accuracy, detail_rows, skipped_non_list_judges


def main() -> int:
    args = parse_args()

    try:
        steps = parse_int_tokens(args.steps)
        retries = parse_int_tokens(args.retries)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not args.base_dir.exists():
        print(f"[ERROR] base-dir does not exist: {args.base_dir}", file=sys.stderr)
        return 2

    targets = discover_target_dirs(args.base_dir, steps, retries)
    if not targets:
        print("[ERROR] No target folders found for the given filters.", file=sys.stderr)
        return 1

    summary_rows: List[Dict] = []
    error_rows: List[Dict] = []
    accuracy_rows: List[Dict] = []
    error_step_rows: List[Dict] = []
    total_skipped = 0
    total_skipped_judge_rows = 0

    for folder_path, max_steps, max_retries in targets:
        summary, error_counter, error_step_counter, skipped = aggregate_folder(
            folder_path=folder_path,
            include_correct_error_type=args.include_correct_error_type,
        )
        total_accuracy, folder_accuracy_rows, skipped_judge_rows = aggregate_accuracy_folder(folder_path)

        total_skipped += skipped
        total_skipped_judge_rows += skipped_judge_rows

        summary_row = {
            "folder_name": summary["folder_name"],
            "max_steps": max_steps,
            "max_retries": max_retries,
            "log_file_count": summary["log_file_count"],
            "total_questions": summary["total_questions"],
            "avg_steps_per_question": summary["avg_steps_per_question"],
            "avg_retries_per_question": summary["avg_retries_per_question"],
            "total_accuracy": total_accuracy,
        }
        summary_rows.append(summary_row)

        for error_type, count in sorted(error_counter.items(), key=lambda t: (-t[1], t[0])):
            error_rows.append(
                {
                    "folder_name": summary["folder_name"],
                    "max_steps": max_steps,
                    "max_retries": max_retries,
                    "error_type": error_type,
                    "count": count,
                }
            )

        for (error_type, step_num), count in sorted(
            error_step_counter.items(),
            key=lambda t: (t[0][0], t[0][1]),
        ):
            total_for_error_type = error_counter.get(error_type, 0)
            ratio = (count / total_for_error_type) if total_for_error_type > 0 else 0.0
            error_step_rows.append(
                {
                    "folder_name": summary["folder_name"],
                    "max_steps": max_steps,
                    "max_retries": max_retries,
                    "error_type": error_type,
                    "step_num": step_num,
                    "count": count,
                    "ratio_within_error_type": ratio,
                    "pct_within_error_type": ratio * 100.0,
                }
            )

        for row in folder_accuracy_rows:
            accuracy_rows.append(
                {
                    "folder_name": summary["folder_name"],
                    "max_steps": max_steps,
                    "max_retries": max_retries,
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "judge_file": row["judge_file"],
                    "total_samples": row["total_samples"],
                    "correct_samples": row["correct_samples"],
                    "accuracy": row["accuracy"],
                }
            )

    summary_rows.sort(key=lambda r: (r["max_steps"], r["max_retries"], r["folder_name"]))
    error_rows.sort(
        key=lambda r: (
            r["max_steps"],
            r["max_retries"],
            r["folder_name"],
            -int(r["count"]),
            r["error_type"],
        )
    )
    accuracy_rows.sort(
        key=lambda r: (
            r["max_steps"],
            r["max_retries"],
            r["folder_name"],
            r["model"],
            r["dataset"],
        )
    )
    error_step_rows.sort(
        key=lambda r: (
            r["max_steps"],
            r["max_retries"],
            r["folder_name"],
            r["error_type"],
            int(r["step_num"]),
        )
    )

    write_csv(
        args.summary_csv,
        fieldnames=[
            "folder_name",
            "max_steps",
            "max_retries",
            "log_file_count",
            "total_questions",
            "avg_steps_per_question",
            "avg_retries_per_question",
            "total_accuracy",
        ],
        rows=summary_rows,
    )
    write_csv(
        args.error_csv,
        fieldnames=[
            "folder_name",
            "max_steps",
            "max_retries",
            "error_type",
            "count",
        ],
        rows=error_rows,
    )
    write_csv(
        args.accuracy_csv,
        fieldnames=[
            "folder_name",
            "max_steps",
            "max_retries",
            "model",
            "dataset",
            "judge_file",
            "total_samples",
            "correct_samples",
            "accuracy",
        ],
        rows=accuracy_rows,
    )
    write_csv(
        args.error_step_csv,
        fieldnames=[
            "folder_name",
            "max_steps",
            "max_retries",
            "error_type",
            "step_num",
            "count",
            "ratio_within_error_type",
            "pct_within_error_type",
        ],
        rows=error_step_rows,
    )

    print(f"[INFO] Target folders: {len(summary_rows)}")
    print(f"[INFO] Wrote summary CSV: {args.summary_csv}")
    print(f"[INFO] Wrote error CSV:   {args.error_csv}")
    print(f"[INFO] Wrote accuracy CSV: {args.accuracy_csv}")
    print(f"[INFO] Wrote error-step CSV: {args.error_step_csv}")
    print(f"[INFO] Skipped non-list steps_history records: {total_skipped}")
    print(f"[INFO] Skipped non-list llm_judge files: {total_skipped_judge_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
