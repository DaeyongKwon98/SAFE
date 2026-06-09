#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_TARGET_DIR = Path(
    "/workspace/daeyong/inference_results/self_feedback_kg_correct_1k_sample_7_1"
)
DEFAULT_DETAIL_CSV = Path(
    "/workspace/daeyong/inference_results/self_feedback_kg_correct_1k_sample_7_1_metrics_detail.csv"
)
DEFAULT_SUMMARY_CSV = Path(
    "/workspace/daeyong/inference_results/self_feedback_kg_correct_1k_sample_7_1_metrics_summary.csv"
)

JUDGE_SUFFIX = "_llm_judge.json"
LOG_SUFFIX = "_logs.json"

DEFAULT_MODELS: Sequence[str] = ("qwen4b", "qwen8b", "qwen14b", "llama8b", "gemma12b")
DEFAULT_DATASETS: Sequence[str] = ("2wiki", "hotpotqa", "musique")

CORRECT_TOKENS: Set[str] = {"correct", "true", "1", "yes"}
WRONG_TOKENS: Set[str] = {"wrong", "incorrect", "false", "0", "no"}


@dataclass
class PairMetrics:
    prefix: str
    model: str
    dataset: str
    judge_file: str
    log_file: str
    total_questions: int
    correct_questions: int
    unknown_label_count: int
    total_steps: int
    total_retries: int
    max_attempts: int
    max_retry_index: int
    steps_with_multiple_attempts: int

    @property
    def accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.correct_questions / self.total_questions

    @property
    def avg_steps_per_question(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.total_steps / self.total_questions

    @property
    def avg_retries_per_question(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.total_retries / self.total_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute overall accuracy/step/retry metrics from *_llm_judge.json and "
            "*_logs.json in self_feedback_kg_correct_1k_sample_7_1."
        )
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory containing paired *_llm_judge.json and *_logs.json files.",
    )
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=DEFAULT_DETAIL_CSV,
        help="Output CSV path for per model/dataset metrics.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Output CSV path for overall summary metrics.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated expected model prefixes.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated expected dataset suffixes.",
    )
    parser.add_argument(
        "--strict-grid",
        dest="strict_grid",
        action="store_true",
        default=True,
        help="Validate exact model x dataset grid and fail on mismatch (default: true).",
    )
    parser.add_argument(
        "--no-strict-grid",
        dest="strict_grid",
        action="store_false",
        help="Disable strict grid validation.",
    )
    return parser.parse_args()


def parse_tokens(value: str) -> Tuple[str, ...]:
    out = []
    for token in value.split(","):
        stripped = token.strip()
        if stripped:
            out.append(stripped)
    if not out:
        raise ValueError(f"No valid tokens parsed from: {value}")
    return tuple(dict.fromkeys(out))


def expected_prefixes(models: Sequence[str], datasets: Sequence[str]) -> Set[str]:
    return {f"{m}_{d}" for m in models for d in datasets}


def load_json_list(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"JSON root is not list: {path}")
    return data


def discover_by_suffix(target_dir: Path, suffix: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(target_dir.glob(f"*{suffix}")):
        if not path.is_file():
            continue
        prefix = path.name[: -len(suffix)]
        if prefix in out:
            raise ValueError(f"Duplicate prefix for suffix {suffix}: {prefix}")
        out[prefix] = path
    return out


def parse_model_dataset(prefix: str, datasets: Sequence[str]) -> Tuple[str, str]:
    ordered_datasets = sorted(datasets, key=len, reverse=True)
    for dataset in ordered_datasets:
        suffix = f"_{dataset}"
        if prefix.endswith(suffix):
            model = prefix[: -len(suffix)]
            if model:
                return model, dataset
    raise ValueError(
        f"Cannot parse model/dataset from prefix '{prefix}' with datasets={datasets}"
    )


def normalize_is_correct(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in CORRECT_TOKENS:
        return True
    if token in WRONG_TOKENS:
        return False
    return None


def calc_pair_metrics(prefix: str, judge_path: Path, log_path: Path, datasets: Sequence[str]) -> PairMetrics:
    model, dataset = parse_model_dataset(prefix, datasets)
    judges = load_json_list(judge_path)
    logs = load_json_list(log_path)

    if len(judges) != len(logs):
        raise ValueError(
            f"Length mismatch for {prefix}: judges={len(judges)} logs={len(logs)}"
        )

    correct_questions = 0
    unknown_label_count = 0

    for idx, row in enumerate(judges):
        if not isinstance(row, dict):
            raise TypeError(f"Judge row is not dict at index={idx}: {judge_path}")
        norm = normalize_is_correct(row.get("is_correct"))
        if norm is True:
            correct_questions += 1
        elif norm is None:
            unknown_label_count += 1

    total_steps = 0
    total_retries = 0
    max_attempts = 0
    max_retry_index = 0
    steps_with_multiple_attempts = 0

    for idx, row in enumerate(logs):
        if not isinstance(row, dict):
            raise TypeError(f"Log row is not dict at index={idx}: {log_path}")
        steps_history = row.get("steps_history")
        if not isinstance(steps_history, list):
            raise TypeError(f"steps_history is not list at index={idx}: {log_path}")

        total_steps += len(steps_history)
        for step in steps_history:
            if not isinstance(step, dict):
                continue
            attempts = step.get("attempts", [])
            if not isinstance(attempts, list):
                continue
            attempts_len = len(attempts)
            if attempts_len > max_attempts:
                max_attempts = attempts_len
            if attempts_len > 1:
                steps_with_multiple_attempts += 1
            total_retries += max(0, attempts_len - 1)

            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                retry_idx = attempt.get("retry_index")
                if isinstance(retry_idx, int) and retry_idx > max_retry_index:
                    max_retry_index = retry_idx

    return PairMetrics(
        prefix=prefix,
        model=model,
        dataset=dataset,
        judge_file=judge_path.name,
        log_file=log_path.name,
        total_questions=len(judges),
        correct_questions=correct_questions,
        unknown_label_count=unknown_label_count,
        total_steps=total_steps,
        total_retries=total_retries,
        max_attempts=max_attempts,
        max_retry_index=max_retry_index,
        steps_with_multiple_attempts=steps_with_multiple_attempts,
    )


def to_csv_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_csv_value(row.get(k)) for k in fieldnames})


def main() -> int:
    args = parse_args()

    try:
        models = parse_tokens(args.models)
        datasets = parse_tokens(args.datasets)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not args.target_dir.exists():
        print(f"[ERROR] target-dir does not exist: {args.target_dir}", file=sys.stderr)
        return 2

    judge_map = discover_by_suffix(args.target_dir, JUDGE_SUFFIX)
    log_map = discover_by_suffix(args.target_dir, LOG_SUFFIX)
    expected = expected_prefixes(models=models, datasets=datasets)

    judge_prefixes = set(judge_map.keys())
    log_prefixes = set(log_map.keys())
    paired_prefixes = judge_prefixes & log_prefixes

    missing_judge = sorted(expected - judge_prefixes)
    missing_log = sorted(expected - log_prefixes)
    missing_pair = sorted(expected - paired_prefixes)

    if args.strict_grid:
        if missing_judge:
            print(f"[ERROR] Missing llm_judge files for prefixes: {missing_judge}", file=sys.stderr)
            return 1
        if missing_log:
            print(f"[ERROR] Missing logs files for prefixes: {missing_log}", file=sys.stderr)
            return 1
        if missing_pair:
            print(f"[ERROR] Missing paired prefixes: {missing_pair}", file=sys.stderr)
            return 1
        if len(judge_map) != len(expected):
            print(
                f"[ERROR] Expected {len(expected)} llm_judge files but found {len(judge_map)}.",
                file=sys.stderr,
            )
            return 1
        if len(log_map) != len(expected):
            print(
                f"[ERROR] Expected {len(expected)} logs files but found {len(log_map)}.",
                file=sys.stderr,
            )
            return 1
        target_prefixes = sorted(expected)
    else:
        target_prefixes = sorted(paired_prefixes)

    if not target_prefixes:
        print("[ERROR] No paired prefixes found.", file=sys.stderr)
        return 1

    pair_metrics: List[PairMetrics] = []
    for prefix in target_prefixes:
        judge_path = judge_map.get(prefix)
        log_path = log_map.get(prefix)
        if judge_path is None or log_path is None:
            continue
        pair_metrics.append(calc_pair_metrics(prefix, judge_path, log_path, datasets=datasets))

    pair_metrics.sort(key=lambda m: (m.model, m.dataset))

    total_questions = sum(m.total_questions for m in pair_metrics)
    total_correct_questions = sum(m.correct_questions for m in pair_metrics)
    total_unknown_labels = sum(m.unknown_label_count for m in pair_metrics)
    total_steps = sum(m.total_steps for m in pair_metrics)
    total_retries = sum(m.total_retries for m in pair_metrics)

    overall_accuracy = (total_correct_questions / total_questions) if total_questions > 0 else 0.0
    overall_avg_steps = (total_steps / total_questions) if total_questions > 0 else 0.0
    overall_avg_retries = (total_retries / total_questions) if total_questions > 0 else 0.0

    macro_accuracy = (
        sum(m.accuracy for m in pair_metrics) / len(pair_metrics) if pair_metrics else 0.0
    )
    macro_avg_steps = (
        sum(m.avg_steps_per_question for m in pair_metrics) / len(pair_metrics)
        if pair_metrics
        else 0.0
    )
    macro_avg_retries = (
        sum(m.avg_retries_per_question for m in pair_metrics) / len(pair_metrics)
        if pair_metrics
        else 0.0
    )

    global_max_attempts = max((m.max_attempts for m in pair_metrics), default=0)
    global_max_retry_index = max((m.max_retry_index for m in pair_metrics), default=0)
    total_steps_with_multiple_attempts = sum(
        m.steps_with_multiple_attempts for m in pair_metrics
    )

    detail_rows: List[Dict] = []
    for m in pair_metrics:
        detail_rows.append(
            {
                "prefix": m.prefix,
                "model": m.model,
                "dataset": m.dataset,
                "judge_file": m.judge_file,
                "log_file": m.log_file,
                "total_questions": m.total_questions,
                "correct_questions": m.correct_questions,
                "unknown_label_count": m.unknown_label_count,
                "accuracy": m.accuracy,
                "avg_steps_per_question": m.avg_steps_per_question,
                "avg_retries_per_question": m.avg_retries_per_question,
                "total_steps": m.total_steps,
                "total_retries": m.total_retries,
                "max_attempts": m.max_attempts,
                "max_retry_index": m.max_retry_index,
                "steps_with_multiple_attempts": m.steps_with_multiple_attempts,
            }
        )

    summary_rows = [
        {
            "target_dir": str(args.target_dir),
            "judge_file_count": len(judge_map),
            "log_file_count": len(log_map),
            "matched_pair_count": len(pair_metrics),
            "expected_pair_count": len(expected),
            "total_questions": total_questions,
            "total_correct_questions": total_correct_questions,
            "total_unknown_label_count": total_unknown_labels,
            "overall_accuracy_micro": overall_accuracy,
            "overall_avg_steps_per_question": overall_avg_steps,
            "overall_avg_retries_per_question": overall_avg_retries,
            "macro_accuracy": macro_accuracy,
            "macro_avg_steps_per_question": macro_avg_steps,
            "macro_avg_retries_per_question": macro_avg_retries,
            "global_max_attempts": global_max_attempts,
            "global_max_retry_index": global_max_retry_index,
            "total_steps_with_multiple_attempts": total_steps_with_multiple_attempts,
        }
    ]

    write_csv(
        path=args.detail_csv,
        fieldnames=[
            "prefix",
            "model",
            "dataset",
            "judge_file",
            "log_file",
            "total_questions",
            "correct_questions",
            "unknown_label_count",
            "accuracy",
            "avg_steps_per_question",
            "avg_retries_per_question",
            "total_steps",
            "total_retries",
            "max_attempts",
            "max_retry_index",
            "steps_with_multiple_attempts",
        ],
        rows=detail_rows,
    )
    write_csv(
        path=args.summary_csv,
        fieldnames=[
            "target_dir",
            "judge_file_count",
            "log_file_count",
            "matched_pair_count",
            "expected_pair_count",
            "total_questions",
            "total_correct_questions",
            "total_unknown_label_count",
            "overall_accuracy_micro",
            "overall_avg_steps_per_question",
            "overall_avg_retries_per_question",
            "macro_accuracy",
            "macro_avg_steps_per_question",
            "macro_avg_retries_per_question",
            "global_max_attempts",
            "global_max_retry_index",
            "total_steps_with_multiple_attempts",
        ],
        rows=summary_rows,
    )

    print("[SUMMARY]")
    print(f"target_dir: {args.target_dir}")
    print(f"llm_judge files: {len(judge_map)}")
    print(f"logs files: {len(log_map)}")
    print(f"paired model-dataset runs: {len(pair_metrics)}")
    print(f"total questions: {total_questions}")
    print(f"overall accuracy (micro): {overall_accuracy:.6f}")
    print(f"avg steps/question (micro): {overall_avg_steps:.6f}")
    print(f"avg retries/question (micro): {overall_avg_retries:.6f}")
    print(f"macro accuracy (15-run mean): {macro_accuracy:.6f}")
    print(f"macro avg steps/question: {macro_avg_steps:.6f}")
    print(f"macro avg retries/question: {macro_avg_retries:.6f}")
    print(f"global max attempts per step: {global_max_attempts}")
    print(f"global max retry_index: {global_max_retry_index}")
    print(f"steps with multiple attempts: {total_steps_with_multiple_attempts}")
    print(f"detail CSV: {args.detail_csv}")
    print(f"summary CSV: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
