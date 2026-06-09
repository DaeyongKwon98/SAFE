#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MODELS: Sequence[str] = ("qwen4b", "qwen8b", "qwen14b", "llama8b", "gemma12b")
DATASETS: Sequence[str] = ("2wiki", "hotpotqa", "musique")
QWEN_MODELS = {"qwen4b", "qwen8b", "qwen14b"}

CORRECT_TOKENS = {"correct", "true", "1", "yes"}
WRONG_TOKENS = {"wrong", "incorrect", "false", "0", "no"}

DEFAULT_BASE_DIR = Path("/workspace/daeyong/inference_results")
DEFAULT_SUMMARY_CSV = (
    DEFAULT_BASE_DIR / "self_feedback_kg_correct_1k_sample_grid_metrics_summary.csv"
)
DEFAULT_DETAIL_CSV = (
    DEFAULT_BASE_DIR / "self_feedback_kg_correct_1k_sample_grid_metrics_detail.csv"
)


@dataclass
class PairResult:
    max_steps: int
    max_retries: int
    model: str
    dataset: str
    strategy: str
    source_dir: Path
    log_file: str
    judge_file: str
    status: str
    note: str
    total_questions: int = 0
    correct_questions: int = 0
    unknown_label_count: int = 0
    total_steps: int = 0
    total_retries: int = 0

    @property
    def accuracy(self) -> Optional[float]:
        if self.total_questions <= 0:
            return None
        return self.correct_questions / self.total_questions

    @property
    def avg_steps_per_question(self) -> Optional[float]:
        if self.total_questions <= 0:
            return None
        return self.total_steps / self.total_questions

    @property
    def avg_retries_per_question(self) -> Optional[float]:
        if self.total_questions <= 0:
            return None
        return self.total_retries / self.total_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate self-feedback metrics over "
            "self_feedback_kg_correct_1k_sample_{max_steps}_{max_retries} grid."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base inference_results directory.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="7,10,13",
        help="Comma-separated max_steps values.",
    )
    parser.add_argument(
        "--retries",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated max_retries values.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS),
        help="Comma-separated model keys.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS),
        help="Comma-separated dataset keys.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Output CSV path for per-(max_steps,max_retries) summary.",
    )
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=DEFAULT_DETAIL_CSV,
        help="Output CSV path for per model/dataset detail rows.",
    )
    return parser.parse_args()


def parse_int_tokens(value: str) -> Tuple[int, ...]:
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
    return tuple(sorted(set(out)))


def parse_text_tokens(value: str) -> Tuple[str, ...]:
    out: List[str] = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            out.append(token)
    if not out:
        raise ValueError(f"No valid tokens parsed from: {value}")
    return tuple(dict.fromkeys(out))


def load_json_list(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"JSON root is not list: {path}")
    return data


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


def resolve_paths(
    base_dir: Path, max_steps: int, max_retries: int, model: str, dataset: str
) -> Tuple[str, Path, str, str]:
    if max_steps == 10 and max_retries == 3:
        if model in QWEN_MODELS:
            source_dir = base_dir / "self_feedback_kg_correct_1k_sample"
            log_file = f"{model}_{dataset}_logs.json"
        elif model == "llama8b":
            source_dir = base_dir / "self_feedback_Meta-Llama-3.1-8B-Instruct"
            log_file = f"{dataset}_logs.json"
        elif model == "gemma12b":
            source_dir = base_dir / "self_feedback_gemma-3-12b-it"
            log_file = f"{dataset}_logs.json"
        else:
            source_dir = base_dir / "self_feedback_kg_correct_1k_sample"
            log_file = f"{model}_{dataset}_logs.json"
        judge_file = f"{model}_{dataset}_llm_judge.json"
        return "fallback_10_3", source_dir, log_file, judge_file

    source_dir = base_dir / f"self_feedback_kg_correct_1k_sample_{max_steps}_{max_retries}"
    log_file = f"{model}_{dataset}_logs.json"
    judge_file = f"{model}_{dataset}_llm_judge.json"
    return "default_grid_folder", source_dir, log_file, judge_file


def compute_pair_result(
    base_dir: Path, max_steps: int, max_retries: int, model: str, dataset: str
) -> PairResult:
    strategy, source_dir, log_file, judge_file = resolve_paths(
        base_dir=base_dir,
        max_steps=max_steps,
        max_retries=max_retries,
        model=model,
        dataset=dataset,
    )
    log_path = source_dir / log_file
    judge_path = source_dir / judge_file

    if not log_path.exists() and not judge_path.exists():
        return PairResult(
            max_steps=max_steps,
            max_retries=max_retries,
            model=model,
            dataset=dataset,
            strategy=strategy,
            source_dir=source_dir,
            log_file=log_file,
            judge_file=judge_file,
            status="missing_both",
            note="log and judge files are both missing",
        )
    if not log_path.exists():
        return PairResult(
            max_steps=max_steps,
            max_retries=max_retries,
            model=model,
            dataset=dataset,
            strategy=strategy,
            source_dir=source_dir,
            log_file=log_file,
            judge_file=judge_file,
            status="missing_log",
            note="log file is missing",
        )
    if not judge_path.exists():
        return PairResult(
            max_steps=max_steps,
            max_retries=max_retries,
            model=model,
            dataset=dataset,
            strategy=strategy,
            source_dir=source_dir,
            log_file=log_file,
            judge_file=judge_file,
            status="missing_judge",
            note="judge file is missing",
        )

    try:
        logs = load_json_list(log_path)
    except Exception as exc:  # pylint: disable=broad-except
        return PairResult(
            max_steps=max_steps,
            max_retries=max_retries,
            model=model,
            dataset=dataset,
            strategy=strategy,
            source_dir=source_dir,
            log_file=log_file,
            judge_file=judge_file,
            status="invalid_log_json",
            note=str(exc),
        )

    try:
        judges = load_json_list(judge_path)
    except Exception as exc:  # pylint: disable=broad-except
        return PairResult(
            max_steps=max_steps,
            max_retries=max_retries,
            model=model,
            dataset=dataset,
            strategy=strategy,
            source_dir=source_dir,
            log_file=log_file,
            judge_file=judge_file,
            status="invalid_judge_json",
            note=str(exc),
        )

    if len(logs) != len(judges):
        return PairResult(
            max_steps=max_steps,
            max_retries=max_retries,
            model=model,
            dataset=dataset,
            strategy=strategy,
            source_dir=source_dir,
            log_file=log_file,
            judge_file=judge_file,
            status="length_mismatch",
            note=f"logs={len(logs)} judges={len(judges)}",
        )

    correct_questions = 0
    unknown_label_count = 0
    for row in judges:
        if not isinstance(row, dict):
            return PairResult(
                max_steps=max_steps,
                max_retries=max_retries,
                model=model,
                dataset=dataset,
                strategy=strategy,
                source_dir=source_dir,
                log_file=log_file,
                judge_file=judge_file,
                status="invalid_judge_schema",
                note="judge row is not dict",
            )
        norm = normalize_is_correct(row.get("is_correct"))
        if norm is True:
            correct_questions += 1
        elif norm is None:
            unknown_label_count += 1

    total_steps = 0
    total_retries = 0
    for row in logs:
        if not isinstance(row, dict):
            return PairResult(
                max_steps=max_steps,
                max_retries=max_retries,
                model=model,
                dataset=dataset,
                strategy=strategy,
                source_dir=source_dir,
                log_file=log_file,
                judge_file=judge_file,
                status="invalid_log_schema",
                note="log row is not dict",
            )
        steps_history = row.get("steps_history")
        if not isinstance(steps_history, list):
            return PairResult(
                max_steps=max_steps,
                max_retries=max_retries,
                model=model,
                dataset=dataset,
                strategy=strategy,
                source_dir=source_dir,
                log_file=log_file,
                judge_file=judge_file,
                status="invalid_log_schema",
                note="steps_history is not list",
            )

        total_steps += len(steps_history)
        for step in steps_history:
            if not isinstance(step, dict):
                continue
            attempts = step.get("attempts", [])
            if not isinstance(attempts, list):
                continue
            total_retries += max(0, len(attempts) - 1)

    return PairResult(
        max_steps=max_steps,
        max_retries=max_retries,
        model=model,
        dataset=dataset,
        strategy=strategy,
        source_dir=source_dir,
        log_file=log_file,
        judge_file=judge_file,
        status="ok",
        note="",
        total_questions=len(judges),
        correct_questions=correct_questions,
        unknown_label_count=unknown_label_count,
        total_steps=total_steps,
        total_retries=total_retries,
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


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def format_row(values: Sequence[str]) -> str:
        return " | ".join(v.ljust(widths[idx]) for idx, v in enumerate(values))

    print(format_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(format_row(row))


def format_float_or_na(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def build_detail_row(result: PairResult) -> Dict:
    return {
        "max_steps": result.max_steps,
        "max_retries": result.max_retries,
        "model": result.model,
        "dataset": result.dataset,
        "strategy": result.strategy,
        "source_dir": str(result.source_dir),
        "log_file": result.log_file,
        "judge_file": result.judge_file,
        "status": result.status,
        "note": result.note,
        "total_questions": result.total_questions if result.status == "ok" else "NA",
        "correct_questions": result.correct_questions if result.status == "ok" else "NA",
        "unknown_label_count": result.unknown_label_count if result.status == "ok" else "NA",
        "accuracy": format_float_or_na(result.accuracy),
        "avg_steps_per_question": format_float_or_na(result.avg_steps_per_question),
        "avg_retries_per_question": format_float_or_na(result.avg_retries_per_question),
        "total_steps": result.total_steps if result.status == "ok" else "NA",
        "total_retries": result.total_retries if result.status == "ok" else "NA",
    }


def build_summary_row(
    max_steps: int, max_retries: int, expected_pair_count: int, pair_results: Sequence[PairResult]
) -> Dict:
    ok_pairs = [r for r in pair_results if r.status == "ok"]
    pair_count = len(ok_pairs)

    total_questions = sum(r.total_questions for r in ok_pairs)
    total_correct = sum(r.correct_questions for r in ok_pairs)
    total_steps = sum(r.total_steps for r in ok_pairs)
    total_retries = sum(r.total_retries for r in ok_pairs)

    if pair_count == 0 or total_questions == 0:
        return {
            "max_steps": max_steps,
            "max_retries": max_retries,
            "avg_steps_per_question": "NA",
            "avg_retries_per_question": "NA",
            "accuracy": "NA",
            "pair_count": pair_count,
            "expected_pair_count": expected_pair_count,
            "total_questions": total_questions,
        }

    return {
        "max_steps": max_steps,
        "max_retries": max_retries,
        "avg_steps_per_question": total_steps / total_questions,
        "avg_retries_per_question": total_retries / total_questions,
        "accuracy": total_correct / total_questions,
        "pair_count": pair_count,
        "expected_pair_count": expected_pair_count,
        "total_questions": total_questions,
    }


def main() -> int:
    args = parse_args()

    try:
        steps = parse_int_tokens(args.steps)
        retries = parse_int_tokens(args.retries)
        models = parse_text_tokens(args.models)
        datasets = parse_text_tokens(args.datasets)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    expected_pair_count = len(models) * len(datasets)

    detail_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for max_steps in steps:
        for max_retries in retries:
            pair_results: List[PairResult] = []
            for model in models:
                for dataset in datasets:
                    pair_result = compute_pair_result(
                        base_dir=args.base_dir,
                        max_steps=max_steps,
                        max_retries=max_retries,
                        model=model,
                        dataset=dataset,
                    )
                    pair_results.append(pair_result)
                    detail_rows.append(build_detail_row(pair_result))

            summary_rows.append(
                build_summary_row(
                    max_steps=max_steps,
                    max_retries=max_retries,
                    expected_pair_count=expected_pair_count,
                    pair_results=pair_results,
                )
            )

    summary_rows.sort(key=lambda row: (int(row["max_steps"]), int(row["max_retries"])))
    detail_rows.sort(
        key=lambda row: (
            int(row["max_steps"]),
            int(row["max_retries"]),
            row["model"],
            row["dataset"],
        )
    )

    write_csv(
        path=args.summary_csv,
        fieldnames=[
            "max_steps",
            "max_retries",
            "avg_steps_per_question",
            "avg_retries_per_question",
            "accuracy",
            "pair_count",
            "expected_pair_count",
            "total_questions",
        ],
        rows=summary_rows,
    )
    write_csv(
        path=args.detail_csv,
        fieldnames=[
            "max_steps",
            "max_retries",
            "model",
            "dataset",
            "strategy",
            "source_dir",
            "log_file",
            "judge_file",
            "status",
            "note",
            "total_questions",
            "correct_questions",
            "unknown_label_count",
            "accuracy",
            "avg_steps_per_question",
            "avg_retries_per_question",
            "total_steps",
            "total_retries",
        ],
        rows=detail_rows,
    )

    print("[SUMMARY] self-feedback grid metrics")
    table_rows: List[Tuple[str, ...]] = []
    for row in summary_rows:
        table_rows.append(
            (
                str(row["max_steps"]),
                str(row["max_retries"]),
                str(row["avg_steps_per_question"]),
                str(row["avg_retries_per_question"]),
                str(row["accuracy"]),
                f"{row['pair_count']}/{row['expected_pair_count']}",
                str(row["total_questions"]),
            )
        )
    print_table(
        headers=(
            "max_steps",
            "max_retries",
            "avg_steps/question",
            "avg_retries/question",
            "accuracy",
            "pairs",
            "total_questions",
        ),
        rows=table_rows,
    )
    print()
    print(f"summary CSV: {args.summary_csv}")
    print(f"detail CSV: {args.detail_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
