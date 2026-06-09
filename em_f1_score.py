#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_INPUT_DIR = "/workspace/daeyong/inference_results/dev_kg_correct_1ksample_2wiki_ver3_newprompt_v2_qwen3_8b_2wiki_added_ver3_maxlength_maxlength_caching"
DEFAULT_BENCHMARK_DIR = "/workspace/daeyong/benchmarks"
DEFAULT_PATTERN = "*_final_answer.json"
VALID_DATASETS = ("2wiki", "hotpotqa", "musique")
FILE_RE = re.compile(r"^(?P<model>.+)_(?P<dataset>2wiki|hotpotqa|musique)_final_answer\.json$")
REQUIRED_KEYS = {"id", "question", "final_answer", "ground_truth"}
FINAL_ANSWER_SUFFIX_RE = re.compile(r"\s*\(\s*final\s*answer\s*\)\s*$", re.IGNORECASE)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text.strip()


def strip_final_answer_suffix(text: Any) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return FINAL_ANSWER_SUFFIX_RE.sub("", text).strip()


def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction == ground_truth else 0.0


def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: Sequence[str]) -> float:
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def parse_answer_list(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        values = raw_value
    elif isinstance(raw_value, (tuple, set)):
        values = list(raw_value)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                values = parsed
            elif isinstance(parsed, (tuple, set)):
                values = list(parsed)
            else:
                values = [parsed]
        except Exception:
            values = [raw_value]
    else:
        values = [raw_value]

    output: List[str] = []
    for value in values:
        text = "" if value is None else str(value)
        if text.strip():
            output.append(text)
    return output


def load_musique_aliases(benchmark_dir: Path, strict: bool) -> Dict[str, List[str]]:
    csv_path = benchmark_dir / "musique_dev_kg_correct.csv"
    if not csv_path.exists():
        msg = f"Musique benchmark CSV not found: {csv_path}"
        if strict:
            fail(msg)
        warn(msg)
        return {}

    aliases_by_id: Dict[str, List[str]] = {}

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                sample_id = str(row.get("id", "")).strip()
                if not sample_id:
                    continue

                answer_candidates = parse_answer_list(row.get("answer_list"))
                if not answer_candidates:
                    answer_value = row.get("answer")
                    if answer_value is not None:
                        answer_candidates = [str(answer_value)]

                normalized = [normalize_text(x) for x in answer_candidates if normalize_text(x)]
                if normalized:
                    aliases_by_id[sample_id] = normalized
                elif strict:
                    fail(f"Empty answer aliases at musique_dev_kg_correct.csv row {row_idx}")
    except Exception as exc:
        msg = f"Failed to load musique aliases from {csv_path}: {exc}"
        if strict:
            fail(msg)
        warn(msg)
        return {}

    return aliases_by_id


def discover_files(
    input_dir: Path,
    pattern: str,
    datasets: Sequence[str],
    models: Optional[Sequence[str]],
    strict: bool,
) -> List[Tuple[Path, str, str]]:
    if not input_dir.exists() or not input_dir.is_dir():
        fail(f"Input directory does not exist or is not a directory: {input_dir}")

    wanted_datasets = set(datasets)
    wanted_models = set(models) if models else None

    discovered: List[Tuple[Path, str, str]] = []
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue

        match = FILE_RE.match(path.name)
        if not match:
            msg = f"Skipping file with unmatched naming format: {path.name}"
            if strict:
                fail(msg)
            warn(msg)
            continue

        model = match.group("model")
        dataset = match.group("dataset")

        if dataset not in wanted_datasets:
            continue
        if wanted_models is not None and model not in wanted_models:
            continue

        discovered.append((path, model, dataset))

    return discovered


def load_json_records(path: Path, strict: bool) -> Optional[List[Dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        msg = f"Failed to read JSON file {path}: {exc}"
        if strict:
            fail(msg)
        warn(msg)
        return None

    if not isinstance(data, list):
        msg = f"JSON root must be a list in {path}"
        if strict:
            fail(msg)
        warn(msg)
        return None

    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            msg = f"Row {idx} in {path} is not an object"
            if strict:
                fail(msg)
            warn(msg)
            return None

        missing = REQUIRED_KEYS - set(row.keys())
        if missing:
            msg = f"Row {idx} in {path} is missing required keys: {sorted(missing)}"
            if strict:
                fail(msg)
            warn(msg)
            return None

    return data


def compute_metrics(
    records: Sequence[Dict[str, Any]],
    dataset: str,
    musique_aliases_by_id: Dict[str, List[str]],
) -> Dict[str, float]:
    if not records:
        return {
            "n_samples": 0,
            "empty_prediction_count": 0,
            "em": 0.0,
            "f1": 0.0,
        }

    em_sum = 0.0
    f1_sum = 0.0
    empty_pred_count = 0

    for row in records:
        pred_norm = normalize_text(strip_final_answer_suffix(row.get("final_answer", "")))
        if pred_norm == "":
            empty_pred_count += 1

        if dataset == "musique":
            sample_id = str(row.get("id", "")).strip()
            gt_candidates = musique_aliases_by_id.get(sample_id)
            if not gt_candidates:
                gt_candidates = [normalize_text(row.get("ground_truth", ""))]
        else:
            gt_candidates = [normalize_text(row.get("ground_truth", ""))]

        gt_candidates = [gt for gt in gt_candidates if gt != ""]
        if not gt_candidates:
            gt_candidates = [""]

        em_sum += metric_max_over_ground_truths(calculate_exact_match, pred_norm, gt_candidates)
        f1_sum += metric_max_over_ground_truths(calculate_f1_score, pred_norm, gt_candidates)

    n_samples = len(records)
    return {
        "n_samples": float(n_samples),
        "empty_prediction_count": float(empty_pred_count),
        "em": em_sum / n_samples,
        "f1": f1_sum / n_samples,
    }


def print_results(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        print("No files matched the current filters.")
        return

    header = f"{'model':<14} {'dataset':<9} {'n':>7} {'EM(%)':>8} {'F1(%)':>8} {'empty_pred':>11}"
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['model']:<14} {row['dataset']:<9} "
            f"{int(row['n_samples']):>7} {row['em'] * 100:>8.2f} {row['f1'] * 100:>8.2f} "
            f"{int(row['empty_prediction_count']):>11}"
        )

    em_file_avg = mean([row["em"] for row in rows])
    f1_file_avg = mean([row["f1"] for row in rows])
    total_samples = sum(int(row["n_samples"]) for row in rows)

    print("-" * len(header))
    print(
        f"FILE-AVG (across {len(rows)} files): EM={em_file_avg * 100:.2f}, "
        f"F1={f1_file_avg * 100:.2f}, total_samples={total_samples}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Exact Match and F1 for *_final_answer.json files.",
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory with *_final_answer.json files")
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR, help="Directory with benchmark CSV files")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern for input file discovery")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(VALID_DATASETS),
        choices=list(VALID_DATASETS),
        help="Datasets to include",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Optional model-name filters")
    parser.add_argument("--strict", action="store_true", help="Fail immediately on malformed files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    benchmark_dir = Path(args.benchmark_dir)

    discovered_files = discover_files(
        input_dir=input_dir,
        pattern=args.pattern,
        datasets=args.datasets,
        models=args.models,
        strict=args.strict,
    )

    if not discovered_files:
        print("No files found for the given filters.")
        return 0

    needs_musique = any(dataset == "musique" for _, _, dataset in discovered_files)
    musique_aliases_by_id = load_musique_aliases(benchmark_dir, strict=args.strict) if needs_musique else {}

    results: List[Dict[str, Any]] = []

    for file_path, model, dataset in discovered_files:
        records = load_json_records(file_path, strict=args.strict)
        if records is None:
            continue

        metrics = compute_metrics(records, dataset=dataset, musique_aliases_by_id=musique_aliases_by_id)
        results.append(
            {
                "file": str(file_path),
                "model": model,
                "dataset": dataset,
                **metrics,
            }
        )

    results.sort(key=lambda x: (x["model"], x["dataset"]))
    print_results(results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        raise SystemExit(1)
