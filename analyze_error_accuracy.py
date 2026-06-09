#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CORRECT_ERROR_TYPE = "Correct (No Error)"
CORRECT_LABEL = "correct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze *_logs.json and *_llm_judge.json pairs for error statistics "
            "and error-accuracy relations."
        )
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument(
        "--exclude_models",
        nargs="*",
        default=["oss20b"],
        help="Model names to exclude, e.g. --exclude_models oss20b",
    )
    parser.add_argument("--max_questions", type=int, default=1000)
    parser.add_argument(
        "--output_format",
        type=str,
        default="markdown,csv",
        help="Comma-separated output formats: markdown,csv",
    )
    return parser.parse_args()


def parse_list_tokens(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def parse_output_formats(value: str) -> List[str]:
    formats = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    if not formats:
        raise ValueError("output_format must include at least one of: markdown,csv")
    valid = {"markdown", "csv"}
    unknown = [fmt for fmt in formats if fmt not in valid]
    if unknown:
        raise ValueError(f"Unknown output_format values: {unknown}")
    return formats


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values)) / float(len(values))


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for xv, yv in zip(x, y):
        dx = xv - mx
        dy = yv - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    denom = math.sqrt(den_x * den_y)
    if denom == 0.0:
        return float("nan")
    return num / denom


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        return float("nan")
    return pearson_corr(rankdata(x), rankdata(y))


def to_csv_value(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return value


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_csv_value(row.get(k)) for k in fieldnames})


def stem_to_model_dataset(stem: str) -> Tuple[str, str]:
    if "_" not in stem:
        raise ValueError(f"Invalid stem without model/dataset separator: {stem}")
    model, dataset = stem.rsplit("_", 1)
    return model, dataset


def discover_pairs(input_dir: Path, exclude_models: Sequence[str]) -> List[Tuple[str, str, str, Path, Path]]:
    exclude_set = set(exclude_models)
    pairs: List[Tuple[str, str, str, Path, Path]] = []
    for log_path in sorted(input_dir.glob("*_logs.json")):
        stem = log_path.name[: -len("_logs.json")]
        model, dataset = stem_to_model_dataset(stem)
        if model in exclude_set:
            continue
        judge_path = input_dir / f"{stem}_llm_judge.json"
        if not judge_path.exists():
            raise FileNotFoundError(f"Missing llm_judge pair for {log_path.name}")
        pairs.append((model, dataset, stem, log_path, judge_path))
    return sorted(pairs, key=lambda x: (x[0], x[1]))


def validate_ratio_sum(rows: Sequence[Dict], key: str, group_key: str, tolerance: float = 1e-6):
    grouped: Dict[str, float] = {}
    for row in rows:
        g = str(row[group_key])
        grouped[g] = grouped.get(g, 0.0) + float(row[key])
    for group_name, ratio_sum in grouped.items():
        if abs(ratio_sum - 1.0) > tolerance:
            raise ValueError(
                f"Ratio sum check failed for group {group_name}: {ratio_sum:.10f} (expected 1.0)"
            )


def analyze_pair(
    model: str,
    dataset: str,
    log_path: Path,
    judge_path: Path,
    max_questions: int,
) -> Dict:
    logs = load_json(log_path)
    judges = load_json(judge_path)
    if not isinstance(logs, list) or not isinstance(judges, list):
        raise TypeError(f"{log_path.name} or {judge_path.name} is not a JSON list")
    if len(logs) < max_questions or len(judges) < max_questions:
        raise ValueError(
            f"{model}/{dataset}: insufficient rows logs={len(logs)} judges={len(judges)} "
            f"for max_questions={max_questions}"
        )

    logs = logs[:max_questions]
    judges = judges[:max_questions]

    total_attempts = 0
    total_error_attempts = 0
    error_counter: Counter = Counter()
    step_error_counter: Counter = Counter()
    location_counter: Counter = Counter()
    question_error_counts: List[Counter] = [Counter() for _ in range(max_questions)]
    labels: List[int] = []

    for idx in range(max_questions):
        log_item = logs[idx]
        judge_item = judges[idx]
        if not isinstance(log_item, dict) or not isinstance(judge_item, dict):
            raise TypeError(f"{model}/{dataset}: row {idx} is not a dict in logs/judges")

        q_log = ((log_item.get("meta_data") or {}).get("question") or "").strip()
        q_judge = (judge_item.get("question") or "").strip()
        if q_log != q_judge:
            raise ValueError(
                f"{model}/{dataset}: question mismatch at index={idx}\n"
                f"log={q_log}\njudge={q_judge}"
            )

        is_correct = str(judge_item.get("is_correct", "")).strip().lower() == CORRECT_LABEL
        labels.append(1 if is_correct else 0)

        steps_history = log_item.get("steps_history", [])
        if not isinstance(steps_history, list):
            raise TypeError(f"{model}/{dataset}: steps_history is not list at row {idx}")

        for step in steps_history:
            if not isinstance(step, dict):
                continue
            step_num = safe_int(step.get("step_num"))
            attempts = step.get("attempts", [])
            if not isinstance(attempts, list):
                continue

            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue

                retry_index = safe_int(attempt.get("retry_index"))
                evaluation = attempt.get("evaluation")
                if isinstance(evaluation, dict):
                    error_type = str(evaluation.get("error_type", "")).strip()
                else:
                    error_type = ""

                if not error_type:
                    error_type = "<MISSING_ERROR_TYPE>"

                total_attempts += 1
                error_counter[error_type] += 1
                question_error_counts[idx][error_type] += 1
                location_counter[(error_type, step_num, retry_index)] += 1

                if error_type != CORRECT_ERROR_TYPE:
                    total_error_attempts += 1
                    step_error_counter[step_num] += 1

    if sum(error_counter.values()) != total_attempts:
        raise ValueError(
            f"{model}/{dataset}: sum(error_type count)={sum(error_counter.values())} "
            f"!= total_attempts={total_attempts}"
        )
    if sum(step_error_counter.values()) != total_error_attempts:
        raise ValueError(
            f"{model}/{dataset}: sum(step error_count)={sum(step_error_counter.values())} "
            f"!= total_error_attempts={total_error_attempts}"
        )

    summary_rows: List[Dict] = []
    for error_type, count in sorted(error_counter.items(), key=lambda t: (-t[1], t[0])):
        summary_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "total_attempts": total_attempts,
                "total_error_attempts": total_error_attempts,
                "unique_error_types": len(error_counter),
                "error_type": error_type,
                "count": count,
                "ratio": (count / total_attempts) if total_attempts > 0 else float("nan"),
            }
        )

    step_rows: List[Dict] = []
    for step_num, count in sorted(step_error_counter.items(), key=lambda t: (t[0] is None, t[0])):
        step_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "step_num": step_num,
                "error_count": count,
                "error_ratio": (count / total_error_attempts) if total_error_attempts > 0 else float("nan"),
            }
        )

    location_rows: List[Dict] = []
    for error_type, total_for_type in sorted(error_counter.items(), key=lambda t: (-t[1], t[0])):
        local_items = [
            (step_num, retry_index, count)
            for (etype, step_num, retry_index), count in location_counter.items()
            if etype == error_type
        ]
        local_items.sort(key=lambda t: (-t[2], t[0] is None, t[0], t[1] is None, t[1]))
        for step_num, retry_index, count in local_items:
            location_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "error_type": error_type,
                    "step_num": step_num,
                    "retry_index": retry_index,
                    "count": count,
                    "ratio_within_error_type": (
                        count / total_for_type if total_for_type > 0 else float("nan")
                    ),
                }
            )

    ratio_check_rows = [
        {
            "error_type_key": f"{row['model']}::{row['dataset']}::{row['error_type']}",
            "ratio_within_error_type": row["ratio_within_error_type"],
        }
        for row in location_rows
    ]
    validate_ratio_sum(
        rows=ratio_check_rows,
        key="ratio_within_error_type",
        group_key="error_type_key",
    )

    relation_rows: List[Dict] = []
    for error_type, _count in sorted(error_counter.items(), key=lambda t: (-t[1], t[0])):
        x_counts = [counter.get(error_type, 0) for counter in question_error_counts]
        with_idx = [i for i, c in enumerate(x_counts) if c > 0]
        without_idx = [i for i, c in enumerate(x_counts) if c == 0]

        y_with = [labels[i] for i in with_idx]
        y_without = [labels[i] for i in without_idx]
        acc_with = mean(y_with)
        acc_without = mean(y_without)
        if math.isnan(acc_with) or math.isnan(acc_without):
            acc_gap = float("nan")
        else:
            acc_gap = acc_with - acc_without

        relation_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "error_type": error_type,
                "n_with": len(with_idx),
                "acc_with": acc_with,
                "n_without": len(without_idx),
                "acc_without": acc_without,
                "acc_gap": acc_gap,
                "point_biserial_r": pearson_corr(x_counts, labels),
                "spearman_rho": spearman_corr(x_counts, labels),
            }
        )

    accuracy = mean(labels)
    return {
        "model": model,
        "dataset": dataset,
        "total_attempts": total_attempts,
        "total_error_attempts": total_error_attempts,
        "error_only_ratio": (
            total_error_attempts / total_attempts if total_attempts > 0 else float("nan")
        ),
        "accuracy": accuracy,
        "summary_rows": summary_rows,
        "step_rows": step_rows,
        "location_rows": location_rows,
        "relation_rows": relation_rows,
    }


def fmt_float(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def build_markdown_report(
    input_dir: Path,
    max_questions: int,
    excluded_models: Sequence[str],
    combo_results: Sequence[Dict],
    summary_rows: Sequence[Dict],
    step_rows: Sequence[Dict],
    location_rows: Sequence[Dict],
    relation_rows: Sequence[Dict],
) -> str:
    lines: List[str] = []
    lines.append("# Error/Accuracy Analysis Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {dt.datetime.utcnow().isoformat()}Z")
    lines.append(f"- Input directory: `{input_dir}`")
    lines.append(f"- Excluded models: `{', '.join(excluded_models)}`")
    lines.append(f"- Max questions per pair: `{max_questions}`")
    lines.append(f"- Analyzed combinations: `{len(combo_results)}`")
    lines.append("")

    lines.append("## 15-Combination Summary")
    lines.append("")
    lines.append(
        "| model | dataset | total_attempts | total_error_attempts | error_only_ratio | llm_judge_accuracy | unique_error_types |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for combo in sorted(combo_results, key=lambda x: (x["model"], x["dataset"])):
        unique_error_types = len(
            {row["error_type"] for row in combo["summary_rows"] if row["model"] == combo["model"]}
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    combo["model"],
                    combo["dataset"],
                    str(combo["total_attempts"]),
                    str(combo["total_error_attempts"]),
                    fmt_float(combo["error_only_ratio"]),
                    fmt_float(combo["accuracy"]),
                    str(unique_error_types),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Top Error Types and Step/Retry Hotspots")
    lines.append("")
    for combo in sorted(combo_results, key=lambda x: (x["model"], x["dataset"])):
        model = combo["model"]
        dataset = combo["dataset"]
        lines.append(f"### {model} / {dataset}")

        local_summary = [
            row
            for row in combo["summary_rows"]
            if row["error_type"] != CORRECT_ERROR_TYPE
        ]
        local_summary.sort(key=lambda r: (-r["count"], r["error_type"]))
        top_errors = ", ".join(
            [f"{row['error_type']}={row['count']}" for row in local_summary[:5]]
        )
        if not top_errors:
            top_errors = "No non-correct errors"

        local_locations = [
            row
            for row in combo["location_rows"]
            if row["error_type"] != CORRECT_ERROR_TYPE
        ]
        local_locations.sort(key=lambda r: (-r["count"], r["step_num"] is None, r["step_num"]))
        top_hotspots = ", ".join(
            [
                f"{row['error_type']}@step{row['step_num']}/retry{row['retry_index']}={row['count']}"
                for row in local_locations[:5]
            ]
        )
        if not top_hotspots:
            top_hotspots = "No non-correct hotspots"

        local_rel = [
            row
            for row in combo["relation_rows"]
            if row["error_type"] != CORRECT_ERROR_TYPE
            and row["n_with"] > 0
            and row["n_without"] > 0
            and not math.isnan(row["acc_gap"])
        ]
        local_rel.sort(key=lambda r: (r["acc_gap"], -r["n_with"]))
        if local_rel:
            strongest = local_rel[0]
            rel_text = (
                f"{strongest['error_type']} "
                f"(acc_gap={fmt_float(strongest['acc_gap'])}, "
                f"n_with={strongest['n_with']}, "
                f"point_biserial={fmt_float(strongest['point_biserial_r'])}, "
                f"spearman={fmt_float(strongest['spearman_rho'])})"
            )
        else:
            rel_text = "Not available"

        lines.append(f"- Top non-correct error types: {top_errors}")
        lines.append(f"- Top error hotspots (step/retry): {top_hotspots}")
        lines.append(f"- Strongest negative accuracy association: {rel_text}")
        lines.append("")

    lines.append("## Cross-Combination Insights")
    lines.append("")

    global_error_counter: Counter = Counter()
    for row in summary_rows:
        if row["error_type"] == CORRECT_ERROR_TYPE:
            continue
        global_error_counter[row["error_type"]] += int(row["count"])
    top_global_errors = ", ".join(
        [f"{name}={count}" for name, count in global_error_counter.most_common(10)]
    )
    lines.append(f"- Most frequent non-correct error types overall: {top_global_errors}")

    global_step_counter: Counter = Counter()
    for row in step_rows:
        global_step_counter[row["step_num"]] += int(row["error_count"])
    top_global_steps = ", ".join(
        [f"step{step}={count}" for step, count in global_step_counter.most_common(7)]
    )
    lines.append(f"- Error concentration by step index: {top_global_steps}")

    relation_candidates = [
        row
        for row in relation_rows
        if row["error_type"] != CORRECT_ERROR_TYPE
        and row["n_with"] >= 30
        and row["n_without"] >= 30
        and not math.isnan(row["acc_gap"])
    ]
    relation_candidates.sort(key=lambda r: (r["acc_gap"], -r["n_with"]))
    strongest_rows = relation_candidates[:10]
    if strongest_rows:
        lines.append("- Strongest negative acc_gap examples (n_with>=30):")
        for row in strongest_rows:
            lines.append(
                "  - "
                f"{row['model']}/{row['dataset']} | {row['error_type']} | "
                f"acc_gap={fmt_float(row['acc_gap'])} | "
                f"point_biserial={fmt_float(row['point_biserial_r'])} | "
                f"spearman={fmt_float(row['spearman_rho'])}"
            )
    else:
        lines.append("- No relation rows met the n_with/n_without threshold for robust comparison.")

    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `summary_error_counts.csv`")
    lines.append("- `error_step_stats.csv`")
    lines.append("- `error_location_stats.csv`")
    lines.append("- `error_accuracy_relation.csv`")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist or is not a directory: {input_dir}")

    exclude_models = parse_list_tokens(args.exclude_models)
    output_formats = parse_output_formats(args.output_format)

    pairs = discover_pairs(input_dir=input_dir, exclude_models=exclude_models)
    if len(pairs) != 15:
        raise ValueError(
            f"Expected exactly 15 combinations after exclusion; found {len(pairs)}. "
            f"Pairs={[(m, d) for m, d, _, _, _ in pairs]}"
        )

    all_summary_rows: List[Dict] = []
    all_step_rows: List[Dict] = []
    all_location_rows: List[Dict] = []
    all_relation_rows: List[Dict] = []
    combo_results: List[Dict] = []

    for model, dataset, _stem, log_path, judge_path in pairs:
        result = analyze_pair(
            model=model,
            dataset=dataset,
            log_path=log_path,
            judge_path=judge_path,
            max_questions=args.max_questions,
        )
        combo_results.append(result)
        all_summary_rows.extend(result["summary_rows"])
        all_step_rows.extend(result["step_rows"])
        all_location_rows.extend(result["location_rows"])
        all_relation_rows.extend(result["relation_rows"])

    if "csv" in output_formats:
        write_csv(
            input_dir / "summary_error_counts.csv",
            [
                "model",
                "dataset",
                "total_attempts",
                "total_error_attempts",
                "unique_error_types",
                "error_type",
                "count",
                "ratio",
            ],
            all_summary_rows,
        )
        write_csv(
            input_dir / "error_step_stats.csv",
            [
                "model",
                "dataset",
                "step_num",
                "error_count",
                "error_ratio",
            ],
            all_step_rows,
        )
        write_csv(
            input_dir / "error_location_stats.csv",
            [
                "model",
                "dataset",
                "error_type",
                "step_num",
                "retry_index",
                "count",
                "ratio_within_error_type",
            ],
            all_location_rows,
        )
        write_csv(
            input_dir / "error_accuracy_relation.csv",
            [
                "model",
                "dataset",
                "error_type",
                "n_with",
                "acc_with",
                "n_without",
                "acc_without",
                "acc_gap",
                "point_biserial_r",
                "spearman_rho",
            ],
            all_relation_rows,
        )

    if "markdown" in output_formats:
        report = build_markdown_report(
            input_dir=input_dir,
            max_questions=args.max_questions,
            excluded_models=exclude_models,
            combo_results=combo_results,
            summary_rows=all_summary_rows,
            step_rows=all_step_rows,
            location_rows=all_location_rows,
            relation_rows=all_relation_rows,
        )
        report_path = input_dir / "error_analysis_report.md"
        report_path.write_text(report, encoding="utf-8")

    print("Analysis complete.")
    print(f"input_dir={input_dir}")
    print(f"exclude_models={exclude_models}")
    print(f"combinations={len(pairs)}")
    if "csv" in output_formats:
        print("generated: summary_error_counts.csv, error_step_stats.csv, error_location_stats.csv, error_accuracy_relation.csv")
    if "markdown" in output_formats:
        print("generated: error_analysis_report.md")


if __name__ == "__main__":
    main()
