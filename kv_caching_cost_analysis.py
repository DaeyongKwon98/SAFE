import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer


GENERATOR_TOKENIZER_PATHS = {
    "qwen7b": "/workspace/hf_transformers/Qwen2.5-7B-Instruct",
    "llama8b": "/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct",
    "gemma12b": "/workspace/hf_transformers/gemma-3-12b-it",
}
EVALUATOR_TOKENIZER_PATH = "/workspace/hf_transformers/Qwen2.5-7B-Instruct"


@dataclass
class RunMetrics:
    run_name: str
    samples: int
    attempts_total: int
    attempts_avg: float
    avg_qp_tokens_gen: float
    avg_qp_tokens_eval: float
    avg_qp_tokens_total: float
    total_tokens_stats: int
    kv_saving_tokens_est: int
    kv_saving_ratio_pct: float
    tokens_after_kv_est: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze KV caching savings from *_logs.json and *_stats.json."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory containing run-level *_logs.json and *_stats.json files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path. Default: <target_dir>/kv_caching_cost_analysis.csv",
    )
    return parser.parse_args()


def safe_load_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"missing file: {path}"
    except json.JSONDecodeError as e:
        return None, f"json decode error: {path} ({e})"
    except Exception as e:
        return None, f"failed to load {path}: {e}"


def normalize_retrieved_passages(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except Exception:
                pass
        return [value]
    return []


def build_qp_segment(question: str, passages: List[str]) -> str:
    passages_str = "\n".join([f"Passage {idx + 1}: {p}" for idx, p in enumerate(passages)])
    return f"Question:\n{question}\n\nRetrieved Passages:\n{passages_str}\n\n"


def find_log_files(target_dir: Path) -> List[Path]:
    return sorted(target_dir.glob("*_logs.json"))


def validate_and_get_final_stats(
    stats_data: Any,
    run_name: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not isinstance(stats_data, list) or not stats_data:
        return None, [f"[{run_name}] stats is empty or not a list"]
    final = stats_data[-1]
    if not isinstance(final, dict):
        return None, [f"[{run_name}] final stats entry is not a dict"]
    required = ["generator_calls", "evaluator_calls", "total_tokens", "completed_count"]
    missing = [k for k in required if k not in final]
    if missing:
        warnings.append(f"[{run_name}] final stats missing keys: {missing}")
    return final, warnings


def compute_run_metrics(
    run_name: str,
    logs_data: Any,
    stats_final: Dict[str, Any],
    gen_tokenizer,
    eval_tokenizer,
) -> Tuple[Optional[RunMetrics], List[str]]:
    warnings: List[str] = []
    if not isinstance(logs_data, list):
        return None, [f"[{run_name}] logs is not a list"]
    if not logs_data:
        return None, [f"[{run_name}] logs is empty"]

    samples = len(logs_data)
    attempts_total = 0
    qp_tokens_gen_sum = 0
    qp_tokens_eval_sum = 0
    kv_saving_tokens_est = 0

    for record in logs_data:
        meta = record.get("meta_data", {}) if isinstance(record, dict) else {}
        question = str(meta.get("question", ""))
        passages = normalize_retrieved_passages(meta.get("retrieved_passages", []))
        qp_segment = build_qp_segment(question, passages)

        qp_gen = len(gen_tokenizer.encode(qp_segment, add_special_tokens=False))
        qp_eval = len(eval_tokenizer.encode(qp_segment, add_special_tokens=False))

        steps_history = record.get("steps_history", []) if isinstance(record, dict) else []
        attempts_per_sample = 0
        if isinstance(steps_history, list):
            for step in steps_history:
                attempts = step.get("attempts", []) if isinstance(step, dict) else []
                if isinstance(attempts, list):
                    attempts_per_sample += len(attempts)

        attempts_total += attempts_per_sample
        qp_tokens_gen_sum += qp_gen
        qp_tokens_eval_sum += qp_eval
        kv_saving_tokens_est += max(0, attempts_per_sample - 1) * (qp_gen + qp_eval)

    total_tokens_stats = int(stats_final.get("total_tokens", 0))
    generator_calls = int(stats_final.get("generator_calls", 0))
    evaluator_calls = int(stats_final.get("evaluator_calls", 0))
    completed_count = int(stats_final.get("completed_count", 0))

    if attempts_total != generator_calls or attempts_total != evaluator_calls:
        warnings.append(
            f"[{run_name}] integrity mismatch: attempts_total={attempts_total}, "
            f"generator_calls={generator_calls}, evaluator_calls={evaluator_calls}"
        )
    if samples != completed_count:
        warnings.append(
            f"[{run_name}] sample mismatch: samples={samples}, completed_count={completed_count}"
        )

    attempts_avg = attempts_total / samples
    avg_qp_tokens_gen = qp_tokens_gen_sum / samples
    avg_qp_tokens_eval = qp_tokens_eval_sum / samples
    avg_qp_tokens_total = avg_qp_tokens_gen + avg_qp_tokens_eval
    tokens_after_kv_est = total_tokens_stats - kv_saving_tokens_est
    kv_saving_ratio_pct = 0.0
    if total_tokens_stats > 0:
        kv_saving_ratio_pct = (kv_saving_tokens_est / total_tokens_stats) * 100.0

    if tokens_after_kv_est < 0:
        warnings.append(
            f"[{run_name}] invalid tokens_after_kv_est={tokens_after_kv_est} (< 0)"
        )
    if kv_saving_ratio_pct < 0 or kv_saving_ratio_pct > 100:
        warnings.append(
            f"[{run_name}] invalid kv_saving_ratio_pct={kv_saving_ratio_pct:.6f} (outside 0~100)"
        )

    metrics = RunMetrics(
        run_name=run_name,
        samples=samples,
        attempts_total=attempts_total,
        attempts_avg=attempts_avg,
        avg_qp_tokens_gen=avg_qp_tokens_gen,
        avg_qp_tokens_eval=avg_qp_tokens_eval,
        avg_qp_tokens_total=avg_qp_tokens_total,
        total_tokens_stats=total_tokens_stats,
        kv_saving_tokens_est=kv_saving_tokens_est,
        kv_saving_ratio_pct=kv_saving_ratio_pct,
        tokens_after_kv_est=tokens_after_kv_est,
    )
    return metrics, warnings


def aggregate_overall(rows: List[RunMetrics]) -> RunMetrics:
    total_samples = sum(r.samples for r in rows)
    attempts_total = sum(r.attempts_total for r in rows)
    total_tokens_stats = sum(r.total_tokens_stats for r in rows)
    kv_saving_tokens_est = sum(r.kv_saving_tokens_est for r in rows)
    tokens_after_kv_est = total_tokens_stats - kv_saving_tokens_est

    if total_samples == 0:
        return RunMetrics(
            run_name="OVERALL",
            samples=0,
            attempts_total=0,
            attempts_avg=0.0,
            avg_qp_tokens_gen=0.0,
            avg_qp_tokens_eval=0.0,
            avg_qp_tokens_total=0.0,
            total_tokens_stats=0,
            kv_saving_tokens_est=0,
            kv_saving_ratio_pct=0.0,
            tokens_after_kv_est=0,
        )

    weighted_avg_qp_gen = sum(r.avg_qp_tokens_gen * r.samples for r in rows) / total_samples
    weighted_avg_qp_eval = sum(r.avg_qp_tokens_eval * r.samples for r in rows) / total_samples
    weighted_avg_qp_total = weighted_avg_qp_gen + weighted_avg_qp_eval
    kv_saving_ratio_pct = 0.0
    if total_tokens_stats > 0:
        kv_saving_ratio_pct = (kv_saving_tokens_est / total_tokens_stats) * 100.0

    return RunMetrics(
        run_name="OVERALL",
        samples=total_samples,
        attempts_total=attempts_total,
        attempts_avg=attempts_total / total_samples,
        avg_qp_tokens_gen=weighted_avg_qp_gen,
        avg_qp_tokens_eval=weighted_avg_qp_eval,
        avg_qp_tokens_total=weighted_avg_qp_total,
        total_tokens_stats=total_tokens_stats,
        kv_saving_tokens_est=kv_saving_tokens_est,
        kv_saving_ratio_pct=kv_saving_ratio_pct,
        tokens_after_kv_est=tokens_after_kv_est,
    )


def metrics_to_dict(m: RunMetrics) -> Dict[str, Any]:
    return {
        "run_name": m.run_name,
        "samples": m.samples,
        "attempts_total": m.attempts_total,
        "attempts_avg": round(m.attempts_avg, 6),
        "avg_qp_tokens_gen": round(m.avg_qp_tokens_gen, 6),
        "avg_qp_tokens_eval": round(m.avg_qp_tokens_eval, 6),
        "avg_qp_tokens_total": round(m.avg_qp_tokens_total, 6),
        "total_tokens_stats": m.total_tokens_stats,
        "kv_saving_tokens_est": m.kv_saving_tokens_est,
        "kv_saving_ratio_pct": round(m.kv_saving_ratio_pct, 6),
        "tokens_after_kv_est": m.tokens_after_kv_est,
    }


def write_csv(path: Path, rows: List[RunMetrics]) -> None:
    fieldnames = [
        "run_name",
        "samples",
        "attempts_total",
        "attempts_avg",
        "avg_qp_tokens_gen",
        "avg_qp_tokens_eval",
        "avg_qp_tokens_total",
        "total_tokens_stats",
        "kv_saving_tokens_est",
        "kv_saving_ratio_pct",
        "tokens_after_kv_est",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(metrics_to_dict(row))


def print_table(rows: List[RunMetrics]) -> None:
    headers = [
        "run_name",
        "samples",
        "attempts_total",
        "attempts_avg",
        "avg_qp_tokens_gen",
        "avg_qp_tokens_eval",
        "avg_qp_tokens_total",
        "total_tokens_stats",
        "kv_saving_tokens_est",
        "kv_saving_ratio_pct",
        "tokens_after_kv_est",
    ]

    rendered: List[List[str]] = []
    for m in rows:
        rendered.append(
            [
                m.run_name,
                str(m.samples),
                str(m.attempts_total),
                f"{m.attempts_avg:.3f}",
                f"{m.avg_qp_tokens_gen:.2f}",
                f"{m.avg_qp_tokens_eval:.2f}",
                f"{m.avg_qp_tokens_total:.2f}",
                str(m.total_tokens_stats),
                str(m.kv_saving_tokens_est),
                f"{m.kv_saving_ratio_pct:.3f}",
                str(m.tokens_after_kv_est),
            ]
        )

    widths = []
    for col_idx, h in enumerate(headers):
        col_vals = [row[col_idx] for row in rendered]
        widths.append(max([len(h)] + [len(v) for v in col_vals]))

    def fmt_row(values: List[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rendered:
        print(fmt_row(row))


def main() -> int:
    args = parse_args()
    target_dir = Path(args.target_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"ERROR: invalid target_dir: {target_dir}")
        return 1

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else target_dir / "kv_caching_cost_analysis.csv"
    )

    log_files = find_log_files(target_dir)
    if not log_files:
        print(f"ERROR: no *_logs.json found in {target_dir}")
        return 1

    print("Loading tokenizers...")
    eval_tokenizer = AutoTokenizer.from_pretrained(
        EVALUATOR_TOKENIZER_PATH, trust_remote_code=True
    )
    generator_tokenizers = {}
    for model_prefix, model_path in GENERATOR_TOKENIZER_PATHS.items():
        generator_tokenizers[model_prefix] = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    rows: List[RunMetrics] = []
    warnings: List[str] = []

    for log_file in log_files:
        run_name = log_file.name.replace("_logs.json", "")
        model_prefix = run_name.split("_")[0]

        if model_prefix not in generator_tokenizers:
            warnings.append(
                f"[{run_name}] unknown model prefix '{model_prefix}', skipping run"
            )
            continue

        stats_file = log_file.with_name(f"{run_name}_stats.json")
        logs_data, err = safe_load_json(log_file)
        if err:
            warnings.append(f"[{run_name}] {err}")
            continue
        stats_data, err = safe_load_json(stats_file)
        if err:
            warnings.append(f"[{run_name}] {err}")
            continue

        stats_final, stat_warnings = validate_and_get_final_stats(stats_data, run_name)
        warnings.extend(stat_warnings)
        if stats_final is None:
            continue

        metrics, run_warnings = compute_run_metrics(
            run_name=run_name,
            logs_data=logs_data,
            stats_final=stats_final,
            gen_tokenizer=generator_tokenizers[model_prefix],
            eval_tokenizer=eval_tokenizer,
        )
        warnings.extend(run_warnings)
        if metrics is None:
            continue
        rows.append(metrics)

    if not rows:
        print("ERROR: no valid run metrics produced.")
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"- {w}")
        return 1

    rows = sorted(rows, key=lambda x: x.run_name)
    overall = aggregate_overall(rows)
    all_rows = rows + [overall]

    write_csv(output_csv, all_rows)
    print_table(all_rows)
    print(f"\nSaved CSV: {output_csv}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("\nWarnings: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
