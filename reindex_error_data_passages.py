import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_INPUT_ROOT = "/workspace/daeyong/filtering_noise_data/false_triples_oss120b_seed42"
DEFAULT_OUTPUT_DIR_NAME = "error_data_reindexed"
DEFAULT_DATASETS = ["2wiki", "hotpotqa", "musique"]
DEFAULT_ERROR_TYPES = [
    "logical_fallacy",
    "information_miss",
    "inefficiency",
    "redundancy",
    "off_topic",
    "overthinking",
    "contradictory",
    "unsupported",
    "premature_attribution",
    "wrong_conclusion",
]


@dataclass
class DatasetAssets:
    sample_by_index: Dict[str, dict]
    sample_by_question: Dict[str, dict]
    benchmark_by_id: Dict[str, dict]
    benchmark_by_question: Dict[str, dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reindex Passage N references in error_data JSONs from GT-passage indices "
            "to benchmark retrieved_passages indices."
        )
    )
    parser.add_argument("--input_root", type=str, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_dir_name", type=str, default=DEFAULT_OUTPUT_DIR_NAME)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--error_types", nargs="+", default=DEFAULT_ERROR_TYPES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Validation only; no output writes.")
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail-fast on any join/mapping/reference mismatch (default).",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Continue and keep unresolved references unchanged.",
    )
    parser.set_defaults(strict=True)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_list_like(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed]
        return [str(parsed).strip()]
    except Exception:
        return [text]


def write_json(path: Path, payload, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_id(value) -> str:
    return str(value).strip()


def load_samples(input_root: Path, dataset: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    sample_path = input_root / "samples" / f"{dataset}_sampled_200_input.json"
    data = json.loads(sample_path.read_text(encoding="utf-8"))

    by_index: Dict[str, dict] = {}
    by_question: Dict[str, dict] = {}
    for row in data:
        sample_index = normalize_id(row.get("sample_index"))
        question = normalize_text(row.get("question"))
        if sample_index and sample_index not in by_index:
            by_index[sample_index] = row
        if question and question not in by_question:
            by_question[question] = row
    return by_index, by_question


def load_benchmark_maps(dataset: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    csv_path = Path("/workspace/daeyong/benchmarks") / f"{dataset}.csv"
    by_id: Dict[str, dict] = {}
    by_question: Dict[str, dict] = {}
    csv.field_size_limit(2**31 - 1)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        id_col = None
        if reader.fieldnames:
            if "id" in reader.fieldnames:
                id_col = "id"
            elif "_id" in reader.fieldnames:
                id_col = "_id"
        if id_col is None:
            raise ValueError(f"No id column ('id' or '_id') in {csv_path}")
        for row in reader:
            row_id = normalize_id(row.get(id_col))
            if not row_id or row_id in by_id:
                pass
            else:
                by_id[row_id] = row
            q = normalize_text(row.get("question"))
            if q and q not in by_question:
                by_question[q] = row
    return by_id, by_question


def load_assets(input_root: Path, dataset: str) -> DatasetAssets:
    sample_by_index, sample_by_question = load_samples(input_root, dataset)
    benchmark_by_id, benchmark_by_question = load_benchmark_maps(dataset)
    return DatasetAssets(
        sample_by_index=sample_by_index,
        sample_by_question=sample_by_question,
        benchmark_by_id=benchmark_by_id,
        benchmark_by_question=benchmark_by_question,
    )


def get_sample_row(item: dict, assets: DatasetAssets, strict: bool) -> dict:
    q = normalize_text(item.get("question"))
    idx = normalize_id(item.get("sample_index"))

    row = None
    if idx:
        row = assets.sample_by_index.get(idx)
        if row and q and normalize_text(row.get("question")) != q:
            if strict:
                raise ValueError(
                    f"Question mismatch for sample_index={idx}: "
                    f"item='{q[:80]}', sample='{normalize_text(row.get('question'))[:80]}'"
                )
            row = None

    if row is None and q:
        row = assets.sample_by_question.get(q)

    if row is None:
        raise ValueError(f"Cannot join sample row for question='{q[:120]}' sample_index='{idx}'")
    return row


def get_new_retrieved_passages(
    sample_row: dict,
    item_question: str,
    assets: DatasetAssets,
    strict: bool,
) -> List[str]:
    benchmark_id = normalize_id(sample_row.get("benchmark_id"))
    row = None
    if benchmark_id and benchmark_id.lower() != "none":
        row = assets.benchmark_by_id.get(benchmark_id)
        if row is None and strict:
            raise ValueError(f"benchmark_id '{benchmark_id}' not found in benchmark csv")
    if row is None:
        # Fallback for datasets where benchmark_id was not saved in sampled_input.
        q = normalize_text(item_question) or normalize_text(sample_row.get("question"))
        row = assets.benchmark_by_question.get(q)
        if row is None:
            raise ValueError(
                f"Cannot find benchmark row by question fallback: '{q[:120]}'"
            )
    passages = parse_list_like(row.get("retrieved_passages"))
    if not passages:
        raise ValueError(f"Empty retrieved_passages for benchmark_id '{benchmark_id}'")
    return passages


def build_old_to_new_map(old_passages: List[str], new_passages: List[str], strict: bool) -> Dict[int, int]:
    new_norm = [normalize_text(x) for x in new_passages]
    mapping: Dict[int, int] = {}
    for i, old in enumerate(old_passages, start=1):
        old_norm = normalize_text(old)
        matches = [j for j, cand in enumerate(new_norm, start=1) if cand == old_norm]
        if len(matches) == 1:
            mapping[i] = matches[0]
        elif len(matches) == 0:
            raise ValueError(f"Cannot map old Passage {i} into benchmark retrieved_passages")
        else:
            if strict:
                raise ValueError(
                    f"Ambiguous mapping for old Passage {i}: matched indices={matches}"
                )
            mapping[i] = matches[0]
    return mapping


PASSAGE_REF_RE = re.compile(r"\bPassage\s*(\d+)\b")


def rewrite_step_text(
    step_text: str,
    mapping: Dict[int, int],
    strict: bool,
) -> Tuple[str, List[int]]:
    unresolved: List[int] = []

    def _repl(match: re.Match) -> str:
        old_idx = int(match.group(1))
        new_idx = mapping.get(old_idx)
        if new_idx is None:
            unresolved.append(old_idx)
            if strict:
                raise ValueError(
                    f"Reference to old Passage {old_idx} has no mapping in current item"
                )
            return match.group(0)
        return f"Passage {new_idx}"

    rewritten = PASSAGE_REF_RE.sub(_repl, step_text)
    return rewritten, unresolved


def rewrite_steps(
    steps: List[str],
    mapping: Dict[int, int],
    strict: bool,
) -> Tuple[List[str], List[dict]]:
    out: List[str] = []
    unresolved_logs: List[dict] = []
    for idx, s in enumerate(steps, start=1):
        if not isinstance(s, str):
            raise ValueError(f"Step at index {idx} is not string: type={type(s).__name__}")
        rewritten, unresolved = rewrite_step_text(s, mapping, strict)
        out.append(rewritten)
        if unresolved:
            unresolved_logs.append({"step_index": idx, "unresolved_old_passages": unresolved})
    return out, unresolved_logs


def check_passage_index_range(steps: List[str], retrieved_len: int) -> Optional[dict]:
    for idx, s in enumerate(steps, start=1):
        refs = [int(x) for x in PASSAGE_REF_RE.findall(s)]
        if not refs:
            continue
        if min(refs) < 1 or max(refs) > retrieved_len:
            return {
                "step_index": idx,
                "refs": refs,
                "retrieved_len": retrieved_len,
            }
    return None


def process_file(
    input_file: Path,
    output_file: Path,
    report_file: Path,
    assets: DatasetAssets,
    strict: bool,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    items = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Input JSON is not a list: {input_file}")

    out_items = []
    report = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "item_count_in": len(items),
        "item_count_out": 0,
        "join_failures": 0,
        "mapping_failures": 0,
        "rewrite_failures": 0,
        "range_failures": 0,
        "failure_examples": [],
        "unresolved_reference_count": 0,
        "status": "ok",
    }

    for idx, item in enumerate(items):
        try:
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {idx} is not object")
            sample_row = get_sample_row(item, assets, strict)
            new_retrieved = get_new_retrieved_passages(
                sample_row=sample_row,
                item_question=item.get("question", ""),
                assets=assets,
                strict=strict,
            )
            old_retrieved = parse_list_like(item.get("retrieved_passages"))
            if not old_retrieved:
                raise ValueError("Item has empty old retrieved_passages")

            mapping = build_old_to_new_map(old_retrieved, new_retrieved, strict)

            ideal_steps = item.get("ideal_steps", [])
            corrupted_steps = item.get("corrupted_steps", [])
            if not isinstance(ideal_steps, list) or not isinstance(corrupted_steps, list):
                raise ValueError("ideal_steps/corrupted_steps must both be lists")

            new_ideal, unresolved_ideal = rewrite_steps(ideal_steps, mapping, strict)
            new_corr, unresolved_corr = rewrite_steps(corrupted_steps, mapping, strict)

            unresolved_count = len(unresolved_ideal) + len(unresolved_corr)
            report["unresolved_reference_count"] += unresolved_count

            range_issue = check_passage_index_range(new_ideal, len(new_retrieved))
            if range_issue is None:
                range_issue = check_passage_index_range(new_corr, len(new_retrieved))
            if range_issue is not None:
                raise ValueError(
                    f"Passage index out of range after rewrite: {range_issue}"
                )

            new_item = dict(item)
            new_item["retrieved_passages"] = new_retrieved
            new_item["ideal_steps"] = new_ideal
            new_item["corrupted_steps"] = new_corr
            out_items.append(new_item)

        except Exception as e:
            msg = str(e)
            if "join" in msg or "sample" in msg or "benchmark_id" in msg:
                report["join_failures"] += 1
            elif "map" in msg or "Ambiguous" in msg:
                report["mapping_failures"] += 1
            elif "range" in msg:
                report["range_failures"] += 1
            else:
                report["rewrite_failures"] += 1

            if len(report["failure_examples"]) < 20:
                report["failure_examples"].append(
                    {
                        "item_index": idx,
                        "question": normalize_text(item.get("question", ""))[:200],
                        "error": msg,
                    }
                )
            if strict:
                report["status"] = "failed"
                report["item_count_out"] = len(out_items)
                if not dry_run:
                    write_json(report_file, report, overwrite)
                raise

    report["item_count_out"] = len(out_items)
    if (
        report["join_failures"]
        or report["mapping_failures"]
        or report["rewrite_failures"]
        or report["range_failures"]
    ):
        report["status"] = "partial_failed"

    if not dry_run:
        write_json(output_file, out_items, overwrite)
        write_json(report_file, report, overwrite)
    return report


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    input_error_root = input_root / "error_data"
    output_root = input_root / args.output_dir_name
    output_logs_root = output_root / "logs"

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "input_error_root": str(input_error_root),
        "output_root": str(output_root),
        "strict": args.strict,
        "dry_run": args.dry_run,
        "datasets": args.datasets,
        "error_types": args.error_types,
        "files": [],
        "totals": {
            "files": 0,
            "items_in": 0,
            "items_out": 0,
            "failed_files": 0,
        },
    }

    assets_cache: Dict[str, DatasetAssets] = {}
    for dataset in args.datasets:
        assets_cache[dataset] = load_assets(input_root, dataset)

    for error_type in args.error_types:
        for dataset in args.datasets:
            in_file = input_error_root / error_type / f"{dataset}_sampled_200_{error_type}.json"
            if not in_file.exists():
                entry = {
                    "dataset": dataset,
                    "error_type": error_type,
                    "input_file": str(in_file),
                    "status": "missing_input",
                }
                summary["files"].append(entry)
                summary["totals"]["failed_files"] += 1
                if args.strict:
                    raise FileNotFoundError(f"Missing input file: {in_file}")
                continue

            out_file = output_root / error_type / in_file.name
            report_file = output_logs_root / f"{dataset}_{error_type}_reindex_report.json"

            report = process_file(
                input_file=in_file,
                output_file=out_file,
                report_file=report_file,
                assets=assets_cache[dataset],
                strict=args.strict,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

            summary["files"].append(
                {
                    "dataset": dataset,
                    "error_type": error_type,
                    "input_file": str(in_file),
                    "output_file": str(out_file),
                    "report_file": str(report_file),
                    "status": report["status"],
                    "item_count_in": report["item_count_in"],
                    "item_count_out": report["item_count_out"],
                }
            )
            summary["totals"]["files"] += 1
            summary["totals"]["items_in"] += report["item_count_in"]
            summary["totals"]["items_out"] += report["item_count_out"]
            if report["status"] != "ok":
                summary["totals"]["failed_files"] += 1

            print(
                f"[{dataset}/{error_type}] in={report['item_count_in']} out={report['item_count_out']} "
                f"status={report['status']}"
            )

    summary_path = output_root / "summary.json"
    if not args.dry_run:
        write_json(summary_path, summary, args.overwrite)
        print(f"✅ Completed. Summary saved: {summary_path}")
    else:
        print("✅ Dry-run completed.")
        print(
            f"files={summary['totals']['files']} items_in={summary['totals']['items_in']} "
            f"items_out={summary['totals']['items_out']} failed_files={summary['totals']['failed_files']}"
        )


if __name__ == "__main__":
    main()
