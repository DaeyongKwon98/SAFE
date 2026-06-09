#!/usr/bin/env python3
"""Build a deduplicated passage corpus from benchmark CSV retrieved_passages."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


DEFAULT_CSV_PATHS = [
    "benchmarks/2wiki.csv",
    "benchmarks/hotpotqa.csv",
    "benchmarks/musique.csv",
    "benchmarks/2wiki_dev.csv",
    "benchmarks/hotpotqa_dev.csv",
    "benchmarks/musique_dev.csv",
]
DEFAULT_OUTPUT_PATH = "benchmarks/retrieved_passages_corpus.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse retrieved_passages from CSV files, deduplicate them as a set, "
            "and save a BM25-ready JSON corpus."
        )
    )
    parser.add_argument("--csv_paths", nargs="+", default=DEFAULT_CSV_PATHS)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--passage_field", default="retrieved_passages")
    parser.add_argument(
        "--dedupe_mode",
        choices=("exact", "whitespace"),
        default="whitespace",
        help="exact keeps passage strings as-is; whitespace collapses whitespace before dedupe.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort passage text before writing. Default preserves first-seen order.",
    )
    return parser.parse_args()


def normalize_passage(text: Any, dedupe_mode: str) -> str:
    value = str(text)
    if dedupe_mode == "whitespace":
        value = re.sub(r"\s+", " ", value)
    return value.strip()


def dataset_from_path(path: Path) -> str:
    name = path.name
    for suffix in ("_dev.csv", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    seen: set[str] = set()
    passages: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {}
    total_entries = 0
    total_rows = 0

    for csv_path_str in args.csv_paths:
        csv_path = Path(csv_path_str)
        dataset = dataset_from_path(csv_path)
        file_stats = {
            "rows": 0,
            "passage_entries": 0,
            "new_unique_passages": 0,
            "parse_errors": 0,
        }
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if args.passage_field not in (reader.fieldnames or []):
                raise ValueError(f"{csv_path} has no field named {args.passage_field!r}")
            for row_idx, row in enumerate(tqdm(reader, desc=str(csv_path))):
                total_rows += 1
                file_stats["rows"] += 1
                raw = (row.get(args.passage_field) or "").strip()
                try:
                    values = ast.literal_eval(raw)
                except Exception:
                    file_stats["parse_errors"] += 1
                    continue
                if not isinstance(values, list):
                    file_stats["parse_errors"] += 1
                    continue
                for passage in values:
                    text = normalize_passage(passage, args.dedupe_mode)
                    if not text:
                        continue
                    total_entries += 1
                    file_stats["passage_entries"] += 1
                    if text in seen:
                        continue
                    seen.add(text)
                    file_stats["new_unique_passages"] += 1
                    passages.append(
                        {
                            "passage_index": len(passages),
                            "passage_text": text,
                            "source_dataset": dataset,
                            "source_file": str(csv_path),
                            "source_row": row_idx,
                        }
                    )
        stats[str(csv_path)] = file_stats

    if args.sort:
        passages.sort(key=lambda item: item["passage_text"])
        for idx, item in enumerate(passages):
            item["passage_index"] = idx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summary = {
        "csv_paths": args.csv_paths,
        "output_path": str(output_path),
        "dedupe_mode": args.dedupe_mode,
        "total_rows": total_rows,
        "total_passage_entries": total_entries,
        "unique_passages": len(passages),
        "files": stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
