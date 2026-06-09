#!/usr/bin/env python3
"""Evaluate BM25 recall for missing-evidence search-query guidance.

The input records are expected to contain a search query in ``guidance`` and
the target evidence passage in ``removed_passage``. The retrieval database is
the concatenation of benchmarks/{dataset}_corpus.json for the selected
datasets.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


DEFAULT_INPUT_PATH = "fourth_finetuning_data/only_correct_missing_evidence_queries.json"
DEFAULT_OUTPUT_PATH = "fourth_finetuning_data/missing_evidence_bm25_recall_results.json"
DEFAULT_DATASETS = ("2wiki", "hotpotqa", "musique")
DEFAULT_K_VALUES = (1, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class PassageDoc:
    global_index: int
    dataset: str
    local_row_index: int
    passage_index: Any
    text: str


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use each record's guidance as a BM25 search query, retrieve topK "
            "from merged benchmark corpora, and compute Recall@K against "
            "removed_passage."
        )
    )
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--corpus_dir", default="benchmarks")
    parser.add_argument(
        "--corpus_json_path",
        default=None,
        help=(
            "Optional BM25 corpus JSON file with passage_text fields. If set, "
            "--datasets/--corpus_dir are ignored."
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--query_field", default="guidance")
    parser.add_argument("--gold_field", default="removed_passage")
    parser.add_argument(
        "--k_values",
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated K values, e.g. 1,5,10,20,50,100.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "bm25s", "python"),
        default="auto",
        help="BM25 backend. auto uses bm25s if available, otherwise pure Python.",
    )
    parser.add_argument(
        "--match_mode",
        choices=("exact", "whitespace", "aggressive", "title_containment"),
        default="aggressive",
        help=(
            "How to match removed_passage to corpus passages. aggressive "
            "case-folds and removes punctuation; title_containment additionally "
            "allows same-title containment matches."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dedupe_passages",
        action="store_true",
        help="Deduplicate merged corpus passages by aggressive normalized text.",
    )
    parser.add_argument(
        "--no_stemmer",
        action="store_true",
        help="Disable the English stemmer when using the bm25s backend.",
    )
    parser.add_argument(
        "--no_corpus_stopwords",
        action="store_true",
        help="Do not remove English stopwords from corpus tokens in bm25s.",
    )
    parser.add_argument(
        "--save_top_passages",
        action="store_true",
        help="Store retrieved passage text in the output JSON. This can be large.",
    )
    parser.add_argument(
        "--save_gold_passages",
        action="store_true",
        help="Store matched gold passage text in the output JSON.",
    )
    return parser.parse_args()


def parse_k_values(value: str) -> list[int]:
    output = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        output.append(k)
    if not output:
        raise ValueError("At least one K value is required")
    return sorted(set(output))


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at row {idx}, got {type(item).__name__}")
    return data


def normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_aggressive(text: Any) -> str:
    value = unicodedata.normalize("NFKD", str(text))
    value = value.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def get_normalizer(match_mode: str) -> Callable[[Any], str]:
    if match_mode == "exact":
        return lambda text: str(text)
    if match_mode == "whitespace":
        return normalize_whitespace
    return normalize_aggressive


def passage_title(text: str) -> str:
    return text.split(":", 1)[0].strip() if ":" in text else ""


def load_corpus(
    corpus_dir: Path,
    datasets: Iterable[str],
    dedupe_passages: bool,
) -> list[PassageDoc]:
    docs: list[PassageDoc] = []
    seen: set[str] = set()
    for dataset in datasets:
        path = corpus_dir / f"{dataset}_corpus.json"
        rows = load_json_list(path)
        for local_row_index, row in enumerate(rows):
            text = str(row.get("passage_text", ""))
            if not text.strip():
                continue
            if dedupe_passages:
                dedupe_key = normalize_aggressive(text)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
            docs.append(
                PassageDoc(
                    global_index=len(docs),
                    dataset=dataset,
                    local_row_index=local_row_index,
                    passage_index=row.get("passage_index", local_row_index),
                    text=text,
                )
            )
    if not docs:
        raise ValueError("No corpus passages loaded")
    return docs


def load_corpus_json(path: Path, dedupe_passages: bool) -> list[PassageDoc]:
    rows = load_json_list(path)
    docs: list[PassageDoc] = []
    seen: set[str] = set()
    for local_row_index, row in enumerate(rows):
        text = str(row.get("passage_text", ""))
        if not text.strip():
            continue
        if dedupe_passages:
            dedupe_key = normalize_aggressive(text)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
        docs.append(
            PassageDoc(
                global_index=len(docs),
                dataset=str(row.get("source_dataset") or row.get("dataset") or path.stem),
                local_row_index=local_row_index,
                passage_index=row.get("passage_index", local_row_index),
                text=text,
            )
        )
    if not docs:
        raise ValueError(f"No corpus passages loaded from {path}")
    return docs


def build_gold_lookup(
    docs: list[PassageDoc],
    normalizer: Callable[[Any], str],
    match_mode: str,
) -> tuple[dict[str, list[int]], dict[str, list[tuple[int, str]]]]:
    exact_lookup: dict[str, list[int]] = defaultdict(list)
    title_lookup: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for doc in docs:
        normalized_text = normalizer(doc.text)
        if normalized_text:
            exact_lookup[normalized_text].append(doc.global_index)
        if match_mode == "title_containment":
            title_key = normalizer(passage_title(doc.text))
            if title_key:
                title_lookup[title_key].append((doc.global_index, normalized_text))
    return exact_lookup, title_lookup


def find_gold_doc_ids(
    removed_passage: Any,
    normalizer: Callable[[Any], str],
    exact_lookup: dict[str, list[int]],
    title_lookup: dict[str, list[tuple[int, str]]],
    match_mode: str,
) -> list[int]:
    removed_text = str(removed_passage or "")
    removed_key = normalizer(removed_text)
    if not removed_key:
        return []
    exact_matches = exact_lookup.get(removed_key, [])
    if exact_matches or match_mode != "title_containment":
        return list(exact_matches)

    title_key = normalizer(passage_title(removed_text))
    if not title_key:
        return []
    matches = []
    for doc_id, doc_key in title_lookup.get(title_key, []):
        if removed_key in doc_key or doc_key in removed_key:
            matches.append(doc_id)
    return matches


class BM25sBackend:
    def __init__(self, docs: list[PassageDoc], args: argparse.Namespace):
        import bm25s

        self.bm25s = bm25s
        self.stemmer = None
        if not args.no_stemmer:
            try:
                import Stemmer

                self.stemmer = Stemmer.Stemmer("english")
            except ImportError:
                self.stemmer = None

        corpus_texts = [doc.text for doc in docs]
        stopwords = None if args.no_corpus_stopwords else "en"
        print(f"Tokenizing {len(corpus_texts)} corpus passages with bm25s...")
        corpus_tokens = bm25s.tokenize(
            corpus_texts,
            stopwords=stopwords,
            stemmer=self.stemmer,
        )
        self.retriever = bm25s.BM25()
        print("Indexing corpus with bm25s...")
        self.retriever.index(corpus_tokens)

    def retrieve_batch(self, queries: list[str], k: int) -> list[tuple[list[int], list[float]]]:
        active_positions = []
        active_queries = []
        output: list[tuple[list[int], list[float]]] = [([], []) for _ in queries]
        for pos, query in enumerate(queries):
            if str(query).strip():
                active_positions.append(pos)
                active_queries.append(str(query))
        if not active_queries:
            return output

        query_tokens = self.bm25s.tokenize(active_queries, stemmer=self.stemmer)
        result_ids, scores = self.retriever.retrieve(query_tokens, k=k)
        result_ids = result_ids.tolist()
        scores = scores.tolist()
        for pos, ids, row_scores in zip(active_positions, result_ids, scores):
            output[pos] = ([int(doc_id) for doc_id in ids], [float(score) for score in row_scores])
        return output


class PythonBM25Backend:
    def __init__(self, docs: list[PassageDoc], args: argparse.Namespace):
        del args
        self.k1 = 1.5
        self.b = 0.75
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        print(f"Building pure-Python BM25 index for {len(docs)} corpus passages...")
        for doc in tqdm(docs, desc="Index corpus"):
            tokens = self.tokenize(doc.text)
            self.doc_lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((doc.global_index, tf))
        self.num_docs = len(docs)
        self.avgdl = sum(self.doc_lengths) / max(1, self.num_docs)
        self.idf = {
            term: math.log(1.0 + (self.num_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]

    def retrieve_batch(self, queries: list[str], k: int) -> list[tuple[list[int], list[float]]]:
        return [self.retrieve_one(str(query), k) for query in queries]

    def retrieve_one(self, query: str, k: int) -> tuple[list[int], list[float]]:
        query_terms = set(self.tokenize(query))
        if not query_terms:
            return [], []
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for doc_id, tf in posting:
                doc_len = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / denom
        if not scores:
            return [], []
        top = heapq.nlargest(k, scores.items(), key=lambda item: (item[1], -item[0]))
        return [doc_id for doc_id, _ in top], [score for _, score in top]


def create_backend(docs: list[PassageDoc], args: argparse.Namespace):
    if args.backend in ("auto", "bm25s"):
        try:
            return "bm25s", BM25sBackend(docs, args)
        except ImportError:
            if args.backend == "bm25s":
                raise
            print("bm25s is not installed; falling back to pure-Python BM25.")
    return "python", PythonBM25Backend(docs, args)


def chunked(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def doc_metadata(doc: PassageDoc, include_text: bool) -> dict[str, Any]:
    value = {
        "doc_id": doc.global_index,
        "dataset": doc.dataset,
        "local_row_index": doc.local_row_index,
        "passage_index": doc.passage_index,
    }
    if include_text:
        value["passage_text"] = doc.text
    return value


def safe_float(value: float) -> float | None:
    if math.isfinite(value):
        return value
    return None


def evaluate() -> dict[str, Any]:
    args = parse_args()
    start_time = time.time()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    corpus_dir = Path(args.corpus_dir)
    k_values = parse_k_values(args.k_values)

    records = load_json_list(input_path)
    selected_indices = list(range(len(records)))
    selected_indices = selected_indices[args.start_index : args.end_index]
    if args.limit is not None:
        selected_indices = selected_indices[: args.limit]
    selected_records = [records[idx] for idx in selected_indices]
    if not selected_records:
        raise ValueError("No input records selected")

    if args.corpus_json_path:
        corpus_json_path = Path(args.corpus_json_path)
        docs = load_corpus_json(corpus_json_path, args.dedupe_passages)
        corpus_description = str(corpus_json_path)
    else:
        docs = load_corpus(corpus_dir, args.datasets, args.dedupe_passages)
        corpus_description = str(corpus_dir)
    max_k = min(max(k_values), len(docs))
    normalizer = get_normalizer(args.match_mode)
    exact_lookup, title_lookup = build_gold_lookup(docs, normalizer, args.match_mode)
    gold_doc_ids_by_row = [
        find_gold_doc_ids(
            record.get(args.gold_field, ""),
            normalizer,
            exact_lookup,
            title_lookup,
            args.match_mode,
        )
        for record in selected_records
    ]

    backend_name, backend = create_backend(docs, args)

    results = []
    batches = list(chunked(list(enumerate(selected_records)), args.batch_size))
    for batch in tqdm(batches, desc="Retrieve"):
        batch_queries = [str(record.get(args.query_field, "") or "") for _, record in batch]
        retrieved_batch = backend.retrieve_batch(batch_queries, max_k)
        for (row_pos, record), (top_doc_ids, scores) in zip(batch, retrieved_batch):
            original_index = selected_indices[row_pos]
            gold_doc_ids = gold_doc_ids_by_row[row_pos]
            gold_doc_id_set = set(gold_doc_ids)
            rank = None
            for retrieved_rank, doc_id in enumerate(top_doc_ids, start=1):
                if doc_id in gold_doc_id_set:
                    rank = retrieved_rank
                    break
            hits = {str(k): bool(rank is not None and rank <= k) for k in k_values}
            item = {
                "index": original_index,
                "question": record.get("question", ""),
                "query": record.get(args.query_field, ""),
                "removed_passage": record.get(args.gold_field, ""),
                "gold_found_in_corpus": bool(gold_doc_ids),
                "gold_doc_ids": gold_doc_ids,
                "rank": rank,
                "hits": hits,
                "top_doc_ids": top_doc_ids,
                "top_scores": [safe_float(score) for score in scores],
            }
            if args.save_gold_passages:
                item["gold_passages"] = [
                    doc_metadata(docs[doc_id], include_text=True) for doc_id in gold_doc_ids
                ]
            if args.save_top_passages:
                item["top_passages"] = [
                    doc_metadata(docs[doc_id], include_text=True) for doc_id in top_doc_ids
                ]
            results.append(item)

    total = len(results)
    covered = sum(1 for item in results if item["gold_found_in_corpus"])
    nonempty_queries = sum(1 for item in results if str(item["query"]).strip())
    recall_at_k = {}
    recall_at_k_covered = {}
    hit_counts = {}
    covered_hit_counts = {}
    for k in k_values:
        key = str(k)
        hits = sum(1 for item in results if item["hits"][key])
        covered_hits = sum(
            1
            for item in results
            if item["gold_found_in_corpus"] and item["hits"][key]
        )
        hit_counts[key] = hits
        covered_hit_counts[key] = covered_hits
        recall_at_k[key] = hits / total if total else 0.0
        recall_at_k_covered[key] = covered_hits / covered if covered else 0.0

    summary = {
        "input_path": str(input_path),
        "corpus_dir": str(corpus_dir),
        "corpus_json_path": args.corpus_json_path,
        "corpus": corpus_description,
        "datasets": [] if args.corpus_json_path else args.datasets,
        "num_records": total,
        "num_corpus_docs": len(docs),
        "backend": backend_name,
        "match_mode": args.match_mode,
        "k_values": k_values,
        "query_field": args.query_field,
        "gold_field": args.gold_field,
        "nonempty_queries": nonempty_queries,
        "gold_found_in_corpus": covered,
        "gold_missing_from_corpus": total - covered,
        "gold_coverage": covered / total if total else 0.0,
        "hit_counts": hit_counts,
        "covered_hit_counts": covered_hit_counts,
        "recall_at_k": recall_at_k,
        "recall_at_k_covered_gold_only": recall_at_k_covered,
        "elapsed_seconds": time.time() - start_time,
    }

    output = {"summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print()
    print(f"Input records: {total}")
    print(f"Corpus docs: {len(docs)}")
    print(f"Gold found in corpus: {covered}/{total} ({summary['gold_coverage']:.4f})")
    print()
    print("K\tHits(all)\tRecall(all)\tHits(covered)\tRecall(covered)")
    for k in k_values:
        key = str(k)
        print(
            f"{k}\t{hit_counts[key]}\t{recall_at_k[key]:.4f}\t"
            f"{covered_hit_counts[key]}\t{recall_at_k_covered[key]:.4f}"
        )
    print()
    print(f"Wrote: {output_path}")
    return output


if __name__ == "__main__":
    evaluate()
