import argparse
import ast
import csv
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from prompts import (
    ideal_reasoning_generation_2wiki,
    ideal_reasoning_generation_hotpotqa,
    ideal_reasoning_generation_musique,
    plan_generation_2wiki,
    plan_generation_hotpotqa,
    plan_generation_musique,
)

BASE_DIR = Path("/workspace/daeyong")
SUPPORTED_DATASETS = ("2wiki", "hotpotqa", "musique")
DATASET_OFFSETS = {name: idx for idx, name in enumerate(SUPPORTED_DATASETS)}

PLAN_PROMPTS = {
    "2wiki": plan_generation_2wiki,
    "hotpotqa": plan_generation_hotpotqa,
    "musique": plan_generation_musique,
}

IDEAL_PROMPTS = {
    "2wiki": ideal_reasoning_generation_2wiki,
    "hotpotqa": ideal_reasoning_generation_hotpotqa,
    "musique": ideal_reasoning_generation_musique,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-dataset false-triple samples and run plan/ideal-step generation "
            "with gpt-oss-120b in one pipeline."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=list(SUPPORTED_DATASETS),
        help="Datasets to process in order.",
    )
    parser.add_argument("--sample_size", type=int, default=200, help="Samples per dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for sampling.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/hf_transformers/gpt-oss-120b",
        help="Model path for vLLM.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/workspace/daeyong/filtering_noise_data",
        help="Root output directory.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="false_triples_oss120b_seed42",
        help="Run folder name under output_root.",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare false sets and sampled inputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing run folder and replacing files.",
    )
    parser.add_argument(
        "--plan_batch_size",
        type=int,
        default=500,
        help="Batch size for plan generation.",
    )
    parser.add_argument(
        "--ideal_batch_size",
        type=int,
        default=100,
        help="Batch size for ideal-step generation.",
    )
    parser.add_argument(
        "--plan_max_tokens",
        type=int,
        default=6000,
        help="Max tokens for plan generation.",
    )
    parser.add_argument(
        "--ideal_max_tokens",
        type=int,
        default=4000,
        help="Max tokens for ideal-step generation.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization for vLLM.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=10000,
        help="vLLM max model length.",
    )
    return parser.parse_args()


def normalize_question(question: str) -> str:
    if question is None:
        return ""
    return str(question).strip()


def write_json(path: Path, payload, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def chunked(items: Sequence[dict], batch_size: int) -> Iterable[Sequence[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def parse_passages(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(v).strip() for v in raw_value]

    text = str(raw_value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(v).strip() for v in parsed]
        return [str(parsed).strip()]
    except Exception:
        return [text]


def load_false_questions(dataset: str) -> Tuple[List[str], Path]:
    verified_path = BASE_DIR / "knowledge_graphs" / f"{dataset}_verified_triples_gleaned.json"
    with verified_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected list in {verified_path}, got {type(records).__name__}")

    false_questions = set()
    for item in records:
        if not isinstance(item, dict):
            continue
        if item.get("all_triples_supported") is False:
            question = normalize_question(item.get("question"))
            if question:
                false_questions.add(question)
    return sorted(false_questions), verified_path


def load_benchmark_rows_for_questions(
    dataset: str, target_questions: set
) -> Tuple[Dict[str, dict], set, int, List[str]]:
    csv_path = BASE_DIR / "benchmarks" / f"{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV missing: {csv_path}")

    csv.field_size_limit(sys.maxsize)
    matched_counts = defaultdict(int)
    question_to_row = {}
    total_rows = 0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        if "question" not in columns:
            raise ValueError(f"Missing 'question' column in {csv_path}")
        if "gt_passages" not in columns:
            raise ValueError(f"Missing 'gt_passages' column in {csv_path}")

        for row in reader:
            total_rows += 1
            question = normalize_question(row.get("question"))
            if not question or question not in target_questions:
                continue

            matched_counts[question] += 1
            if matched_counts[question] > 1:
                continue

            question_to_row[question] = {
                "id": row.get("id"),
                "question": question,
                "answer": row.get("answer", ""),
                "gt_passages": parse_passages(row.get("gt_passages")),
                "question_decomposition": row.get("question_decomposition", ""),
            }

    duplicate_questions = {q for q, count in matched_counts.items() if count > 1}
    matched_questions = set(matched_counts.keys())
    missing_questions = sorted(target_questions - matched_questions)
    return question_to_row, duplicate_questions, total_rows, missing_questions


def build_samples(
    dataset: str,
    false_questions: List[str],
    question_to_row: Dict[str, dict],
    duplicate_questions: set,
    sample_size: int,
    seed: int,
) -> Tuple[List[dict], dict]:
    false_set = set(false_questions)
    duplicate_in_false = sorted(duplicate_questions & false_set)
    eligible_questions = sorted(
        q for q in false_questions if q in question_to_row and q not in duplicate_questions
    )

    if len(eligible_questions) < sample_size:
        raise ValueError(
            f"{dataset}: not enough eligible questions after duplicate exclusion "
            f"(eligible={len(eligible_questions)}, requested={sample_size})"
        )

    rng = random.Random(seed + DATASET_OFFSETS[dataset])
    sampled_questions = rng.sample(eligible_questions, sample_size)

    sampled_inputs = []
    for idx, question in enumerate(sampled_questions, 1):
        row = question_to_row[question]
        sampled_inputs.append(
            {
                "dataset": dataset,
                "sample_index": idx,
                "benchmark_id": row.get("id"),
                "question": question,
                "gt_answer": row.get("answer", ""),
                "gt_passages": row.get("gt_passages", []),
                "question_decomposition": row.get("question_decomposition", ""),
            }
        )

    stats = {
        "false_total": len(false_questions),
        "eligible_total": len(eligible_questions),
        "duplicate_excluded_total": len(duplicate_in_false),
        "duplicate_excluded_questions": duplicate_in_false,
        "sampled_total": len(sampled_inputs),
    }
    return sampled_inputs, stats


def extract_step_list(text: str) -> str:
    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1].strip()

    matches = re.findall(r"\[.*?\]", text, re.DOTALL)
    if matches:
        return matches[-1]

    steps = re.findall(r"Step\s*\d+\s*:[^,\n]+", text)
    if steps:
        return "[" + ", ".join(s.strip() for s in steps) + "]"

    return text.strip()


def safe_parse_plan(plan_str: str) -> List[str]:
    plan_str = plan_str.strip()
    if plan_str.startswith("[") and plan_str.endswith("]"):
        inner = plan_str[1:-1].strip()
    else:
        inner = plan_str

    if not inner:
        return []

    steps_raw = re.split(r",?\s*(?=Step\s*\d+:)", inner)
    clean_steps = []
    for step in steps_raw:
        step = step.strip().strip("'").strip('"')
        if step and step.lower().startswith("step"):
            clean_steps.append(step)
    return clean_steps


def parse_plan_output(raw_text: str) -> Tuple[List[str], List[str]]:
    parsed_steps = safe_parse_plan(extract_step_list(raw_text))
    clean_plan = []
    invalid_steps = []
    for step in parsed_steps:
        normalized = step.strip().strip(",")
        if normalized.endswith("(Attribution)") or normalized.endswith("(Logical)"):
            clean_plan.append(normalized)
        else:
            invalid_steps.append(normalized)
    return clean_plan, invalid_steps


def extract_json_output(text):
    if not isinstance(text, str):
        return text

    text = text.strip()
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    if not match:
        match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)

    if match:
        content = match.group(1).strip()
    else:
        return text

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return content

    if isinstance(result, str):
        return extract_json_output(result)
    return result


def init_llm_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer
    from vllm import LLM

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    return llm, tokenizer


def run_plan_generation(
    dataset: str,
    samples: List[dict],
    llm,
    tokenizer,
    args: argparse.Namespace,
) -> Tuple[List[dict], List[dict]]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.plan_max_tokens)
    system_prompt = PLAN_PROMPTS[dataset]

    plan_results = []
    failures = []

    for batch in chunked(samples, args.plan_batch_size):
        prompts = []
        metadata = []

        for item in batch:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {item['question']}"},
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(full_prompt)
            metadata.append(item)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for item, output in zip(metadata, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            clean_plan, invalid_steps = parse_plan_output(raw_text)
            plan_results.append({"question": item["question"], "plan": clean_plan})

            if not clean_plan or invalid_steps:
                failures.append(
                    {
                        "question": item["question"],
                        "plan_length": len(clean_plan),
                        "invalid_steps": invalid_steps,
                        "raw_output": raw_text,
                    }
                )

    return plan_results, failures


def run_ideal_generation(
    dataset: str,
    samples: List[dict],
    plans: List[dict],
    llm,
    tokenizer,
    args: argparse.Namespace,
) -> Tuple[List[dict], List[dict], int]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.ideal_max_tokens)
    system_prompt = IDEAL_PROMPTS[dataset]

    plan_map = {item["question"]: item.get("plan", []) for item in plans}
    runnable_samples = []
    skipped_no_plan = 0
    for sample in samples:
        plan_steps = plan_map.get(sample["question"], [])
        if not plan_steps:
            skipped_no_plan += 1
            continue
        runnable_samples.append({**sample, "plan": plan_steps})

    ideal_results = []
    failures = []

    for batch in chunked(runnable_samples, args.ideal_batch_size):
        prompts = []
        metadata = []
        for item in batch:
            passages = item.get("gt_passages", [])
            formatted_passages = "\n".join(
                f"Passage {idx + 1}: {passage}" for idx, passage in enumerate(passages)
            )
            formatted_plan = "\n".join(item["plan"])
            user_content = (
                f"Question: {item['question']}\n\n"
                f"Ground Truth Context:\n{formatted_passages}\n\n"
                f"Reasoning Plan:\n{formatted_plan}"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(full_prompt)
            metadata.append(item)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for item, output in zip(metadata, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            parsed = extract_json_output(raw_text.split("assistantfinal")[-1].strip())
            ideal_results.append(
                {
                    "question": item["question"],
                    "plan": item["plan"],
                    "ideal_steps": parsed,
                }
            )

            if not isinstance(parsed, list):
                failures.append(
                    {
                        "question": item["question"],
                        "error": "ideal_steps_not_list",
                        "parsed_type": type(parsed).__name__,
                        "raw_output": raw_text,
                    }
                )

    return ideal_results, failures, skipped_no_plan


def ensure_run_dir(args: argparse.Namespace) -> Dict[str, Path]:
    run_dir = Path(args.output_root) / args.run_name
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run directory already exists. Use --overwrite to write into it: {run_dir}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_dir": run_dir,
        "samples_dir": run_dir / "samples",
        "plans_dir": run_dir / "plans",
        "ideal_dir": run_dir / "ideal_steps",
        "logs_dir": run_dir / "logs",
        "summary_path": run_dir / "summary.json",
    }
    for path in paths.values():
        if path.suffix:
            continue
        path.mkdir(parents=True, exist_ok=True)
    return paths


def unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def main():
    args = parse_args()
    args.datasets = unique_preserve_order(args.datasets)
    paths = ensure_run_dir(args)

    summary = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            "datasets": args.datasets,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "model_path": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "prepare_only": args.prepare_only,
            "overwrite": args.overwrite,
            "output_root": args.output_root,
            "plan_batch_size": args.plan_batch_size,
            "ideal_batch_size": args.ideal_batch_size,
            "plan_max_tokens": args.plan_max_tokens,
            "ideal_max_tokens": args.ideal_max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
        },
        "datasets": {},
        "totals": {
            "false_total": 0,
            "sampled_total": 0,
            "plan_generated_total": 0,
            "ideal_generated_total": 0,
            "plan_failure_total": 0,
            "ideal_failure_total": 0,
            "skipped_no_plan_total": 0,
        },
    }

    sampled_by_dataset = {}

    for dataset in args.datasets:
        print(f"\n=== Preparing dataset: {dataset} ===")
        false_questions, verified_path = load_false_questions(dataset)
        false_set = set(false_questions)
        (
            question_to_row,
            duplicate_questions,
            benchmark_row_count,
            missing_questions,
        ) = load_benchmark_rows_for_questions(dataset, false_set)

        sampled_inputs, sample_stats = build_samples(
            dataset=dataset,
            false_questions=false_questions,
            question_to_row=question_to_row,
            duplicate_questions=duplicate_questions,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        sampled_by_dataset[dataset] = sampled_inputs

        all_false_path = paths["samples_dir"] / f"{dataset}_all_false_questions.json"
        sampled_input_path = (
            paths["samples_dir"] / f"{dataset}_sampled_{args.sample_size}_input.json"
        )
        sampling_log_path = paths["logs_dir"] / f"{dataset}_sampling_log.json"

        write_json(
            all_false_path,
            {
                "dataset": dataset,
                "source_file": str(verified_path),
                "false_total": len(false_questions),
                "false_questions": false_questions,
            },
            args.overwrite,
        )
        write_json(sampled_input_path, sampled_inputs, args.overwrite)
        write_json(
            sampling_log_path,
            {
                "dataset": dataset,
                "benchmark_rows": benchmark_row_count,
                "matched_false_questions": len(question_to_row) + len(duplicate_questions),
                "missing_in_benchmark_total": len(missing_questions),
                "missing_in_benchmark_questions": missing_questions,
                "duplicate_questions_in_benchmark_total": len(duplicate_questions),
                "duplicate_questions_in_benchmark": sorted(duplicate_questions),
                **sample_stats,
            },
            args.overwrite,
        )

        summary["datasets"][dataset] = {
            "source_file": str(verified_path),
            "benchmark_file": str(BASE_DIR / "benchmarks" / f"{dataset}.csv"),
            "false_total": len(false_questions),
            "benchmark_rows": benchmark_row_count,
            "missing_in_benchmark_total": len(missing_questions),
            "duplicate_excluded_total": sample_stats["duplicate_excluded_total"],
            "eligible_total": sample_stats["eligible_total"],
            "sampled_total": sample_stats["sampled_total"],
            "plan_generated_total": 0,
            "plan_failure_total": 0,
            "ideal_generated_total": 0,
            "ideal_failure_total": 0,
            "skipped_no_plan_total": 0,
        }

        summary["totals"]["false_total"] += len(false_questions)
        summary["totals"]["sampled_total"] += sample_stats["sampled_total"]

        print(
            f"[{dataset}] false={len(false_questions)} eligible={sample_stats['eligible_total']} "
            f"sampled={sample_stats['sampled_total']}"
        )

    if args.prepare_only:
        write_json(paths["summary_path"], summary, args.overwrite)
        print("\n✅ Prepare-only completed.")
        print(f"📂 Outputs: {paths['run_dir']}")
        return

    print("\n=== Loading model for generation ===")
    llm, tokenizer = init_llm_and_tokenizer(args)

    for dataset in args.datasets:
        print(f"\n=== Plan generation: {dataset} ===")
        samples = sampled_by_dataset[dataset]
        plan_results, plan_failures = run_plan_generation(
            dataset=dataset,
            samples=samples,
            llm=llm,
            tokenizer=tokenizer,
            args=args,
        )

        plan_output_path = paths["plans_dir"] / f"{dataset}_sampled_{args.sample_size}_plan.json"
        plan_failure_path = paths["logs_dir"] / f"{dataset}_plan_failures.json"
        write_json(plan_output_path, plan_results, args.overwrite)
        write_json(plan_failure_path, plan_failures, args.overwrite)

        summary["datasets"][dataset]["plan_generated_total"] = len(plan_results)
        summary["datasets"][dataset]["plan_failure_total"] = len(plan_failures)
        summary["totals"]["plan_generated_total"] += len(plan_results)
        summary["totals"]["plan_failure_total"] += len(plan_failures)

        print(
            f"[{dataset}] plan generated={len(plan_results)} plan failures={len(plan_failures)}"
        )

        print(f"=== Ideal-step generation: {dataset} ===")
        ideal_results, ideal_failures, skipped_no_plan = run_ideal_generation(
            dataset=dataset,
            samples=samples,
            plans=plan_results,
            llm=llm,
            tokenizer=tokenizer,
            args=args,
        )

        ideal_output_path = (
            paths["ideal_dir"] / f"{dataset}_sampled_{args.sample_size}_ideal_steps.json"
        )
        ideal_failure_path = paths["logs_dir"] / f"{dataset}_ideal_failures.json"
        write_json(ideal_output_path, ideal_results, args.overwrite)
        write_json(ideal_failure_path, ideal_failures, args.overwrite)

        summary["datasets"][dataset]["ideal_generated_total"] = len(ideal_results)
        summary["datasets"][dataset]["ideal_failure_total"] = len(ideal_failures)
        summary["datasets"][dataset]["skipped_no_plan_total"] = skipped_no_plan
        summary["totals"]["ideal_generated_total"] += len(ideal_results)
        summary["totals"]["ideal_failure_total"] += len(ideal_failures)
        summary["totals"]["skipped_no_plan_total"] += skipped_no_plan

        print(
            f"[{dataset}] ideal generated={len(ideal_results)} ideal failures={len(ideal_failures)} "
            f"skipped_no_plan={skipped_no_plan}"
        )

    write_json(paths["summary_path"], summary, args.overwrite)
    print("\n🎉 Pipeline completed.")
    print(f"📂 Outputs: {paths['run_dir']}")


if __name__ == "__main__":
    main()
