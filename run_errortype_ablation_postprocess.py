#!/usr/bin/env python3
import argparse
import ast
import csv
import gc
import os
import re
import string
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

DEFAULT_ROOT_DIR = (
    "/workspace/daeyong/inference_results/"
    "dev_kg_correct_1ksample_with_noises_10_3_errortype_ablation"
)
DATASET_CHOICES = ("2wiki", "hotpotqa", "musique")
EXCLUDE_SUFFIXES = (
    "_logs.json",
    "_stats.json",
    "_cache_stats.json",
    "_final_answer.json",
    "_llm_judge.json",
)
INFERENCE_FILE_RE = re.compile(r"^(?P<model>.+)_(?P<dataset>2wiki|hotpotqa|musique)\.json$")


@dataclass(frozen=True)
class InferenceTask:
    drop_dir: Path
    model: str
    dataset: str
    input_path: Path

    @property
    def final_answer_path(self) -> Path:
        return self.input_path.with_name(f"{self.model}_{self.dataset}_final_answer.json")

    @property
    def judge_path(self) -> Path:
        return self.input_path.with_name(f"{self.model}_{self.dataset}_llm_judge.json")


@dataclass
class StageStats:
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def parse_cli_tokens(values: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for item in values:
        for token in str(item).split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    # Deduplicate while preserving order.
    return list(dict.fromkeys(tokens))


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).strip()


def parse_maybe_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return [str(parsed)]
        except Exception:
            return [raw]
    if value is None:
        return []
    return [str(value)]


def load_musique_aliases(musique_csv_path: Path) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    if not musique_csv_path.exists():
        print(f"[WARN] musique alias csv not found: {musique_csv_path}")
        return aliases

    with musique_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("id", "")).strip()
            if not sample_id:
                continue

            answer_candidates = parse_maybe_list(row.get("answer_list", ""))
            normalized = [normalize_text(x) for x in answer_candidates if normalize_text(x)]
            if normalized:
                aliases[sample_id] = normalized

    return aliases


def resolve_drop_dirs(root_dir: Path, drop_dirs: Optional[Sequence[str]]) -> List[Path]:
    if drop_dirs:
        resolved: List[Path] = []
        for token in parse_cli_tokens(drop_dirs):
            path = Path(token)
            if not path.is_absolute():
                path = root_dir / token
            path = path.resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"Drop directory not found: {path}")
            resolved.append(path)
        return list(dict.fromkeys(resolved))

    auto_dirs = sorted(
        [p.resolve() for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("drop_")],
        key=lambda p: p.name,
    )
    if not auto_dirs:
        raise FileNotFoundError(f"No drop_* directories found under: {root_dir}")
    return auto_dirs


def parse_inference_file_name(file_name: str, allowed_datasets: Sequence[str]) -> Optional[Tuple[str, str]]:
    if not file_name.endswith(".json"):
        return None
    if any(file_name.endswith(suffix) for suffix in EXCLUDE_SUFFIXES):
        return None

    match = INFERENCE_FILE_RE.match(file_name)
    if not match:
        return None

    model = match.group("model")
    dataset = match.group("dataset")
    if dataset not in set(allowed_datasets):
        return None
    return model, dataset


def discover_inference_tasks(drop_dirs: Sequence[Path], datasets: Sequence[str]) -> List[InferenceTask]:
    tasks: List[InferenceTask] = []
    for drop_dir in drop_dirs:
        for file_path in sorted(drop_dir.glob("*.json")):
            parsed = parse_inference_file_name(file_path.name, datasets)
            if not parsed:
                continue
            model, dataset = parsed
            tasks.append(
                InferenceTask(
                    drop_dir=drop_dir,
                    model=model,
                    dataset=dataset,
                    input_path=file_path.resolve(),
                )
            )

    tasks.sort(key=lambda x: (x.drop_dir.name, x.model, x.dataset))
    return tasks


def run_final_answer_stage(tasks: Sequence[InferenceTask], overwrite: bool, max_model_len: int) -> StageStats:
    from final_answer_self_feedback import (
        MODEL_MAPPING as FINAL_ANSWER_MODEL_MAPPING,
        load_vllm_model,
        run_answer_generation_for_dataset,
    )

    fallback_mapping = {
        "llama8b": "/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct",
        "gemma12b": "/workspace/hf_transformers/gemma-3-12b-it",
        "qwen4b": "/workspace/hf_transformers/Qwen3-4B-Instruct-2507",
        "qwen8b": "/workspace/hf_transformers/Qwen3-8B",
        "qwen14b": (
            "/workspace/hf_transformers/models--Qwen--Qwen2.5-14B-Instruct/"
            "snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
        ),
    }
    model_mapping = dict(fallback_mapping)
    model_mapping.update(FINAL_ANSWER_MODEL_MAPPING)

    stats = StageStats()
    model_groups: Dict[str, List[InferenceTask]] = defaultdict(list)
    for task in tasks:
        model_groups[task.model].append(task)

    for model_short in sorted(model_groups.keys()):
        model_tasks = sorted(model_groups[model_short], key=lambda x: (x.drop_dir.name, x.dataset))
        model_path = model_mapping.get(model_short, "")

        if not model_path:
            print(f"[FINAL][FAIL] Unsupported model prefix '{model_short}'.")
            stats.failed += len(model_tasks)
            continue
        if not os.path.exists(model_path):
            print(f"[FINAL][FAIL] Model path missing for '{model_short}': {model_path}")
            stats.failed += len(model_tasks)
            continue

        print(f"\n[FINAL] Loading generator model: {model_short} -> {model_path}")
        try:
            llm, tokenizer = load_vllm_model(
                model_path=model_path,
                max_model_len=max_model_len,
            )
        except Exception as e:
            print(f"[FINAL][FAIL] Failed to load model '{model_short}': {e}")
            stats.failed += len(model_tasks)
            continue

        disable_thinking = "qwen3-8b" in model_path.lower()

        for task in model_tasks:
            output_path = task.final_answer_path
            if output_path.exists() and not overwrite:
                print(f"[FINAL][SKIP] {output_path}")
                stats.skipped += 1
                continue

            if output_path.exists() and overwrite:
                output_path.unlink()

            print(f"[FINAL][RUN] input={task.input_path} output={output_path}")
            try:
                df = pd.read_json(task.input_path)
            except Exception as e:
                print(f"[FINAL][FAIL] Failed to read input: {task.input_path} ({e})")
                stats.failed += 1
                continue

            try:
                run_answer_generation_for_dataset(
                    df=df,
                    dataset=task.dataset,
                    source_file_name=task.input_path.name,
                    llm=llm,
                    tokenizer=tokenizer,
                    result_file_path=str(output_path),
                    disable_thinking=disable_thinking,
                )
            except Exception as e:
                print(f"[FINAL][FAIL] Processing failed: {task.input_path} ({e})")
                stats.failed += 1
                continue

            if not output_path.exists():
                print(f"[FINAL][FAIL] Output missing after processing: {output_path}")
                stats.failed += 1
                continue

            stats.processed += 1

        del llm
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return stats


def get_musique_ground_truth_list(row: pd.Series, aliases_by_id: Dict[str, List[str]]) -> List[str]:
    sample_id = str(row.get("id", "")).strip()
    if sample_id and sample_id in aliases_by_id:
        return aliases_by_id[sample_id]

    answer_list = parse_maybe_list(row.get("answer_list_norm", ""))
    normalized = [normalize_text(x) for x in answer_list if normalize_text(x)]
    if normalized:
        return normalized

    gt = normalize_text(str(row.get("ground_truth", "")))
    if gt:
        return [gt]
    return [""]


def run_judge_stage(tasks: Sequence[InferenceTask], overwrite: bool, musique_aliases: Dict[str, List[str]]) -> StageStats:
    from oss_answer_binary_self_feedback import (
        get_generated_answer,
        load_judge_model,
        parse_llm_output,
        system_prompt,
    )

    stats = StageStats()
    pending_tasks: List[InferenceTask] = []

    for task in tasks:
        output_path = task.judge_path
        if output_path.exists() and not overwrite:
            print(f"[JUDGE][SKIP] {output_path}")
            stats.skipped += 1
            continue

        final_input = task.final_answer_path
        if not final_input.exists():
            print(f"[JUDGE][FAIL] Missing final answer input: {final_input}")
            stats.failed += 1
            continue
        pending_tasks.append(task)

    if not pending_tasks:
        return stats

    print(f"\n[JUDGE] Loading judge model once (gpt-oss-120b) for {len(pending_tasks)} files...")
    try:
        llm, tokenizer, sampling_params = load_judge_model()
    except Exception as e:
        print(f"[JUDGE][FAIL] Failed to load judge model: {e}")
        stats.failed += len(pending_tasks)
        return stats

    for task in pending_tasks:
        final_input = task.final_answer_path
        output_path = task.judge_path

        if output_path.exists() and overwrite:
            output_path.unlink()

        print(f"[JUDGE][RUN] input={final_input} output={output_path}")
        try:
            df = pd.read_json(final_input)
        except Exception as e:
            print(f"[JUDGE][FAIL] Failed to read final answer file: {final_input} ({e})")
            stats.failed += 1
            continue

        prompts: List[str] = []
        generated_answers: List[str] = []
        empty_flags: List[bool] = []

        for _, row in df.iterrows():
            if task.dataset == "musique":
                gt_list = get_musique_ground_truth_list(row, aliases_by_id=musique_aliases)
            else:
                gt_list = [row.get("ground_truth", "")]

            generated_answer = get_generated_answer(row)
            generated_answers.append(generated_answer)
            empty_flags.append(generated_answer == "")

            user_content = (
                f"### Input Data\n"
                f"**Question**: {row.get('question', '')}\n"
                f"**Ground Truth List**: {gt_list}\n"
                f"**Generated Answer**: {generated_answer}\n\n"
                f"### Task\n"
                f"Is the generated answer correct based on the ground truth list?"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        if prompts:
            try:
                outputs = llm.generate(prompts, sampling_params)
            except Exception as e:
                print(f"[JUDGE][FAIL] Generation failed: {final_input} ({e})")
                stats.failed += 1
                continue

            is_correct_list: List[str] = []
            reasoning_list: List[str] = []

            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text.split("assistantfinal")[-1].strip()
                is_correct, reasoning = parse_llm_output(generated_text)

                if empty_flags[i]:
                    reasoning = f"[empty_generated_answer] {reasoning}"

                is_correct_list.append(is_correct)
                reasoning_list.append(reasoning)
        else:
            is_correct_list = []
            reasoning_list = []

        df["generated_answer_used"] = generated_answers
        df["is_correct"] = is_correct_list
        df["reasoning"] = reasoning_list

        try:
            df.to_json(output_path, orient="records", force_ascii=False, indent=2)
        except Exception as e:
            print(f"[JUDGE][FAIL] Failed to write output: {output_path} ({e})")
            stats.failed += 1
            continue

        stats.processed += 1

    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run errortype-ablation post-processing over drop_* folders: "
            "final answer generation and/or gpt-oss-120b judge."
        )
    )
    parser.add_argument("--root-dir", type=str, default=DEFAULT_ROOT_DIR)
    parser.add_argument(
        "--drop-dirs",
        nargs="+",
        default=None,
        help="Optional subset of drop directories (name or absolute path).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CHOICES),
        help=f"Datasets to include. Supported: {', '.join(DATASET_CHOICES)}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_final_answer.json and *_llm_judge.json files.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "final", "judge"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=3000,
        help="max_model_len passed to final answer generation model loader.",
    )
    return parser


def print_summary(stage_name: str, stats: StageStats) -> None:
    print(
        f"[SUMMARY][{stage_name}] "
        f"processed={stats.processed} skipped={stats.skipped} failed={stats.failed}"
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    datasets = parse_cli_tokens(args.datasets)
    invalid_datasets = [d for d in datasets if d not in DATASET_CHOICES]
    if invalid_datasets:
        raise ValueError(f"Unsupported dataset(s): {invalid_datasets}. Supported: {DATASET_CHOICES}")
    if not datasets:
        raise ValueError("No datasets selected.")

    drop_dirs = resolve_drop_dirs(root_dir, args.drop_dirs)
    tasks = discover_inference_tasks(drop_dirs=drop_dirs, datasets=datasets)

    print("===========================================================")
    print("Errortype Ablation Postprocess")
    print(f"root_dir={root_dir}")
    print(f"stage={args.stage} overwrite={args.overwrite} datasets={datasets}")
    print(f"drop_dirs={len(drop_dirs)}")
    print(f"discovered_inference_files={len(tasks)}")
    print("===========================================================")

    if not tasks:
        print("No target inference files found. Nothing to process.")
        return 0

    final_stats = StageStats()
    judge_stats = StageStats()

    if args.stage in {"all", "final"}:
        final_stats = run_final_answer_stage(
            tasks=tasks,
            overwrite=args.overwrite,
            max_model_len=args.max_model_len,
        )

    if args.stage in {"all", "judge"}:
        musique_csv = Path(__file__).resolve().parent / "benchmarks" / "musique_dev.csv"
        musique_aliases = load_musique_aliases(musique_csv)
        judge_stats = run_judge_stage(
            tasks=tasks,
            overwrite=args.overwrite,
            musique_aliases=musique_aliases,
        )

    print("\n============================= Final Summary =============================")
    if args.stage in {"all", "final"}:
        print_summary("final", final_stats)
    if args.stage in {"all", "judge"}:
        print_summary("judge", judge_stats)

    total_failed = 0
    if args.stage in {"all", "final"}:
        total_failed += final_stats.failed
    if args.stage in {"all", "judge"}:
        total_failed += judge_stats.failed

    if total_failed > 0:
        print(f"[RESULT] Completed with failures: {total_failed}")
        return 1

    print("[RESULT] Completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
