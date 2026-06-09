import argparse
import ast
import json
import os
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompts import generate_single_step_fixed_system_prompt


CLEAN_MARKERS = ["<start_of_turn>", "User:", "## Question"]
PROMPT_LEAK_MARKERS = ["Step to evaluate", "Question:", "error_type", "JSON", "###"]
HIGH_REPETITION_RUN_THRESHOLD = 6


def load_vllm_model(model_id: str, gpu_memory_utilization: float = 0.90):
    print(f"Loading Single Model (vLLM)... Model: '{model_id}'")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_id,
        tensor_parallel_size=4,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=10000,
        enforce_eager=False,
        enable_prefix_caching=True,
    )

    print("✅ Model loaded successfully.")
    return llm, tokenizer


def normalize_step_text(gen_text: str, expected_step_num: int = 1) -> str:
    expected_start = f"Step {expected_step_num}:"
    text = str(gen_text).strip()

    if not text:
        return expected_start

    first_line, sep, remaining = text.partition("\n")
    first_line = first_line.strip()

    if first_line.startswith(expected_start):
        return text

    strict_step_match = re.match(r"^Step\s+(\d+)\s*:\s*(.*)$", first_line)
    if strict_step_match:
        suffix = strict_step_match.group(2).strip()
        normalized_first = expected_start if not suffix else f"{expected_start} {suffix}"
        return normalized_first + (f"\n{remaining}" if sep else "")

    if re.match(r"(?i)^step", first_line):
        tail = re.sub(r"(?i)^step", "", first_line, count=1).strip()
        if ":" in tail:
            tail = tail.split(":", 1)[1].strip()
        else:
            tail = re.sub(
                r"^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b[\s\.\-\)]*",
                "",
                tail,
                flags=re.IGNORECASE,
            ).strip()

        normalized_first = expected_start if not tail else f"{expected_start} {tail}"
        return normalized_first + (f"\n{remaining}" if sep else "")

    return f"{expected_start} {text}".strip()


def parse_passages(context_source: Any) -> list[str]:
    if isinstance(context_source, str):
        try:
            parsed = ast.literal_eval(context_source)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return [str(context_source)]
        except Exception:
            return [str(context_source)]
    if isinstance(context_source, list):
        return [str(x) for x in context_source]
    return []


def build_step1_user_prompt(question: str, passages: list[str]) -> str:
    passages_str = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)])
    previous_steps_str = "(No previous steps.)"
    feedback_str = "Status: N/A (First attempt at this step)"

    prompt_user = f"""Question:
{question}

Retrieved Passages:
{passages_str}

Previous Reasoning Steps:
{previous_steps_str}

Feedback:
{feedback_str}

Generate next step (start with `Step 1:`)"""
    return prompt_user


def append_jsonl(file_path: str, payload: dict | list[dict]):
    if isinstance(payload, list):
        rows = payload
    else:
        rows = [payload]
    with open(file_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _max_consecutive_same_token_run(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9']+", str(text or "").lower())
    if not tokens:
        return 0
    run = 1
    best = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def contains_prompt_leak_marker(text: str) -> bool:
    lower = str(text or "").lower()
    return any(marker.lower() in lower for marker in PROMPT_LEAK_MARKERS)


def to_builtin(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def load_processed_ids(results_file_path: str) -> set:
    if not os.path.exists(results_file_path):
        return set()

    processed_ids = set()
    with open(results_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            row_id = row.get("id")
            if row_id is not None:
                processed_ids.add(str(row_id))
    return processed_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="/workspace/hf_transformers/Qwen2.5-7B-Instruct")
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_prompt_text", action="store_true")
    parser.add_argument("--debug_small", action="store_true")
    parser.add_argument("--disable_resume", action="store_true")
    args = parser.parse_args()

    if args.temperature < 0.0:
        parser.error("--temperature must be >= 0.0")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        parser.error("--top_p must be in (0.0, 1.0]")
    if args.repetition_penalty <= 0.0:
        parser.error("--repetition_penalty must be > 0.0")

    if args.debug_small:
        args.sample_size = 8
        args.batch_size = 4
        args.log_every = 1
        if not args.output_suffix.strip():
            args.output_suffix = "debug8"
        print(
            "Debug-small profile enabled: "
            f"sample_size={args.sample_size}, batch_size={args.batch_size}, log_every={args.log_every}"
        )

    base_path = "/workspace/daeyong"
    dataset_path = f"{base_path}/benchmarks/{args.dataset}_dev_kg_correct.csv"
    output_root = f"{base_path}/qwen_problem/inference_results"
    os.makedirs(output_root, exist_ok=True)

    model_name_clean = args.model_id.split("/")[-1]
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", args.output_suffix.strip())
    if suffix:
        output_dir = os.path.join(output_root, f"step1_only_{model_name_clean}_{suffix}")
    else:
        output_dir = os.path.join(output_root, f"step1_only_{model_name_clean}")
    os.makedirs(output_dir, exist_ok=True)

    results_file_path = os.path.join(output_dir, f"{args.dataset}_step1_results.jsonl")
    progress_file_path = os.path.join(output_dir, f"{args.dataset}_step1_progress.jsonl")
    summary_file_path = os.path.join(output_dir, f"{args.dataset}_step1_summary.json")

    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} total rows from {args.dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.disable_resume:
        print("Resume disabled. Starting fresh output files.")
        for path in [results_file_path, progress_file_path, summary_file_path]:
            if os.path.exists(path):
                os.remove(path)
    else:
        processed_ids = load_processed_ids(results_file_path)
        if processed_ids:
            if "id" in df.columns:
                df = df[~df["id"].astype(str).isin(processed_ids)]
            print(f"Resuming... {len(processed_ids)} already processed. Remaining rows: {len(df)}")

    if args.sample_size > 0:
        sample_n = min(args.sample_size, len(df))
        if sample_n < len(df):
            print(f"Sampling {sample_n} rows (seed={args.sample_seed}) from {len(df)} rows.")
        else:
            print(f"Using all rows ({sample_n}); sample_size={args.sample_size}.")
        df = df.sample(n=sample_n, random_state=args.sample_seed)
    else:
        print(f"sample_size={args.sample_size}; using all remaining rows ({len(df)}).")

    if len(df) == 0:
        print("No rows to process after resume/sample filtering.")
        empty_summary = {
            "num_samples": 0,
            "num_empty_step1": 0,
            "num_prefix_ok": 0,
            "num_contains_prompt_leak_markers": 0,
            "num_high_repetition": 0,
            "high_repetition_threshold": HIGH_REPETITION_RUN_THRESHOLD,
            "avg_prompt_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_step1_chars": 0.0,
            "results_file_path": results_file_path,
            "progress_file_path": progress_file_path,
            "created_at": int(time.time()),
        }
        with open(summary_file_path, "w", encoding="utf-8") as f:
            json.dump(empty_summary, f, ensure_ascii=False, indent=2)
        return

    llm, tokenizer = load_vllm_model(args.model_id)

    stop_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    stop_tokens = [token for token in stop_tokens if token is not None]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=256,
        stop_token_ids=stop_tokens,
    )
    print(
        "Sampling config: "
        f"temperature={args.temperature}, top_p={args.top_p}, "
        f"repetition_penalty={args.repetition_penalty}, max_tokens=256"
    )

    generator_calls_total = 0
    total_tokens = 0
    prompt_tokens_sum = 0
    output_tokens_sum = 0
    step1_chars_sum = 0

    num_empty_step1 = 0
    num_prefix_ok = 0
    num_contains_prompt_leak_markers = 0
    num_high_repetition = 0

    start_time = time.time()
    processed_total = 0
    total_rows = len(df)
    pbar = tqdm(total=total_rows, desc="Step1-only vLLM Processing")

    for iteration_idx, start_idx in enumerate(range(0, total_rows, args.batch_size), start=1):
        batch_df = df.iloc[start_idx : start_idx + args.batch_size]
        batch_items = []
        gen_prompts = []

        for row_idx, row in batch_df.iterrows():
            passages = parse_passages(row.get("retrieved_passages", ""))
            prompt_user = build_step1_user_prompt(str(row.get("question", "")), passages)
            messages = [
                {"role": "system", "content": generate_single_step_fixed_system_prompt},
                {"role": "user", "content": prompt_user},
            ]
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            batch_items.append(
                {
                    "row_idx": int(row_idx),
                    "id": to_builtin(row.get("id", f"row_{row_idx}")),
                    "question": to_builtin(row.get("question", "")),
                    "ground_truth": to_builtin(row.get("answer", "N/A")),
                    "retrieved_passages": to_builtin(passages),
                    "prompt_text": prompt_text,
                }
            )
            gen_prompts.append(prompt_text)

        gen_outputs = llm.generate(gen_prompts, sampling_params=sampling_params, use_tqdm=False)
        generator_calls_total += len(gen_outputs)

        batch_results = []
        for i, output in enumerate(gen_outputs):
            item = batch_items[i]
            raw_generation_text = output.outputs[0].text
            gen_text = output.outputs[0].text.strip()

            prompt_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)
            sample_total_tokens = prompt_tokens + output_tokens

            total_tokens += sample_total_tokens
            prompt_tokens_sum += prompt_tokens
            output_tokens_sum += output_tokens

            for marker in CLEAN_MARKERS:
                if marker in gen_text:
                    gen_text = gen_text.split(marker)[0].strip()

            step1_text = normalize_step_text(gen_text, expected_step_num=1)
            step1_chars_sum += len(step1_text)

            if not step1_text.strip() or step1_text.strip() == "Step 1:":
                num_empty_step1 += 1
            if step1_text.startswith("Step 1:"):
                num_prefix_ok += 1
            if contains_prompt_leak_marker(step1_text):
                num_contains_prompt_leak_markers += 1
            if _max_consecutive_same_token_run(step1_text) >= HIGH_REPETITION_RUN_THRESHOLD:
                num_high_repetition += 1

            result_row = {
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "retrieved_passages": item["retrieved_passages"],
                "step1_text": step1_text,
                "raw_generation_text": raw_generation_text,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": sample_total_tokens,
                "has_answer_tag": "####ANSWER" in step1_text,
                "has_end_reasoning_tag": "[END_OF_REASONING]" in step1_text,
            }
            if args.save_prompt_text:
                result_row["prompt_text"] = item["prompt_text"]

            batch_results.append(result_row)

        append_jsonl(results_file_path, batch_results)
        processed_in_batch = len(batch_results)
        processed_total += processed_in_batch
        remaining = total_rows - processed_total

        progress_row = {
            "iteration": iteration_idx,
            "timestamp": int(time.time()),
            "processed_in_batch": processed_in_batch,
            "processed_total": processed_total,
            "remaining": remaining,
            "generator_calls_total": generator_calls_total,
            "total_tokens": total_tokens,
            "elapsed_sec": round(time.time() - start_time, 2),
        }
        append_jsonl(progress_file_path, progress_row)

        pbar.update(processed_in_batch)

        if args.log_every > 0 and iteration_idx % args.log_every == 0:
            print(
                f"[Iter {iteration_idx}] processed_in_batch={processed_in_batch} "
                f"processed_total={processed_total}/{total_rows} "
                f"remaining={remaining} calls={generator_calls_total} "
                f"total_tokens={total_tokens}"
            )

    pbar.close()

    num_samples = processed_total
    summary = {
        "num_samples": num_samples,
        "num_empty_step1": num_empty_step1,
        "num_prefix_ok": num_prefix_ok,
        "num_contains_prompt_leak_markers": num_contains_prompt_leak_markers,
        "num_high_repetition": num_high_repetition,
        "high_repetition_threshold": HIGH_REPETITION_RUN_THRESHOLD,
        "avg_prompt_tokens": round(prompt_tokens_sum / max(1, num_samples), 4),
        "avg_output_tokens": round(output_tokens_sum / max(1, num_samples), 4),
        "avg_step1_chars": round(step1_chars_sum / max(1, num_samples), 4),
        "generator_calls_total": generator_calls_total,
        "total_tokens": total_tokens,
        "sampling_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": 256,
            "stop_token_ids": stop_tokens,
        },
        "results_file_path": results_file_path,
        "progress_file_path": progress_file_path,
        "created_at": int(time.time()),
    }
    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("✅ Step1-only generation complete.")
    print(f"Results: {results_file_path}")
    print(f"Progress: {progress_file_path}")
    print(f"Summary: {summary_file_path}")


if __name__ == "__main__":
    main()
