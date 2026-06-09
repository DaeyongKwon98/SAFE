import argparse
import ast
import gc
import json
import os
import re
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ANSWER_GENERATION_SYSTEM_PROMPT = """You are an expert answering agent.
The reasoning process is complete. Your task is to formulate the FINAL ANSWER based on the provided history.

INSTRUCTIONS:
1. Do not generate any new reasoning steps.
2. Directly output the final answer.
3. YOU MUST USE THE FOLLOWING FORMAT:
####ANSWER: your_final_answer_here (Final Answer)"""

MODEL_MAPPING = {
    # "qwen7b": "/workspace/hf_transformers/Qwen2.5-7B-Instruct",
    "llama8b": "/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct",
    "gemma12b": "/workspace/hf_transformers/gemma-3-12b-it",
    "qwen4b": "/workspace/hf_transformers/Qwen3-4B-Instruct-2507",
    "qwen8b": "/workspace/hf_transformers/Qwen3-8B",
    "qwen36_27b": "/workspace/hf_transformers/Qwen3.6-27B",
    "qwen14b": "/workspace/hf_transformers/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
}

DEFAULT_MODELS = ["qwen4b", "qwen8b", "qwen14b"]
DATASETS = ["musique", "hotpotqa", "2wiki"]

def load_vllm_model(
    model_path: str,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 3000,
) -> Tuple[LLM, AutoTokenizer]:
    print(f"\\n🔵 Loading Model: '{model_path}'")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        enforce_eager=False,
        enable_prefix_caching=True,
    )

    print("✅ Model loaded successfully.")
    return llm, tokenizer


def parse_answer_from_last_step(last_step: Any) -> str:
    if not isinstance(last_step, str):
        return ""

    match = re.search(r"####ANSWER:\s*(.*)", last_step, re.DOTALL)
    if not match:
        return ""

    answer_line = match.group(1).strip().split("\n", 1)[0]
    answer_line = answer_line.replace("(Final Answer)", "").strip()
    return answer_line


def get_reasoning_list(response_field: Any) -> List[str]:
    if isinstance(response_field, list):
        return [str(x) for x in response_field]

    if isinstance(response_field, str):
        text = response_field.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (ValueError, SyntaxError):
                pass
        return [response_field]

    return []


def make_generation_prompt(
    question: str,
    reasoning_steps: List[str],
    tokenizer,
    disable_thinking: bool = False,
) -> str:
    reasoning_str = "\n".join(reasoning_steps) if reasoning_steps else "No reasoning provided."

    user_content = f"""Question: {question}

Provided Reasoning Process:
{reasoning_str}

Based on the reasoning process above, what is the final answer?
Remember to start your response with ####ANSWER:"""

    messages = [
        {"role": "system", "content": ANSWER_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    template_kwargs = {"add_generation_prompt": True, "tokenize": False}
    if disable_thinking:
        template_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def run_answer_generation_for_dataset(
    df: pd.DataFrame,
    dataset: str,
    source_file_name: str,
    llm: LLM,
    tokenizer,
    result_file_path: str,
    disable_thinking: bool = False,
) -> None:
    processed_ids = set()
    existing_results: List[Dict[str, Any]] = []

    if os.path.exists(result_file_path):
        try:
            with open(result_file_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            processed_ids = {item["id"] for item in existing_results if "id" in item}
            print(f"🔄 Resuming... {len(processed_ids)} already processed in {os.path.basename(result_file_path)}")
        except Exception as e:
            print(f"⚠️ Failed to read existing output ({result_file_path}): {e}. Rewriting from scratch.")
            existing_results = []
            processed_ids = set()

    df = df[~df["id"].isin(processed_ids)]
    if len(df) == 0:
        print(f"⏭️  All data already processed for {dataset}.")
        return

    prompts: List[str] = []
    prompt_indices: List[int] = []
    pending_records: List[Dict[str, Any]] = []

    for row_idx, row in df.iterrows():
        reasoning_steps = get_reasoning_list(row.get("response"))
        last_step = reasoning_steps[-1] if reasoning_steps else ""
        parsed_answer = parse_answer_from_last_step(last_step)

        base_record = {
            "id": row.get("id"),
            "question": row.get("question", ""),
            "ground_truth": row.get("ground_truth", "N/A"),
            "source_file": source_file_name,
            "input_last_step": last_step if isinstance(last_step, str) else str(last_step),
            "input_reasoning": "\n".join(reasoning_steps) if reasoning_steps else "No reasoning provided.",
            "answer_source": "parsed_last_step" if parsed_answer else "generated",
            "full_response": "",
            "final_answer_extracted": parsed_answer,
        }

        if parsed_answer:
            pending_records.append(base_record)
            continue

        prompts.append(
            make_generation_prompt(
                base_record["question"],
                reasoning_steps,
                tokenizer,
                disable_thinking=disable_thinking,
            )
        )
        prompt_indices.append(len(pending_records))
        pending_records.append(base_record)

    if prompts:
        stop_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        stop_tokens = [t for t in stop_tokens if t is not None]

        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=128, stop_token_ids=stop_tokens),
            use_tqdm=False,
        )

        for out_idx, output in enumerate(outputs):
            rec_idx = prompt_indices[out_idx]
            generated_text = output.outputs[0].text.strip()

            extracted = parse_answer_from_last_step(generated_text)
            if not extracted:
                extracted = generated_text

            pending_records[rec_idx]["full_response"] = generated_text
            pending_records[rec_idx]["final_answer_extracted"] = extracted.strip()

    merged_results = existing_results + pending_records

    with open(result_file_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    parsed_count = sum(1 for r in pending_records if r["answer_source"] == "parsed_last_step")
    generated_count = len(pending_records) - parsed_count
    print(
        f"✅ Saved {len(pending_records)} rows ({dataset}) -> {result_file_path} "
        f"[parsed: {parsed_count}, generated: {generated_count}]"
    )


def parse_cli_tokens(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in values:
        for token in str(item).split(","):
            token = token.strip()
            if token:
                out.append(token)
    return list(dict.fromkeys(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--max_model_len", type=int, default=3000)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    folder_path = args.folder_path
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    model_list = parse_cli_tokens(args.models)
    dataset_list = parse_cli_tokens(args.datasets)

    print(f"🚀 Start Final Answer Generation on self-feedback folder: {folder_path}")
    print(
        f"⚙️ Settings: max_model_len={args.max_model_len}, "
        f"models={model_list}, datasets={dataset_list}"
    )

    # Keep the existing default behavior (qwen4b/qwen8b/qwen14b) unless overridden.
    for model_short in model_list:
        model_path = MODEL_MAPPING.get(model_short, "")
        if not model_path or not os.path.exists(model_path):
            print(f"❌ Model path not found for {model_short}: {model_path}")
            continue

        is_qwen3_model = "qwen3" in model_path.lower()

        try:
            llm, tokenizer = load_vllm_model(
                model_path=model_path,
                max_model_len=args.max_model_len,
            )
        except Exception as e:
            print(f"❌ Failed to load model {model_short}: {e}")
            continue

        for dataset in dataset_list:
            input_file_name = f"{model_short}_{dataset}_results.json"
            input_path = os.path.join(folder_path, input_file_name)
            output_path = os.path.join(folder_path, f"{model_short}_{dataset}_final_answer.json")

            if not os.path.exists(input_path):
                print(f"⚠️ Input file missing, skipping: {input_path}")
                continue

            try:
                df = pd.read_json(input_path)
            except Exception as e:
                print(f"❌ Failed to read {input_path}: {e}")
                continue

            print("-----------------------------------------------------------")
            print(f"📂 Processing: [{model_short}] -> [{dataset}]")
            print("-----------------------------------------------------------")

            try:
                run_answer_generation_for_dataset(
                    df=df,
                    dataset=dataset,
                    source_file_name=input_file_name,
                    llm=llm,
                    tokenizer=tokenizer,
                    result_file_path=output_path,
                    disable_thinking=is_qwen3_model,
                )
            except Exception as e:
                print(f"❌ Error processing {dataset} with {model_short}: {e}")

        print(f"🧹 Unloading model {model_short}...")
        del llm
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Memory cleared.\n")

    print("🎉 final_answer_self_feedback.py finished.")


if __name__ == "__main__":
    main()
