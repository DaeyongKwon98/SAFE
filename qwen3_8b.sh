#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7

ROOT="/workspace/daeyong"
MODEL_ID="/workspace/hf_transformers/Qwen3-8B"
NOFB_DIR="${ROOT}/inference_results/no_feedback_Qwen3-8B"
SELF_DIR="${ROOT}/inference_results/self_feedback_kg_correct_1k_sample"
DATASETS=(2wiki hotpotqa musique)

if [[ -x "${ROOT}/conda_envs/vllm/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/conda_envs/vllm/bin/python"
else
  PYTHON_BIN="python3"
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

run_qwen8b_judge() {
  # args: folder input_template output_template
  # template examples:
  #   "{dataset}_final_answer.json"
  #   "qwen8b_{dataset}_final_answer.json"
  local folder="$1"
  local input_template="$2"
  local output_template="$3"

  "${PYTHON_BIN}" - "$folder" "$input_template" "$output_template" <<'PY'
import ast
import csv
import json
import os
import re
import string
import sys

import pandas as pd

ROOT = "/workspace/daeyong"
sys.path.insert(0, ROOT)

from oss_answer_binary_self_feedback import (  # noqa: E402
    get_generated_answer,
    load_judge_model,
    parse_llm_output,
    system_prompt,
)


def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s.strip()


def load_musique_aliases():
    aliases = {}
    csv_path = os.path.join(ROOT, "benchmarks", "musique_dev.csv")
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("id", "")).strip()
            raw = row.get("answer_list", "")
            candidates = []
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, list):
                        candidates = parsed
                    else:
                        candidates = [parsed]
                except Exception:
                    candidates = [raw]
            elif raw:
                candidates = [raw]
            norm = [normalize_text(str(x)) for x in candidates if normalize_text(str(x))]
            if norm and sample_id:
                aliases[sample_id] = norm
    return aliases


folder = sys.argv[1]
input_template = sys.argv[2]
output_template = sys.argv[3]
datasets = ["2wiki", "hotpotqa", "musique"]

llm, tokenizer, sampling_params = load_judge_model()
musique_aliases = load_musique_aliases()

for dataset in datasets:
    input_path = os.path.join(folder, input_template.format(dataset=dataset))
    output_path = os.path.join(folder, output_template.format(dataset=dataset))

    if not os.path.exists(input_path):
        print(f"Skipping: {input_path} (File not found)")
        continue

    print(f"\n🚀 Judge Processing: {dataset}")
    df = pd.read_json(input_path)
    if df.empty:
        print("Empty dataframe. Skipping.")
        continue

    prompts = []
    generated_answers = []
    empty_flags = []

    for _, row in df.iterrows():
        if dataset == "musique":
            gt_list = musique_aliases.get(str(row.get("id", "")).strip(), [normalize_text(str(row.get("ground_truth", "")))])
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
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, sampling_params)

    is_correct_list = []
    reasoning_list = []

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.split("assistantfinal")[-1].strip()
        is_correct, reasoning = parse_llm_output(generated_text)
        if empty_flags[i]:
            reasoning = f"[empty_generated_answer] {reasoning}"
        is_correct_list.append(is_correct)
        reasoning_list.append(reasoning)

    df["generated_answer_used"] = generated_answers
    df["is_correct"] = is_correct_list
    df["reasoning"] = reasoning_list
    df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    print(f"✅ Saved: {output_path}")

print("\n✅ Judge step done.")
PY
}

log "Using python: ${PYTHON_BIN}"
log "Target model: ${MODEL_ID}"

mkdir -p "${NOFB_DIR}" "${SELF_DIR}"

log "1) No feedback (1k kg correct) inference"
for d in "${DATASETS[@]}"; do
  log "  - dataset=${d}"
  "${PYTHON_BIN}" "${ROOT}/inference_baseline_vllm.py" \
    --dataset "${d}" \
    --model_id "${MODEL_ID}"
done

log "2) No feedback final answer generation (Qwen3-8B only)"
"${PYTHON_BIN}" - <<'PY'
import gc
import os
import sys

import pandas as pd
import torch

ROOT = "/workspace/daeyong"
MODEL_PATH = "/workspace/hf_transformers/Qwen3-8B"
INPUT_DIR = f"{ROOT}/inference_results/no_feedback_Qwen3-8B"
DATASETS = ["2wiki", "hotpotqa", "musique"]

sys.path.insert(0, ROOT)
from inference_baseline_final_answer import load_vllm_model, run_answer_generation  # noqa: E402

llm, tokenizer = load_vllm_model(MODEL_PATH)

for dataset in DATASETS:
    input_file = os.path.join(INPUT_DIR, f"{dataset}_results.json")
    output_file = os.path.join(INPUT_DIR, f"{dataset}_final_answer.json")
    if not os.path.exists(input_file):
        print(f"Skipping missing input: {input_file}")
        continue
    df = pd.read_json(input_file)
    if df.empty:
        print(f"Skipping empty input: {input_file}")
        continue
    run_answer_generation(
        df=df,
        llm=llm,
        tokenizer=tokenizer,
        result_file_path=output_file,
        disable_thinking=True,
    )
    print(f"✅ Saved: {output_file}")

del llm
del tokenizer
gc.collect()
torch.cuda.empty_cache()
PY

log "3) OSS binary answer (no feedback, Qwen3-8B only)"
run_qwen8b_judge "${NOFB_DIR}" "{dataset}_final_answer.json" "{dataset}_llm.json"

log "4) Self feedback (1k kg correct) inference"
for d in "${DATASETS[@]}"; do
  log "  - dataset=${d}"
  "${PYTHON_BIN}" "${ROOT}/inference_self_feedback_vllm.py" \
    --dataset "${d}" \
    --model_id "${MODEL_ID}" \
    --sample_size 1000 \
    --sample_seed 42 \
    --batch_size 256 \
    --max_steps 10 \
    --max_retries 3 \
    --log_every 10
done

log "5) Self feedback final answer generation (Qwen3-8B only)"
"${PYTHON_BIN}" - <<'PY'
import gc
import os
import sys

import pandas as pd
import torch

ROOT = "/workspace/daeyong"
MODEL_PATH = "/workspace/hf_transformers/Qwen3-8B"
FOLDER = f"{ROOT}/inference_results/self_feedback_kg_correct_1k_sample"
DATASETS = ["musique", "hotpotqa", "2wiki"]
MODEL_SHORT = "qwen8b"

sys.path.insert(0, ROOT)
from final_answer_self_feedback import load_vllm_model, run_answer_generation_for_dataset  # noqa: E402

llm, tokenizer = load_vllm_model(MODEL_PATH)

for dataset in DATASETS:
    input_file_name = f"{MODEL_SHORT}_{dataset}_results.json"
    input_path = os.path.join(FOLDER, input_file_name)
    output_path = os.path.join(FOLDER, f"{MODEL_SHORT}_{dataset}_final_answer.json")

    if not os.path.exists(input_path):
        print(f"Skipping missing input: {input_path}")
        continue

    df = pd.read_json(input_path)
    run_answer_generation_for_dataset(
        df=df,
        dataset=dataset,
        source_file_name=input_file_name,
        llm=llm,
        tokenizer=tokenizer,
        result_file_path=output_path,
        disable_thinking=True,
    )

del llm
del tokenizer
gc.collect()
torch.cuda.empty_cache()
PY

log "6) OSS binary answer (self feedback, Qwen3-8B only)"
run_qwen8b_judge "${SELF_DIR}" "qwen8b_{dataset}_final_answer.json" "qwen8b_{dataset}_llm_judge.json"

log "🎉 All Qwen3-8B pipeline steps completed."
