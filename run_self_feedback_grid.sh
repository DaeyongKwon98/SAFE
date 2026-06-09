#!/usr/bin/env bash
set -u -o pipefail

ROOT="/workspace/daeyong"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "${ROOT}/conda_envs/vllm/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/conda_envs/vllm/bin/python"
else
  PYTHON_BIN="python3"
fi

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LOG_EVERY="${LOG_EVERY:-10}"

MODELS=(qwen4b qwen8b qwen14b llama8b gemma12b)
DATASETS=(2wiki hotpotqa musique)
MAX_STEPS_LIST=(7 10 13)
MAX_RETRIES_LIST=(1 2 3 4 5)

declare -A MODEL_PATHS=(
  [qwen4b]="/workspace/hf_transformers/Qwen3-4B-Instruct-2507"
  [qwen8b]="/workspace/hf_transformers/Qwen3-8B"
  [qwen14b]="/workspace/hf_transformers/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
  [llama8b]="/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct"
  [gemma12b]="/workspace/hf_transformers/gemma-3-12b-it"
)

TOTAL_STEPS=0
FAIL_COUNT=0
FAILED_STEPS=()
EXECUTED_COMBOS=0
SKIPPED_COMBOS=0

TOTAL_COMBOS=0
for max_steps in "${MAX_STEPS_LIST[@]}"; do
  for max_retries in "${MAX_RETRIES_LIST[@]}"; do
    TOTAL_COMBOS=$((TOTAL_COMBOS + 1))
  done
done

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

run_step() {
  local step_name="$1"
  shift

  TOTAL_STEPS=$((TOTAL_STEPS + 1))
  log "START: ${step_name}"
  echo "CMD: $*"

  if "$@"; then
    log "OK: ${step_name}"
    return 0
  fi

  local exit_code=$?
  log "FAIL: ${step_name} (exit=${exit_code})"
  FAIL_COUNT=$((FAIL_COUNT + 1))
  FAILED_STEPS+=("${step_name}::exit=${exit_code}")
  return 0
}

run_stage2_final_answer() {
  local folder_path="$1"

  env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" - "${folder_path}" <<'PY'
import gc
import os
import sys

import pandas as pd
import torch

ROOT = "/workspace/daeyong"
FOLDER_PATH = sys.argv[1]
MODEL_ORDER = ["qwen4b", "qwen8b", "qwen14b", "llama8b", "gemma12b"]
DATASETS = ["2wiki", "hotpotqa", "musique"]
MODEL_PATHS = {
    "qwen4b": "/workspace/hf_transformers/Qwen3-4B-Instruct-2507",
    "qwen8b": "/workspace/hf_transformers/Qwen3-8B",
    "qwen14b": "/workspace/hf_transformers/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    "llama8b": "/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct",
    "gemma12b": "/workspace/hf_transformers/gemma-3-12b-it",
}

sys.path.insert(0, ROOT)
from final_answer_self_feedback import load_vllm_model, run_answer_generation_for_dataset  # noqa: E402

hard_errors = 0

if not os.path.isdir(FOLDER_PATH):
    print(f"[ERROR] Folder not found: {FOLDER_PATH}")
    sys.exit(1)

print(f"[INFO] Stage2 final answer generation start: {FOLDER_PATH}")

for model_short in MODEL_ORDER:
    model_path = MODEL_PATHS.get(model_short, "")
    if not model_path or not os.path.exists(model_path):
        print(f"[ERROR] Model path not found for {model_short}: {model_path}")
        hard_errors += 1
        continue

    is_qwen3_8b = "qwen3-8b" in model_path.lower()

    try:
        llm, tokenizer = load_vllm_model(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_short}: {e}")
        hard_errors += 1
        continue

    for dataset in DATASETS:
        input_file_name = f"{model_short}_{dataset}_results.json"
        input_path = os.path.join(FOLDER_PATH, input_file_name)
        output_path = os.path.join(FOLDER_PATH, f"{model_short}_{dataset}_final_answer.json")

        if not os.path.exists(input_path):
            print(f"[WARN] Input missing, skipping: {input_path}")
            continue

        try:
            df = pd.read_json(input_path)
        except Exception as e:
            print(f"[ERROR] Failed to read {input_path}: {e}")
            hard_errors += 1
            continue

        try:
            run_answer_generation_for_dataset(
                df=df,
                dataset=dataset,
                source_file_name=input_file_name,
                llm=llm,
                tokenizer=tokenizer,
                result_file_path=output_path,
                disable_thinking=is_qwen3_8b,
            )
        except Exception as e:
            print(f"[ERROR] Failed final answer generation: model={model_short} dataset={dataset} error={e}")
            hard_errors += 1

    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if hard_errors > 0:
    print(f"[ERROR] Stage2 completed with errors: {hard_errors}")
    sys.exit(1)

print("[OK] Stage2 completed successfully.")
PY
}

run_stage3_judge() {
  local folder_path="$1"
  env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" "${ROOT}/oss_answer_binary_self_feedback.py" \
    --folder_path "${folder_path}" \
    --models "${MODELS[@]}" \
    --datasets "${DATASETS[@]}"
}

log "Starting 14-grid pipeline: self-feedback -> final answer -> gpt-oss-120b judge"
log "PYTHON_BIN=${PYTHON_BIN}"
log "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
log "SAMPLE_SIZE=${SAMPLE_SIZE}, SAMPLE_SEED=${SAMPLE_SEED}, BATCH_SIZE=${BATCH_SIZE}, LOG_EVERY=${LOG_EVERY}"
log "Grid max_steps={${MAX_STEPS_LIST[*]}} max_retries={${MAX_RETRIES_LIST[*]}} with skip=(10,3)"

for max_steps in "${MAX_STEPS_LIST[@]}"; do
  for max_retries in "${MAX_RETRIES_LIST[@]}"; do
    if [[ "${max_steps}" == "10" && "${max_retries}" == "3" ]]; then
      SKIPPED_COMBOS=$((SKIPPED_COMBOS + 1))
      log "Skipping already-finished combo: max_steps=${max_steps}, max_retries=${max_retries}"
      continue
    fi

    EXECUTED_COMBOS=$((EXECUTED_COMBOS + 1))
    combo_folder="${ROOT}/inference_results/self_feedback_kg_correct_1k_sample_${max_steps}_${max_retries}"

    log "============================================================"
    log "Combo ${EXECUTED_COMBOS}: max_steps=${max_steps}, max_retries=${max_retries}"
    log "Output folder: ${combo_folder}"

    for model_short in "${MODELS[@]}"; do
      model_path="${MODEL_PATHS[$model_short]}"
      for dataset in "${DATASETS[@]}"; do
        run_step \
          "Stage1 self_feedback combo=${max_steps}_${max_retries} model=${model_short} dataset=${dataset}" \
          env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" "${ROOT}/inference_self_feedback_vllm.py" \
            --dataset "${dataset}" \
            --model_id "${model_path}" \
            --sample_size "${SAMPLE_SIZE}" \
            --sample_seed "${SAMPLE_SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --max_steps "${max_steps}" \
            --max_retries "${max_retries}" \
            --log_every "${LOG_EVERY}"
      done
    done

    run_step \
      "Stage2 final_answer combo=${max_steps}_${max_retries}" \
      run_stage2_final_answer "${combo_folder}"

    run_step \
      "Stage3 llm_judge combo=${max_steps}_${max_retries}" \
      run_stage3_judge "${combo_folder}"
  done
done

log "==================== Pipeline Summary ===================="
echo "Total grid combos: ${TOTAL_COMBOS}"
echo "Skipped combos: ${SKIPPED_COMBOS} (expected: 1)"
echo "Executed combos: ${EXECUTED_COMBOS} (expected: 14)"
echo "Total stage runs: ${TOTAL_STEPS}"
echo "Failures: ${FAIL_COUNT}"

if [[ "${EXECUTED_COMBOS}" -ne 14 ]]; then
  echo "WARNING: Executed combo count is ${EXECUTED_COMBOS}, expected 14."
fi

if [[ "${FAIL_COUNT}" -eq 0 ]]; then
  echo "All pipeline stages completed successfully."
  exit 0
fi

echo "Failed steps:"
for item in "${FAILED_STEPS[@]}"; do
  echo "- ${item}"
done
exit 1
