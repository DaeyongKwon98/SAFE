#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_ROOT="${INPUT_ROOT:-/workspace/daeyong/filtering_noise_data/false_triples_oss120b_seed42}"
MODEL_PATH="${MODEL_PATH:-/workspace/hf_transformers/gpt-oss-120b}"
TP_SIZE="${TP_SIZE:-4}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-false}"

CMD=(
  "$PYTHON_BIN"
  /workspace/daeyong/inject_errors_false_triples_oss120b.py
  --input_root "$INPUT_ROOT"
  --datasets 2wiki hotpotqa musique
  --error_types wrong_conclusion
  --model_path "$MODEL_PATH"
  --tensor_parallel_size "$TP_SIZE"
  --seed "$SEED"
  --max_items_per_dataset 200
)

if [[ "$OVERWRITE" == "true" ]]; then
  CMD+=(--overwrite)
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE"
echo "Running: ${CMD[*]}"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "${CMD[@]}"
