#!/usr/bin/env bash
set -euo pipefail

cd /workspace/daeyong

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

TORCHRUN="${TORCHRUN:-/workspace/daeyong/conda_envs/training/bin/torchrun}"

"${TORCHRUN}" --standalone --nproc_per_node=4 \
  /workspace/daeyong/finetune_validationset_fast.py \
  --model_id /workspace/hf_transformers/Qwen3-8B \
  --dataset_path /workspace/daeyong/fourth_finetuning_data/final_answer_with_wrong_conclusion_off_topic_first_premature_attribution_gpt_oss20b.jsonl \
  --output_dir /workspace/daeyong/trained_models/qwen3-8b-final-answer-gpt-oss20b-lora4bit \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --max_grad_norm 0.3 \
  --lr_scheduler_type cosine \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 10 \
  --max_length 8500 \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --report_to tensorboard \
  "$@"
