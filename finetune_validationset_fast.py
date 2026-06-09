import argparse
import ast
from collections import Counter
import json
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from prompts import evaluate_system_prompt_premature_attribution

### 학습 명령어 ###
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 finetune_validationset_fast.py
##################

DEFAULT_MODEL_ID = "/workspace/hf_transformers/Qwen3-8B"
DEFAULT_DATASET_PATH = (
	"/workspace/daeyong/fourth_finetuning_data/"
	"final_answer_with_wrong_conclusion_off_topic_first_premature_attribution_gpt_oss20b.jsonl"
)
DEFAULT_OUTPUT_DIR = (
	"/workspace/daeyong/trained_models/"
	"qwen3-8b-final-answer-gpt-oss20b-lora4bit"
)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Qwen3-8B 4bit LoRA SFT trainer for reasoning feedback data."
	)
	parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
	parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
	parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--test_size", type=float, default=0.1)
	parser.add_argument("--max_length", type=int, default=8500)
	parser.add_argument("--num_train_epochs", type=float, default=2.0)
	parser.add_argument("--per_device_train_batch_size", type=int, default=1)
	parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
	parser.add_argument("--eval_steps", type=int, default=100)
	parser.add_argument("--save_steps", type=int, default=100)
	parser.add_argument("--logging_steps", type=int, default=10)
	parser.add_argument("--learning_rate", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=0.01)
	parser.add_argument("--max_grad_norm", type=float, default=0.3)
	parser.add_argument("--warmup_ratio", type=float, default=0.03)
	parser.add_argument("--lr_scheduler_type", default="cosine")
	parser.add_argument("--optim", default="adamw_torch")
	parser.add_argument("--report_to", default="tensorboard")
	parser.add_argument("--lora_r", type=int, default=64)
	parser.add_argument("--lora_alpha", type=int, default=128)
	parser.add_argument("--lora_dropout", type=float, default=0.05)
	parser.add_argument("--overwrite_output_dir", action="store_true")
	parser.add_argument("--disable_gradient_checkpointing", action="store_true")
	parser.add_argument(
		"--resume_from_checkpoint",
		default=None,
		help="Checkpoint path to resume from. Leave unset to start a new run.",
	)
	return parser.parse_args()


def parse_passages_to_text(passages_value):
	if isinstance(passages_value, list):
		return "\n".join(f"Passage {idx + 1}: {p}" for idx, p in enumerate(passages_value))
	if isinstance(passages_value, str):
		text = passages_value.strip()
		if not text:
			return ""
		try:
			parsed = ast.literal_eval(text)
			if isinstance(parsed, list):
				return "\n".join(f"Passage {idx + 1}: {p}" for idx, p in enumerate(parsed))
		except Exception:
			pass
		return text
	return str(passages_value)


def format_steps(steps):
	if isinstance(steps, str):
		try:
			parsed = ast.literal_eval(steps)
			if isinstance(parsed, list):
				steps = parsed
		except Exception:
			pass
	if isinstance(steps, list) and steps:
		return "\n".join(str(step).strip() for step in steps if str(step).strip())
	return "(No previous steps.)"


def build_training_input_from_raw(item):
	question = str(item.get("question", "")).strip()
	passages_text = parse_passages_to_text(item.get("retrieved_passages", []))
	previous_text = format_steps(item.get("previous_steps", []))
	current_step = str(item.get("current_step", "")).strip()
	return (
		"### Task: Evaluate the Correctness of the Reasoning Step\n\n"
		f"Question:\n{question}\n\n"
		f"Retrieved Passages:\n{passages_text}\n\n"
		f"Previous Steps:\n{previous_text}\n\n"
		f"Step to evaluate:\n{current_step}"
	)


def load_raw_items(dataset_path):
	if dataset_path.endswith(".jsonl"):
		items = []
		with open(dataset_path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if line:
					items.append(json.loads(line))
		return items

	with open(dataset_path, "r", encoding="utf-8") as f:
		loaded = json.load(f)
	if isinstance(loaded, list):
		return loaded
	if isinstance(loaded, dict) and isinstance(loaded.get("data"), list):
		return loaded["data"]
	if isinstance(loaded, dict):
		return [loaded]
	raise ValueError(f"Unsupported dataset structure: {dataset_path}")


def extract_error_type(output_text):
	if isinstance(output_text, dict):
		return str(output_text.get("error_type", "")).strip()
	try:
		parsed = json.loads(str(output_text))
		if isinstance(parsed, dict):
			return str(parsed.get("error_type", "")).strip()
	except Exception:
		pass
	return ""


def coerce_training_record(item):
	if not isinstance(item, dict):
		return None

	if "input" in item and "output" in item:
		input_text = item["input"]
		output_text = item["output"]
		if isinstance(output_text, dict):
			output_text = json.dumps(output_text, ensure_ascii=False)
		return {
			"input": str(input_text),
			"output": str(output_text),
			"error_type": extract_error_type(output_text),
		}

	required = ("question", "current_step", "error_type", "diagnosis", "guidance")
	if not all(item.get(key) for key in required):
		return None
	output = {
		"error_type": item.get("error_type", ""),
		"diagnosis": item.get("diagnosis", ""),
		"guidance": item.get("guidance", ""),
	}
	return {
		"input": build_training_input_from_raw(item),
		"output": json.dumps(output, ensure_ascii=False),
		"error_type": str(item.get("error_type", "")).strip(),
	}


def load_training_records(dataset_path):
	raw_items = load_raw_items(dataset_path)
	records = []
	dropped = 0
	for item in raw_items:
		record = coerce_training_record(item)
		if record is None or not record["input"] or not record["output"]:
			dropped += 1
			continue
		records.append(record)
	return records, {
		"raw_total": len(raw_items),
		"used_total": len(records),
		"dropped_total": dropped,
	}


def main():
	args = parse_args()

	# 1. [필수] LOCAL_RANK 환경변수 가져오기
	local_rank = int(os.environ.get("LOCAL_RANK", 0))

	# 2. [필수] 현재 프로세스가 사용할 GPU를 강제로 지정
	# 여기서 local_rank가 0이면 -> 실제로는 물리 GPU 2번을 잡습니다.
	torch.cuda.set_device(local_rank)

	# --- 1. 설정 (Configuration) ---
	model_id = args.model_id
	dataset_path = args.dataset_path
	output_dir = args.output_dir

	# --- 2. 데이터셋 로드 ---
	try:
		records, dataset_stats = load_training_records(dataset_path)
		error_distribution = Counter(
			record["error_type"] for record in records if record.get("error_type")
		)
		data = []
		for item in records:
			# Conversational prompt-completion 형식으로 변환
			data.append({
				"prompt": [
					{"role": "system", "content": evaluate_system_prompt_premature_attribution},
					{"role": "user", "content": item["input"]}
				],
				"completion": [
					{"role": "assistant", "content": item["output"]}
				],
				"chat_template_kwargs": {"enable_thinking": False}
			})
		full_dataset = Dataset.from_list(data)
		
		# 메인 프로세스(GPU 0)에서만 로그 출력
		if int(os.environ.get("LOCAL_RANK", 0)) == 0:
			print(f"📂 데이터셋: {dataset_path}")
			print(
				f"✅ 전체 데이터셋 로드 완료. "
				f"총 {len(full_dataset)}개의 샘플 "
				f"(raw={dataset_stats['raw_total']}, dropped={dataset_stats['dropped_total']})."
			)
			print(f"🔎 error_type 분포: {dict(error_distribution)}")

		dataset_split = full_dataset.train_test_split(test_size=args.test_size, seed=args.seed)
		train_dataset = dataset_split['train']
		eval_dataset = dataset_split['test']

	except Exception as e:
		print(f"❌ 데이터셋 로드 실패: {e}")
		return

	# --- 3. 모델 및 토크나이저 로드 ---
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.bfloat16,
		bnb_4bit_use_double_quant=False,
	)

	# [수정 2] DDP를 위한 device_map 설정
	# 기존: device_map = {"": torch.cuda.current_device()} 
	# 수정: local_rank를 직접 명시하는 것이 가장 안전합니다.
	device_map = {"": local_rank}

	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		quantization_config=bnb_config,
		device_map=device_map,
		trust_remote_code=True,
		# DDP 호환성을 위해 cache 비활성화
		use_cache=False 
	)
	
	tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"

	# --- 4. LoRA 설정 ---
	peft_config = LoraConfig(
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		r=args.lora_r,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules=[
			"q_proj", "k_proj", "v_proj", "o_proj",
			"gate_proj", "up_proj", "down_proj",
		],
	)


	# --- 6. SFTConfig 설정---
	sft_config = SFTConfig(
		output_dir=output_dir,
		num_train_epochs=args.num_train_epochs,
		per_device_train_batch_size=args.per_device_train_batch_size,
		per_device_eval_batch_size=args.per_device_eval_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
  		gradient_checkpointing=(not args.disable_gradient_checkpointing),
		optim=args.optim,
		do_eval=True,                 
		eval_strategy="steps",
		eval_steps=args.eval_steps,
		save_steps=args.save_steps,
		logging_steps=args.logging_steps,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		fp16=False,
		bf16=True,
		max_grad_norm=args.max_grad_norm,
		warmup_ratio=args.warmup_ratio,
		group_by_length=True,
		lr_scheduler_type=args.lr_scheduler_type,
		report_to=args.report_to,
		ddp_find_unused_parameters=False,
		max_length=args.max_length,
		packing=False,
		seed=args.seed,
		overwrite_output_dir=args.overwrite_output_dir,
	)

	# --- 7. Trainer ---
	trainer = SFTTrainer(
		model=model,
		processing_class=tokenizer,
		train_dataset=train_dataset, 
		eval_dataset=eval_dataset,   
		peft_config=peft_config,
		args=sft_config,
	)

	if int(os.environ.get("LOCAL_RANK", 0)) == 0:
		print("🚀 파인튜닝을 시작합니다...")
	
	if args.resume_from_checkpoint:
		trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
	else:
		trainer.train()
	# trainer.train(resume_from_checkpoint=True) # 가장 최근 checkpoint부터 이어서 진행

	if int(os.environ.get("LOCAL_RANK", 0)) == 0:
		print("✅ 파인튜닝 완료!")
		trainer.save_model(output_dir)
		tokenizer.save_pretrained(output_dir)
		print(f"✔ 학습된 LoRA 어댑터가 '{output_dir}'에 저장되었습니다.")

		# 1. 저장할 설정들을 딕셔너리로 구성
		# peft_config와 sft_config는 .to_dict() 메서드를 제공합니다.
		hyperparameters = {
			"meta_info": {
				"model_id": model_id,
				"dataset_path": dataset_path,
				"output_dir": output_dir,
				"dataset_stats": dataset_stats,
				"error_distribution": dict(error_distribution),
			},
			"lora_config": peft_config.to_dict(),  # LoRA 설정
			"training_args": sft_config.to_dict(), # 학습 파라미터 (Epoch, LR, Batch 등)
		}

		# 2. JSON 파일로 저장
		hp_save_path = os.path.join(output_dir, "hyperparameters.json")
		
		try:
			with open(hp_save_path, "w", encoding="utf-8") as f:
				json.dump(hyperparameters, f, indent=4, ensure_ascii=False)
			print(f"✔ 하이퍼파라미터 설정이 '{hp_save_path}'에 저장되었습니다.")
		except Exception as e:
			print(f"❌ 하이퍼파라미터 저장 중 오류 발생: {e}")
			# 혹시 직렬화가 안 되는 객체가 있을 경우를 대비해 문자열로 변환해서 저장 시도
			with open(hp_save_path, "w", encoding="utf-8") as f:
				f.write(str(hyperparameters))

if __name__ == "__main__":
	main()
