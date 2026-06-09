import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os
import json
from datasets import Dataset

from prompts import evaluate_system_prompt_premature_attribution_missing_evidence

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

### 학습 명령어 ###
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 finetune_validationset_fast_missing_evidence.py
##################

def main():
	# 1. [필수] LOCAL_RANK 환경변수 가져오기
	local_rank = int(os.environ.get("LOCAL_RANK", 0))

	# 2. [필수] 현재 프로세스가 사용할 GPU를 강제로 지정
	# 여기서 local_rank가 0이면 -> 실제로는 물리 GPU 2번을 잡습니다.
	torch.cuda.set_device(local_rank)

	# --- 1. 설정 (Configuration) ---
	# model_id = "/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct"
	model_id = "/workspace/hf_transformers/Qwen3-8B"
	# model_id = "/workspace/hf_transformers/Qwen2.5-7B-Instruct"
	# model_id = "/workspace/hf_transformers/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
	dataset_path = "/workspace/daeyong/fourth_finetuning_data/missing_evidence_training_data.jsonl"
	output_dir = "./trained_models/qwen3-8b-missing_evidence_training_data"
	system_prompt = evaluate_system_prompt_premature_attribution_missing_evidence

	# --- 2. 데이터셋 로드 ---
	try:
		data = []
		with open(dataset_path, 'r', encoding='utf-8') as f:
			for line in f:
				if line.strip():
					item = json.loads(line)
     
					# Conversational prompt-completion 형식으로 변환
					data.append({
						"prompt": [
							{"role": "system", "content": system_prompt},
							{"role": "user", "content": item['input']}
						],
						"completion": [
							{"role": "assistant", "content": item['output']}
						],
						"chat_template_kwargs": {"enable_thinking": False} # Qwen3 8B할때 켜기
					})
		full_dataset = Dataset.from_list(data)
		
		# 메인 프로세스(GPU 0)에서만 로그 출력
		if int(os.environ.get("LOCAL_RANK", 0)) == 0:
			print(f"✅ 전체 데이터셋 로드 완료. 총 {len(full_dataset)}개의 샘플.")

		dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
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
		lora_alpha=128,
		lora_dropout=0.05,
		r=64,
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
		num_train_epochs=2,
		per_device_train_batch_size=1,
		per_device_eval_batch_size=1,
		gradient_accumulation_steps=16,
  		gradient_checkpointing=True,
		optim="adamw_torch",
		do_eval=True,                 
		eval_strategy="steps",
		eval_steps=100,
		save_steps=100,
		logging_steps=10,
		learning_rate=1e-4,
		weight_decay=0.01,
		fp16=False,
		bf16=True,
		max_grad_norm=0.3,
		warmup_ratio=0.03,
		group_by_length=True,
		lr_scheduler_type="cosine",
		report_to="tensorboard",
		ddp_find_unused_parameters=False,
		max_length=8500,
		packing=False,
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
				"system_prompt": "evaluate_system_prompt_premature_attribution_missing_evidence",
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
