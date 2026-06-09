import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
import os
import json
import pandas as pd
from datasets import Dataset
from accelerate import Accelerator

# CUDA 할당 문제 방지
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

################# 학습 명령어 #####################
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes=8 dpo.py
# accelerate config로 설정 수정해야 할 수도 있음!
##################################################

system_prompt = """You are an expert reasoning evaluator.
Your task is to critically assess the reasoning step (STEP TO EVALUATE) by:
1. Classifying the step into one of the error categories.
2. Providing consise diagnosis (why) and guidance (what next).

### Error Type Definitions:
- **Correct (No Error)**: The step is logically sound, fully supported by the retrieved passages, and moves the reasoning forward correctly.
- **Off-topic**: The step is irrelevant to the overall goal of the question and the specific step it is replacing. It introduces a new, unrelated piece of information or inference that leads the reasoning process astray.
- **Redundancy**: The step repeats information or conclusions from previous steps without providing any significant new progression. It stalls the reasoning process by repeating what is already known.
- **Overthinking**: The step continues *after* the reasoning is sufficient to answer the question. It introduces a new, *unnecessary* line of reasoning that is no longer required to find the final answer.
- **Inefficiency**: The step provides meta-discussion, procedural intent, planning statements, or placeholder reasoning instead of executing meaningful inference. It does not extract evidence or logically reason but merely describes what the model plans to do rather than doing it.
- **Logical Fallacy**: The step contains a flawed reasoning process. The facts gathered from previous steps are correct, but the conclusion drawn from them is incorrect.
- **Unsupported**: The step makes a factual claim using information that **cannot** be found in any of the `Retrieved Passages`. The step hallucinates a new, false piece of information and presents it as fact.
- **Contradictory**: The step makes a factual claim that **directly conflicts** or **contradicts** information explicitly stated in the `Retrieved Passages`.
- **Information Miss**: The step incorrectly concludes that specific information is unavailable, unknown, or missing—even though the information is present in the retrieved passages. The failure lies in not recognizing or retrieving relevant evidence that already exists.

### Feedback Requirements:
Your feedback must be split into two distinct parts:
   - **Diagnosis:** Explain *why* the step is correct or erroneous based on the specific definition above. Be specific about what fact or logic is involved.
   - **Guidance:**
	 - If **Correct**: Briefly confirm the finding and suggest the logical next step (e.g., "Now that you've found [Entity], look for [Next Info]...").
	 - If **Error**: Explicitly point out the mistake and guide the user on what they *should* have done or looked for instead to advance the reasoning correctly.

### Output Requirements:
Always output a JSON object with `"error_type"`, `"diagnosis"`, and `"guidance"` keys:
`"error_type"` — The one error category selected.
`"diagnosis"` — The explanation of why the step is correct or erroneous.
`"guidance"` — The instruction on what to do next or how to correct the error."""

def format_dpo_data(example):
	"""
	데이터셋을 DPO Chat Template 형식으로 변환하는 함수
	"""
	# 1. Prompt 구성 (SFT 입력과 동일하게 Chat Template 적용)
	prompt_msg = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": example['input']}
	]
	
	# 2. Chosen / Rejected 구성
	chosen_msg = [
		{"role": "assistant", "content": example['chosen']} 
	]
	rejected_msg = [
		{"role": "assistant", "content": example['rejected']}
	]

	return {
		"prompt": prompt_msg,
		"chosen": chosen_msg,
		"rejected": rejected_msg
	}

def construct_user_input(row):
	"""
	DataFrame의 컬럼들을 조합하여 모델에 들어갈 User Input String을 생성합니다.
	(SFT 학습 때 사용한 포맷과 띄어쓰기, 줄바꿈까지 정확히 일치시키는 것이 좋습니다)
	"""
	# 리스트나 문자열 처리를 위한 안전장치
	passages = row['retrieved_passages']
	prev_steps = row['previous_steps']
	
	# 문자열로 들어온 경우 리스트로 변환 (필요시)
	if isinstance(passages, str):
		passages = eval(passages)
	if isinstance(prev_steps, str):
		prev_steps = eval(prev_steps)

	input_str = f"Question: {row['question']}\n"
	input_str += f"Retrieved Passages: {passages}\n"
	input_str += f"Previous Steps: {prev_steps}\n"
	input_str += f"Current Step: {row['current_step']}"
	
	return input_str

def main():
	# --- 1. 설정 (Configuration) ---
	model_id = "./trained_models/qwen2.5-7b-third-fixed-merged" 
	
	# JSON 파일 경로
	dataset_path = "/workspace/daeyong/first_dpo_training_data_rewrited_length.json"
	output_dir = "./trained_models/qwen2.5-7b-dpo-v1-beta0.2_rewrited_400sample"

	# --- 2. 데이터셋 로드 및 전처리 ---
	try:
		# if local_rank == 0:
		#     print(f"📂 데이터셋 로딩 중: {dataset_path}")
		print(f"📂 데이터셋 로딩 중: {dataset_path}")
			
		# Pandas로 JSON 읽기
		df = pd.read_json(dataset_path)
		
		# [END_OF_REASONING]이 row['guidance_gpt']에 포함된 데이터는 모두 포함하고, 이외의 데이터들 중 500개만 랜덤 샘플링
		df_end_of_reasoning = df[df['guidance_gpt'].apply(lambda x: "[END_OF_REASONING]" in x)]
		print(f"[INFO] 'END_OF_REASONING' 포함된 데이터 수: {len(df_end_of_reasoning)}")
		df_others = df[~df['guidance_gpt'].apply(lambda x: "[END_OF_REASONING]" in x)]
		df_others_sampled = df_others.sample(n=200, random_state=42)
		df = pd.concat([df_end_of_reasoning, df_others_sampled]).reset_index(drop=True)
		print(f"[INFO] 최종 데이터셋 크기: {len(df)}")
		
		processed_data = []
		
		for _, row in df.iterrows():
			# 1) User Input 생성
			user_input = construct_user_input(row)
			
			# 2) Chosen (Good Data) 생성 -> GPT 결과 사용
			# Feedback 포맷: "Diagnosis: ... Guidance: ..."
			chosen_feedback_text = f"Diagnosis: {row['diagnosis_gpt']} Guidance: {row['guidance_gpt']}"
			chosen_json = json.dumps({
				"error_type": row['error_type_gpt'],
				"feedback": chosen_feedback_text
			}, ensure_ascii=False)
			
			# 3) Rejected (Bad Data) 생성 -> 기존 모델 결과 사용
			rejected_feedback_text = f"Diagnosis: {row['diagnosis']} Guidance: {row['guidance']}"
			rejected_json = json.dumps({
				"error_type": row['error_type'],
				"feedback": rejected_feedback_text
			}, ensure_ascii=False)
			
			processed_data.append({
				"input": user_input,
				"chosen": chosen_json,
				"rejected": rejected_json
			})
			
		# HuggingFace Dataset으로 변환
		full_dataset = Dataset.from_list(processed_data)
		
		# DPO 포맷 매핑 (Chat Template 적용을 위해)
		formatted_dataset = full_dataset.map(format_dpo_data)
		
		# if local_rank == 0:
		#     print(f"✅ DPO 데이터셋 준비 완료. 총 {len(formatted_dataset)}쌍.")
		print(f"✅ DPO 데이터셋 준비 완료. 총 {len(formatted_dataset)}쌍.")

		# 데이터 분할 (Train 90% / Test 10%)
		dataset_split = formatted_dataset.train_test_split(test_size=0.1, seed=42)
		train_dataset = dataset_split['train']
		eval_dataset = dataset_split['test']

	except Exception as e:
		print(f"❌ 데이터셋 처리 실패: {e}")
		return

	# Accelerator 초기화 (분산 학습 환경 감지)
	accelerator = Accelerator()
	
	# 현재 프로세스의 로컬 랭크 가져오기
	local_rank = accelerator.local_process_index

	# --- 3. 모델 로드 (4-bit QLoRA) ---
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.bfloat16,
		bnb_4bit_use_double_quant=False,
	)

	# 현재 프로세스가 사용할 GPU 명시적으로 지정
	device_map = {"": local_rank} 

	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		quantization_config=bnb_config,
		device_map=device_map,
		trust_remote_code=True,
		use_cache=False
	)
	
	# [필수 추가] 이 코드가 있어야 "Gradients will be None" 경고가 사라지고 실제 학습이 됩니다.
	model.enable_input_require_grads() 

	# gradient_checkpointing 활성화 (이미 config에 있다면 생략 가능)
	model.gradient_checkpointing_enable()
	
	tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
	tokenizer.pad_token = tokenizer.eos_token
	# DPO나 Right-padding이 일반적이지만, 모델/라이브러리 버전에 따라 다를 수 있음.
	# Qwen2.5는 보통 left padding 없이도 잘 동작하나, 학습 시엔 right padding 권장
	tokenizer.padding_side = "right" 

	# --- 4. LoRA 설정 ---
	peft_config = LoraConfig(
		r=64,
		lora_alpha=128,
		lora_dropout=0.05,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules=[
			"q_proj", "k_proj", "v_proj", "o_proj",
			"gate_proj", "up_proj", "down_proj",
		],
	)

	# --- 5. DPO Config ---
	dpo_config = DPOConfig(
		output_dir=output_dir,
		per_device_train_batch_size=2,   # 메모리 절약
		per_device_eval_batch_size=2,
		gradient_accumulation_steps=1,   # 배치 크기 보정
		gradient_checkpointing=True,
		optim="adamw_torch",
		
		learning_rate=1e-6,              # SFT보다 낮게 설정
		num_train_epochs=1,
		beta=0.2,                        # DPO Temperature
		
		eval_strategy="steps",
		eval_steps=5,
		save_steps=5,
		logging_steps=1,
		
		warmup_ratio=0.1,
		lr_scheduler_type="cosine",
		fp16=False,
		bf16=True,
		report_to="tensorboard",
		ddp_find_unused_parameters=False,
		
		max_length=4096,
		max_prompt_length=3072,
		
		# [중요] remove_unused_columns=False 설정을 해야 map에서 만든 컬럼들이 날아가지 않음
		remove_unused_columns=False 
	)

	# --- 6. Trainer ---
	trainer = DPOTrainer(
		model=model,
		ref_model=None,         # PEFT 사용 시 None 권장 (자동으로 adapter disable/enable 하며 reference 역할 수행)
		args=dpo_config,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		processing_class=tokenizer,
		peft_config=peft_config,
	)

	# if local_rank == 0:
	#     print("🚀 DPO 학습을 시작합니다...")
	print("🚀 DPO 학습을 시작합니다...")
		
	trainer.train()

	# if local_rank == 0:
	#     print("✅ DPO 학습 완료!")
	#     trainer.save_model(output_dir)
	#     tokenizer.save_pretrained(output_dir)
	print("✅ DPO 학습 완료!")
	trainer.save_model(output_dir)
	tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
	main()