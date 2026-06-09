import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import json
import argparse
import os
import ast
import torch
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def load_generator_model(model_id: str):
	print(f"Generator 모델 로딩 중... Model: '{model_id}'")
	
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		device_map="auto",
		torch_dtype=torch.bfloat16
	)
	
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	tokenizer.pad_token = tokenizer.eos_token
	
	print("✅ Generator 모델 로드 완료.")
	return model, tokenizer

def evaluate_single_step(question, context, previous_steps, current_step):
	previous_steps_str = "\n".join(previous_steps)

	user_prompt = f"""### Evaluate the following:

Question: {question}

Retrieved Passages:
{context}

PREVIOUS STEPS:
{previous_steps_str}

STEP TO EVALUATE:
{current_step}
""".strip()

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
Your feedback must include two parts:
	- **Diagnosis:** Explain *why* the step is correct or erroneous based on the specific definition above.
	- **Guidance:** - If **Correct**: Briefly confirm the finding and suggest the logical next step (e.g., "Now that you've found [Entity], look for [Next Info]...").
	 - If **Error**: Explicitly point out the mistake and guide the user on what they *should* have done or looked for instead to advance the reasoning correctly.

### Output Requirements:
Always output a JSON object with `"error_type"` and `"feedback"` keys:
`"error_type"` — The one error category selected.
`"feedback"` — The diagnosis and guidance text.
""".strip()

	try:
		# completion = client.chat.completions.create(
		# 	model="gpt-4o-mini",
		# 	messages=[
		# 			{"role": "system", "content": system_prompt},
		# 			{"role": "user", "content": user_prompt},
		# 	],
		# 	max_tokens=256,
		# 	temperature=0,
		# )
  
		completion = client.chat.completions.create(
			model="gpt-5.1",
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			reasoning_effort="medium",
			max_output_tokens=256
		)

		response = completion.choices[0].message.content.strip()
		
		# 토큰 정보 추출
		input_tokens = completion.usage.prompt_tokens
		output_tokens = completion.usage.completion_tokens

		# JSON 추출
		match = re.search(r'\{.*\}', response, re.DOTALL)
		if match:
			json_str = match.group(0)
			parsed = json.loads(json_str)
			if "error_type" not in parsed: parsed["error_type"] = "Unknown"
			if "feedback" not in parsed: parsed["feedback"] = ""
			return parsed

		return {"error_type": "Parsing Error", "feedback": f"Raw output: {response}"}, input_tokens, output_tokens

	except Exception as e:
		print(f"⚠️ OpenAI API Error: {e}")
		return {"error_type": "API Error", "feedback": str(e)}, input_tokens, output_tokens

def generate_single_step(
	query: str, 
	retrieved_passages: list, 
	previous_steps: list, 
	last_feedback: dict, 
	generator_model, 
	generator_tokenizer
) -> str:
	"""
	Generator 모델을 사용하여 다음 한 단계의 추론을 생성합니다.
	"""
	system_prompt = f"""You are a meticulous, step-by-step logical reasoner. Your task is to solve a complex question by generating **ONLY THE NEXT SINGLE, ATOMIC STEP** in a chain of thought.

## Core Task Definition
You must analyze the `Question`, `Retrieved Passages`, `Previous Reasoning Steps`, and critically, the `Error Type` and `Feedback` on the last step.

Determine your action based on the following **Logic Flow**:

### 1. Feedback & Error Analysis (CRITICAL PRIORITY)
If the previous step received an Error, you **MUST** change your approach based on the specific `Error Type`:

- **Off-topic**: The step is irrelevant to the overall goal of the question and the specific step it is replacing. It introduces a new, unrelated piece of information or inference that leads the reasoning process astray.
- **Redundancy**: The step repeats information or conclusions from previous steps without providing any significant new progression. It stalls the reasoning process by repeating what is already known.
- **Overthinking**: The step continues *after* the reasoning is sufficient to answer the question. It introduces a new, *unnecessary* line of reasoning that is no longer required to find the final answer.
- **Inefficiency**: The step provides meta-discussion, procedural intent, planning statements, or placeholder reasoning instead of executing meaningful inference. It does not extract evidence or logically reason but merely describes what the model plans to do rather than doing it.
- **Logical Fallacy**: The step contains a flawed reasoning process. The facts gathered from previous steps are correct, but the conclusion drawn from them is incorrect.
- **Unsupported**: The step makes a factual claim using information that **cannot** be found in any of the `Retrieved Passages`. The step hallucinates a new, false piece of information and presents it as fact.
- **Contradictory**: The step makes a factual claim that **directly conflicts** or **contradicts** information explicitly stated in the `Retrieved Passages`.
- **Information Miss**: The step incorrectly concludes that specific information is unavailable, unknown, or missing—even though the information is present in the retrieved passages. The failure lies in not recognizing or retrieving relevant evidence that already exists.

For correct reasoning steps, there will be Correct (No Error).
- **Correct (No Error)**: The step is logically sound, fully supported by the retrieved passages, and moves the reasoning forward correctly.

### 2. Termination Check (Can I Finish?)
- **Condition**: IF the last step was a **Logical Step** (ends with `(Logical)`) AND it explicitly stated the final answer...
- **Action**: Output `[Reasoning Finished]`.
- **Constraint**: You **MUST NOT** output `[Reasoning Finished]` if the last step was an **Attribution Step**. Even if you know the answer, you must generate a **Logical Step** to explicitly state the conclusion.

### 3. Continuation Logic (What to generate next?)
If you cannot stop, determine the next step based on the current state:

- **Case A: Sufficient Information Gathered**
  - If you have gathered all necessary facts from previous Attribution Steps to answer the question:
  - **Action**: Generate a **Final Logical Step** that combines these facts to explicitly state the answer.

- **Case B: Insufficient Information**
  - If you still need more facts or intermediate deductions:
  - **Action**: Generate the next necessary step (this could be another **Attribution Step** to find new facts, or an intermediate **Logical Step** to process current facts).

---

## Step Classifications
Every step you generate must be strictly classified into one of two types:

### 1. Attribution Step
- **Definition**: Extracts exactly ONE explicit fact from a **single** retrieved passage.
- **Requirement**: You MUST explicitly cite the source passage (e.g., "According to Passage X...", "Passage X states...", "As seen in Passage X...", etc.)
- **Constraint**: Do not combine information from multiple passages. Do not make external inferences.
- **Format suffix**: End the sentence with `(Attribution)`.

### 2. Logical Step
- **Definition**: Performs a logical operation (comparison, calculation, bridging, or concluding) based **ONLY** on information found in `Previous Reasoning Steps`.
- **Requirement**: Do NOT look up new information from passages. Use what you have already found.
- **Format suffix**: End the sentence with `(Logical)`.

---

## Strict Formatting Rules
1. **Numbering**: Start your response with `Step K:`, where `K` is the next integer after the last step number.
2. **Atomic Nature**: One step = One action. Do not combine an Attribution and a Logical inference in the same step.
3. **Suffix Mandatory**: Every step must end with either `(Attribution)` or `(Logical)`.
4. **Termination**: If you output `[Reasoning Finished]`, do **not** output any other text or step number.

---

## Examples of Valid Steps

### Attribution Step Examples
- **Example 1**: 
  Step 1: According to Passage 3, the director of the film "Inception" is Christopher Nolan. (Attribution)
- **Example 2**: 
  Step 2: Passage 1 states that World War I ended on November 11, 1918. (Attribution)
- **Example 3**: 
  Step 4: The father of King George VI was King George V, as mentioned in Passage 5. (Attribution)

### Logical Step Examples
- **Example 1**: 
  Step 3: Since Step 1 identified the director as Christopher Nolan, and Step 2 identified his wife as Emma Thomas, the person being sought is Emma Thomas. (Logical)
- **Example 2**: 
  Step 5: Comparing the dates from Step 3 (1918) and Step 4 (1939), the start of World War II was later than the end of World War I. (Logical)
- **Example 3**: 
  Step 6: Based on the evidence in Step 5, the capital city Paris is the answer. (Logical)
""".strip()
	
	passages_str = '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(retrieved_passages)])
	previous_steps_str = '\n'.join(previous_steps)

	# --- Construct Feedback String with Error Type ---
	feedback_str = ""
	if not last_feedback:
		feedback_str = "Status: First Step. No feedback yet."
	else:
		error_type = last_feedback.get("error_type", "Unknown")
		feedback_text = last_feedback.get("feedback", "No feedback provided.")
		is_error = error_type != "Correct (No Error)"
		
		if not is_error:
			feedback_str = f"Status: Correct\nFeedback: {feedback_text}"
		else:
			feedback_str = f"Status: Error Detected\nError Type: {error_type}\nFeedback: {feedback_text}"
			
	prompt_user = f"""Question: {query}

Retrieved Passages:
{passages_str}

Previous Reasoning Steps:
{previous_steps_str}

Feedback on Last Step:
{feedback_str}

Generate next step (start with `Step {len(previous_steps) + 1}:`)
"""

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": prompt_user}
	]

	try:
		# Chat Template 적용
		input_ids = generator_tokenizer.apply_chat_template(
			messages,
			add_generation_prompt=True,
			return_tensors="pt"
		).to(generator_model.device)

		terminators = [generator_tokenizer.eos_token_id]

		# 추가 안전 처리
		eot_id = generator_tokenizer.convert_tokens_to_ids("<|eot_id|>")
		if eot_id is not None:
			terminators.append(eot_id)

		if generator_tokenizer.pad_token_id is None:
			generator_tokenizer.pad_token_id = generator_tokenizer.eos_token_id

		outputs = generator_model.generate(
			input_ids,
			attention_mask=input_ids.ne(generator_tokenizer.pad_token_id).long(),      
			pad_token_id=generator_tokenizer.pad_token_id,
			max_new_tokens=256,
			eos_token_id=terminators,
			do_sample=False,
		)

		response = generator_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

		# 출력 정제 (찌꺼기 제거 및 포맷 확인)
		dirty_markers = [
			"<start_of_turn>", "</start_of_turn>",
			"<end_of_turn>", "</end_of_turn>",
			"User:", "## Question"
		]
		
		for marker in dirty_markers:
			if marker in response:
				response = response.split(marker)[0].strip()

		# 스텝 번호 강제 보정
		expected_start = f"Step {len(previous_steps) + 1}:"
		if not response.startswith(expected_start):
			if not response.startswith("Step"):
				print(f"⚠️ Generated step missing prefix. Adding '{expected_start}'")
				response = f"{expected_start} " + response.lstrip()
			elif response.startswith("Step"):
					response = response.split('\n')[0]

		return response
		
	except Exception as e:
		print(f"LLM step generation error: {e}")
		return ""

def generate_and_evaluate_iteratively(
	query: str, 
	retrieved_passages: list, 
	generator_model,
	generator_tokenizer,
	max_steps: int = 7,
	max_retries: int = 3
) -> tuple[list, list, list]:

	total_input_tokens, total_output_tokens = 0, 0

	# [수정] 로그 구조를 딕셔너리로 변경하여 질문과 지문 정보를 포함
	full_logs = {
		"meta_data": {
			"question": query,
			"retrieved_passages": retrieved_passages
		},
		"steps_history": [] # 여기에 스텝별 상세 기록이 들어감
	}
	step_texts = []      # "검증된" 스텝들만 저장하는 리스트 (Clean History)
	feedback_list = []   # 각 스텝의 최종 피드백 저장

	print(f"🚀 Reasoning Start: {query}")

	# 전체 추론 단계 루프 (Step 1 -> Step 2 -> ...)
	while len(step_texts) < max_steps:
		
		if total_input_tokens + total_output_tokens > 950000:
			print(f"⛔ Token budget exceeded during retry ({total_input_tokens + total_output_tokens}). Stopping.")
			full_logs["meta_data"]["stop_reason"] = "Token Limit Exceeded"
			return step_texts, feedback_list, full_logs
  
  
		current_step_num = len(step_texts) + 1
		current_retry = 0
		last_feedback = None  # 새 스텝 시작 시 피드백 초기화
		step_accepted = False # 현재 스텝 승인 여부
		
		# 현재 스텝을 위한 로그 컨테이너 생성
		current_step_log = {
			"step_num": current_step_num,
			"status": "In Progress",
			"attempts": []
		}
		
		# [Inner Loop] 올바른 스텝이 나올 때까지 재시도
		while current_retry < max_retries:
			print(f"Total tokens used: {total_input_tokens} + {total_output_tokens} = {total_input_tokens + total_output_tokens}")
			if total_input_tokens+total_output_tokens > 950000:
				full_logs["meta_data"]["stop_reason"] = "Token Limit Exceeded"
				return step_texts, feedback_list, full_logs
			
			# 1. 생성 (Generator)
			# 중요: step_texts에는 "오류 스텝"이 들어있지 않음 (Clean Context)
			# 재시도 중일 경우 last_feedback에 오류 내용이 들어있음
			next_step_text = generate_single_step(
				query, 
				retrieved_passages, 
				step_texts, 
				last_feedback, 
				generator_model, 
				generator_tokenizer
			)
			
			if not next_step_text:
				print("⚠️ Generator returned empty response.")
				break

			# 문맥 구성 (Evaluator용)
			context_str = 'Retrieved Passages:\n' + '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(retrieved_passages)])
			
			# 2. 평가 (Evaluator)
			# 중요: Evaluator에게도 "Clean History" + "Current Candidate"를 보여줌
			evaluation_result, input_tokens, output_tokens = evaluate_single_step(
				query, 
				context_str, 
				step_texts, 
				next_step_text
			)
			
			error_type = evaluation_result.get("error_type", "Unknown")
			feedback_msg = evaluation_result.get("feedback", "")

			total_input_tokens += input_tokens
			total_output_tokens += output_tokens

			# 시도 기록 저장
			attempt_record = {
				"retry_index": current_retry,
				"generated_text": next_step_text,
				"evaluation": evaluation_result, # error_type, feedback 포함
				"result": "Pending" # 나중에 Accepted/Rejected로 업데이트
			}

			# 3. 판단 로직
			if error_type == "Correct (No Error)":
				print(f"✅ Step {current_step_num} Accepted: {error_type}")
				
				# 로그 업데이트
				attempt_record["result"] = "Accepted"
				current_step_log["attempts"].append(attempt_record)
				current_step_log["status"] = "Completed"
				
				# [승인] 역사에 기록 (Commit)
				step_texts.append(next_step_text)
				feedback_list.append(evaluation_result)
				step_accepted = True
				break # 재시도 루프 탈출
			
			else:
				print(f"🔄 Step {current_step_num} Retry ({current_retry+1}/{max_retries}): {error_type} -> Rolling back...")
				
				# 로그 업데이트
				attempt_record["result"] = "Rejected (Rollback)"
				current_step_log["attempts"].append(attempt_record)
				
				# [거절] 역사에 기록하지 않음 (Rollback)
				# 다음 생성을 위한 피드백만 업데이트
				last_feedback = evaluation_result
				current_retry += 1
				
				# 만약 재시도 횟수가 남았다면, 루프의 처음으로 돌아가서 generate_single_step 다시 호출

		# Inner Loop 종료 후 처리
		if not step_accepted:
			print(f"⚠️ Step {current_step_num}: Max retries reached. Forcing progression.")
			current_step_log["status"] = "Max retries"
			# 정책 결정: 계속 틀리면 그냥 틀린 채로 갈 것인가, 멈출 것인가?
			# 여기서는 틀린 채로라도 추가하고 진행하는 방식(Lenient)을 적용
			if next_step_text:
				if current_step_log["attempts"]:
					current_step_log["attempts"][-1]["result"] = "Max retries"
				step_texts.append(next_step_text)
				feedback_list.append(last_feedback if last_feedback else evaluation_result)
			else:
				current_step_log["status"] = "Failed (Empty)"
				full_logs["steps_history"].append(current_step_log) # 실패했더라도 로그 저장하고 종료
				break 

		# 현재 스텝의 로그를 전체 로그에 추가
		full_logs["steps_history"].append(current_step_log)
		
		# ---------------------------------------------------------
		# 종료 조건 강화 (Generator OR Evaluator Signal)
		# ---------------------------------------------------------
		
		# 1. Generator가 스스로 종료를 선언했는지 확인
		generator_signal = step_texts and "[Reasoning Finished]" in step_texts[-1]
		
		# 2. Feedback Model이 종료를 지시했는지 확인
		evaluator_signal = False
		if feedback_list:
			last_eval_result = feedback_list[-1]
			last_error_type = last_eval_result.get("error_type", "")
			last_feedback_msg = last_eval_result.get("feedback", "").lower()
			
			# 종료 트리거 문구들 정의
			stop_phrases = [
				"the reasoning process is successfully completed",
				"the reasoning process is compeleted",
				"do not generate any further steps",
				"you must stop reasoning",
				"you should stop reasoning",
				"stop reasoning now",
				"reasoning is complete",
				"reasoning is finished",
	 
			]
			
			# 조건: Error Type이 Correct이면서, 피드백 메시지에 종료 문구가 포함된 경우
			if last_error_type == "Correct (No Error)" and any(phrase in last_feedback_msg for phrase in stop_phrases):
				evaluator_signal = True

		# 둘 중 하나라도 만족하면 루프 종료
		if generator_signal or evaluator_signal:
			trigger_source = "Generator" if generator_signal else "Evaluator"
			print(f"🎉 Reasoning Process Finished. (Stopped by {trigger_source})")
			break 

	if len(step_texts) == max_steps:
		print(f"⚠️ 최대 단계({max_steps})에 도달하여 추론을 종료합니다.")

	return step_texts, feedback_list, full_logs



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, required=True)
	args = parser.parse_args()

	# 1. Generator 모델 초기화 (Llama 3.1 8B Instruct)
	generator_model_id = "meta-llama/Llama-3.1-8B-Instruct"
	# generator_model_id = "Qwen/Qwen2.5-7B-Instruct"
	gen_model, gen_tokenizer = load_generator_model(generator_model_id)

	if args.dataset == '2wiki':
		df = pd.read_csv("/workspace/daeyong/2wiki_dev.csv").sample(n=100, random_state=42)
	elif args.dataset == 'hotpotqa':
		df = pd.read_csv("/workspace/daeyong/hotpotqa_validation.csv").sample(n=100, random_state=42)
	elif args.dataset == 'musique':
		df = pd.read_csv("/workspace/daeyong/musique_dev.csv").sample(n=100, random_state=42)
	else:
		print(f"❌ 알 수 없는 데이터셋: {args.dataset}")
		exit()

	full_logs = []
	response_list = []
	for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating and Evaluating"):
		query = row['question']
		context_source = row['retrieved_passages']
		
		if isinstance(context_source, str):
			try:
				retrieved_passages = ast.literal_eval(context_source)
			except (ValueError, SyntaxError):
				retrieved_passages = [context_source]
		elif isinstance(context_source, list):
			retrieved_passages = context_source
		else: retrieved_passages = []
		
		# 3. 단계별 생성 및 평가 동시 진행
		response_steps, feedback_list, logs = generate_and_evaluate_iteratively(
			query, 
			retrieved_passages, 
			gen_model, 
			gen_tokenizer,
			max_steps=7
		)
		
		response_list.append({
			"id": row['id'],
			"question": query,
			"context": 'Retrieved Passages:\n' + '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(retrieved_passages)]),
			"response": response_steps,
			"feedback": [f"Feedback for Step {i+1}: {f}" for i, f in enumerate(feedback_list)],
			"ground_truth": row.get('answer', 'N/A')
		})
  
		full_logs.append(logs)
		
		# 로그 기록 저장
		with open(f"/workspace/daeyong/ours_fourth_llama8b_{args.dataset}_logs_gpt.json", "w", encoding="utf-8") as f:
			json.dump(full_logs, f, ensure_ascii=False, indent=2)
  
		# 중간 저장
		output_filename = f"/workspace/daeyong/ours_fourth_llama8b_{args.dataset}_gpt.json"
		with open(output_filename, "w") as f:
			json.dump(response_list, f, ensure_ascii=False, indent=2)