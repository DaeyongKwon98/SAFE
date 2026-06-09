import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pandas as pd

# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

SYSTEM_PROMPT = """You are an expert reasoning evaluator and verifier.

You will receive:
1. Question: The user's original question.
2. Retrieved Passages: Contextual information.
3. Ideal Reasoning Steps: The *correct, full* reasoning path to the answer.
4. Previous Steps: The reasoning steps before the current step.
5. Current Step: The current reasoning step with no error.

Your task:
1. Identify the position of the current step in the ideal reasoning steps sequence.
2. Provide one clear guidance sentence describing what the next reasoning step should be.
3. If the current step is already the final step, tell the student to produce the final answer.

Output *only* the feedback text, and nothing else.

---
Examples
---

Question: When did Princess Alexandrine Of Baden's mother die?

Retrieved Passages:
Passage 1: Princess Sophie of Sweden (born 21 May 1801 – died 6 July 1865)
Passage 2: Princess Alexandrine of Baden is the daughter of Princess Sophie of Sweden.

Ideal Reasoning Steps:
[
 "Step 1. According to Passage 2, the mother of Princess Alexandrine Of Baden is Princess Sophie of Sweden. (Attribution)",
 "Step 2. According to Passage 1, the date of death of Princess Sophie of Sweden is 6 July 1865. (Attribution)",
 "Step 3. Therefore, 6 July 1865 is the date of death of Princess Alexandrine Of Baden's mother. (Logical)"
]

Previous Steps:
[
 "Step 1. According to Passage 2, the mother of Princess Alexandrine Of Baden is Princess Sophie of Sweden. (Attribution)",
 "Step 2. According to Passage 1, the date of death of Princess Sophie of Sweden is 6 July 1865. (Attribution)"
]

Current Step:
Step 3: Therefore, 6 July 1865 is the date of death of Princess Alexandrine Of Baden's mother. (Logical)

Feedback:
You have successfully completed the reasoning process by logically connecting the mother identified in Step 1 to the death date retrieved in Step 2 to reach the correct final answer.
---

Question: Journey to the Center of the Earth starred a former Metro-Goldwyn-Mayer contract star who is the mother of what actor?

Retrieved Passages: 
Passage 1: Journey to the Center of the Earth (1959 film): Journey to the Center of the Earth (also called Jules Verne's Journey to the Center of the Earth) is a 1959 adventure film adapted by Charles Brackett from the novel of the same name by Jules Verne. "Journey to the Center of the Earth" was directed by Henry Levin and stars James Mason, Pat Boone and Arlene Dahl. 
Passage 2: Arlene Dahl: Arlene Carol Dahl (born August 11, 1925) is an American actress and former Metro-Goldwyn-Mayer contract star, who achieved notability during the 1950s. She has three children, the eldest of whom is actor Lorenzo Lamas. Passage 3: Laraine Day: Laraine Day (October 13, 1920 – November 10, 2007) was an American actress and a former Metro-Goldwyn-Mayer contract star.

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the former Metro-Goldwyn-Mayer contract star who starred in "Journey to the Center of the Earth" is Arlene Dahl. (Attribution)", 
 "Step 2: According to Passage 2, one of the children of Arlene Dahl is actor Lorenzo Lamas. (Attribution)", 
 "Step 3: Therefore, the child found in Step 2, Lorenzo Lamas, is the answer. (Logical)" 
]

Previous Steps: 
[]

Current Step: 
Step 1: According to Passage 1, the former Metro-Goldwyn-Mayer contract star who starred in "Journey to the Center of the Earth" is Arlene Dahl. (Attribution)

Feedback: 
You have correctly identified Arlene Dahl as the star; in the next step, please find the name of her actor son.

---
Question: What record label does the performer of What'cha Gonna Do About It belong to?

Retrieved Passages: 
Passage 1: What'cha Gonna Do About It: What'cha Gonna Do About It is a 1964 song by Doris Troy. It made #37 on the UK Singles Chart in 1964 and #38 in 1965. 
Passage 2: Doris Troy (album): Doris Troy is an album released in 1970 on the Beatles' Apple Records label by American soul singer Doris Troy. 

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the performer of "What'cha Gonna Do About It" is Doris Troy. (Attribution)", 
 "Step 2: According to Passage 2, the record label of Doris Troy is Apple Records. (Attribution)", 
 "Step 3: Therefore, the record label found in Step 2, Apple Records, is the answer. (Logical)"
]

Previous Steps: 
[ 
 "Step 1: According to Passage 1, the performer of "What'cha Gonna Do About It" is Doris Troy. (Attribution)"
]

Current Step: 
Step 2: According to Passage 2, Doris Troy's album was released on the Apple Records label. (Attribution)

Feedback: 
The current step have successfully identified the record label as Apple Records; now, please proceed to state the record label as the final answer.
""".strip()


SYSTEM_PROMPT = """You are an expert reasoning evaluator and verifier.

You will receive:
1. Question: The user's original question.
2. Ideal Reasoning Steps: The *correct, full* reasoning path to the answer.
3. Previous Steps: The reasoning steps before the current step.
4. Current Step: The current reasoning step with no error.

Your task:
1. Identify the position of the current step in the ideal reasoning steps sequence.
2. If the current step corresponds to the **FINAL step** of the ideal reasoning (meaning the answer is explicitly found):
	- You MUST output a **Strong Termination Command**.
	- Explicitly state that the reasoning is complete.
	- **Command to STOP reasoning immediately**.
	- Do NOT ask them to "state the answer".

3. If the current step is **NOT** the final step:
	- Provide one clear guidance sentence describing what the next reasoning step should be.
	- You MUST NOT include the specific passage number in your feedback. (e.g., do NOT say "find ... from Passage 2")

Output *only* the feedback text, and nothing else.

---
Examples
---

Question: When did Princess Alexandrine Of Baden's mother die?

Ideal Reasoning Steps:
[
 "Step 1. According to Passage 2, the mother of Princess Alexandrine Of Baden is Princess Sophie of Sweden. (Attribution)",
 "Step 2. According to Passage 1, the date of death of Princess Sophie of Sweden is 6 July 1865. (Attribution)",
 "Step 3. Therefore, 6 July 1865 is the date of death of Princess Alexandrine Of Baden's mother. (Logical)"
]

Previous Steps:
[
 "Step 1. According to Passage 2, the mother of Princess Alexandrine Of Baden is Princess Sophie of Sweden. (Attribution)",
 "Step 2. According to Passage 1, the date of death of Princess Sophie of Sweden is 6 July 1865. (Attribution)"
]

Current Step:
Step 3: Therefore, 6 July 1865 is the date of death of Princess Alexandrine Of Baden's mother. (Logical)

Feedback:
You have successfully completed the reasoning process by finding the death date of Princess Alexandrine Of Baden's mother as 6 July 1865. Do not generate any further steps. You must stop reasoning immediately.

---

Question: Journey to the Center of the Earth starred a former Metro-Goldwyn-Mayer contract star who is the mother of what actor?

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the former MGM star in "Journey to the Center of the Earth" is Arlene Dahl. (Attribution)", 
 "Step 2: According to Passage 2, the actor son of Arlene Dahl is Lorenzo Lamas. (Attribution)", 
 "Step 3: Therefore, the actor found in Step 2, Lorenzo Lamas, is the answer. (Logical)" 
]

Previous Steps: 
[]

Current Step: 
Step 1: According to Passage 1, the former Metro-Goldwyn-Mayer contract star who starred in "Journey to the Center of the Earth" is Arlene Dahl. (Attribution)

Feedback: 
You have correctly identified Arlene Dahl as the MGM star; in the next step, please find the name of her actor son.

---

Question: What record label does the performer of What'cha Gonna Do About It belong to?

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the performer is Doris Troy. (Attribution)", 
 "Step 2: According to Passage 2, the record label of Doris Troy is Apple Records. (Attribution)", 
 "Step 3: Therefore, the record label found in Step 2, Apple Records, is the answer. (Logical)"
]

Previous Steps: 
[ 
 "Step 1: According to Passage 1, the performer is Doris Troy. (Attribution)",
 "Step 2: According to Passage 2, the record label of Doris Troy is Apple Records. (Attribution)"
]

Current Step: 
Step 3: Therefore, the record label found in Step 2, Apple Records, is the answer. (Logical)

Feedback: 
The reasoning is complete. You have identified the answer as Apple Records. Stop reasoning now and terminate.

---

Question: "Which film whose director is younger, Skoplje '63 or Abused Confidence?"

Ideal Reasoning Steps:
[
 "Step 1: According to Passage 6, the director of the film Skoplje '63 is Veljko Bulajić. (Attribution)",
 "Step 2: According to Passage 2, the birth date of Veljko Bulajić (from Step 1) is 22 March 1928. (Attribution)",
 "Step 3: According to Passage 7, the director of the film Abused Confidence is Henri Decoin. (Attribution)",
 "Step 4: According to Passage 1, the birth date of Henri Decoin (from Step 3) is 18 March 1890. (Attribution)",
 "Step 5: Since 1928 (from Step 2) is later than 1890 (from Step 4), Veljko Bulajić is younger than Henri Decoin, so the film Skoplje '63 has a younger director. (Logical)"
]

Previous Steps:
[
 "Step 1: According to Passage 6, the director of the film Skoplje '63 is Veljko Bulajić. (Attribution)",
 "Step 2: According to Passage 2, the birth date of Veljko Bulajić (from Step 1) is 22 March 1928. (Attribution)",
 "Step 3: According to Passage 7, the director of the film Abused Confidence is Henri Decoin. (Attribution)"
]

Current Step:
Step 4: According to Passage 1, the birth date of Henri Decoin (from Step 3) is 18 March 1890. (Attribution)

Feedback:
You have correctly identified the birth date of Henri Decoin; in the next step, please compare the birth dates of the two directors to determine which one is younger.
""".strip()


def init_model(model_name: str):
	print(f"Loading model: {model_name}")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		dtype=torch.bfloat16,
		device_map="auto"
	)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id
	return tokenizer, model



def generate_feedback(tokenizer, model, item):
	"""Generate feedback for correct reasoning step (next-step guidance)."""

	question = item["question"]
	ideal_steps = item["ideal_steps"]
	previous_steps = item.get("previous_steps", [])
	current_step = item["current_step"]

	user_prompt = f"""Question: {question}

Ideal Reasoning Steps:
{json.dumps(ideal_steps, indent=2, ensure_ascii=False)}

Previous Steps:
{json.dumps(previous_steps, indent=2, ensure_ascii=False)}

Current Step:
{json.dumps(current_step, indent=2, ensure_ascii=False)}
""".strip()

	messages = [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": user_prompt}
	]

	input_ids = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		tokenize=True,
		return_tensors="pt"
	).to(model.device)
	
	# input_ids와 동일한 device에 attention_mask 생성 (생성 품질 안정성을 위해 권장)
	attention_mask = torch.ones_like(input_ids)

	with torch.no_grad():
		output = model.generate(
			input_ids=input_ids,          
			attention_mask=attention_mask,
			max_new_tokens=1024,
			pad_token_id=tokenizer.eos_token_id,
			do_sample=False
		)
	
	answer = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
	
	return answer.split("assistantfinal")[-1].strip()


def run_feedback_generation(df, model_name="/workspace/hf_transformers/gpt-oss-120b", save_path=None):
	# 1. 기존 진행상황 확인 및 로드 (Resume Logic)
	if save_path and os.path.exists(save_path):
		print(f"Found existing progress file at {save_path}. Resuming...")
		try:
			# 저장된 파일을 불러옵니다.
			saved_df = pd.read_json(save_path)
			
			# 인덱스를 기준으로 기존 진행된 feedback 내용을 메인 df에 병합합니다.
			# 데이터 순서가 바뀌지 않았다고 가정합니다.
			if 'feedback' in saved_df.columns:
				# feedback 컬럼이 없다면 생성
				if 'feedback' not in df.columns:
					df['feedback'] = None
				
				# 저장된 데이터의 feedback을 현재 df에 업데이트
				# update를 사용하거나 인덱스 매핑을 합니다. 여기서는 단순 대입
				# (주의: 입력 json과 저장된 json의 행 개수와 순서가 동일해야 합니다)
				df.update(saved_df[['feedback']])
				
				processed_count = df['feedback'].notna().sum()
				print(f"Resumed! {processed_count}/{len(df)} items already processed.")
		except ValueError as e:
			print(f"Error loading existing file: {e}. Starting from scratch.")
			df['feedback'] = None
	else:
		print("Starting from scratch.")
		df['feedback'] = None

	# 모델 로드 (이미 다 처리된 경우 불필요하게 로드하지 않도록 체크 가능하지만, 여기선 단순화)
	# 만약 100% 완료되었다면 함수 종료
	if df['feedback'].notna().all():
		print("All items are already processed!")
		return df

	# tokenizer, model = init_model(model_name)
	tokenizer, model = init_model(model_name)

	# 2. Loop 실행 (Skip Logic)
	for idx in tqdm(range(len(df)), desc="Generating next-step reasoning feedback"):
		# 이미 피드백이 존재하는 경우 건너뜀 (Skip)
		if pd.notna(df.at[idx, 'feedback']) and df.at[idx, 'feedback'] != "":
			continue

		try:
			row = df.iloc[idx]
			feedback = generate_feedback(tokenizer, model, row.to_dict())
			df.at[idx, 'feedback'] = feedback

			# 중간 저장
			if idx % 10 == 0 and save_path:
				df.to_json(save_path, orient="records", indent=2, force_ascii=False)
				
		except Exception as e:
			print(f"Error at index {idx}: {e}")
			# 에러 발생 시에도 저장을 시도할 수 있음
			if save_path:
				df.to_json(save_path, orient="records", indent=2, force_ascii=False)

	# 최종 저장
	if save_path:
		df.to_json(save_path, orient="records", indent=2, force_ascii=False)

	return df


if __name__ == "__main__":
	MODEL_PATH = "/workspace/hf_transformers/gpt-oss-120b"
	OUTPUT_FILE = "/workspace/daeyong/feedback/correct_feedback_morethan4_musique.json"
	  
	df = pd.read_json("/workspace/daeyong/ideal_steps/combined_correct_steps_paraphrased.json")
	# --- 실행 ---
	result_df = run_feedback_generation(df, model_name=MODEL_PATH, save_path=OUTPUT_FILE)
	
	# --- 결과 확인 ---
	print("\n--- Generation Example ---")
	print(result_df[['current_step', 'feedback']].head())