import argparse
import pandas as pd
import json
import random
import re
import ast

DEFAULT_INPUT_PATH = "/workspace/daeyong/fourth_finetuning_data/training_data_with_noises.json"


def parse_args():
	parser = argparse.ArgumentParser(
		description="Convert evaluator training records from JSON to JSONL."
	)
	parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH)
	parser.add_argument(
		"--output_path",
		default=None,
		help="Defaults to input_path with the .json suffix replaced by .jsonl.",
	)
	return parser.parse_args()

def delete_comma_at_the_end(df):
	df['previous_steps'] = df['previous_steps'].apply(lambda l: [x.strip(",") for x in l])
	df['current_step'] = df['current_step'].apply(lambda x: x.strip(","))
	return df

def get_paraphrased_text(text):
	"""
	입력된 텍스트와 인덱스(0~18)를 받아, 
	해당하는 단 하나의 Paraphrasing된 문자열을 반환하는 함수
	"""
	# 1. 인덱스 유효성 검사
	index = random.randint(0, 18)

	# 2. 정규표현식 파싱
	pattern = r"According to Passage (\d+), (.+?)\. \(Attribution\)"
	match = re.search(pattern, text)
	
	if not match:
		print(f"Error: 입력 텍스트 형식이 올바르지 않습니다. -> {text}")
		return None

	p_id = match.group(1)
	content = match.group(2)

	# 3. 사용할 Content 결정 (Group C인 13번부터는 대문자 변환 필요)
	if index >= 13:
		# 문장 앞으로 오므로 첫 글자 대문자 처리
		final_content = content[0].upper() + content[1:]
	else:
		# 문장 뒤에 오므로 원본 유지
		final_content = content

	# 4. 인덱스별 템플릿 정의 (Dictionary)
	templates = {
		# --- Group A: Preposition (전치사구) ---
		0: f"According to Passage {p_id}, {final_content}. (Attribution)",
		1: f"Based on Passage {p_id}, {final_content}. (Attribution)",
		2: f"Refer to Passage {p_id}, {final_content}. (Attribution)",
		3: f"As stated in Passage {p_id}, {final_content}. (Attribution)",
		4: f"From the information in Passage {p_id}, {final_content}. (Attribution)",
		5: f"As you can see in Passage {p_id}, {final_content}. (Attribution)",
		
		# --- Group B: Verb Phrases (주어+동사) ---
		6: f"Passage {p_id} states that {final_content}. (Attribution)",
		7: f"Passage {p_id} confirms that {final_content}. (Attribution)",
		8: f"Passage {p_id} reveals that {final_content}. (Attribution)",
		9: f"Passage {p_id} indicates that {final_content}. (Attribution)",
		10: f"It is mentioned in Passage {p_id} that {final_content}. (Attribution)",
		11: f"We can find in Passage {p_id} that {final_content}. (Attribution)",
		12: f"Looking at Passage {p_id}, we see that {final_content}. (Attribution)",

		# --- Group C: Inverted (도치/후수식) ---
		13: f"{final_content}, according to Passage {p_id}. (Attribution)",
		14: f"{final_content}, based on Passage {p_id}. (Attribution)",
		15: f"{final_content}, as shown in Passage {p_id}. (Attribution)",
		16: f"{final_content}, as stated in Passage {p_id}. (Attribution)",
		17: f"{final_content}, which is supported by Passage {p_id}. (Attribution)",
		18: f"{final_content}, as referencing Passage {p_id}. (Attribution)"
	}

	return templates[index]

def more_than_1_passage_filter(df):
	print("1. More than 1 passage filter")
	filter1 = []
	for i, row in df.iterrows():
		is_wrong=False
		for step in row["ideal_steps"]:
			passage_refs = []
			matches = re.findall(r"Passage\s+\d+", step)
			passage_refs.extend(matches)

			# Passage 언급이 2개 이상이면 제외
			if len(passage_refs) >= 2:
				is_wrong = True
				# print(f"prev Row {i}: {step}.")
				break

		if is_wrong:
			continue
		filter1.append(row)

	filter1_df = pd.DataFrame(filter1)
	print(f"{len(filter1_df)} after passage count filtering")
	return filter1_df

def says_no_information_filter(df):
	# filter_list = [
	# 	"do not mention", "does not mention", "do not provide", "does not provide",
	# 	"do not state", "does not state", "not provided", "not mentioned", "not stated", 
	# 	"not explicitly mentioned", "not explicitly provided", "not explicitly stated", 
	# 	"not present in", "no information in the provided", "not explicitly state", 
	# 	"not explicitly mention", "does not specify", "do not specify", "is not specified", 
	# 	"there is no explicit information", "there is no information",
	# 	"is not directly mentioned", "is not directly stated", "is not directly provided", 
	# 	"there's no direct information", "is not found in", "do not directly mention", "does not directly mention",
	# ]
	filter_list = []
	print("2. Says no information filtering")
	filter2 = []
	for i, row in df.iterrows():
		flag = False
		for step in row['ideal_steps']:
			step_lower = step.lower()
			# 하나라도 금지 문구가 포함되어 있으면 플래그 설정
			if any(phrase in step_lower for phrase in filter_list):
				flag = True
				break

		if not flag:
			filter2.append(row)

	filter2_df = pd.DataFrame(filter2)
	print(len(filter2_df))
	return filter2_df

def previous_steps_paraphrase_filter(df):
	rows = []
	for i, row in df.iterrows():
		new_previous_steps = []
		is_wrong = False
		for j, step in enumerate(row['previous_steps']):
			if step.endswith("(Logical)"):
				new_previous_steps.append(step)
				continue
			if get_paraphrased_text(step):
				new_previous_steps.append(f"Step {j+1}: {get_paraphrased_text(step)}")
			else:
				is_wrong = True
				break
		if is_wrong:
			# print(f"Warning: Some steps in row {i} could not be paraphrased correctly.")
			continue
		row['previous_steps'] = new_previous_steps
		rows.append(row)
	print(f"{len(rows)} after previous steps paraphrase filtering")
	return pd.DataFrame(rows)

def current_step_paraphrase_filter(df):
	for i, row in df.iterrows():
		if row['current_step'].endswith("(Logical)"):
			continue
		
		if get_paraphrased_text(row['current_step']):
			df.at[i, 'current_step'] = f"Step {j+1}: {get_paraphrased_text(row['current_step'])}"
			
	return df


def fix_step_index(df):
	for i, row in df.iterrows():
	
		if isinstance(row['previous_steps'], str):
			steps = ast.literal_eval(row['previous_steps'])
		else:
			steps = row['previous_steps']

		expected_step_num = len(steps) + 1
		expected_prefix = f"Step {expected_step_num}:"

		current_step_text = row['current_step']

		# 숫자 추출 시도
		try:
			current_k = int(current_step_text.split()[1].replace(":", ""))
		except:
			current_k = None

		# 잘못된 경우 = auto correct
		if current_k != expected_step_num or not current_step_text.startswith("Step "):

			if ":" in current_step_text:
				after_colon = current_step_text.split(":", 1)[1].strip()
				new_step = f"{expected_prefix} {after_colon}"
			else:
				new_step = expected_prefix

			df.at[i, "current_step"] = new_step

	print("✔ Step index correction 완료.")
	return df

# -----------------------------
# 2️⃣ 입력 프롬프트 생성 함수
# -----------------------------
def make_input_prompt(row):
	question = row["question"]
	context = row["retrieved_passages"]
	previous_steps = row["previous_steps"]
	current_step = row["current_step"]
	
	if isinstance(context, str):
		context = eval(context)
	if isinstance(previous_steps, str):
		previous_steps = eval(previous_steps)

	# context의 각 원소(passage text)앞에 Passage i: 번호를 추가한 뒤, 각각 줄바꿈으로 합치기
	context_list = ""
	for i, passage in enumerate(context):
		context_list += f"Passage {i+1}: {passage}\n"
	context_list = context_list.strip()

	# previous_steps를 문자열로 변환
	if len(previous_steps)>0:
		previous_steps_str = "\n".join(previous_steps).strip()
	else:
		previous_steps_str = "(No previous steps.)"

	prompt = f"""### Task: Evaluate the Correctness of the Reasoning Step

Question:
{question}

Retrieved Passages:
{context_list}

Previous Steps:
{previous_steps_str}

Step to evaluate:
{current_step}
""".strip()

	return prompt

def main():
	# -----------------------------
	# 3️⃣ input/output 구성
	# -----------------------------
	args = parse_args()
	input_path = args.input_path
	output_path = args.output_path or input_path.replace(".json", ".jsonl")
	df = pd.read_json(input_path)

	df = delete_comma_at_the_end(df)
	df = fix_step_index(df)

	train_data = []
	for _, row in df.iterrows():    
		output_dict = {"error_type": row['error_type'], "diagnosis": row['diagnosis'], "guidance": row['guidance']}
		output_str = json.dumps(output_dict, ensure_ascii=False)
		
		entry = {
			"input": make_input_prompt(row),
			"output": output_str
		}
		train_data.append(entry)

	print(f"총 {len(train_data)}개의 training sample 생성됨.")

	# -----------------------------
	# 4️⃣ JSONL 형식으로 저장
	# -----------------------------
	with open(output_path, "w", encoding="utf-8") as f:
		for ex in train_data:
			f.write(json.dumps(ex, ensure_ascii=False) + "\n")

	print(f"✅ 저장 완료: {output_path}")


if __name__ == "__main__":
	main()
