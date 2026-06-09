import json
import os
from openai import OpenAI
from tqdm import tqdm
import ast
import pandas as pd

from prompts import sft_data_validation_prompt

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Global Token Counter
TOKEN_STATS = {
    "total_input": 0,
    "total_output": 0
}

# --- 2. System Prompt (The GPT Judge) ---
system_prompt = """# Role
You are an expert AI Judge for evaluating reasoning chains and feedback quality. Your goal is to verify if the provided "Error Type", "Diagnosis", and "Guidance" are accurate according to the strict definitions below.

# Error Type Definitions (CRITICAL)
Use these definitions to validate the provided data:

- **Correct (No Error)**: The step is logically sound, fully supported by the retrieved passages, and moves the reasoning forward correctly.
- **Off-topic**: The step is irrelevant to the overall goal of the question and the specific step it is replacing. It introduces a new, unrelated piece of information or inference that leads the reasoning process astray.
- **Redundancy**: The step repeats information or conclusions from previous steps without providing any significant new progression. It stalls the reasoning process by repeating what is already known.
- **Overthinking**: The step continues *after* the reasoning is sufficient to answer the question. It introduces a new, *unnecessary* line of reasoning that is no longer required to find the final answer.
- **Inefficiency**: The step provides meta-discussion, procedural intent, planning statements, or placeholder reasoning instead of executing meaningful inference. It does not extract evidence or logically reason but merely describes what the model plans to do rather than doing it.
- **Logical Fallacy**: The step contains a flawed reasoning process. The facts gathered from previous steps are correct, but the conclusion drawn from them is incorrect.
- **Unsupported**: The step makes a factual claim using information that **cannot** be found in any of the `Retrieved Passages`. The step hallucinates a new, false piece of information and presents it as fact.
- **Contradictory**: The step makes a factual claim that **directly conflicts** or **contradicts** information explicitly stated in the `Retrieved Passages`.
- **Information Miss**: The step incorrectly concludes that specific information is unavailable, unknown, or missing—even though the information is present in the retrieved passages. The failure lies in not recognizing or retrieving relevant evidence that already exists.

# Task
Determine if the Input's `Error Type`, `Diagnosis`, and `Guidance` are ACCURATE based on the `Current Step` and the definitions above.

## Step 1: Analyze the Current Step independent of the input label
- Check logical continuity from `Previous Steps`.
- Check factual support from `Retrieved Passages`.
- Check relevance to the `Question`.

## Step 2: Evaluate the Input Label (Error Type & Feedback)
- Compare your independent analysis with the provided `Error Type`, `Diagnosis`, and `Guidance`.
- **Accurate**: If the provided label matches the actual error (or correctness) of the step.
- **Inaccurate**:
    - If the provided `Error Type` is wrong (e.g., labeled 'Correct' but is actually 'Unsupported').
    - If the `Diagnosis` is factually incorrect or illogical.
    - If the `Guidance` is misleading or suggests the wrong next step.

## Step 3: Generate Golden Feedback (Required if Inaccurate)
- If the input is Inaccurate, or if it needs refinement, follow these strict rules for true_guidance:
    - Atomic Single Step: The guidance MUST point to exactly one next action. Do not provide a multi-step plan or combine multiple inferences.
    - Termination Logic: Determine if the information gathered so far (including the corrected step) is sufficient to answer the question.
        - If the current reasoning has reached a final conclusion (Final Logical Step: comparison, total sum, or definitive answer), the guidance MUST include: "All necessary evidence and logical deductions have been gathered for the final answer process. Stop reasoning now. [END_OF_REASONING]"
        - A "Final Logical Step" must precede this termination signal.
        - No Meta-talk: Avoid "You should..." or "The model must...". Direct instructions only.

# Output Format (JSON Only)
{
  "status": "Accurate" or "Inaccurate",
  "reasoning": "Brief explanation of why the input label was right or wrong.",
  "true_error_type": "The correct error type selected from the definitions.",
  "true_diagnosis": "Correct explanation of the error.",
  "true_guidance": "Correct instruction for the next single step."
}
""".strip()

system_prompt = sft_data_validation_prompt

def evaluate_dpo_candidate(item):
    """
    Calls GPT-5-Mini to evaluate one DPO candidate entry.
    """
    # Construct User Prompt
    retrieved_passages_list = eval(item['retrieved_passages']) if isinstance(item['retrieved_passages'], str) else item['retrieved_passages']
    
    passages_text = "\n".join([f"Passage {idx+1}: {p}" for idx, p in enumerate(retrieved_passages_list)])
    previous_steps_text = "\n".join(item['previous_steps']) if item['previous_steps'] else "(None)"
    
    user_prompt = f"""# Input Data
1. Question: {item['question']}

2. Retrieved Passages:
{passages_text}

3. Previous Steps:
{previous_steps_text}

4. Current Step: {item['current_step']}

5. Model's Predicted Error Type: {item['error_type']}

6. Model's Diagnosis: {item['diagnosis']}

7. Model's Guidance: {item['guidance']}
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=4096,
        )
        
        response_content = completion.choices[0].message.content
        usage = completion.usage
        
        return json.loads(response_content), usage

    except Exception as e:
        print(f"API Error: {e}")
        return None, None


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


def main():
    input_path = "/workspace/daeyong/third_finetuning_data/training_data_GPT_end_fixed.json"
    output_path = "/workspace/daeyong/third_finetuning_data/training_data_GPT_end_fixed_isvalid.json"
    
    # 1. Load Data
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print("❌ Input file not found.")
        return
    
    df = fix_step_index(pd.read_json(input_path))
    
    # 순서 랜덤으로 섞기
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Resume Logic
    results = []
    processed_count = 0
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_count = len(results)
            print(f"🔄 Resuming from {processed_count} records.")
        except:
            print("⚠️ Output file corrupted. Starting over.")

    # 3. Processing Loop
    print(f"🚀 Starting Evaluation on {len(df)} items...")
    
    # Slice data to skip processed items
    data_to_process_df = df[processed_count:]
    
    for i, item in tqdm(data_to_process_df.iterrows(), total=len(data_to_process_df), desc="Evaluating"):
        evaluation, usage = evaluate_dpo_candidate(item)
        
        if evaluation:
            # Update Token Stats
            TOKEN_STATS["total_input"] += usage.prompt_tokens
            TOKEN_STATS["total_output"] += usage.completion_tokens
            
            # Merge Evaluation into Item
            result_item = item.to_dict()
            result_item["judge_evaluation"] = evaluation
            
            results.append(result_item)
            
            # Save progressively
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Processed item. Total tokens used so far: {TOKEN_STATS['total_input']:,} + {TOKEN_STATS['total_output']:,}")
            
            # Token Limit Safety (Optional)
            if (TOKEN_STATS["total_input"] + TOKEN_STATS["total_output"]) > 5_000_000:
                print("⚠️ Token limit reached. Stopping.")
                break
        else:
            print("⚠️ Skipping item due to error.")

    print(f"✅ Completed. Total Tokens Used: {TOKEN_STATS['total_input'] + TOKEN_STATS['total_output']:,}")

if __name__ == "__main__":
    main()