import json
import os
from openai import OpenAI
from tqdm import tqdm
import hashlib

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Global Token Counter
TOKEN_STATS = {
    "total_input": 0,
    "total_output": 0
}

# --- 2. System Prompt (The GPT Judge) ---
system_prompt = """# Role
You are a Senior Reasoning Data Architect. Your mission is to audit and refine feedback data (Error Type, Diagnosis, Guidance).

# System Logic: Core Constraints

## 1. The Single Next-Step Rule
- Guidance must only provide instructions for the IMMEDIATE NEXT ATOMIC STEP.
- DO not provide multi-step instructions (e.g., "Find X and then compare it with Y"). 
- If multiple actions are needed to reach the answer, only instruct the first missing action. The model should receive instructions for the subsequent actions only after it completes the first one.

## 2. The Atomic Action Definition
An action is atomic if it is either:
- Attribution: Extracting exactly ONE fact from exactly ONE passage.
- Logical: Performing exactly ONE inference or comparison based on previously extracted facts.

## 3. Sufficiency & Termination
- Fact Sufficiency: All raw data needed to answer the question must be extracted.
- Logical Finalization: Even if all facts are present, the reasoning is INCOMPLETE until a final logical bridge (e.g., "Therefore, A and B are...") is explicitly stated.
- Stop Signal: The tag `[END_OF_REASONING]` must ONLY appear in the Guidance when both Fact Sufficiency and Logical Finalization are achieved.

# Error Type Definitions (You should choose only one of these types)
- Correct (No Error): The step is logically sound and adheres to the Atomic Step Rule. It either moves the reasoning forward or provides the final logical conclusion to appropriately end the chain.
- Off-topic: The step introduces information or inferences irrelevant to the question or the intended reasoning path.
- Redundancy: The step repeats previous information or conclusions without new meaningful progression.
- Overthinking: The step continues after the final logical conclusion has already been reached.
- Inefficiency: The step provides procedural meta-talk (e.g., "I will now look for...") instead of performing an actual attribution or inference.
- Logical Fallacy: The underlying facts are correct, but the deduction drawn from them is flawed (e.g., incorrect age comparison).
- Unsupported: The step claims information that cannot be found in the Retrieved Passages (Hallucination).
- Contradictory: The step claims information that directly conflicts with the Retrieved Passages.
- Information Miss: The step incorrectly claims information is missing when it is actually present in the passages.

# Task Instructions
Evaluate the provided reasoning context and feedback labels. 

1. Check for Premature Termination: Did the original guidance say "Stop" before a Logical Bridge was built?
2. Check for Missed Termination: Is the logic finished but the guidance failed to provide `[END_OF_REASONING]`?

# Output Generation Protocol
If the input feedback is perfect, set status to "Accurate" and repeat the input values.
If the input is flawed, set status to "Inaccurate" and generate "Golden Feedback":
- True Diagnosis: Explain exactly why the original was wrong.
- True Guidance: 
    - If COMPLETE: Start with: "All necessary evidence and logical deductions have been gathered for the final answer process. Stop reasoning now. [END_OF_REASONING]"
    - If INCOMPLETE: Provide direct, atomic instructions for the next step.

# Input Data
- Question
- Retrieved Passages
- Previous Steps
- Current Step
- Original Error Type
- Original Diagnosis
- Original Guidance

# Final Output (JSON Only)
{
  "status": "Accurate" or "Inaccurate",
  "reasoning": "Explanation of the rule check",
  "true_error_type": "The correct error type",
  "true_diagnosis": "The refined diagnosis",
  "true_guidance": "The refined guidance (with [END_OF_REASONING] if all criteria are met)"
}
""".strip()

def get_unique_id(item):
    """
    항목의 고유성을 식별하기 위한 해시 키를 생성합니다.
    수정된 후에도 변하지 않는 '원본 필드'들만 사용해야 Resume 기능이 작동합니다.
    """
    check_str = (
        f"{item.get('question', '')}"
        f"{item.get('current_step', '')}"
        f"{item.get('error_type', '')}" # 원본 에러 타입 (수정 안 됨)
        f"{item.get('diagnosis', '')}"  # 원본 모델의 진단 (수정 안 됨)
        f"{item.get('guidance', '')}"   # 원본 모델의 가이드 (수정 안 됨)
    )
    return hashlib.md5(check_str.encode('utf-8')).hexdigest()

def evaluate_dpo_candidate(item):
    """
    Calls GPT-5.1 to evaluate one DPO candidate entry.
    """
    # Construct User Prompt
    
    # retrieved_passages와 previous_steps가 str인 경우 리스트로 변환
    if isinstance(item['retrieved_passages'], str):
        item['retrieved_passages'] = eval(item['retrieved_passages'])
    if isinstance(item['previous_steps'], str):
        item['previous_steps'] = eval(item['previous_steps'])
    
    passages_text = "\n".join([f"Passage {idx+1}: {p}" for idx, p in enumerate(item['retrieved_passages'])])
    previous_steps_text = "\n".join(item['previous_steps']) if item['previous_steps'] else "(None)"
    
    user_prompt = f"""# Input Data
1. Question: {item.get('question')}
2. Retrieved Passages:
{passages_text}
3. Previous Steps:
{previous_steps_text}
4. Current Step: {item.get('current_step')}
5. Original Error Type: {item.get('error_type')}
6. Original Diagnosis: {item.get('diagnosis')}
7. Original Guidance: {item.get('guidance')}
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


def main():
    input_path = "/workspace/daeyong/third_finetuning_data/training_data_rewritten_deduplicated_splitted.json"
    output_path = "/workspace/daeyong/third_finetuning_data/training_data_rewritten_deduplicated_splitted_gpt.json"
    
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print("❌ Input file not found.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    processed_ids = set() 
    
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                for res in results:
                    processed_ids.add(get_unique_id(res))
            print(f"🔄 Resuming: {len(processed_ids)} unique records already processed.")
        except Exception as e:
            print(f"⚠️ Output file read error: {e}. Starting over.")

    print(f"🚀 Starting Evaluation on {len(data)} items...")
    
    for item in tqdm(data, desc="Refining"):
        current_id = get_unique_id(item)
        if current_id in processed_ids:
            continue
        
        refined_data, usage = evaluate_dpo_candidate(item)
        
        if refined_data:
            TOKEN_STATS["total_input"] += usage.prompt_tokens
            TOKEN_STATS["total_output"] += usage.completion_tokens
            
            # 원본 데이터 복사 후 GPT가 정제한 diagnosis_gpt, guidance_gpt 덮어쓰기
            new_item = item.copy()
            new_item['status'] = refined_data.get("status")
            new_item['reasoning'] = refined_data.get("reasoning")
            new_item["error_type_gpt"] = refined_data.get("true_error_type")
            new_item["diagnosis_gpt"] = refined_data.get("true_diagnosis")
            new_item["guidance_gpt"] = refined_data.get("true_guidance")
            
            results.append(new_item)
            processed_ids.add(current_id)
            
            # 진행 상황 수시 저장
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Total tokens used: {TOKEN_STATS['total_input'] + TOKEN_STATS['total_output']:,}")
                
            if (TOKEN_STATS["total_input"] + TOKEN_STATS["total_output"]) > 1_000_000:
                print("⚠️ Token limit safety stop.")
                break
        else:
            print("⚠️ Skipping item due to error.")

    print(f"✅ Completed. Total Tokens Used: {TOKEN_STATS['total_input'] + TOKEN_STATS['total_output']:,}")

if __name__ == "__main__":
    main()