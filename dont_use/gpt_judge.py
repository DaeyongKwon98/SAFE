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
You are a Senior Reasoning Data Architect. Your mission is to audit and refine feedback data (Error Type, Diagnosis, Guidance) used for training a reasoning model. You must ensure every feedback entry strictly adheres to the "Atomic Step Rule" and "Logical Completeness" standards.

# System Logic: Core Constraints

## 1. The Atomic Step Rule (STRICT)
Every reasoning step must be "Atomic" (one action only):
- Attribution Step: Extract exactly ONE fact from exactly ONE passage.
- Logical Step: Perform exactly ONE inference or comparison based on previously cited facts.
- A step is "Double-Action" (and thus needs correction) if it cites a passage AND makes a conclusion simultaneously.

## 2. Sufficiency & Termination (The Bridge Rule)
- Fact Sufficiency: All raw data needed to answer the question must be extracted.
- Logical Finalization: Even if all facts are present, the reasoning is INCOMPLETE until a final logical bridge (e.g., "Therefore, A and B are...") is explicitly stated.
- Stop Signal: The tag `[END_OF_REASONING]` must ONLY appear in the Guidance when both Fact Sufficiency and Logical Finalization are achieved.

# Error Type Definitions
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
3. Check for Atomic Violations: Did the step mix attribution and logic?

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
    """
    # 중복 체크 기준 5개 항목 추출
    check_str = (
        f"{item.get('question', '')}"
        f"{item.get('current_step', '')}"
        f"{item.get('error_type', '')}"
        f"{item.get('diagnosis', '')}"
        f"{item.get('guidance', '')}"
    )
    # 공백 제거 및 해싱을 통해 고유 키 생성
    return hashlib.md5(check_str.encode('utf-8')).hexdigest()

def evaluate_dpo_candidate(item):
    """
    Calls GPT-5-Mini to evaluate one DPO candidate entry.
    """
    # Construct User Prompt
    passages_text = "\n".join([f"Passage {idx+1}: {p}" for idx, p in enumerate(item['retrieved_passages'])])
    previous_steps_text = "\n".join(item['previous_steps']) if item['previous_steps'] else "(None)"
    
    user_prompt = f"""# Input Data
1. Question: {item['question']}

2. Retrieved Passages:
{passages_text}

3. Previous Steps:
{previous_steps_text}

4. Current Step: {item['current_step']}

5. Original Error Type: {item['error_type']}

6. Original Diagnosis: {item['diagnosis']}

7. Original Guidance: {item['guidance']}
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


def process_reasoning_data(json_data):
    dpo_candidates = []

    for item in json_data:
        # 1. 메타 데이터 추출
        meta = item.get("meta_data", {})
        question = meta.get("question", "")
        retrieved_passages = meta.get("retrieved_passages", [])
        
        # 이전 단계들을 저장할 리스트 (Accepted된 스텝만 쌓임)
        previous_steps_accumulator = []
        
        # steps_history를 step_num 순서대로 정렬
        steps_history = sorted(item.get("steps_history", []), key=lambda x: x["step_num"])
        
        for step in steps_history:
            # attempts를 retry_index 순서대로 정렬
            attempts = sorted(step.get("attempts", []), key=lambda x: x["retry_index"])
            
            for attempt in attempts:
                current_step_text = attempt.get("generated_text", "")
                evaluation = attempt.get("evaluation", {})
                result = attempt.get("result", "")
                
                # 2. 데이터 구성
                # 주의: 리스트는 Mutable이므로 복사(.copy())해서 저장해야 함
                entry = {
                    "question": question,
                    "retrieved_passages": retrieved_passages,
                    "previous_steps": previous_steps_accumulator.copy(), 
                    "current_step": current_step_text,
                    "error_type": evaluation.get("error_type", ""),
                    "diagnosis": evaluation.get("diagnosis", ""),
                    "guidance": evaluation.get("guidance", "")
                }
                
                dpo_candidates.append(entry)
                
                # 3. Accepted인 경우에만 History에 추가
                # Rejected(Rollback)인 경우, 다음 attempt도 현재와 동일한 previous_steps를 가져야 하므로 추가하지 않음
                if result == "Accepted":
                    previous_steps_accumulator.append(current_step_text)

    return dpo_candidates


def main():
    input_path = "/workspace/daeyong/ours_dpo_600_qwen7b_musique_logs_qwen.json"
    output_path = "/workspace/daeyong/judge_qwen7b_musique.json"
    
    # 1. Load Data
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print("❌ Input file not found.")
        return
    with open(input_path, "r", encoding="utf-8") as f:
        data = process_reasoning_data(json.load(f))
    
    # 2. Resume Logic (중복 체크용 Set 구성)
    results = []
    processed_ids = set() # 이미 처리된 ID 저장
    
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                for res in results:
                    # 결과 파일에 있는 항목들로 ID 세트 생성
                    processed_ids.add(get_unique_id(res))
            print(f"🔄 Resuming: {len(processed_ids)} unique records already processed.")
        except Exception as e:
            print(f"⚠️ Output file read error: {e}. Starting over.")

    # 3. Processing Loop
    print(f"🚀 Starting Evaluation on {len(data)} items...")
    
    for item in tqdm(data, desc="Judging"):
        # 고유 ID 생성
        current_id = get_unique_id(item)
        
        # 이미 처리된 ID라면 건너뛰기
        if current_id in processed_ids:
            continue
        
        evaluation, usage = evaluate_dpo_candidate(item)
        
        if evaluation:
            # Update Token Stats
            TOKEN_STATS["total_input"] += usage.prompt_tokens
            TOKEN_STATS["total_output"] += usage.completion_tokens
            
            # Merge Evaluation into Item
            result_item = item.copy()
            result_item["judge_evaluation"] = evaluation
            
            results.append(result_item)
            processed_ids.add(current_id)
            
            # Save progressively
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Processed item. Total tokens used so far: {TOKEN_STATS['total_input'] + TOKEN_STATS['total_output']:,}")
            
            # Token Limit Safety (Optional)
            if (TOKEN_STATS["total_input"] + TOKEN_STATS["total_output"]) > 3_000_000:
                print("⚠️ Token limit reached. Stopping.")
                break
        else:
            print("⚠️ Skipping item due to error.")

    print(f"✅ Completed. Total Tokens Used: {TOKEN_STATS['total_input'] + TOKEN_STATS['total_output']:,}")

if __name__ == "__main__":
    main()