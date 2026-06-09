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
You are a DPO Data Optimization Expert. Your goal is to achieve "Length Equilibrium" by stylistically expanding the original feedback (Rejected: `diagnosis`, `guidance`) to match the combined length of the GPT-refined feedback (Chosen: `diagnosis_gpt`, `guidance_gpt`).

# Core Principle: Preservation of Flaws (CRITICAL)
- You MUST maintain the "Rejected" nature of the original feedback. 
- DO NOT improve the logical quality, DO NOT fix the errors, and DO NOT make the guidance more helpful.
- If the original diagnosis is vague, incorrect, or incomplete, the expanded version MUST remain equally vague, incorrect, or incomplete.
- The expansion should be purely STYLISTIC, not INFORMATIVE.

# Goal
1. Stylistically expand 'diagnosis' and 'guidance' (Rejected) into complete sentences until their combined length matches the input 'diagnosis_gpt' and 'guidance_gpt'.
2. Return ONLY the rewritten 'diagnosis' and 'guidance' in the final JSON output.

# Rules for Rewriting (Expansion without Improvement)

## 1. Stylistic Expansion (No Quality Buffs)
- Instead of adding new insights, use more words to express the **exact same flawed idea**.
- Use phrases that add volume but no new information: "It appears that within the current context of this reasoning step," "Based on the observations made regarding the provided chain so far."
- Convert short, punchy errors into longer, formal descriptions of the same error. 
  - (Example: "Redundant" -> "This specific step is characterized by a high degree of redundancy as it restates previous points.")
- Keep the guidance "Rejected" by ensuring it remains as unhelpful or incorrect as the original.

## 2. Precise Target Length Matching
- Observe the combined length (word/token count) of the input `diagnosis_gpt` + `guidance_gpt`.
- Target Parity: Expand the Rejected pair so its total length is roughly equal to the Chosen pair (within a 5-10% margin).
- If the Chosen pair is short, keep the Rejected expansion minimal to avoid length bias in the other direction.

# Input Data
You will receive a JSON containing:
- `question`, `retrieved_passages`, `previous_steps`, `current_step`
- `diagnosis`, `guidance` (To be lengthened - KEEP THE ERRORS)
- `diagnosis_gpt`, `guidance_gpt` (Reference length - DO NOT RETURN IN OUTPUT)

# Output Format (JSON Only)
{
  "diagnosis": "Stylistically lengthened original diagnosis (flaws preserved)",
  "guidance": "Stylistically lengthened original guidance (unhelpfulness preserved)"
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
    )
    return hashlib.md5(check_str.encode('utf-8')).hexdigest()

def evaluate_dpo_candidate(item):
    """
    Calls GPT-5-Mini to evaluate one DPO candidate entry.
    """
    # Construct User Prompt
    passages_text = "\n".join([f"Passage {idx+1}: {p}" for idx, p in enumerate(item['retrieved_passages'])])
    previous_steps_text = "\n".join(item['previous_steps']) if item['previous_steps'] else "(None)"
    
    user_prompt = f"""# Input Data
1. Question: {item.get('question')}
2. Retrieved Passages:
{passages_text}
3. Previous Steps:
{previous_steps_text}
4. Current Step: {item.get('current_step')}

5. Model's Diagnosis (To be lengthened): {item.get('diagnosis')}
6. Model's Guidance (To be lengthened): {item.get('guidance')}

7. GPT's Diagnosis: {item.get('diagnosis_gpt')}
8. GPT's Guidance: {item.get('guidance_gpt')}
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
    input_path = "/workspace/daeyong/first_dpo_training_data_rewrited.json"
    output_path = "/workspace/daeyong/first_dpo_training_data_rewrited_length.json"
    
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
            new_item["diagnosis"] = refined_data.get("diagnosis")
            new_item["guidance"] = refined_data.get("guidance")
            
            results.append(new_item)
            processed_ids.add(current_id)
            
            # 진행 상황 수시 저장
            if len(results) % 5 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Total tokens used: {TOKEN_STATS['total_input'] + TOKEN_STATS['total_output']:,}")
                
            if (TOKEN_STATS["total_input"] + TOKEN_STATS["total_output"]) > 10_000_000:
                print("⚠️ Token limit safety stop.")
                break
        else:
            print("⚠️ Skipping item due to error.")

    print(f"✅ Completed. Total Tokens Used: {TOKEN_STATS['total_input'] + TOKEN_STATS['total_output']:,}")

if __name__ == "__main__":
    main()