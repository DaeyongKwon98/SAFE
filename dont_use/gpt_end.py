import json
import os
from openai import OpenAI
from tqdm import tqdm
import hashlib
import argparse
import math

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Global Token Counter
TOKEN_STATS = {
    "total_input": 0,
    "total_output": 0
}

# --- 2. System Prompt (The GPT Judge) ---
system_prompt = """# Role
You are a Precision Reasoning Auditor. Your goal is to review and refine the `Diagnosis` and `Guidance` for reasoning steps that were previously marked as **FINISHED** (containing `[END_OF_REASONING]`).

# Input Data
- **Question**: The main query.
- **Previous Steps**: Context gathered so far.
- **Current Step**: The step immediately preceding the stop signal.
- **Original Diagnosis**: The previous assessment of the step.
- **Original Guidance**: The instruction that told the model to stop.

# Audit Standards

## 1. Termination Validity Check
You must determine if the "Stop Signal" (`[END_OF_REASONING]`) is valid based on the content of the `Current Step`.
- **VALID Termination**: The `Current Step` explicitly performs the final logical deduction OR explicitly states the final answer text.
- **INVALID Termination (Premature)**:
  - The step is merely an `(Attribution)` (extracting a raw fact).
  - The step lists facts (e.g., "A is 5, B is 3") but does not explicitly state the comparison result.
  - The step contains meta-talk or planning (e.g., "I will now compare...").

    **CRITICAL EXCEPTION (Previous Step Check):**
    - BEFORE checking the `Current Step`, look at the last step in `Previous Steps`.
    - If the last previous step explicitly says "is the answer", "are the answer", or "is the correct answer":
    - The reasoning is ALREADY FINISHED.
    - The `Current Step` is likely Overthinking.
    - **Verdict:** The Stop Signal is **VALID**. Do NOT ask to state the answer again.

## 2. Diagnosis Refinement Rules (Content Validation Only)
Evaluate the `Current Step` exclusively on its own merit as an atomic action.
- **Valid Content**: If the step correctly extracts a fact or performs valid logic based on previous steps, praise its accuracy.
  - *Do NOT* criticize the step for not being the final answer.
  - *Do NOT* say "The step fails to compare..." (That is a guidance issue, not a diagnosis of the current step's accuracy).
  - *Example:* "The step accurately extracts the birth date 'June 8, 1952' from Passage 5." (Correct)
- **Invalid Content**: Only criticize the diagnosis if the content *itself* is factually wrong (hallucination) or logically flawed (wrong math/inference).

## 3. Guidance Refinement Rules (Next Atomic Step)
If the termination was INVALID, you must remove `[END_OF_REASONING]` and provide guidance for the **Immediate Next Atomic Step**.

**Definition of "Atomic Step" for Guidance:**
- **Allowed (Logic + Conclusion)**: Instructing the model to compare known facts and state the answer IS allowed as a single step.
  - *Example:* "Compare the birth years of Director A and Director B to determine who is older and state the answer."
- **Prohibited (Retrieval + Logic)**: Instructing the model to find *new* information AND perform logic is NOT allowed.
  - *Bad Example:* "Find the director of Film B and compare their nationality." (Split this: First instruct to find the director).

# Task Instructions
1. Analyze if the `Current Step` truly concludes the answer.
2. If **VALID**: Keep the status as "Pass".
3. If **INVALID**:
   - Set status to "Fixed".
   - **Refine Diagnosis**: Validate the accuracy of what was *actually done* in the current step.
   - **Refine Guidance**: Provide the instruction for the next atomic action (usually a logical comparison or final statement).

# Output Format (JSON Only)
{
  "status": "Pass" or "Fixed",
  "refined_diagnosis": "The corrected diagnosis string.",
  "refined_guidance": "The corrected guidance string."
}

# Few-shot Demonstrations

## Example 1: Premature Stop (Attribution Tag)
- Question: "What is the date of birth of the director of film Freeway Ii: Confessions Of A Trickbaby?"
- Previous Steps: ["Step 1: The director is Matthew Bright. (Attribution)"]
- Current Step: "Step 2: From the information in Passage 5, the date of birth of Matthew Bright (from Step 1) is June 8, 1952. (Attribution)"
- Original Diagnosis: "The reasoning accurately connects the director identified in Passage 7 with the birth date given in Passage 5."
- Original Guidance: "State explicitly that June 8, 1952 is the answer. Stop reasoning now. [END_OF_REASONING]"
- Output:
{
  "status": "Fixed",
  "refined_diagnosis": "The step accurately connects the director identified in Passage 7 (Matthew Bright) with the birth date given in Passage 5, correctly establishing that his date of birth is June 8, 1952.",
  "refined_guidance": "State explicitly that June 8, 1952 is the answer to the question about the director’s date of birth and conclude the reasoning."
}

## Example 2: Premature Stop (Meta-talk / Planning)
- Question: "Are the directors of both films Tanya'S Island and Madhurappathinezhu from the same country?"
- Previous Steps: ["Step 4: The director of Madhurappathinezhu is Hariharan (Indian). (Attribution)"]
- Current Step: "Step 5: To finalize my answer, I will now review the countries of origin for both film directors to ensure I haven't missed any crucial information that could alter the conclusion. (Logical)"
- Original Diagnosis: "The step outlines the plan to review the countries of origin."
- Original Guidance: "Directly state the final conclusion. Stop reasoning now. [END_OF_REASONING]"
- Output:
{
  "status": "Fixed",
  "refined_diagnosis": "The step states a procedural intention to review information but does not strictly perform a new retrieval or logical inference.",
  "refined_guidance": "Directly state the final conclusion that the directors are from different countries, based on the established facts."
}

## Example 3: Premature Stop (Facts Gathered, Comparison Pending)
- Question: "Do director of film Being Cyrus and director of film Papa Kehte Hai share the same nationality?"
- Previous Steps: ["Step 3: Homi Adajania is the director of Being Cyrus. (Attribution)"]
- Current Step: "Step 4: According to Passage 8, the nationality of Mahesh Bhatt (from Step 3) is Indian. (Attribution)"
- Original Diagnosis: "Step 4 correctly extracts Mahesh Bhatt's nationality as Indian."
- Original Guidance: "Compare them and conclude. Stop reasoning now. [END_OF_REASONING]"
- Output:
{
  "status": "Fixed",
  "refined_diagnosis": "Step 4 correctly extracts Mahesh Bhatt's nationality as Indian from Passage 8, which aligns with the prior step identifying him as the director of Papa Kehte Hai.",
  "refined_guidance": "Compare the nationalities of Homi Adajania and Mahesh Bhatt to determine if they share the same nationality and state the final answer."
}

## Example 4: Correct Termination (Valid Logical Conclusion)
- Question: "Who wrote an unofficial sequel-series of J.K. Rowling's Harry Potter universe...?"
- Previous Steps: ["Step 4: G. Norman Lippert wrote the James Potter series. (Attribution)"]
- Current Step: "Step 5: Therefore, G. Norman Lippert (from Step 4) is the author who wrote the unofficial sequel-series. (Logical)"
- Original Diagnosis: "The step is a valid logical conclusion from the attributed evidence."
- Original Guidance: "All necessary evidence... Stop reasoning now. [END_OF_REASONING]"
- Output:
{
  "status": "Pass",
  "refined_diagnosis": "The step is a valid logical conclusion from the attributed evidence, correctly identifying G. Norman Lippert as the author who wrote the unofficial sequel-series.",
  "refined_guidance": "All necessary evidence and logical deductions have been gathered for the final answer process. Stop reasoning now. [END_OF_REASONING]"
}

## Example 5: Correct Termination (Comparison Performed)
- Question: "Which genus has more species Fatsia or Leuchtenbergia?"
- Previous Steps: ["Step 2: Fatsia has 3 species.", "Step 4: Leuchtenbergia has 1 species."]
- Current Step: "Step 5: 3 (from Step 2) is more than 1 (from Step 4), so Fatsia has more species. (Logical)"
- Original Diagnosis: "The step correctly compares the species counts."
- Original Guidance: "Stop reasoning now. [END_OF_REASONING]"
- Output:
{
  "status": "Pass",
  "refined_diagnosis": "The step accurately uses the earlier derived species counts for Fatsia and Leuchtenbergia to compare them and reaches the correct conclusion that Fatsia has more species.",
  "refined_guidance": "All necessary evidence and logical deductions have been gathered for the final answer process. Stop reasoning now. [END_OF_REASONING]"
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
    # previous_steps가 str인 경우 리스트로 변환
    if isinstance(item['previous_steps'], str):
        item['previous_steps'] = eval(item['previous_steps'])
    
    previous_steps_text = "\n".join(item['previous_steps']) if item['previous_steps'] else "(None)"
    
    user_prompt = f"""# Input Data
1. Question: {item.get('question')}
2. Previous Steps:
{previous_steps_text}
3. Current Step: {item.get('current_step')}
4. Original Diagnosis: {item.get('diagnosis')}
5. Original Guidance: {item.get('guidance')}
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
    # ---------------------------------------------------------
    # [설정] argparse로 외부 인자 받아오기
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Run DPO evaluation in chunks.")
    parser.add_argument("--part_index", type=int, required=True, help="Index of the current chunk (0, 1, 2...)")
    parser.add_argument("--total_parts", type=int, default=4, help="Total number of chunks to split the data into")
    args = parser.parse_args()

    PART_INDEX = args.part_index
    TOTAL_PARTS = args.total_parts
    # ---------------------------------------------------------

    # 경로 설정
    input_path = "/workspace/daeyong/third_finetuning_data/training_data_GPT.json"
    
    # [중요] 전체 통합 파일 (이미 처리된 목록 확인용 - Read Only)
    total_output_path = "/workspace/daeyong/third_finetuning_data/training_data_GPT_end_total.json"
    
    # [중요] 각 파트별 개별 저장 파일 (쓰기 전용 - 충돌 방지)
    # 예: ..._part0.json, ..._part1.json
    part_output_path = f"/workspace/daeyong/third_finetuning_data/training_data_GPT_end_part{PART_INDEX}.json"
    
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print("❌ Input file not found.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # ---------------------------------------------------------
    # 1. 이미 처리된 ID 수집 (Resume 기능)
    # ---------------------------------------------------------
    processed_ids = set()
    
    # (1) 전체 통합 파일에서 확인
    if os.path.exists(total_output_path):
        try:
            with open(total_output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for res in existing_data:
                    processed_ids.add(get_unique_id(res))
            print(f"🔍 Found {len(processed_ids)} processed items in TOTAL file.")
        except Exception as e:
            print(f"⚠️ Error reading total output file: {e}")

    # (2) 현재 파트 파일에서 확인 (내 작업 이어하기)
    part_results = []
    if os.path.exists(part_output_path):
        try:
            with open(part_output_path, "r", encoding="utf-8") as f:
                part_results = json.load(f)
                for res in part_results:
                    processed_ids.add(get_unique_id(res))
            print(f"🔄 Resuming Part {PART_INDEX}: Found {len(part_results)} items already in part file.")
        except Exception as e:
            print(f"⚠️ Error reading part output file: {e}")

    # ---------------------------------------------------------
    # 2. 처리 대상 후보군 선정 및 분할 (Chunking)
    # ---------------------------------------------------------
    # 조건: 'END_OF_REASONING'이 포함된 데이터만 대상
    # 주의: 여기서 processed_ids로 필터링하지 않습니다. (인덱스가 밀리는 것을 방지하기 위해)
    all_candidates = [item for item in data if "END_OF_REASONING" in item.get("guidance", "")]
    
    total_candidates = len(all_candidates)
    if total_candidates == 0:
        print("✅ No candidates found with END_OF_REASONING.")
        return

    # 3등분 계산
    chunk_size = math.ceil(total_candidates / TOTAL_PARTS)
    start_idx = PART_INDEX * chunk_size
    end_idx = start_idx + chunk_size
    
    # 내 몫(Slice) 챙기기
    my_chunk = all_candidates[start_idx:end_idx]
    
    print(f"🔹 [Part {PART_INDEX}/{TOTAL_PARTS}] Processing indices {start_idx} ~ {end_idx}")
    print(f"🔹 My chunk size: {len(my_chunk)}")

    # ---------------------------------------------------------
    # 3. 실제 처리 루프
    # ---------------------------------------------------------
    for item in tqdm(my_chunk, desc=f"Part {PART_INDEX} Processing"):
        current_id = get_unique_id(item)
        
        # 이미 처리된 데이터면 스킵 (통합 파일 or 내 파트 파일)
        if current_id in processed_ids:
            continue
        
        refined_data, usage = evaluate_dpo_candidate(item)
        
        if refined_data:
            TOKEN_STATS["total_input"] += usage.prompt_tokens
            TOKEN_STATS["total_output"] += usage.completion_tokens
            
            # 결과 객체 생성 (원본 + 수정된 필드)
            new_item = item.copy()
            new_item["status"] = refined_data.get("status")
            new_item["refined_diagnosis"] = refined_data.get("refined_diagnosis")
            new_item["refined_guidance"] = refined_data.get("refined_guidance")
            
            # 리스트에 추가
            part_results.append(new_item)
            processed_ids.add(current_id)
            
            # 파일 저장 (Overwrite 방식 - 데이터가 크지 않으므로 안전함)
            with open(part_output_path, "w", encoding="utf-8") as f:
                json.dump(part_results, f, ensure_ascii=False, indent=2)
                
            # 토큰 제한 확인
            if (TOKEN_STATS["total_input"] + TOKEN_STATS["total_output"]) > 3_000_000:
                print("⚠️ Token limit safety stop.")
                break
        else:
            print("⚠️ Skipping item due to API error.")

    print(f"✅ Part {PART_INDEX} Completed. Saved to {part_output_path}")
    print(f"💰 Tokens Used in this session: {TOKEN_STATS['total_input'] + TOKEN_STATS['total_output']:,}")

if __name__ == "__main__":
    main()