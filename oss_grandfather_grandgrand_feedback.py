import json
import os
import re
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================================================================
# 1. Configuration & Prompts
# =============================================================================
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/grandfather_grandgrand_generated.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/grandfather_grandgrand_feedback.json"

KINSHIP_FEEDBACK_SYSTEM_PROMPT = """You are an expert AI evaluator providing feedback on reasoning chains involving family relationships.
Your task is to evaluate a specific **Logical Step** in a reasoning chain where the user commits a "Kinship Definition Error" (specifically, confusing a Parent with a Grandparent).

**Task:**
Identify the "Logical Fallacy" where the reasoning incorrectly claims that a direct **Parent** is the **Grandparent**.

**Requirements:**
1.  **error_type**: ALWAYS return "Logical Fallacy".
2.  **diagnosis**: Explain the specific kinship error. State: "The reasoning incorrectly identifies the parent ([Parent Name]) as the grandparent. The parent of a person is their father/mother, not their grandfather/grandmother."
3.  **guidance**: Provide the corrective logic. E.g., "To find the grandfather, you must identify the father of [Parent Name]. Do not stop at the direct parent."

**Output Format:**
Provide ONLY a valid JSON object:
{
  "error_type": "Logical Fallacy",
  "diagnosis": "...",
  "guidance": "..."
}

**Few-Shot Examples:**

Input:

Question: Who is the paternal grandfather of Birger Brosa?
Retrieved Passages:
Passage 1: Birger Brosa (died 1202) was the son of Bengt Snivil.
Passage 2: Bengt Snivil was a Swedish jarl and the son of Folke the Fat.
Passage 3: Birger Brosa had several sons, including Philip and Knut.
Reasoning Chain:
Step 1: According to Passage 1, Birger Brosa is the son of Bengt Snivil. (Attribution)
Step 2: According to Passage 2, Bengt Snivil is the son of Folke the Fat. (Attribution)
Step 3: Since Bengt Snivil is the father of Birger Brosa, Bengt Snivil is the paternal grandfather of Birger Brosa. (Logical)

Output:
{
  "error_type": "Logical Fallacy",
  "diagnosis": "The step incorrectly defines the relationship. It correctly identifies Bengt Snivil as the father in the premise, but then logically concludes he is the grandfather. A father is a parent, not a grandparent.",
  "guidance": "Recognize that Bengt Snivil is the father, not the grandfather. To find the paternal grandfather, you must look for the father of Bengt Snivil (Folke the Fat) mentioned in Passage 2."
}

Input:

Question: Who is the maternal grandmother of Princess Marie of Edinburgh?
Retrieved Passages:
Passage 1: Princess Marie was the eldest daughter of Prince Alfred and Grand Duchess Maria Alexandrovna of Russia.
Passage 2: Maria Alexandrovna was the daughter of Alexander II of Russia and Marie of Hesse.
Passage 3: Prince Alfred was the second son of Queen Victoria and Prince Albert.
Reasoning Chain:
Step 1: According to Passage 1, Princess Marie is the daughter of Grand Duchess Maria Alexandrovna. (Attribution)
Step 2: According to Passage 2, Maria Alexandrovna is the daughter of Marie of Hesse. (Attribution)
Step 3: Since Maria Alexandrovna is the mother of Princess Marie, Maria Alexandrovna is the maternal grandmother of Princess Marie. (Logical)

Output:
{
  "error_type": "Logical Fallacy",
  "diagnosis": "The reasoning commits a generation skip error. Maria Alexandrovna is explicitly identified as the mother of Princess Marie, but the conclusion incorrectly labels her as the grandmother.",
  "guidance": "Do not conflate the mother with the grandmother. Identify the mother of Maria Alexandrovna (Marie of Hesse) from Passage 2 to find the correct maternal grandmother."
}

Input:

Question: Who is the paternal grandmother of Sir Robert Long?
Retrieved Passages:
Passage 1: Sir Robert Long, 1st Baronet (1600–1673) was the son of Sir Walter Long of South Wraxall.
Passage 2: Sir Walter Long was the son of Thomas Long and Mary Cocks.
Passage 3: Robert Long served as Secretary of State to Charles II.
Reasoning Chain:
Step 1: According to Passage 1, Sir Robert Long is the son of Sir Walter Long. (Attribution)
Step 2: According to Passage 2, the mother of Sir Walter Long is Mary Cocks. (Attribution)
Step 3: Since Sir Walter Long is the father of Sir Robert Long, Sir Walter Long is the paternal grandmother of Sir Robert Long. (Logical)

Output:
{
  "error_type": "Logical Fallacy",
  "diagnosis": "The step makes a logical error by identifying the father (Sir Walter Long) as the paternal grandmother. This conflates both the generation (Parent vs Grandparent) and the gender (Father vs Grandmother).",
  "guidance": "Identify the mother of Sir Walter Long (Mary Cocks) from Passage 2 to find the correct paternal grandmother, rather than mislabeling the father."
}
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json(text):
    text = text.strip()
    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1].strip()
        
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            return text

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content

# =============================================================================
# 3. Main Logic (오류 수정 버전)
# =============================================================================
def main():
    print(f"📂 Input Path: {INPUT_FILE}")
    print(f"📂 Output Path: {OUTPUT_FILE}")

    # 1. 원본 데이터 로드
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # 2. 기존 결과 로드 및 중복 체크용 셋 생성
    existing_results = []
    processed_questions = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
                existing_results = json.load(f)
                # 공백 등 미세한 차이 방지를 위해 strip() 사용
                processed_questions = {str(item.get('question')).strip() for item in existing_results}
                print(f"🔄 Resuming... Found {len(processed_questions)} processed items.")
        except Exception as e:
            print(f"⚠️ Failed to load existing output: {e}")

    # 3. 미처리 데이터만 필터링
    to_process = [item for item in full_data if str(item.get('question')).strip() not in processed_questions]
    
    if not to_process:
        print("✅ All items have already been processed.")
        return

    print(f"🚀 To be processed: {len(to_process)} records.")

    # vLLM 설정
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        max_model_len=10000,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    # 4. 배치 처리
    BATCH_SIZE = 100 
    
    for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Processing Batches"):
        batch_records = to_process[i : i + BATCH_SIZE]
        batch_prompts = []
        
        for row in batch_records:
            question = row.get('question', '')
            passages = row.get('retrieved_passages', [])
            reasoning_steps = row.get('generated_kinship_error_reasoning', [])

            if not reasoning_steps: continue

            # 지문 포맷팅
            if isinstance(passages, list):
                passages_text = "\n".join([f"Passage {j+1}: {p}" for j, p in enumerate(passages)])
            else:
                passages_text = str(passages)
            
            reasoning_text = "\n".join(reasoning_steps)
            user_content = f"Question: {question}\n\nRetrieved Passages:\n{passages_text}\n\nReasoning Chain:\n{reasoning_text}\n\n**Task:** Locate the step that makes the logical conclusion about the grandparent relationship and evaluate it."

            messages = [{"role": "system", "content": KINSHIP_FEEDBACK_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
            batch_prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

        # 추론 수행
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # 5. 결과 누적 및 '전체 리스트' 저장
        for row, output in zip(batch_records, outputs):
            feedback_json = extract_json(output.outputs[0].text.strip())
            
            result_entry = row.copy()
            result_entry["gold_feedback"] = feedback_json if isinstance(feedback_json, dict) else {"error": "parse_fail"}
            
            # 기존 리스트에 새 결과를 하나씩 추가
            existing_results.append(result_entry)

        # 매 배치 완료 후 전체(기존+신규)를 파일에 덮어씀
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 Completed. Total items in {OUTPUT_FILE}: {len(existing_results)}")

if __name__ == "__main__":
    main()