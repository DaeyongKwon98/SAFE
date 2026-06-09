import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import pandas as pd
import re

# --- 1. System Prompt for Paraphrasing with Constraints ---
PARAPHRASE_SYSTEM_PROMPT = """You are an expert data augmentor specializing in NLP robustness.
Your task is to rewrite the given "Reasoning Step" into a new sentence that preserves the exact same meaning and logic but changes the format, style, or structure (e.g., date formats, units, active/passive voice).

## Strict Constraints (MUST FOLLOW):

### Citation & Reference Preservation (CRITICAL): (Applied for (Attribution) steps.)
- You must identify the phrase used to cite a passage (e.g., "According to Passage X", "Passage X states", "Based on Passage X").
- DO NOT CHANGE this citation phrase. Keep it exactly as it appears in the input.
- Only paraphrase the factual content that follows the citation.

### Logical Step References: (Applied for (Logical) steps.)
- If the step refers to previous steps (e.g., "Step 1", "from Step 2"), you MUST KEEP these references exactly as they appear.

### Label Tags:
- You MUST KEEP the classification tag (e.g., "(Attribution)", "(Logical)") at the very end of the sentence.

### Semantic Preservation (CRITICAL - DO NOT FAIL THIS):
- Absolute Factual Consistency: The paraphrased output must convey the exact same information as the input.
- No Information Loss: Do not simplify specific roles into general terms (e.g., do not change "writer and director" to "creator").
- No Information Gain: Do not add details that were not in the original text (e.g., do not change "worked with" to "produced" or "created").
- Precise Relationships: Do not alter the nature of relationships (e.g., NEVER change "father" to "gave birth to", never change "marrying" to "passing as").

### Paraphrasing Targets (Apply these while obeying constraint #4):
- Dates: Change formats (e.g., "14 January 1987" → "1987-01-14", "Jan 14, '87").
- Units: Change units or notation (e.g., "1.5 kg" → "1500 grams", "50%" → "half").
- Sentence Structure: Switch between Active and Passive voice, or change the word order of the fact.
- Verbs: Use synonyms for the main action ONLY IF they are exact substitutes (e.g., "died on" → "passed away on").

## Output:
Output ONLY the rewritten sentence. Do not add explanations.

## Examples:

Input: "Step 1. According to Passage 2, the mother of Princess Alexandrine is Princess Sophie. (Attribution)"
Output: "Step 1. According to Passage 2, Princess Sophie is identified as the parent who gave birth to Princess Alexandrine. (Attribution)"

Input: "Step 2. Passage 1 states that he was born on Jan 14, 1987. (Attribution)"
Output: "Step 2. Passage 1 states that his date of birth is recorded as 1987-01-14. (Attribution)"

Input: "Step 3. Since 1928 (from Step 2) is later than 1890 (from Step 4), Veljko is younger. (Logical)"
Output: "Step 3. Because the year 1928 (from Step 2) comes after the year 1890 (from Step 4), Veljko is considered to be younger. (Logical)
""".strip()

def generate_paraphrase(tokenizer, llm, original_step):
    """Generate a paraphrased version of the reasoning step."""

    user_prompt = f"""Target Reasoning Step:
"{original_step}"

Rewritten Sentence:"""

    messages = [
        {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )

    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    answer = outputs[0].outputs[0].text
    return answer.split("assistantfinal")[-1].strip().strip(chr(34))

def validate_consistency(original, paraphrased):
    """
    Rule-based 검증 함수 (수정됨)
    
    [공통 검증]
    1. 문장 시작 부분의 Step 번호(K)가 동일한지 확인
    2. 문장 끝 부분의 태그((Attribution)/(Logical))가 동일한지 확인

    [조건부 검증]
    3. (Attribution)인 경우: Passage 번호들의 등장 횟수와 값이 동일한지 확인 (Step 참조는 달라도 허용)
    4. (Logical)인 경우: Step 번호들의 등장 횟수와 값이 동일한지 확인 (Passage 참조는 달라도 허용)
    """
    try:
        # 공백 제거 및 정규화
        original = original.strip()
        paraphrased = paraphrased.strip()

        # ---------------------------------------------------------
        # 1. 시작 Step 번호 검증 ("Step K" 형태)
        # ---------------------------------------------------------
        orig_start = re.search(r"^Step\s+(\d+)", original, re.IGNORECASE)
        para_start = re.search(r"^Step\s+(\d+)", paraphrased, re.IGNORECASE)

        if not orig_start or not para_start:
            # print("Fail: Start Step format mismatch")
            return False 
        
        if orig_start.group(1) != para_start.group(1):
            # print(f"Fail: Start Step number mismatch ({orig_start.group(1)} vs {para_start.group(1)})")
            return False

        # ---------------------------------------------------------
        # 2. 끝 태그 검증 ((Attribution) 또는 (Logical))
        # ---------------------------------------------------------
        tag_pattern = r"\((Attribution|Logical)\)\s*[.,;]?\s*$"
        orig_tag = re.search(tag_pattern, original, re.IGNORECASE)
        para_tag = re.search(tag_pattern, paraphrased, re.IGNORECASE)

        if not orig_tag or not para_tag:
            # print("Fail: End Tag format mismatch")
            return False

        if orig_tag.group(1).lower() != para_tag.group(1).lower():
            # print(f"Fail: End Tag mismatch ({orig_tag.group(1)} vs {para_tag.group(1)})")
            return False

        tag_type = orig_tag.group(1).lower() # 'attribution' or 'logical'

        # ---------------------------------------------------------
        # 3. 조건부 검증 (Tag 타입에 따라 다름)
        # ---------------------------------------------------------
        
        # CASE A: (Attribution) -> Passage 번호 엄격 검사
        if tag_type == 'attribution':
            orig_passages = sorted(re.findall(r"Passage\s+(\d+)", original, re.IGNORECASE))
            para_passages = sorted(re.findall(r"Passage\s+(\d+)", paraphrased, re.IGNORECASE))

            if orig_passages != para_passages:
                # print(f"Fail (Attribution): Passage mismatch {orig_passages} vs {para_passages}")
                return False
            
            # Attribution에서는 "(from Step 1)" 같은 참조가 사라져도 되므로 Step 검사는 생략

        # CASE B: (Logical) -> Step 번호 엄격 검사 (참조 유지가 핵심)
        elif tag_type == 'logical':
            orig_steps = sorted(re.findall(r"Step\s+(\d+)", original, re.IGNORECASE))
            para_steps = sorted(re.findall(r"Step\s+(\d+)", paraphrased, re.IGNORECASE))

            if orig_steps != para_steps:
                # print(f"Fail (Logical): Step ref mismatch {orig_steps} vs {para_steps}")
                return False
            
            # Logical에서는 Passage 인용이 드물거나 덜 중요하므로 Passage 검사는 완화 (필요시 추가 가능)

        return True

    except Exception as e:
        print(f"Validation Error: {e}")
        return False

def run_paraphrase_generation(df, model_name, save_path=None):
    # 1. Resume Logic (중단된 지점부터 이어하기)
    if save_path and os.path.exists(save_path):
        print(f"Found existing progress file at {save_path}. Resuming...")
        try:
            saved_df = pd.read_json(save_path)
            
            # 컬럼 확인 및 병합
            if 'paraphrased_step' in saved_df.columns:
                if 'paraphrased_step' not in df.columns:
                    df['paraphrased_step'] = None
                
                # 기존에 생성된 내용을 현재 데이터프레임에 업데이트
                df.update(saved_df[['paraphrased_step']])
                
                processed_count = df['paraphrased_step'].notna().sum()
                print(f"Resumed! {processed_count}/{len(df)} items already processed.")
        except ValueError as e:
            print(f"Error loading existing file: {e}. Starting from scratch.")
            df['paraphrased_step'] = None
    else:
        print("Starting from scratch.")
        df['paraphrased_step'] = None

    # 모든 작업이 완료되었는지 확인
    if df['paraphrased_step'].notna().all():
        print("All items are already processed!")
        return df

    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        max_model_len=3000,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    # 원본 데이터가 있는 컬럼명을 확인하세요. (여기서는 'current_step'으로 가정)
    for idx in tqdm(range(len(df)), desc="Generating paraphrases"):
        # 이미 처리된 행은 건너뜀
        if pd.notna(df.at[idx, 'paraphrased_step']) and df.at[idx, 'paraphrased_step'] != "":
            continue

        try:
            original_text = df.iloc[idx]['current_step']
            
            # 텍스트 유효성 검사
            if not isinstance(original_text, str) or len(original_text) < 5:
                df.at[idx, 'paraphrased_step'] = original_text
                continue

            # --- Retry Logic Start ---
            max_retries = 5
            success = False
            last_generated_text = "" # 실패 로그 출력을 위해 마지막 생성문 저장

            for attempt in range(max_retries):
                # 생성 수행
                paraphrased = generate_paraphrase(tokenizer, llm, original_text)
                
                # 검증 수행
                if validate_consistency(original_text, paraphrased):
                    df.at[idx, 'paraphrased_step'] = paraphrased
                    success = True
                    break # 성공 시 재시도 루프 탈출
                else:
                    # 이번 시도 실패 시 기록만 해두고 다음 시도로 넘어감
                    last_generated_text = paraphrased
                    # (선택사항) 디버깅을 위해 재시도 중임을 출력하고 싶다면 주석 해제
                    # print(f"Retry {attempt+1}/{max_retries} for index {idx}...")

            # 5번 다 실패했을 경우에만 로그 출력
            if not success:
                print(f"\n[Validation Failed ID: {idx} after {max_retries} attempts]")
                print(f"Orig: {original_text}")
                print(f"Last Para: {last_generated_text}")
                # 실패했으므로 저장하지 않음 (None 유지)
            
            # --- Retry Logic End ---

            # 중간 저장 (5개 마다)
            if idx % 5 == 0 and save_path:
                df.to_json(save_path, orient="records", indent=2, force_ascii=False)
                
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            # 에러 발생 시에도 안전하게 저장 시도
            if save_path:
                df.to_json(save_path, orient="records", indent=2, force_ascii=False)

    # 최종 저장
    if save_path:
        df.to_json(save_path, orient="records", indent=2, force_ascii=False)

    return df

if __name__ == "__main__":
    # --- 설정 (경로 확인 필수) ---
    MODEL_PATH = "/workspace/hf_transformers/gpt-oss-120b"
    # INPUT_FILE = "/workspace/daeyong/second_finetuning_data/correct.json"
    INPUT_FILE = "/workspace/daeyong/ideal_steps/combined_correct_steps.json"
    # OUTPUT_FILE = "/workspace/daeyong/second_finetuning_data/correct_paraphrased.json"
    OUTPUT_FILE = "/workspace/daeyong/ideal_steps/combined_correct_steps_paraphrased.json"
      
    # 데이터 로드
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE)
    
    # --- 실행 ---
    result_df = run_paraphrase_generation(df, model_name=MODEL_PATH, save_path=OUTPUT_FILE)