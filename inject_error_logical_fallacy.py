import pandas as pd
from tqdm import tqdm
import json
import os
import re
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

system_prompt = """You are an expert in logical reasoning simulation, tasked with generating "Hard Negative" reasoning steps.
Your goal is to replace a single, correct reasoning step with a 'Logical Fallacy' that looks plausible but is fundamentally flawed.

### Your Goal:
Generate a `(Logical)` step that contains a specific reasoning error. The error should not be a simple negation. It should mimic how a human or LLM might misinterpret data, make a calculation slip, or jump to a hasty conclusion.

### Error Definition: 'Logical Fallacy' (Hard Negative Mode)
The step must accept the facts from previous steps as true but draw an incorrect conclusion from them.

**Target Error Sub-types (Choose one appropriate for the context):**
1.  **Numeric/Date Misinterpretation:**
    * *Incorrect:* "Since 1964 is a larger number than 1932, the person born in 1964 is older." (Confusing birth year magnitude with age).
    * *Incorrect:* "The meeting was in 1963-64, so the answer is 1965." (Choosing a date slightly outside the range or averaging incorrectly).
2.  **Entity/Category Confusion:**
    * *Incorrect:* "The composer is part of the Rose Consort, so he must be the founder of it." (Confusing member with founder).
    * *Incorrect:* "Ferrari 250 GTO is the car, so the song is about the Pontiac GTO." (Confusing different entities with similar names).
3.  **Hasty Generalization / Causality Error:**
    * *Incorrect:* "The director was born in the US, so the film must be a Hollywood blockbuster." (Unwarranted assumption).
    * *Incorrect:* "He acted in 3 films, so he is the most prolific actor in history." (Exaggeration).
4.  **Counting/Grouping Error:**
    * *Incorrect:* "The list has A, B, and C. A and B are a team, so I count them as 1. Total is 2." (Arbitrary grouping logic).
5.  **Set/Subset Logic Error:**
    * *Incorrect:* "Director A is Indian and Director B is American. Since they speak different languages, they cannot both be from the 'Northern Hemisphere'." (Confusing cultural traits with geographic subsets).
    * *Incorrect:* "Rochester is in Western New York. Since Albany is in Eastern New York, Rochester is not part of New York State." (Confusing distinct regions with the parent state).
6.  **Temporal/Causal Reversal:**
    * *Incorrect:* "The 1939 film was released before the 1921 film because 1939 is a higher version number." (Confusing dates with versioning).
    * *Incorrect:* "Since the 2000 remake is more famous, the 1950 original was likely based on the 2000 version." (Reversing cause and effect).

### Constraints for 'Logical Fallacy' Generation (CRITICAL)
1. **NO External Knowledge/Subjective Opinion:**
   - Do NOT construct the error based on "cultural significance", "fame", "symbolism", or facts not found in the passages.
   - The error must be derived PURELY from the provided facts. You are allowed to twist the logic (A > B), but NOT the facts (A's birth year).
   - The error must come from misinterpreting the *mechanics* of the facts provided (e.g., set theory, arithmetic, causal direction).

2. **Stay "On-topic":**
   - The corrupted step must still *attempt* to answer the specific question asked. Do not drift into discussing related trivia.
   - *Bad Example:* "The film was released in 1995, which was a great year for cinema..." (Off-topic)
   - *Good Example:* "The film was released in 1995, which implies it is older than the 1920 film because numbers cycle every century." (Logical Fallacy)

### Input Format:
You will receive:
1.  **Question:** The user's original question.
2.  **Retrieved Passages:** Contextual information.
3.  **Ideal Reasoning Steps:** The correct, multi-step reasoning.
4.  **Target Step to Corrupt:** The specific `(Logical)` step you must replace.

### Output Format:
-   Output **ONLY** the single, new, erroneous reasoning step.
-   The step MUST start with the "Step X:" prefix and end with `(Logical)`.
-   **DO NOT** simply add "not" to the correct sentence. Make the error specific and descriptive.

---
EXAMPLES
---

Question: "Are Markhal and Now Khaleh-Ye Jafari located in the same country?"

Retrieved Passages: 
"Passage 1: Markhal... is a village in... Gilan Province, Iran."
"Passage 2: Now Khaleh-ye Jafari... is a village in... Gilan Province, Iran."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, Markhal is located in Iran. (Attribution)",
 "Step 2: According to Passage 2, Now Khaleh-Ye Jafari is located in Iran. (Attribution)",
 "Step 3: Iran (from Step 1) and Iran (from Step 2) are the same country. (Logical)"
]

Target Step to Corrupt: 
Step 3

Output: 
Step 3: Although both are in Iran, Passage 1 and Passage 2 describe distinct villages, so logically they cannot be considered the "same" location or country context in a strict geographical sense. (Logical)

---

Question: "Who is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven?"

Retrieved Passages: 
"Passage 1: James Tuchet, 3rd Earl of Castlehaven... was the son of Mervyn Tuchet, 2nd Earl..." 
"Passage 2: Mervyn Tuchet, 2nd Earl... A son of George Tuchet, 1st Earl..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the father of James Tuchet, 3rd Earl of Castlehaven is Mervyn Tuchet, 2nd Earl of Castlehaven. (Attribution)", 
 "Step 2: According to Passage 2, the father of Mervyn Tuchet, 2nd Earl of Castlehaven (from Step 1) is George Tuchet, 1st Earl of Castlehaven. (Attribution)", 
 "Step 3: Therefore, the person found in Step 2, George Tuchet, 1st Earl of Castlehaven, is the paternal grandfather. (Logical)" 
]

Target Step to Corrupt: 
Step 3

Output: 
Step 3: Since titles are inherited directly from father to son, Mervyn Tuchet (the 2nd Earl) holds the primary paternal lineage, making him the functional paternal grandfather figure in the succession line. (Logical)

---

Question: "Who lived longer, Ignace Matondo Kwa Nzambi or Claudia Zobel?"

Retrieved Passages: 
"Passage 1: Ignace Matondo Kwa Nzambi (12 April 1932 – 9 September 2011)..." 
"Passage 2: Claudia Zobel (February 27, 1964 – February 10, 1984)..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, Ignace Matondo Kwa Nzambi was born in 1932 and died in 2011. (Attribution)", 
 "Step 2: Based on the dates from Step 1, Ignace Matondo Kwa Nzambi lived for 79 years. (Logical)", 
 "Step 3: According to Passage 2, Claudia Zobel was born in 1964 and died in 1984. (Attribution)", 
 "Step 4: Based on the dates from Step 3, Claudia Zobel lived for 19 years. (Logical)", 
 "Step 5: Comparing the lifespans from Step 2 (79 years) and Step 4 (19 years), Ignace Matondo Kwa Nzambi lived longer. (Logical)" 
]

Target Step to Corrupt: 
Step 5

Output: 
Step 5: Since Claudia Zobel was born in 1964, which is a larger number than 1932, she belongs to a more recent generation and thus represents a longer span of modern history. (Logical)
""".strip()

def generate_response(tokenizer, llm, messages):
    """Chat template 기반 gpt-oss-120b 응답 생성"""
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )

    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    response = outputs[0].outputs[0].text
    return response.split("assistantfinal")[-1].strip()

def parse_step(step_str: str) -> Tuple[str, str]:
    """
    "Step 1: Do something. (Attribution)" -> ("Step 1:", "(Attribution)")
    """
    step_str = step_str.strip()
    
    # Regex 수정: (Attribution) 또는 (Logical) 레이블을 정확히 찾음
    match = re.match(r"^(Step\s*\d+:)(.*)(\((Attribution|Logical)\))$", step_str, re.DOTALL)
    
    if match:
        prefix = match.group(1).strip()
        label = match.group(3).strip()
        return prefix, label
    else:
        # 레이블 파싱 실패 시 기본값
        print(f"⚠️ Warning: Could not parse label for step: {step_str}. Defaulting to (Logical).")
        prefix_match = re.match(r"^(Step\s*\d+:)", step_str)
        prefix = prefix_match.group(1).strip() if prefix_match else f"Step {step_str.split(':')[0]}:"
        return prefix, "(Logical)"

def save_results(data: List[Dict[str, Any]], filepath: str):
    """결과를 JSON 파일로 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main(args):
    # ✅ 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        max_model_len=3000,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    input_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_ideal_steps_passage_mapped.json"
    output_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_logical_fallacy_2.json" 
    
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {input_filepath}")
    except Exception as e:
        print(f"Error loading data from {input_filepath}: {e}")
        return

    # 1: 기존 결과 로드
    results = []
    processed_ids = set() # (question, target_index)를 저장할 Set
    
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
            # 2. 이미 처리된 ID를 Set에 저장
            for res in results:
                if 'question' in res and 'corrupted_step_index' in res:
                    processed_ids.add((res['question'], res['corrupted_step_index']))
            print(f"Loaded {len(results)} existing results from {output_filepath}. Resuming...")
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_filepath}. Starting fresh. Error: {e}")
            results = []
            processed_ids = set()
    
    # =======================================================
    # 🔹 루프: 각 항목에 대해 오류 주입
    # =======================================================
    # 1. 각 '항목'(질문)에 대해 루프
    for item in tqdm(data, desc="Injecting 'Logical Fallacy' errors"):
        try:
            question = item['question']
            passages = item['retrieved_passages']
            ideal_steps = item['ideal_steps']
            
            # 2. 1부터 K까지 각 스텝을 타겟으로 내부 루프
            for i, step_text in enumerate(ideal_steps):
                
                # 1-based index로 변환
                target_index = i + 1
                
                # (Logical) 스텝이 아니면 건너뛰기
                if not step_text.strip().endswith("(Logical)"):
                    continue
                
                # 3. 개별 스텝 주입(try-except로 감싸서 한 스텝이 실패해도 다음 스텝으로 넘어가도록 함)
                try:
                    # 3: 이미 처리된 항목인지 확인
                    current_id = (question, target_index)
                    if current_id in processed_ids:
                        continue # 이미 처리되었으므로 스킵
                    
                    # 컨텍스트로 사용할 passages 포맷팅
                    passages_context = "\n".join(f"Passage {i+1}: {p}" for i, p in enumerate(passages))

                    # LLM에 전달할 프롬프트 구성
                    user_prompt = f"""Question: {question}

Retrieved Passages:
{passages_context}

Ideal Reasoning Steps:
{json.dumps(ideal_steps, indent=2, ensure_ascii=False)}

Target Step to Corrupt:
Step {target_index}
""".strip()

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]

                    # 오류가 주입된 스텝 생성
                    corrupted_step = generate_response(tokenizer, llm, messages).strip().strip(",")

                    # 생성된 스텝 검증 (간단)
                    # Logical Fallacy는 반드시 (Logical)로 끝나야 함
                    if not (corrupted_step.startswith(f"Step {target_index}:") and 
                            corrupted_step.endswith("(Logical)")):
                        print(f"\n⚠️ Warning: Model output format mismatch for Q: {question[:50]}... (Step {target_index})")
                        print(f"  Expected prefix: 'Step {target_index}:'")
                        print(f"  Expected suffix: '(Logical)'") # Logical만 허용
                        print(f"  Got: {corrupted_step}")
                        # 포맷이 망가졌으면 이 target_index는 건너뛰기
                        continue

                    # 새 오류 스텝 생성 (N+1 이후 스텝은 제외)
                    base_steps = ideal_steps[:target_index-1] 
                    corrupted_steps = base_steps + [corrupted_step]

                    # 결과 저장
                    new_item = item.copy()
                    new_item['corrupted_steps'] = corrupted_steps
                    new_item['corrupted_step_index'] = target_index # 1-based
                    # Error type 변경
                    new_item['error_type'] = 'Logical Fallacy' 
                    results.append(new_item)

                    # 4: 방금 처리한 ID를 Set에 추가
                    processed_ids.add(current_id)

                    # 중간 저장 (5개마다)
                    if len(results) % 5 == 0:
                        save_results(results, output_filepath)

                except Exception as e:
                    print(f"\nFailed to process question {question[:50]}... on Step {target_index}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue # 이 스텝은 실패했으므로 다음 target_index로 넘어감

        except Exception as e:
            # item 로딩 등 외부 루프에서 에러 발생 시
            print(f"\nFailed to process item {question[:50]}...: {e}")
            import traceback
            traceback.print_exc()
            continue # 이 항목(질문)은 실패했으므로 다음 항목으로 넘어감

    # 최종 저장
    save_results(results, output_filepath)
    print(f"✅ Completed error injection. Total {len(results)} corrupted items saved to {output_filepath}.")

if __name__ == "__main__":
    # 🔴 [수정됨] ArgumentParser 설명 변경
    parser = argparse.ArgumentParser(description="Inject 'Logical Fallacy' errors into reasoning steps.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., '2wiki') to determine input/output filenames.")
    parser.add_argument("--model_name", type=str, 
                        default="/workspace/hf_transformers/gpt-oss-120b",
                        help="Path to the HuggingFace model directory.")
    
    args = parser.parse_args()
    main(args)
