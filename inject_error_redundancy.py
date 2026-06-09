import pandas as pd
from tqdm import tqdm
import json
import os
import re
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# system_prompt = """You are an expert in logical reasoning, tasked with intentionally introducing a specific logical error into a reasoning steps.

# Your Goal: Replace a single, correct reasoning step with a 'Redundancy' error.

# Error Definition: 'Redundancy'
# A 'Redundancy' error is a reasoning step that repeats information or conclusions from previous steps without providing any significant new progression. It stalls the reasoning process by repeating what is already known.
# - MUST be a repetition or simple restatement of a conclusion from a *previous* step (e.g., Step N-1, N-2, etc.).
# - MUST adopt the label (Attribution or Logical) from the previous step that is being repeated.
# - MUST NOT be a repetition of the *original* step it is replacing (as that would be correct).
# - MUST NOT introduce *any* new 'off-topic' information, even if it's about an entity from a previous step (e.g., If Step 1 is 'Elvis is the performer', the error *cannot* be 'Elvis was written by...').
# - The error's content MUST be inspired *only* by steps *before* the target step. Do NOT use the original content of the step being replaced as inspiration.
# - The step MUST be a direct statement of fact or logic (e.g., "According to..."), NOT a meta-commentary about the reasoning (e.g., "This repeats Step 1..." or "...which was already established").

# Input Format:
# You will receive:
# 1. Question: The user's original question.
# 2. Ideal Reasoning Steps: The correct, multi-step reasoning.
# 3. Target Step to Corrupt: The specific step from the ideal steps that you must replace. (ex. Step 1)

# Output Format:
# - You MUST output only the single, new, erroneous reasoning step.
# - The new step MUST be formatted exactly like the target step, including the "Step X:" prefix and the "(Label)" suffix.

# ---
# EXAMPLES
# ---

# Question: "What was the first EP of the singer who recorded What Ifs?"

# Ideal Reasoning Steps:
# [
#  "Step 1: According to Passage 1, the singer who recorded \"What Ifs\" is Kane Brown. (Attribution)",
#  "Step 2: According to Passage 2, the first EP of Kane Brown (from Step 1) is titled \"Closer\". (Attribution)",
#  "Step 3: Therefore, the EP found in Step 2, \"Closer\", is the answer. (Logical)"
# ]

# Target Step to Corrupt:
# Step 2

# Output:
# Step 2: According to Passage 1, Kane Brown is the singer who performed "What Ifs". (Attribution)

# ---

# Question: "What Bengali political film was censored by one of the most powerful film censor boards in the world?"

# Ideal Reasoning Steps: 
# [ 
#  "Step 1: According to Passage 1, the Central Board of Film Certification (CBFC) is one of the most powerful film censor boards in the world. (Attribution)", 
#  "Step 2: According to Passage 2, the Bengali political film 'Kangal Malsat' was censored by the Central Board of Film Certification (from Step 1). (Attribution)", 
#  "Step 3: Therefore, the film found in Step 2, 'Kangal Malsat', is the answer. (Logical)" 
# ]

# Target Step to Corrupt: 
# Step 2

# Output: 
# Step 2: According to Passage 1, the Central Board of Film Certification (CBFC) is one of the most powerful film censor boards in the world. (Attribution)

# ---

# Question: "Are David Nixon and Charlie Chaplin from the same country originally?"

# Ideal Reasoning Steps: 
# [ 
#  "Step 1: According to Passage 1, David Nixon is an American film director. (Attribution)", 
#  "Step 2: According to Passage 2, Charlie Chaplin was an English comic actor. (Attribution)", 
#  "Step 3: Based on Step 1 (American) and Step 2 (English), their countries of origin are not the same. (Logical)" 
# ]

# Target Step to Corrupt: 
# Step 3

# Output: 
# Step 3: According to Passage 1, David Nixon is an American film director. (Attribution)

# ---

# Question: "Who is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven?"

# Ideal Reasoning Steps: 
# [ 
#  "Step 1: According to Passage 1, the father of James Tuchet, 3rd Earl of Castlehaven is Mervyn Tuchet, 2nd Earl of Castlehaven. (Attribution)", 
#  "Step 2: According to Passage 2, the father of Mervyn Tuchet, 2nd Earl of Castlehaven (from Step 1) is George Tuchet, 1st Earl of Castlehaven. (Attribution)", 
#  "Step 3: Therefore, the person found in Step 2, George Tuchet, 1st Earl of Castlehaven, is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven. (Logical)" 
# ]

# Target Step to Corrupt: 
# Step 3

# Output: 
# Step 3: Based on Passage 2, George Tuchet, 1st Earl of Castlehaven is the father of the person found in Step 1. (Attribution)

# ---

# Question: "Who lived longer, Ignace Matondo Kwa Nzambi or Claudia Zobel?"

# Ideal Reasoning Steps: 
# [ 
#  "Step 1: According to Passage 1, Ignace Matondo Kwa Nzambi was born on 12 April 1932 and died on 9 September 2011. (Attribution)", 
#  "Step 2: Based on the dates from Step 1, Ignace Matondo Kwa Nzambi lived for 79 years. (Logical)", 
#  "Step 3: According to Passage 2, Claudia Zobel was born on February 27, 1964 and died on February 10, 1984. (Attribution)", 
#  "Step 4: Based on the dates from Step 3, Claudia Zobel lived for 19 years. (Logical)", 
#  "Step 5: Comparing the lifespans from Step 2 (79 years) and Step 4 (19 years), Ignace Matondo Kwa Nzambi lived longer. (Logical)" 
# ]

# Target Step to Corrupt: 
# Step 4

# Output: 
# Step 4: According to the dates in Step 1, Ignace Matondo Kwa Nzambi’s lifespan was 79 years. (Logical)
# """.strip()

system_prompt = """You are an expert in logical reasoning, tasked with intentionally introducing a specific logical error into a reasoning steps.

Your Goal: Replace a single, correct reasoning step with a 'Redundancy' error.

Error Definition: 'Redundancy'
A 'Redundancy' error is a reasoning step that repeats information or conclusions from previous steps without providing any significant new progression. It mimics how LLMs sometimes get stuck in a loop or unnecessarily re-state found facts.

Your generated step must reflect one of the following realistic behaviors:
1. **Paraphrasing:** Re-stating a fact from a previous step using different words or sentence structure.
2. **Unnecessary Confirmation:** Explicitly confirming a fact that was already established (e.g., "Thus, it is confirmed that [Fact from Step 1]...").
3. **Circular Reference:** Referring back to a previous step to state the exact same conclusion again.

Constraints:
- MUST be a repetition of a *previous* step (e.g., Step N-1, N-2).
- MUST adopt the label (Attribution or Logical) from the previous step that is being repeated.
- MUST NOT be a repetition of the *original* step it is replacing.
- MUST NOT introduce new 'off-topic' information.
- The step MUST be a direct statement of fact or logic, NOT a meta-plan (e.g., Do NOT say "I will now check Step 1 again").

Input Format:
You will receive:
1. Question: The user's original question.
2. Ideal Reasoning Steps: The correct, multi-step reasoning.
3. Target Step to Corrupt: The specific step from the ideal steps that you must replace.

(Note: Retrieved Passages are intentionally omitted. You must ONLY use information from the 'Ideal Reasoning Steps' to create a redundancy.)

Output Format:
- You MUST output only the single, new, erroneous reasoning step.
- The new step MUST be formatted exactly like the target step, including the "Step X:" prefix and the "(Label)" suffix.

---
EXAMPLES
---

Question: "What was the first EP of the singer who recorded What Ifs?"

Ideal Reasoning Steps:
[
 "Step 1: According to Passage 1, the singer who recorded \"What Ifs\" is Kane Brown. (Attribution)",
 "Step 2: According to Passage 2, the first EP of Kane Brown (from Step 1) is titled \"Closer\". (Attribution)",
 "Step 3: Therefore, the EP found in Step 2, \"Closer\", is the answer. (Logical)"
]

Target Step to Corrupt:
Step 2

Output:
Step 2: According to Passage 1, the artist responsible for the song "What Ifs" is identified as Kane Brown. (Attribution)

---

Question: "What Bengali political film was censored by one of the most powerful film censor boards in the world?"

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the Central Board of Film Certification (CBFC) is one of the most powerful film censor boards in the world. (Attribution)", 
 "Step 2: According to Passage 2, the Bengali political film 'Kangal Malsat' was censored by the Central Board of Film Certification (from Step 1). (Attribution)", 
 "Step 3: Therefore, the film found in Step 2, 'Kangal Malsat', is the answer. (Logical)" 
]

Target Step to Corrupt: 
Step 2

Output: 
Step 2: Based on the information found in Passage 1, the CBFC is indeed considered a highly powerful film censor board globally. (Attribution)

---

Question: "Are David Nixon and Charlie Chaplin from the same country originally?"

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, David Nixon is an American film director. (Attribution)", 
 "Step 2: According to Passage 2, Charlie Chaplin was an English comic actor. (Attribution)", 
 "Step 3: Based on Step 1 (American) and Step 2 (English), their countries of origin are not the same. (Logical)" 
]

Target Step to Corrupt: 
Step 3

Output: 
Step 3: David Nixon has been identified as an American film director according to Passage 1. (Attribution)

---

Question: "Who is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven?"

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the father of James Tuchet, 3rd Earl of Castlehaven is Mervyn Tuchet, 2nd Earl of Castlehaven. (Attribution)", 
 "Step 2: According to Passage 2, the father of Mervyn Tuchet, 2nd Earl of Castlehaven (from Step 1) is George Tuchet, 1st Earl of Castlehaven. (Attribution)", 
 "Step 3: Therefore, the person found in Step 2, George Tuchet, 1st Earl of Castlehaven, is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven. (Logical)" 
]

Target Step to Corrupt: 
Step 3

Output: 
Step 3: Thus, it is clear from Passage 2 that Mervyn Tuchet, the 2nd Earl of Castlehaven, is the son of George Tuchet. (Attribution)

---

Question: "Who lived longer, Ignace Matondo Kwa Nzambi or Claudia Zobel?"

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, Ignace Matondo Kwa Nzambi was born on 12 April 1932 and died on 9 September 2011. (Attribution)", 
 "Step 2: Based on the dates from Step 1, Ignace Matondo Kwa Nzambi lived for 79 years. (Logical)", 
 "Step 3: According to Passage 2, Claudia Zobel was born on February 27, 1964 and died on February 10, 1984. (Attribution)", 
 "Step 4: Based on the dates from Step 3, Claudia Zobel lived for 19 years. (Logical)", 
 "Step 5: Comparing the lifespans from Step 2 (79 years) and Step 4 (19 years), Ignace Matondo Kwa Nzambi lived longer. (Logical)" 
]

Target Step to Corrupt: 
Step 4

Output: 
Step 4: As calculated previously, the total lifespan of Ignace Matondo Kwa Nzambi is confirmed to be 79 years. (Logical)
""".strip()

def generate_response(tokenizer, llm, messages):
    """Chat template 기반 gpt-oss-120b 응답 생성"""
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

# def main(args):
#     # ✅ 모델 로드

#     # ✅ 데이터 로드
#     input_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_ideal_steps_passage_mapped.json"
#     output_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_redundancy.json"
    
#     try:
#         with open(input_filepath, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         print(f"Loaded {len(data)} items from {input_filepath}")
#     except Exception as e:
#         print(f"Error loading data from {input_filepath}: {e}")
#         return

#     # 💡 [수정됨] 1: 기존 결과 로드
#     results = []
#     processed_ids = set() # (question, target_index)를 저장할 Set
    
#     if os.path.exists(output_filepath):
#         try:
#             with open(output_filepath, "r", encoding="utf-8") as f:
#                 results = json.load(f)
#             # 2. 이미 처리된 ID를 Set에 저장
#             # (질문, 타겟스텝인덱스) 조합이 고유 키가 됩니다.
#             for res in results:
#                 if 'question' in res and 'corrupted_step_index' in res:
#                     processed_ids.add((res['question'], res['corrupted_step_index']))
#             print(f"Loaded {len(results)} existing results from {output_filepath}. Resuming...")
#         except Exception as e:
#             print(f"Warning: Could not load existing results from {output_filepath}. Starting fresh. Error: {e}")
#             results = []
#             processed_ids = set()
    
#     # =======================================================
#     # 🔹 루프: 각 항목에 대해 오류 주입
#     # =======================================================
#     # 1. 각 '항목'(질문)에 대해 루프
#     for item in tqdm(data, desc="Injecting 'Redundancy' errors"):
#         try:
#             question = item['question']
#             passages = item['retrieved_passages']
#             ideal_steps = item['ideal_steps']
#             total_steps = len(ideal_steps)
            
#             # 2. (요청 1) 1부터 K까지 각 스텝을 타겟으로 내부 루프
#             for target_index in range(1, total_steps + 1):
#                 # target_index는 1-based (1, 2, ..., K)
                
#                 # (수정) Redundancy 오류는 정의상 Step 1에는 주입할 수 없음 (반복할 이전 스텝이 없음)
#                 if target_index == 1:
#                     continue

#                 # 3. 개별 스텝 주입(try-except로 감싸서 한 스텝이 실패해도 다음 스텝으로 넘어가도록 함)
#                 try:
#                     # 💡 [수정됨] 3: 이미 처리된 항목인지 확인
#                     current_id = (question, target_index)
#                     if current_id in processed_ids:
#                         continue # 이미 처리되었으므로 스킵
                    
#                     # 컨텍스트로 사용할 passages 포맷팅
#                     passages_context = "\n".join(f"Passage {i+1}: {p}" for i, p in enumerate(passages))

#                     # LLM에 전달할 프롬프트 구성
#                     user_prompt = f"""Question: {question}

# Retrieved Passages:
# {passages_context}

# Ideal Reasoning Steps:
# {json.dumps(ideal_steps, indent=2, ensure_ascii=False)}

# Target Step to Corrupt:
# Step {target_index}
# """.strip()

#                     messages = [
#                         {"role": "system", "content": system_prompt},
#                         {"role": "user", "content": user_prompt}
#                     ]

#                     # 오류가 주입된 스텝 생성
#                     corrupted_step = generate_response(tokenizer, llm, messages).strip().strip(",")

#                     # 생성된 스텝 검증 (간단)
#                     if not (corrupted_step.startswith(f"Step {target_index}:") and 
#                             (corrupted_step.endswith("(Attribution)") or corrupted_step.endswith("(Logical)"))):
#                         print(f"\n⚠️ Warning: Model output format mismatch for Q: {question[:50]}... (Step {target_index})")
#                         print(f"  Expected prefix: 'Step {target_index}:'")
#                         print(f"  Expected suffix: '(Attribution)' or '(Logical)'")
#                         print(f"  Got: {corrupted_step}")
#                         # 포맷이 망가졌으면 이 target_index는 건너뛰기
#                         continue

#                     # (요청 2) 새 오류 스텝 생성 (N+1 이후 스텝은 제외)
#                     # 1. 타겟 스텝 이전의 스텝들 (0-based 인덱싱)
#                     base_steps = ideal_steps[:target_index-1] 
#                     # 2. [이전 스텝들] + [새로 생성된 오류 스텝]
#                     corrupted_steps = base_steps + [corrupted_step]

#                     # 결과 저장
#                     new_item = item.copy()
#                     new_item['corrupted_steps'] = corrupted_steps
#                     new_item['corrupted_step_index'] = target_index # 1-based
#                     # 🔴 [수정됨] Error type 변경
#                     new_item['error_type'] = 'Redundancy'
#                     # (요청 3) 'injected_error' 키는 저장하지 않음
#                     results.append(new_item)

#                     # 💡 [수정됨] 4: 방금 처리한 ID를 Set에 추가
#                     processed_ids.add(current_id)

#                     # 중간 저장 (5개마다)
#                     if len(results) % 5 == 0:
#                         save_results(results, output_filepath)

#                 except Exception as e:
#                     print(f"\nFailed to process question {question[:50]}... on Step {target_index}: {e}")
#                     import traceback
#                     traceback.print_exc()
#                     continue # 이 스텝은 실패했으므로 다음 target_index로 넘어감

#         except Exception as e:
#             # item 로딩 등 외부 루프에서 에러 발생 시
#             print(f"\nFailed to process item {question[:50]}...: {e}")
#             import traceback
#             traceback.print_exc()
#             continue # 이 항목(질문)은 실패했으므로 다음 항목으로 넘어감

#     # 최종 저장
#     save_results(results, output_filepath)
#     print(f"✅ Completed error injection. Total {len(results)} corrupted items saved to {output_filepath}.")

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

    # ✅ 데이터 로드
    input_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_ideal_steps_passage_mapped.json"
    output_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_redundancy_rewriting.json"
    
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {input_filepath}")
    except Exception as e:
        print(f"Error loading data from {input_filepath}: {e}")
        return

    # 💡 [수정됨] 1: 기존 결과 로드 및 처리된 질문 추적
    results = []
    processed_questions = set() # 질문 문자열 자체를 저장할 Set (질문당 1개 생성이므로)
    
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
            # 이미 처리된 '질문'을 Set에 저장
            for res in results:
                if 'question' in res:
                    processed_questions.add(res['question'])
            print(f"Loaded {len(results)} existing results from {output_filepath}. Resuming...")
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_filepath}. Starting fresh. Error: {e}")
            results = []
            processed_questions = set()
    
    # =======================================================
    # 🔹 루프: 각 항목에 대해 오류 주입 (질문당 1회)
    # =======================================================
    for item in tqdm(data, desc="Injecting 'Redundancy' errors"):
        try:
            question = item['question']
            
            # 1. 이미 처리된 질문이면 스킵
            if question in processed_questions:
                continue

            ideal_steps = item['ideal_steps']
            total_steps = len(ideal_steps)
            
            # 2. Step 2부터 가능하므로, 최소 2스텝 이상이어야 함
            if total_steps < 2:
                continue

            # 3. [핵심] Target Step 선택 (가중치 샘플링)
            # 후보군: Step 2 ~ Step K
            candidates = list(range(2, total_steps + 1))
            
            # 가중치 계산: 스텝 번호의 제곱(i^2)에 비례하게 확률 부여
            # 예: candidates=[2, 3, 4] -> weights=[4, 9, 16]
            # 이유: 뒷번호 스텝은 긴 체인에서만 등장하므로, 기회가 왔을 때 더 많이 뽑혀야 전체 분포가 균일해짐
            weights = [i**2 for i in candidates]
            
            target_index = random.choices(candidates, weights=weights, k=1)[0]
            
            # 4. 개별 스텝 생성 시작
            try:
                user_prompt = f"""Question: {question}

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

                # 생성된 스텝 검증 (형식 확인)
                if not (corrupted_step.startswith(f"Step {target_index}:") and 
                        (corrupted_step.endswith("(Attribution)") or corrupted_step.endswith("(Logical)"))):
                    print(f"\n⚠️ Warning: Model output format mismatch for Q: {question[:50]}... (Step {target_index})")
                    print(f"  Got: {corrupted_step}")
                    continue # 포맷 불일치 시 스킵

                # 5. 결과 구성 (Truncation 적용)
                # [이전 스텝들] + [새로 생성된 오류 스텝] (이후 스텝은 잘라냄)
                base_steps = ideal_steps[:target_index-1] 
                corrupted_steps = base_steps + [corrupted_step]

                # 결과 저장
                new_item = item.copy()
                new_item['corrupted_steps'] = corrupted_steps
                new_item['corrupted_step_index'] = target_index
                new_item['error_type'] = 'Redundancy'
                
                results.append(new_item)
                
                # 처리된 질문 목록에 추가
                processed_questions.add(question)

                # 중간 저장 (5개마다)
                if len(results) % 5 == 0:
                    save_results(results, output_filepath)

            except Exception as e:
                print(f"\nFailed to process question {question[:50]}... on Step {target_index}: {e}")
                import traceback
                traceback.print_exc()
                continue 

        except Exception as e:
            # item 로딩 등 외부 루프에서 에러 발생 시
            print(f"\nFailed to process item {question[:50]}...: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 최종 저장
    save_results(results, output_filepath)
    print(f"✅ Completed error injection. Total {len(results)} corrupted items saved to {output_filepath}.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject 'Redundancy' errors into reasoning steps.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., '2wiki') to determine input/output filenames.")
    parser.add_argument("--model_name", type=str, 
                        default="/workspace/hf_transformers/gpt-oss-120b",
                        help="Path to the HuggingFace model directory.")
    
    args = parser.parse_args()
    main(args)
