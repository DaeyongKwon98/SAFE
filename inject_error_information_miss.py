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
from collections import Counter

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

system_prompt = """You are an expert in logical reasoning, tasked with intentionally introducing a specific logical error into a reasoning steps.

Your Goal: Replace a single, correct reasoning step with an 'Information Miss' error.

Error Definition: 'Information Miss' (False Negative / Missed Retrieval)
An 'Information Miss' error occurs when the model **fails to extract an existing piece of factual evidence** from the `Retrieved Passages` and instead asserts that the required information **is absent, unknown, or not provided**. The information *must* actually be present in the passages.

- MUST assert that the information needed for that step is **not available** (e.g., "The passages do not mention...", "The date is unknown").
- MUST NOT state what the information *is* (as that would be correct).
- MUST NOT extract the information from the passages, even if it is only procedural meta-talk.
- MUST NOT repeat a specific fact or conclusion found in a previous step (as that would be a 'Redundancy' error).
- MUST NOT be off-topic; it should be relevant to the reasoning process but fail to find the needed fact.

Input Format:
You will receive:
1. Question: The user's original question.
2. Retrieved Passages: Contextual information.
3. Ideal Reasoning Steps: The correct, multi-step reasoning.
4. Target Step to Corrupt: The specific step from the ideal steps that you must replace. (ex. Step 2)

Output Format:
- You MUST output only the single, new, erroneous reasoning step.
- The new step MUST be formatted exactly like the target step, including the "Step X:" prefix and the "(Label)" suffix (Use "(Attribution)" for this error type, as it's a failure of factual retrieval).

---
EXAMPLES
---

Question: "What was the first EP of the singer who recorded What Ifs?"

Retrieved Passages:
"Passage 1: \"What Ifs\" is a song recorded by American country music singer Kane Brown for his self-titled debut album..."
"Passage 2: Kane Brown: Kane Allen Brown (born October 21, 1993) is an American country music singer and songwriter... He released his first EP, titled \"Closer\", in June 2015..."

Ideal Reasoning Steps:
[
 "Step 1: According to Passage 1, the singer who recorded \"What Ifs\" is Kane Brown. (Attribution)",
 "Step 2: According to Passage 2, the first EP of Kane Brown (from Step 1) is titled \"Closer\". (Attribution)",
 "Step 3: Therefore, the EP found in Step 2, \"Closer\", is the answer. (Logical)"
]

Target Step to Corrupt:
Step 2

Output:
Step 2: Passage 2 mention Kane Brown but do not explicitly state the title of his first EP. (Attribution)

---

Question: "Who is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven?"

Retrieved Passages: 
"Passage 1: James Tuchet, 3rd Earl of Castlehaven (c. 1617 – 11 October 1684) was the son of Mervyn Tuchet, 2nd Earl of Castlehaven..." 
"Passage 2: Mervyn Tuchet, 2nd Earl of Castlehaven (1593 – 14 May 1631)... A son of George Tuchet, 1st Earl of Castlehaven and 11th Baron Audley..."
"Passage 3: George Tuchet... was an English peer..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the father of James Tuchet, 3rd Earl of Castlehaven is Mervyn Tuchet, 2nd Earl of Castlehaven. (Attribution)", 
 "Step 2: According to Passage 2, the father of Mervyn Tuchet, 2nd Earl of Castlehaven (from Step 1) is George Tuchet, 1st Earl of Castlehaven. (Attribution)", 
 "Step 3: Therefore, the person found in Step 2, George Tuchet, 1st Earl of Castlehaven, is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven. (Logical)" 
]

Target Step to Corrupt: 
Step 2

Output: 
Step 2: While Passage 2 discusses Mervyn Tuchet, the identity of his father is not provided in the retrieved texts. (Attribution)

---

Question: "Who lived longer, Ignace Matondo Kwa Nzambi or Claudia Zobel?"

Retrieved Passages: 
"Passage 1: Ignace Matondo Kwa Nzambi (12 April 1932 – 9 September 2011) was a Congolese politician..." 
"Passage 2: Claudia Zobel (February 27, 1964 – February 10, 1984) was a Filipina actress..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, Ignace Matondo Kwa Nzambi was born on 12 April 1932 and died on 9 September 2011. (Attribution)", 
 "Step 2: Based on the dates from Step 1, Ignace Matondo Kwa Nzambi lived for 79 years. (Logical)", 
 "Step 3: According to Passage 2, Claudia Zobel was born on February 27, 1964 and died on February 10, 1984. (Attribution)", 
 "Step 4: Based on the dates from Step 3, Claudia Zobel lived for 19 years. (Logical)", 
 "Step 5: Comparing the lifespans from Step 2 (79 years) and Step 4 (19 years), Ignace Matondo Kwa Nzambi lived longer. (Logical)" 
]

Target Step to Corrupt: 
Step 1

Output: 
Step 1: The passages provide the names but do not contain the necessary birth and death dates for Ignace Matondo Kwa Nzambi to calculate his lifespan. (Attribution)
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
    output_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_information_miss.json"
    
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {input_filepath}")
    except Exception as e:
        print(f"Error loading data from {input_filepath}: {e}")
        return

    # 💡 [수정됨] 1: 기존 결과 로드
    results = []
    processed_questions = set()
    
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
            # 2. 이미 처리된 ID를 Set에 저장
            # 질문이 고유 키가 됩니다.
            for res in results:
                if 'question' in res and 'corrupted_step_index' in res:
                    processed_questions.add(res['question'])
            print(f"Loaded {len(results)} existing results from {output_filepath}. Resuming...")
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_filepath}. Starting fresh. Error: {e}")
            results = []
            processed_questions = set()
    
    WEIGHT_MAP = {
        1: 1,
        2: 1,
        3: 3,   # Step 3까지는 데이터가 충분함
        4: 12,   # Step 4부터 급격히 줄어드므로 가중치 상향
        5: 30,  # Step 5는 전체의 10% 수준이므로 매우 높게 설정
        6: 50,  # Step 6 이상은 극히 드무므로 보이면 거의 무조건 선택
        7: 50
    }

    # 기본값 (매핑에 없는 더 큰 숫자가 나올 경우)
    DEFAULT_WEIGHT = 50
    
    c = Counter()
    
    # =======================================================
    # 🔹 루프: 각 항목에 대해 오류 주입
    # =======================================================
    # 1. 각 '항목'(질문)에 대해 루프
    for item in tqdm(data, desc="Injecting 'Information Miss' errors with scarcity-aware sampling"):
        try:
            question = item['question']
            passages = item['retrieved_passages']
            ideal_steps = item['ideal_steps']

            # 이미 처리된 항목인지 확인
            if question in processed_questions:
                continue

            possible_indices = []
            for i, step_text in enumerate(ideal_steps):
                # 1-based index로 변환
                target_index = i + 1
                
                # (Attribution) 스텝이 아니면 건너뛰기
                if not "(Attribution)" in step_text:
                    continue
                
                possible_indices.append(target_index)

            
            # 💡 [핵심 수정] 희소성 기반 가중치 적용
            # 단순히 i를 쓰는 것보다, 데이터 분포 역수를 반영한 매핑 사용이 훨씬 균일함
            weights = [WEIGHT_MAP.get(i, DEFAULT_WEIGHT) for i in possible_indices]
            
            # 가중치 기반 랜덤 샘플링 (1개 선택)
            target_index = random.choices(possible_indices, weights=weights, k=1)[0]
            c[target_index] += 1

            # 이하 에러 주입 및 저장 로직 동일
            passages_context = "\n".join(f"Passage {i+1}: {p}" for i, p in enumerate(passages))

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

            corrupted_step = generate_response(tokenizer, llm, messages)

            # Format 검증
            if not (corrupted_step.startswith(f"Step {target_index}:") and ("(Attribution)" in corrupted_step)):
                continue

            base_steps = ideal_steps[:target_index-1] 
            corrupted_steps = base_steps + [corrupted_step]

            new_item = item.copy()
            new_item['corrupted_steps'] = corrupted_steps
            new_item['corrupted_step_index'] = target_index
            new_item['error_type'] = 'Information Miss'
            results.append(new_item)

            processed_questions.add(question)

            if len(results) % 5 == 0:
                save_results(results, output_filepath)
                
            print(c)

        except Exception as e:
            print(f"\nFailed to process item {question[:30]}...: {e}")
            continue 

    save_results(results, output_filepath)
    print(f"✅ Completed. Total {len(results)} items saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject 'Information Miss' errors into reasoning steps.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., '2wiki') to determine input/output filenames.")
    parser.add_argument("--model_name", type=str, 
                        default="/workspace/hf_transformers/gpt-oss-120b",
                        help="Path to the HuggingFace model directory.")
    
    args = parser.parse_args()
    main(args)
