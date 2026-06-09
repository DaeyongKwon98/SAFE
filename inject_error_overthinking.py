import pandas as pd
from tqdm import tqdm
import json
import os
import re
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

system_prompt = """You are an expert in logical reasoning, tasked with intentionally introducing a specific logical error into a reasoning steps.

Your Goal: Add a single, correct reasoning step with an 'Overthinking' error.

Error Definition: 'Overthinking'
An 'Overthinking' error is a reasoning step that continues *after* the reasoning is sufficient to answer the question. It introduces a new, *unnecessary* line of reasoning that is no longer required to find the final answer.
- MUST be a *new* step appended to the *end* of the ideal steps.
- MUST be a plausible, but superfluous, inquiry that logically follows from a previous step (often the final step or the entity identified in it).
- MUST be labeled (Attribution) or (Logical) based on its own content.

Input Format:
You will receive:
1. Question: The user's original question.
2. Retrieved Passages: Contextual information.
3. Ideal Reasoning Steps: The correct, multi-step reasoning, which is *already complete* and sufficient to answer the question.

Output Format:
- You MUST output only the single, new, erroneous reasoning step.
- The new step MUST be formatted as the *next* logical step in the sequence (e.g., if Ideal Steps has 3 steps, the output MUST start with "Step 4:").
- The new step MUST end with its correct (Attribution) or (Logical) label.

---
EXAMPLES
---

Question: "What was the first EP of the singer who recorded What Ifs?"

Retrieved Passages:
"Passage 1: \"What Ifs\" is a song recorded by American country music singer Kane Brown for his self-titled debut album..."
"Passage 2: Kane Brown: Kane Allen Brown (born October 21, 1993) is an American country music singer and songwriter... He released his first EP, titled \"Closer\", in June 2015..."
"Passage 3: Lauren Alaina: Lauren Alaina is an American singer... She duetted with Kane Brown on the song \"What Ifs\"."

Ideal Reasoning Steps:
[
 "Step 1: According to Passage 1, the singer who recorded \"What Ifs\" is Kane Brown. (Attribution)",
 "Step 2: According to Passage 2, the first EP of Kane Brown (from Step 1) is titled \"Closer\". (Attribution)",
 "Step 3: Therefore, the EP found in Step 2, \"Closer\", is the answer. (Logical)"
]

Output:
Step 4: According to Passage 3, Lauren Alaina duetted with Kane Brown (from Step 1) on the song "What Ifs". (Attribution)

---

Question: "What Bengali political film was censored by one of the most powerful film censor boards in the world?"

Retrieved Passages: 
"Passage 1: Central Board of Film Certification: ... (CBFC) ... considered to be one of the most powerful film censor boards in the world..."
"Passage 2: Maalik (2016 film): Maalik ... is a 2016 Pakistani political-thriller film ... The film was banned in Pakistan..."
"Passage 3: Kangal Malsat: Kangal Malsat ... is a Bengali political film ... based on the novel with same title written by Nabarun Bhattacharya. ... the Central Board of Film Certification denied approval to the film..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the Central Board of Film Certification (CBFC) is one of the most powerful film censor boards in the world. (Attribution)", 
 "Step 2: According to Passage 3, the Bengali political film 'Kangal Malsat' was censored by the Central Board of Film Certification (from Step 1). (Attribution)", 
 "Step 3: Therefore, the film found in Step 2, 'Kangal Malsat', is the answer. (Logical)" 
]

Output: 
Step 4: According to Passage 2, 'Maalik' is a 2016 Pakistani political-thriller film. (Attribution)

---

Question: "Are David Nixon and Charlie Chaplin from the same country originally?"

Retrieved Passages: 
"Passage 1: David Nixon (director): David Nixon is an American film director and film producer."
"Passage 2: Charlie Chaplin: Sir Charles Spencer 'Charlie' Chaplin... was an English comic actor, filmmaker, and composer..."
"Passage 3: Academy Awards: The Academy Awards, also known as the Oscars, are awards for artistic and technical merit in the international film industry."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, David Nixon is an American film director. (Attribution)", 
 "Step 2: According to Passage 2, Charlie Chaplin was an English comic actor. (Attribution)", 
 "Step 3: Based on Step 1 (American) and Step 2 (English), their countries of origin are not the same. (Logical)" 
]

Output: 
Step 4: Based on Step 1 and Step 2, both David Nixon and Charlie Chaplin work in the film industry. (Logical)

---

Question: "Who is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven?"

Retrieved Passages: 
"Passage 1: James Tuchet, 3rd Earl of Castlehaven (c. 1617 – 11 October 1684) was the son of Mervyn Tuchet, 2nd Earl of Castlehaven..." 
"Passage 2: Mervyn Tuchet, 2nd Earl of Castlehaven (1593 – 14 May 1631)... A son of George Tuchet, 1st Earl of Castlehaven and 11th Baron Audley..."
"Passage 3: Earl of Castlehaven: The title Earl of Castlehaven was created in 1616 for George Tuchet, 1st Baron Audley."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the father of James Tuchet, 3rd Earl of Castlehaven is Mervyn Tuchet, 2nd Earl of Castlehaven. (Attribution)", 
 "Step 2: According to Passage 2, the father of Mervyn Tuchet, 2nd Earl of Castlehaven (from Step 1) is George Tuchet, 1st Earl of Castlehaven. (Attribution)", 
 "Step 3: Therefore, the person found in Step 2, George Tuchet, 1st Earl of Castlehaven, is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven. (Logical)" 
]

Output: 
Step 4: According to Passage 3, the title Earl of Castlehaven was created in 1616 for George Tuchet (from Step 2). (Attribution)

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

Output: 
Step 6: According to Passage 1, Ignace Matondo Kwa Nzambi (from Step 1) was a Congolese politician. (Attribution)
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
        temperature=0.0,
        top_p=1.0,
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
    # 🔴 [수정됨] Output 파일 이름 변경
    output_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_overthinking.json"
    
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {input_filepath}")
    except Exception as e:
        print(f"Error loading data from {input_filepath}: {e}")
        return

    # 💡 1: 기존 결과 로드 (이제 (question)만 키로 사용)
    results = []
    processed_ids = set() # (question)을 저장할 Set
    
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
            # 2. 이미 처리된 ID를 Set에 저장
            for res in results:
                if 'question' in res:
                    processed_ids.add(res['question'])
            print(f"Loaded {len(results)} existing results from {output_filepath}. Resuming...")
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_filepath}. Starting fresh. Error: {e}")
            results = []
            processed_ids = set()
    
    # =======================================================
    # 🔹 루프: 각 항목에 대해 오류 주입
    # =======================================================
    # 1. 각 '항목'(질문)에 대해 루프
    for item in tqdm(data, desc="Injecting 'Overthinking' errors"):
        try:
            question = item['question']
            
            # 💡 [수정됨] 3: 이미 처리된 항목인지 확인 (질문 단위로)
            if question in processed_ids:
                continue # 이미 처리되었으므로 스킵

            passages = item['retrieved_passages']
            ideal_steps = item['ideal_steps']
            
            # 다음 스텝 번호(K+1)를 계산
            next_step_index = len(ideal_steps) + 1
            
            # 컨텍스트로 사용할 passages 포맷팅
            passages_context = "\n".join(f"Passage {i+1}: {p}" for i, p in enumerate(passages))

            # 'Target Step to Corrupt'가 프롬프트에서 제거됨
            user_prompt = f"""Question: {question}

Retrieved Passages:
{passages_context}

Ideal Reasoning Steps:
{json.dumps(ideal_steps, indent=2, ensure_ascii=False)}
""".strip()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 오류가 주입된 스텝(K+1번째 스텝) 생성
            corrupted_step = generate_response(tokenizer, llm, messages).strip().strip(",")

            # 🔴 [수정됨] 생성된 스텝 검증 (다음 스텝 인덱스 기준)
            if not (corrupted_step.startswith(f"Step {next_step_index}:") and 
                    (corrupted_step.endswith("(Attribution)") or corrupted_step.endswith("(Logical)"))):
                print(f"\n⚠️ Warning: Model output format mismatch for Q: {question[:50]}...")
                print(f"  Expected prefix: 'Step {next_step_index}:'")
                print(f"  Expected suffix: '(Attribution)' or '(Logical)'")
                print(f"  Got: {corrupted_step}")
                continue

            # 🔴 [수정됨] 새 오류 스텝 생성 (단순 append)
            corrupted_steps = ideal_steps + [corrupted_step]

            # 결과 저장
            new_item = item.copy()
            new_item['corrupted_steps'] = corrupted_steps
            # 🔴 [수정됨] corrupted_step_index는 이제 항상 마지막 스텝 인덱스
            new_item['corrupted_step_index'] = next_step_index # 1-based
            # 🔴 [수정됨] Error type 변경
            new_item['error_type'] = 'Overthinking'
            results.append(new_item)
            
            # 💡 [수정됨] 4: 방금 처리한 ID(질문)를 Set에 추가
            processed_ids.add(question)

            # 중간 저장 (3개마다)
            if len(results) % 3 == 0:
                save_results(results, output_filepath)

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
    parser = argparse.ArgumentParser(description="Inject 'Overthinking' errors into reasoning steps.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., '2wiki') to determine input/output filenames.")
    parser.add_argument("--model_name", type=str, 
                        default="/workspace/hf_transformers/gpt-oss-120b",
                        help="Path to the HuggingFace model directory.")
    args = parser.parse_args()
    main(args)