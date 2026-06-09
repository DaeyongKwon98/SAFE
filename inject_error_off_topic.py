import pandas as pd
from tqdm import tqdm
import json
import os
import re
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple

system_prompt = """You are an expert in logical reasoning, tasked with intentionally introducing a specific logical error into a reasoning steps.

Your Goal: Replace a single, correct reasoning step with an 'Off-topic' error.

Error Definition: 'Off-topic'
An 'Off-topic' error is a reasoning step that is irrelevant to the overall goal of the question and the specific step it is replacing. It introduces a new, unrelated piece of information or inference that leads the reasoning process astray.
- MUST be a new, irrelevant line of reasoning.
- MUST NOT be a repetition of a previous step's conclusion.
- MUST NOT be a repetition of the original step it is replacing.
- MUST maintain the original step's label (e.g., (Attribution) or (Logical)).

Input Format:
You will receive:
1. Question: The user's original question.
2. Retrieved Passages: Contextual information.
3. Ideal Reasoning Steps: The correct, multi-step reasoning.
4. Target Step to Corrupt: The specific step from the ideal steps that you must replace. (ex. Step 1)

Output Format:
- You MUST output only the single, new, erroneous reasoning step.
- The new step MUST be formatted exactly like the target step, including the "Step X:" prefix and the "(Label)" suffix.

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
Step 2: According to Passage 2, Kane Brown (from Step 1) is an American country music singer and songwriter. (Attribution)

---

Question: "What Bengali political film was censored by one of the most powerful film censor boards in the world?"

Retrieved Passages: 
"Passage 1: Central Board of Film Certification: ... (CBFC) ... considered to be one of the most powerful film censor boards in the world..."
"Passage 2: Kangal Malsat: Kangal Malsat ... is a Bengali political film ... based on the novel with same title written by Nabarun Bhattacharya. ... the Central Board of Film Certification denied approval to the film..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the Central Board of Film Certification (CBFC) is one of the most powerful film censor boards in the world. (Attribution)", 
 "Step 2: According to Passage 2, the Bengali political film 'Kangal Malsat' was censored by the Central Board of Film Certification (from Step 1). (Attribution)", 
 "Step 3: Therefore, the film found in Step 2, 'Kangal Malsat', is the answer. (Logical)" 
]

Target Step to Corrupt: 
Step 2

Output: 
Step 2: According to Passage 2, the film 'Kangal Malsat' is based on a novel written by Nabarun Bhattacharya. (Attribution)

---

Question: "Journey to the Center of the Earth starred a former Metro-Goldwyn-Mayer contract star who is the mother of what actor?"

Retrieved Passages: 
"Passage 1: Journey to the Center of the Earth (1959 film): ... stars James Mason, Pat Boone and Arlene Dahl." 
"Passage 2: Arlene Dahl: Arlene Carol Dahl (born August 11, 1925) is an American actress and former Metro-Goldwyn-Mayer contract star... She has three children, the eldest of whom is actor Lorenzo Lamas."

Ideal Reasoning Steps:
[ 
 "Step 1: According to Passage 1, 'Journey to the Center of the Earth' stars Arlene Dahl. (Attribution)", 
 "Step 2: According to Passage 2, Arlene Dahl (from Step 1) is a former Metro-Goldwyn-Mayer contract star. (Attribution)", 
 "Step 3: According to Passage 2, the actor son of Arlene Dahl (from Step 1) is Lorenzo Lamas. (Attribution)", 
 "Step 4: Therefore, the actor found in Step 3, Lorenzo Lamas, is the answer. (Logical)" 
]

Target Step to Corrupt: 
Step 3

Output: 
Step 3: According to Passage 2, Arlene Dahl (from Step 1) was born on August 11, 1925. (Attribution)

---

Question: "Are David Nixon and Charlie Chaplin from the same country originally?"

Retrieved Passages: 
"Passage 1: David Nixon (director): David Nixon is an American film director and film producer."
"Passage 2: Charlie Chaplin: Sir Charles Spencer 'Charlie' Chaplin... was an English comic actor, filmmaker, and composer..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, David Nixon is an American film director. (Attribution)", 
 "Step 2: According to Passage 2, Charlie Chaplin was an English comic actor. (Attribution)", 
 "Step 3: Based on Step 1 (American) and Step 2 (English), their countries of origin are not the same. (Logical)" 
]

Target Step to Corrupt: 
Step 3

Output: 
Step 3: Based on Step 1 and Step 2, both David Nixon and Charlie Chaplin work in the film industry. (Logical)

---

Question: "What Bengali political film was censored by one of the most powerful film censor boards in the world?"

Retrieved Passages: 
"Passage 1: Central Board of Film Certification: ... (CBFC) ... is a statutory censorship ... body under the ... Government of India. ... CBFC India is considered to be one of the most powerful film censor boards in the world..."
"Passage 2: Maalik (2016 film): Maalik (Urdu مالک) is a 2016 Pakistani political-thriller film ... The film was banned in Pakistan for political reasons after being cleared by all three Censor Boards..."
"Passage 3: Kangal Malsat: Kangal Malsat ('War Cry of beggars') is a Bengali political film ... As of 2013, the Central Board of Film Certification denied approval to the film..."

Ideal Reasoning Steps: 
[
 "Step 1: According to Passage 1, the Central Board of Film Certification (CBFC) is one of the most powerful film censor boards in the world. (Attribution)", 
 "Step 2: According to Passage 3, the Bengali political film 'Kangal Malsat' was censored by the Central Board of Film Certification (from Step 1). (Attribution)", 
 "Step 3: Therefore, the film found in Step 2, 'Kangal Malsat', is the answer. (Logical)" 
]

Target Step to Corrupt: 
Step 2

Output: 
Step 2: According to Passage 2, 'Maalik' is a 2016 Pakistani political-thriller film. (Attribution)
""".strip()

system_prompt = """You are an expert in logical reasoning, tasked with intentionally introducing a specific logical error into a reasoning process.

Your Goal: Replace a single, correct reasoning step with a 'Off-topic' error.

Error Definition: 'Off-topic' (Navigation Failure)
A 'Off-topic' error occurs when the reasoning agent gets distracted by a keyword or a related entity in the text and retrieves information that is factually correct but **fails to address the specific sub-goal of the target step**.
This simulates a "wrong turn" in the search process.

The error MUST fall into one of these two categories:
1. **Attribute Drift:** The agent finds the CORRECT entity but retrieves the WRONG attribute (e.g., asked for 'birth year', retrieved 'birth place').
2. **Entity Confusion:** The agent finds a WRONG entity that shares a keyword or relationship and retrieves information about that distractor (e.g., asked for 'Director of Film A', retrieved 'Director of Film B' because both films are mentioned).

Input Format:
You will receive:
1. Question: The user's original question.
2. Retrieved Passages: Contextual information.
3. Ideal Reasoning Steps: The correct, multi-step reasoning.
4. Target Step to Corrupt: The specific step from the ideal steps that you must replace.

Output Format:
- You MUST output only the single, new, erroneous reasoning step.
- The step MUST use information present in the Retrieved Passages.
- The step MUST maintain the original step's label (e.g., (Attribution) or (Logical)).
- **CRITICAL:** Do NOT just repeat the previous step. Provide new, specific information that is irrelevant to the current sub-goal.

---
EXAMPLES
---

Question: "Kim Sŏng-ae is the wife of which leader of North Korea who died in 1994?"

Retrieved Passages:
"Passage 1: Kim Il-sung ... died in 1994..."
"Passage 5: Kang Pan-sok ... was the mother of North Korean leader Kim Il-sung..."
"Passage 6: Kim Sŏng-ae ... is ... the second wife of North Korean leader Kim Il-sung."

Ideal Reasoning Steps:
[
 "Step 1: According to Passage 1, the leader of North Korea who died in 1994 is Kim Il-sung. (Attribution)",
 "Step 2: According to Passage 6, the wife of Kim Il-sung is Kim Sŏng-ae. (Attribution)",
 "Step 3: Therefore, the leader is Kim Il-sung. (Logical)"
]

Target Step to Corrupt:
Step 2

Output:
Step 2: According to Passage 5, Kang Pan-sok was the mother of North Korean leader Kim Il-sung. (Attribution)

---

Question: "When is the director of film For Love Or Money (2014 Film)'s birthday?"

Retrieved Passages:
"Passage 5: For Love or Money (1963 film): ... directed by Michael Gordon..."
"Passage 6: For Love or Money (2014 film): ... The film was directed by Gao Xixi..."
"Passage 1: Gao Xixi (born June 16, 1962)..."

Ideal Reasoning Steps:
[
 "Step 1: According to Passage 6, the director of the film 'For Love or Money (2014)' is Gao Xixi. (Attribution)",
 "Step 2: According to Passage 1, the birthday of Gao Xixi is June 16, 1962. (Attribution)",
 "Step 3: So the director of film For Love Or Money's birthday is June 16, 1962. (Logical)"
]

Target Step to Corrupt:
Step 1

Output:
Step 1: According to Passage 5, the film 'For Love or Money (1963 film)' was directed by Michael Gordon. (Attribution)

---

Question: "What nationality is the performer of song In The Heat Of The Night?"

Retrieved Passages:
"Passage 4: In the Heat of the Night... 1967 song performed by Ray Charles..."
"Passage 10: Ray Charles... was an American singer... often referred to as 'The Genius'."

Ideal Reasoning Steps:
[
 "Step 1: According to Passage 4, the performer of the song 'In The Heat Of The Night' is Ray Charles. (Attribution)",
 "Step 2: According to Passage 10, the nationality of Ray Charles is American. (Attribution)",
 "Step 3: So the final answer for nationality of Ray Charles is American. (Logical)"
]

Target Step to Corrupt:
Step 2

Output:
Step 2: According to Passage 10, Ray Charles (from Step 1) was often referred to as "The Genius." (Attribution)

---

Question: "Which film has the director who was born first, The Seventh Sign or The Old Fritz?"

Retrieved Passages:
"Passage 1: Carl Schultz (born 19 September 1939) is a Hungarian-Australian film director..."
"Passage 4: The Seventh Sign... directed by Carl Schultz."
"Passage 5: Hassan Zee... Pakistani-American film director who was born in Chakwal, Pakistan."
"Passage 8: The Old Fritz... directed by Gerhard Lamprecht..."
"Passage 9: Gerhard Lamprecht (6 October 1897 – 4 May 1974) was a German film director..."

Ideal Reasoning Steps:
[
 "Step 1: According to Passage 4, the director of 'The Seventh Sign' is Carl Schultz. (Attribution)",
 "Step 2: According to Passage 1, Carl Schultz (from Step 1) was born on September 19, 1939. (Attribution)",
 "Step 3: According to Passage 8, the director of 'The Old Fritz' is Gerhard Lamprecht. (Attribution)",
 "Step 4: According to Passage 9, Gerhard Lamprecht (from Step 3) was born on October 6, 1897. (Attribution)",
 "Step 5: Comparing the birth years found in Step 2 (1939) and Step 4 (1897), Gerhard Lamprecht was born first, so 'The Old Fritz' is the answer. (Logical)"
]

Target Step to Corrupt:
Step 3

Output:
Step 3: According to Passage 5, Hassan Zee is a Pakistani-American film director who was born in Chakwal, Pakistan. (Attribution)
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
    output_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_off_topic.json"
    
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {input_filepath}")
    except Exception as e:
        print(f"Error loading data from {input_filepath}: {e}")
        return

    # 💡 [수정됨] 1: 기존 결과 로드
    results = []
    processed_ids = set() # (question, target_index)를 저장할 Set
    
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
            # 2. 이미 처리된 ID를 Set에 저장
            # (질문, 타겟스텝인덱스) 조합이 고유 키가 됩니다.
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
    for item in tqdm(data, desc="Injecting 'Off-topic' errors"):
        try:
            question = item['question']
            passages = item['retrieved_passages']
            ideal_steps = item['ideal_steps']
            total_steps = len(ideal_steps)
            
            # 2. (요청 1) 1부터 K까지 각 스텝을 타겟으로 내부 루프
            for target_index in range(1, total_steps + 1):
                # target_index는 1-based (1, 2, ..., K)
                
                # 3. 개별 스텝 주입(try-except로 감싸서 한 스텝이 실패해도 다음 스텝으로 넘어가도록 함)
                try:
                    # 💡 [수정됨] 3: 이미 처리된 항목인지 확인
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
                    if not (corrupted_step.startswith(f"Step {target_index}:") and 
                            (corrupted_step.endswith("(Attribution)") or corrupted_step.endswith("(Logical)"))):
                        print(f"\n⚠️ Warning: Model output format mismatch for Q: {question[:50]}... (Step {target_index})")
                        print(f"  Expected prefix: 'Step {target_index}:'")
                        print(f"  Expected suffix: '(Attribution)' or '(Logical)'")
                        print(f"  Got: {corrupted_step}")
                        # 포맷이 망가졌으면 이 target_index는 건너뛰기
                        continue

                    # (요청 2) 새 오류 스텝 생성 (N+1 이후 스텝은 제외)
                    # 1. 타겟 스텝 이전의 스텝들 (0-based 인덱싱)
                    base_steps = ideal_steps[:target_index-1] 
                    # 2. [이전 스텝들] + [새로 생성된 오류 스텝]
                    corrupted_steps = base_steps + [corrupted_step]

                    # 결과 저장
                    new_item = item.copy()
                    new_item['corrupted_steps'] = corrupted_steps
                    new_item['corrupted_step_index'] = target_index # 1-based
                    new_item['error_type'] = 'Off-topic'
                    # (요청 3) 'injected_error' 키는 저장하지 않음
                    results.append(new_item)
                    
                    # 💡 [수정됨] 4: 방금 처리한 ID를 Set에 추가
                    processed_ids.add(current_id)

                    # 중간 저장 (3개마다)
                    if len(results) % 3 == 0:
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
    parser = argparse.ArgumentParser(description="Inject 'Off-topic' errors into reasoning steps.")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., '2wiki') to determine input/output filenames.")
    
    parser.add_argument("--model_name", type=str, 
                        default="/workspace/hf_transformers/gpt-oss-120b",
                        help="Path to the HuggingFace model directory.")
    args = parser.parse_args()
    main(args)
