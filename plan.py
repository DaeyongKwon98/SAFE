import pandas as pd
from tqdm import tqdm
import json
import os
import re
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts import plan_generation_2wiki, plan_generation_hotpotqa, plan_generation_musique

MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"

def extract_step_list(text: str) -> str:
    # 1. 모델 특유의 구분자(assistantfinal) 처리
    # 모델이 CoT를 출력한 뒤 'assistantfinal' 뒤에 진짜 답을 내놓는 패턴 대응
    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1].strip()

    # 2. 대괄호 [...] 패턴 찾기
    # re.DOTALL을 쓰되, 여러 개의 리스트가 있을 수 있으므로 findall로 모두 찾음
    matches = re.findall(r"\[.*?\]", text, re.DOTALL)
    
    if matches:
        # 가장 마지막에 등장한 대괄호 묶음을 선택 (보통 최종 정답이 끝에 위치함)
        return matches[-1]
    
    # 3. 대괄호가 없는 경우: Step 패턴으로 강제 추출 시도
    # "Step 1: ... Step 2: ..." 형태의 텍스트가 있는지 확인
    steps = re.findall(r"Step\s*\d+\s*:[^,\n]+", text)
    if steps:
        # 강제로 리스트 포맷으로 변환
        return "[" + ", ".join(s.strip() for s in steps) + "]"
        
    return text.strip()

def safe_parse_plan(plan_str: str):
    plan_str = plan_str.strip()
    
    # 대괄호 제거
    if plan_str.startswith('[') and plan_str.endswith(']'):
        inner = plan_str[1:-1].strip()
    else:
        inner = plan_str

    if not inner:
        print(f"⚠️ Empty plan string after removing brackets: {plan_str}")
        return []
    
    # 4. 텍스트 내부 파싱 로직 개선
    # 단순히 쉼표(,)로만 자르면 문장 안에 쉼표가 있을 때 오작동할 수 있음.
    # "Step 숫자:" 패턴을 기준으로 분리하는 것이 안전함.
    
    # 정규식 설명: 
    # ,? (쉼표가 있을수도 없을수도)
    # \s* (공백)
    # (?=Step\s*\d+:) -> 뒤에 'Step 숫자:'가 오는 위치를 찾음 (Lookahead)
    steps_raw = re.split(r",?\s*(?=Step\s*\d+:)", inner)
    
    clean_steps = []
    for s in steps_raw:
        s = s.strip()
        # 불필요한 따옴표 제거 (모델이 'Step 1: ...' 처럼 따옴표를 쓸 때가 있음)
        s = s.strip("'").strip('"')
        
        # 빈 문자열이나, Step으로 시작하지 않는 찌꺼기 데이터 제외
        if s and s.lower().startswith("step"):
            clean_steps.append(s)
            
    return clean_steps

def main(args):
    if args.dataset == "2wiki":
        INPUT_PATH = "/workspace/daeyong/benchmarks/2wiki_dev.csv"
        RESULT_PATH = "/workspace/daeyong/reasoning_plans/2wiki_dev_plan.json"
        system_prompt = plan_generation_2wiki
    elif args.dataset == "hotpotqa":
        INPUT_PATH = "/workspace/daeyong/benchmarks/hotpotqa_dev.csv"
        RESULT_PATH = "/workspace/daeyong/reasoning_plans/hotpotqa_dev_plan.json"
        system_prompt = plan_generation_hotpotqa
    elif args.dataset == "musique":
        INPUT_PATH = "/workspace/daeyong/benchmarks/musique_dev.csv"
        RESULT_PATH = "/workspace/daeyong/reasoning_plans/musique_dev_plan.json"
        system_prompt = plan_generation_musique
    
    # 1. Initialize vLLM
    print(f"🚀 Loading vLLM model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        max_model_len=10000,
        enable_prefix_caching=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=6000,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Load Data & Filter
    df = pd.read_csv(INPUT_PATH)
    
    # Load existing results for resuming
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, "r") as f:
            existing_results = json.load(f)
        processed_questions = {r["question"] for r in existing_results}
        print(f"🔄 Resuming... Found {len(processed_questions)} existing results.")
    else:
        existing_results, processed_questions = [], set()

    # Filter out currently processed questions
    df_target = df[~df['question'].isin(processed_questions)]
    print(f"🎯 Total questions to process: {len(df_target)}")

    if df_target.empty:
        print("✅ No new questions to process.")
        return

    # =========================================================================
    # ✅ 수정된 부분: 배치 단위 처리 및 중간 저장
    # =========================================================================
    
    BATCH_SIZE = 500
    questions = df_target['question'].tolist()
    final_results = existing_results  # 기존 결과 리스트를 계속 업데이트

    print(f"🚀 Starting processing in batches of {BATCH_SIZE}...")

    # 전체 질문을 BATCH_SIZE만큼 잘라서 반복
    for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="Processing Batches"):
        batch_questions = questions[i : i + BATCH_SIZE]
        batch_prompts = []

        # 3. Construct Prompts for Current Batch
        for q in batch_questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {q}"}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            batch_prompts.append(full_prompt)

        # 4. Generate in Batch (현재 배치만 수행)
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # 5. Process Outputs for Current Batch
        new_batch_results = []
        for question, output in zip(batch_questions, outputs):
            generated_text = output.outputs[0].text
            
            try:
                plan_str = extract_step_list(generated_text)
                plan_list = safe_parse_plan(plan_str)
                
                # Validation
                clean_plan = []
                for step in plan_list:
                    step = step.strip().strip(",")
                    if not (step.endswith("(Attribution)") or step.endswith("(Logical)")):
                        print(f"⚠️ Invalid step format in Q={question[:30]}...: {step}")
                        continue
                    clean_plan.append(step)

                new_batch_results.append({
                    "question": question,
                    "plan": clean_plan
                })

            except Exception as e:
                print(f"⚠️ Parsing failed for Q={question[:30]}...: {e}")

        # 6. Save Intermediate Results (각 배치 끝날 때마다 저장)
        final_results.extend(new_batch_results)
        
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved: {len(final_results)}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, choices=["2wiki", "hotpotqa", "musique"], required=True)
    args = argparser.parse_args()
    main(args)