import pandas as pd
from tqdm import tqdm
import json
import os
import re
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ast

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

system_prompt = """You are a sophisticated binary classifier evaluating answers from a Question Answering (QA) system.
Your task is to judge whether the 'predicted answer' (Pred) is semantically 'Correct' or 'Incorrect' based on the 'question' and 'ground truth answer' (GT).

[Output Format]
You MUST respond with only a single digit: `1` (Correct) or `0` (Incorrect). Do not include any other text.

[Evaluation Guidelines]

Judge as `1` (Correct) in the following cases:
1.  **Semantic Equivalence:** When `predicted answer` has the same meaning as `ground truth answer`.
    * (GT: `PRC` / Pred: `People's Republic of China`)
    * (GT: `196,000-600,000` / Pred: `Between 196,000 and 600,000`)
2.  **Superset Information:** When `predicted answer` includes the core information of `ground truth answer` and adds *correct and relevant* details.
    * (GT: `Tampa` / Pred: `Tampa, Florida`)
    * (GT: `−9 °F` / Pred: `−9 °F (−23 °C)`)
    * (GT: `just south of Yabucoa` / Pred: `Just south of Yabucoa, Puerto Rico`)
3.  **Core Information (Subset):** When `predicted answer` includes the most crucial information from `ground truth answer`, even if some supplementary details or aliases are missing.
    * (GT: `The Rungrado 1st of May Stadium, also known as the May Day Stadium` / Pred: `Rungrado 1st of May Stadium`)
    * (GT: `taxi dancer ``All the Way ''Mae Mordabito` / Pred: `"All the Way" Mae Mordabito`)
4.  **Minor Formatting:** Ignore differences in case, quotes, periods, or articles (a, the).

Judge as `0` (Incorrect) in the following cases:
1.  **Factual Error:** When `predicted answer` states a fact clearly different from `ground truth answer`.
    * (GT: `violin` / Pred: `Guitar`)
    * (GT: `Caroline Islands` / Pred: `Tuvalu`)
2.  **Refusal to Answer:** When `predicted answer` refuses to answer, e.g., 'No information', 'Cannot determine', 'Not specified'.
    * (Pred: `Not specified in the passages provided`)
3.  **Missing Core Information:** When `predicted answer` misses the most essential information from `ground truth answer`.
4.  **Misunderstood Question:** When `predicted answer` provides an irrelevant answer that misunderstands the `question`'s intent.

Begin evaluation. Judge based on the provided `question`, `ground truth answer` (GT), and `predicted answer` (Pred).
""".strip()

def generate_response(tokenizer, llm, messages):
    """Chat template 기반 gpt-oss-120b 응답 생성"""
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
    )

    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    response = outputs[0].outputs[0].text
    return response.split("assistantfinal")[-1].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        max_model_len=3000,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    RESULT_PATH = f"/workspace/daeyong/ideal_steps/{args.dataset}_is_correct_2.json"
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    # -------------------
    # 🔥 기존 처리 결과 불러오기 (resume 지원)
    # -------------------
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, "r") as f:
            existing_results = json.load(f)
        print(f"🔄 Resuming — already processed: {len(existing_results)} rows")
    else:
        existing_results = []
        print("🆕 Starting new run, no previous file found.")

    # -------------------
    # 데이터를 로드하고 필요한 컬럼 정리
    # -------------------
    gt_df = pd.read_csv(f"/workspace/daeyong/training_data/{args.dataset}.csv")[['question', 'answer']]
    df = pd.read_json(f"/workspace/daeyong/ideal_steps/{args.dataset}_final_answer_2.json")

    df = df.merge(gt_df, on="question", how="inner")
    df = df.rename(columns={"answer": "answer_gt"})

    # -------------------
    # 🔥 Resume 시작 index 계산
    # -------------------
    start_index = len(existing_results)
    print(f"▶ Starting from index: {start_index}/{len(df)}")

    results = existing_results.copy()

    for index, item in tqdm(df.iloc[start_index:].iterrows(), total=len(df)-start_index, initial=start_index):

        question = str(item["question"]).strip()
        gt = str(item["answer_gt"]).strip()
        pred = str(item["final_answer"]).strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\nGround Truth Answer: {gt}\nPredicted Answer: {pred}"}
        ]

        try:
            response = generate_response(tokenizer, llm, messages)
        except Exception as e:
            print(f"⚠️ Generation failed at index {index}: {e}")
            response = "ERROR"

        results.append({
            "question": question,
            "answer_gt": gt,
            "answer_pred": pred,
            "is_correct": response
        })

        # 중간 저장
        if len(results) % 5 == 0:
            with open(RESULT_PATH, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # 최종 저장
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Completed: total processed {len(results)} rows.")