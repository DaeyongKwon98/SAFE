import pandas as pd
from tqdm import tqdm
import json
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse

# -------------------------
# 기존 모델 로드 함수 그대로 유지
# -------------------------
system_prompt = """You are a reasoning model that takes a question and ideal reasoning steps.
Generate the final concise answer based ONLY on the reasoning steps.
Do not include reasoning in the output — only provide the final answer.

Answer format:
<final answer here>
""".strip()

def generate_response(tokenizer, llm, messages):
    """Chat template 기반 gpt-oss-120b 응답 생성"""
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
    )

    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    response = outputs[0].outputs[0].text
    return response.split("assistantfinal")[-1].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 2wiki, hotpotqa, musique)")
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
    INPUT_PATH = f"/workspace/daeyong/ideal_steps/{args.dataset}_ideal_steps_2.json"
    OUTPUT_PATH = f"/workspace/daeyong/ideal_steps/{args.dataset}_final_answer_2.json"

    print(f"📂 Loading: {INPUT_PATH}")
    df = pd.read_json(INPUT_PATH)

    results = []
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        reasoning = row.get("ideal_steps", "N/A")

        user_message = f"""
Question:
{question}

Reasoning Steps:
{reasoning}

Now provide ONLY the final answer based on the reasoning.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            answer = generate_response(tokenizer, llm, messages)
        except Exception as e:
            answer = f"ERROR: {e}"

        results.append({
            "question": question,
            "ideal_steps": reasoning,
            "final_answer": answer
        })

        if len(results) % 5 == 0:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Done: Saved {len(results)} results → {OUTPUT_PATH}")
