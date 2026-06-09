import argparse
import os
import json
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def load_model_and_tokenizer(model_name: str):
    """
    vLLM 모델 엔진과 토크나이저를 로드합니다.
    """
    print(f"Loading Model (vLLM): {model_name}...")
    
    # 1. 토크나이저 로드 (프롬프트 템플릿 적용용)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. vLLM 엔진 로드
    # 단일 모델만 로드하므로 메모리를 넉넉히(0.9) 사용합니다.
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=8,
        gpu_memory_utilization=0.9, 
        trust_remote_code=True,
        max_model_len=1000,
    )
    
    return llm, tokenizer

def prepare_prompt(query: str, reasoning_steps: list, tokenizer) -> str:
    """
    질문과 추론 단계를 받아 vLLM에 입력할 프롬프트 '문자열'을 생성합니다.
    """
    system_prompt = f"""You are a strict output formatter. 
Your output will be parsed by a computer program using a Regular Expression. 
Any text outside the specified format will cause a system crash.

Your Task:
1. Analyze the `Question` and `Reasoning Steps`.
2. Determine the final conclusion.
3. Format it into a SINGLE line.

# Critical Rules
1. **ENTITY ONLY (NO SENTENCES)**: 
   - Extract ONLY the specific entity, name, date, number, or short phrase.
   - **Do NOT** output full sentences. Remove words like "The answer is", "is", "was", "Therefore".
   - Bad: "The director is Steven Spielberg"
   - Good: "Steven Spielberg"
2. **FAITHFULNESS**: You must base your answer ONLY on the provided `Reasoning Steps`. 
   - Even if the reasoning contains a factual error (e.g., says "1+1=3"), you must output the result derived from that logic (e.g., "3").
   - **DO NOT correct errors** using your external knowledge. Trust the steps completely.
3. **NO EXPLANATIONS**: Do NOT explain logic. Do NOT mention "According to the steps". Just output the final value.
4. **STRICT FORMAT**: Your output must match this format exactly:
   Step {len(reasoning_steps)+1}: ####ANSWER: <Final_Value> (Final Answer)

# Handling Missing Information
- If and ONLY IF the reasoning steps fail to reach ANY conclusion or explicitly state the answer is unknown:
  Output: Step {len(reasoning_steps)+1}: ####ANSWER: Cannot Answer (Final Answer)
""".strip()

    user_prompt = f"""## Question ##
{query}

## Reasoning Steps ##
{reasoning_steps}

## Final Answer Step (Step {len(reasoning_steps)+1})##
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # vLLM은 리스트 입력을 받을 때 tokenize=False로 된 문자열을 선호합니다.
    full_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False 
    )
    
    return full_prompt

def main(args):
    llm, tokenizer = load_model_and_tokenizer(args.model_id)

    try:
        a = pd.read_json("/workspace/daeyong/ideal_steps/2wiki_ideal_steps_passage_mapped_2.json").sample(100, random_state=42)
        b = pd.read_json("/workspace/daeyong/ideal_steps/hotpotqa_ideal_steps_passage_mapped_2.json").sample(100, random_state=42)
        c = pd.read_json("/workspace/daeyong/ideal_steps/musique_ideal_steps_passage_mapped_2.json").sample(100, random_state=42)
    except ValueError as e:
        print(f"⚠️ Error loading data (check file paths or sample size): {e}")
        return

    df = pd.concat([a, b, c], ignore_index=True)
    
    output_file = f"/workspace/daeyong/ideal_steps/final_step_300data.json"
    print(f"📊 Total samples to process: {len(df)}")

    # 1. 프롬프트 배치 생성
    prompts = []
    print("🛠️ Preparing prompts...")
    for _, item in df.iterrows():
        query = item['question']
        reasoning_steps = item['ideal_steps']
        prompt = prepare_prompt(query, reasoning_steps, tokenizer)
        prompts.append(prompt)

    # 2. Sampling Params
    stop_candidates = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    actual_stop_tokens = [t for t in stop_candidates if t is not None]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop_token_ids=actual_stop_tokens
    )

    # 3. vLLM 추론
    print("🚀 Running vLLM Inference...")
    outputs = llm.generate(prompts, sampling_params)

    # 4. 결과 매핑
    results = []
    # zip을 사용하여 안전하게 순회
    for item_tuple, output in zip(df.iterrows(), outputs):
        _, item = item_tuple # iterrows는 (index, Series)를 반환
        
        generated_text = output.outputs[0].text.split("assistantfinal")[-1].strip()
        
        # Series -> Dict 변환 후 결과 추가
        item_dict = item.to_dict()
        item_dict['generated_answer'] = generated_text
        results.append(item_dict)

    # 5. 저장
    print(f"💾 Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/workspace/hf_transformers/gpt-oss-120b")
    args = parser.parse_args()
    
    main(args)