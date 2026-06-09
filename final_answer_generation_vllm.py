import argparse
import os
import json
from tqdm import tqdm
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
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9, 
        trust_remote_code=True,
        max_model_len=1500,
    )
    
    return llm, tokenizer

def prepare_prompt(query: str, reasoning_steps: list, tokenizer) -> str:
    """
    질문과 추론 단계를 받아 vLLM에 입력할 프롬프트 '문자열'을 생성합니다.
    """
    previous_steps_str = '\n'.join(reasoning_steps)

    system_prompt = """You are an expert answer extractor.
Your task is to examine the provided `Question` and the `Reasoning Process` (a list of steps), and determine the final answer.

## Critical Instructions (Priority)
1. **Target Identification**: First, identify the Target Entity Type requested by the `Question` (e.g., Film Name, Person Name, Yes/No).
2. **Contextual Bridging (Look Back)**:
    - The final reasoning step might only contain an intermediate value (e.g., a date "1951", a person "Director A", or a boolean "True").
    - If this happens, you **MUST** review the **entire** `Reasoning Process` (previous steps) or the `Question` text to find the specific Entity associated with that conclusion.
    - *Example*: If the reasoning concludes "The director of film A died first", find the name of "film A" mentioned in the earlier steps or the question.
3. **Logical Inference**: You are allowed to extract the answer if it is **unambiguously derived** from the provided text, even if it is not explicitly repeated in the final step.

## Rules
1. **Short Answer Only**: Output the entity, name, date, or number directly. Do not write a full sentence.
2. **No Hallucination**: Do not invent information not present in the provided text.
3. **Unanswerable**: Output `Cannot Answer` **ONLY** if the reasoning steps provide **insufficient information** to link the conclusion back to the requested entity. Do not output `Cannot Answer` just because the exact string is missing in the *last* step.
""".strip()

    user_prompt = f"""## Question ##
{query}

## Reasoning Process ##
{previous_steps_str}

## Final Answer ##
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

def main(llm, tokenizer, dataset, model_id, prefix):
    if "llama" in model_id.lower():
        name = "llama8b"
    elif "gemma" in model_id.lower():
        name = "gemma12b"
    elif "qwen" in model_id.lower():
        name = "qwen7b"
    else:        
        name = "unknown_model"
    
    input_file = f"/workspace/daeyong/ours_{prefix}_{name}_{dataset}.json"
    output_file = f"/workspace/daeyong/ours_{prefix}_{name}_{dataset}_answer.json"
    
    print(f"Loading data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"⚠️ File not found: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # 1. 모든 데이터에 대해 프롬프트 미리 생성 (Batch Preparation)
    prompts = []
    for item in data:
        query = item.get('question', '')
        reasoning_steps = item.get('response', [])
        prompt = prepare_prompt(query, reasoning_steps, tokenizer)
        prompts.append(prompt)

    # 2. Sampling Params 설정
    stop_candidates = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    actual_stop_tokens = [t for t in stop_candidates if t is not None]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        stop_token_ids=actual_stop_tokens
    )

    # 3. vLLM 일괄 추론 실행 (High Throughput)
    print("🚀 Running vLLM Inference...")
    outputs = llm.generate(prompts, sampling_params)

    # 4. 결과 매핑
    results = []
    for i, item in enumerate(data):
        # vLLM output 객체에서 텍스트 추출
        generated_text = outputs[i].outputs[0].text.strip()
        
        new_item = item.copy()
        new_item['generated_answer'] = generated_text
        results.append(new_item)

    # 5. 결과 저장
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    args = parser.parse_args()
    
    # 모델 및 토크나이저 로드 (한 번만 실행)
    llm, tokenizer = load_model_and_tokenizer(args.model_id)

    # 데이터셋 루프
    for dataset in ["2wiki", "hotpotqa", "musique"]:
        main(llm, tokenizer, dataset, args.model_id, args.prefix)