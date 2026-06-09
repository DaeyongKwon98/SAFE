import json
import os
import re
import argparse
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# vLLM 임포트
from vllm import LLM, SamplingParams

# =======================================================
# 1. System Prompt 수정 (JSON 출력 및 Step 1 집중)
# =======================================================
system_prompt = """You are an expert in logical reasoning, tasked with intentionally introducing a specific logical error into the VERY FIRST STEP of a reasoning process.

Your Goal: Replace "Step 1" of the correct reasoning with an 'Off-topic' error.

Error Definition: 'Off-topic' (Start Failure)
An 'Off-topic' error in Step 1 means the reasoning agent starts by retrieving information that is factually correct based on the documents but **irrelevant to the specific question asked**.
- It captures the "Distraction" phenomenon: The agent sees a keyword in the question (e.g., a movie name) and retrieves an unrelated fact about it (e.g., release date instead of director) or a fact about a similarly named entity.
- The step MUST be based on the provided Passages.

Output Format:
You MUST output a valid JSON object with the following keys:
1. "off_topic_step": The new erroneous Step 1 string. It must start with "Step 1:" and end with "(Attribution)".
2. "diagnosis": A brief explanation of why this step is irrelevant to the question.
3. "guidance": A concise instruction on what the correct Step 1 should have focused on.

---
EXAMPLES
---

Question: "Kim Sŏng-ae is the wife of which leader of North Korea who died in 1994?"

Retrieved Passages:
"Passage 1: Kim Il-sung ... died in 1994..."
"Passage 5: Kang Pan-sok ... was the mother of North Korean leader Kim Il-sung..."

Ideal Step 1: 
"Step 1: According to Passage 1, the leader of North Korea who died in 1994 is Kim Il-sung. (Attribution)"

Output:
{
  "off_topic_step": "Step 1: According to Passage 5, Kang Pan-sok was the mother of North Korean leader Kim Il-sung. (Attribution)",
  "diagnosis": "The step provides information about Kim Il-sung's mother, which is irrelevant to identifying the leader himself based on the death year.",
  "guidance": "Start by identifying the North Korean leader who died in 1994 using Passage 1."
}

---

Question: "When is the director of film For Love Or Money (2014 Film)'s birthday?"

Retrieved Passages:
"Passage 5: For Love or Money (1963 film): ... directed by Michael Gordon..."
"Passage 6: For Love or Money (2014 film): ... The film was directed by Gao Xixi..."

Ideal Step 1: 
"Step 1: According to Passage 6, the director of the film 'For Love or Money (2014)' is Gao Xixi. (Attribution)"

Output:
{
  "off_topic_step": "Step 1: According to Passage 5, the film 'For Love or Money (1963 film)' was directed by Michael Gordon. (Attribution)",
  "diagnosis": "The step retrieves information about the 1963 film instead of the requested 2014 film due to the similar title.",
  "guidance": "Focus on the 2014 version of the film 'For Love or Money' mentioned in Passage 6 to identify the correct director."
}
"""

def parse_json_response(response: str) -> Dict[str, str]:
    """
    LLM 응답에서 JSON 객체를 추출합니다.
    Markdown code block (```json ... ```) 또는 순수 JSON 문자열을 처리합니다.
    """
    response = response.strip()
    
    # 1. Code block 제거
    match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 2. 중괄호로 시작하고 끝나는 부분 찾기
        match = re.search(r"(\{.*\})", response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError:
        print(f"\n⚠️ JSON Parsing Failed. Raw response:\n{response}")
        return None

def save_results(data: List[Dict[str, Any]], filepath: str):
    """결과를 JSON 파일로 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main(args):
    # =======================================================
    # 2. vLLM 모델 초기화
    # =======================================================
    print(f"🚀 Loading vLLM model: {args.model_name}")
    
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=4,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=6000
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048
    )

    # ✅ 데이터 로드
    input_filepath = f"/workspace/daeyong/fourth_finetuning_data/off_topic_first_ingredient.json"
    output_filepath = f"/workspace/daeyong/fourth_finetuning_data/off_topic_first.json"
    
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {input_filepath}")
    except Exception as e:
        print(f"Error loading data from {input_filepath}: {e}")
        return

    # 기존 결과 로드 (Resume 기능)
    results = []
    processed_ids = set()
    
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
            # 이미 처리된 질문을 Set에 저장 (Target Step은 항상 1이므로 질문만 키로 사용해도 무방하나, 구조 유지)
            for res in results:
                if 'question' in res:
                    processed_ids.add(res['question'])
            print(f"Loaded {len(results)} existing results from {output_filepath}. Resuming...")
        except Exception as e:
            print(f"Warning: Could not load existing results. Starting fresh. Error: {e}")
            results = []
            processed_ids = set()

    # =======================================================
    # 3. 데이터 처리 루프 (vLLM Batch 처리 대신 로직 유지 위해 개별 처리)
    #    * 속도를 위해 리스트를 만들어 llm.generate에 한 번에 넣을 수도 있지만,
    #      Resume 기능과 중간 저장을 위해 루프 방식을 유지합니다.
    # =======================================================
    
    # vLLM은 리스트 단위 추론이 빠르므로, 처리되지 않은 아이템만 모아서 배치를 구성할 수 있습니다.
    # 하지만 복잡도를 낮추기 위해 여기서는 루프 안에서 하나씩 처리하는 구조를 유지하되,
    # vLLM 호출 방식을 따릅니다.
    
    items_to_process = []
    original_indices = []

    for idx, item in enumerate(data):
        if item['question'] not in processed_ids:
            items_to_process.append(item)
            original_indices.append(idx)
    
    print(f"Items left to process: {len(items_to_process)}")
    
    if not items_to_process:
        print("All items processed.")
        return

    # 프롬프트 구성
    prompts = []
    for item in items_to_process:
        question = item['question']
        passages = item['retrieved_passages']
        ideal_steps = item['ideal_steps']
        
        # Step 1을 타겟으로 함
        target_step_content = ideal_steps[0] if len(ideal_steps) > 0 else "Step 1: Unknown"

        passages_context = "\n".join(f"Passage {i+1}: {p}" for i, p in enumerate(passages))

        user_prompt = f"""Question: {question}

Retrieved Passages:
{passages_context}

Ideal Step 1:
{target_step_content}

Target Step to Corrupt:
Step 1
"""
        # Chat template 적용
        # vLLM의 LLM 클래스는 prompt_token_ids나 prompt 문자열을 받는데,
        # Chat Model을 쓴다면 chat format을 맞춰주는 것이 좋습니다.
        # 여기서는 raw prompt string을 만들기보다 tokenizer를 이용해 포맷팅합니다.
        
        tokenizer = llm.get_tokenizer()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted_prompt)

    # =======================================================
    # vLLM Generate (Batch Inference)
    # =======================================================
    print("Generating responses with vLLM...")
    # vLLM은 리스트로 던지면 내부적으로 최적화하여 배칭 처리함
    outputs = llm.generate(prompts, sampling_params)

    # 결과 처리 및 저장
    for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
        item = items_to_process[i]
        generated_text = output.outputs[0].text.split("assistantfinal")[-1].strip()
        
        parsed_data = parse_json_response(generated_text)
        
        if parsed_data and "off_topic_step" in parsed_data:
            off_topic_step = parsed_data["off_topic_step"]
            diagnosis = parsed_data.get("diagnosis", "")
            guidance = parsed_data.get("guidance", "")
            
            # Step 1만 교체하므로 logic은 간단함
            # 기존 ideal steps 가져오기 (혹시 모르니 copy)
            corrupted_steps = []
            corrupted_steps.append(off_topic_step) # Step 1 교체
            
            # 원래 Step 1 이후의 스텝들은 그대로 둘지, 아니면 Step 1에서 망가졌으니 끊을지 결정해야 함.
            # 데이터셋의 목적상 "망가진 Step 1" + "원래 Step 1"은 논리적으로 말이 안되지만,
            # "Off-topic" 데이터 포맷에 맞추기 위해 'Target index'까지만 생성하거나
            # 혹은 전체 스텝 중 1번만 바꿈.
            # 보통 에러 데이터셋은 [이전 스텝들] + [에러 스텝] 형태임.
            # 즉, Step 1 에러 데이터는 [에러 Step 1] 만 가지고 끝나는 것이 일반적임 (더 이상 진행 불가).
            
            # 요청하신 이전 코드 로직: corrupted_steps = base_steps + [corrupted_step]
            # base_steps는 target_index가 1이므로 빈 리스트 []
            # 따라서 결과는 [off_topic_step] 하나만 있는 리스트가 됨.
            
            new_item = item.copy()
            new_item['corrupted_steps'] = [off_topic_step]
            new_item['corrupted_step_index'] = 1
            new_item['error_type'] = 'Off-topic'
            new_item['diagnosis'] = diagnosis
            new_item['guidance'] = guidance
            
            results.append(new_item)
        else:
            print(f"Failed to parse output for question: {item['question'][:30]}...")
            # 파싱 실패한 경우 원본 로그 등을 남기거나 skip

        # 안전을 위해 일정 주기마다 저장 (루프 밖에서 일괄 처리되지만, 대량일 경우 중간 저장 로직 추가 가능)
        if (i + 1) % 5 == 0:
            save_results(results, output_filepath)

    # 최종 저장
    save_results(results, output_filepath)
    print(f"✅ Completed error injection. Total {len(results)} items saved to {output_filepath}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject 'Off-topic' errors into Step 1 using vLLM.")
    parser.add_argument("--model_name", type=str, 
                        default="/workspace/hf_transformers/gpt-oss-120b",
                        help="Path to the HuggingFace model directory.")
    
    args = parser.parse_args()
    main(args)