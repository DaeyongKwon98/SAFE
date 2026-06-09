from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import json
import re
from tqdm import tqdm
import pandas as pd
import argparse
import math

system_prompt = """You are a data processing assistant specialized in structuring feedback text.
Your task is to parse the provided "Feedback" text into a structured JSON object containing "Diagnosis" and "Guidance".

### Definitions:
1. **Diagnosis**: The part of the text that evaluates the current step (e.g., "You correctly identified the birth date", "The step is incorrect because it cites the wrong passage", "The reasoning is complete").
2. **Guidance**: The part of the text that instructs what to do next (e.g., "in the next step, please compare the dates", "You should directly attribute this information", "Stop reasoning now and terminate").

### Strict Rules:
- **Preserve Wording**: Do not rewrite or summarize. Extract the exact substrings from the original text.
- **Full Extraction**: Do not truncate the text. Do not add ellipses (...) at the end. Extract the full sentence or clause as it appears in the input.
- **Missing Information**: If the feedback contains ONLY a diagnosis or ONLY guidance, set the missing field to an empty string `""`. Do NOT hallucinate or generate new text.
- **Split Logically**: If a single sentence contains both, split it logically into the two fields.

### Examples:

**Input:**
"You have correctly identified the death date of John Spencer; in the next step, please use this information to determine the answer."
**Output:**
{
  "Diagnosis": "You have correctly identified the death date of John Spencer;",
  "Guidance": "in the next step, please use this information to determine the answer."
}

**Input:**
"The step incorrectly claims that the information is missing. Passage 7 explicitly describes the characteristics."
**Output:**
{
  "Diagnosis": "The step incorrectly claims that the information is missing. Passage 7 explicitly describes the characteristics.",
  "Guidance": ""
}

**Input:**
"Please compare the birth dates of the two directors to determine which one is younger."
**Output:**
{
  "Diagnosis": "",
  "Guidance": "Please compare the birth dates of the two directors to determine which one is younger."
}
""".strip()

# ==========================================
# 2. Helper Functions
# ==========================================

def clean_and_parse_json(response_text: str):
    """
    Extracts JSON object from LLM response.
    Handles Markdown code blocks, raw text, and unescaped quotes inside strings.
    """
    # 1. Try extracting from code block first
    code_block_match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
    else:
        # 2. Try finding the first '{' and last '}'
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx : end_idx + 1]
        else:
            json_str = response_text.strip()

    # 3. First Attempt: Standard JSON parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 4. Fallback: Manual Extraction using Regex
        # "Diagnosis"와 "Guidance" 사이의 구조적 특징(키와 콤마)을 이용하여 내용을 강제 추출합니다.
        try:
            # 패턴 설명:
            # "Diagnosis" 키 뒤의 콜론과 따옴표를 찾고 -> (내용 캡처) -> 뒤따라오는 "Guidance" 키 직전의 콤마와 따옴표까지
            # "Guidance" 키 뒤의 콜론과 따옴표를 찾고 -> (내용 캡처) -> 닫는 중괄호 직전의 따옴표까지
            
            # 주의: re.DOTALL을 써서 개행 문자도 포함해서 캡처해야 합니다.
            diagnosis_pattern = r'"Diagnosis"\s*:\s*"(.*?)"\s*,\s*"Guidance"'
            guidance_pattern = r'"Guidance"\s*:\s*"(.*?)"\s*\}'
            
            diagnosis_match = re.search(diagnosis_pattern, json_str, re.DOTALL)
            guidance_match = re.search(guidance_pattern, json_str, re.DOTALL)
            
            if diagnosis_match and guidance_match:
                diagnosis_content = diagnosis_match.group(1).strip()
                guidance_content = guidance_match.group(1).strip()
                
                # 캡처된 내용 안에 있을 수 있는 이스케이프 문자 처리 (선택 사항)
                # LLM이 \" 라고 잘 썼는데 Regex로 캡처하면 \가 그대로 들어올 수 있으므로 unescape 처리
                # 하지만 단순히 큰따옴표 문제라면 raw string 그대로 써도 무방합니다.
                
                return {
                    "Diagnosis": diagnosis_content,
                    "Guidance": guidance_content
                }
            else:
                # 패턴 매칭도 실패한 경우
                raise ValueError("Regex extraction failed")
                
        except Exception as e:
            print(f"⚠️ JSON Decode & Repair Failed: {e}\nRaw Output: {response_text}")
            return None

def generate_rewrite(tokenizer, llm, feedback_text):
    """
    Generates the structured Diagnosis/Guidance JSON using the system prompt.
    """
    user_content = f"**Input:**\n\"{feedback_text}\"\n\n**Output (JSON):**"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
    )

    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    response = outputs[0].outputs[0].text

    return response.split("assistantfinal")[-1].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_id", type=int, required=True, help="0 to 3")
    parser.add_argument("--total_splits", type=int, default=4, help="Total partitions")
    args = parser.parse_args()

    split_id = args.split_id
    total_splits = args.total_splits

    # 파일 경로
    input_path = "/workspace/daeyong/inaccurate_cases.json"
    base_output_path = "/workspace/daeyong/inaccurate_cases_rewritten.json"
    
    # 내 결과 파일 (Local Part File)
    root, ext = os.path.splitext(base_output_path)
    my_output_path = f"{root}_part_{split_id}{ext}"

    # 1. 원본 전체 로드
    df_all = pd.read_json(input_path)
    print(f"📂 [Input] Original Total: {len(df_all)} items.")

    # 2. Global Offset 계산 (이미 처리된 개수만큼 앞에서 제외)
    global_offset = 0
    if os.path.exists(base_output_path):
        try:
            with open(base_output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                global_offset = len(existing_data)
            print(f"🛑 [Global Skip] Skipping first {global_offset} items (already in base file).")
        except:
            print("⚠️ Base output file error. Assuming 0 processed.")

    # 3. 남은 데이터 슬라이싱 (Offset 이후 데이터만 사용)
    # iloc[global_offset:] -> 앞에서부터 global_offset 개수만큼 자름
    df_remaining = df_all.iloc[global_offset:].reset_index(drop=True)
    total_remaining = len(df_remaining)
    print(f"📝 [Remaining To-Do] {total_remaining} items left to process.")

    if total_remaining == 0:
        print("🎉 Nothing left to do! Exiting.")
        return

    # 4. 남은 데이터를 4등분 (Chunking)
    chunk_size = math.ceil(total_remaining / total_splits)
    start_idx = split_id * chunk_size
    end_idx = min((split_id + 1) * chunk_size, total_remaining)

    # 내 할당량 (Chunk)
    my_chunk = df_remaining.iloc[start_idx:end_idx]
    print(f"🔥 [Process {split_id}] Assigned Chunk Size: {len(my_chunk)} (Index {start_idx} ~ {end_idx})")

    # 5. Local Resume (내 파트 파일 확인)
    # 이미 part_X.json에 저장된 개수만큼, my_chunk의 앞에서부터 건너뜀
    my_results = []
    local_done_count = 0

    if os.path.exists(my_output_path):
        try:
            with open(my_output_path, 'r', encoding='utf-8') as f:
                my_results = json.load(f)
                local_done_count = len(my_results)
            print(f"🔄 [Local Resume] Found {local_done_count} items in {my_output_path}. Resuming from there.")
        except:
            print("⚠️ Local part file empty/invalid. Starting fresh.")

    # 실제로 작업해야 할 데이터 (Local Resume 적용)
    # my_chunk에서 이미 처리한 개수만큼 제외
    df_to_process = my_chunk.iloc[local_done_count:]
    
    if len(df_to_process) == 0:
        print("🎉 Local chunk already finished! Exiting.")
        return

    # 6. 모델 로드
    model_name = "/workspace/hf_transformers/gpt-oss-120b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        max_model_len=3000,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    save_interval = 10
    
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc=f"Split-{split_id}"):
        feedback = row['feedback']

        try:
            # Generate
            raw_response = generate_rewrite(tokenizer, llm, feedback)
            
            # Parse
            parsed_json = clean_and_parse_json(raw_response)
            
            if not parsed_json:
                print(f"⚠️ Parsing Failed. Skipping.")
                continue
            
            # 결과 저장
            row_dict = row.to_dict()
            row_dict['feedback_rewritten'] = parsed_json
            
            my_results.append(row_dict)
            
            # 중간 저장
            if len(my_results) % save_interval == 0:
                with open(my_output_path, "w", encoding="utf-8") as f:
                    json.dump(my_results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

    # 최종 저장
    with open(my_output_path, "w", encoding="utf-8") as f:
        json.dump(my_results, f, indent=2, ensure_ascii=False)

    print(f"✅ [Process {split_id}] Completed! Saved {len(my_results)} items to {my_output_path}")

if __name__ == "__main__":
    main()