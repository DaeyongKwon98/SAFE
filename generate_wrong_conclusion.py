import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import re

# =============================================================================
# 1. API Setup
# =============================================================================
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Global Token Counter
TOKEN_STATS = {
    "total_input": 0,
    "total_output": 0
}


def robust_json_parse(text: str):
    """
    LLM 출력 텍스트에서 JSON 객체를 robust하게 추출합니다.
    - 마크다운(```json) 제거
    - 앞뒤 불필요한 텍스트 제거
    - 잘린 JSON(Missing brackets/quotes) 복구 시도
    """
    if not text:
        return None

    # 1. 마크다운 코드 블록 제거 (```json ... ```)
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)
    
    # 2. 첫 번째 '{' 위치 찾기
    start_idx = text.find('{')
    if start_idx == -1:
        return None # JSON 객체가 없음
    
    # '{' 부터 끝까지만 잘라냄 (앞부분의 자연어 설명 제거)
    json_candidate = text[start_idx:].strip()
    
    # 3. 뒤쪽의 불필요한 자연어 제거 시도 (마지막 '}' 찾기)
    # 다만, JSON이 잘린 상태라면 마지막 '}'가 없을 수도 있으므로
    # 먼저 원본(또는 마지막 '}'까지)으로 시도해보고, 안되면 복구 로직으로 넘어감.
    end_idx = json_candidate.rfind('}')
    
    candidates = []
    
    # 후보 1: 잘라낸 문자열 그대로
    candidates.append(json_candidate)
    
    # 후보 2: 마지막 '}' 까지만 잘라냄 (뒤에 자연어가 붙은 경우 대비)
    if end_idx != -1:
        candidates.append(json_candidate[:end_idx+1])
    
    # 후보 3~N: 잘린 JSON 복구 시도 (Bracket/Quote Completion)
    # LLM이 생성하다가 끊긴 경우를 위해 끝에 닫는 기호들을 순차적으로 붙여봄
    completion_patterns = ['}', '"}', '"]', '}}', '"}}', '}]', '"}]']
    for pattern in completion_patterns:
        candidates.append(json_candidate + pattern)

    # 4. 파싱 시도
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
            
    # 5. 모든 시도가 실패했을 때 (Python의 eval 사용 시도 - 최후의 수단)
    # 주의: 보안상 안전한 데이터라고 확신할 때만 사용해야 함.
    # JSON 표준은 아니지만 Python Dict 형태인 경우(Single quote 등)를 처리
    try:
        import ast
        return ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        pass

    print(f"Error: Failed to parse JSON from text: {text[:50]}...")
    return None

# =============================================================================
# 2. System Prompt (Strict Entity Extractor)
# =============================================================================
system_prompt = """You are an expert in generating synthetic data for evaluating reasoning models. Your task is to generate a specific type of error called **"Wrong Conclusion"**, along with a **Diagnosis** explaining the error and **Guidance** to correct it.

### Task Description
You will be provided with:
1. **Question**: A multi-hop question.
2. **Reasoning Steps**: A list of correct logical steps leading up to the answer.
3. **Correct Step**: The valid final answer derived from the reasoning.

Your goal is to generate a JSON object containing:
- "generated_wrong_step": A final step where the model fails at the last moment.
- "diagnosis": An explanation of why the final step is wrong.
- "guidance": Instructions on how to generate the correct answer.

### Error Generation Strategy
Analyze the **Question** and **Correct Step** to decide which error type to apply:

#### **Type A: Logical Reversal (For Binary/Comparative Questions)**
* **Trigger:** If the question asks for **Yes/No**, **True/False**, or a **Comparison** (e.g., "Which is older?", "Who died first?").
* **Action:** Generate an answer that is the **direct opposite** of the Correct Step.
* **Diagnosis:** State that the reasoning steps logically point to one conclusion, but the final answer incorrectly selects the opposite/other option.
* **Guidance:** Instruct to select the answer that aligns with the logical conclusion derived in the previous steps.

#### **Type B: Entity Mismatch (For Open-ended/Attribute Questions)**
* **Trigger:** If the question asks for a specific attribute (Nationality, Date, Name, Place) and involves intermediate entities.
* **Action:** Answer with a prominent **Intermediate Entity** from the `Reasoning Steps` instead of the requested target attribute.
* **Diagnosis:** State that the question asks for [Target Type], but the answer provides [Wrong Entity Type] (e.g., "The question asks for the nationality, but the answer provides the performer's name").
* **Guidance:** Instruct to provide the specific attribute requested by the question, based on the identified entity.

### Examples

**Example 1 (Type A - Boolean Reversal)**
* **Input:**
    * Question: "Are James and Bob from the same country?"
    * Reasoning Steps: ["Step 1: James is from Germany.", "Step 2: Bob is from America.", "Step 3: Germany and America are different countries."]
    * Correct Step: "Step 4: ####ANSWER: No (Final Answer)"
* **Output:**
{
"generated_wrong_step": "Step 4: ####ANSWER: Yes (Final Answer)",
"diagnosis": "The previous steps clearly establish that James and Bob are from different countries (Germany and America). However, the final answer contradicts this deduction by stating 'Yes'.",
"guidance": "Based on the conclusion in Step 3 that the countries are different, the correct final answer should be 'No'. Submit the final answer as ####ANSWER: No."
}

**Example 2 (Type A - Comparative Reversal)**
* **Input:**
    * Question: "Which film has the director died first, movie A or movie B?"
    * Reasoning Steps: ["Step 1: Director X (movie A) died in 1980.", "Step 2: Director Y (movie B) died in 2013.", "Step 3: 1980 is earlier than 2013, so Director X died first."]
    * Correct Step: "Step 4: ####ANSWER: movie A (Final Answer)"
* **Output:**
{
"generated_wrong_step": "Step 4: ####ANSWER: movie B (Final Answer)",
"diagnosis": "The reasoning correctly identifies that Director X (Movie A) died earlier than Director Y. However, the final answer incorrectly names 'movie B' instead of 'movie A', contradicting the comparison in Step 3.",
"guidance": "Since Step 3 establishes that the director of movie A died first, the final answer must match this conclusion. Submit the final answer as ####ANSWER: movie A."
}

**Example 3 (Type B - Entity Mismatch)**
* **Input:**
    * Question: "What nationality is the performer of song Happiness Forever?"
    * Reasoning Steps: ["Step 1: The performer is Singer X.", "Step 2: Singer X is American."]
    * Correct Step: "Step 3: ####ANSWER: American (Final Answer)"
* **Output:**
{
"generated_wrong_step": "Step 3: ####ANSWER: Singer X (Final Answer)",
"diagnosis": "The question explicitly asks for the 'nationality' of the performer. While the reasoning correctly identifies the performer as Singer X, the final answer provides the performer's name instead of their nationality.",
"guidance": "The question asks for the nationality, not the name. Use the information from Step 2 that Singer X is American to provide the correct answer. Submit the final answer as ####ANSWER: American."
}

---

### Current Task

**Input:**
* Question: question
* Reasoning Steps: reasoning_steps
* Correct Step: current_step

**Output:**
Generate only the JSON object

Your generated_wrong_step should have the exact same format as the Correct Step, starting with "Step X: ####ANSWER: answer_here (Final Answer)".
""".strip()

# =============================================================================
# 3. GPT Generation Function
# =============================================================================
def generate_final_step(item):
    """
    Calls GPT to generate the final answer step.
    """
    query = item['question']
    reasoning_steps = item['ideal_steps']

    # 리스트를 텍스트로 변환 및 Step 번호 계산
    if isinstance(reasoning_steps, list):
        steps_text = "\n".join(reasoning_steps)
        next_step_num = len(reasoning_steps) + 1
    else:
        steps_text = str(reasoning_steps)
        next_step_num = steps_text.count('\n') + 2

    user_prompt = f"""## Question ##
{query}

## Reasoning Steps ##
{steps_text}

## Correct Step ##
{item['generated_answer']}
""".strip()

    try:
        completion = client.chat.completions.create(
            model="gpt-5.1", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=2048,
        )
        
        response_content = completion.choices[0].message.content.strip()
        usage = completion.usage
        
        return response_content, usage

    except Exception as e:
        print(f"API Error: {e}")
        return None, None

# =============================================================================
# 4. Main Execution Loop
# =============================================================================
def main():
    # 데이터 로드 경로
    input_path = "/workspace/daeyong/ideal_steps/correct_data_for_wrong_conclusion_generation.json"
    output_path = "/workspace/daeyong/fourth_finetuning_data/wrong_conclusion.json"

    # 1. Load & Merge Data
    print("📂 Loading data...")
    try:
        df = pd.read_json(input_path)
    except ValueError as e:
        print(f"⚠️ Error loading {input_path}: {e}")
        return

    print(f"📊 Total samples to process: {len(df)}")

    # 2. Resume Logic
    results = []
    processed_count = 0
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_count = len(results)
            print(f"🔄 Resuming from {processed_count} records.")
        except:
            print("⚠️ Output file corrupted. Starting over.")

    # 3. Processing Loop
    print(f"🚀 Starting Evaluation on {len(df) - processed_count} items...")
    
    # 이미 처리된 데이터 건너뛰기
    data_to_process_df = df.iloc[processed_count:]
    
    for i, row in tqdm(data_to_process_df.iterrows(), total=len(data_to_process_df), desc="Generating"):
        item = row.to_dict()
        
        generated_text, usage = generate_final_step(item)
        generated_data = robust_json_parse(generated_text)
        
        if generated_data:
            # Update Token Stats
            TOKEN_STATS["total_input"] += usage.prompt_tokens
            TOKEN_STATS["total_output"] += usage.completion_tokens
            
            # Save Result
            item["error_type"] = "Wrong Conclusion"
            item["generated_wrong_step"] = generated_data["generated_wrong_step"]
            item["diagnosis"] = generated_data["diagnosis"]
            item["guidance"] = generated_data["guidance"]
            results.append(item)

            if len(results) % 5 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                print(f"Total Tokens: {TOKEN_STATS['total_input'] + TOKEN_STATS['total_output']:,}")

        else:
            print("⚠️ Skipping item due to API error.")

    print(f"✅ Completed. Final Total Tokens: {TOKEN_STATS['total_input'] + TOKEN_STATS['total_output']:,}")

if __name__ == "__main__":
    main()