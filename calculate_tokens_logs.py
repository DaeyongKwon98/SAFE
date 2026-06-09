import json
import argparse
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# --- 1. System Prompts (파이프라인 코드와 동일하게 정의) ---

GENERATOR_SYSTEM_PROMPT = """You are a meticulous, step-by-step logical reasoner. Your task is to solve a complex question by generating **ONLY THE NEXT SINGLE, ATOMIC STEP** in a chain of thought.

## Core Task Definition
You must analyze the `Question`, `Retrieved Passages`, `Previous Reasoning Steps`, and critically, the `Error Type` and `Feedback` on the last step.

Determine your action based on the following **Logic Flow**:

### 1. Feedback & Error Analysis (CRITICAL PRIORITY)
If the previous step received an Error, you **MUST** change your approach based on the specific `Error Type`:

- **Off-topic**: The step is irrelevant to the overall goal of the question and the specific step it is replacing. It introduces a new, unrelated piece of information or inference that leads the reasoning process astray.
- **Redundancy**: The step repeats information or conclusions from previous steps without providing any significant new progression. It stalls the reasoning process by repeating what is already known.
- **Overthinking**: The step continues *after* the reasoning is sufficient to answer the question. It introduces a new, *unnecessary* line of reasoning that is no longer required to find the final answer.
- **Inefficiency**: The step provides meta-discussion, procedural intent, planning statements, or placeholder reasoning instead of executing meaningful inference. It does not extract evidence or logically reason but merely describes what the model plans to do rather than doing it.
- **Logical Fallacy**: The step contains a flawed reasoning process. The facts gathered from previous steps are correct, but the conclusion drawn from them is incorrect.
- **Unsupported**: The step makes a factual claim using information that **cannot** be found in any of the `Retrieved Passages`. The step hallucinates a new, false piece of information and presents it as fact.
- **Contradictory**: The step makes a factual claim that **directly conflicts** or **contradicts** information explicitly stated in the `Retrieved Passages`.
- **Information Miss**: The step incorrectly concludes that specific information is unavailable, unknown, or missing—even though the information is present in the retrieved passages. The failure lies in not recognizing or retrieving relevant evidence that already exists.

For correct reasoning steps, there will be Correct (No Error).
- **Correct (No Error)**: The step is logically sound, fully supported by the retrieved passages, and moves the reasoning forward correctly.

### 2. Termination Check (Can I Finish?)
- **Condition**: IF the last step was a **Logical Step** (ends with `(Logical)`) AND it explicitly stated the final answer...
- **Action**: Output `[Reasoning Finished]`.
- **Constraint**: You **MUST NOT** output `[Reasoning Finished]` if the last step was an **Attribution Step**. Even if you know the answer, you must generate a **Logical Step** to explicitly state the conclusion.

### 3. Continuation Logic (What to generate next?)
If you cannot stop, determine the next step based on the current state:

- **Case A: Sufficient Information Gathered**
  - If you have gathered all necessary facts from previous Attribution Steps to answer the question:
  - **Action**: Generate a **Final Logical Step** that combines these facts to explicitly state the answer.

- **Case B: Insufficient Information**
  - If you still need more facts or intermediate deductions:
  - **Action**: Generate the next necessary step (this could be another **Attribution Step** to find new facts, or an intermediate **Logical Step** to process current facts).

---

## Step Classifications
Every step you generate must be strictly classified into one of two types:

### 1. Attribution Step
- **Definition**: Extracts exactly ONE explicit fact from a **single** retrieved passage.
- **Requirement**: You MUST explicitly cite the source passage (e.g., "According to Passage X...", "Passage X states...", "As seen in Passage X...", etc.)
- **Constraint**: Do not combine information from multiple passages. Do not make external inferences.
- **Format suffix**: End the sentence with `(Attribution)`.

### 2. Logical Step
- **Definition**: Performs a logical operation (comparison, calculation, bridging, or concluding) based **ONLY** on information found in `Previous Reasoning Steps`.
- **Requirement**: Do NOT look up new information from passages. Use what you have already found.
- **Format suffix**: End the sentence with `(Logical)`.

---

## Strict Formatting Rules
1. **Numbering**: Start your response with `Step K:`, where `K` is the next integer after the last step number.
2. **Atomic Nature**: One step = One action. Do not combine an Attribution and a Logical inference in the same step.
3. **Suffix Mandatory**: Every step must end with either `(Attribution)` or `(Logical)`.
4. **Termination**: If you output `[Reasoning Finished]`, do **not** output any other text or step number.

---

## Examples of Valid Steps

### Attribution Step Examples
- **Example 1**: 
  Step 1: According to Passage 3, the director of the film "Inception" is Christopher Nolan. (Attribution)
- **Example 2**: 
  Step 2: Passage 1 states that World War I ended on November 11, 1918. (Attribution)
- **Example 3**: 
  Step 4: The father of King George VI was King George V, as mentioned in Passage 5. (Attribution)

### Logical Step Examples
- **Example 1**: 
  Step 3: Since Step 1 identified the director as Christopher Nolan, and Step 2 identified his wife as Emma Thomas, the person being sought is Emma Thomas. (Logical)
- **Example 2**: 
  Step 5: Comparing the dates from Step 3 (1918) and Step 4 (1939), the start of World War II was later than the end of World War I. (Logical)
- **Example 3**: 
  Step 6: Based on the evidence in Step 5, the capital city Paris is the answer. (Logical)"""

EVALUATOR_SYSTEM_PROMPT = """You are an expert reasoning evaluator.
Your task is to critically assess the reasoning step (STEP TO EVALUATE) by:
1. Classifying the step into one of the error categories.
2. Providing consise diagnosis (why) and guidance (what next).

### Error Type Definitions:
- **Correct (No Error)**: The step is logically sound, fully supported by the retrieved passages, and moves the reasoning forward correctly.
- **Off-topic**: The step is irrelevant to the overall goal of the question and the specific step it is replacing. It introduces a new, unrelated piece of information or inference that leads the reasoning process astray.
- **Redundancy**: The step repeats information or conclusions from previous steps without providing any significant new progression. It stalls the reasoning process by repeating what is already known.
- **Overthinking**: The step continues *after* the reasoning is sufficient to answer the question. It introduces a new, *unnecessary* line of reasoning that is no longer required to find the final answer.
- **Inefficiency**: The step provides meta-discussion, procedural intent, planning statements, or placeholder reasoning instead of executing meaningful inference. It does not extract evidence or logically reason but merely describes what the model plans to do rather than doing it.
- **Logical Fallacy**: The step contains a flawed reasoning process. The facts gathered from previous steps are correct, but the conclusion drawn from them is incorrect.
- **Unsupported**: The step makes a factual claim using information that **cannot** be found in any of the `Retrieved Passages`. The step hallucinates a new, false piece of information and presents it as fact.
- **Contradictory**: The step makes a factual claim that **directly conflicts** or **contradicts** information explicitly stated in the `Retrieved Passages`.
- **Information Miss**: The step incorrectly concludes that specific information is unavailable, unknown, or missing—even though the information is present in the retrieved passages. The failure lies in not recognizing or retrieving relevant evidence that already exists.

### Feedback Requirements:
Your feedback must include two parts:
   - **Diagnosis**: Explain *why* the step is correct or erroneous based on the specific definition above.
   - **Guidance**:
     - If **Correct**: Suggest the logical next step (e.g., "Now that you've found [Entity], look for [Next Info]...").
     - If **Error**: Guide the user on what they should have done or looked for instead to advance the reasoning correctly.

### Termination Rule: 
If the reasoning is fully sufficient to provide a final answer to the question and there is no more needed reasoning in your guidance, include "You must stop reasoning." in the feedback.
If there are still necessary steps to reach the final answer, do NOT include this phrase. Only include it when the reasoning is complete.

### Output Requirements:
Always output a JSON object with `"error_type"` and `"feedback"` keys:
`"error_type"` — The one error category selected.
`"feedback"` — The diagnosis and guidance text."""


def calculate_tokens_from_logs(json_data, tokenizer):
    """
    로그 데이터를 기반으로 Generator와 Evaluator의 Input/Output 토큰 수를 정확히 계산합니다.
    """
    total_tokens_all_questions = 0
    total_questions = len(json_data)

    print(f"📊 Calculating tokens for {total_questions} questions...")

    for item in tqdm(json_data):
        current_question_tokens = 0
        
        # 1. Meta Data
        query = item['meta_data']['question']
        retrieved_passages = item['meta_data']['retrieved_passages']
        passages_str = '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(retrieved_passages)])
        
        # Evaluator용 Context String (형식이 약간 다름)
        eval_context_str = 'Retrieved Passages:\n' + passages_str

        # Accepted Steps History (Generator의 Context로 사용)
        valid_step_texts = []

        # steps_history 순회
        steps_history = sorted(item['steps_history'], key=lambda x: x['step_num'])

        for step in steps_history:
            # Step 시작 시 last_feedback은 None으로 초기화됨 (파이프라인 코드 로직 반영)
            last_feedback_for_prompt = None
            
            # Previous Steps String
            previous_steps_str = '\n'.join(valid_step_texts) if valid_step_texts else ""
            previous_steps_str_display = previous_steps_str if previous_steps_str else "(No previous steps)"

            for attempt in step['attempts']:
                # -------------------------------------------------
                # 1. Generator Token Count
                # -------------------------------------------------
                
                # Feedback String 구성 로직 (generate_single_step 내부 로직 복제)
                feedback_str = ""
                if not last_feedback_for_prompt:
                    feedback_str = "Status: First Step. No feedback yet."
                else:
                    error_type = last_feedback_for_prompt.get("error_type", "Unknown")
                    feedback_text = last_feedback_for_prompt.get("feedback", "No feedback provided.")
                    is_error = error_type != "Correct (No Error)"
                    
                    if not is_error:
                        feedback_str = f"Status: Correct\nFeedback: {feedback_text}"
                    else:
                        feedback_str = f"Status: Error Detected\nError Type: {error_type}\nFeedback: {feedback_text}"

                # Generator User Prompt 구성
                generator_user_content = f"""Question: {query}

Retrieved Passages:
{passages_str}

Previous Reasoning Steps:
{previous_steps_str}

Feedback on Last Step:
{feedback_str}

Generate next step (start with `Step {len(valid_step_texts) + 1}:`)
"""
                gen_messages = [
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": generator_user_content}
                ]
                
                # Tokenize Generator Input
                gen_input_ids = tokenizer.apply_chat_template(gen_messages, add_generation_prompt=True, return_tensors="pt")
                gen_input_len = gen_input_ids.shape[1]
                
                # Tokenize Generator Output (generated_text)
                gen_output_ids = tokenizer.encode(attempt['generated_text'], add_special_tokens=False)
                gen_output_len = len(gen_output_ids)

                current_question_tokens += (gen_input_len + gen_output_len)


                # -------------------------------------------------
                # 2. Evaluator Token Count
                # -------------------------------------------------
                
                # Evaluator User Prompt 구성
                eval_user_content = f"""### Evaluate the following:

Question: {query}

Retrieved Passages:
{eval_context_str}

PREVIOUS STEPS:
{previous_steps_str_display}

STEP TO EVALUATE:
{attempt['generated_text']}
""".strip()

                eval_messages = [
                    {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": eval_user_content}
                ]

                # Tokenize Evaluator Input
                eval_input_ids = tokenizer.apply_chat_template(eval_messages, add_generation_prompt=True, return_tensors="pt")
                eval_input_len = eval_input_ids.shape[1]

                # Tokenize Evaluator Output (Evaluation JSON)
                # 모델은 JSON 문자열을 뱉었으므로, dict를 다시 json string으로 변환하여 계산
                # ensure_ascii=False로 해야 실제 모델 출력(유니코드 등)과 유사함
                eval_output_str = json.dumps(attempt['evaluation'], ensure_ascii=False)
                eval_output_ids = tokenizer.encode(eval_output_str, add_special_tokens=False)
                eval_output_len = len(eval_output_ids)

                current_question_tokens += (eval_input_len + eval_output_len)


                # -------------------------------------------------
                # 3. State Update for Next Loop
                # -------------------------------------------------
                if attempt['result'] == "Accepted":
                    valid_step_texts.append(attempt['generated_text'])
                    # Accept 되면 루프 탈출 (다음 Step으로)
                    # last_feedback_for_prompt는 갱신하지 않아도 됨 (Step 바뀔때 초기화되므로)
                else:
                    # Reject 되면 현재 evaluation이 다음 재시도의 feedback input이 됨
                    last_feedback_for_prompt = attempt['evaluation']
        
        total_tokens_all_questions += current_question_tokens

    if total_questions == 0:
        return 0

    return total_tokens_all_questions / total_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID for tokenizer")
    args = parser.parse_args()

    # 1. Load Tokenizer
    print(f"Loading tokenizer: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    for data in ["2wiki", "hotpotqa", "musique"]:
        # 2. Load Log Data
        log_file = f"/workspace/daeyong/self_feedback_gemma12b_{data}_logs_500.json"
        print(f"Loading log file: {log_file}...")
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        # 3. Calculate
        avg_tokens = calculate_tokens_from_logs(log_data, tokenizer)

        print(f"\n==========================================")
        print(f"📂 Processed File: {log_file}")
        print(f"❓ Total Questions: {len(log_data)}")
        print(f"🔢 Average Total Tokens per Question: {avg_tokens:.2f}")
        print(f"==========================================\n")