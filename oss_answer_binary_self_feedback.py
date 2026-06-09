import argparse
import ast
import json
import os
import re
from typing import Tuple

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --------------------------
# 1. 설정 및 모델 로드
# --------------------------
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"


def load_judge_model() -> Tuple[LLM, AutoTokenizer, SamplingParams]:
    print(f"Loading vLLM model for Judging: {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        max_model_len=5000,
        dtype="bfloat16",
        enable_prefix_caching=True,
        seed=42
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4000,
        stop=None,
    )
    return llm, tokenizer, sampling_params


# --------------------------
# 2. System Prompt
# --------------------------
system_prompt = """You are an expert evaluator for a Question Answering task.
Your goal is to determine if the 'Generated Answer' is correct based on the 'Ground Truth List'.

**Evaluation Criteria:**
1. **Semantic Equivalence:** If the 'Generated Answer' refers to the same real-world entity, concept, or event as **ANY** item in the 'Ground Truth List', it is "correct". This includes aliases, abbreviations, and common synonyms.
2. **Granularity & Hierarchy:** Accept answers that are factually accurate but differ in specificity or granularity, provided they refer to the same location or entity.
    - **Geographic Inclusion:** Accept constituent countries, states, or specific locations if they are part of the broader Ground Truth entity (e.g., "England" is correct for "United Kingdom"; "New York" is correct for "USA" if the context implies origin).
    - **Specificity:** Accept broader correct terms if they encompass the specific Ground Truth (e.g., "UK" is correct for "England" if the question asks for country).
3. **Logical Entailment & Event Description:** Accept answers that describe the same event or fact using different but factually compatible attributes.
    - **Cause vs. Nature:** If Ground Truth specifies the mechanism (e.g., "shot") and Generated Answer specifies the nature of the event (e.g., "homicide" or "murder"), and both describe the same factual occurrence, it is "correct".
    - **Implication:** If the Generated Answer logically implies the Ground Truth or vice versa in the given context (e.g., "shot by father" implies "killed by family member"), it is "correct".
4. **Robustness:** Ignore minor formatting, casing, punctuation, or conversational fillers (e.g., "The answer is...").
5. **Contextual Correctness:** If the answer is factually different, references a completely distinct entity, or introduces contradictory information compared to the ground truth (e.g., "died of old age" vs "shot"), it is "wrong".

**Output Format:**
You must output ONLY a valid JSON object with exactly two keys: "is_correct" and "reasoning". Do not include any markdown styling or extra text.
- "is_correct": Must be either "correct" or "wrong".
- "reasoning": A brief, high-density explanation of why the answer was judged this way, specifically mentioning any logical entailment or hierarchy logic if applicable.

Example 1:
{"is_correct": "correct", "reasoning": "The generated answer 'England' is a constituent country of the ground truth 'United Kingdom'. Both refer to the correct origin in this context."}

Example 2:
{"is_correct": "correct", "reasoning": "The generated answer 'Steve Jobs' matches the ground truth 'Steven Paul Jobs' as a valid alias."}

Example 3:
{"is_correct": "correct", "reasoning": "Ground Truth is 'shot' (mechanism) and Generated Answer is 'homicide by family member' (nature of event). Since being shot by one's father is a form of homicide by a family member, they describe the same factual event."}
""".strip()


# --------------------------
# 3. 유틸리티 함수
# --------------------------
def parse_llm_output(generated_text: str) -> Tuple[str, str]:
    """LLM 출력을 JSON/dict로 robust하게 파싱합니다."""

    def strip_json_prefix(s: str) -> str:
        return re.sub(r"^\s*json\s*[:=-]?\s*", "", s, flags=re.IGNORECASE).strip()

    def extract_balanced_objects(s: str):
        objects = []
        depth = 0
        start = None
        in_string = False
        escape = False
        quote_char = ""

        for i, ch in enumerate(s):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
                continue

            if ch in ("'", '"'):
                in_string = True
                quote_char = ch
                continue

            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(s[start : i + 1])
                    start = None
        return objects

    def try_parse_obj(candidate: str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "is_correct" in item:
                            return item
            except Exception:
                continue
        return None

    def normalize_is_correct(value):
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"correct", "true", "1", "yes", "y"}:
                return "correct"
            if s in {"wrong", "incorrect", "false", "0", "no", "n"}:
                return "wrong"
            return "error"
        if isinstance(value, bool):
            return "correct" if value else "wrong"
        if isinstance(value, (int, float)):
            if value == 1:
                return "correct"
            if value == 0:
                return "wrong"
        return "error"

    text = str(generated_text or "").strip()
    candidates = []

    def add_candidate(value: str):
        value = str(value or "").strip()
        if value and value not in candidates:
            candidates.append(value)

    add_candidate(text)
    add_candidate(strip_json_prefix(text))

    fence_matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in fence_matches:
        add_candidate(block)
        add_candidate(strip_json_prefix(block))

    for c in list(candidates):
        for obj in extract_balanced_objects(c):
            add_candidate(obj)
            add_candidate(strip_json_prefix(obj))

    for candidate in candidates:
        parsed = try_parse_obj(candidate)
        if isinstance(parsed, dict):
            is_correct = normalize_is_correct(parsed.get("is_correct"))
            if is_correct == "error":
                is_correct = normalize_is_correct(parsed.get("label"))
            if is_correct == "error":
                is_correct = normalize_is_correct(parsed.get("result"))

            reasoning = parsed.get("reasoning")
            if reasoning is None:
                reasoning = parsed.get("explanation")
            if reasoning is None:
                reasoning = parsed.get("message")
            if reasoning is None:
                reasoning = "Parsing successful but keys missing."
            return is_correct, str(reasoning)

    low = text.lower()
    if "incorrect" in low:
        is_correct = "wrong"
    elif "wrong" in low and "correct" not in low:
        is_correct = "wrong"
    elif "correct" in low and "wrong" not in low:
        is_correct = "correct"
    else:
        is_correct = "error"
    return is_correct, f"JSON Parsing Failed: {text[:100]}"


def get_generated_answer(row: pd.Series) -> str:
    """self-feedback final answer 결과 스키마 우선순위에 맞춰 정답 후보를 추출합니다."""
    for key in ["final_answer_extracted", "final_answer"]:
        val = row.get(key, "")
        if pd.isna(val):
            continue
        text = str(val).strip()
        if text:
            return text
    return ""


# --------------------------
# 4. 메인 실행 루프
# --------------------------
def main(args):
    datasets = args.datasets or ["2wiki", "hotpotqa", "musique"]
    models = args.models or ["gemma12b", "llama8b", "qwen14b", "qwen4b", "qwen8b"]

    llm, tokenizer, sampling_params = load_judge_model()

    for dataset in datasets:
        for model_name in models:
            input_path = f"{args.folder_path}/{model_name}_{dataset}_final_answer.json"
            output_path = f"{args.folder_path}/{model_name}_{dataset}_llm_judge.json"

            if not os.path.exists(input_path):
                print(f"Skipping: {input_path} (File not found)")
                continue

            print(f"\n🚀 Processing: {dataset} - {model_name}")
            try:
                df = pd.read_json(input_path)
            except Exception as e:
                print(f"Skipping {input_path}: invalid JSON or read error ({e})")
                continue

            if df.empty:
                print("Empty dataframe. Skipping.")
                continue

            prompts = []
            empty_flags = []
            generated_answers = []

            for _, row in df.iterrows():
                if dataset == "musique":
                    gt_list = row.get("answer_list_norm", [row.get("ground_truth", "")])
                    if not isinstance(gt_list, list):
                        gt_list = [row.get("ground_truth", "")]
                else:
                    gt_list = [row.get("ground_truth", "")]

                generated_answer = get_generated_answer(row)
                generated_answers.append(generated_answer)
                empty_flags.append(generated_answer == "")

                user_content = (
                    f"### Input Data\n"
                    f"**Question**: {row.get('question', '')}\n"
                    f"**Ground Truth List**: {gt_list}\n"
                    f"**Generated Answer**: {generated_answer}\n\n"
                    f"### Task\n"
                    f"Is the generated answer correct based on the ground truth list?"
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]

                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)

            print(f"Generating {len(prompts)} judgments...")
            outputs = llm.generate(prompts, sampling_params)

            is_correct_list = []
            reasoning_list = []

            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text.split("assistantfinal")[-1].strip()
                is_correct, reasoning = parse_llm_output(generated_text)

                if empty_flags[i]:
                    reasoning = f"[empty_generated_answer] {reasoning}"

                is_correct_list.append(is_correct)
                reasoning_list.append(reasoning)

            df["generated_answer_used"] = generated_answers
            df["is_correct"] = is_correct_list
            df["reasoning"] = reasoning_list

            df.to_json(output_path, orient="records", force_ascii=False, indent=2)
            print(f"✅ Saved results to: {output_path}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--folder_path", type=str, required=True)
    argparser.add_argument("--datasets", nargs="+", default=None)
    argparser.add_argument("--models", nargs="+", default=None)
    args = argparser.parse_args()
    main(args)
