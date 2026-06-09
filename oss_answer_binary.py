import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --------------------------
# 1. 설정 및 모델 로드
# --------------------------
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
DEFAULT_DATASETS = ("musique", "2wiki", "hotpotqa")
DEFAULT_MODELS = ("qwen4b", "qwen8b", "qwen14b", "gemma12b", "llama8b")


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
def parse_cli_tokens(values: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return list(dict.fromkeys(tokens))



def load_judge_model(max_model_len: int = 2000) -> Tuple[LLM, AutoTokenizer, SamplingParams]:
    print(f"Loading vLLM model for Judging: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
        seed=42,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1000,
        stop=None,
    )
    return llm, tokenizer, sampling_params



def parse_llm_output(generated_text: str) -> Tuple[str, str]:
    """LLM 출력을 JSON으로 파싱하고 에러 처리를 수행합니다."""
    generated_text = str(generated_text).strip()
    try:
        if generated_text.startswith("```json"):
            generated_text = generated_text[7:]
        if generated_text.endswith("```"):
            generated_text = generated_text[:-3]

        result = json.loads(generated_text.strip())
        return result.get("is_correct", "error"), result.get("reasoning", "Parsing successful but keys missing.")
    except Exception:
        is_correct = "error"
        lowered = generated_text.lower()
        if "correct" in lowered and "wrong" not in lowered:
            is_correct = "correct"
        elif "wrong" in lowered:
            is_correct = "wrong"
        return is_correct, f"JSON Parsing Failed: {generated_text[:100]}"



def load_json_records(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {file_path}")
    return data



def save_json_records(file_path: str, rows: List[Dict]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)



def build_ground_truth_list(row: Dict, dataset: str) -> List:
    if dataset == "musique":
        gt_list = row.get("answer_list_norm")
        if isinstance(gt_list, list) and gt_list:
            return gt_list
    return [row.get("ground_truth")]



def build_prompts(rows: Sequence[Dict], dataset: str, tokenizer) -> List[str]:
    prompts: List[str] = []
    for row in rows:
        gt_list = build_ground_truth_list(row, dataset)
        user_content = (
            f"### Input Data\n"
            f"**Question**: {row.get('question', '')}\n"
            f"**Ground Truth List**: {gt_list}\n"
            f"**Generated Answer**: {row.get('final_answer', '')}\n\n"
            f"### Task\n"
            f"Is the generated answer correct based on the ground truth list?"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts



def judge_rows(
    rows: Sequence[Dict],
    dataset: str,
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
) -> Dict[str, Dict]:
    if not rows:
        return {}

    prompts = build_prompts(rows, dataset, tokenizer)
    print(f"Generating {len(prompts)} judgments...")
    outputs = llm.generate(prompts, sampling_params)

    judged_by_id: Dict[str, Dict] = {}
    for row, output in zip(rows, outputs):
        generated_text = output.outputs[0].text.split("assistantfinal")[-1].strip()
        is_correct, reasoning = parse_llm_output(generated_text)

        merged_row = dict(row)
        merged_row["is_correct"] = is_correct
        merged_row["reasoning"] = reasoning
        judged_by_id[str(row.get("id"))] = merged_row

    return judged_by_id



def select_one_char_ids(existing_rows: Sequence[Dict]) -> List[str]:
    ids: List[str] = []
    for row in existing_rows:
        final_answer = str(row.get("final_answer", "") if row.get("final_answer", "") is not None else "").strip()
        if len(final_answer) == 1:
            ids.append(str(row.get("id")))
    return list(dict.fromkeys(ids))



def run_full_regeneration(
    folder_path: str,
    datasets: Sequence[str],
    models: Sequence[str],
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
) -> None:
    for dataset in datasets:
        for model_name in models:
            input_path = f"{folder_path}/{model_name}_{dataset}_final_answer.json"
            output_path = f"{folder_path}/{model_name}_{dataset}_llm_judge.json"

            if not os.path.exists(input_path):
                print(f"Skipping: {input_path} (File not found)")
                continue

            print(f"\n🚀 Processing: {dataset} - {model_name}")
            rows = load_json_records(input_path)
            if not rows:
                print("Empty input rows. Skipping.")
                continue

            judged_by_id = judge_rows(rows, dataset, llm, tokenizer, sampling_params)
            judged_rows = [judged_by_id[str(row.get('id'))] for row in rows]
            save_json_records(output_path, judged_rows)
            print(f"✅ Saved results to: {output_path}")



def run_selective_one_char_update(
    folder_path: str,
    datasets: Sequence[str],
    models: Sequence[str],
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
) -> None:
    for dataset in datasets:
        for model_name in models:
            input_path = f"{folder_path}/{model_name}_{dataset}_final_answer.json"
            output_path = f"{folder_path}/{model_name}_{dataset}_llm_judge.json"

            if not os.path.exists(input_path):
                print(f"Skipping: {input_path} (final_answer file not found)")
                continue
            if not os.path.exists(output_path):
                print(f"Skipping: {output_path} (existing llm_judge file required for selective update)")
                continue

            print(f"\n🔄 Selective update: {dataset} - {model_name}")
            input_rows = load_json_records(input_path)
            existing_rows = load_json_records(output_path)

            input_by_id = {str(row.get('id')): row for row in input_rows}
            target_ids = select_one_char_ids(existing_rows)

            if not target_ids:
                print("No one-character final_answer rows found in existing llm_judge. Keeping file as-is.")
                continue

            rerun_rows: List[Dict] = []
            missing_input_ids: List[str] = []
            for row_id in target_ids:
                if row_id in input_by_id:
                    rerun_rows.append(input_by_id[row_id])
                else:
                    missing_input_ids.append(row_id)

            if missing_input_ids:
                print(
                    f"⚠️ {len(missing_input_ids)} ids were selected from existing llm_judge but missing in final_answer input. "
                    "They will be left unchanged."
                )

            if not rerun_rows:
                print("No valid rerun rows found after matching ids. Keeping file as-is.")
                continue

            judged_by_id = judge_rows(rerun_rows, dataset, llm, tokenizer, sampling_params)

            merged_rows: List[Dict] = []
            updated_count = 0
            for existing_row in existing_rows:
                row_id = str(existing_row.get('id'))
                if row_id in judged_by_id:
                    merged_row = dict(existing_row)
                    merged_row.update(input_by_id[row_id])
                    merged_row['is_correct'] = judged_by_id[row_id]['is_correct']
                    merged_row['reasoning'] = judged_by_id[row_id]['reasoning']
                    merged_rows.append(merged_row)
                    updated_count += 1
                else:
                    merged_rows.append(existing_row)

            save_json_records(output_path, merged_rows)
            print(
                f"✅ Updated {updated_count} one-character rows in: {output_path} "
                f"(unchanged rows preserved: {len(existing_rows) - updated_count})"
            )


# --------------------------
# 4. 메인 실행 루프
# --------------------------
def main(args) -> None:
    datasets = parse_cli_tokens(args.datasets) if args.datasets else list(DEFAULT_DATASETS)
    models = parse_cli_tokens(args.models) if args.models else list(DEFAULT_MODELS)

    llm, tokenizer, sampling_params = load_judge_model(max_model_len=args.max_model_len)

    if args.only_existing_one_char_final_answer:
        run_selective_one_char_update(
            folder_path=args.folder_path,
            datasets=datasets,
            models=models,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
        )
    else:
        run_full_regeneration(
            folder_path=args.folder_path,
            datasets=datasets,
            models=models,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--folder_path", type=str, required=True)
    argparser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    argparser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    argparser.add_argument("--max_model_len", type=int, default=2000)
    argparser.add_argument(
        "--only-existing-one-char-final-answer",
        action="store_true",
        help=(
            "Read the existing *_llm_judge.json, select ids whose stored final_answer has length 1, "
            "rejudge only those ids using the current *_final_answer.json, and merge the updates back."
        ),
    )
    args = argparser.parse_args()
    main(args)
