import pandas as pd
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import gc
import torch

try:
    from hf_bnb4_generator import HFBitsAndBytesGenerator
except ModuleNotFoundError:
    HFBitsAndBytesGenerator = None

# -------------------------------------------------------------------------
# 1. Configuration & Prompts
# -------------------------------------------------------------------------

FORCE_ANSWER_SYSTEM_PROMPT = """You are an expert answering agent.
The reasoning process is complete. Your task is to formulate the FINAL ANSWER based on the provided history.

INSTRUCTIONS:
1. Do not generate any new reasoning steps.
2. Directly output the final answer.
3. YOU MUST USE THE FOLLOWING FORMAT:
####ANSWER: your_final_answer_here"""

DEFAULT_DATASETS = ("musique", "hotpotqa", "2wiki")
DEFAULT_MODELS = ("qwen4b", "qwen8b", "qwen14b", "gemma12b", "llama8b")

# -------------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------------

def parse_cli_tokens(values):
    tokens = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return list(dict.fromkeys(tokens))

def resolve_model_config(model_name):
    generator_quantization = "bnb4" if "bnb4" in model_name else "none"
    if "qwen36_27b" in model_name or "qwen3.6" in model_name:
        model_id = "/workspace/hf_transformers/Qwen3.6-27B"
    elif "qwen7b" in model_name:
        model_id = "/workspace/hf_transformers/Qwen2.5-7B-Instruct"
    elif "qwen4b" in model_name:
        model_id = "/workspace/hf_transformers/Qwen3-4B-Instruct-2507"
    elif "qwen8b" in model_name:
        model_id = "/workspace/hf_transformers/Qwen3-8B"
    elif "qwen14b" in model_name:
        model_id = "/workspace/hf_transformers/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    elif "gemma4" in model_name:
        model_id = "/workspace/hf_transformers/gemma-4-31B-it"
    elif "gemma" in model_name:
        model_id = "/workspace/hf_transformers/gemma-3-12b-it"
    elif "llama" in model_name:
        model_id = "/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct"
    elif "oss" in model_name:
        model_id = "/workspace/hf_transformers/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    else:
        raise ValueError("Model type not recognized in the given path.")
    return model_id, generator_quantization

def load_vllm_model(
    model_id: str,
    gpu_memory_utilization: float = 0.90,
    generator_quantization: str = "none",
    max_model_len: int = 3000,
):
    """Reasoning과 Self-Feedback을 모두 수행할 단일 모델 로드"""
    backend_label = "HF bitsandbytes 4bit" if generator_quantization == "bnb4" else "vLLM"
    print(f"Loading Single Model ({backend_label})... Model: '{model_id}'")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "enforce_eager": False,
        "enable_prefix_caching": True,
        "seed": 42,
    }
    if generator_quantization == "bnb4":
        if HFBitsAndBytesGenerator is None:
            raise ModuleNotFoundError(
                "hf_bnb4_generator is required for bnb4 final-answer generation"
            )
        print("Final-answer model quantization: Transformers bitsandbytes 4bit (NF4)")
        llm = HFBitsAndBytesGenerator(
            model_id,
            tokenizer,
            max_input_tokens=llm_kwargs["max_model_len"],
        )
        print("✅ Model loaded successfully.")
        return llm, tokenizer
    elif generator_quantization != "none":
        raise ValueError(f"Unsupported generator_quantization: {generator_quantization}")

    llm = LLM(**llm_kwargs)
    
    print("✅ Model loaded successfully.")
    return llm, tokenizer

def extract_final_answer(text):
    """텍스트에서 ####ANSWER: 뒤의 내용을 추출"""
    if not isinstance(text, str):
        return ""
    match = re.search(r"####ANSWER:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip().replace("(Final Answer)", "")
    return ""

def get_answer_from_response_list(resp):
    """Response 리스트의 마지막 스텝에서 정답 추출 시도"""
    if not isinstance(resp, list) or len(resp) == 0:
        return ""
    last_step = resp[-1]
    return extract_final_answer(last_step)

# -------------------------------------------------------------------------
# 3. Main Processing Logic
# -------------------------------------------------------------------------

def main(args):
    path = args.folder_path # path 예시: /workspace/daeyong/inference_results/self_feedback_Qwen2.5-7B-Instruct
    datasets = parse_cli_tokens(args.datasets) if args.datasets else list(DEFAULT_DATASETS)
    model_names = parse_cli_tokens(args.models) if args.models else list(DEFAULT_MODELS)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128) # oss는 늘려야함. 원래 128임.
    for model_name in model_names:
        model_id, generator_quantization = resolve_model_config(model_name)
        is_qwen3_model = "qwen3" in model_id.lower()
        llm = None
        tokenizer = None
        
        for dataset in datasets:
            # ours는 _results가 없음
            input_path = f"{path}/{model_name}_{dataset}.json"
            output_path = f"{path}/{model_name}_{dataset}_final_answer.json"
            
            print(f"Processing: {input_path}")
            try:
                df = pd.read_json(input_path)
            except ValueError:
                print(f"Skipping {input_path}: File not found or invalid JSON.")
                continue

            # 1차 시도: 기존 Response에서 Regex로 추출
            df["final_answer"] = df["response"].apply(get_answer_from_response_list)

            # 정답을 찾지 못한 행(row) 식별
            missing_mask = df["final_answer"] == ""
            missing_indices = df[missing_mask].index.tolist()
            
            if not missing_indices:
                print(f"All answers extracted successfully for {dataset}.")
                df.to_json(output_path, orient="records", lines=False, indent=2)
                continue

            print(f"Found {len(missing_indices)} samples missing strict answer format. Generating with vLLM...")
            if llm is None or tokenizer is None:
                llm, tokenizer = load_vllm_model(
                    model_id,
                    generator_quantization=generator_quantization,
                    max_model_len=args.max_model_len,
                )

            # vLLM 배치 처리를 위한 프롬프트 리스트 생성
            prompts = []
            
            for idx in missing_indices:
                row = df.loc[idx]
                question = row['question']
                # 리스트로 된 response를 문자열로 합침 (Step 구분)
                history_steps = row['response'] if isinstance(row['response'], list) else [str(row['response'])]
                history_str = "\n".join(history_steps)
                
                # User Prompt 구성
                user_content = f"""Question: {question}

Reasoning History:
{history_str}

The reasoning seems complete, but the final answer format is missing.
Please output ONLY the final answer now based on the history above.
Remember to use the format: ####ANSWER: final_answer_here"""

                # Chat Template 적용
                messages = [
                    {"role": "system", "content": FORCE_ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ]
                
                # Tokenizer를 사용해 모델 입력 포맷으로 변환 (tokenize=False로 string 반환)
                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if is_qwen3_model:
                    template_kwargs["enable_thinking"] = False
                full_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
                prompts.append(full_prompt)

            # vLLM Batch Inference 수행
            outputs = llm.generate(prompts, sampling_params)

            # 결과 파싱 및 DataFrame 업데이트
            for idx, output in zip(missing_indices, outputs):
                generated_text = output.outputs[0].text.split("assistantfinal")[-1].strip()
                
                # 생성된 텍스트에서 다시 ####ANSWER 추출 시도
                extracted = extract_final_answer(generated_text)
                
                # 만약 모델이 포맷을 지켰다면 추출, 아니면 생성된 텍스트 전체를 정답 후보로 저장 (혹은 후처리)
                final_val = extracted if extracted else generated_text.strip()
                
                # DataFrame 업데이트
                df.at[idx, "final_answer"] = final_val

            # 최종 저장
            df.to_json(output_path, orient="records", lines=False, indent=2)
            print(f"Saved processed results to {output_path}")

        if llm is not None:
            del llm
            del tokenizer

            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--folder_path", type=str, required=True)
    arg_parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    arg_parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    arg_parser.add_argument("--max_model_len", type=int, default=3000)
    args = arg_parser.parse_args()
    
    main(args)
