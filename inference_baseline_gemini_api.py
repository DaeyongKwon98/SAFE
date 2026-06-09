import pandas as pd
import numpy as np
import argparse
import os
import json
import ast
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from prompts import generate_single_step_system_prompt_nofeedback

@dataclass
class GeminiResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    raw_response: Any


class GeminiBaselineClient:
    """Gemini REST API-backed baseline reasoner."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "GEMINI_API_KEY",
        base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 120.0,
        max_retries: int = 5,
        retry_sleep: float = 2.0,
        concurrency: int = 8,
    ):
        api_key = os.getenv(api_key_env) if api_key_env else None
        if not api_key and api_key_env == "GEMINI_API_KEY":
            api_key = os.getenv("GOOGLE_API_KEY")
        if api_key_env and not api_key:
            raise ValueError(
                f"Gemini API key not found. Set {api_key_env} or GOOGLE_API_KEY."
            )

        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.concurrency = max(1, concurrency)

    def _endpoint(self) -> str:
        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"
        return f"{self.base_url}/{urllib.parse.quote(model_path, safe='/')}:generateContent"

    @staticmethod
    def _split_messages(messages: list[dict]) -> tuple[str, str]:
        system_parts = []
        user_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
        return "\n\n".join(system_parts), "\n\n".join(user_parts)

    def _build_payload(
        self,
        messages: list[dict],
        max_completion_tokens: int,
        temperature: Optional[float] = None,
    ) -> dict:
        system_text, user_text = self._split_messages(messages)
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": user_text}]}
            ],
            "generationConfig": {
                "maxOutputTokens": max_completion_tokens,
            },
        }
        if system_text:
            payload["system_instruction"] = {"parts": [{"text": system_text}]}
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        return payload

    @staticmethod
    def _extract_text(response_data: dict) -> str:
        candidates = response_data.get("candidates") or []
        if not candidates:
            return ""
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        return "".join(str(part.get("text", "")) for part in parts).strip()

    @staticmethod
    def _usage(response_data: dict) -> tuple[int, int, int]:
        usage = response_data.get("usageMetadata") or {}
        prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
        output_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
        total_tokens = int(usage.get("totalTokenCount", 0) or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + output_tokens
        return prompt_tokens, output_tokens, total_tokens

    def _create_completion(
        self,
        messages: list[dict],
        max_completion_tokens: int,
        temperature: Optional[float] = None,
    ) -> GeminiResult:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                payload = self._build_payload(messages, max_completion_tokens, temperature)
                request = urllib.request.Request(
                    self._endpoint(),
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                text = self._extract_text(response_data)
                prompt_tokens, output_tokens, total_tokens = self._usage(response_data)
                return GeminiResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    raw_response=response_data,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {body}")
                if exc.code in {400, 401, 403, 404} or attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep * (2 ** attempt))
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep * (2 ** attempt))

        raise RuntimeError(f"Gemini baseline call failed after retries: {last_error}") from last_error

    def generate_batch(
        self,
        messages_batch: list[list[dict]],
        max_completion_tokens: int = 256,
        temperature: Optional[float] = None,
    ) -> list[GeminiResult]:
        if not messages_batch:
            return []

        results: list[Optional[GeminiResult]] = [None] * len(messages_batch)
        max_workers = min(self.concurrency, len(messages_batch))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._create_completion, messages, max_completion_tokens, temperature): idx
                for idx, messages in enumerate(messages_batch)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()

        return [result for result in results if result is not None]


def load_gemini_model(
    model_id: str,
    api_key_env: str = "GEMINI_API_KEY",
    base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta",
    timeout: float = 120.0,
    max_retries: int = 5,
    retry_sleep: float = 2.0,
    concurrency: int = 8,
):
    print(f"Loading Single Model (Gemini API)... Model: '{model_id}'")
    client = GeminiBaselineClient(
        model=model_id,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        concurrency=concurrency,
    )
    print("✅ Gemini API model initialized successfully.")
    return client

def run_iterative_reasoning(
    df: pd.DataFrame,
    client: GeminiBaselineClient,
    result_file_path: str,
    stats_file_path: str,
    max_steps: int = 10,
    batch_size: int = 32,
    max_completion_tokens: int = 256,
    temperature: Optional[float] = None,
):
    # 1. 초기화
    pending_queue = deque()
    for _, row in df.iterrows():
        pending_queue.append(row)

    active_states = [] 
    total_stats = {"generator_calls": 0, "total_tokens": 0, "completed_count": 0}
    
    def append_to_json_file(file_path, new_data):
        if not new_data: return
        data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except: pass
        if isinstance(new_data, list): data.extend(new_data)
        else: data.append(new_data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_new_state(row):
        context_source = row['retrieved_passages']
        if isinstance(context_source, str):
            try: passages = ast.literal_eval(context_source)
            except: passages = [context_source]
        elif isinstance(context_source, list): passages = context_source
        else: passages = []
        
        return {
            "id": row['id'],
            "question": row['question'],
            "passages": passages,
            "step_texts": [],       # 생성된 Step들을 저장하는 리스트
            "finished": False,
            "ground_truth": row.get('answer', 'N/A')
        }

    pbar = tqdm(total=len(df), desc="Baseline Gemini API Reasoning Processing")

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    while pending_queue or active_states:
        
        # [Refill] 배치 크기만큼 채우기
        while len(active_states) < batch_size and pending_queue:
            active_states.append(create_new_state(pending_queue.popleft()))
        if not active_states: break

        # =========================================================================
        # SINGLE PHASE: GENERATION (Next Step)
        # =========================================================================
        gen_messages_batch = []
        gen_indices = []
        
        for idx, state in enumerate(active_states):          
            passages_str = '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(state["passages"])])
            previous_steps_str = '\n'.join(state["step_texts"]) if state["step_texts"] else "(No previous steps. Start with Step 1.)"
            
            # 다음 스텝 번호 계산
            next_step_num = len(state["step_texts"]) + 1
            
            # 프롬프트 구성
            user_content = f"""Question: {state['question']}

Retrieved Passages:
{passages_str}

Previous Reasoning Steps:
{previous_steps_str}

Generate next step (start with `Step {next_step_num}:`)."""

            messages = [
                {"role": "system", "content": generate_single_step_system_prompt_nofeedback}, 
                {"role": "user", "content": user_content}
            ]

            gen_messages_batch.append(messages)
            gen_indices.append(idx)

        # Gemini API Batch Inference
        if gen_messages_batch:
            gen_outputs = client.generate_batch(
                gen_messages_batch,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
            total_stats["generator_calls"] += len(gen_outputs)

            for i, output in enumerate(gen_outputs):
                state_idx = gen_indices[i]
                state = active_states[state_idx]
                
                gen_text = output.text.strip()
                total_stats["total_tokens"] += output.total_tokens

                # 메타데이터 토큰 제거 (Cleaning)
                for marker in ["<start_of_turn>", "User:", "## Question"]:
                    if marker in gen_text: gen_text = gen_text.split(marker)[0].strip()
                
                # 결과 저장
                state["step_texts"].append(gen_text)
                
                # 종료 태그 발견
                if "END_OF_REASONING" in gen_text:
                    state["finished"] = True

        # =========================================================================
        # CLEANUP & SAVE
        # =========================================================================
        next_active_states = []
        finished_results = []

        for state in active_states:
            # 종료 조건 체크 2: 최대 스텝 도달
            if not state["finished"] and len(state["step_texts"]) >= max_steps:
                state["finished"] = True

            if state["finished"]:
                res_obj = {
                    "id": state["id"],
                    "question": state["question"],
                    "passages": state['passages'],
                    "generated_steps": state["step_texts"],
                    "ground_truth": state["ground_truth"]
                }
                finished_results.append(res_obj)
                pbar.update(1)
            else:
                next_active_states.append(state)

        if finished_results:
            append_to_json_file(result_file_path, finished_results)
            total_stats["completed_count"] += len(finished_results)
            append_to_json_file(stats_file_path, [deepcopy(total_stats)])

        active_states = next_active_states

    pbar.close()
    return total_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--gemini_api_key_env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--gemini_base_url", type=str, default="https://generativelanguage.googleapis.com/v1beta")
    parser.add_argument("--gemini_timeout", type=float, default=120.0)
    parser.add_argument("--gemini_max_retries", type=int, default=5)
    parser.add_argument("--gemini_retry_sleep", type=float, default=2.0)
    parser.add_argument("--gemini_concurrency", type=int, default=8)
    parser.add_argument("--gemini_max_completion_tokens", type=int, default=256)
    parser.add_argument("--gemini_temperature", type=float, default=None)
    args = parser.parse_args()

    base_path = "/workspace/daeyong"
    
    try:
        # 데이터셋 로드
        df = pd.read_csv(f"{base_path}/benchmarks/{args.dataset}_dev_kg_correct.csv")
        if args.sample_size > 0:
            df = df.sample(n=min(args.sample_size, len(df)), random_state=args.sample_seed)[:100] # 임시로 100개!
        print(f"Loaded {len(df)} samples from {args.dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    model_name_clean = re.sub(r"[^A-Za-z0-9]+", "_", args.model_id.split("/")[-1]).strip("_").lower()
    output_dir = f"{base_path}/inference_results/no_feedback_{model_name_clean}"
    os.makedirs(output_dir, exist_ok=True)
    
    result_file_path = os.path.join(output_dir, f"{args.dataset}_results.json")
    stats_file_path = os.path.join(output_dir, f"{args.dataset}_stats.json")

    # Resume Logic
    processed_ids = set()
    if os.path.exists(result_file_path):
        try:
            with open(result_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                processed_ids = {item['id'] for item in data}
            print(f"Resuming... {len(processed_ids)} already processed.")
        except: pass
    
    df = df[~df['id'].isin(processed_ids)]

    if len(df) > 0:
        client = load_gemini_model(
            args.model_id,
            api_key_env=args.gemini_api_key_env,
            base_url=args.gemini_base_url,
            timeout=args.gemini_timeout,
            max_retries=args.gemini_max_retries,
            retry_sleep=args.gemini_retry_sleep,
            concurrency=args.gemini_concurrency,
        )

        run_iterative_reasoning(
            df=df,
            client=client,
            result_file_path=result_file_path,
            stats_file_path=stats_file_path,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            max_completion_tokens=args.gemini_max_completion_tokens,
            temperature=args.gemini_temperature,
        )
    else:
        print("All data processed.")
