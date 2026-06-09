import pandas as pd
import numpy as np
import argparse
import os
import json
import ast
import re
import time
from tqdm import tqdm
from copy import deepcopy
from collections import deque
from typing import Any, Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from prompts import (
    evaluate_system_prompt_premature_attribution,
)

generate_single_step_only_diagnosis = """
You are a meticulous, step-by-step logical reasoner. Your task is to solve a complex question by generating **ONLY THE NEXT SINGLE, ATOMIC STEP** in a chain of thought.

## CRITICAL: FEEDBACK COMPLIANCE PROTOCOL
**You must analyze the `Feedback` received on the previous step with the highest priority.**
Feedback consists of:
- **Diagnosis**: An explanation about the last step.

**Discard Priors**: Do not repeat the same error. Discard your previous assumption and look at the `Retrieved Passages` afresh.

---

## Step Classifications
Every step must be strictly classified into one of three types.
Attribution, Logical, and Final Answer actions cannot be mixed in a single step.

### 1. Attribution Step
- **Definition**: Extracts **ONE** explicit fact from a **SINGLE** retrieved passage.
- **Requirement**: You MUST explicitly cite the source (e.g., "According to Passage X...").
- **Constraint 1 (Anti-Hallucination)**: Verify the **Subject** (who/what) and the **Attribute/Relation** (is what/did what) exactly against the text.
    - **Do NOT** misattribute a date, location, action, or property to the wrong subject simply because of text proximity.
- **Constraint 2**: Do **NOT** combine information from multiple passages.
- **Constraint 3**: You must follow the strict logical chain starting from the `Question`.
    - **Rule**: You are ONLY allowed to extract facts or an entity if it is **explicitly mentioned in the `Question`** OR **explicitly discovered in `Previous Reasoning Steps`**.
    - **Prohibition**: Do **NOT** jump to an intermediate entity unless a previous step has already established its connection to the starting entity.
- **Format suffix**: End the sentence with `(Attribution)`.

### 2. Logical Step
- **Definition**: Performs **ONE** logical operation (comparison, calculation, or inference) based **ONLY** on `Previous Reasoning Steps`.
- **Requirement**: You must NOT look up any new information from passages.
- **Constraint**: This step is for **Intermediate Reasoning** only.
    - You must NOT output the final answer marker here.
- **Format suffix**: End the sentence with `(Logical)`.

### 3. Final Answer Step
- **Definition**: Submits the final answer.
- **Strict Syntax Rule**: `Step K: ####ANSWER: <answer_value> (Final Answer)`
- **Constraint (Type Check)**: Verify that the `<answer_value>` strictly matches the **Entity Type** or **Format** requested by the original `Question`.
    - If the question asks "Who", the answer MUST be a **Name** (Person/Org).
    - If the question asks "When", the answer MUST be a **Date/Time**.
    - If the question asks "How many", the answer MUST be a **Number**.
    - **Do NOT** output a full sentence unless explicitly asked. Just provide the specific value.

---

## Strict Formatting Rules
1. **Numbering**: Start your response with `Step K:`, where `K` is the next integer after the last step number.
2. **Atomic Nature**: Adhere strictly to the "One Step = One Action" rule.
3. **Suffix Mandatory**: End every step with `(Attribution)`, `(Logical)`, or `(Final Answer)`.

---

## Examples of Valid Atomic Steps

### 1. Attribution Step Examples
- **Correct**:
  Step K: According to Passage 3, the director of the film "Inception" is Christopher Nolan. (Attribution)

- **Incorrect (Mixed Types: Attribution + Logical)**: 
  Step K: According to Passage 2, the singer X born in 1977. So he is older than singer Y. (Attribution)
  -> **WRONG!** "So he is older than ..." is a logical inference. **Stop after "... born in 1977".**

- **Incorrect (No Citation)**: 
  Step K: According to the provided passages, the film was released in 2010. (Attribution)
  -> **WRONG!** You must explicitly state "According to Passage X".

### 2. Logical Step Examples
- **Correct**:
  Step K: Comparing the date in Step 1 (1918) and Step 2 (1939), the start of World War II was later than the end of World War I. (Logical)

- **Incorrect (New Fact Lookup)**:
  Step K: Since Step 1 mentions "Titanic", and Passage 2 says it won 11 Oscars, it is successful. (Logical)
  -> **WRONG!** Do not cite Passage 2 in a Logical step. Make a separate Attribution step first.

### 3. Final Answer Step Examples
- **Correct**:
  Step K: ####ANSWER: Paris (Final Answer)
""".strip()


generate_single_step_only_guidance = """
You are a meticulous, step-by-step logical reasoner. Your task is to solve a complex question by generating **ONLY THE NEXT SINGLE, ATOMIC STEP** in a chain of thought.

## CRITICAL: FEEDBACK COMPLIANCE PROTOCOL
**You must analyze the `Feedback` received on the previous step with the highest priority.**
Feedback consists of:
- **Guidance**: Specific instruction on what you should do in this current step. (e.g. Fixing previous step's error, Proceeding to extract new information, or Making the final conclusion).

1. **Read the Guidance**: Treat the `Guidance` field as a **MANDATORY COMMAND**.
    - If Guidance says "Use Passage X", you **MUST** start your step with "According to Passage X...".
    - If Guidance says "The entity was wrong", you **MUST** re-read the passage and select the correct entity.
2. **Discard Priors**: Do not repeat the same error. Discard your previous assumption and look at the `Retrieved Passages` afresh.
3. **Strict Adherence**: Any step that ignores the specific instruction in `Guidance` will be considered a failure.

---

## Step Classifications
Every step must be strictly classified into one of three types.
Attribution, Logical, and Final Answer actions cannot be mixed in a single step.

### 1. Attribution Step
- **Definition**: Extracts **ONE** explicit fact from a **SINGLE** retrieved passage.
- **Requirement**: You MUST explicitly cite the source (e.g., "According to Passage X...").
- **Constraint 1 (Anti-Hallucination)**: Verify the **Subject** (who/what) and the **Attribute/Relation** (is what/did what) exactly against the text.
    - **Do NOT** misattribute a date, location, action, or property to the wrong subject simply because of text proximity.
- **Constraint 2**: Do **NOT** combine information from multiple passages.
- **Constraint 3**: You must follow the strict logical chain starting from the `Question`.
    - **Rule**: You are ONLY allowed to extract facts or an entity if it is **explicitly mentioned in the `Question`** OR **explicitly discovered in `Previous Reasoning Steps`**.
    - **Prohibition**: Do **NOT** jump to an intermediate entity unless a previous step has already established its connection to the starting entity.
- **Format suffix**: End the sentence with `(Attribution)`.

### 2. Logical Step
- **Definition**: Performs **ONE** logical operation (comparison, calculation, or inference) based **ONLY** on `Previous Reasoning Steps`.
- **Requirement**: You must NOT look up any new information from passages.
- **Constraint**: This step is for **Intermediate Reasoning** only.
    - You must NOT output the final answer marker here.
- **Format suffix**: End the sentence with `(Logical)`.

### 3. Final Answer Step
- **Definition**: Submits the final answer.
- **Trigger Condition**: You **MUST** generate this step if the previous `Guidance` instructs you to submit the final answer.
- **Strict Syntax Rule**: `Step K: ####ANSWER: <answer_value> (Final Answer)`
- **Constraint (Type Check)**: Verify that the `<answer_value>` strictly matches the **Entity Type** or **Format** requested by the original `Question`.
    - If the question asks "Who", the answer MUST be a **Name** (Person/Org).
    - If the question asks "When", the answer MUST be a **Date/Time**.
    - If the question asks "How many", the answer MUST be a **Number**.
    - **Do NOT** output a full sentence unless explicitly asked. Just provide the specific value.

---

## Strict Formatting Rules
1. **Numbering**: Start your response with `Step K:`, where `K` is the next integer after the last step number.
2. **Atomic Nature**: Adhere strictly to the "One Step = One Action" rule.
3. **Suffix Mandatory**: End every step with `(Attribution)`, `(Logical)`, or `(Final Answer)`.

---

## Examples of Valid Atomic Steps

### 1. Attribution Step Examples
- **Correct**:
  Step K: According to Passage 3, the director of the film "Inception" is Christopher Nolan. (Attribution)

- **Incorrect (Mixed Types: Attribution + Logical)**: 
  Step K: According to Passage 2, the singer X born in 1977. So he is older than singer Y. (Attribution)
  -> **WRONG!** "So he is older than ..." is a logical inference. **Stop after "... born in 1977".**

- **Incorrect (No Citation)**: 
  Step K: According to the provided passages, the film was released in 2010. (Attribution)
  -> **WRONG!** You must explicitly state "According to Passage X".

### 2. Logical Step Examples
- **Correct**:
  Step K: Comparing the date in Step 1 (1918) and Step 2 (1939), the start of World War II was later than the end of World War I. (Logical)

- **Incorrect (New Fact Lookup)**:
  Step K: Since Step 1 mentions "Titanic", and Passage 2 says it won 11 Oscars, it is successful. (Logical)
  -> **WRONG!** Do not cite Passage 2 in a Logical step. Make a separate Attribution step first.

### 3. Final Answer Step Examples
- **Correct**:
  Step K: ####ANSWER: Paris (Final Answer)
""".strip()


FORCE_ANSWER_SYSTEM_PROMPT = """You are an expert answering agent.
The reasoning process is complete. Your task is to formulate the FINAL ANSWER based on the provided history.

INSTRUCTIONS:
1. Do not generate any new reasoning steps.
2. Directly output the final answer.
3. YOU MUST USE THE FOLLOWING FORMAT:
####ANSWER: your_final_answer_here (Final Answer)"""


GENERATION_FEEDBACK_MODES = {"diagnosis_only", "guidance_only"}


def select_generation_system_prompt(generation_feedback_mode: str) -> str:
    if generation_feedback_mode == "diagnosis_only":
        return generate_single_step_only_diagnosis
    if generation_feedback_mode == "guidance_only":
        return generate_single_step_only_guidance
    raise ValueError(
        "generation_feedback_mode must be one of "
        f"{sorted(GENERATION_FEEDBACK_MODES)}, got: {generation_feedback_mode}"
    )


def resolve_adapter_path(feedback_model_arg: str) -> str:
    if os.path.isabs(feedback_model_arg):
        return feedback_model_arg
    return f"/workspace/daeyong/trained_models/{feedback_model_arg}"


def strip_think_blocks(text: str) -> str:
    """Remove Qwen thinking traces so downstream parsing uses answer content only."""
    if not text:
        return text

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Handle truncated generations where <think> starts but </think> is missing.
    dangling_start = cleaned.lower().find("<think>")
    if dangling_start != -1:
        cleaned = cleaned[:dangling_start]

    cleaned = cleaned.replace("</think>", "")
    return cleaned.strip()


def build_generator_sampling_params(
    is_qwen3_8b: bool,
    qwen3_thinking_mode: str,
    stop_token_ids: list[int],
) -> SamplingParams:
    if is_qwen3_8b:
        if qwen3_thinking_mode == "on":
            return SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                max_tokens=1024,
                stop_token_ids=stop_token_ids,
            )
        # 일단 deterministic으로 먼저 해보기
        # return SamplingParams(
        #     temperature=0.7,
        #     top_p=0.8,
        #     top_k=20,
        #     min_p=0.0,
        #     max_tokens=256,
        #     stop_token_ids=stop_token_ids
        # )

    return SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop_token_ids=stop_token_ids,
    )


class CacheTracker:
    """Tracks cache-related token statistics without affecting inference behavior."""

    def __init__(
        self,
        enabled: bool,
        mode: str,
        trace_file_path: Optional[str],
        stats_file_path: Optional[str],
        gen_tokenizer,
        eval_tokenizer,
        evaluator_system_prompt: str,
        generation_system_prompt: str,
    ):
        self.enabled = enabled
        self.mode = mode
        self.trace_file_path = trace_file_path
        self.stats_file_path = stats_file_path
        self.gen_tokenizer = gen_tokenizer
        self.eval_tokenizer = eval_tokenizer
        self.evaluator_system_prompt = evaluator_system_prompt
        self.generation_system_prompt = generation_system_prompt
        self.cached_le_prompt_all_calls = True
        self.record_errors = 0

        self.role_global_call_index = {"generator": 0, "evaluator": 0}
        self.sample_role_call_count = {}
        self.qp_token_cache = {}

        self.role_stats = {
            "generator": {
                "calls": 0,
                "prompt_tokens_total": 0,
                "output_tokens_total": 0,
                "cached_tokens_total": 0,
                "used_input_tokens_total": 0,
                "exact_calls": 0,
                "fallback_calls": 0,
            },
            "evaluator": {
                "calls": 0,
                "prompt_tokens_total": 0,
                "output_tokens_total": 0,
                "cached_tokens_total": 0,
                "used_input_tokens_total": 0,
                "exact_calls": 0,
                "fallback_calls": 0,
            },
        }

        if not self.enabled:
            self.system_tokens = {"generator": 0, "evaluator": 0}
            return

        # System prompt token lengths for fallback mode assumptions.
        self.system_tokens = {
            "generator": len(
                self.gen_tokenizer.encode(
                    self.generation_system_prompt,
                    add_special_tokens=False,
                )
            ),
            "evaluator": len(
                self.eval_tokenizer.encode(
                    self.evaluator_system_prompt,
                    add_special_tokens=False,
                )
            ),
        }

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    def _extract_cached_from_obj(self, obj: Any) -> Optional[int]:
        if obj is None:
            return None

        candidate_keys = [
            "num_cached_tokens",
            "cached_tokens",
            "prefix_cached_tokens",
            "prompt_cached_tokens",
        ]

        # dict-like
        if isinstance(obj, dict):
            for k in candidate_keys:
                if k in obj:
                    val = self._safe_int(obj.get(k))
                    if val is not None:
                        return val
            for v in obj.values():
                val = self._extract_cached_from_obj(v)
                if val is not None:
                    return val
            return None

        # object attributes
        for k in candidate_keys:
            if hasattr(obj, k):
                val = self._safe_int(getattr(obj, k))
                if val is not None:
                    return val

        if hasattr(obj, "__dict__"):
            return self._extract_cached_from_obj(vars(obj))
        return None

    def _extract_exact_cached_tokens(self, output_obj: Any) -> Optional[int]:
        # Primary source (vLLM RequestOutput field)
        if hasattr(output_obj, "num_cached_tokens"):
            val = self._safe_int(getattr(output_obj, "num_cached_tokens"))
            if val is not None:
                return val

        # Secondary source (metrics object)
        metrics = getattr(output_obj, "metrics", None)
        return self._extract_cached_from_obj(metrics)

    def _get_qp_tokens(self, role: str, state: dict) -> int:
        cache_key = (role, state.get("id"))
        if cache_key in self.qp_token_cache:
            return self.qp_token_cache[cache_key]

        passages = state.get("passages", [])
        passages_str = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)])
        qp_segment = (
            f"Question:\n{state.get('question', '')}\n\n"
            f"Retrieved Passages:\n{passages_str}\n\n"
        )

        tokenizer = self.gen_tokenizer if role == "generator" else self.eval_tokenizer
        qp_tokens = len(tokenizer.encode(qp_segment, add_special_tokens=False))
        self.qp_token_cache[cache_key] = qp_tokens
        return qp_tokens

    def _append_trace(self, payload: dict):
        if not self.enabled or not self.trace_file_path:
            return
        with open(self.trace_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def record_call(
        self,
        role: str,
        state: dict,
        output_obj: Any,
        prompt_tokens: int,
        output_tokens: int,
        step_num: int,
        retry_index: int,
    ):
        if not self.enabled:
            return

        global_idx = self.role_global_call_index[role]
        sample_key = (state.get("id"), role)
        sample_role_idx = self.sample_role_call_count.get(sample_key, 0)

        exact_cached = self._extract_exact_cached_tokens(output_obj)
        fallback_components = None

        if exact_cached is not None:
            cached_tokens = max(0, min(prompt_tokens, int(exact_cached)))
            cache_source = "exact_num_cached_tokens"
            self.role_stats[role]["exact_calls"] += 1
        else:
            if self.mode == "exact_only":
                raise RuntimeError(
                    f"cache_stats_mode=exact_only but exact cached tokens are unavailable "
                    f"(role={role}, sample_id={state.get('id')}, step={step_num}, retry={retry_index})."
                )

            sys_cached = self.system_tokens[role] if global_idx > 0 else 0
            qp_cached = self._get_qp_tokens(role, state) if sample_role_idx > 0 else 0
            cached_tokens = min(prompt_tokens, sys_cached + qp_cached)
            cache_source = "fallback_sys_plus_qp"
            fallback_components = {
                "system_tokens_assumed": sys_cached,
                "qp_tokens_assumed": qp_cached,
            }
            self.role_stats[role]["fallback_calls"] += 1

        if cached_tokens > prompt_tokens:
            self.cached_le_prompt_all_calls = False
            cached_tokens = prompt_tokens

        used_input_tokens = prompt_tokens - cached_tokens

        role_stat = self.role_stats[role]
        role_stat["calls"] += 1
        role_stat["prompt_tokens_total"] += prompt_tokens
        role_stat["output_tokens_total"] += output_tokens
        role_stat["cached_tokens_total"] += cached_tokens
        role_stat["used_input_tokens_total"] += used_input_tokens

        trace = {
            "timestamp": time.time(),
            "role": role,
            "sample_id": state.get("id"),
            "question": state.get("question", ""),
            "step_num": step_num,
            "retry_index": retry_index,
            "global_call_index": global_idx,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "used_input_tokens": used_input_tokens,
            "cache_source": cache_source,
            "fallback_components": fallback_components,
        }
        self._append_trace(trace)

        self.role_global_call_index[role] = global_idx + 1
        self.sample_role_call_count[sample_key] = sample_role_idx + 1

    def dump_stats(self, total_stats: dict):
        if not self.enabled or not self.stats_file_path:
            return

        overall_prompt = sum(v["prompt_tokens_total"] for v in self.role_stats.values())
        overall_output = sum(v["output_tokens_total"] for v in self.role_stats.values())
        overall_cached = sum(v["cached_tokens_total"] for v in self.role_stats.values())
        overall_used_input = sum(v["used_input_tokens_total"] for v in self.role_stats.values())
        overall_calls = sum(v["calls"] for v in self.role_stats.values())
        overall_exact_calls = sum(v["exact_calls"] for v in self.role_stats.values())
        overall_fallback_calls = sum(v["fallback_calls"] for v in self.role_stats.values())

        saving_ratio_input_pct = (
            (overall_cached / overall_prompt) * 100.0 if overall_prompt > 0 else 0.0
        )

        payload = {
            "config": {
                "enabled": self.enabled,
                "mode": self.mode,
                "trace_file_path": self.trace_file_path,
                "stats_file_path": self.stats_file_path,
            },
            "system_tokens": self.system_tokens,
            "roles": self.role_stats,
            "overall": {
                "calls": overall_calls,
                "prompt_tokens_total": overall_prompt,
                "output_tokens_total": overall_output,
                "cached_tokens_total": overall_cached,
                "used_input_tokens_total": overall_used_input,
                "exact_calls": overall_exact_calls,
                "fallback_calls": overall_fallback_calls,
                "saving_ratio_input_pct": saving_ratio_input_pct,
            },
            "integrity": {
                "generator_calls_matches_total_stats": (
                    self.role_stats["generator"]["calls"] == total_stats.get("generator_calls", -1)
                ),
                "evaluator_calls_matches_total_stats": (
                    self.role_stats["evaluator"]["calls"] == total_stats.get("evaluator_calls", -1)
                ),
                "cached_le_prompt_all_calls": self.cached_le_prompt_all_calls,
            },
        }

        with open(self.stats_file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

def load_generator_model(model_id: str, gpu_memory_utilization: float = 0.4):
    """
    Reasoning을 수행할 Base Generator 모델을 vLLM으로 로드합니다.
    """
    print(f"Generator 모델 로딩 중 (vLLM)... Model: '{model_id}'")
    
    # 1. 프롬프트 생성용 토크나이저 로드 (HuggingFace)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. 추론용 vLLM 엔진 로드
    llm = LLM(
        model=model_id,
        tensor_parallel_size=4,
        gpu_memory_utilization=gpu_memory_utilization, # 두 모델을 띄우기 위해 메모리 제한
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=10000,
        enable_prefix_caching=True,
        seed=42
    )
    
    print("✅ Generator 모델 로드 완료.")
    return llm, tokenizer

def load_finetuned_evaluator(base_model_id: str, adapter_path: str, gpu_memory_utilization: float = 0.4):
    """
    파인튜닝된 LoRA 어댑터를 사용할 평가 모델을 vLLM으로 로드합니다.
    (bitsandbytes 4bit 양자화 적용)
    """
    print(f"평가자(Evaluator) 모델 로딩 중 (vLLM 4bit)... Base: '{base_model_id}'")
    
    # 1. 프롬프트 생성용 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 추론용 vLLM 엔진 (LoRA + 4bit bitsandbytes 활성화)
    llm = LLM(
        model=base_model_id,
        enable_lora=True,              # LoRA 활성화
        quantization="bitsandbytes",   # ★ 핵심: 4bit (nf4) 로딩을 위해 설정
        load_format="bitsandbytes",    # 가중치 포맷 명시 (필수는 아니지만 권장)
        tensor_parallel_size=4,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=10000,
        max_lora_rank=64,
        enable_prefix_caching=True,
        seed=42
    )

    print("✅ 평가자 모델(4bit) 로드 완료.")
    return llm, tokenizer

def parse_eval_response(response_text: str) -> dict:
    """
    Evaluator 응답 텍스트를 강력하게(Robust) 파싱하고, 필수 키를 보정하는 함수.
    """
    # 1. 초기값 설정 (실패 시 반환할 형태)
    fallback_result = {
        "error_type": "Parsing Error",
        "diagnosis": "No JSON object found or parsing failed.",
        "guidance": "Check model output format."
    }

    if not response_text:
        return fallback_result

    text = response_text.strip()

    # 2. 마크다운 코드 블록 제거 (Regex 사용이 split보다 안전함)
    # ```json ... ``` 또는 ``` ... ``` 패턴 추출
    markdown_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if markdown_match:
        text = markdown_match.group(1)

    # 3. JSON 객체 시작('{') 찾기
    start_idx = text.find('{')
    if start_idx == -1:
        return fallback_result
    
    # 4. JSON 후보 문자열 추출 (앞부분 잡음 제거)
    json_str = text[start_idx:].strip()

    # 5. 파싱 시도 (3단계 전략)
    parsed = None
    
    # 전략 A: 뒤쪽 잡음 제거 후 파싱
    end_idx = json_str.rfind('}')
    if end_idx != -1:
        candidate = json_str[:end_idx+1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 전략 B: 전략 A 실패 시, 잘린 JSON 복구 (Truncation Repair)
    if parsed is None:
        repair_patterns = [
            '', '}', '"}', ']}', '"]}', '}}', '"}}'
        ]
        for pattern in repair_patterns:
            try:
                parsed = json.loads(json_str + pattern)
                break
            except json.JSONDecodeError:
                continue

    # 전략 C: 전략 B 실패 시, Python Literal Eval (Single Quote 대응)
    if parsed is None:
        try:
            parsed = ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            pass

    # 6. 결과 반환 및 키 보정
    if parsed and isinstance(parsed, dict):
        # 필수 키가 없으면 기본값 주입
        if "error_type" not in parsed: parsed["error_type"] = "Unknown"
        if "diagnosis" not in parsed: parsed["diagnosis"] = "No diagnosis provided."
        if "guidance" not in parsed: parsed["guidance"] = "No guidance provided."
        return parsed
    else:
        # 최후의 수단: 실패 반환 시 원본 텍스트 일부를 진단에 포함 (디버깅용)
        fallback_result["diagnosis"] = f"Parsing failed. Raw text start: {response_text[:50]}..."
        return fallback_result

def run_dynamic_batch_inference(
    df: pd.DataFrame,
    gen_llm: LLM,
    gen_tokenizer,
    eval_llm: LLM,
    eval_tokenizer,
    adapter_path: str,
    result_file_path: str,
    log_file_path: str,
    stats_file_path: str,
    cache_trace_file_path: Optional[str] = None,
    cache_stats_file_path: Optional[str] = None,
    track_cache_stats: bool = False,
    cache_stats_mode: str = "exact_or_fallback",
    evaluator_system_prompt: str = evaluate_system_prompt_premature_attribution,
    max_steps: int = 10,
    max_retries: int = 3,
    batch_size: int = 32,
    is_qwen3_8b: bool = False,
    qwen3_thinking_mode: str = "off",
    disable_evaluator_thinking: bool = False,
    skip_missing_assistantfinal_for_oss20b: bool = False,
    generation_feedback_mode: str = "guidance_only",
):
    generation_system_prompt = select_generation_system_prompt(generation_feedback_mode)

    # 1. 대기열(Queue) 생성: 처리해야 할 모든 데이터를 큐에 넣음
    pending_queue = deque()
    for _, row in df.iterrows():
        pending_queue.append(row)

    # 2. 현재 작업 중인 슬롯 (Active Slots)
    active_states = [] 
    
    total_stats = {
        "generator_calls": 0,
        "evaluator_calls": 0,
        "total_tokens": 0,
        "completed_count": 0,
        "skipped_missing_assistantfinal": 0,
    }
    cache_tracker = CacheTracker(
        enabled=track_cache_stats,
        mode=cache_stats_mode,
        trace_file_path=cache_trace_file_path,
        stats_file_path=cache_stats_file_path,
        gen_tokenizer=gen_tokenizer,
        eval_tokenizer=eval_tokenizer,
        evaluator_system_prompt=evaluator_system_prompt,
        generation_system_prompt=generation_system_prompt,
    )
    
    # 저장 헬퍼 함수
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

    # 상태 초기화 헬퍼 함수
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
            "step_texts": [],
            "feedback_list": [],
            "logs": {
                "meta_data": {"question": row['question'], "retrieved_passages": passages},
                "steps_history": []
            },
            "current_retry": 0,
            "last_feedback": None,
            "finished": False,
            "ground_truth": row.get('answer', 'N/A'),
            "temp_gen_text": None,
            "current_step_log": None,
            "skipped": False,
            "skip_reason": None,
        }

    # -------------------------------------------------------------------------
    # Main Loop: 대기열이 있거나, 작업 중인 것이 있으면 계속 돔
    # -------------------------------------------------------------------------
    pbar = tqdm(total=len(df), desc="Dynamic Batch Processing")
    
    while pending_queue or active_states:
        
        # [Step 1] Refill: 빈 자리가 있고 대기열이 남았으면 채워넣기
        while len(active_states) < batch_size and pending_queue:
            new_row = pending_queue.popleft()
            active_states.append(create_new_state(new_row))
        
        if not active_states: break # 안전장치

        # [Step 2] Generator Phase
        # 현재 활성 상태인 모든 요청에 대해 프롬프트 생성
        gen_prompts = []
        
        # active_states의 순서가 유지되므로 인덱스로 매핑 가능
        for state in active_states:
            
            # 상태 관리 (Retry 0일 때 새 로그 생성)
            if state["current_retry"] == 0:
                state["current_step_log"] = {
                    "step_num": len(state["step_texts"]) + 1, "status": "In Progress", "attempts": []
                }
            
            # Max Step 체크 -> finished 처리 (여기서는 바로 종료시키지 않고 빈 텍스트 생성 유도하거나 처리 필요)
            # 깔끔한 처리를 위해, 이미 finished 된 상태라면 프롬프트 생성을 건너뛰어야 하지만,
            # active_states는 항상 진행 중인 것만 남기므로 여기엔 unfinished만 있음.
            # 단, max_step 도달 검사는 루프 끝에서 제거하면서 수행.
            
            # Prompt 구성
            passages_str = '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(state["passages"])])
            previous_steps_str = '\n'.join(state["step_texts"]) if state["step_texts"] else "(No previous steps.)"
            
            # Feedback 문자열 구조화
            force_answer_mode = False
            feedback_str = ""
            if not state["last_feedback"]:
                # 첫 스텝이거나, 이전 스텝이 성공해서 그냥 넘어온 경우
                feedback_str = "Status: N/A (First attempt at this step)"
            else:
                # 에러가 발생해서 재시도(Retry)하는 경우, 또는 이전 단계 성공 후 피드백이 있는 경우
                err = state["last_feedback"].get("error_type", "Unknown")
                failed_text = state["last_feedback"].get("failed_text", None)

                if generation_feedback_mode == "guidance_only":
                    guid = state["last_feedback"].get("guidance", "Proceed logically.")

                    # Guidance에 ####ANSWER: 포함 시, 최종 정답 생성 모드로 진입
                    cond_format = "####ANSWER:" in guid
                    if cond_format:
                        force_answer_mode = True

                    if "Correct" in err:
                        feedback_str = f"""[Previous Step Was Correct. You must follow the Guidance for the next step.]
- Guidance: {guid}"""

                    # 조건: 에러가 있고(else) + 현재 retry 카운트가 0임
                    elif state["current_retry"] == 0:
                        # 이때 failed_text는 이미 previous_steps_str에 포함되어 있으므로 다시 보여주지 않음. 대신 강력한 경고 메시지를 줌.
                        feedback_str = f"""[WARNING: The previous step (Step {len(state["step_texts"])}) failed verification multiple times but was retained. Proceed with caution, following the Guidance.]
- Guidance for this step: {guid}"""

                    else:
                        # 에러가 나서 재시도 하는 경우 (가장 중요)
                        if failed_text:
                            feedback_str = f"""[Previous Incorrect Attempt at Step {len(state['step_texts']) + 1}]
{failed_text}

[Feedback on Previous Attempt. You must follow the Guidance carefully.]
- Guidance: {guid}"""
                        else: # 이 경우는 발생하지 않아야 함.
                            feedback_str = f"""[Previous Step Failed. You must follow the Guidance carefully.]
- Guidance: {guid}"""
                else:
                    diag = state["last_feedback"].get("diagnosis", "None")

                    if "Correct" in err:
                        feedback_str = f"""[Previous Step Was Correct. Use the Diagnosis as context for the next step.]
- Diagnosis: {diag}"""

                    elif state["current_retry"] == 0:
                        feedback_str = f"""[WARNING: The previous step (Step {len(state["step_texts"])}) failed verification multiple times but was retained. Proceed with caution, using the Diagnosis.]
- Previous Error Diagnosis: {diag}"""

                    else:
                        if failed_text:
                            feedback_str = f"""[Previous Incorrect Attempt at Step {len(state['step_texts']) + 1}]
{failed_text}

[Feedback on Previous Attempt. Use the Diagnosis carefully.]
- Diagnosis: {diag}"""
                        else: # 이 경우는 발생하지 않아야 함.
                            feedback_str = f"""[Previous Step Failed. Use the Diagnosis carefully.]
- Diagnosis: {diag}"""

            prompt_user = f"""Question:
{state['question']}

Retrieved Passages:
{passages_str}

Previous Reasoning Steps:
{previous_steps_str}

Feedback:
{feedback_str}

Generate next step (start with `Step {len(state['step_texts']) + 1}:`)"""
            
            if force_answer_mode:
                prompt_user = f"""Question: {state['question']}

Retrieved Passages:
{passages_str}

Reasoning History:
{previous_steps_str}

Feedback on Last Step:
{feedback_str}

The reasoning seems complete, but the final answer format is missing.
Please output ONLY the final answer now. 
Remember to use the format: ####ANSWER: final_answer_here (Final Answer).
""".strip()
                messages = [{"role": "system", "content": FORCE_ANSWER_SYSTEM_PROMPT}, {"role": "user", "content": prompt_user}]
            else: # generation_feedback_mode에 맞는 step-generation system prompt 사용
                messages = [{"role": "system", "content": generation_system_prompt}, {"role": "user", "content": prompt_user}]
            gen_template_kwargs = {"add_generation_prompt": True, "tokenize": False}
            if is_qwen3_8b:
                gen_template_kwargs["enable_thinking"] = qwen3_thinking_mode == "on"
            gen_prompts.append(gen_tokenizer.apply_chat_template(messages, **gen_template_kwargs))

        # Generator 실행 (Batch)
        # stop_token_ids 후보들 중 None이 아닌 정수값만 추출
        stop_candidates = [gen_tokenizer.eos_token_id, gen_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        actual_stop_tokens = [t for t in stop_candidates if t is not None]

        gen_sampling_params = build_generator_sampling_params(
            is_qwen3_8b=is_qwen3_8b,
            qwen3_thinking_mode=qwen3_thinking_mode,
            stop_token_ids=actual_stop_tokens,
        )

        gen_outputs = gen_llm.generate(
            gen_prompts, 
            gen_sampling_params, 
            use_tqdm=False
        )
        total_stats["generator_calls"] += len(gen_outputs)

        # 결과 매핑
        states_for_eval = [] 
        eval_prompts = []

        for i, output in enumerate(gen_outputs):
            state = active_states[i]
            full_generated_text = output.outputs[0].text.strip()
            prompt_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)
            total_stats["total_tokens"] += prompt_tokens + output_tokens

            if track_cache_stats:
                try:
                    cache_tracker.record_call(
                        role="generator",
                        state=state,
                        output_obj=output,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        step_num=state["current_step_log"]["step_num"],
                        retry_index=state["current_retry"],
                    )
                except Exception as e:
                    if cache_stats_mode == "exact_only":
                        raise
                    cache_tracker.record_errors += 1
                    if cache_tracker.record_errors <= 3:
                        print(f"⚠️ Cache tracking warning (generator): {e}")

            if skip_missing_assistantfinal_for_oss20b:
                marker_pos = full_generated_text.lower().find("assistantfinal")
                if marker_pos == -1:
                    state["finished"] = True
                    state["skipped"] = True
                    state["skip_reason"] = "missing_assistantfinal"
                    total_stats["skipped_missing_assistantfinal"] += 1

                    if state["current_step_log"] is None:
                        state["current_step_log"] = {
                            "step_num": len(state["step_texts"]) + 1,
                            "status": "Skipped (Missing assistantfinal)",
                            "attempts": [],
                        }

                    state["current_step_log"]["attempts"].append(
                        {
                            "retry_index": state["current_retry"],
                            "result": "Skipped",
                            "note": "assistantfinal marker missing in generator output for gpt-oss-20b",
                            "generated_preview": full_generated_text[:300],
                        }
                    )
                    state["current_step_log"]["status"] = "Skipped (Missing assistantfinal)"
                    state["logs"]["steps_history"].append(state["current_step_log"])

                    print(
                        f"⚠️ Skipping sample id={state['id']} due to missing assistantfinal marker "
                        "for gpt-oss-20b output."
                    )
                    continue

                raw_generated_text = full_generated_text[marker_pos + len("assistantfinal"):].strip()
            else:
                raw_generated_text = full_generated_text.split("assistantfinal")[-1].strip()

            generated_text = strip_think_blocks(raw_generated_text)

            # Cleaning
            for marker in ["<start_of_turn>", "User:", "## Question"]:
                if marker in generated_text: generated_text = generated_text.split(marker)[0].strip()
            
            expected_start = f"Step {len(state['step_texts']) + 1}:"
            if not generated_text.startswith(expected_start):
                if not generated_text.startswith("Step"): generated_text = f"{expected_start} " + generated_text.lstrip()
                elif generated_text.startswith("Step"): generated_text = generated_text.split('\n')[0]
            
            # final answer step인데 (Final Answer) 태그가 없으면 추가
            if "####ANSWER: " in generated_text and "(Final Answer)" not in generated_text:
                generated_text = generated_text + " (Final Answer)"

            state["temp_gen_text"] = generated_text
            states_for_eval.append(state)
            
            # Eval Prompt 구성
            if isinstance(state['passages'], str):
                passages = eval(state['passages'])
            else:
                passages = state['passages']
            if isinstance(state['step_texts'], str):
                step_texts = eval(state['step_texts'])
            else:
                step_texts = state['step_texts']
            
            context_str = '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)]).strip()
            previous_steps_str = '\n'.join(step_texts).strip()
            user_content = f"""### Task: Evaluate the Correctness of the Reasoning Step

Question:
{state['question']}

Retrieved Passages:
{context_str}

Previous Steps:
{previous_steps_str}

Step to evaluate:
{state['temp_gen_text']}
""".strip()
            messages = [{"role": "system", "content": evaluator_system_prompt}, {"role": "user", "content": user_content}]
            eval_template_kwargs = {"add_generation_prompt": True, "tokenize": False}
            if disable_evaluator_thinking:
                eval_template_kwargs["enable_thinking"] = False
            eval_prompts.append(eval_tokenizer.apply_chat_template(messages, **eval_template_kwargs))

        # [Step 3] Evaluator Phase
        if eval_prompts:
            eval_outputs = eval_llm.generate(eval_prompts, SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=[eval_tokenizer.eos_token_id]), lora_request=LoRARequest("evaluator_adapter", 1, adapter_path), use_tqdm=False)
            total_stats["evaluator_calls"] += len(eval_outputs)

            for i, output in enumerate(eval_outputs):
                state = states_for_eval[i]
                raw_eval = output.outputs[0].text.strip()
                prompt_tokens = len(output.prompt_token_ids)
                output_tokens = len(output.outputs[0].token_ids)
                total_stats["total_tokens"] += prompt_tokens + output_tokens

                if track_cache_stats:
                    try:
                        cache_tracker.record_call(
                            role="evaluator",
                            state=state,
                            output_obj=output,
                            prompt_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            step_num=state["current_step_log"]["step_num"],
                            retry_index=state["current_retry"],
                        )
                    except Exception as e:
                        if cache_stats_mode == "exact_only":
                            raise
                        cache_tracker.record_errors += 1
                        if cache_tracker.record_errors <= 3:
                            print(f"⚠️ Cache tracking warning (evaluator): {e}")
                
                # JSON 결과 파싱
                parsed_eval = parse_eval_response(raw_eval)
                
                attempt_record = {"retry_index": state["current_retry"], "generated_text": state["temp_gen_text"], "evaluation": parsed_eval, "result": "Pending"}
                err_type = parsed_eval.get("error_type", "Unknown")
                
                # 조건 1: 생성된 텍스트에 ####ANSWER 포함
                has_answer_tag = "####ANSWER" in state["temp_gen_text"]
                if generation_feedback_mode == "guidance_only":
                    # guidance_only에서는 guidance의 종료 시그널까지 충족해야 완료
                    guidance_text = parsed_eval.get("guidance", "")
                    has_stop_token = "[END_OF_REASONING]" in guidance_text
                    should_finish = has_answer_tag and has_stop_token
                else:
                    # diagnosis_only에서는 guidance를 참조하지 않음
                    should_finish = has_answer_tag

                # Logic Update
                if 'correct' in err_type.lower():
                    attempt_record["result"] = "Accepted"
                    state["current_step_log"]["attempts"].append(attempt_record)
                    state["current_step_log"]["status"] = "Completed"
                    state["logs"]["steps_history"].append(state["current_step_log"])
                    state["step_texts"].append(state["temp_gen_text"])
                    state["feedback_list"].append(parsed_eval)
                    state["current_retry"] = 0
                    state["last_feedback"] = deepcopy(parsed_eval)
                    state["last_feedback"].pop("failed_text", None)
                    
                    # mode별 종료 조건
                    if should_finish:
                        state["finished"] = True
                        state["current_step_log"]["status"] = "Finished (Verified Answer)"
                    else:
                        # Correct지만 mode별 종료 조건이 충족되지 않으면 계속 진행
                        state["finished"] = False

                # [변경 4] 기타 에러 처리 (Overthinking 등으로 인한 강제 종료 제거 -> 그냥 Retry 로직으로 통합)
                else:
                    # 만약 ####ANSWER가 있는데 틀렸다고 판단되면 여기서 Rejected 됨 (Rollback)
                    attempt_record["result"] = "Rejected (Rollback)"
                    state["current_step_log"]["attempts"].append(attempt_record)
                    state["last_feedback"] = parsed_eval
                    
                    # 직전에 실패한 step도 저장해서 함께 제공
                    state["last_feedback"]["failed_text"] = state["temp_gen_text"]
                    
                    state["current_retry"] += 1
                    
                    if state["current_retry"] >= max_retries:
                        state["current_step_log"]["attempts"][-1]["result"] = "Max retries"
                        state["current_step_log"]["status"] = "Max retries"
                        state["logs"]["steps_history"].append(state["current_step_log"])
                        state["step_texts"].append(state["temp_gen_text"])
                        state["feedback_list"].append(parsed_eval)
                        state["current_retry"] = 0
                        
                        # 다음 step의 첫번째 생성때, 직전 previous steps의 마지막 시도가 잘못되었음을 알려줘야하므로 피드백 포함.
                        # state["last_feedback"] = None
                        state["last_feedback"] = parsed_eval
                        

        # [Step 4] Remove Finished & Save
        # 완료된 것과 계속 진행할 것 분리
        next_active_states = []
        finished_results = []
        finished_logs = []

        for state in active_states:
            # 최대 스텝 체크 (루프 돌면서 자연스럽게 도달했을 수 있음)
            if not state["finished"] and len(state["step_texts"]) >= max_steps:
                state["finished"] = True
                # 마지막 로그가 안 들어갔으면 넣어줌 (보통 Generator 단계에서 처리되지만 안전장치)
                if state["current_step_log"] and state["current_step_log"] not in state["logs"]["steps_history"]:
                     state["logs"]["steps_history"].append(state["current_step_log"])

            if state["finished"]:
                is_skipped = state.get("skipped", False)
                skip_reason = state.get("skip_reason")
                # 결과 포맷팅
                res_obj = {
                    "id": state["id"],
                    "question": state["question"],
                    "context": 'Retrieved Passages:\n' + '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(state['passages'])]),
                    "response": [] if is_skipped else state["step_texts"],
                    "feedback": [] if is_skipped else [f"Feedback for Step {i+1}: {f}" for i, f in enumerate(state['feedback_list'])],
                    "ground_truth": state["ground_truth"],
                    "skipped": is_skipped,
                    "skip_reason": skip_reason,
                    "status": "Skipped" if is_skipped else "Completed",
                }
                finished_results.append(res_obj)
                finished_logs.append(state["logs"])
                pbar.update(1)
            else:
                next_active_states.append(state)

        # 완료된 것들 즉시 저장
        if finished_results:
            append_to_json_file(result_file_path, finished_results)
            append_to_json_file(log_file_path, finished_logs)
            total_stats["completed_count"] += len(finished_results)
            # 통계 업데이트
            append_to_json_file(stats_file_path, [deepcopy(total_stats)])
            if track_cache_stats:
                cache_tracker.dump_stats(total_stats)

        # active 리스트 교체 (다음 루프에서 refill 됨)
        active_states = next_active_states
        
    pbar.close()
    if track_cache_stats:
        cache_tracker.dump_stats(total_stats)
    return total_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--generator_model", type=str, required=True)
    parser.add_argument("--feedback_model", type=str, required=True)
    parser.add_argument(
        "--generation_feedback_mode",
        type=str,
        choices=["diagnosis_only", "guidance_only"],
        required=True,
        help="Ablation mode for generation prompt feedback fields.",
    )
    parser.add_argument(
        "--qwen3_thinking_mode",
        type=str,
        choices=["on", "off"],
        default="off",
        help="Only used when --generator_model is Qwen3-8B.",
    )
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--ablation_output_root",
        type=str,
        default="/workspace/daeyong/inference_results/dev_kg_correct_1ksample_with_noises_10_3_errortype_ablation",
        help="Output root for diagnosis/guidance ablation results.",
    )
    parser.add_argument(
        "--track_cache_stats",
        action="store_true",
        help="Enable cache tracking logs/stats without changing inference behavior.",
    )
    parser.add_argument(
        "--cache_stats_mode",
        type=str,
        choices=["exact_or_fallback", "exact_only"],
        default="exact_or_fallback",
        help="How to handle missing exact cached-token fields from vLLM outputs.",
    )
    args = parser.parse_args()

    generator_model_lower = args.generator_model.lower()
    is_qwen3_8b = "qwen3" in generator_model_lower and "8b" in generator_model_lower
    is_gpt_oss_20b = "oss" in generator_model_lower and "-20b" in generator_model_lower

    # Generator 설정 (파일 저장명 용도)
    if "llama" in generator_model_lower and "-8b" in generator_model_lower:
        model_name = "llama8b"
    elif "llama" in generator_model_lower and "-70b" in generator_model_lower:
        model_name = "llama70b"
    elif is_qwen3_8b:
        model_name = "qwen8b"
    elif "qwen" in generator_model_lower and "-4b" in generator_model_lower:
        model_name = "qwen4b"
    elif "qwen" in generator_model_lower and "-7b" in generator_model_lower:
        model_name = "qwen7b"
    elif "qwen" in generator_model_lower and "-14b" in generator_model_lower:
        model_name = "qwen14b"
    elif "gemma" in generator_model_lower:
        model_name = "gemma12b"
    elif is_gpt_oss_20b:
        model_name = "oss20b"
    else:
        model_name = "unknown"

    # Evaluator 설정
    # [수정 1] 경로를 변수로 관리하여 불일치 방지
    # base_model_id = "/workspace/hf_transformers/Qwen2.5-7B-Instruct"
    base_model_id = "/workspace/hf_transformers/Qwen3-8B"
    # base_model_id = "/workspace/hf_transformers/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    is_qwen3_8b_evaluator = ("qwen3" in base_model_id.lower()) and ("8b" in base_model_id.lower())
    adapter_path = resolve_adapter_path(args.feedback_model)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Evaluator adapter path not found: {adapter_path}")

    evaluator_system_prompt = evaluate_system_prompt_premature_attribution

    # -------------------------------------------------------------------------
    # 1. vLLM 모델 로드 (GPU 메모리 분할)
    # -------------------------------------------------------------------------
    # 주의: 하나의 GPU에 2개의 LLM 엔진을 올리려면 gpu_memory_utilization 합이 1.0 미만이어야 합니다.
    # 여기서는 각각 0.4씩 할당하여 총 0.8 사용을 목표로 합니다.
    
    gen_llm, gen_tokenizer = load_generator_model(
        args.generator_model, 
        gpu_memory_utilization=0.4
    )

    stop_candidates = [gen_tokenizer.eos_token_id, gen_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    actual_stop_tokens = [t for t in stop_candidates if t is not None]
    generator_sampling_params = build_generator_sampling_params(
        is_qwen3_8b=is_qwen3_8b,
        qwen3_thinking_mode=args.qwen3_thinking_mode,
        stop_token_ids=actual_stop_tokens,
    )
    print(
        f"🧠 Generator config | model={args.generator_model} "
        f"is_qwen3_8b={is_qwen3_8b} qwen3_thinking_mode={args.qwen3_thinking_mode}"
    )
    print(f"🎛️ Generator SamplingParams: {generator_sampling_params}")
    print(f"🧪 Evaluator thinking disabled: {is_qwen3_8b_evaluator}")
    print(f"🧪 Skip missing assistantfinal for oss20b: {is_gpt_oss_20b}")
    print(f"🧪 Generation feedback mode: {args.generation_feedback_mode}")
    print("🧪 Evaluator prompt mode: fixed default (premature_attribution)")
    
    # [수정 1 반영] 변수 사용
    eval_llm, eval_tokenizer = load_finetuned_evaluator(
        base_model_id, 
        adapter_path, 
        gpu_memory_utilization=0.4
    )
    
    # -------------------------------------------------------------------------
    # 데이터셋 로드
    # -------------------------------------------------------------------------
    # df = pd.read_csv(f"/workspace/daeyong/benchmarks/{args.dataset}_dev.csv")

    df = pd.read_csv(f"/workspace/daeyong/benchmarks/{args.dataset}_dev_kg_correct.csv").sample(n=1000, random_state=42)
    # if args.dataset == "2wiki":
    #     df = pd.read_csv("/workspace/daeyong/benchmarks/2wiki_20k_sample_yes.csv").sample(n=2000, random_state=42)
    # elif args.dataset == "hotpotqa":
    #     df = pd.read_csv("/workspace/daeyong/benchmarks/hotpotqa_20k_sample_yes.csv").sample(n=2000, random_state=42)
    # elif args.dataset == "musique":
    #     df = pd.read_csv("/workspace/daeyong/benchmarks/musique_yes.csv").sample(n=2000, random_state=42)
    #     # df = pd.read_json("/workspace/daeyong/benchmarks/musique_confusing_entities_filtered.json")
    feedback_model_clean = os.path.basename(os.path.normpath(adapter_path)).replace("-", "_")

    # 2. 저장할 디렉토리 경로 생성
    base_output_dir = args.ablation_output_root
    output_dir = os.path.join(base_output_dir, feedback_model_clean, args.generation_feedback_mode)
    os.makedirs(output_dir, exist_ok=True)  # 폴더가 없으면 생성

    # 3. 파일 경로 설정 (지정된 폴더 내부로)
    log_file_path = os.path.join(output_dir, f"{model_name}_{args.dataset}_logs.json")
    result_file_path = os.path.join(output_dir, f"{model_name}_{args.dataset}.json")
    stats_file_path = os.path.join(output_dir, f"{model_name}_{args.dataset}_stats.json")
    cache_trace_file_path = os.path.join(
        output_dir, f"{model_name}_{args.dataset}_cache_trace.jsonl"
    )
    cache_stats_file_path = os.path.join(
        output_dir, f"{model_name}_{args.dataset}_cache_stats.json"
    )

    print(f"📂 Output Directory: {output_dir}")
    print(f"📄 Result File: {result_file_path}")

    # Resume 로직
    processed_ids = set()
    if os.path.exists(result_file_path):
        # [수정 2] 파일 깨짐 대비 안전장치 추가
        try:
            with open(result_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                processed_ids = {item['id'] for item in data}
            print(f"🔄 Resuming... Found {len(processed_ids)} processed examples.")
        except json.JSONDecodeError:
            print("⚠️ Existing result file is corrupted or empty. Starting from scratch.")
        except Exception as e:
            print(f"⚠️ Error reading result file: {e}. Starting from scratch.")
    
    # 이미 처리된 것 제외
    df = df[~df['id'].isin(processed_ids)]

    if len(df) > 0:
        print(f"🚀 Starting Dynamic Batch Inference for {len(df)} samples...")
        
        # 배치 실행
        final_stats = run_dynamic_batch_inference(
            df, gen_llm, gen_tokenizer, eval_llm, eval_tokenizer, 
            adapter_path=adapter_path,     # 변수 전달
            result_file_path=result_file_path,
            log_file_path=log_file_path,
            stats_file_path=stats_file_path,
            cache_trace_file_path=cache_trace_file_path,
            cache_stats_file_path=cache_stats_file_path,
            track_cache_stats=args.track_cache_stats,
            cache_stats_mode=args.cache_stats_mode,
            evaluator_system_prompt=evaluator_system_prompt,
            max_steps=args.max_steps,
            max_retries=args.max_retries,
            batch_size=256,
            is_qwen3_8b=is_qwen3_8b,
            qwen3_thinking_mode=args.qwen3_thinking_mode,
            disable_evaluator_thinking=is_qwen3_8b_evaluator,
            skip_missing_assistantfinal_for_oss20b=is_gpt_oss_20b,
            generation_feedback_mode=args.generation_feedback_mode,
        )
        print("✅ All processing complete.")
    else:
        print("✅ Nothing to process (All completed).")
