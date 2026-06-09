import pandas as pd
import numpy as np
import argparse
import os
import json
import ast
import math
import re
import time
from tqdm import tqdm
from copy import deepcopy
from collections import Counter, defaultdict, deque
from typing import Any, Optional

os.environ.setdefault("DAEYONG_VLLM_TORCH_PRELOAD", "1")

# vLLM 0.19.1 triggers torch._inductor during import. With torch 2.10.0,
# preloading these Dynamo internals avoids an import-order failure in
# torch._dynamo.utils where NP_SUPPORTED_MODULES is not initialized yet.
import torch._logging  # noqa: F401
import torch._numpy  # noqa: F401
from torch._guards import detect_fake_mode as _torch_detect_fake_mode  # noqa: F401
from torch._logging import LazyString as _TorchLazyString  # noqa: F401
from torch._dynamo import config as _torch_dynamo_config  # noqa: F401
from torch._subclasses.fake_tensor import (  # noqa: F401
    FakeTensor as _TorchFakeTensor,
    is_fake as _torch_is_fake,
    maybe_get_fake_mode as _torch_maybe_get_fake_mode,
)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from prompts import (
    evaluate_system_prompt_premature_attribution_missing_evidence,
    generate_single_step_system_prompt,
    generate_single_step_fixed_system_prompt,
)
from prompts_ablation import (
    evaluate_system_prompt_drop_wrong_conclusion,
    evaluate_system_prompt_drop_overthinking,
    evaluate_system_prompt_drop_off_topic,
    evaluate_system_prompt_drop_redundancy,
    evaluate_system_prompt_drop_inefficiency,
    evaluate_system_prompt_drop_contradictory,
    evaluate_system_prompt_drop_unsupported,
    evaluate_system_prompt_drop_information_miss,
    evaluate_system_prompt_drop_premature_attribution,
    evaluate_system_prompt_drop_logical_fallacy,
    evaluate_system_prompt_drop_contradictory_information_miss_unsupported_premature_attribution,
    evaluate_system_prompt_drop_off_topic_inefficiency_redundancy_overthinking,
)

_GEMMA4_BNB_QKV_PATCHED = False


def _gemma4_k_eq_v_layer_indices(model_id: str) -> set[int]:
    """Return Gemma4 full-attention layers whose V projection is tied to K."""
    if not os.path.isdir(model_id):
        return set()

    config_path = os.path.join(model_id, "config.json")
    if not os.path.exists(config_path):
        return set()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return set()

    text_config = config.get("text_config", config)
    if text_config.get("model_type") != "gemma4_text":
        return set()
    if not text_config.get("attention_k_eq_v", False):
        return set()

    layer_types = text_config.get("layer_types") or []
    return {
        idx
        for idx, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    }


def patch_vllm_bitsandbytes_gemma4_k_eq_v():
    """Patch vLLM BNB online 4bit loading for Gemma4 K==V attention layers."""
    global _GEMMA4_BNB_QKV_PATCHED
    if _GEMMA4_BNB_QKV_PATCHED:
        return

    try:
        from vllm.model_executor.model_loader.bitsandbytes_loader import (
            BitsAndBytesModelLoader,
        )
    except Exception as exc:
        print(f"⚠️ Gemma4 BNB qkv patch unavailable: {exc}")
        return

    original = BitsAndBytesModelLoader._get_quantized_weights_iterator
    if getattr(original, "_daeyong_gemma4_k_eq_v_patch", False):
        _GEMMA4_BNB_QKV_PATCHED = True
        return

    def _map_loader_name(loader, name: str) -> str:
        mapped = loader.weight_mapper(name)
        target_modules = getattr(loader, "target_modules", [])
        if (
            getattr(loader, "is_pool_model", False)
            and target_modules
            and target_modules[0].startswith("model.")
            and not mapped.startswith("model.")
        ):
            mapped = "model." + mapped
        return mapped

    def patched_get_quantized_weights_iterator(self, model_name_or_path, revision):
        iterator, quant_state_dict = original(self, model_name_or_path, revision)
        k_eq_v_layers = _gemma4_k_eq_v_layer_indices(model_name_or_path)
        if not k_eq_v_layers:
            return iterator, quant_state_dict

        def patched_iterator():
            layer_pattern = re.compile(
                r"(?:^|\.)layers\.(\d+)\.self_attn\.k_proj\.weight$"
            )
            for org_name, processed_weight in iterator:
                yield org_name, processed_weight

                match = layer_pattern.search(org_name)
                if match is None or int(match.group(1)) not in k_eq_v_layers:
                    continue

                v_org_name = org_name.replace(".k_proj.", ".v_proj.")
                k_mapped_name = _map_loader_name(self, org_name)
                v_mapped_name = _map_loader_name(self, v_org_name)
                if k_mapped_name in quant_state_dict:
                    quant_state_dict[v_mapped_name] = quant_state_dict[k_mapped_name]

                yield v_org_name, processed_weight

        return patched_iterator(), quant_state_dict

    patched_get_quantized_weights_iterator._daeyong_gemma4_k_eq_v_patch = True
    BitsAndBytesModelLoader._get_quantized_weights_iterator = patched_get_quantized_weights_iterator
    _GEMMA4_BNB_QKV_PATCHED = True

FORCE_ANSWER_SYSTEM_PROMPT = """You are an expert answering agent.
The reasoning process is complete. Your task is to formulate the FINAL ANSWER based on the provided history.

INSTRUCTIONS:
1. Do not generate any new reasoning steps.
2. Directly output the final answer.
3. YOU MUST USE THE FOLLOWING FORMAT:
####ANSWER: your_final_answer_here (Final Answer)"""


DROP_EVALUATOR_PROMPT_MAP = {
    "drop_wrong_conclusion": evaluate_system_prompt_drop_wrong_conclusion,
    "drop_overthinking": evaluate_system_prompt_drop_overthinking,
    "drop_off_topic": evaluate_system_prompt_drop_off_topic,
    "drop_redundancy": evaluate_system_prompt_drop_redundancy,
    "drop_inefficiency": evaluate_system_prompt_drop_inefficiency,
    "drop_contradictory": evaluate_system_prompt_drop_contradictory,
    "drop_unsupported": evaluate_system_prompt_drop_unsupported,
    "drop_information_miss": evaluate_system_prompt_drop_information_miss,
    "drop_premature_attribution": evaluate_system_prompt_drop_premature_attribution,
    "drop_logical_fallacy": evaluate_system_prompt_drop_logical_fallacy,
    "drop_contradictory_information_miss_unsupported_premature_attribution": (
        evaluate_system_prompt_drop_contradictory_information_miss_unsupported_premature_attribution
    ),
    "drop_off_topic_inefficiency_redundancy_overthinking": (
        evaluate_system_prompt_drop_off_topic_inefficiency_redundancy_overthinking
    ),
}


def resolve_adapter_path(feedback_model_arg: str) -> str:
    if os.path.isabs(feedback_model_arg):
        return feedback_model_arg
    return f"/workspace/daeyong/trained_models/{feedback_model_arg}"


def get_drop_key_from_adapter_path(adapter_path: str) -> Optional[str]:
    adapter_name = os.path.basename(os.path.normpath(adapter_path))
    if adapter_name in DROP_EVALUATOR_PROMPT_MAP:
        return adapter_name
    return None


def select_evaluator_system_prompt(adapter_path: str) -> str:
    drop_key = get_drop_key_from_adapter_path(adapter_path)
    if drop_key:
        return DROP_EVALUATOR_PROMPT_MAP[drop_key]
    return evaluate_system_prompt_premature_attribution_missing_evidence


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
    is_qwen3_generator: bool,
    qwen3_thinking_mode: str,
    stop_token_ids: list[int],
) -> SamplingParams:
    if is_qwen3_generator:
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


def build_stop_token_ids(tokenizer) -> list[int]:
    token_ids = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, list):
        token_ids.extend(eos_token_id)
    elif eos_token_id is not None:
        token_ids.append(eos_token_id)

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in ("<|eot_id|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unk_token_id:
            continue
        token_ids.append(token_id)

    return list(dict.fromkeys(token_ids))


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
    ):
        self.enabled = enabled
        self.mode = mode
        self.trace_file_path = trace_file_path
        self.stats_file_path = stats_file_path
        self.gen_tokenizer = gen_tokenizer
        self.eval_tokenizer = eval_tokenizer
        self.evaluator_system_prompt = evaluator_system_prompt
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
                    generate_single_step_fixed_system_prompt,
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
        # Retrieval can append passages mid-sample, so the Q/P cache key must
        # include the current passage content rather than only the sample id.
        passages = state.get("passages", [])
        cache_key = (role, state.get("id"), tuple(map(str, passages)))
        if cache_key in self.qp_token_cache:
            return self.qp_token_cache[cache_key]

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

def load_generator_model(
    model_id: str,
    gpu_memory_utilization: float = 0.4,
    generator_quantization: str = "none",
    tensor_parallel_size: int = 4,
    max_model_len: int = 10000,
):
    """
    Reasoning을 수행할 Base Generator 모델을 vLLM으로 로드합니다.
    """
    print(f"Generator 모델 로딩 중 (vLLM)... Model: '{model_id}'")
    
    # 1. 프롬프트 생성용 토크나이저 로드 (HuggingFace)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generator_quantization = generator_quantization.lower()
    if generator_quantization in {"4bit", "bnb4"}:
        generator_quantization = "bitsandbytes"

    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "enable_prefix_caching": True,
        "seed": 42,
    }
    if generator_quantization == "bitsandbytes":
        patch_vllm_bitsandbytes_gemma4_k_eq_v()
        llm_kwargs.update(
            {
                "quantization": "bitsandbytes",
                "load_format": "bitsandbytes",
                "enforce_eager": True,
            }
        )
    elif generator_quantization != "none":
        raise ValueError(
            f"Unsupported generator_quantization: {generator_quantization}"
        )

    # 2. 추론용 vLLM 엔진 로드
    llm = LLM(**llm_kwargs)
    
    print("✅ Generator 모델 로드 완료.")
    return llm, tokenizer

def load_finetuned_evaluator(
    base_model_id: str,
    adapter_path: str,
    gpu_memory_utilization: float = 0.4,
    evaluator_quantization: str = "none",
    max_model_len: int = 8000,
):
    """
    파인튜닝된 LoRA 어댑터를 사용할 평가 모델을 vLLM으로 로드합니다.
    """
    print(
        f"평가자(Evaluator) 모델 로딩 중 (vLLM, quantization={evaluator_quantization})... "
        f"Base: '{base_model_id}', Adapter: '{adapter_path}'"
    )
    
    # 1. 프롬프트 생성용 토크나이저. LoRA adapter가 tokenizer/chat template을 함께
    # 저장하므로 adapter tokenizer를 우선 사용한다.
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = {
        "model": base_model_id,
        "tokenizer": adapter_path,
        "enable_lora": True,
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "max_lora_rank": 64,
        "enable_prefix_caching": True,
        "seed": 42,
    }
    if evaluator_quantization == "bitsandbytes":
        llm_kwargs.update(
            {
                "quantization": "bitsandbytes",
                "load_format": "bitsandbytes",
                "enforce_eager": True,
            }
        )
    elif evaluator_quantization != "none":
        raise ValueError(f"Unsupported evaluator_quantization: {evaluator_quantization}")

    llm = LLM(**llm_kwargs)

    print("✅ 평가자 모델 로드 완료.")
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


BM25_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
BM25_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "with",
}


def bm25_tokenize(text: Any) -> list[str]:
    tokens = []
    for token in BM25_TOKEN_PATTERN.findall(str(text or "").lower()):
        if token in BM25_STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


def normalize_passage_for_dedup(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def is_missing_evidence_error(error_type: Any) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", str(error_type or "").lower()).strip()
    return "missing evidence" in normalized


def is_correct_error_type(error_type: Any) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", str(error_type or "").lower()).strip()
    if "incorrect" in normalized or "not correct" in normalized:
        return False
    return "correct" in normalized


class BM25PassageRetriever:
    """Small dependency-free BM25 retriever over benchmarks/{dataset}_corpus.json."""

    def __init__(
        self,
        corpus_path: str,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.corpus_path = corpus_path
        self.k1 = k1
        self.b = b
        self.passages: list[str] = []
        self.passage_indices: list[Any] = []
        self.doc_lens: list[int] = []
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self._load_and_index()

    def _load_and_index(self):
        print(f"🔎 Loading BM25 corpus: {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        if not isinstance(corpus, list):
            raise ValueError(f"BM25 corpus must be a JSON list: {self.corpus_path}")

        postings = defaultdict(list)
        total_doc_len = 0
        for doc_id, item in enumerate(tqdm(corpus, desc="Indexing BM25 corpus")):
            if isinstance(item, dict):
                passage_text = item.get("passage_text", "")
                passage_index = item.get("passage_index", doc_id)
            else:
                passage_text = str(item)
                passage_index = doc_id

            self.passages.append(str(passage_text))
            self.passage_indices.append(passage_index)

            token_counts = Counter(bm25_tokenize(passage_text))
            doc_len = sum(token_counts.values())
            self.doc_lens.append(doc_len)
            total_doc_len += doc_len
            for token, tf in token_counts.items():
                postings[token].append((doc_id, tf))

        self.postings = dict(postings)
        num_docs = max(1, len(self.passages))
        self.avgdl = total_doc_len / num_docs if total_doc_len else 1.0
        self.idf = {
            token: math.log(1.0 + (num_docs - len(doc_postings) + 0.5) / (len(doc_postings) + 0.5))
            for token, doc_postings in self.postings.items()
        }
        print(f"✅ BM25 corpus indexed: {len(self.passages)} passages")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        existing_passages: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        query_terms = Counter(bm25_tokenize(query))
        if not query_terms:
            return []

        existing = {
            normalize_passage_for_dedup(passage)
            for passage in (existing_passages or [])
        }
        scores = defaultdict(float)
        for term, query_tf in query_terms.items():
            term_idf = self.idf.get(term)
            if term_idf is None:
                continue
            for doc_id, tf in self.postings.get(term, []):
                doc_len = self.doc_lens[doc_id] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_id] += query_tf * term_idf * (tf * (self.k1 + 1.0) / denom)

        ranked_doc_ids = sorted(scores, key=scores.get, reverse=True)
        results = []
        for doc_id in ranked_doc_ids:
            passage_text = self.passages[doc_id]
            if normalize_passage_for_dedup(passage_text) in existing:
                continue
            results.append(
                {
                    "passage_index": self.passage_indices[doc_id],
                    "passage_text": passage_text,
                    "score": float(scores[doc_id]),
                }
            )
            if len(results) >= top_k:
                break
        return results

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
    evaluator_system_prompt: str = evaluate_system_prompt_premature_attribution_missing_evidence,
    max_steps: int = 10,
    max_retries: int = 3,
    evaluator_max_tokens: int = 256,
    batch_size: int = 32,
    is_qwen3_generator: bool = False,
    qwen3_thinking_mode: str = "off",
    disable_evaluator_thinking: bool = False,
    skip_missing_assistantfinal_for_oss20b: bool = False,
    enable_missing_evidence_retrieval: bool = True,
    retrieval_corpus_path: Optional[str] = None,
    retrieval_top_k: int = 3,
    retrieval_max_per_sample: int = 1,
):
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
        "retrieval_calls": 0,
        "retrieval_passages_appended": 0,
    }
    retriever = None
    if enable_missing_evidence_retrieval:
        if not retrieval_corpus_path:
            raise ValueError("retrieval_corpus_path is required when retrieval is enabled.")
        retriever = BM25PassageRetriever(retrieval_corpus_path)

    cache_tracker = CacheTracker(
        enabled=track_cache_stats,
        mode=cache_stats_mode,
        trace_file_path=cache_trace_file_path,
        stats_file_path=cache_stats_file_path,
        gen_tokenizer=gen_tokenizer,
        eval_tokenizer=eval_tokenizer,
        evaluator_system_prompt=evaluator_system_prompt,
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
        passages = list(passages)
        
        return {
            "id": row['id'],
            "question": row['question'],
            "passages": passages,
            "step_texts": [],
            "feedback_list": [],
            "logs": {
                "meta_data": {
                    "question": row['question'],
                    "initial_retrieved_passages": deepcopy(passages),
                    "retrieved_passages": passages,
                },
                "steps_history": [],
                "retrieval_history": [],
            },
            "current_retry": 0,
            "last_feedback": None,
            "finished": False,
            "ground_truth": row.get('answer', 'N/A'),
            "temp_gen_text": None,
            "current_step_log": None,
            "skipped": False,
            "skip_reason": None,
            "retrieval_count": 0,
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
                # 에러가 발생해서 재시도(Retry)하는 경우, 또는 이전 단계 성공 후 가이드가 있는 경우
                err = state["last_feedback"].get("error_type", "Unknown")
                diag = state["last_feedback"].get("diagnosis", "None")
                guid = state["last_feedback"].get("guidance", "Proceed logically.")
                
                failed_text = state["last_feedback"].get("failed_text", None)
                
                # Guidance에 ####ANSWER: 포함 시, 최종 정답 생성 모드로 진입
                cond_format = "####ANSWER:" in guid

                if cond_format:
                    force_answer_mode = True
                    
                if "Correct" in err:
#                     feedback_str = f"""[Previous Step Was Correct]
# - Error Type: {err}
# - Diagnosis: {diag}
# - Guidance: {guid}"""

                    # Error Type 제거
                    feedback_str = f"""[Previous Step Was Correct. You must follow the Guidance for the next step.]
- Diagnosis: {diag}
- Guidance: {guid}"""

                # 조건: 에러가 있고(else) + 현재 retry 카운트가 0임
                elif state["current_retry"] == 0:
                    # 이때 failed_text는 이미 previous_steps_str에 포함되어 있으므로 다시 보여주지 않음. 대신 강력한 경고 메시지를 줌.
                    feedback_str = f"""[WARNING: The previous step (Step {len(state["step_texts"])}) failed verification multiple times but was retained. Proceed with caution, following the Guidance.]
- Previous Error Diagnosis: {diag}
- Guidance for this step: {guid}"""

                else:
                    # 에러가 나서 재시도 하는 경우 (가장 중요)
#                     feedback_str = f"""[Previous Step Failed]
# - Error Type: {err}
# - Diagnosis: {diag}
# - Guidance: {guid}"""

                    # Error Type 제거
                    if failed_text:
                        feedback_str = f"""[Previous Incorrect Attempt at Step {len(state['step_texts']) + 1}]
{failed_text}

[Feedback on Previous Attempt. You must follow the Guidance carefully.]
- Diagnosis: {diag}
- Guidance: {guid}"""
                    else: # 이 경우는 발생하지 않아야 함.
                        feedback_str = f"""[Previous Step Failed. You must follow the Guidance carefully.]
- Diagnosis: {diag}
- Guidance: {guid}"""

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
            else: # fixed system prompt로 바꾼 상태!
                messages = [{"role": "system", "content": generate_single_step_fixed_system_prompt}, {"role": "user", "content": prompt_user}]
            gen_template_kwargs = {"add_generation_prompt": True, "tokenize": False}
            if is_qwen3_generator:
                gen_template_kwargs["enable_thinking"] = qwen3_thinking_mode == "on"
            gen_prompts.append(gen_tokenizer.apply_chat_template(messages, **gen_template_kwargs))

        # Generator 실행 (Batch)
        # stop_token_ids 후보들 중 None이 아닌 정수값만 추출
        actual_stop_tokens = build_stop_token_ids(gen_tokenizer)

        gen_sampling_params = build_generator_sampling_params(
            is_qwen3_generator=is_qwen3_generator,
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
            eval_outputs = eval_llm.generate(eval_prompts, SamplingParams(temperature=0.0, max_tokens=evaluator_max_tokens, stop_token_ids=build_stop_token_ids(eval_tokenizer)), lora_request=LoRARequest("evaluator_adapter", 1, adapter_path), use_tqdm=False)
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
                # 조건 2: error_type이 Correct (아래 Logic Update에서 확인)
                # 조건 3: guidance에 [END_OF_REASONING] 포함
                guidance_text = parsed_eval.get("guidance", "")
                has_stop_token = "[END_OF_REASONING]" in guidance_text

                # Logic Update
                if is_correct_error_type(err_type):
                    attempt_record["result"] = "Accepted"
                    state["current_step_log"]["attempts"].append(attempt_record)
                    state["current_step_log"]["status"] = "Completed"
                    state["logs"]["steps_history"].append(state["current_step_log"])
                    state["step_texts"].append(state["temp_gen_text"])
                    state["feedback_list"].append(parsed_eval)
                    state["current_retry"] = 0
                    state["last_feedback"] = deepcopy(parsed_eval)
                    state["last_feedback"].pop("failed_text", None)
                    
                    # [변경 3] 엄격한 종료 조건 적용 (3가지 모두 만족 시에만 finished)
                    if has_answer_tag and has_stop_token:
                        state["finished"] = True
                        state["current_step_log"]["status"] = "Finished (Verified Answer)"
                    else:
                        # Correct지만, ANSWER 태그가 없거나 END 토큰이 없으면 계속 진행
                        state["finished"] = False

                elif (
                    retriever is not None
                    and is_missing_evidence_error(err_type)
                    and state["retrieval_count"] < retrieval_max_per_sample
                ):
                    query = str(guidance_text or "").strip()
                    if not query:
                        query = str(parsed_eval.get("diagnosis", "")).strip()
                    retrieved = retriever.retrieve(
                        query,
                        top_k=retrieval_top_k,
                        existing_passages=state["passages"],
                    )
                    appended_passages = [
                        item["passage_text"]
                        for item in retrieved
                        if item.get("passage_text")
                    ]
                    state["passages"].extend(appended_passages)
                    state["retrieval_count"] += 1
                    total_stats["retrieval_calls"] += 1
                    total_stats["retrieval_passages_appended"] += len(appended_passages)

                    retrieval_record = {
                        "trigger_step_num": state["current_step_log"]["step_num"],
                        "retry_index": state["current_retry"],
                        "error_type": err_type,
                        "query": query,
                        "top_k": retrieval_top_k,
                        "appended_count": len(appended_passages),
                        "results": retrieved,
                    }
                    state["logs"]["retrieval_history"].append(retrieval_record)
                    state["logs"]["meta_data"]["retrieved_passages"] = state["passages"]

                    attempt_record["result"] = "Accepted (Missing Evidence Retrieval)"
                    attempt_record["retrieval"] = retrieval_record
                    state["current_step_log"]["attempts"].append(attempt_record)
                    state["current_step_log"]["status"] = "Completed (Missing Evidence Retrieval)"
                    state["logs"]["steps_history"].append(state["current_step_log"])
                    state["step_texts"].append(state["temp_gen_text"])
                    state["feedback_list"].append(parsed_eval)
                    state["current_retry"] = 0
                    state["last_feedback"] = {
                        "error_type": "Correct (No Error)",
                        "diagnosis": (
                            f"{parsed_eval.get('diagnosis', '')} "
                            f"Retrieved {len(appended_passages)} passage(s) using the Missing Evidence query."
                        ).strip(),
                        "guidance": "Use the updated retrieved passages to generate the immediate next reasoning step.",
                    }

                    print(
                        f"🔎 Missing Evidence retrieval | id={state['id']} "
                        f"query={query!r} appended={len(appended_passages)}"
                    )

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
                    "retrieval_count": state.get("retrieval_count", 0),
                    "retrieval_history": state["logs"].get("retrieval_history", []),
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
        "--generator_quantization",
        type=str,
        choices=["none", "bitsandbytes", "bnb4", "4bit"],
        default="none",
        help=(
            "Generator quantization mode. Use bitsandbytes/bnb4/4bit "
            "for vLLM online 4bit."
        ),
    )
    parser.add_argument(
        "--generator_tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for the generator. Default keeps the original TP=4 behavior.",
    )
    parser.add_argument(
        "--generator_max_model_len",
        type=int,
        default=10000,
        help="Maximum model length for the generator vLLM engine.",
    )
    parser.add_argument(
        "--qwen3_thinking_mode",
        type=str,
        choices=["on", "off"],
        default="off",
        help="Only used when --generator_model is a Qwen3-family model.",
    )
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--evaluator_max_tokens",
        type=int,
        default=256,
        help="Maximum evaluator completion tokens.",
    )
    parser.add_argument(
        "--evaluator_quantization",
        type=str,
        choices=["none", "bitsandbytes"],
        default="none",
        help="Evaluator base model quantization. Use none by default; bitsandbytes can corrupt Qwen3 LoRA outputs in vLLM.",
    )
    parser.add_argument(
        "--evaluator_max_model_len",
        type=int,
        default=8000,
        help="Maximum model length for the evaluator vLLM engine.",
    )
    parser.add_argument(
        "--ablation_output_root",
        type=str,
        default="/workspace/daeyong/inference_results/dev_kg_correct_1ksample_with_noises_10_3_errortype_ablation",
        help="Output root used when ablation adapter drop_key is detected.",
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
    parser.add_argument(
        "--disable_missing_evidence_retrieval",
        action="store_true",
        help="Disable BM25 retrieval on evaluator error_type=Missing Evidence.",
    )
    parser.add_argument(
        "--retrieval_top_k",
        type=int,
        default=3,
        help="Number of BM25 passages to append when Missing Evidence is detected.",
    )
    parser.add_argument(
        "--retrieval_max_per_sample",
        type=int,
        default=1,
        help="Maximum number of Missing Evidence retrieval calls per sample.",
    )
    parser.add_argument(
        "--retrieval_corpus_path",
        type=str,
        default=None,
        help="Override BM25 corpus path. Defaults to benchmarks/{dataset}_corpus.json.",
    )
    args = parser.parse_args()

    generator_model_lower = args.generator_model.lower()
    is_qwen3_generator = "qwen3" in generator_model_lower
    is_qwen36_27b = "qwen3.6" in generator_model_lower and "27b" in generator_model_lower
    is_gpt_oss_20b = "oss" in generator_model_lower and "-20b" in generator_model_lower
    enable_missing_evidence_retrieval = not args.disable_missing_evidence_retrieval
    retrieval_corpus_path = args.retrieval_corpus_path or (
        f"/workspace/daeyong/benchmarks/{args.dataset}_corpus.json"
    )
    if enable_missing_evidence_retrieval and not os.path.exists(retrieval_corpus_path):
        raise FileNotFoundError(f"BM25 retrieval corpus not found: {retrieval_corpus_path}")

    # Generator 설정 (파일 저장명 용도)
    if "llama" in generator_model_lower and "-8b" in generator_model_lower:
        model_name = "llama8b"
    elif "llama" in generator_model_lower and "-70b" in generator_model_lower:
        model_name = "llama70b"
    elif is_qwen36_27b:
        model_name = "qwen36_27b"
    elif "qwen" in generator_model_lower and "-27b" in generator_model_lower:
        model_name = "qwen27b"
    elif "qwen" in generator_model_lower and "-8b" in generator_model_lower:
        model_name = "qwen8b"
    elif "qwen" in generator_model_lower and "-4b" in generator_model_lower:
        model_name = "qwen4b"
    elif "qwen" in generator_model_lower and "-7b" in generator_model_lower:
        model_name = "qwen7b"
    elif "qwen" in generator_model_lower and "-14b" in generator_model_lower:
        model_name = "qwen14b"
    elif "gemma-4" in generator_model_lower and "31b" in generator_model_lower:
        model_name = "gemma4_31b"
    elif "gemma" in generator_model_lower and "12b" in generator_model_lower:
        model_name = "gemma12b"
    elif "gemma" in generator_model_lower:
        model_name = "gemma"
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

    drop_key = get_drop_key_from_adapter_path(adapter_path)
    evaluator_system_prompt = select_evaluator_system_prompt(adapter_path)

    # -------------------------------------------------------------------------
    # 1. vLLM 모델 로드 (GPU 메모리 분할)
    # -------------------------------------------------------------------------
    # 주의: 하나의 GPU에 2개의 LLM 엔진을 올리려면 gpu_memory_utilization 합이 1.0 미만이어야 합니다.
    # Gemma 4 31B는 vLLM 0.19의 CUDA graph/KV cache 메모리 산정상 0.4가
    # max_model_len=8000에서 너무 타이트하여 generator만 0.55로 둡니다.
    
    gen_llm, gen_tokenizer = load_generator_model(
        args.generator_model, 
        gpu_memory_utilization=0.5,
        generator_quantization=args.generator_quantization,
        tensor_parallel_size=args.generator_tensor_parallel_size,
        max_model_len=args.generator_max_model_len,
    )

    actual_stop_tokens = build_stop_token_ids(gen_tokenizer)
    generator_sampling_params = build_generator_sampling_params(
        is_qwen3_generator=is_qwen3_generator,
        qwen3_thinking_mode=args.qwen3_thinking_mode,
        stop_token_ids=actual_stop_tokens,
    )
    print(
        f"🧠 Generator config | model={args.generator_model} "
        f"is_qwen3_generator={is_qwen3_generator} qwen3_thinking_mode={args.qwen3_thinking_mode} "
        f"generator_quantization={args.generator_quantization} "
        f"generator_tensor_parallel_size={args.generator_tensor_parallel_size} "
        f"generator_max_model_len={args.generator_max_model_len} "
        f"evaluator_max_model_len={args.evaluator_max_model_len}"
    )
    print(f"🎛️ Generator SamplingParams: {generator_sampling_params}")
    print(f"🧪 Evaluator thinking disabled: {is_qwen3_8b_evaluator}")
    print(f"🧪 Skip missing assistantfinal for oss20b: {is_gpt_oss_20b}")
    if drop_key:
        print(f"🧪 Evaluator prompt mode: ablation ({drop_key})")
    else:
        print("🧪 Evaluator prompt mode: default (premature_attribution_missing_evidence)")
    print(
        "🔎 Missing Evidence retrieval: "
        f"enabled={enable_missing_evidence_retrieval} "
        f"top_k={args.retrieval_top_k} "
        f"max_per_sample={args.retrieval_max_per_sample} "
        f"corpus={retrieval_corpus_path}"
    )
    
    # [수정 1 반영] 변수 사용
    eval_llm, eval_tokenizer = load_finetuned_evaluator(
        base_model_id, 
        adapter_path, 
        gpu_memory_utilization=0.35,
        evaluator_quantization=args.evaluator_quantization,
        max_model_len=args.evaluator_max_model_len,
    )
    
    # -------------------------------------------------------------------------
    # 데이터셋 로드
    # -------------------------------------------------------------------------
    # df = pd.read_csv(f"/workspace/daeyong/benchmarks/{args.dataset}_dev.csv")

    df = pd.read_csv(f"/workspace/daeyong/benchmarks/{args.dataset}_dev_kg_correct.csv").sample(n=200, random_state=42)[:100] # 임시로 100개만!
    # if args.dataset == "2wiki":
    #     df = pd.read_csv("/workspace/daeyong/benchmarks/2wiki_20k_sample_yes.csv").sample(n=2000, random_state=42)
    # elif args.dataset == "hotpotqa":
    #     df = pd.read_csv("/workspace/daeyong/benchmarks/hotpotqa_20k_sample_yes.csv").sample(n=2000, random_state=42)
    # elif args.dataset == "musique":
    #     df = pd.read_csv("/workspace/daeyong/benchmarks/musique_yes.csv").sample(n=2000, random_state=42)
    #     # df = pd.read_json("/workspace/daeyong/benchmarks/musique_confusing_entities_filtered.json")
    feedback_model_clean = os.path.basename(os.path.normpath(adapter_path)).replace("-", "_")

    # 2. 저장할 디렉토리 경로 생성
    if drop_key:
        output_root = args.ablation_output_root
        output_name = f"{drop_key}_bm25_retrieval" if enable_missing_evidence_retrieval else drop_key
        output_dir = os.path.join(output_root, output_name)
    else:
        output_dir = f"/workspace/daeyong/inference_results/dev_kg_correct_1ksample_no_premature_conclusion_{args.max_steps}_{args.max_retries}_{feedback_model_clean}"
        if enable_missing_evidence_retrieval:
            output_dir += "_bm25_retrieval"

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
            evaluator_max_tokens=args.evaluator_max_tokens,
            batch_size=256,
            is_qwen3_generator=is_qwen3_generator,
            qwen3_thinking_mode=args.qwen3_thinking_mode,
            disable_evaluator_thinking=is_qwen3_8b_evaluator,
            skip_missing_assistantfinal_for_oss20b=is_gpt_oss_20b,
            enable_missing_evidence_retrieval=enable_missing_evidence_retrieval,
            retrieval_corpus_path=retrieval_corpus_path,
            retrieval_top_k=args.retrieval_top_k,
            retrieval_max_per_sample=args.retrieval_max_per_sample,
        )
        print("✅ All processing complete.")
    else:
        print("✅ Nothing to process (All completed).")
