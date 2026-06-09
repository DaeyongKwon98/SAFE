import pandas as pd
import numpy as np
import argparse
import os
import json
import ast
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

eval_json_schema = {
    "type": "object",
    "properties": {
        "error_type": {"type": "string"},
        "diagnosis": {"type": "string"},
        "guidance": {"type": "string"},
    },
    "required": ["error_type", "diagnosis", "guidance"],
    "additionalProperties": False,
}

# 기존 prompts 모듈이 있다고 가정
from prompts import (
    generate_single_step_system_prompt, 
    generate_single_step_fixed_system_prompt,
    evaluate_system_prompt_premature_attribution
)

# -------------------------------------------------------------------------
# [추가] 정답 강제 추출을 위한 전용 System Prompt
# -------------------------------------------------------------------------
FORCE_ANSWER_SYSTEM_PROMPT = """You are an expert answering agent.
The reasoning process is complete. Your task is to formulate the FINAL ANSWER based on the provided history.

INSTRUCTIONS:
1. Do not generate any new reasoning steps.
2. Directly output the final answer.
3. YOU MUST USE THE FOLLOWING FORMAT:
####ANSWER: your_final_answer_here (Final Answer)"""


@dataclass
class OpenAIResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    raw_response: Any


class OpenAISelfFeedbackClient:
    """Single OpenAI model used for both generation and self-evaluation."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 5,
        retry_sleep: float = 2.0,
        concurrency: int = 8,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OpenAI Python SDK is required. Install it in the runtime "
                "environment, for example: pip install openai"
            ) from exc

        api_key = os.getenv(api_key_env) if api_key_env else None
        if api_key_env and not api_key:
            raise ValueError(
                f"OpenAI API key not found. Set the {api_key_env} environment variable."
            )

        client_kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.concurrency = max(1, concurrency)

    def _chat_completion(
        self,
        messages: list[dict],
        max_completion_tokens: int,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> OpenAIResult:
        params = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if response_format is not None:
            params["response_format"] = response_format

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(**params)
                message = completion.choices[0].message
                text = (message.content or "").strip()
                usage = getattr(completion, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                if total_tokens == 0:
                    total_tokens = prompt_tokens + output_tokens
                return OpenAIResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    raw_response=completion,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep * (2 ** attempt))

        raise RuntimeError(f"OpenAI call failed after retries: {last_error}") from last_error

    def _run_batch(
        self,
        messages_batch: list[list[dict]],
        max_completion_tokens: int,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> list[OpenAIResult]:
        if not messages_batch:
            return []

        results: list[Optional[OpenAIResult]] = [None] * len(messages_batch)
        max_workers = min(self.concurrency, len(messages_batch))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self._chat_completion,
                    messages,
                    max_completion_tokens,
                    temperature,
                    response_format,
                ): idx
                for idx, messages in enumerate(messages_batch)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()

        return [result for result in results if result is not None]

    def generate_batch(
        self,
        messages_batch: list[list[dict]],
        max_completion_tokens: int = 256,
        temperature: Optional[float] = None,
    ) -> list[OpenAIResult]:
        return self._run_batch(
            messages_batch=messages_batch,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=None,
        )

    def evaluate_batch(
        self,
        messages_batch: list[list[dict]],
        max_completion_tokens: int = 256,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> list[OpenAIResult]:
        return self._run_batch(
            messages_batch=messages_batch,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
        )


def load_openai_model(
    model_id: str,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: Optional[str] = None,
    timeout: float = 120.0,
    max_retries: int = 5,
    retry_sleep: float = 2.0,
    concurrency: int = 8,
):
    """
    Reasoning과 Self-Feedback을 모두 수행할 단일 GPT API 모델을 초기화합니다.
    """
    print(f"Loading Single Model (OpenAI API)... Model: '{model_id}'")
    client = OpenAISelfFeedbackClient(
        model=model_id,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        concurrency=concurrency,
    )
    print("✅ OpenAI API model initialized successfully.")
    return client

def parse_eval_response(response_text: str, expect_structured: bool = False) -> dict:
    """
    Evaluator 응답을 파싱하여 diagnosis, guidance를 추출합니다.
    JSON 뒤에 부가 설명(Reasoning)이 붙어도 무시하고 앞쪽의 JSON 객체만 추출합니다.
    """
    # 기본 결과 구조 (실패 시 대비)
    fallback_result = {
        "error_type": "Parsing Error",
        "diagnosis": "Initialization failed",
        "guidance": "Check model output format.",
        "raw_response": response_text,
        "parse_mode": "failed"
    }

    if not response_text or not response_text.strip():
        # print("Empty response received from evaluator.") # 로그가 너무 많으면 주석 처리
        fallback_result["diagnosis"] = "Parsing failed (empty_response)"
        return fallback_result

    text = response_text.strip()

    # 1. Markdown 코드 블록 제거
    markdown_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if markdown_match:
        text = markdown_match.group(1).strip()

    # 2. JSON 시작점 '{' 찾기
    start_idx = text.find('{')

    # ---------------------------------------------------------
    # Strategy 1: json.JSONDecoder().raw_decode (가장 권장)
    # ---------------------------------------------------------
    # 설명: JSON 규격에 맞는 문자열이 앞에 있고, 뒤에 잡다한 텍스트가 있을 때 유효함.
    if start_idx != -1:
        try:
            decoder = json.JSONDecoder()
            # raw_decode는 (파싱된_객체, 끝난_인덱스) 튜플을 반환합니다.
            # 뒤에 "However..." 같은 텍스트가 있어도 에러 없이 JSON만 가져옵니다.
            parsed, _ = decoder.raw_decode(text[start_idx:])
            
            if isinstance(parsed, dict):
                mode = "structured_json" if expect_structured else "raw_json"
                return _normalize_result(parsed, response_text, parse_mode=mode)
                
        except json.JSONDecodeError:
            # JSON 문법이 깨졌거나(싱글 쿼트 등), 중간에 잘렸을 경우 다음 전략으로 넘어감
            pass

        # ---------------------------------------------------------
        # Strategy 2: ast.literal_eval (싱글 쿼트 대응)
        # ---------------------------------------------------------
        # 설명: LLM이 "key": 'value' 처럼 파이썬 딕셔너리 문법(싱글 쿼트)을 썼을 때 유효함.
        # ast는 뒤에 잡다한 텍스트가 있으면 에러가 나므로, 마지막 '}'를 찾아 잘라줘야 함.
        try:
            last_brace = text.rfind('}')
            if last_brace != -1:
                # '{' 부터 마지막 '}' 까지만 잘라서 시도
                candidate = text[start_idx : last_brace+1]
                parsed = ast.literal_eval(candidate)
                
                if isinstance(parsed, dict):
                    return _normalize_result(parsed, response_text, parse_mode="raw_json")
        except (ValueError, SyntaxError):
            pass

        # ---------------------------------------------------------
        # Strategy 3: Truncated JSON Recovery (잘림 대응) - 선택 사항
        # ---------------------------------------------------------
        # 혹시 토큰 제한으로 JSON이 닫히지 않았을 경우를 대비한 간단한 복구 시도
        # (Reasoning이 뒤에 붙는 문제에서는 보통 필요 없으나 안전장치로 둠)
        patterns = ['}', '"}', ']}', '"]}', '}}', '"}}']
        for pattern in patterns:
            try:
                # 잘린 부분에 강제로 닫는 괄호를 붙여서 시도
                # 여기서는 raw_decode가 아니라 loads를 씀 (이미 잘린 걸 가정하므로)
                candidate = text[start_idx:] + pattern
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return _normalize_result(parsed, response_text, parse_mode="raw_json")
            except json.JSONDecodeError:
                continue

    # ---------------------------------------------------------
    # Strategy 4: Non-JSON header fallback
    # ---------------------------------------------------------
    header_fallback = _parse_header_style_evaluation(text, response_text)
    if header_fallback is not None:
        return header_fallback

    # 모든 시도 실패
    if start_idx == -1:
        fallback_result["diagnosis"] = "Parsing failed (no_json_header_pattern_missing)"
    else:
        fallback_result["diagnosis"] = f"Parsing failed (json_decode_failed). Raw start: {text[start_idx:start_idx+50]}..."
    return fallback_result

def _normalize_result(parsed_dict: dict, raw_text: str, parse_mode: str = "raw_json") -> dict:
    """결과 딕셔너리의 키를 보정하고 raw_response를 추가하는 헬퍼 함수"""
    return {
        "error_type": parsed_dict.get("error_type", "Unknown"),
        "diagnosis": parsed_dict.get("diagnosis", "No diagnosis provided."),
        "guidance": parsed_dict.get("guidance", "No guidance provided."),
        "raw_response": raw_text,
        "parse_mode": parse_mode
    }

def _extract_section_block(text: str, section_name: str) -> str:
    """## Section 헤더 블록을 추출합니다."""
    pattern = rf"(?is)(?:^|\n)\s*#+\s*{re.escape(section_name)}\s*:?\s*\n(.*?)(?=\n\s*#+\s*(?:Error Type|Diagnosis|Guidance)\b|\Z)"
    match = re.search(pattern, text)
    if not match:
        return ""
    return match.group(1).strip()

def _parse_error_type_from_text(text: str) -> str:
    """헤더형 텍스트에서 error_type 문구를 추출합니다."""
    candidates = []

    block = _extract_section_block(text, "Error Type")
    if block:
        candidates.append(block.splitlines()[0].strip())

    m = re.search(r"(?is)\bthe\s+error\s+type\s+is\s+`([^`]+)`", text)
    if m:
        candidates.append(m.group(1).strip())

    m = re.search(r"(?im)^\s*error[_\s-]*type\s*:\s*(.+)$", text)
    if m:
        candidates.append(m.group(1).strip())

    for cand in candidates:
        if not cand:
            continue
        backtick = re.search(r"`([^`]+)`", cand)
        if backtick:
            cand = backtick.group(1).strip()
        cand = re.sub(r"(?is)^\s*the\s+error\s+type\s+is\s*", "", cand).strip()
        cand = cand.strip(" .:-")
        if cand:
            return cand
    return ""

def _parse_header_style_evaluation(text: str, raw_text: str) -> dict | None:
    """
    JSON이 아닌 헤더형 출력(## Error Type/Diagnosis/Guidance)에서 필드를 복구합니다.
    """
    error_type = _parse_error_type_from_text(text)
    diagnosis = _extract_section_block(text, "Diagnosis")
    guidance = _extract_section_block(text, "Guidance")

    if not diagnosis:
        m = re.search(r"(?is)\bdiagnosis\s*:\s*(.+?)(?=\n\s*(?:guidance\s*:|error[_\s-]*type\s*:)|\Z)", text)
        if m:
            diagnosis = m.group(1).strip()

    if not guidance:
        m = re.search(r"(?is)\bguidance\s*:\s*(.+?)(?=\n\s*(?:diagnosis\s*:|error[_\s-]*type\s*:)|\Z)", text)
        if m:
            guidance = m.group(1).strip()

    # 헤더형 복구는 최소한 error_type + (diagnosis 또는 guidance) 확보 시에만 성공으로 간주
    if error_type and (diagnosis or guidance):
        parsed = {
            "error_type": error_type,
            "diagnosis": diagnosis or "No diagnosis provided.",
            "guidance": guidance or "No guidance provided."
        }
        return _normalize_result(parsed, raw_text, parse_mode="header_fallback")
    return None

def is_correct_error_type(err_type: str) -> bool:
    """정답 판정은 정확히 'Correct (No Error)' 문자열과 일치할 때만 허용합니다."""
    return str(err_type).strip() == "Correct (No Error)"

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def _sanitize_feedback_text_for_prompt(text: str) -> str:
    return _normalize_whitespace(text)

def _build_quality_fallback(raw_response: str, reason: str) -> dict:
    return {
        "error_type": "Parsing Error",
        "diagnosis": "Evaluator output invalid; schema or quality checks failed.",
        "guidance": "Regenerate strict JSON with one concise diagnosis and guidance.",
        "raw_response": raw_response,
        "parse_mode": "quality_gate_failed",
        "quality_gate": "failed",
        "quality_gate_reason": reason,
    }

def validate_evaluation(parsed_eval: dict, raw_response: str) -> dict:
    err_type = _normalize_whitespace(parsed_eval.get("error_type", ""))
    diagnosis = _normalize_whitespace(parsed_eval.get("diagnosis", ""))
    guidance = _normalize_whitespace(parsed_eval.get("guidance", ""))

    if err_type == "Parsing Error":
        return _build_quality_fallback(raw_response, "parse_error")

    return {
        "error_type": err_type,
        "diagnosis": diagnosis,
        "guidance": guidance,
        "raw_response": parsed_eval.get("raw_response", raw_response),
        "parse_mode": parsed_eval.get("parse_mode", "unknown"),
        "quality_gate": "passed",
    }

def normalize_step_text(gen_text: str, expected_step_num: int) -> str:
    """
    생성된 step 텍스트를 `Step {expected_step_num}:` 형식으로 정규화합니다.
    - 이미 기대 인덱스면 유지
    - 다른 인덱스면 기대 인덱스로 교체
    - 형식이 깨진 Step 표기도 가능한 범위에서 복구
    """
    expected_start = f"Step {expected_step_num}:"
    text = str(gen_text).strip()

    if not text:
        return expected_start

    first_line, sep, remaining = text.partition("\n")
    first_line = first_line.strip()

    # Case A: 이미 올바른 prefix
    if first_line.startswith(expected_start):
        return text

    # Case B: Step N: ... 형태 (N != K 포함)
    strict_step_match = re.match(r"^Step\s+(\d+)\s*:\s*(.*)$", first_line)
    if strict_step_match:
        suffix = strict_step_match.group(2).strip()
        normalized_first = expected_start if not suffix else f"{expected_start} {suffix}"
        return normalized_first + (f"\n{remaining}" if sep else "")

    # Case C: Step으로 시작하지만 형식이 깨진 경우 복구
    if re.match(r"(?i)^step", first_line):
        tail = re.sub(r"(?i)^step", "", first_line, count=1).strip()

        if ":" in tail:
            tail = tail.split(":", 1)[1].strip()
        else:
            tail = re.sub(
                r"^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b[\s\.\-\)]*",
                "",
                tail,
                flags=re.IGNORECASE
            ).strip()

        normalized_first = expected_start if not tail else f"{expected_start} {tail}"
        return normalized_first + (f"\n{remaining}" if sep else "")

    # Case D: Step prefix가 없으면 앞에 부착
    return f"{expected_start} {text}".strip()


def run_dynamic_batch_inference_self_feedback(
    df: pd.DataFrame,
    client: OpenAISelfFeedbackClient,
    result_file_path: str,
    log_file_path: str,
    stats_file_path: str,
    progress_file_path: str = "",
    max_steps: int = 10,
    max_retries: int = 3,
    batch_size: int = 32,
    log_every: int = 10,
    model_id: str = "",
    max_completion_tokens: int = 256,
    eval_max_completion_tokens: int = 256,
    temperature: Optional[float] = None,
    eval_temperature: Optional[float] = None,
    eval_response_format: str = "json_schema",
):
    # 1. 초기화
    pending_queue = deque()
    for _, row in df.iterrows():
        pending_queue.append(row)

    active_states = [] 
    total_stats = {"generator_calls": 0, "evaluator_calls": 0, "total_tokens": 0, "completed_count": 0}
    iteration_idx = 0
    
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

    def append_jsonl_file(file_path: str, payload: dict):
        if not file_path:
            return
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

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
            "current_step_log": None
        }

    pbar = tqdm(total=len(df), desc="Self-Feedback GPT API Processing")
    
    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------   
    while pending_queue or active_states:
        iteration_idx += 1
        
        # [Refill]
        while len(active_states) < batch_size and pending_queue:
            active_states.append(create_new_state(pending_queue.popleft()))
        if not active_states: break

        active_count_start = len(active_states)
        pending_count_start = len(pending_queue)
        accepted_this_iter = 0
        rejected_this_iter = 0
        quality_gate_failed_this_iter = 0

        # =========================================================================
        # PHASE 1: GENERATION
        # =========================================================================
        gen_messages_batch = []
        gen_indices = []
        gen_force_answer_modes = []
        
        for idx, state in enumerate(active_states):
            # 상태 관리
            if state["current_retry"] == 0:
                state["current_step_log"] = {
                    "step_num": len(state["step_texts"]) + 1, "status": "In Progress", "attempts": []
                }
            
            passages_str = '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(state["passages"])])
            previous_steps_str = '\n'.join(state["step_texts"]) if state["step_texts"] else "(No previous steps.)"
            
            feedback_str = ""
            force_answer_mode = False

            if not state["last_feedback"]:
                # 첫 스텝이거나, 이전 스텝이 성공해서 그냥 넘어온 경우
                feedback_str = "Status: N/A (First attempt at this step)"
            else:
                err = state["last_feedback"].get("error_type", "Unknown")
                diag = _sanitize_feedback_text_for_prompt(
                    state["last_feedback"].get("diagnosis", "No diagnosis provided.")
                ) or "No diagnosis provided."
                guid = _sanitize_feedback_text_for_prompt(
                    state["last_feedback"].get("guidance", "Proceed logically with one concise grounded step.")
                ) or "Proceed logically with one concise grounded step."
                failed_text = state["last_feedback"].get("failed_text", None)
                
                # 강제 정답 모드 확인
                cond_format = "####ANSWER:" in guid
                cond_finish = "[END_OF_REASONING]" in guid # 두 번째 코드의 개선된 종료 시그널 유지

                if cond_format or cond_finish:
                    force_answer_mode = True
                
                # [프롬프트 통일] 첫 번째 코드의 피드백 포맷 적용
                if err == "Correct (No Error)":
                    # Error Type 제거
                    feedback_str = f"""[Previous Step Was Correct. You must follow the Guidance for the next step.]
- Diagnosis: {diag}
- Guidance: {guid}"""

                elif state["current_retry"] == 0:
                    # failed_text는 이미 previous_steps_str에 포함되어 있으므로 다시 보여주지 않음. 대신 강력한 경고 메시지.
                    feedback_str = f"""[WARNING: The previous step (Step {len(state["step_texts"])}) failed verification multiple times but was retained. Proceed with caution, following the Guidance.]
- Previous Error Diagnosis: {diag}
- Guidance for this step: {guid}"""

                else:
                    # 에러가 나서 재시도 하는 경우 (Error Type 제거 및 failed_text 포함)
                    if failed_text:
                        feedback_str = f"""[Previous Incorrect Attempt at Step {len(state['step_texts']) + 1}]
{failed_text}

[Feedback on Previous Attempt. You must follow the Guidance carefully.]
- Diagnosis: {diag}
- Guidance: {guid}"""
                    else:
                        feedback_str = f"""[Previous Step Failed. You must follow the Guidance carefully.]
- Diagnosis: {diag}
- Guidance: {guid}"""

            # 프롬프트 조립
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
                system_prompt = FORCE_ANSWER_SYSTEM_PROMPT
            else:
                prompt_user = f"""Question:
{state['question']}

Retrieved Passages:
{passages_str}

Previous Reasoning Steps:
{previous_steps_str}

Feedback:
{feedback_str}

Generate next step (start with `Step {len(state['step_texts']) + 1}:`)"""
                system_prompt = generate_single_step_fixed_system_prompt

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_user}]
            gen_messages_batch.append(messages)
            gen_indices.append(idx)
            gen_force_answer_modes.append(force_answer_mode)

        # OpenAI API Batch Run (Generator)
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

                # Cleaning
                for marker in ["<start_of_turn>", "User:", "## Question"]:
                    if marker in gen_text: gen_text = gen_text.split(marker)[0].strip()
                
                # Step 형식 보정 (Force mode가 아닐 때만)
                if not gen_force_answer_modes[i]:
                    expected_step_num = len(state["step_texts"]) + 1
                    gen_text = normalize_step_text(gen_text, expected_step_num)
                
                state["temp_gen_text"] = gen_text

        # =========================================================================
        # PHASE 2: EVALUATION (Self-Correction)
        # =========================================================================
        eval_messages_batch = []
        eval_indices = []

        for idx, state in enumerate(active_states):
            # 문자열로 저장된 경우 안전하게 변환 (첫 번째 코드 로직)
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
            
            # [프롬프트 통일] 모델(Qwen 등)에 따른 분기문 삭제, 첫 번째 코드의 프롬프트 100% 동일 적용
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
            
            messages = [{"role": "system", "content": evaluate_system_prompt_premature_attribution}, {"role": "user", "content": user_content}]
            eval_messages_batch.append(messages)
            eval_indices.append(idx)

        if eval_messages_batch:
            response_format = None
            if eval_response_format == "json_schema":
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "self_feedback_evaluation",
                        "strict": True,
                        "schema": eval_json_schema,
                    },
                }
            elif eval_response_format == "json_object":
                response_format = {"type": "json_object"}

            eval_outputs = client.evaluate_batch(
                eval_messages_batch,
                max_completion_tokens=eval_max_completion_tokens,
                temperature=eval_temperature,
                response_format=response_format,
            )
            total_stats["evaluator_calls"] += len(eval_outputs)

            for i, output in enumerate(eval_outputs):
                state_idx = eval_indices[i]
                state = active_states[state_idx]
                
                raw_eval = output.text.strip()
                total_stats["total_tokens"] += output.total_tokens
                
                # JSON 결과 파싱
                parsed_eval_raw = parse_eval_response(raw_eval, expect_structured=True)
                parsed_eval = validate_evaluation(parsed_eval_raw, raw_eval)
                
                attempt_record = {
                    "retry_index": state["current_retry"], 
                    "generated_text": state["temp_gen_text"], 
                    "evaluation": parsed_eval, 
                    "result": "Pending"
                }
                
                err_type = parsed_eval.get("error_type", "Unknown")
                quality_gate_passed = parsed_eval.get("quality_gate") == "passed"
                if not quality_gate_passed:
                    quality_gate_failed_this_iter += 1
                
                # [조건 통일] 첫 번째 코드의 조건 그대로 이식
                has_answer_tag = "####ANSWER" in state["temp_gen_text"]
                guidance_text = parsed_eval.get("guidance", "")
                has_stop_token = "[END_OF_REASONING]" in guidance_text

                # Logic Update
                if is_correct_error_type(err_type) and quality_gate_passed:
                    accepted_this_iter += 1
                    attempt_record["result"] = "Accepted"
                    state["current_step_log"]["attempts"].append(attempt_record)
                    state["current_step_log"]["status"] = "Completed"
                    state["logs"]["steps_history"].append(state["current_step_log"])
                    state["step_texts"].append(state["temp_gen_text"])
                    state["feedback_list"].append(parsed_eval)
                    state["current_retry"] = 0
                    
                    # Correct인 경우에도 다음 step 생성을 위해 guidance를 유지
                    state["last_feedback"] = deepcopy(parsed_eval)
                    # 이전 실패 시도에서 남아있을 수 있는 failed_text는 제거
                    state["last_feedback"].pop("failed_text", None)
                    
                    # [로직 복구] 엄격한 종료 조건 (두 조건 모두 만족해야 함)
                    if has_answer_tag and has_stop_token:
                        state["finished"] = True
                        state["current_step_log"]["status"] = "Finished (Verified Answer)"
                    else:
                        state["finished"] = False

                else:
                    rejected_this_iter += 1
                    attempt_record["result"] = "Rejected (Rollback)"
                    state["current_step_log"]["attempts"].append(attempt_record)
                    state["last_feedback"] = parsed_eval
                    
                    # [로직 복구] 직전에 실패한 step 텍스트 저장
                    state["last_feedback"]["failed_text"] = state["temp_gen_text"]
                    
                    state["current_retry"] += 1
                    
                    if state["current_retry"] >= max_retries:
                        state["current_step_log"]["attempts"][-1]["result"] = "Max retries"
                        state["current_step_log"]["status"] = "Max retries"
                        state["logs"]["steps_history"].append(state["current_step_log"])
                        state["step_texts"].append(state["temp_gen_text"])
                        state["feedback_list"].append(parsed_eval)
                        state["current_retry"] = 0
                        
                        # [로직 복구] 가짜 피드백 주입 등 예외 처리 모두 삭제하고, 실패한 피드백만 남김
                        state["last_feedback"] = parsed_eval

        # =========================================================================
        # PHASE 3: CLEANUP & SAVE
        # =========================================================================
        next_active_states = []
        finished_results = []
        finished_logs = []

        for state in active_states:
            # Max Step 체크
            if not state["finished"] and len(state["step_texts"]) >= max_steps:
                state["finished"] = True
                if state["current_step_log"] and state["current_step_log"] not in state["logs"]["steps_history"]:
                     state["logs"]["steps_history"].append(state["current_step_log"])

            if state["finished"]:
                # [포맷 통일] 결과 저장 시 feedback을 문자열 리스트로 캐스팅
                res_obj = {
                    "id": state["id"],
                    "question": state["question"],
                    "context": 'Retrieved Passages:\n' + '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(state['passages'])]),
                    "response": state["step_texts"],
                    "feedback": [f"Feedback for Step {i+1}: {f}" for i, f in enumerate(state['feedback_list'])],
                    "ground_truth": state["ground_truth"]
                }
                finished_results.append(res_obj)
                finished_logs.append(state["logs"])
                pbar.update(1)
            else:
                next_active_states.append(state)

        if finished_results:
            append_to_json_file(result_file_path, finished_results)
            append_to_json_file(log_file_path, finished_logs)
            total_stats["completed_count"] += len(finished_results)
            append_to_json_file(stats_file_path, [deepcopy(total_stats)])

        finished_this_iter = len(finished_results)
        progress_payload = {
            "iteration": iteration_idx,
            "timestamp": int(time.time()),
            "active_count": active_count_start,
            "pending_count": pending_count_start,
            "accepted_this_iter": accepted_this_iter,
            "rejected_this_iter": rejected_this_iter,
            "quality_gate_failed_this_iter": quality_gate_failed_this_iter,
            "finished_this_iter": finished_this_iter,
            "completed_total": total_stats["completed_count"],
            "generator_calls_total": total_stats["generator_calls"],
            "evaluator_calls_total": total_stats["evaluator_calls"],
            "total_tokens": total_stats["total_tokens"],
            "active_count_next": len(next_active_states),
            "pending_count_next": len(pending_queue),
        }
        append_jsonl_file(progress_file_path, progress_payload)

        if log_every > 0 and (iteration_idx % log_every == 0 or finished_this_iter > 0):
            print(
                f"[Iter {iteration_idx}] active={active_count_start} pending={pending_count_start} "
                f"accepted={accepted_this_iter} rejected={rejected_this_iter} "
                f"qg_failed={quality_gate_failed_this_iter} finished={finished_this_iter} "
                f"completed_total={total_stats['completed_count']}"
            )

        active_states = next_active_states

    pbar.close()
    return total_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="gpt-5.4-mini")
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--openai_timeout", type=float, default=120.0)
    parser.add_argument("--openai_max_retries", type=int, default=5)
    parser.add_argument("--openai_retry_sleep", type=float, default=2.0)
    parser.add_argument("--openai_concurrency", type=int, default=8)
    parser.add_argument("--openai_max_completion_tokens", type=int, default=256)
    parser.add_argument("--openai_eval_max_completion_tokens", type=int, default=256)
    parser.add_argument("--openai_temperature", type=float, default=None)
    parser.add_argument("--openai_eval_temperature", type=float, default=None)
    parser.add_argument(
        "--openai_eval_response_format",
        type=str,
        choices=["json_schema", "json_object", "none"],
        default="json_schema",
    )
    args = parser.parse_args()

    base_path = "/workspace/daeyong"
    
    try:
        df = pd.read_csv(f"{base_path}/benchmarks/{args.dataset}_dev_kg_correct.csv")
        print(f"Loaded {len(df)} total rows from {args.dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    model_name = re.sub(r"[^A-Za-z0-9]+", "_", args.model_id).strip("_").lower()

    output_dir = f"{base_path}/inference_results/self_feedback_kg_correct_1k_sample_{args.max_steps}_{args.max_retries}"
    os.makedirs(output_dir, exist_ok=True)
    
    result_file_path = os.path.join(output_dir, f"{model_name}_{args.dataset}_results.json")
    log_file_path = os.path.join(output_dir, f"{model_name}_{args.dataset}_logs.json")
    stats_file_path = os.path.join(output_dir, f"{model_name}_{args.dataset}_stats.json")
    progress_file_path = os.path.join(output_dir, f"{model_name}_{args.dataset}_progress.jsonl")

    processed_ids = set()
    if os.path.exists(result_file_path):
        try:
            with open(result_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                processed_ids = {item['id'] for item in data}
            print(f"Resuming... {len(processed_ids)} already processed.")
        except: pass

    if processed_ids:
        df = df[~df['id'].isin(processed_ids)]
        print(f"Remaining after resume filter: {len(df)}")

    if args.sample_size > 0:
        sample_n = min(args.sample_size, len(df))
        if sample_n < len(df):
            print(f"Sampling {sample_n} rows (seed={args.sample_seed}) from {len(df)} remaining rows.")
        else:
            print(f"Using all remaining rows ({sample_n}); sample_size={args.sample_size}.")
        df = df.sample(n=sample_n, random_state=args.sample_seed)[:100] # 임시로 이렇게 해둠!! 

    if len(df) > 0:
        client = load_openai_model(
            args.model_id,
            api_key_env=args.openai_api_key_env,
            base_url=args.openai_base_url,
            timeout=args.openai_timeout,
            max_retries=args.openai_max_retries,
            retry_sleep=args.openai_retry_sleep,
            concurrency=args.openai_concurrency,
        )

        run_dynamic_batch_inference_self_feedback(
            df=df,
            client=client,
            result_file_path=result_file_path,
            log_file_path=log_file_path,
            stats_file_path=stats_file_path,
            progress_file_path=progress_file_path,
            max_steps=args.max_steps,
            max_retries=args.max_retries,
            batch_size=args.batch_size,
            log_every=args.log_every,
            model_id=args.model_id,
            max_completion_tokens=args.openai_max_completion_tokens,
            eval_max_completion_tokens=args.openai_eval_max_completion_tokens,
            temperature=args.openai_temperature,
            eval_temperature=args.openai_eval_temperature,
            eval_response_format=args.openai_eval_response_format,
        )
    else:
        print("No rows to process after resume/sample filtering.")
