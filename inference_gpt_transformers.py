import pandas as pd
import numpy as np
import argparse
import os
import tiktoken
from tqdm import tqdm
import json
import ast
import torch
import re
from openai import OpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import evaluate_system_prompt, generate_single_step_system_prompt

# LDI lab
ldi_key = ""

client = OpenAI(api_key=ldi_key)

def load_generator_model(model_id: str):
    """
    Reasoning을 수행할 Base Generator 모델을 로드합니다.
    """
    print(f"Generator 모델 로딩 중... Model: '{model_id}'")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Generator 모델 로드 완료.")
    return model, tokenizer



def evaluate_single_step(question: str, context: str, previous_steps: list, current_step: str) -> tuple[dict, int]:
    """
    OpenAI GPT 모델을 사용하여 현재 단계를 평가합니다.
    """
    # --- 1. User Input 구성 ---
    previous_steps_str = '\n'.join(previous_steps)
    
    user_content = f"""### Evaluate the following:

Question: {question}

Retrieved Passages:
{context}

PREVIOUS STEPS:
{previous_steps_str}

STEP TO EVALUATE:
{current_step}
""".strip()

    # --- 3. OpenAI API 호출 ---
    try:
        completion = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": evaluate_system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=4096,
        )

        assistant_response = completion.choices[0].message.content
        
        # 토큰 사용량 계산 (OpenAI 응답 객체에서 제공)
        total_tokens = completion.usage.total_tokens

        # --- 4. JSON 파싱 ---
        try:
            parsed = json.loads(assistant_response)
            
            # 키 확인 및 기본값 설정
            if "error_type" not in parsed: parsed["error_type"] = "Unknown"
            if "diagnosis" not in parsed: parsed["diagnosis"] = "No diagnosis provided."
            if "guidance" not in parsed: parsed["guidance"] = "No guidance provided."
            
            print("--- GPT FEEDBACK ---")
            print(parsed)
            return parsed, total_tokens

        except json.JSONDecodeError:
            print(f"⚠️ JSON 파싱 실패: {assistant_response}")
            return {
                "error_type": "Parsing Error", 
                "diagnosis": f"Raw output: {assistant_response}",
                "guidance": "Check raw output"
            }, total_tokens
            
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            "error_type": "API Error", 
            "diagnosis": str(e),
            "guidance": "An exception occurred during generation."
        }, 0


def generate_single_step(
    query: str, 
    retrieved_passages: list, 
    previous_steps: list, 
    last_feedback: dict, 
    generator_model, 
    generator_tokenizer
) -> tuple[str, int]:
    """
    Generator 모델을 사용하여 다음 한 단계의 추론을 생성합니다.
    """
    passages_str = '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(retrieved_passages)])
    previous_steps_str = '\n'.join(previous_steps)

    # --- Construct Feedback String with Error Type ---
    feedback_str = ""
    if not last_feedback:
        feedback_str = "Status: First Step. No feedback yet."
    else:
        error_type = last_feedback.get("error_type", "Unknown")
        feedback_text = last_feedback.get("feedback", "No feedback provided.")
        is_error = error_type != "Correct (No Error)"
        
        if not is_error:
            feedback_str = f"Status: Correct\nFeedback: {feedback_text}"
        else:
            feedback_str = f"Status: Error Detected\nError Type: {error_type}\nFeedback: {feedback_text}"
            
    prompt_user = f"""Question: {query}

Retrieved Passages:
{passages_str}

Previous Reasoning Steps:
{previous_steps_str}

Feedback on Last Step:
{feedback_str}

Generate next step (start with `Step {len(previous_steps) + 1}:`)
"""

    messages = [
        {"role": "system", "content": generate_single_step_system_prompt},
        {"role": "user", "content": prompt_user}
    ]

    try:
        # Chat Template 적용
        input_ids = generator_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(generator_model.device)

        # [Token Count] Input
        input_token_count = input_ids.shape[1]

        terminators = [generator_tokenizer.eos_token_id]

        # 추가 안전 처리
        eot_id = generator_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None:
            terminators.append(eot_id)

        if generator_tokenizer.pad_token_id is None:
            generator_tokenizer.pad_token_id = generator_tokenizer.eos_token_id

        outputs = generator_model.generate(
            input_ids,
            attention_mask=input_ids.ne(generator_tokenizer.pad_token_id).long(),      
            pad_token_id=generator_tokenizer.pad_token_id,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=False,
        )

        # [Token Count] Total (Input + Output)
        total_tokens = outputs.shape[1]
        
        response = generator_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

        # 출력 정제 (찌꺼기 제거 및 포맷 확인)
        dirty_markers = [
            "<start_of_turn>", "</start_of_turn>",
            "<end_of_turn>", "</end_of_turn>",
            "User:", "## Question"
        ]
        
        for marker in dirty_markers:
            if marker in response:
                response = response.split(marker)[0].strip()

        # 스텝 번호 강제 보정
        expected_start = f"Step {len(previous_steps) + 1}:"
        if not response.startswith(expected_start):
            if not response.startswith("Step"):
                print(f"⚠️ Generated step missing prefix. Adding '{expected_start}'")
                response = f"{expected_start} " + response.lstrip()
            elif response.startswith("Step"):
                    response = response.split('\n')[0]

        return response, total_tokens
        
    except Exception as e:
        print(f"LLM step generation error: {e}")
        return "", 0


def generate_and_evaluate_iteratively(
    query: str, 
    retrieved_passages: list, 
    generator_model,
    generator_tokenizer,
    max_steps: int = 7,
    max_retries: int = 3
) -> tuple[list, list, list, dict]:

    # [통계] 초기화
    stats = {
        "final_step_count": 0,
        "generator_calls": 0,
        "evaluator_calls": 0,
        "total_tokens": 0
    }

    # [수정] 로그 구조를 딕셔너리로 변경하여 질문과 지문 정보를 포함
    full_logs = {
        "meta_data": {
            "question": query,
            "retrieved_passages": retrieved_passages
        },
        "steps_history": [] # 여기에 스텝별 상세 기록이 들어감
    }
    step_texts = []      # "검증된" 스텝들만 저장하는 리스트 (Clean History)
    feedback_list = []   # 각 스텝의 최종 피드백 저장

    # 종료 트리거 문구들 정의 (대소문자 무시 매칭용)
    stop_phrases = [
        "successfully completed",
        "process is completed",
        "do not generate any further steps",
        "you must stop reasoning",
        "you should stop reasoning",
        "stop reasoning now",
        "reasoning is complete",
        "reasoning is finished",
        "answer has already been found"
    ]

    print(f"🚀 Reasoning Start: {query}")

    # 전체 추론 단계 루프 (Step 1 -> Step 2 -> ...)
    while len(step_texts) < max_steps:
        
        current_step_num = len(step_texts) + 1
        current_retry = 0
        last_feedback = None  # 새 스텝 시작 시 피드백 초기화
        step_accepted = False # 현재 스텝 승인 여부
        
        # 현재 스텝을 위한 로그 컨테이너 생성
        current_step_log = {
            "step_num": current_step_num,
            "status": "In Progress",
            "attempts": []
        }
        
        # [Inner Loop] 올바른 스텝이 나올 때까지 재시도
        while current_retry < max_retries:
            
            # 1. 생성 (Generator)
            next_step_text, gen_tokens = generate_single_step(
                query, 
                retrieved_passages, 
                step_texts, 
                last_feedback, 
                generator_model, 
                generator_tokenizer
            )

            # [통계] Generator 호출 및 토큰 수 업데이트
            stats["generator_calls"] += 1
            stats["total_tokens"] += gen_tokens
            
            if not next_step_text:
                print("⚠️ Generator returned empty response.")
                break

            # 문맥 구성 (Evaluator용)
            context_str = 'Retrieved Passages:\n' + '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(retrieved_passages)])
            
            # 2. 평가 (Evaluator - GPT)
            # evaluator_model, evaluator_tokenizer 인자 제거
            evaluation_result, eval_tokens = evaluate_single_step(
                query, 
                context_str, 
                step_texts, 
                next_step_text
            )
   
            # [통계] Evaluator 호출 및 토큰 수 업데이트
            stats["evaluator_calls"] += 1
            stats["total_tokens"] += eval_tokens
            
            error_type = evaluation_result.get("error_type", "Unknown")
            diagnosis_msg = evaluation_result.get("diagnosis", "Unknown")
            guidance_msg = evaluation_result.get("guidance", "Unknown")

            # 시도 기록 저장
            attempt_record = {
                "retry_index": current_retry,
                "generated_text": next_step_text,
                "evaluation": evaluation_result, # error_type, diagnosis, guidance 포함
                "result": "Pending" # 나중에 Accepted/Rejected로 업데이트
            }

            # -------------------------------------------------------
            # 3. 판단 로직 (수정됨)
            # -------------------------------------------------------
            
            # guidance 메시지 소문자 변환 (안전장치)
            guidance_lower = guidance_msg.lower()
            # 종료 시그널 감지
            is_stop_signal = any(phrase.lower() in guidance_lower for phrase in stop_phrases)

            # Case 1: 정답(Correct)인 경우 -> 스텝 저장 O
            if error_type == "Correct (No Error)":
                print(f"✅ Step {current_step_num} Accepted: {error_type}")
                
                attempt_record["result"] = "Accepted"
                current_step_log["attempts"].append(attempt_record)
                current_step_log["status"] = "Completed"
                
                # [승인] 역사에 기록 (Commit)
                step_texts.append(next_step_text)
                feedback_list.append(evaluation_result)
                step_accepted = True
                
                # [즉시 종료 체크] Correct이면서 종료 신호가 있다면 -> 전체 종료
                if is_stop_signal:
                    print(f"🎉 Evaluator signaled completion with 'Correct'.")
                    full_logs["steps_history"].append(current_step_log)
                    stats["final_step_count"] = len(step_texts)
                    return step_texts, feedback_list, full_logs, stats

                break # 재시도 루프 탈출 (다음 스텝으로)
            
            # Case 2: 과한 생각(Overthinking) 또는 중복(Redundancy)이면서 종료 신호가 있는 경우 -> 스텝 저장 X, 전체 종료
            elif error_type in ["Overthinking", "Redundancy"] and is_stop_signal:
                print(f"🛑 Stop Signal Detected on {error_type}. The previous steps were sufficient.")
                
                attempt_record["result"] = "Rejected (Process Finished)"
                current_step_log["attempts"].append(attempt_record)
                current_step_log["status"] = "Finished by Evaluator"
                full_logs["steps_history"].append(current_step_log)
                
                # ★ 중요: 이번 스텝은 저장하지 않고 함수 전체를 종료함
                stats["final_step_count"] = len(step_texts)
                return step_texts, feedback_list, full_logs, stats

            # Case 3: 그 외 오류 (일반적인 틀린 추론) -> 재시도
            else:
                print(f"🔄 Step {current_step_num} Retry ({current_retry+1}/{max_retries}): {error_type} -> Rolling back...")
                
                attempt_record["result"] = "Rejected (Rollback)"
                current_step_log["attempts"].append(attempt_record)
                
                # [거절] 역사에 기록하지 않음 (Rollback)
                last_feedback = evaluation_result
                current_retry += 1
                
        # Inner Loop 종료 후 처리 (최대 재시도 초과 등)
        if not step_accepted:
            print(f"⚠️ Step {current_step_num}: Max retries reached. Forcing progression.")
            current_step_log["status"] = "Max retries"
            
            if next_step_text:
                if current_step_log["attempts"]:
                    current_step_log["attempts"][-1]["result"] = "Max retries"
                step_texts.append(next_step_text)
                feedback_list.append(last_feedback if last_feedback else evaluation_result)
            else:
                current_step_log["status"] = "Failed (Empty)"
                full_logs["steps_history"].append(current_step_log) # 실패했더라도 로그 저장하고 종료
                break 

        # 현재 스텝의 로그를 전체 로그에 추가
        full_logs["steps_history"].append(current_step_log)
        
        # Generator 자체 종료 시그널 확인 ([Reasoning Finished] 토큰 등)
        generator_signal = step_texts and "[Reasoning Finished]" in step_texts[-1]
        
        if generator_signal:
            print(f"🎉 Reasoning Process Finished. (Stopped by Generator)")
            break

    if len(step_texts) == max_steps:
        print(f"⚠️ 최대 단계({max_steps})에 도달하여 추론을 종료합니다.")
  
    # [통계] 최종 스텝 수 업데이트
    stats["final_step_count"] = len(step_texts)

    return step_texts, feedback_list, full_logs, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    # 1. Generator 모델 초기화
    generator_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # generator_model_id = "Qwen/Qwen2.5-7B-Instruct"
    gen_model, gen_tokenizer = load_generator_model(generator_model_id)
    
    # 데이터셋 로드
    if args.dataset == '2wiki':
        df = pd.read_csv("/workspace/daeyong/2wiki_dev.csv")
        df1 = df.sample(n=100, random_state=42)
        df = df.drop(df1.index).sample(n=400, random_state=42)
        df = pd.concat([df, df1])
    elif args.dataset == 'hotpotqa':
        df = pd.read_json("/workspace/daeyong/hotpotqa_validation.json")
        df1 = df.sample(n=100, random_state=42)
        df = df.drop(df1.index).sample(n=400, random_state=42)
        df = pd.concat([df, df1])
    elif args.dataset == 'musique':
        df = pd.read_csv("/workspace/daeyong/musique_dev.csv")
        df1 = df.sample(n=100, random_state=42)
        df = df.drop(df1.index).sample(n=400, random_state=42)
        df = pd.concat([df, df1]) 
    else:
        print(f"❌ 알 수 없는 데이터셋: {args.dataset}")
        exit()

    # --- 파일 경로 정의 및 Resume 로직 시작 ---
    log_file_path = f"/workspace/daeyong/gpt_llama8b_{args.dataset}_logs.json"
    result_file_path = f"/workspace/daeyong/gpt_llama8b_{args.dataset}_results.json"
    stats_file_path = f"/workspace/daeyong/gpt_llama8b_{args.dataset}_stats.json"

    full_logs = []
    response_list = []
    stats_list = []
    processed_ids = set()

    # 1. 결과 파일이 존재하면 로드
    if os.path.exists(result_file_path):
        print(f"🔄 Found existing result file: {result_file_path}")
        try:
            with open(result_file_path, "r", encoding="utf-8") as f:
                response_list = json.load(f)
            
            # 처리된 ID 추출
            for item in response_list:
                if 'id' in item:
                    processed_ids.add(item['id'])
            
            # 로그 및 통계 파일도 로드 (싱크 유지)
            if os.path.exists(log_file_path):
                with open(log_file_path, "r", encoding="utf-8") as f:
                    full_logs = json.load(f)
            if os.path.exists(stats_file_path):
                with open(stats_file_path, "r", encoding="utf-8") as f:
                    stats_list = json.load(f)
            
            print(f"✅ Resuming... {len(processed_ids)} examples already processed.")
            
        except json.JSONDecodeError:
            print("⚠️ Existing file is corrupted or empty. Starting from scratch.")
        except Exception as e:
            print(f"⚠️ Error loading existing files: {e}. Starting from scratch.")
    # --- Resume 로직 끝 ---

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating and Evaluating"):
        current_id = row['id']

        # 이미 처리된 ID라면 건너뛰기
        if current_id in processed_ids:
            continue

        query = row['question']
        context_source = row['retrieved_passages']
        
        if isinstance(context_source, str):
            try:
                retrieved_passages = ast.literal_eval(context_source)
            except (ValueError, SyntaxError):
                retrieved_passages = [context_source]
        elif isinstance(context_source, list):
            retrieved_passages = context_source
        else: retrieved_passages = []
        
        # 3. 단계별 생성 및 평가 동시 진행
        response_steps, feedback_list, logs, step_stats = generate_and_evaluate_iteratively(
            query, 
            retrieved_passages, 
            gen_model, 
            gen_tokenizer,
            max_steps=7
        )
        
        response_list.append({
            "id": current_id,
            "question": query,
            "context": 'Retrieved Passages:\n' + '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(retrieved_passages)]),
            "response": response_steps,
            "feedback": [f"Feedback for Step {i+1}: {f}" for i, f in enumerate(feedback_list)],
            "ground_truth": row.get('answer', 'N/A')
        })
  
        full_logs.append(logs)

        # 통계 정보에 질문 ID 포함
        step_stats["id"] = current_id
        stats_list.append(step_stats)
        
        # 로그 기록 저장 (덮어쓰기 방식으로 저장하여 JSON 유효성 유지)
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(full_logs, f, ensure_ascii=False, indent=2)
  
        # 중간 저장
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump(response_list, f, ensure_ascii=False, indent=2)
   
        # 중간 저장 (통계)
        with open(stats_file_path, "w", encoding="utf-8") as f:
            json.dump(stats_list, f, ensure_ascii=False, indent=2)