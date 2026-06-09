import json
from transformers import AutoTokenizer

def calculate_cumulative_tokens(json_data, model_id="meta-llama/Llama-3.1-8B-Instruct"):
    """
    매 Step마다 (Question + Context + Previous Steps) + (Current Step)의 토큰 수를 누적 계산
    """
    print(f"Loading tokenizer: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        print("토크나이저 로드 실패. 'gpt2'로 대체합니다.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    total_tokens_all_questions = 0
    total_questions = len(json_data)

    print(f"\n총 {total_questions}개의 데이터에 대해 계산을 시작합니다.")

    for idx, item in enumerate(json_data):
        question = item['question']
        context = item['context']
        steps = item['response']
        
        current_q_tokens = 0
        previous_steps_text = []

        # print(f"Question {idx+1} (Steps: {len(steps)})")

        for step_idx, current_step in enumerate(steps):
            # 1. Input Context 구성 (프롬프트 템플릿과 유사하게 구성)
            # Question + Passages + Previous Steps
            prev_steps_str = "\n".join(previous_steps_text) if previous_steps_text else "(No previous steps)"
            
            input_text = f"""Question: {question}

{context}

Previous Reasoning Steps:
{prev_steps_str}
"""
            # 2. Tokenize Input
            input_tokens = len(tokenizer.encode(input_text))
            
            # 3. Tokenize Output (Current Reasoning Step)
            output_tokens = len(tokenizer.encode(current_step))
            
            # 4. Sum & Accumulate
            step_total = input_tokens + output_tokens
            current_q_tokens += step_total
            
            # 다음 스텝을 위해 현재 스텝 저장
            previous_steps_text.append(current_step)
            
            # (디버깅용 출력 - 필요시 주석 해제)
            # print(f"  Step {step_idx+1}: Input({input_tokens}) + Output({output_tokens}) = {step_total}")

        total_tokens_all_questions += current_q_tokens
        # print(f"  -> Total Tokens: {current_q_tokens}")

    if total_questions == 0:
        return 0

    avg_tokens = total_tokens_all_questions / total_questions
    
    print(f"📊 [결과] Question당 평균 총 토큰 수: {avg_tokens:,.2f}")
    print('----------------------------------------')
    
    return avg_tokens

# 실행
if __name__ == "__main__":
    
    for data in ["2wiki", "hotpotqa", "musique"]:
        input_path = f"/workspace/daeyong/no_feedback_qwen7b_{data}.json"
        print(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        calculate_cumulative_tokens(data)