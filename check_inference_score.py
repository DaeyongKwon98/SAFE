import pandas as pd
import numpy as np
import re
import string
from collections import Counter

def normalize_text(s: str) -> str:
    """
    텍스트를 소문자로 변환하고, 구두점과 관사(a, an, the)를 제거합니다.
    SQuAD 데이터셋 평가에 사용되는 표준 정규화 방식입니다.
    """
    if not isinstance(s, str):
        return ""
    
    s = s.lower()
    # 구두점 제거
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # 관사(a, an, the) 제거
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # 연속된 공백을 하나의 공백으로 변환 및 앞뒤 공백 제거
    s = ' '.join(s.split())
    return s.strip()

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """두 문자열 간의 F1 점수를 토큰 레벨에서 계산합니다."""
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()
    
    # 두 문자열이 모두 비어있으면 완벽히 일치한 것으로 간주
    if not pred_tokens and not truth_tokens:
        return 1.0
    # 어느 한 쪽이라도 비어있으면 F1 점수는 0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = 1.0 * num_common / len(pred_tokens)
    recall = 1.0 * num_common / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """두 문자열이 정확히 일치하는지 확인합니다."""
    return 1.0 if prediction == ground_truth else 0.0

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    prediction과 ground_truths 리스트 내의 각 정답을 비교하여
    최고 점수를 반환합니다.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def calculate_metrics(target_df, dataset_name, gt_df_musique=None):
    """
    주어진 DataFrame에 대해 EM, F1 점수를 계산하여 반환하는 함수
    """
    CANNOT_ANSWER_NORM = normalize_text("Cannot Answer")
    
    if len(target_df) == 0:
        return 0.0, 0.0

    if dataset_name == "musique":
        em_scores = []
        f1_scores = []
        
        for _, row in target_df.iterrows():
            pred = row['generated_answer']
            
            # [중요] Reference 로직 반영: 
            # Ground Truth가 "Cannot Answer"인 경우(No split 등)에는 리스트 검색 없이 바로 처리
            if row['ground_truth'] == CANNOT_ANSWER_NORM:
                gts = [CANNOT_ANSWER_NORM]
            else:
                # Yes split인 경우 answer_list에서 후보군 가져오기
                gts_raw = gt_df_musique.loc[gt_df_musique['question'] == row['question'], 'answer_list_norm'].values
                if len(gts_raw) > 0:
                    gts = gts_raw[0]
                else:
                    gts = [row['ground_truth']] # 예외 처리

            if not gts:
                gts = [row['ground_truth']]
                
            em_scores.append(metric_max_over_ground_truths(calculate_exact_match, pred, gts))
            f1_scores.append(metric_max_over_ground_truths(calculate_f1_score, pred, gts))
            
        return np.mean(em_scores), np.mean(f1_scores)

    else:
        # 2wiki, hotpotqa (1:1 비교)
        em_scores = [calculate_exact_match(pred, gt) for pred, gt in zip(target_df['generated_answer'], target_df['ground_truth'])]
        f1_scores = [calculate_f1_score(pred, gt) for pred, gt in zip(target_df['generated_answer'], target_df['ground_truth'])]
        return np.mean(em_scores), np.mean(f1_scores)

model = "gemma12b"
for dataset in ["2wiki", "hotpotqa", "musique"]:
    df = pd.read_json(f"/workspace/daeyong/inference_results/dev_kg_wrong_qwen2.5_7b_2wiki_added_checkpoint_320_10steps/{model}_{dataset}_final_answer.json")
    # df = pd.read_json(f"/workspace/daeyong/ours_SFT_GPT_end_fixed_logical_leap_{model}_{dataset}_answer.json")
    # df = pd.read_json(f"/workspace/daeyong/reasoning_results/no_feedback_{model}_{dataset}_answer_fixed.json")
    # df = pd.read_json(f"/workspace/daeyong/reasoning_results/self_feedback_{model}_{dataset}_result_500_answer_fixed.json")
    
    # 1. 텍스트 정규화
    df['ground_truth'] = df['ground_truth'].apply(normalize_text)
    df['generated_answer'] = df['final_answer'].apply(normalize_text)

    # 2. Musique 데이터셋 추가 정답지 로드 및 정규화
    # gt_df = pd.read_csv(f"/workspace/daeyong/benchmarks/{dataset}_dev_yes.csv")
    if dataset == "2wiki":
        gt_df = pd.read_csv("/workspace/daeyong/benchmarks/2wiki_dev.csv")
    elif dataset == "hotpotqa":
        gt_df = pd.read_csv("/workspace/daeyong/benchmarks/hotpotqa_dev.csv")
    elif dataset == "musique":
        gt_df = pd.read_csv("/workspace/daeyong/benchmarks/musique_dev.csv")
        
    if dataset == "musique":
        gt_df['answer_list_norm'] = gt_df['answer_list'].apply(
            lambda gts: [normalize_text(str(gt)) for gt in gts] if isinstance(gts, list) else [normalize_text(str(gts))]
        )
    
    # gt_df = gt_df[~gt_df['error_place'].isin(["question", "gt_passages", "gt_answer"])]
    # df = df[df['question'].isin(gt_df['question'])]
    print(f"\n[{dataset}] Total Records: {len(df)}")

    # 5. 점수 계산 및 출력
    em_all, f1_all = calculate_metrics(df, dataset, gt_df)

    print(f"  - All: EM={em_all*100:.2f}, F1={f1_all*100:.2f}")
    # print(f"  - Yes: EM={em_yes*100:.2f}, F1={f1_yes*100:.2f}")
    # print(f"  - No: EM={em_no*100:.2f}, F1={f1_no*100:.2f}")
    
    # df = pd.read_json(f"/workspace/daeyong/ours_SFT_GPT_end_fixed_2e_4_{model}_{dataset}_stats_qwen_vllm.json")
    # print(f"Final step count: {np.mean(df['final_step_count'])}")
    # print(f"Generator calls: {np.mean(df['generator_calls'])}")
    # print(f"Evaluator calls: {np.mean(df['evaluator_calls'])}")
    # print(f"Total tokens: {np.mean(df['total_tokens'])}")