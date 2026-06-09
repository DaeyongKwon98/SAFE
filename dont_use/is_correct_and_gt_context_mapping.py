import pandas as pd
from tqdm import tqdm
import json
import os
import re
import ast
import difflib

dataset = "musique"
ideal_steps_path = f"/workspace/daeyong/ideal_steps/{dataset}_ideal_steps_2.json"
correct_path = f"/workspace/daeyong/ideal_steps/{dataset}_is_correct_2.json"
output_path = f"/workspace/daeyong/ideal_steps/{dataset}_ideal_steps_filtered_2.json"

# 1. Load ideal steps data
with open(ideal_steps_path, "r") as f:
    ideal_steps_data = json.load(f)

# 2. Load correctness data
with open(correct_path, "r") as f:
    correctness_data = json.load(f)

# 3. Collect questions with is_correct == "0"
remove_questions = {item["question"] for item in correctness_data if item["is_correct"] == "0"}

print(f"⚠️ Removing {len(remove_questions)} incorrect items...")

# 4. Remove entries from ideal_steps_data
filtered_data = [item for item in ideal_steps_data if item["question"] not in remove_questions]

print(f"Before filtering: {len(ideal_steps_data)}")
print(f"After filtering: {len(filtered_data)} (Removed {len(ideal_steps_data) - len(filtered_data)})")

# 5. Save filtered result
with open(output_path, "w") as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"✅ Saved filtered data to: {output_path}")

csv_path = f"/workspace/daeyong/benchmarks/{dataset}.csv"
output_path = f"/workspace/daeyong/ideal_steps/{dataset}_ideal_steps_passage_mapped_2.json"

# --- 헬퍼 함수 (100% ROBUST V6) ---

def get_key_and_remainder(text: str) -> tuple[str, str]:
    """
    텍스트를 [첫 번째 콜론(:) 앞의 L1 키]와 [나머지 텍스트]로 분리합니다.
    """
    text = text.strip()
    
    # 첫 번째 콜론의 위치를 찾음
    first_colon_index = text.find(':')
    
    if first_colon_index != -1:
        # 콜론이 있음
        key_l1 = text[:first_colon_index].strip()
        # ❗️ [V6] 텍스트 정규화: 비교 오류를 줄이기 위해 공백을 통일합니다.
        remainder_text = " ".join(text[first_colon_index+1:].strip().split())
        return key_l1, remainder_text
    else:
        # 콜론이 없음 (e.g., "No Colon Text")
        return text.strip(), ""


def map_passage_index(gt_context: list[str], retrieved_passages: list[str]) -> dict[int, int]:
    """
    100% 견고한 매핑 (V6):
    1. L1 키(타이틀)로 1:1 매핑을 시도합니다.
    2. L1 키가 모호한 경우(e.g., "FBI", "Zeitgeist"),
       `difflib`을 사용하여 'remainder' 텍스트가 가장 유사한
       단 하나의 1:1 매칭을 찾습니다.
    """
    
    # 1. [Pre-processing] retrieved_passages의 L1 키 및 튜플 맵 생성
    # {key_l1: [list_of_rp_indices_0_based]}
    rp_map_l1 = {}
    # (key_l1, remainder) 튜플 미리 생성 (V6)
    rp_data = []
    
    for i, rp_text in enumerate(retrieved_passages):
        key_l1, remainder = get_key_and_remainder(rp_text)
        rp_data.append((key_l1, remainder))
        rp_map_l1.setdefault(key_l1, []).append(i)

    # 2. [Mapping] 매핑 수행
    final_mapping = {} # { ideal_passage_num (1-based) : new_passage_num (1-based) }
    
    for gt_index, gt_text in enumerate(gt_context):
        ideal_num = gt_index + 1 # 1-based
        key_l1, gt_remainder = get_key_and_remainder(gt_text)
        
        # L1 키가 일치하는 retrieved_passages 후보군
        l1_candidates_indices = rp_map_l1.get(key_l1, [])
        
        # --- 3. [V6 로직 적용] ---
        
        # --- Case A: L1 키가 고유함 (1:1 매핑) ---
        if len(l1_candidates_indices) == 1:
            final_match_index = l1_candidates_indices[0]
            final_mapping[ideal_num] = final_match_index + 1 # 1-based
        
        # --- Case B: L1 키가 모호함 (N:M 매핑) ---
        # ❗️ [100% 핵심 V6 수정] difflib으로 "Best-Fit" 1:1 매칭
        elif len(l1_candidates_indices) > 1:
            
            # 후보군(rp)의 [remainder 텍스트]와 [원본 인덱스]
            candidate_remainders_map = {
                rp_data[rp_index][1]: rp_index 
                for rp_index in l1_candidates_indices
            }
            
            # difflib.get_close_matches를 사용하여
            # gt_remainder와 가장 유사한 rp_remainder를 *단 하나* 찾음
            best_match_list = difflib.get_close_matches(
                gt_remainder, 
                candidate_remainders_map.keys(), 
                n=1, 
                cutoff=0.0
            )
            
            if best_match_list:
                # 가장 유사한 remainder 텍스트
                best_match_remainder = best_match_list[0]
                # 해당 텍스트의 원본 rp_index (0-based)
                final_match_index = candidate_remainders_map[best_match_remainder]
                final_mapping[ideal_num] = final_match_index + 1 # 1-based
            
            else:
                print(f"CRITICAL ERROR (gt_index {gt_index}): L1은 모호했으나, 0% 이상 유사한 remainder를 찾지 못했습니다. (L1 키: '{key_l1}')")
                final_mapping[ideal_num] = -1

        # --- Case C: L1 키를 찾을 수 없음 ---
        else: # len(l1_candidates_indices) == 0
            print(f"CRITICAL ERROR (gt_index {gt_index}): L1 키 '{key_l1}'를 rp에서 찾을 수 없습니다.")
            final_mapping[ideal_num] = -1

    return final_mapping


def replace_passage_refs(step_text: str, mapping: dict[int, int]) -> str:
    """
    ideal_steps 내 "Passage X" → "Passage Y" 로 교체 (버그 수정)
    """
    
    def replacer(match):
        old_num_str = match.group(1)
        try:
            old_num = int(old_num_str)
            new_num = mapping.get(old_num, old_num) 
            if new_num == -1:
                return f"Passage [MAPPING_ERROR]"
            return f"Passage {new_num}"
        except ValueError:
            return match.group(0) 

    return re.sub(r"Passage\s*(\d+)\b", replacer, step_text)

# --- 메인 로직 ---

print("Loading data...")
ideal_data = filtered_data

# Load 2wiki csv
df = pd.read_csv(csv_path)
df["retrieved_passages"] = df["retrieved_passages"].apply(ast.literal_eval)
df["gt_index"] = df["gt_index"].apply(ast.literal_eval) # gt_index는 이제 참조용

# Question -> (retrieved_passages, gt_index) 매핑
csv_map = {
    row["question"].strip(): (row["retrieved_passages"], row["gt_index"])
    for _, row in df.iterrows()
}

# 새로운 결과 리스트
new_results = []
error_count = 0

print(f"Starting 100% robust mapping (V6 - L1 + difflib) for {len(ideal_data)} items...")
for item in tqdm(ideal_data, desc="Mapping passages"):
    q = item["question"].strip()
    
    if q not in csv_map:
        new_results.append(item)
        continue
    
    retrieved_passages, gt_index = csv_map[q]
    
    # [핵심 V6] L1 키(타이틀) + difflib Best-Fit 매칭
    mapping = map_passage_index(item["gt_context"], retrieved_passages)

    # ideal_steps 업데이트
    new_ideal_steps = [replace_passage_refs(step, mapping) for step in item["ideal_steps"]]

    # (선택적) 매핑 실패한 항목 감지
    if -1 in mapping.values():
        print(f"\nWarning: Mapping failure detected for Q: {q[:50]}...")
        error_count += 1

    # 저장
    new_results.append({
        "question": item["question"],
        "retrieved_passages": retrieved_passages,
        "gt_index": gt_index, # 원본 gt_index는 보존 (참고용)
        "gt_context": item["gt_context"],
        "plan": item["plan"],
        "ideal_steps": new_ideal_steps,
        "_mapping_debug": mapping # (선택적) 디버깅을 위해 매핑 결과 저장
    })

# 저장
with open(output_path, "w") as f:
    json.dump(new_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ 완료: Passage 번호가 'V6 (L1 + difflib)' 키 기반으로 100% 매핑되었습니다.")
print(f"Total mapping errors (unresolvable): {error_count}")
print(f"📁 저장 위치: {output_path}")