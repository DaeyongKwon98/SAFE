import json
import re
import argparse
import os
from tqdm import tqdm

def extract_passage_ids(text):
    """
    텍스트에서 'Passage X' 형태의 번호를 모두 추출하여 집합(Set)으로 반환합니다.
    예: "According to Passage 1 and Passage 10..." -> {'1', '10'}
    """
    # 대소문자 무시, Passage 뒤에 공백이 있거나 없을 수 있음
    matches = re.findall(r"Passage\s*(\d+)", text, re.IGNORECASE)
    return set(matches)

def main(args):
    input_filepath = f"/workspace/daeyong/feedback/{args.dataset}_redundancy.json"
    output_filepath = f"/workspace/daeyong/feedback/{args.dataset}_redundancy_filtered.json"

    print(f"📂 Loading data from: {input_filepath}")
    
    if not os.path.exists(input_filepath):
        print(f"❌ File not found: {input_filepath}")
        return

    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"   Total items loaded: {len(data)}")

    filtered_data = []
    removed_count = 0

    for item in tqdm(data, desc="Filtering invalid redundancy"):
        steps = item.get('corrupted_steps', [])
        
        # 데이터가 비어있거나 스텝이 1개 이하인 경우 (비교 불가) -> 일단 유지하거나 스킵
        if not steps or len(steps) < 2:
            filtered_data.append(item)
            continue

        last_step = steps[-1]
        previous_steps = steps[:-1]

        # 1. 마지막 스텝이 Attribution인지 확인
        if "(Attribution)" in last_step:
            # 2. 마지막 스텝에서 Passage ID 추출
            current_passage_ids = extract_passage_ids(last_step)
            
            # 3. 이전 스텝들에서 등장한 모든 Passage ID 추출
            previous_passage_ids = set()
            for step in previous_steps:
                previous_passage_ids.update(extract_passage_ids(step))
            
            # 4. 검증: 현재 스텝의 Passage ID가 이전 스텝들에 존재하지 않는 것이 있다면 '잘못된 데이터'
            # (current_passage_ids가 previous_passage_ids의 부분집합이어야 함)
            if not current_passage_ids.issubset(previous_passage_ids):
                # 새로운 Passage를 인용했으므로 Redundancy가 아님 -> 제거 대상
                removed_count += 1
                # 디버깅용 출력 (첫 5개만)
                if removed_count <= 5:
                    print(f"\n[Removed Item ID: {item.get('id', 'N/A')}]")
                    print(f" - Prev Passages: {previous_passage_ids}")
                    print(f" - Last Step (Invalid): {last_step}")
                continue

        # 문제가 없거나 Logical Step인 경우 데이터 유지
        filtered_data.append(item)

    print("-" * 50)
    print(f"✅ Filtering Completed.")
    print(f"   Original count: {len(data)}")
    print(f"   Removed count : {removed_count}")
    print(f"   Final count   : {len(filtered_data)}")
    
    # 결과 저장
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved filtered data to: {output_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out invalid redundancy samples where new passages are cited.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., '2wiki')")
    
    args = parser.parse_args()
    main(args)