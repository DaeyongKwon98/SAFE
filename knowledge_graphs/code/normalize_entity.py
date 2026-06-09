import json
import os
from tqdm import tqdm
import argparse

def main(args):
    BASE_DIR = "/workspace/daeyong/knowledge_graphs"

    ENTITY_MAPPING_PATH = os.path.join(BASE_DIR, f"{args.dataset}_same_entity_gleaned.json")
    # ENTITY_MAPPING_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_same_entity_gleaned.json"
    TRIPLES_PATH = os.path.join(BASE_DIR, f"{args.dataset}_triples_gleaned.json")
    # TRIPLES_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_triples_gleaned.json"
    OUTPUT_PATH = os.path.join(BASE_DIR, f"{args.dataset}_triples_normalized_gleaned.json")
    # OUTPUT_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_triples_normalized_gleaned.json"
    
    # --- 1. Load Entity Resolution Data ---
    print(f"📂 Loading Entity Mappings from: {ENTITY_MAPPING_PATH}")
    if not os.path.exists(ENTITY_MAPPING_PATH):
        print(f"❌ File not found: {ENTITY_MAPPING_PATH}")
        return

    with open(ENTITY_MAPPING_PATH, "r", encoding="utf-8") as f:
        entity_data = json.load(f)

    # --- 2. Build Lookup Dictionary ---
    # 수정: passage_indices가 리스트이므로, 내부의 index를 하나씩 꺼내서 key로 사용
    print("🔄 Building Mapping Dictionary...")
    
    global_mapping = {} 
        
    for item in tqdm(entity_data, desc="Indexing mappings"):
        p_indices = item.get('passage_indices', [])
        synonym_groups = item.get('synonym_groups', [])
        
        local_map = {}
        for group in synonym_groups:
            if not group: continue
            
            # 대표 이름(Canonical Name) 공백 제거
            canonical = str(group[0]).strip()
            
            for variant in group:
                # 변형(Variant) 이름도 공백 제거하여 매핑
                local_map[str(variant).strip()] = canonical
        
        for p_idx in p_indices:
            # 타입을 통일하기 위해 문자열로 강제 변환
            str_p_idx = str(p_idx) 
            if str_p_idx in global_mapping:
                global_mapping[str_p_idx].update(local_map)
            else:
                global_mapping[str_p_idx] = local_map.copy()

    # --- 3. Load Triples Data ---
    print(f"📂 Loading Triples from: {TRIPLES_PATH}")
    if not os.path.exists(TRIPLES_PATH):
        print(f"❌ File not found: {TRIPLES_PATH}")
        return

    with open(TRIPLES_PATH, "r", encoding="utf-8") as f:
        triples_data = json.load(f)

    # --- 4. Normalize Triples ---
    print("🔄 Normalizing Triples...")
    normalized_count = 0
    
    # enumerate를 사용하여 리스트의 인덱스(i)를 함께 가져옵니다.
    for i, entry in enumerate(tqdm(triples_data, desc="Processing triples")):
        # 원래 값에서 가져오되, 만약 비어있거나 없다면 현재 줄 번호(i)를 인덱스로 사용!
        p_indices = entry.get('passage_indices')
        if not p_indices: 
            p_indices = [i]
            
        original_triples = entry.get('triples_updated', [])
        
        current_mapping = {}
        for p_idx in p_indices:
            str_p_idx = str(p_idx)
            if str_p_idx in global_mapping:
                current_mapping.update(global_mapping[str_p_idx])
        
        normalized_triples = []
        
        for triple in original_triples:
            if not isinstance(triple, list) or len(triple) < 3:
                normalized_triples.append(triple)
                continue
                
            subj, pred, obj = triple[0], triple[1], triple[2]
            
            # Subject 치환
            clean_subj = str(subj).strip() if isinstance(subj, str) else subj
            new_subj = current_mapping.get(clean_subj, clean_subj)  # 수정: 원본 대신 clean_subj 사용
            
            # Object 치환
            clean_obj = str(obj).strip() if isinstance(obj, str) else obj
            new_obj = current_mapping.get(clean_obj, clean_obj)     # 수정: 원본 대신 clean_obj 사용
                
            normalized_triples.append([new_subj, pred, new_obj])
            
            # 카운팅 (실제 변경이 일어난 경우)
            if new_subj != clean_subj or new_obj != clean_obj:
                normalized_count += 1
        
        entry['triples_normalized'] = normalized_triples

    # --- 5. Save Result ---
    print(f"💾 Saving normalized data to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(triples_data, f, indent=2, ensure_ascii=False)

    print(f"🎉 Done! Total entities normalized: {normalized_count}")
    
    # --- [디버깅 블록 시작] 데이터 구조 눈으로 확인하기 ---
    print("\n" + "="*50)
    print("🚨 [DEBUG] 데이터 구조 점검 🚨")
    
    # 1. global_mapping 상태 확인
    print(f"✅ 생성된 global_mapping 개수: {len(global_mapping)}")
    if len(global_mapping) > 0:
        sample_p_idx = list(global_mapping.keys())[0]
        print(f"👉 매핑 인덱스 예시: '{sample_p_idx}' (타입: {type(sample_p_idx)})")
        print(f"👉 매핑 데이터 예시: {list(global_mapping[sample_p_idx].items())[:2]}...")
    else:
        print("❌ 앗! global_mapping이 텅 비어있습니다. Entity Mapping 파일을 확인하세요.")

    # 2. triples_data 상태 확인
    print(f"\n✅ 로드된 트리플 데이터 개수: {len(triples_data)}")
    if len(triples_data) > 0:
        sample_entry = triples_data[0]
        t_p_indices = sample_entry.get('passage_indices', [])
        t_triples = sample_entry.get('triples_updated', [])
        
        print(f"👉 트리플 인덱스 예시: {t_p_indices}")
        if not t_triples:
            print("❌ 'triples_updated' 키에 데이터가 없습니다! JSON 키 이름이 맞는지 확인하세요. (혹시 'triples' 아닐까요?)")
            print(f"   현재 존재하는 키 목록: {list(sample_entry.keys())}")
        else:
            print(f"👉 트리플 데이터 예시: {t_triples[:2]}")
    print("="*50 + "\n")
    # --- [디버깅 블록 끝] ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize entity names in triples.")
    parser.add_argument("--dataset", type=str, choices=["2wiki", "hotpotqa", "musique"], required=True)
    args = parser.parse_args()

    main(args)