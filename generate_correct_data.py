import pandas as pd
from collections import Counter

dfs = []
for dataset in ["2wiki", "hotpotqa", "musique"]:
	df = pd.read_json(f"/workspace/daeyong/ideal_steps/{dataset}_ideal_steps_passage_mapped_2.json")[['question', 'retrieved_passages', 'ideal_steps']]
	processed_data = []

	for i, row in df.iterrows():
		steps = row['ideal_steps']

		# 길이가 4 미만이면 skip
		if len(steps) < 4:
			continue

		# step index 3부터 마지막까지 (0-based 기준)
		for k in range(3, len(steps)):  
			previous_steps = steps[:k]           # 리스트
			current_step = steps[k]              # 문자열

			processed_data.append({
				"question": row["question"],
				"retrieved_passages": row.get("retrieved_passages", None),
				"ideal_steps": row['ideal_steps'],
				"previous_steps": previous_steps,
				"current_step": current_step
			})

	processed_df = pd.DataFrame(processed_data)
	dfs.append(processed_df)

df = pd.concat(dfs, ignore_index=True)
print(Counter(df['previous_steps'].apply(len)+1))
df.to_json("/workspace/daeyong/ideal_steps/combined_correct_steps_musique.json", orient="records", lines=False, indent=2)