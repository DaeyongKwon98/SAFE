import json
from pathlib import Path
import csv

ROOT = Path('/workspace/daeyong/inference_results')
TARGET_PREFIXES = ('self_feedback_kg_correct_1k_sample', 'no_feedback')
OUT_SUMMARY = ROOT / 'feedback_mode_one_char_final_answer_summary.csv'
OUT_FILES = ROOT / 'feedback_mode_one_char_final_answer_files.csv'


def classify(value):
    s = '' if value is None else str(value).strip()
    if len(s) != 1:
        return None
    if s.isdigit():
        return 'digit'
    if s.isalpha():
        return 'alpha'
    return 'other'

rows = []
file_rows = []
folders = sorted(
    p for p in ROOT.iterdir()
    if p.is_dir() and p.name.startswith(TARGET_PREFIXES)
)

for folder in folders:
    final_files = sorted(folder.glob('*_final_answer.json'))
    for path in final_files:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        total = len(data)
        one_char = 0
        digit = 0
        alpha = 0
        other = 0
        examples = []
        for row in data:
            c = classify(row.get('final_answer'))
            if c is None:
                continue
            one_char += 1
            if c == 'digit':
                digit += 1
            elif c == 'alpha':
                alpha += 1
            else:
                other += 1
            if len(examples) < 5:
                examples.append(f"{row.get('id')}::{str(row.get('final_answer')).strip()}")
        file_rows.append({
            'group': 'self_feedback' if folder.name.startswith('self_feedback_kg_correct_1k_sample') else 'no_feedback',
            'folder': folder.name,
            'file': path.name,
            'file_path': str(path),
            'total_rows': total,
            'one_char_rows': one_char,
            'one_char_digit_rows': digit,
            'one_char_alpha_rows': alpha,
            'one_char_other_rows': other,
            'one_char_pct': round(one_char / total * 100, 2) if total else 0.0,
            'examples': ' | '.join(examples),
        })

summary = {}
for row in file_rows:
    key = (row['group'], row['folder'])
    item = summary.setdefault(key, {
        'group': row['group'],
        'folder': row['folder'],
        'files': 0,
        'affected_files': 0,
        'total_rows': 0,
        'one_char_rows': 0,
        'one_char_digit_rows': 0,
        'one_char_alpha_rows': 0,
        'one_char_other_rows': 0,
    })
    item['files'] += 1
    item['affected_files'] += int(row['one_char_rows'] > 0)
    item['total_rows'] += row['total_rows']
    item['one_char_rows'] += row['one_char_rows']
    item['one_char_digit_rows'] += row['one_char_digit_rows']
    item['one_char_alpha_rows'] += row['one_char_alpha_rows']
    item['one_char_other_rows'] += row['one_char_other_rows']

rows = []
for (_, _), item in sorted(summary.items()):
    total = item['total_rows']
    item['one_char_pct'] = round(item['one_char_rows'] / total * 100, 2) if total else 0.0
    rows.append(item)

with OUT_SUMMARY.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['group','folder'])
    writer.writeheader()
    writer.writerows(rows)

with OUT_FILES.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(file_rows[0].keys()) if file_rows else ['group','folder','file'])
    writer.writeheader()
    writer.writerows(file_rows)

print(f'summary_csv={OUT_SUMMARY}')
print(f'files_csv={OUT_FILES}')
print(f'folders={len(folders)} files={len(file_rows)}')
for row in rows:
    print('\t'.join(str(row[k]) for k in ['group','folder','affected_files','files','one_char_rows','one_char_digit_rows','one_char_alpha_rows','one_char_other_rows','one_char_pct']))
