#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7

MANIFEST_PATH="/workspace/daeyong/inference_results/no_premature_conclusion_rerun_manifest.csv"
FINAL_ANSWER_SCRIPT="/workspace/daeyong/final_answer.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
START_FROM=""
END_AT=""
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  bash rerun_no_premature_conclusion_final_answers.sh [options]

Options:
  --manifest PATH        CSV manifest path.
  --script PATH          final_answer.py path.
  --python-bin PATH      Python executable to use. Default: python3
  --start-from FOLDER    Start from this folder name (inclusive).
  --end-at FOLDER        End at this folder name (inclusive).
  --dry-run              Print commands without executing them.
  -h, --help             Show this help.

Examples:
  bash rerun_no_premature_conclusion_final_answers.sh
  bash rerun_no_premature_conclusion_final_answers.sh --start-from dev_kg_correct_1ksample_no_premature_conclusion_10_3_qwen3_8b_no_premature_conclusion
  bash rerun_no_premature_conclusion_final_answers.sh --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --script)
      FINAL_ANSWER_SCRIPT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --start-from)
      START_FROM="$2"
      shift 2
      ;;
    --end-at)
      END_AT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

if [[ ! -f "$FINAL_ANSWER_SCRIPT" ]]; then
  echo "final_answer.py not found: $FINAL_ANSWER_SCRIPT" >&2
  exit 1
fi

mapfile -t FOLDERS < <(
  "$PYTHON_BIN" - "$MANIFEST_PATH" "$START_FROM" "$END_AT" <<'PY'
import csv
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
start_from = sys.argv[2]
end_at = sys.argv[3]
base_dir = Path("/workspace/daeyong/inference_results")

rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
folders = {}
for row in rows:
    folder = row["folder"]
    folder_path = str(base_dir / folder)
    key = (int(row["max_steps"]), int(row["max_retries"]), folder)
    folders[key] = folder_path

ordered = [folders[key] for key in sorted(folders.keys())]

if start_from:
    target = str(base_dir / start_from)
    if target not in ordered:
        raise SystemExit(f"start-from folder not found in manifest: {start_from}")
    ordered = ordered[ordered.index(target):]

if end_at:
    target = str(base_dir / end_at)
    if target not in ordered:
        raise SystemExit(f"end-at folder not found in manifest slice: {end_at}")
    ordered = ordered[: ordered.index(target) + 1]

for folder_path in ordered:
    print(folder_path)
PY
)

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
  echo "No folders selected." >&2
  exit 1
fi

echo "Selected ${#FOLDERS[@]} folders."
for folder_path in "${FOLDERS[@]}"; do
  cmd=( "$PYTHON_BIN" "$FINAL_ANSWER_SCRIPT" --folder_path "$folder_path" )
  echo
  echo ">>> ${cmd[*]}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${cmd[@]}"
  fi
done

echo
echo "All requested final answer reruns are complete."
