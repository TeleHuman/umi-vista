#!/usr/bin/env bash
set -euo pipefail

# Batch runner: execute run_replay_all.sh sequentially for every subfolder under
# one or more task-parent directories.

cd "$(dirname "$0")"

BASE_DIRS_STR="${BASE_DIRS:-${PHYSICAL_VALIDATION_BASE_DIRS:-}}"
if [[ -z "${BASE_DIRS_STR}" ]]; then
  echo "[ERROR] Set BASE_DIRS or PHYSICAL_VALIDATION_BASE_DIRS to one or more task-parent directories." >&2
  echo "Example: BASE_DIRS='/path/to/parent1 /path/to/parent2' ./run_replay_all_batch.sh" >&2
  exit 2
fi

read -r -a BASE_DIRS <<< "${BASE_DIRS_STR}"

for base_dir in "${BASE_DIRS[@]}"; do
  if [[ ! -d "$base_dir" ]]; then
    echo "[ERROR] BASE_DIR not found: $base_dir" >&2
    exit 1
  fi

  subdirs=()
  mapfile -d '' subdirs < <(find "$base_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

  if [[ ${#subdirs[@]} -eq 0 ]]; then
    echo "[WARN] No subfolders found under: $base_dir" >&2
    continue
  fi

  echo "[INFO] BASE_DIR=$base_dir"
  echo "[INFO] Found ${#subdirs[@]} subfolders"

  idx=0
  for score_dir in "${subdirs[@]}"; do
    idx=$((idx + 1))
    echo ""
    echo "========================================"
    echo "[BATCH] ($idx/${#subdirs[@]}) SCORE_FOLDER_PATH=$score_dir"
    echo "========================================"

    SCORE_FOLDER_PATH="$score_dir" bash ./run_replay_all.sh "$@"
  done
done
