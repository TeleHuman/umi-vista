#!/usr/bin/env bash
set -euo pipefail

# Run replay.py sequentially for all configured robot types.
# Usage:
#   ./run_replay_all.sh                       # score mode for each robot (uses default SCORE_FOLDER_PATH in replay.py)
#   ./run_replay_all.sh <idx>                 # replay one trajectory for each robot
#   ./run_replay_all.sh <score_folder>        # score mode with explicit folder
#   ./run_replay_all.sh <score_folder> <idx>  # replay one trajectory with explicit folder
#
# You can also pass path via env:
#   export SCORE_FOLDER_PATH=/path/to/task_folder
#   ./run_replay_all.sh [idx]

cd "$(dirname "$0")"

robots=(acone r1pro rm75)

score_folder="${SCORE_FOLDER_PATH:-}"
args=("$@")

if [[ ${#args[@]} -ge 1 ]]; then
  # Backwards compatible: if first arg is an integer, treat it as idx.
  if [[ "${args[0]}" =~ ^[0-9]+$ ]]; then
    :
  else
    score_folder="${args[0]}"
    args=("${args[@]:1}")
  fi
fi

if [[ -n "${score_folder}" ]]; then
  export SCORE_FOLDER_PATH="${score_folder}"
  echo "[ENV] SCORE_FOLDER_PATH=${SCORE_FOLDER_PATH}"
fi

for robot in "${robots[@]}"; do
  echo ""
  echo "==============================="
  echo "[RUN] ROBOT_NAME=${robot} python3 replay.py ${args[*]:-}"
  echo "==============================="

  ROBOT_NAME="${robot}" python3 replay.py "${args[@]}"
done
