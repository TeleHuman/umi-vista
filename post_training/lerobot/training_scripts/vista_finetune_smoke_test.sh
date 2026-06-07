#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/data_ysy/app/codex_space/temp/lerobot_finetune_smoke_test}"
LOG_DIR="${OUTPUT_ROOT}/logs"
RUN_ROOT="${OUTPUT_ROOT}/outputs"
LOG_FILE="${LOG_DIR}/vista_finetune_smoke_test.log"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========== LeRobot UMI temp-env smoke test =========="
echo "date: $(date '+%F %T %Z')"
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "script: $0"
echo "log: ${LOG_FILE}"

ENV_DIR="${ENV_DIR:-/data/ysy/app/miniconda3/envs/rhodeslerobot_pi_temp}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1

PYTHON_BIN="${ENV_DIR}/bin/python"
ACCELERATE_BIN="${ENV_DIR}/bin/accelerate"

DATASET_REPO_ID="${DATASET_REPO_ID:-/data/guolinzheng/umi_work_space_0324/umi_dataset_for_bench/simulation/v30/libero_umi_v2_ee_lerobot_720_150_0.6}"
DATASET_ROOT="${DATASET_ROOT:-${DATASET_REPO_ID}}"
DATASET_REVISION="${DATASET_REVISION:-v3.0}"
POLICY_PATH="${POLICY_PATH:-/data_ysy/app/codex_space/share/vista_checkpoints/vista_pres2_16w/pretrained_model}"
REPO_DIR="${REPO_DIR:-/data_ysy/app/codex_space/project/2026_06_05/umi-vista/post_training/lerobot}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-/data/guolinzheng/umi_work_space_0324/lerobot-2602/train_scripts/accelerate_configs/accelerate_default.yaml}"

GPUS="${GPUS:-1}"
GPU_IDS="${GPU_IDS:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-2}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-2}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-2}"
LOG_FREQ="${LOG_FREQ:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
ENFORCE_REPLACE="${ENFORCE_REPLACE:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"

DATE1=$(date "+%y-%m-%d")
TIME1=$(date "+%H-%M-%S")
MODEL_NAME="${MODEL_NAME:-vista_pres2_16w_finetune_smoke}"
DATASET_NAME="${DATASET_REPO_ID##*/}"
OUTPUT_DIR="${RUN_ROOT}/${DATE1}_${TIME1}_${MODEL_NAME}_${DATASET_NAME}_2steps"

echo "python: $(which python)"
"${PYTHON_BIN}" --version
echo "accelerate: $(which accelerate)"
echo "dataset: ${DATASET_REPO_ID}"
echo "policy: ${POLICY_PATH}"
echo "repo: ${REPO_DIR}"
echo "output_dir: ${OUTPUT_DIR}"
echo "total_steps: ${TOTAL_STEPS}"
echo "num_workers: ${NUM_WORKERS}"

for required_path in "${PYTHON_BIN}" "${ACCELERATE_BIN}" "${DATASET_REPO_ID}" "${POLICY_PATH}" "${REPO_DIR}" "${ACCELERATE_CONFIG}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "missing required path: ${required_path}"
    exit 2
  fi
done

if [[ -x "${REPO_DIR}/third_party/prepare_pi_transformers.sh" ]]; then
  bash "${REPO_DIR}/third_party/prepare_pi_transformers.sh"
fi

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/third_party/pi_transformers/src:${PYTHONPATH:-}"
echo "PYTHONPATH: ${PYTHONPATH}"

"${PYTHON_BIN}" - <<'PY'
import lerobot, transformers
print("lerobot_file:", getattr(lerobot, "__file__", None))
print("transformers_file:", getattr(transformers, "__file__", None))
PY

cd "${REPO_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch \
  --num_processes="${GPUS}" \
  --config_file="${ACCELERATE_CONFIG}" \
  --main_process_port="${MAIN_PROCESS_PORT}" \
  src/lerobot/scripts/lerobot_train_umi.py \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.revision="${DATASET_REVISION}" \
  --dataset.image_transforms.enable=true \
  --dataset.wrist_transforms.enable=true \
  --policy.max_state_dim=16 \
  --policy.max_action_dim=32 \
  --policy.dtype=bfloat16 \
  --policy.path="${POLICY_PATH}" \
  --policy.push_to_hub=false \
  --policy.chunk_size="${ACTION_CHUNK_SIZE}" \
  --policy.n_action_steps="${ACTION_CHUNK_SIZE}" \
  --policy.optimizer_lr="${LEARNING_RATE}" \
  --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}" \
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
  --policy.use_delta_action=true \
  --output_dir="${OUTPUT_DIR}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${TOTAL_STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --num_workers="${NUM_WORKERS}" \
  --enforce_input_output_replace="${ENFORCE_REPLACE}" \
  --seed="${SEED}"

status=$?
echo "smoke test exit code: ${status}"
exit "${status}"
