#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data_ysy/app/codex_space/project/2026_06_05/umi-vista}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${PROJECT_ROOT}/post_training/lerobot}"
ENV_DIR="${ENV_DIR:-/data/ysy/app/miniconda3/envs/rhodeslerobot_pi_temp}"
POLICY_PATH="${POLICY_PATH:-/data_ysy/app/codex_space/share/vista_checkpoints/vista_libero_umi_v2_40k/pretrained_model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data_ysy/app/codex_space/temp/libero_umi_eval/$(date +%Y%m%d_%H%M%S)_vista}"
N_EPISODES_PER_TASK="${N_EPISODES_PER_TASK:-50}"
GPU_ID="${GPU_ID:-0}"
SUITES="${SUITES:-libero_10 libero_goal libero_object libero_spatial}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/root/.libero}"
LIBERO_PACKAGE_DIR="${LIBERO_PACKAGE_DIR:-${ENV_DIR}/lib/python3.10/site-packages/libero/libero}"
LIBERO_DATASETS_DIR="${LIBERO_DATASETS_DIR:-/data_ysy/app/codex_space/share/libero_datasets/datasets}"
LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:-${LIBERO_PACKAGE_DIR}/assets}"
LIBERO_BDDL_DIR="${LIBERO_BDDL_DIR:-${LIBERO_PACKAGE_DIR}/bddl_files}"
LIBERO_INIT_STATES_DIR="${LIBERO_INIT_STATES_DIR:-${LIBERO_PACKAGE_DIR}/init_files}"
LIBERO_OVERWRITE_CONFIG="${LIBERO_OVERWRITE_CONFIG:-0}"
GLVND_RUNTIME_DIR="${GLVND_RUNTIME_DIR:-/data_ysy/app/codex_space/share/glvnd_runtime}"
PI_TRANSFORMERS_PREPARE="${PI_TRANSFORMERS_PREPARE:-${LEROBOT_ROOT}/third_party/prepare_pi_transformers.sh}"

if [[ ! -x "${PI_TRANSFORMERS_PREPARE}" ]]; then
  echo "missing executable dependency preparation script: ${PI_TRANSFORMERS_PREPARE}" >&2
  exit 2
fi

bash "${PI_TRANSFORMERS_PREPARE}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${GLVND_RUNTIME_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/third_party/pi_transformers/src:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH

mkdir -p "${OUTPUT_ROOT}/logs" "${LIBERO_CONFIG_PATH}" "${LIBERO_DATASETS_DIR}"

if [[ "${LIBERO_OVERWRITE_CONFIG}" == "1" || ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
assets: ${LIBERO_ASSETS_DIR}
bddl_files: ${LIBERO_BDDL_DIR}
benchmark_root: ${LIBERO_PACKAGE_DIR}
datasets: ${LIBERO_DATASETS_DIR}
init_states: ${LIBERO_INIT_STATES_DIR}
EOF
fi

printf 'PROJECT_ROOT=%s\n' "${PROJECT_ROOT}"
printf 'LEROBOT_ROOT=%s\n' "${LEROBOT_ROOT}"
printf 'POLICY_PATH=%s\n' "${POLICY_PATH}"
printf 'OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
printf 'N_EPISODES_PER_TASK=%s\n' "${N_EPISODES_PER_TASK}"
printf 'GPU_ID=%s\n' "${GPU_ID}"
printf 'SUITES=%s\n' "${SUITES}"
printf 'SUMMARY_PATH=%s\n' "${SUMMARY_PATH}"
printf 'LIBERO_CONFIG_PATH=%s\n' "${LIBERO_CONFIG_PATH}"
printf 'LIBERO_DATASETS_DIR=%s\n' "${LIBERO_DATASETS_DIR}"
printf 'LIBERO_ASSETS_DIR=%s\n' "${LIBERO_ASSETS_DIR}"
printf 'LIBERO_BDDL_DIR=%s\n' "${LIBERO_BDDL_DIR}"
printf 'LIBERO_INIT_STATES_DIR=%s\n' "${LIBERO_INIT_STATES_DIR}"
printf 'LIBERO_OVERWRITE_CONFIG=%s\n' "${LIBERO_OVERWRITE_CONFIG}"
printf 'GLVND_RUNTIME_DIR=%s\n' "${GLVND_RUNTIME_DIR}"

for required_path in "${ENV_DIR}/bin/lerobot-eval" "${POLICY_PATH}" "${LEROBOT_ROOT}" "${LIBERO_BDDL_DIR}" "${LIBERO_INIT_STATES_DIR}" "${LIBERO_ASSETS_DIR}" "${GLVND_RUNTIME_DIR}/libEGL.so.0" "${GLVND_RUNTIME_DIR}/libOpenGL.so.0"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "missing required path: ${required_path}" >&2
    exit 2
  fi
done

cd "${LEROBOT_ROOT}"
for suite in ${SUITES}; do
  suite_output="${OUTPUT_ROOT}/${suite}"
  mkdir -p "${suite_output}"
  echo "========== LIBERO-UMI suite: ${suite} ==========" | tee "${OUTPUT_ROOT}/logs/${suite}.log"
  lerobot-eval \
    --output_dir="${suite_output}" \
    --env.type=libero \
    --env.task="${suite}" \
    --eval.batch_size=1 \
    --eval.n_episodes="${N_EPISODES_PER_TASK}" \
    --policy.path="${POLICY_PATH}" \
    --policy.n_action_steps=10 \
    --env.max_parallel_tasks=1 2>&1 | tee -a "${OUTPUT_ROOT}/logs/${suite}.log"
done

"${ENV_DIR}/bin/python" "${PROJECT_ROOT}/simulation_evaluation/libero_umi/summarize_libero_umi.py" \
  --output-root "${OUTPUT_ROOT}" \
  --suites ${SUITES} \
  --episodes-per-task "${N_EPISODES_PER_TASK}" \
  --summary-path "${SUMMARY_PATH}"

echo "summary: ${SUMMARY_PATH}"
