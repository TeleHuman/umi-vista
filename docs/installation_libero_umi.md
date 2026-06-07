# LIBERO-UMI Evaluation Installation

This guide assumes the VISTA fine-tuning installation has already been completed. Follow `docs/installation_finetuning.md` first so the conda environment, LeRobot package, and converted VISTA checkpoints are available.

## Environment

Use the same conda environment and project paths:

```bash
source /data/ysy/app/miniconda3/etc/profile.d/conda.sh
conda activate rhodeslerobot_pi_temp
export PROJECT_ROOT=/data_ysy/app/codex_space/project/2026_06_05/umi-vista
export LEROBOT_ROOT=${PROJECT_ROOT}/post_training/lerobot
cd ${LEROBOT_ROOT}
bash third_party/prepare_pi_transformers.sh
pip install --no-build-isolation -e ".[pi,libero]"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH=/root/.libero
export ENV_DIR=/data/ysy/app/miniconda3/envs/rhodeslerobot_pi_temp
export LIBERO_PACKAGE_DIR=${ENV_DIR}/lib/python3.10/site-packages/libero/libero
export LIBERO_DATASETS_DIR=/data_ysy/app/codex_space/share/libero_datasets/datasets
export LIBERO_ASSETS_DIR=${LIBERO_PACKAGE_DIR}/assets
export LIBERO_BDDL_DIR=${LIBERO_PACKAGE_DIR}/bddl_files
export LIBERO_INIT_STATES_DIR=${LIBERO_PACKAGE_DIR}/init_files
export LIBERO_OVERWRITE_CONFIG=0
export GLVND_RUNTIME_DIR=/data_ysy/app/codex_space/share/glvnd_runtime
export LD_LIBRARY_PATH=${GLVND_RUNTIME_DIR}:${LD_LIBRARY_PATH:-}
```

The runner writes `LIBERO_CONFIG_PATH/config.yaml` when it is missing. Set `LIBERO_OVERWRITE_CONFIG=1` to regenerate it from the exported paths. This avoids LIBERO's first-import dataset prompt and keeps runtime dataset/cache files under `LIBERO_DATASETS_DIR`. It also prepends `GLVND_RUNTIME_DIR` to `LD_LIBRARY_PATH` so MuJoCo can find `libEGL.so.0` and `libOpenGL.so.0` on headless machines.

## LIBERO Assets And Config

The evaluated LIBERO package reads runtime paths from `LIBERO_CONFIG_PATH/config.yaml`. The validated environment uses:

```text
LIBERO package:      /data/ysy/app/miniconda3/envs/rhodeslerobot_pi_temp/lib/python3.10/site-packages/libero/libero
assets:              /data/ysy/app/miniconda3/envs/rhodeslerobot_pi_temp/lib/python3.10/site-packages/libero/libero/assets
bddl_files:          /data/ysy/app/miniconda3/envs/rhodeslerobot_pi_temp/lib/python3.10/site-packages/libero/libero/bddl_files
init_states:         /data/ysy/app/miniconda3/envs/rhodeslerobot_pi_temp/lib/python3.10/site-packages/libero/libero/init_files
datasets/cache root: /data_ysy/app/codex_space/share/libero_datasets/datasets
config file:         /root/.libero/config.yaml
```

On this host, the installed `libero` package already contains the simulation assets, BDDL files, and initial states. Their approximate sizes are:

```text
assets       405M
bddl_files   572K
init_files    13M
```

Create the config file manually when running LIBERO outside the project runner:

```bash
mkdir -p ${LIBERO_CONFIG_PATH} ${LIBERO_DATASETS_DIR}
cat > ${LIBERO_CONFIG_PATH}/config.yaml <<EOF
assets: ${LIBERO_ASSETS_DIR}
bddl_files: ${LIBERO_BDDL_DIR}
benchmark_root: ${LIBERO_PACKAGE_DIR}
datasets: ${LIBERO_DATASETS_DIR}
init_states: ${LIBERO_INIT_STATES_DIR}
EOF
```

Check the resolved LIBERO paths:

```bash
python - <<'PY'
from libero.libero import get_libero_path

for key in ["benchmark_root", "assets", "bddl_files", "init_states", "datasets"]:
    print(f"{key}: {get_libero_path(key)}")
PY
```

The VISTA LIBERO-UMI evaluation does not require downloading official LIBERO HDF5 demonstration datasets. It evaluates in simulation with the package assets, BDDL files, and initial states above. If you need the official LIBERO datasets for other experiments, download them into the configured dataset directory:

```bash
python - <<'PY'
from pathlib import Path
from libero.libero.utils.download_utils import libero_dataset_download

download_dir = "/data_ysy/app/codex_space/share/libero_datasets/datasets"
for name in ["libero_object", "libero_goal", "libero_spatial"]:
    libero_dataset_download(
        datasets=name,
        download_dir=download_dir,
        check_overwrite=False,
        use_huggingface=True,
    )

for name in ["libero_object", "libero_goal", "libero_spatial"]:
    hdf5_files = sorted((Path(download_dir) / name).glob("*.hdf5"))
    print(f"{name}: {len(hdf5_files)} hdf5 files")
PY
```

If an installed `libero` package does not include `assets`, download the assets with Hugging Face Hub support and place them where `LIBERO_ASSETS_DIR` points:

```bash
python - <<'PY'
from libero.libero.utils.download_utils import download_assets_from_huggingface

download_assets_from_huggingface(
    "/data/ysy/app/miniconda3/envs/rhodeslerobot_pi_temp/lib/python3.10/site-packages/libero/libero/assets"
)
PY
```

The asset downloader uses the `jadechoghari/libero-assets` Hugging Face repo in this installed LIBERO package. The official dataset downloader uses the `yifengzhu-hf/LIBERO-datasets` dataset repo when `use_huggingface=True`.

## Checkpoint

The default VISTA LIBERO-UMI checkpoint is:

```text
/data_ysy/app/codex_space/share/vista_checkpoints/vista_libero_umi_v2_40k/pretrained_model
```

The checkpoint metadata must contain `"type": "vista"` in `config.json`.

## Evaluation

The project runner evaluates every task in each selected LIBERO suite. By default, `N_EPISODES_PER_TASK=50`, which means 50 episodes for every task, not 50 total episodes per suite.

```bash
cd /data_ysy/app/codex_space/project/2026_06_05/umi-vista
GPU_ID=0 \
bash simulation_evaluation/libero_umi/run_libero_umi_eval.sh
```

The runner writes per-suite outputs, logs, and `summary.tsv` under `OUTPUT_ROOT`. By default, `OUTPUT_ROOT` is a timestamped directory under:

```text
/data_ysy/app/codex_space/temp/libero_umi_eval
```

## Quick Check

For a shorter five-episodes-per-task check, override the default:

```bash
N_EPISODES_PER_TASK=5 bash simulation_evaluation/libero_umi/run_libero_umi_eval.sh
```

Historical full-run reference rates for the VISTA predecessor checkpoint are:

```text
libero_10      88.8%
libero_goal    91.6%
libero_object  98.8%
libero_spatial 97.8%
```

## Summary TSV

The summary file contains:

```text
suite
pc_success
n_tasks
episodes_per_task
total_episodes
status
eval_info
```

Use `pc_success` for each suite and the `Avg.` row for the mean across completed suites.
