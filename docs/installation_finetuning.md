# VISTA Fine-Tuning Installation

This guide sets up the environment for VISTA fine-tuning smoke tests and downstream fine-tuning.

## Environment

Use the shared conda environment:

```bash
source /data/ysy/app/miniconda3/etc/profile.d/conda.sh
conda activate rhodeslerobot_pi_temp
```

Set the project paths before running training commands:

```bash
export PROJECT_ROOT=/data_ysy/app/codex_space/project/2026_06_05/umi-vista
export LEROBOT_ROOT=${PROJECT_ROOT}/post_training/lerobot
export PYTHONNOUSERSITE=1
```

## Prepare the Vendored Transformers Dependency

VISTA keeps the LeRobot-compatible Transformers fork as a zip archive to keep the repository small. Prepare it before installing or running LeRobot:

```bash
cd ${LEROBOT_ROOT}
bash third_party/prepare_pi_transformers.sh
```

This creates:

```text
${LEROBOT_ROOT}/third_party/pi_transformers
```

The generated directory is a local installation artifact and should not be committed. The archive that should stay in the repository is:

```text
${LEROBOT_ROOT}/third_party/transformers-dcddb970176382c0fcf4521b0c0e6fc15894dfe0.zip
```

Install the copied LeRobot package after the archive has been prepared:

```bash
cd ${LEROBOT_ROOT}
pip install --no-build-isolation -e ".[pi]"
export PYTHONPATH=${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/third_party/pi_transformers/src:${PYTHONPATH:-}
```

## Required Inputs

The smoke test uses the shared LIBERO-UMI LeRobot dataset:

```text
/data/guolinzheng/umi_work_space_0324/umi_dataset_for_bench/simulation/v30/libero_umi_v2_ee_lerobot_720_150_0.6
```

Use the converted VISTA two-stage base checkpoint:

```text
/data_ysy/app/codex_space/share/vista_checkpoints/vista_pres2_16w/pretrained_model
```

The checkpoint metadata must contain `"type": "vista"` in `config.json`.

## Fine-Tuning Smoke Test

Run the smoke test from the project root or the LeRobot root. Outputs should stay under `/data_ysy/app/codex_space/temp`.

```bash
cd /data_ysy/app/codex_space/project/2026_06_05/umi-vista/post_training/lerobot
bash training_scripts/vista_finetune_smoke_test.sh
```

The launcher accepts environment overrides. For example:

```bash
POLICY_PATH=/data_ysy/app/codex_space/share/vista_checkpoints/vista_pres2_16w/pretrained_model \
OUTPUT_ROOT=/data_ysy/app/codex_space/temp/lerobot_finetune_smoke_test \
TOTAL_STEPS=2 \
GPU_IDS=0 \
bash training_scripts/vista_finetune_smoke_test.sh
```

A successful run prints both training steps, saves a checkpoint, and ends with exit code 0. Expected log markers:

```text
step:1
step:2
Checkpoint policy after step 2
End of training
smoke test exit code: 0
```

The saved smoke-test checkpoint should also have `config.json` with `"type": "vista"`.
