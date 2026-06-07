<div align="center">

# VISTA: Vision-Grounded and Physics-Validated Adaptation of UMI data for VLA Training

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2606.04708)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://tele-umi-vista.github.io)
[![Datasets](https://img.shields.io/badge/Datasets-HuggingFace-yellow.svg)](https://huggingface.co/collections/TeleEmbodied/vista)
[![Models](https://img.shields.io/badge/Models-HuggingFace-orange.svg)](https://huggingface.co/collections/TeleEmbodied/vista)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

<img src="assets/teaser.png" alt="VISTA overview" width="95%">

</div>

## Overview

VISTA adapts UMI-collected demonstrations for VLA policy training. It addresses two practical gaps: wrist-fisheye observations are out of distribution for pretrained vision-language models, and human-collected trajectories can be physically infeasible for a target robot. The released code includes:

- VISTA policy integration in LeRobot for post-training and downstream fine-tuning.
- LIBERO-UMI evaluation runners for VISTA checkpoints.
- Cross-embodiment physical validation tools for replaying and scoring UMI-style trajectories.

Datasets and model checkpoints are released through the Hugging Face collection:

```text
https://huggingface.co/collections/TeleEmbodied/vista
```

## Repository Layout

```text
umi-vista/
  assets/                                           # README images and lightweight media
  docs/                                             # Detailed installation and evaluation guides
  post_training/lerobot/                            # Vendored LeRobot with VISTA policy support
  simulation_evaluation/libero_umi/                 # LIBERO-UMI runner and summary script
  physical_validation/cross_embodiment_replay_and_score/
                                                    # Physical replay and scoring tools
```

RoboTwin-UMI evaluation code is planned for a later release.

## Choose What To Install

You can install only the part you need:

- **Post-training / fine-tuning:** install LeRobot with the VISTA policy and the local Transformers fork.
- **LIBERO-UMI evaluation:** install the post-training environment plus LIBERO simulation dependencies.
- **Physical validation:** install the replay/scoring Python requirements and unpack robot model assets.

The commands below assume Linux, Python 3.10, a CUDA-capable PyTorch installation, and a fresh clone of this repository.

```bash
git clone https://github.com/TeleHuman/umi-vista.git
cd umi-vista
export VISTA_ROOT=$PWD
```

Create and activate an environment:

```bash
conda create -n vista python=3.10 -y
conda activate vista
python -m pip install --upgrade pip
```

Install a PyTorch build that matches your CUDA driver before installing LeRobot. For example, follow the command generated at:

```text
https://pytorch.org/get-started/locally/
```

## Post-Training And Fine-Tuning

Prepare the vendored Transformers fork and install LeRobot with VISTA policy support:

```bash
export LEROBOT_ROOT=${VISTA_ROOT}/post_training/lerobot
cd ${LEROBOT_ROOT}
bash third_party/prepare_pi_transformers.sh
pip install --no-build-isolation -e ".[pi]"
```

`prepare_pi_transformers.sh` unpacks the repository archive:

```text
post_training/lerobot/third_party/transformers-dcddb970176382c0fcf4521b0c0e6fc15894dfe0.zip
```

into:

```text
post_training/lerobot/third_party/pi_transformers/
```

The editable install then uses this local Transformers fork through the LeRobot `pi` optional dependency.

Download a VISTA base checkpoint and a LeRobot-format training dataset from the Hugging Face collection, then launch fine-tuning with the same parameter scale used for VISTA UMI fine-tuning:

```bash
cd ${LEROBOT_ROOT}

export DATASET_REPO_ID=/path/to/lerobot_dataset
export DATASET_ROOT=${DATASET_REPO_ID}
export DATASET_REVISION=v2.0
export POLICY_PATH=/path/to/vista_base_checkpoint/pretrained_model
export OUTPUT_ROOT=/path/to/vista_outputs/fine_tuning

export GPU_IDS=0
export GPUS=1
export MAIN_PROCESS_PORT=29510
export BATCH_SIZE=32
export TOTAL_STEPS=40000
export SAVE_FREQ=10000
export LOG_FREQ=50
export LEARNING_RATE=5e-5
export ACTION_CHUNK_SIZE=50
export NUM_WORKERS=32
export SEED=42
export GRADIENT_CHECKPOINTING=true

DATE_TAG=$(date "+%y-%m-%d_%H-%M-%S")
DATASET_NAME="${DATASET_REPO_ID##*/}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${DATE_TAG}_vista_gpu${GPUS}_ck${ACTION_CHUNK_SIZE}_lr5e-5_bs${BATCH_SIZE}_s40K_seed${SEED}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} accelerate launch \
  --num_processes=${GPUS} \
  --main_process_port=${MAIN_PROCESS_PORT} \
  src/lerobot/scripts/lerobot_train_umi.py \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.revision=${DATASET_REVISION} \
  --dataset.image_transforms.enable=false \
  --dataset.wrist_transforms.enable=true \
  --policy.dtype=float32 \
  --policy.path=${POLICY_PATH} \
  --policy.push_to_hub=false \
  --policy.chunk_size=${ACTION_CHUNK_SIZE} \
  --policy.n_action_steps=${ACTION_CHUNK_SIZE} \
  --policy.optimizer_lr=${LEARNING_RATE} \
  --policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
  --policy.scheduler_decay_steps=36000 \
  --policy.use_delta_action=true \
  --output_dir=${OUTPUT_DIR} \
  --batch_size=${BATCH_SIZE} \
  --steps=${TOTAL_STEPS} \
  --save_freq=${SAVE_FREQ} \
  --log_freq=${LOG_FREQ} \
  --num_workers=${NUM_WORKERS} \
  --enforce_input_output_replace=true \
  --seed=${SEED}
```

The saved checkpoint metadata should contain:

```json
{"type": "vista"}
```

For a short environment validation run, use the smoke-test wrapper documented in [docs/installation_finetuning.md](docs/installation_finetuning.md).

## LIBERO-UMI Evaluation

LIBERO-UMI evaluation builds on the post-training installation. Install LeRobot with the LIBERO extra:

```bash
cd ${LEROBOT_ROOT}
bash third_party/prepare_pi_transformers.sh
pip install --no-build-isolation -e ".[pi,libero]"
```

Set the checkpoint and runtime paths. By default, the runner evaluates 50 episodes for every task in each selected suite, not 50 total episodes per suite.

```bash
cd ${VISTA_ROOT}
POLICY_PATH=/path/to/vista_libero_umi_checkpoint/pretrained_model \
OUTPUT_ROOT=/path/to/vista_outputs/libero_umi_eval \
GPU_ID=0 \
bash simulation_evaluation/libero_umi/run_libero_umi_eval.sh
```

By default the runner evaluates:

```text
libero_10 libero_goal libero_object libero_spatial
```

Override `SUITES` to run a subset:

```bash
SUITES="libero_goal" \
POLICY_PATH=/path/to/vista_libero_umi_checkpoint/pretrained_model \
bash simulation_evaluation/libero_umi/run_libero_umi_eval.sh
```

Run a quick five-episodes-per-task check by explicitly overriding the default:

```bash
POLICY_PATH=/path/to/vista_libero_umi_checkpoint/pretrained_model \
N_EPISODES_PER_TASK=5 \
bash simulation_evaluation/libero_umi/run_libero_umi_eval.sh
```

The runner writes per-suite logs, `eval_info.json` files, and a `summary.tsv` under `OUTPUT_ROOT`.

If your LIBERO installation does not already include assets, BDDL files, or initial states, see [docs/installation_libero_umi.md](docs/installation_libero_umi.md) for the required `LIBERO_CONFIG_PATH`, asset paths, and optional download commands.

## Physical Validation

The physical-validation tool replays UMI-style trajectories in robot-specific MuJoCo models and writes trajectory quality scores.

```bash
cd ${VISTA_ROOT}/physical_validation/cross_embodiment_replay_and_score
python3 -m pip install -r requirements.txt
bash prepare_model_assets.sh
```

`prepare_model_assets.sh` unpacks:

```text
physical_validation/cross_embodiment_replay_and_score/model_assets.tar.gz
```

and creates the generated directory:

```text
physical_validation/cross_embodiment_replay_and_score/model/
```

Run simulation scoring for one robot:

```bash
export SCORE_FOLDER_PATH=/path/to/task_trajectory_folder
export ROBOT_NAME=rm75
python3 replay.py
```

Supported robot names are:

```text
acone
r1pro
rm75
```

Run one indexed trajectory:

```bash
export SCORE_FOLDER_PATH=/path/to/task_trajectory_folder
export ROBOT_NAME=r1pro
python3 replay.py 0
```

Score one task folder across all supported robots:

```bash
export SCORE_FOLDER_PATH=/path/to/task_trajectory_folder
bash run_replay_all.sh
```

Score all task folders under one or more parent directories:

```bash
BASE_DIRS="/path/to/task_parent_a /path/to/task_parent_b" bash run_replay_all_batch.sh
```

Results are written under:

```text
physical_validation/cross_embodiment_replay_and_score/log/
```

## Detailed Guides

- [Fine-tuning installation](docs/installation_finetuning.md)
- [LIBERO-UMI evaluation installation](docs/installation_libero_umi.md)
- [Physical validation installation](docs/installation_physical_validation.md)

## Citation

If you find VISTA useful, please cite the paper:

```bibtex
@misc{yang2026vistavisiongroundedphysicsvalidatedadaptation,
      title={VISTA: Vision-Grounded and Physics-Validated Adaptation of UMI data for VLA Training}, 
      author={Siyuan Yang and Linzheng Guo and Ouyang Lu and Zhaxizhuoma and Daoran Zhang and Xinmiao Wang and Ting Xiao and Fangzheng Yan and Zhijun Chen and Yan Ding and Chao Yu and Chenjia Bai and Xuelong Li},
      year={2026},
      eprint={2606.04708},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.04708}, 
}
```

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
