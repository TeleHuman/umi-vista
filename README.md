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
export PYTHONPATH=${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/third_party/pi_transformers/src:${PYTHONPATH:-}
```

`prepare_pi_transformers.sh` unpacks the repository archive:

```text
post_training/lerobot/third_party/transformers-dcddb970176382c0fcf4521b0c0e6fc15894dfe0.zip
```

and creates the generated directory:

```text
post_training/lerobot/third_party/pi_transformers/
```

The generated directory is ignored by git and should not be committed.

Download a VISTA base checkpoint and a LeRobot-format training dataset from the Hugging Face collection, then point the launcher at those local paths:

```bash
cd ${LEROBOT_ROOT}
DATASET_REPO_ID=/path/to/lerobot_dataset \
DATASET_ROOT=/path/to/lerobot_dataset \
POLICY_PATH=/path/to/vista_base_checkpoint/pretrained_model \
OUTPUT_ROOT=/path/to/vista_outputs/lerobot_finetune_smoke_test \
TOTAL_STEPS=2 \
GPU_IDS=0 \
bash training_scripts/vista_finetune_smoke_test.sh
```

The smoke test should print `step:1`, `step:2`, save a checkpoint, and finish with `smoke test exit code: 0`. The checkpoint metadata should contain:

```json
{"type": "vista"}
```

For full fine-tuning, reuse the same launcher with larger values for `TOTAL_STEPS`, `SAVE_FREQ`, `BATCH_SIZE`, `GPU_IDS`, and `OUTPUT_ROOT`.

## LIBERO-UMI Evaluation

LIBERO-UMI evaluation builds on the post-training installation. Install LeRobot with the LIBERO extra:

```bash
cd ${LEROBOT_ROOT}
bash third_party/prepare_pi_transformers.sh
pip install --no-build-isolation -e ".[pi,libero]"
export PYTHONPATH=${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/third_party/pi_transformers/src:${PYTHONPATH:-}
```

Set the checkpoint and runtime paths. `N_EPISODES_PER_TASK=5` means five episodes for every task in each selected suite, not five total episodes per suite.

```bash
cd ${VISTA_ROOT}
POLICY_PATH=/path/to/vista_libero_umi_checkpoint/pretrained_model \
OUTPUT_ROOT=/path/to/vista_outputs/libero_umi_eval \
N_EPISODES_PER_TASK=5 \
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
N_EPISODES_PER_TASK=5 \
bash simulation_evaluation/libero_umi/run_libero_umi_eval.sh
```

Run a full 50-episodes-per-task evaluation with:

```bash
POLICY_PATH=/path/to/vista_libero_umi_checkpoint/pretrained_model \
N_EPISODES_PER_TASK=50 \
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

The generated `model/` directory is ignored by git and should not be committed.

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
