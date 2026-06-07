# Physical Validation Installation

This guide sets up the VISTA physical-validation replay and scoring tool.

## Environment

The tool was tested on Ubuntu 22.04 with Python 3.10. Use an isolated environment when possible:

```bash
cd /data_ysy/app/codex_space/project/2026_06_05/umi-vista/physical_validation/cross_embodiment_replay_and_score
python3 -m pip install -r requirements.txt
```

Prepare robot model assets before running replay or scoring:

```bash
bash prepare_model_assets.sh
```

This creates a generated local directory:

```text
physical_validation/cross_embodiment_replay_and_score/model
```

The generated `model/` directory is ignored by git. The archive that should stay in the repository is:

```text
physical_validation/cross_embodiment_replay_and_score/model_assets.tar.gz
```

## Simulation Scoring

Set the task trajectory folder and choose one robot embodiment:

```bash
cd /data_ysy/app/codex_space/project/2026_06_05/umi-vista/physical_validation/cross_embodiment_replay_and_score
bash prepare_model_assets.sh
export SCORE_FOLDER_PATH=/path/to/task_path
export ROBOT_NAME=rm75
python3 replay.py
```

Supported robot names are:

```text
acone
r1pro
rm75
```

To replay one indexed trajectory instead of scoring all trajectories:

```bash
export SCORE_FOLDER_PATH=/path/to/task_path
export ROBOT_NAME=r1pro
python3 replay.py 0
```

To score the same task for all three robot embodiments:

```bash
export SCORE_FOLDER_PATH=/path/to/task_path
bash run_replay_all.sh
```

To score all task folders under one or more parent directories:

```bash
BASE_DIRS="/path/to/parent1 /path/to/parent2" bash run_replay_all_batch.sh
```

Each task writes result files under:

```text
physical_validation/cross_embodiment_replay_and_score/log
```

The main outputs are:

```text
<ROBOT_NAME>_score_results_<timestamp>.txt
<ROBOT_NAME>_score_summary_<timestamp>.txt
```

## Real-Robot Replay

Real-robot replay depends on robot-specific ROS, network, USB, and SDK setup. The provided startup scripts are examples from the validated lab setup and must be adapted before use on another robot:

```bash
bash start_acone.sh
bash start_r1pro.sh
bash start_rm75.sh
```

After starting the robot stack, run:

```bash
export SCORE_FOLDER_PATH=/path/to/task_path
export ROBOT_NAME=<acone|r1pro|rm75>
python3 replay.py 0
```

## Sanity Checks

Check that the scripts parse and the required task path error is clear:

```bash
bash -n prepare_model_assets.sh
bash -n run_replay_all.sh
bash -n run_replay_all_batch.sh
python3 replay.py
```

The last command should fail until `SCORE_FOLDER_PATH` points to a real task trajectory folder.
