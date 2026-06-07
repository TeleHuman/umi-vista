# Cross-Embodiment Replay And Score

This tool replays UMI-style trajectories in robot-specific MuJoCo models and writes trajectory quality scores.

## Installation

We tested the project on:

    Ubuntu 22.04

(1) Install Python 3:

```
apt update
apt install -y python3 python3-pip
```

(2) Install the requirements:

```
python3 -m pip install -r requirements.txt
```

(3) Prepare robot model assets:

```
bash prepare_model_assets.sh
```

## Usage - Replay And Score In Simulation

(1) Configure runtime parameters:

Set RUN_MODE_REAL=0.

Set START_VIEWER / QUIET_MODE / LOG_FLAG according to your needs.

`SCORE_FOLDER_PATH` must be set to the task trajectory folder:

```
export SCORE_FOLDER_PATH=/path/to/task_path
```

Set FILE_TYPE according to the dataset format.

Set FILE_SEARCH_LEVEL according to the subdirectory depth.

    Example 1: FILE_SEARCH_LEVEL=1 when _DEFAULT_SCORE_FOLDER_PATH = "/task_path/" and the folder structure is

```
/task_path/
│
├── session_001/
├── session_002/
└── .../
```

    Example 2: FILE_SEARCH_LEVEL=3 when _DEFAULT_SCORE_FOLDER_PATH = "/task_path/" and the folder structure is

```
/task_path/
│
├── subdir_1/
    │
    ├── multi_session_001/
	│
	├── session_001/
	├── session_002/
	└── .../
    ├── multi_session_002/
    └── .../
├── subdir_2/
└── .../
```

(2) Replay and score one trajectory:

```
export ROBOT_NAME="rm75" # or acone/r1pro
python3 replay.py <index> # eg. python3 replay.py 0. Here <index> is the index of the file folder named by session_***.
```

(3) Score all trajectories in one task on a single robot embodiment:

Set _DEFAULT_SCORE_FOLDER_PATH and FILE_SEARCH_LEVEL, then

```
export SCORE_FOLDER_PATH=/path/to/task_path
export ROBOT_NAME="rm75" # or acone/r1pro
python3 replay.py
```

(4) Score all trajectories in one task across all three robot embodiments:

```
./run_replay_all.sh
```

(5) Score all trajectories for multiple tasks across all three robot embodiments:

```
./run_replay_all_batch.sh
```

Set `BASE_DIRS` to a space-separated list of the parent directories of the task paths.

For example:

```
BASE_DIRS="/parent_dir1_of_tasks /parent_dir2_of_tasks" ./run_replay_all_batch.sh
```

for the following folder structure:

```
/parent_dir1_of_tasks/
│
├── task_path1/
├── task_path2/
└── .../
/parent_dir2_of_tasks/
│
├── task_path3/
├── task_path4/
└── .../
```

(6) Two files containing the score results are generated in log/ for each task after scoring. They are named as follows:

```
<ROBOT_NAME>_score_results_<time_stamp>.txt
<ROBOT_NAME>_score_summary_<time_stamp>.txt
```

Each row in <ROBOT_NAME>_score_results_<time_stamp>.txt stores the score of one trajectory in the task.

<ROBOT_NAME>_score_summary_<time_stamp>.txt summarizes the distribution, counts, duration, and other statistics.

## Usage - Replay On Real Robot

Replay on a real robot depends on robot-specific paths and topic names (note: acone/r1pro uses ROS2). The hardware interfaces in this repository have been validated only on our robot setup. To deploy on your robot, update the configuration to match your hardware setup.

RUN_MODE_REAL=1 and other parameters are set similarly to replay in the simulation.

```
./start_acone.sh # or start_r1pro.sh or start_rm75.sh
python3 replay.py <index>
```

## License

This project is licensed under the MIT License.

## Note

The task path must be set correctly, or the program may crash. The paths in the current files are only examples from the authors' machine.
