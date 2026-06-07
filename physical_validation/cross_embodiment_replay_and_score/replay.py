# -*- coding: utf-8 -*-

# UMI trajectory replay and scoring.

import time, math, os, sys
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from termcolor import cprint
import multiprocessing as mp

# Set this via env var: export SCORE_FOLDER_PATH=/path/to/task_folder
_DEFAULT_SCORE_FOLDER_PATH = ""


def _resolve_score_folder_path() -> str:
    score_folder_path = (os.environ.get("SCORE_FOLDER_PATH") or "").strip() or _DEFAULT_SCORE_FOLDER_PATH
    if not score_folder_path:
        raise RuntimeError("SCORE_FOLDER_PATH is required. Set it to a task trajectory folder before running replay.py.")
    return score_folder_path


try:
    SCORE_FOLDER_PATH = _resolve_score_folder_path()
except RuntimeError as exc:
    print(f"[CONFIG] {exc}", file=sys.stderr)
    sys.exit(2)

import utils.self_math as smath
import utils.lowpass_filter as flt
from utils.logger import TrajLogger
from utils.folder_indexer import FolderIndexer
from utils.common import concat_qpos_parts, State

ROBOT = os.environ.get("ROBOT_NAME", "r1pro").strip().lower()  # rm75, r1pro, acone, telebotw, rm75_side, sailor
os.environ["ROBOT_NAME"] = ROBOT
if ROBOT == "rm75":
    from robot.rm75_double_mink import DoubleRm75Mink as RbtKin
    import robot.rm75_interface as RbtCtrl
    print("[ROBOT] Using rm75 robot model and interface")
elif ROBOT == "r1pro":
    from robot.r1pro_mink import R1ProMink as RbtKin
    import robot.r1pro_interface as RbtCtrl
    print("[ROBOT] Using r1pro robot model and interface")
elif ROBOT == "acone":
    from robot.acone_mink import AconeMink as RbtKin
    import robot.acone_interface as RbtCtrl
    print("[ROBOT] Using acone robot model and interface")

from robot.robot_joint_config import (
    N_CHASSIS,
    N_LEFT_ARM,
    N_LEFT_GRIPPER,
    N_QPOS_TOTAL,
    N_RIGHT_ARM,
    N_RIGHT_GRIPPER,
    N_TORSO,
    SL_CHASSIS,
    SL_LEFT_ARM,
    SL_LEFT_GRIPPER,
    SL_RIGHT_ARM,
    SL_RIGHT_GRIPPER,
    SL_TORSO,
    GRIPPER_OPEN,
    GRIPPER_CLOSE,
    _fatal,
    _load_qpos_vector_from_cfg,
    _require_1d_float_array,
    _require_fb_vec,
)

# ---------------- Configuration ----------------
RUN_MODE_REAL     = 0 # 1=real robot, 0=simulation; simulation has score and replay modes
START_VIEWER      = 1 # 1=start viewer (simulation only), 0=disable; forced to 0 in score mode
QUIET_MODE        = 0 # 1=suppress all prints except exceptions; 0=normal prints; forced to 1 in score mode
LOG_FLAG          = True # whether to record replay logs (simulation only); forced to 0 in score mode
FILE_TYPE         = "txt" # "txt" or "parquet", trajectory file type for replay (supports LeRobot parquet v2.1 single-episode files and v3.0 multi-episode single files)
REPLAY_TYPE       = 0 # 0=pose; 1=jnt;
FILE_SEARCH_LEVEL = 3 # Subdirectory traversal depth
if FILE_TYPE == "parquet":
    REPLAY_TYPE = 0
elif FILE_TYPE == "txt":
    if REPLAY_TYPE == 0:
        FILE_SEARCH_MODE = "session_ee"
    elif REPLAY_TYPE == 1:
        FILE_SEARCH_MODE = "session_jnt"
else:
    raise ValueError(f"Unsupported FILE_TYPE={FILE_TYPE!r} (expected 'txt' or 'parquet')")
    
CONTROL_HZ        = 50.0
SCALE_POS_L       = 1.0
SCALE_POS_R       = 1.0

R_W2B_L = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype=float)
R_W2B_R = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype=float)

LPF_CUTOFF_HZ     = 9.0
MAX_STEP_XYZ      = 0.1
MAX_STEP_ROT      = 0.1

WS_MIN_L          = np.array([ -0.2, -0.7,  0.20])
WS_MAX_L          = np.array([  0.8,  0.7,  1.80])
WS_MIN_R          = np.array([ -0.2, -0.7,  0.20])
WS_MAX_R          = np.array([  0.8,  0.7,  1.80])

# At HOME_POS pose, gripper-to-table height difference is 14.65 cm, consistent with UMI.
HOME_POS = _load_qpos_vector_from_cfg("home_pos")
EXIT_POS = _load_qpos_vector_from_cfg("exit_pos")
MAX_STEP_RAD = _load_qpos_vector_from_cfg("max_step_rad")
STEP_SCALE     = 1.0 # The effective scaling factor is MAX_STEP_RAD * STEP_SCALE
ACC_LMT        = 2
ADJUST_EE_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # End-effector adjustment offsets: left xyz + right xyz.
WAIT_TIME      = 0.5 # Wait duration during state transitions
HOME_TIME      = 5.0
ADJUST_TIME    = 2.0 # Duration of end-effector adjustment
    
# Unified print wrapper (controlled by QUIET_MODE)
def _info(msg: str):
    try:
        if not QUIET_MODE:
            print(msg)
    except Exception:
        pass

# ---------- Txt Trajectory loading ----------
def _load_joint_traj(file_path: str, joint_count: int) -> np.ndarray:
    """Load a joint trajectory text file.

    Returns an array shaped (N, 1+joint_count): [t, q1..qK].
    For compatibility:
    - If the file has only K columns (no time), a zero time column is prepended.
    - If joint_count==0, returns (N,1) with only the time column.
    """
    joint_count = int(joint_count)
    if not os.path.isfile(file_path):
        _fatal(f"trajectory file not found: {file_path}")
    arr = np.loadtxt(file_path, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        _fatal(f"Empty joint trajectory file: {file_path}")

    if joint_count <= 0:
        if arr.shape[1] != 1:
            _fatal(f"Unexpected column count in {file_path}: got {arr.shape[1]}, expected 1 (time only)")
        return np.asarray(arr[:, :1], dtype=float)

    expected_with_t = 1 + joint_count
    if arr.shape[1] == expected_with_t:
        return np.asarray(arr, dtype=float)
    if arr.shape[1] == joint_count:
        # Allowed format: no time column
        t = np.zeros((arr.shape[0], 1), dtype=float)
        return np.concatenate([t, np.asarray(arr, dtype=float)], axis=1)
    _fatal(f"Unexpected column count in {file_path}: got {arr.shape[1]}, expected {joint_count} (no time) or {expected_with_t} (with time)")
    return np.zeros((0, expected_with_t), dtype=float)

def _load_pose_traj(file_path: str) -> np.ndarray:
    """Load pose trajectory: columns = [t, x, y, z, qx, qy, qz, qw]. Returns (N,8)."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"trajectory file not found: {file_path}")
    arr = np.loadtxt(file_path, dtype=float)
    if arr.ndim == 1:
        if arr.size == 8:
            return arr.reshape(1, 8)
        else:
            raise ValueError(f"Unexpected columns in {file_path}: {arr.size}")
    if arr.shape[1] != 8:
        raise ValueError(f"Expected 8 columns in {file_path}, got {arr.shape[1]}")
    return arr

def _load_gripper_traj(file_path: str) -> np.ndarray:
    """Load gripper trajectory from clamp/gripper file.

    Returns (N, 2): [t, qpos], where the first column is timestamp
    and the second column is gripper qpos.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"gripper file not found: {file_path}")
    arr = np.loadtxt(file_path, dtype=float)
    if arr.ndim == 1:
        # Single row: need at least time and qpos
        if arr.size >= 2:
            return np.array([[float(arr[0]), float(arr[1])]], dtype=float)
        else:
            raise ValueError(f"Unexpected gripper row format in {file_path}: {arr}")
    # Multi-row: ensure we have at least two columns [t, qpos]
    if arr.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in gripper file {file_path}, got {arr.shape[1]}")
    # Keep only first two columns: [t, qpos]
    return np.asarray(arr[:, :2], dtype=float)


# ---------- LeRobot dataset loading (parquet) ----------
def _detect_episode_column(col_names) -> Optional[str]:
    """Pick an episode label column name from parquet columns."""
    if not col_names:
        return None

    lower_map = {str(c).lower(): str(c) for c in col_names}
    preferred = [
        "episode_index",
        "episode",
        "episode_id",
        "episode_idx",
        "episode_number",
        "episode_name",
    ]
    for k in preferred:
        if k in lower_map:
            return lower_map[k]

    for c in col_names:
        try:
            if "episode" in str(c).lower():
                return str(c)
        except Exception:
            continue
    return None


def _list_lerobot_v3_dataset_files(task_dir: str) -> list:
    """List LeRobot v3 dataset parquet files under a dataset directory.

    Expected layout (v3):
      dataset_dir/data/chunk-*/file-*.parquet
    """
    import glob

    base = os.path.abspath(str(task_dir))
    if not os.path.isdir(base):
        return []

    pattern = os.path.join(base, "data", "chunk-*", "file-*.parquet")
    matches = [p for p in glob.glob(pattern) if os.path.isfile(p)]

    def _natural_key(path: str):
        rel = os.path.relpath(path, base).replace(os.sep, "/")
        parts = re.split(r"(\d+)", rel)
        key = []
        for part in parts:
            if part.isdigit():
                try:
                    key.append(int(part))
                except Exception:
                    key.append(part)
            else:
                key.append(part.lower())
        return key

    matches.sort(key=_natural_key)
    return matches


def _infer_lerobot_fps(dataset_dir: str, default_fps: float = 30.0) -> float:
    """Best-effort infer fps from LeRobot dataset metadata files."""
    try:
        import json

        candidates = [
            os.path.join(dataset_dir, "meta.json"),
            os.path.join(dataset_dir, "metadata.json"),
            os.path.join(dataset_dir, "dataset.json"),
            os.path.join(dataset_dir, "info.json"),
        ]
        for p in candidates:
            if not os.path.isfile(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for k in ("fps", "frame_rate", "sampling_rate"):
                if k in meta:
                    fps = float(meta[k])
                    if fps > 0:
                        return fps
    except Exception:
        pass
    return float(default_fps)


def _list_task_parquet_files(task_dir: str) -> list:
    """List LeRobot episode parquet files under a dataset directory.

    Expected layout:
      dataset_dir/data/chunk-*/episode_XXXXXX.parquet

    Sorting:
      Natural-sorted by relative path to keep stable ordering.

    Also supports passing a direct parquet file path.
    """
    import glob

    base = os.path.abspath(str(task_dir))
    if os.path.isfile(base) and base.lower().endswith(".parquet"):
        return [base]
    if not os.path.isdir(base):
        return []

    pattern = os.path.join(base, "data", "chunk-*", "episode_*.parquet")
    matches = [p for p in glob.glob(pattern) if os.path.isfile(p)]

    def _natural_key(path: str):
        rel = os.path.relpath(path, base).replace(os.sep, "/")
        parts = re.split(r"(\d+)", rel)
        key = []
        for part in parts:
            if part.isdigit():
                try:
                    key.append(int(part))
                except Exception:
                    key.append(part)
            else:
                key.append(part.lower())
        return key

    matches.sort(key=_natural_key)
    return matches


def _find_lerobot_episode_parquet(dataset_dir: str, episode_index: int) -> str:
    """Resolve an episode parquet path.

        Supports:
        - LeRobot v2.x layout: `dataset_dir/data/chunk-*/episode_XXXXXX.parquet`.
            `episode_index` selects the Nth episode (0-based) in stable natural order.
        - LeRobot v3.x layout: `dataset_dir/data/chunk-*/file-*.parquet`.
            Returns the dataset parquet file; actual episode slicing happens during read.
        - Direct parquet file path passed as `dataset_dir` (then `episode_index` is ignored).
    """
    import glob

    p = os.path.abspath(str(dataset_dir))
    if os.path.isfile(p) and p.lower().endswith(".parquet"):
        return p

    # LeRobot dataset layout (preferred/expected)
    files = _list_task_parquet_files(p)
    if files:
        idx = int(episode_index)
        if idx < 0 or idx >= len(files):
            raise IndexError(f"episode_index {idx} out of range (0..{len(files)-1}) for dataset_dir={p!r}")
        return files[idx]

    # LeRobot v3 dataset layout: a single (or a few) file-*.parquet containing many episodes
    v3_files = _list_lerobot_v3_dataset_files(p)
    if v3_files:
        # Episode selection happens later by filtering on episode label column.
        return v3_files[0]

    # Helpful error: show the exact pattern we expect
    ep_name = f"episode_{int(episode_index):06d}.parquet"
    pattern = os.path.join(p, "data", "chunk-*", ep_name)
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No LeRobot episodes found under {p!r}. Expected pattern like {pattern!r}"
    )


def _count_lerobot_episodes_in_parquet(parquet_path: str) -> int:
    """Count number of episodes in a single LeRobot parquet file.

    - v2.x single-episode parquet typically has 1 unique episode label (even if an episode column exists).
    - v3.x file-*.parquet usually contains multiple episode labels.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pyarrow is required to count episodes in parquet: {e!r}")

    try:
        col_names = list(pq.ParquetFile(parquet_path).schema.names)
    except Exception:
        col_names = []

    ep_col = _detect_episode_column(col_names or [])
    if not ep_col:
        return 1

    try:
        table = pq.read_table(parquet_path, columns=[ep_col])
        col = table[ep_col]
        try:
            values = np.asarray(col.to_numpy())
        except Exception:
            values = np.asarray(col.to_pylist())
        if values.size <= 0:
            return 0
        try:
            uniq = np.unique(values)
            return int(len(uniq))
        except Exception:
            # Fallback: manual unique count
            return int(len(set([str(x) for x in values.tolist()])))
    except Exception:
        # Best-effort fallback: treat as single episode
        return 1


def _list_lerobot_episode_refs(dataset_dir: str) -> list:
    """Build a stable list of (parquet_path, episode_index) references for scoring.

    - v2.x: one episode per file -> [(episode_file, 0), ...]
    - v3.x: multiple episodes per file-*.parquet -> [(file, ep_ordinal), ...]
    - direct parquet path: enumerate episodes within that file
    """
    p = os.path.abspath(str(dataset_dir))
    refs = []

    if os.path.isfile(p) and p.lower().endswith(".parquet"):
        n = _count_lerobot_episodes_in_parquet(p)
        if n <= 0:
            return []
        return [(p, int(i)) for i in range(int(n))]

    # v2 layout: episode_*.parquet
    v2_files = _list_task_parquet_files(p)
    if v2_files:
        return [(str(fp), 0) for fp in v2_files]

    # v3 layout: file-*.parquet
    v3_files = _list_lerobot_v3_dataset_files(p)
    for fp in v3_files:
        n = _count_lerobot_episodes_in_parquet(str(fp))
        for i in range(int(max(0, n))):
            refs.append((str(fp), int(i)))
    return refs


def _read_lerobot_episode_state(
    parquet_path: str,
    episode_index: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read LeRobot parquet and return (state[T,16], timestamp[T] or None).

    - For v2.x: parquet is a single episode file.
    - For v3.x: parquet may contain multiple episodes; we filter by an episode label column
      (prefer `episode_index` if present).
    """
    # Try to infer columns without loading the full file
    col_names = None
    try:
        import pyarrow.parquet as pq  # type: ignore

        col_names = list(pq.ParquetFile(parquet_path).schema.names)
    except Exception:
        col_names = None

    ep_col = _detect_episode_column(col_names or [])
    ts_candidates = ["timestamp", "t", "time", "observation.timestamp"]
    order_candidates = ["frame_index", "index"]

    cols_to_read = ["observation.state"]
    if col_names:
        for c in ts_candidates:
            if c in col_names:
                cols_to_read.append(c)
                break
        if episode_index is not None and ep_col and ep_col in col_names:
            cols_to_read.append(ep_col)
        for c in order_candidates:
            if c in col_names:
                cols_to_read.append(c)
                break
    else:
        # unknown schema; read full file
        cols_to_read = None  # type: ignore

    # Prefer pandas if available (common in this repo's data tools)
    try:
        import pandas as pd  # type: ignore
        df = pd.read_parquet(parquet_path, columns=cols_to_read)
        labels_arr = None
        if episode_index is not None and ep_col and ep_col in df.columns:
            labels_arr = np.asarray(df[ep_col].to_numpy())
            # v2.x single-episode files may still have an episode column, but only one unique value.
            # In that case, do not filter by episode_index to avoid accidental empty selection.
            try:
                if int(len(np.unique(labels_arr))) <= 1:
                    labels_arr = None
            except Exception:
                pass

        if episode_index is not None and ep_col and ep_col in df.columns and labels_arr is not None:
            # Accept either direct label match or ordinal index in sorted unique labels
            try:
                uniq = np.unique(labels_arr)
                uniq_sorted = np.sort(uniq)
            except Exception:
                uniq_sorted = None

            selected = None
            try:
                if bool(np.any(labels_arr == episode_index)):
                    selected = episode_index
            except Exception:
                selected = None

            if selected is None and uniq_sorted is not None and 0 <= int(episode_index) < int(len(uniq_sorted)):
                selected = uniq_sorted[int(episode_index)]

            if selected is None:
                raise IndexError(
                    f"episode_index {episode_index} not found in column {ep_col!r} of {parquet_path!r}"
                )

            df = df[df[ep_col] == selected]
            # Keep deterministic ordering within an episode
            for oc in order_candidates:
                if oc in df.columns:
                    try:
                        df = df.sort_values(by=oc, kind="mergesort")
                    except Exception:
                        pass
                    break

        if "observation.state" not in df.columns:
            raise KeyError("missing column 'observation.state'")
        state_list = df["observation.state"].tolist()
        state = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in state_list], axis=0)
        ts = None
        for ts_col in ts_candidates:
            if ts_col in df.columns:
                try:
                    ts = np.asarray(df[ts_col].to_numpy(), dtype=float).reshape(-1)
                except Exception:
                    ts = None
                break
        return state, ts
    except ImportError:
        pass

    # Fallback to pyarrow
    try:
        import pyarrow.parquet as pq  # type: ignore
        import pyarrow.compute as pc  # type: ignore

        table = pq.read_table(parquet_path, columns=cols_to_read)
        table_col_names = set(table.column_names)

        col_pa = None

        if episode_index is not None and ep_col and ep_col in table_col_names:
            col_pa = table[ep_col]
            # v2.x single-episode files may still have an episode column, but only one unique value.
            # In that case, do not filter by episode_index to avoid accidental empty selection.
            try:
                vals0 = np.asarray(col_pa.to_numpy())
                if int(len(np.unique(vals0))) <= 1:
                    col_pa = None
            except Exception:
                pass

        if episode_index is not None and ep_col and ep_col in table_col_names and col_pa is not None:
            # Direct label match or ordinal in sorted unique
            selected = None
            try:
                values = np.asarray(col_pa.to_numpy())
                if bool(np.any(values == episode_index)):
                    selected = episode_index
                else:
                    uniq = np.unique(values)
                    uniq_sorted = np.sort(uniq)
                    if 0 <= int(episode_index) < int(len(uniq_sorted)):
                        selected = uniq_sorted[int(episode_index)]
            except Exception:
                selected = episode_index

            if selected is None:
                raise IndexError(
                    f"episode_index {episode_index} not found in column {ep_col!r} of {parquet_path!r}"
                )

            mask = pc.equal(col_pa, selected)
            table = table.filter(mask)

        table_col_names = set(table.column_names)
        if "observation.state" not in table_col_names:
            raise KeyError("missing column 'observation.state'")

        # Deterministic ordering if order columns exist
        for oc in order_candidates:
            if oc in table_col_names:
                try:
                    table = table.sort_by([(oc, "ascending")])
                except Exception:
                    pass
                break

        state_py = table["observation.state"].to_pylist()
        state = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in state_py], axis=0)

        ts = None
        for ts_col in ts_candidates:
            if ts_col in table_col_names:
                try:
                    ts = np.asarray(table[ts_col].to_numpy(), dtype=float).reshape(-1)
                except Exception:
                    ts = None
                break
        return state, ts
    except Exception as e:
        raise RuntimeError(
            f"Failed to read LeRobot episode parquet: {parquet_path!r}. "
            f"Install pandas+pyarrow or pyarrow. err={e!r}"
        )

def _load_lerobot_episode_as_trajs(
    dataset_dir: str,
    episode_index: int = 0,
    fps: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a LeRobot parquet episode and convert to replay trajectories.

    Expects observation.state with 16 dims:
      [x_l,y_l,z_l,qx_l,qy_l,qz_l,qw_l,grip_l, x_r,y_r,z_r,qx_r,qy_r,qz_r,qw_r,grip_r]
    """
    parquet_path = _find_lerobot_episode_parquet(dataset_dir, int(episode_index))
    state, ts = _read_lerobot_episode_state(parquet_path, episode_index=int(episode_index))
    if state.ndim != 2 or state.shape[1] < 16:
        raise ValueError(f"Unexpected observation.state shape in {parquet_path}: {state.shape}")
    T = int(state.shape[0])
    if T <= 0:
        raise ValueError(f"Empty episode: {parquet_path}")

    if ts is not None and ts.size == T:
        t = np.asarray(ts, dtype=float)
    else:
        _fps = float(fps) if (fps is not None and float(fps) > 0) else _infer_lerobot_fps(os.path.dirname(parquet_path))
        t = (np.arange(T, dtype=float) / _fps).astype(float)

    left_pose7 = np.asarray(state[:, 0:7], dtype=float)
    left_grip = np.asarray(state[:, 7], dtype=float).reshape(-1)
    right_pose7 = np.asarray(state[:, 8:15], dtype=float)
    right_grip = np.asarray(state[:, 15], dtype=float).reshape(-1)

    traj_left_pose = np.concatenate([t.reshape(-1, 1), left_pose7], axis=1)
    traj_right_pose = np.concatenate([t.reshape(-1, 1), right_pose7], axis=1)
    traj_left_gripper = np.stack([t, left_grip], axis=1)
    traj_right_gripper = np.stack([t, right_grip], axis=1)

    return traj_left_pose, traj_right_pose, traj_left_gripper, traj_right_gripper

def _nearest_index(times: np.ndarray, target_t: float) -> int:
    """Find index of the nearest timestamp in sorted `times` to `target_t`."""
    if times.size == 0:
        return 0
    # Assume times ascending; use searchsorted
    pos = int(np.searchsorted(times, target_t, side='left'))
    if pos <= 0:
        return 0
    if pos >= times.size:
        return times.size - 1
    prev_idx = pos - 1
    # Choose closer neighbor
    if abs(times[pos] - target_t) < abs(target_t - times[prev_idx]):
        return pos
    return prev_idx

# ---------- Continuity Evaluation ----------
def check_continuous(last_pose_L: Optional[Dict[str, Any]],
                     last_pose_R: Optional[Dict[str, Any]],
                     pose_L: Optional[Dict[str, Any]],
                     pose_R: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate continuity between consecutive poses for left/right hands.

    Scoring rules:
    - Position distance: 0 m -> 100, 0.04 m -> 60, >0.04 m -> 0 (linear in between)
    - Angle distance: 0 deg -> 100, 8 deg -> 60, >8 deg -> 0 (linear in between)

    Returns a flat dict with per-hand metrics and overall score (min of hands).
    """
    def _pose_to_vec(pose: Optional[Dict[str, Any]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if pose is None:
            return None
        try:
            tx = float(pose["transform"]["translation"]["x"]) 
            ty = float(pose["transform"]["translation"]["y"]) 
            tz = float(pose["transform"]["translation"]["z"]) 
            qx = float(pose["transform"]["rotation"]["x"]) 
            qy = float(pose["transform"]["rotation"]["y"]) 
            qz = float(pose["transform"]["rotation"]["z"]) 
            qw = float(pose["transform"]["rotation"]["w"]) 
            p = np.array([tx, ty, tz], dtype=float)
            q = np.array([qx, qy, qz, qw], dtype=float)
            return p, q
        except Exception:
            return None

    def _angle_between_quats_deg(q1: np.ndarray, q2: np.ndarray) -> float:
        # Ensure unit quats
        def _norm_q(q):
            n = np.linalg.norm(q)
            return q / n if n > 0 else q
        q1 = _norm_q(q1); q2 = _norm_q(q2)
        # Use absolute dot to handle double-cover
        dot = float(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
        angle_rad = 2.0 * math.acos(dot)
        return math.degrees(angle_rad)

    def _score_linear(val: float, thr: float, score_at_thr: float) -> float:
        if val <= 0.0:
            return 100.0
        if val >= thr:
            return 60.0 if score_at_thr == 60.0 and thr == 0.0 else 0.0
        # Map linearly from 0->100 down to thr->score_at_thr
        return 100.0 - (100.0 - score_at_thr) * (val / thr)

    def _score_pos(dist_m: float) -> float:
        if dist_m<=0.005:
            score_pos = 100.0
        elif dist_m <= 0.045:
            score_pos = 100.0 - 40.0 * ((dist_m-0.005) / 0.04)
        else:
            # Progressive decay: continuous at p=0.045 with score 60, then exponentially decays toward 0
            delta_p = dist_m - 0.045
            decay_scale_p = 0.1  # Decay scale of about 10 cm
            score_pos = 60.0 * math.exp(-max(0.0, delta_p) / max(1e-6, decay_scale_p))
        return score_pos

    def _score_ang(angle_deg: float) -> float:
        if angle_deg <= 1.0:
            score_ori = 100.0
        elif angle_deg <= 9.0:
            score_ori = 100.0 - 40.0 * ((angle_deg - 1.0) / 8.0)
        else:
            # Progressive decay: continuous at a=9 degrees with score 60, then exponentially decays toward 0
            delta_a = angle_deg - 9.0
            decay_scale_a = 20.0  # Decay scale of about 20 degrees
            score_ori = 60.0 * math.exp(-max(0.0, delta_a) / max(1e-6, decay_scale_a))
        return score_ori

    def _compute_for_hand(last_pose: Optional[Dict[str, Any]], pose: Optional[Dict[str, Any]]) -> Dict[str, float]:
        v_last = _pose_to_vec(last_pose)
        v_now = _pose_to_vec(pose)
        if (v_last is None) or (v_now is None):
            return {"pos_dist": float("nan"), "ang_deg": float("nan"), "pos_score": 0.0, "ang_score": 0.0, "score": 0.0}
        p_last, q_last = v_last
        p_now, q_now = v_now
        pos_dist = float(np.linalg.norm(p_now - p_last))
        ang_deg = _angle_between_quats_deg(q_last, q_now)
        pos_score = _score_pos(pos_dist)
        ang_score = _score_ang(ang_deg)
        return {"pos_dist": pos_dist, "ang_deg": ang_deg, "pos_score": pos_score, "ang_score": ang_score, "score": min(pos_score, ang_score)}

    left = _compute_for_hand(last_pose_L, pose_L)
    right = _compute_for_hand(last_pose_R, pose_R)
    overall = min(left["score"], right["score"]) if (not math.isnan(left["pos_dist"]) and not math.isnan(right["pos_dist"])) else max(left["score"], right["score"]) 
    return {"left": left, "right": right, "score": overall}

# ---------- Reused UDP Utility Functions ----------
def hand_pose_to_T(pose_dict):
    tr = pose_dict["transform"]["translation"]
    ro = pose_dict["transform"]["rotation"]   # x,y,z,w
    tx = float(tr["x"]); ty = float(tr["y"]); tz = float(tr["z"])
    R  = smath.quat_to_R(float(ro["w"]), float(ro["x"]), float(ro["y"]), float(ro["z"]))
    T  = np.eye(4); T[:3,:3] = R; T[:3, 3] = [tx, ty, tz]
    return T

# ---------- MuJoCo Stability Helpers ----------
def safe_start_viewer(rbt, retries: int = 3, delay_s: float = 0.3) -> bool:
    for i in range(max(1, int(retries))):
        try:
            rbt.start_viewer_thread()
            return True
        except Exception as e:
            print(f"[VIEWER] start failed (try {i+1}/{retries}): {e!r}")
            time.sleep(delay_s)
    return False

# ---------- Robot Motion-Control Functions ----------
def go_home(start_pos: np.ndarray, start_count:int, count:int)->Tuple[bool, np.ndarray]:
    dt = 1.0 / CONTROL_HZ
    totaltime = HOME_TIME
    target_pos= HOME_POS
    s=0.5*(1-math.cos((count-start_count)*dt/totaltime*math.pi)) #0-1
    cur_pos = (1-s)*start_pos + s*target_pos
    isFinished = (count - start_count)*dt >= totaltime
    return isFinished, cur_pos

def go_init(start_pos: np.ndarray, start_count:int, count:int)->Tuple[bool, np.ndarray]:
    dt = 1.0 / CONTROL_HZ
    totaltime = HOME_TIME
    target_pos= EXIT_POS
    s=0.5*(1-math.cos((count-start_count)*dt/totaltime*math.pi)) #0-1
    cur_pos = (1-s)*start_pos + s*target_pos
    isFinished = (count - start_count)*dt >= totaltime
    return isFinished, cur_pos

def adjust_ee(start_pos: np.ndarray, start_count:int, count:int, adjust_dist: np.ndarray)->Tuple[bool, np.ndarray]:
    dt = 1.0 / CONTROL_HZ
    totaltime = ADJUST_TIME
    target_pos= adjust_dist + start_pos
    s=0.5*(1-math.cos((count-start_count)*dt/totaltime*math.pi)) #0-1
    cur_pos = (1-s)*start_pos + s*target_pos
    isFinished = (count - start_count)*dt >= totaltime
    return isFinished, cur_pos

def jog(flag, vLmt, pos, prev_v, dt):
    vLmt=abs(vLmt)
    aLmt=vLmt/0.2  # Accelerate/decelerate to max speed within 0.2s
    # flag: true=open, false=close
    if flag:
        if pos<GRIPPER_OPEN:
            vCmd = prev_v + aLmt * dt
            vCmd = min(vCmd, vLmt)
        else: # Decelerate to zero after hitting limit
            vCmd = prev_v - aLmt * dt
            vCmd = max(vCmd, 0.0)
    else:
        if pos>GRIPPER_CLOSE:
            vCmd = prev_v - aLmt * dt
            vCmd = max(vCmd, -vLmt)
        else: # Accelerate back to zero after hitting limit
            vCmd = prev_v + aLmt * dt
            vCmd = min(vCmd, 0.0)
    pCmd = pos + vCmd * dt
    return pCmd, vCmd

def slow_stop(start_pos: np.ndarray, start_vel:np.ndarray, start_count:int, count:int)->Tuple[bool, np.ndarray]:
    dt = 1.0 / CONTROL_HZ
    totaltime = np.max(np.abs(start_vel)) / ACC_LMT
    if totaltime < dt:
        totaltime = dt
    cur_vel=(1-(count-start_count)*dt/totaltime)*start_vel
    if np.linalg.norm(cur_vel)<1e-6:
        cur_vel=np.array([0.0]*len(start_vel))
    cur_pos= start_pos+cur_vel*dt
    isFinished = (count - start_count)*dt >= totaltime
    return isFinished, cur_pos

def clamp_joint_step(q_from: np.ndarray, q_to: np.ndarray) -> Tuple[np.ndarray, bool]:
    dq = q_to - q_from
    ratio = dq / (MAX_STEP_RAD * STEP_SCALE * 1.0 / CONTROL_HZ) # MAX_STEP_RAD is in rad/s or mm/s
    max_ratio = np.max(np.abs(ratio))
    if max_ratio <= 1.0:
        return q_to, False
    else:
        # print(f"[IK] Exceeded max joint speed limit, scaling ratio: {max_ratio:.3f}")
        dq_scaled = dq / max_ratio
        return q_from + dq_scaled, True

# ---------------- replay & score ----------------
def replay(trajectory_index: int = 1,
           adjust_ee_dist: Optional[np.ndarray] = None,
           traj_left_pose: Optional[np.ndarray] = None,
           traj_right_pose: Optional[np.ndarray] = None,
           traj_left_gripper: Optional[np.ndarray] = None,
           traj_right_gripper: Optional[np.ndarray] = None,
           rbt_tgt_obj: Optional[RbtKin] = None,
           rbt_act_obj: Optional[RbtKin] = None):
    # Trajectory playback mode: no UDP/keyboard threads
    if (traj_left_pose is None) or (traj_right_pose is None) or (traj_left_gripper is None) or (traj_right_gripper is None):
        if FILE_TYPE == "parquet":
            # SCORE_FOLDER_PATH points to a task folder containing many parquet files.
            # Each parquet is one trajectory; trajectory_index selects by natural-sorted filename.
            traj_left_pose, traj_right_pose, traj_left_gripper, traj_right_gripper = _load_lerobot_episode_as_trajs(
                SCORE_FOLDER_PATH, episode_index=int(trajectory_index), fps=30)
        else:
            fi=FolderIndexer(SCORE_FOLDER_PATH,FILE_SEARCH_MODE,FILE_SEARCH_LEVEL)
            traj_path = fi.build_traj_paths(trajectory_index)
            if isinstance(traj_path,tuple):# txt file
                left_path, right_path, left_grip_path, right_grip_path = traj_path
                try:
                    if REPLAY_TYPE==1:
                        traj_left_pose = _load_joint_traj(left_path, N_LEFT_ARM)
                        traj_right_pose = _load_joint_traj(right_path, N_RIGHT_ARM)
                    elif REPLAY_TYPE==0:
                        traj_left_pose = _load_pose_traj(left_path)
                        traj_right_pose = _load_pose_traj(right_path)
                    else:
                        pass
                    traj_left_gripper = _load_gripper_traj(left_grip_path)
                    traj_right_gripper = _load_gripper_traj(right_grip_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to load trajectories: {e!r}")
    assert traj_left_pose is not None
    assert traj_right_pose is not None
    assert traj_left_gripper is not None
    assert traj_right_gripper is not None
    # Build a unified playback timeline by left hand timestamps
    tL_pose = np.asarray(traj_left_pose[:, 0], dtype=float)
    tR_pose = np.asarray(traj_right_pose[:, 0], dtype=float)
    timeline = np.asarray(tL_pose, dtype=float)
    n_steps = int(timeline.size)
    # Prepare gripper time/qpos arrays
    tLG = np.asarray(traj_left_gripper[:, 0], dtype=float)
    qLG = np.asarray(traj_left_gripper[:, 1], dtype=float)
    tRG = np.asarray(traj_right_gripper[:, 0], dtype=float)
    qRG = np.asarray(traj_right_gripper[:, 1], dtype=float)
    rbt_tgt = rbt_tgt_obj if rbt_tgt_obj is not None else RbtKin()
    rbt_act = rbt_act_obj if rbt_act_obj is not None else RbtKin()# background virtual robot for FK only; no sim startup and no IK coupling
    logger: Optional[TrajLogger] = None
    if LOG_FLAG:
        # act_vec = qpos_total + left_tcp7 + right_tcp7
        logger = TrajLogger(out_dir="log", filename_prefix=f"{ROBOT}_replay", vector_len=int(N_QPOS_TOTAL + 14))
    # Always initialize data dicts (real/sim both use them)
    act_data={'qpos':{
              'chassis':[0.0]*N_CHASSIS,
              'torso':[0.0]*N_TORSO,
              'left_arm':[0.0]*N_LEFT_ARM,
              'right_arm':[0.0]*N_RIGHT_ARM,
              'left_gripper':[0.0]*N_LEFT_GRIPPER,
              'right_gripper':[0.0]*N_RIGHT_GRIPPER},
          'tcp':{'left_arm':[0.0]*7,'right_arm':[0.0]*7},
          'qvel':{
              'torso':[0.0]*N_TORSO,
              'left_arm':[0.0]*N_LEFT_ARM,
              'right_arm':[0.0]*N_RIGHT_ARM} }
    tgt_data={'qpos':{
              'chassis':[0.0]*N_CHASSIS,
              'torso':[0.0]*N_TORSO,
              'left_arm':[0.0]*N_LEFT_ARM,
              'right_arm':[0.0]*N_RIGHT_ARM,
              'left_gripper':[0.0]*N_LEFT_GRIPPER,
              'right_gripper':[0.0]*N_RIGHT_GRIPPER},
          'tcp':{'left_arm':[0.0]*7,'right_arm':[0.0]*7} }
    
    if RUN_MODE_REAL:
        _info("[MODE] Run mode: real robot")
    else:
        _info("[MODE] Run mode: simulation")

    # Input state
    # input-free playback
    playback_idx = 0  # current frame index in trajectories
    recalib_pending = True
    pause_until = 0.0
    next_state_after_pause: Optional[State] = None
    # Trajectory score stats: minimum score of continuous/workspace/collision (and corresponding frame)
    continuous_min_score = float('inf')
    continuous_min_index = -1
    continuous_min_detail: Optional[Dict[str, Any]] = None
    out_ws_min_score = float('inf')
    out_ws_min_index = -1
    out_ws_min_detail: Optional[Dict[str, Any]] = None
    collision_min_score = float('inf')
    collision_min_index = -1
    collision_min_detail: Optional[Dict[str, Any]] = None

    # Home / first-frame cache for this run
    home7 = None
    home_axisAngle_L  = None
    home_axisAngle_R  = None
    T_hand0_L = None
    T_hand0_R = None
    target7_L = None
    target7_R = None
    pose_L = None
    pose_R = None
    last_pose_L = None
    last_pose_R = None
    last_axisAngle_L = None
    last_axisAngle_R = None
    last_p_L = np.zeros((3,), dtype=float)
    last_p_R = np.zeros((3,), dtype=float)
    delta_pose_L = None
    delta_pose_R = None
    
    dt     = 1.0/CONTROL_HZ
    filt_L = [flt.RealTimeButterworthFilter(lowcut=10, fs=100, btype='low') for _ in range(7)]
    filt_R = [flt.RealTimeButterworthFilter(lowcut=10, fs=100, btype='low') for _ in range(7)]
    count=0
    state=State.GO_HOME
    last_state=State.GO_HOME
    start_home_count=0
    start_home_pos=np.array([0.0]*N_QPOS_TOTAL)
    start_init_count=0
    start_init_pos=np.array([0.0]*N_QPOS_TOTAL)
    start_adjust_count=0
    start_adjust_pos=np.array([0.0]*N_QPOS_TOTAL)
    start_slow_stop_count=0
    start_slow_stop_pos=np.array([0.0]*N_QPOS_TOTAL)
    start_slow_stop_vel=np.array([0.0]*N_QPOS_TOTAL)
    cmd=np.array([0.0]*N_QPOS_TOTAL)
    last_cmd=np.array([0.0]*N_QPOS_TOTAL)
    
    ctrl: Optional[RbtCtrl.Dual_arm_controller] = None
    # Start viewer with retries to avoid occasional startup failures
    if not RUN_MODE_REAL:
        if START_VIEWER:
            ok = safe_start_viewer(rbt_tgt, retries=3, delay_s=0.5)
            if not ok:
                _info("[VIEWER] Viewer startup failed; continue computation-only (headless)")
    else:
        ctrl = RbtCtrl.Dual_arm_controller()
    # Fixed-period scheduling: use monotonic clock for next cycle deadline
    next_deadline = time.perf_counter() + dt
    try:
        while True:
            t_loop_start = time.time()
            
            try:
                if RUN_MODE_REAL:
                    assert ctrl is not None
                    fb = ctrl.get_real_feedback()
                else:
                    fb = rbt_tgt.get_sim_feedback()
                if isinstance(fb, dict) and fb.get("warmed", True) is False:
                    _info("[SIM] Robot not ready yet, waiting...")
                    time.sleep(0.1)
                    continue
                qpos_fb = fb.get("qpos", {}) if isinstance(fb, dict) else {}
                qvel_fb = fb.get("qvel", {}) if isinstance(fb, dict) else {}
                act_data["qpos"]["chassis"] = _require_fb_vec(qpos_fb, "chassis", N_CHASSIS, "feedback.qpos.chassis").tolist()
                act_data["qpos"]["torso"] = _require_fb_vec(qpos_fb, "torso", N_TORSO, "feedback.qpos.torso").tolist()
                act_data["qpos"]["left_arm"] = _require_fb_vec(qpos_fb, "left_arm", N_LEFT_ARM, "feedback.qpos.left_arm").tolist()
                act_data["qpos"]["right_arm"] = _require_fb_vec(qpos_fb, "right_arm", N_RIGHT_ARM, "feedback.qpos.right_arm").tolist()
                act_data["qpos"]["left_gripper"] = _require_fb_vec(qpos_fb, "left_gripper", N_LEFT_GRIPPER, "feedback.qpos.left_gripper").tolist()
                act_data["qpos"]["right_gripper"] = _require_fb_vec(qpos_fb, "right_gripper", N_RIGHT_GRIPPER, "feedback.qpos.right_gripper").tolist()
                act_data["qvel"]["torso"] = _require_fb_vec(qvel_fb, "torso", N_TORSO, "feedback.qvel.torso").tolist()
                act_data["qvel"]["left_arm"] = _require_fb_vec(qvel_fb, "left_arm", N_LEFT_ARM, "feedback.qvel.left_arm").tolist()
                act_data["qvel"]["right_arm"] = _require_fb_vec(qvel_fb, "right_arm", N_RIGHT_ARM, "feedback.qvel.right_arm").tolist()
                if logger is not None: # for logging
                    _act_qpos = concat_qpos_parts(act_data["qpos"])
                    rbt_act.set_real_qpos(_act_qpos)
                    act_pose7=rbt_act.solve_fk()
                    act_data["tcp"]["left_arm"]=smath.pose7_wxyz_to_xyzw(act_pose7[0]).tolist()
                    act_data["tcp"]["right_arm"]=smath.pose7_wxyz_to_xyzw(act_pose7[1]).tolist()
            except Exception as e:
                _info(f"[SIM] get feedback failed: {e!r}")
            if count==0:
                cmd = concat_qpos_parts(act_data["qpos"])
                last_cmd=cmd.copy()
                start_home_count = count
                start_home_pos = cmd.copy()
            
            # Automatic state transition: GO_HOME -> REPLAY_POSE(playback) -> GO_INIT -> exit
            if not state==last_state:
                _info(f"state change from {last_state} to {state} at count {count}")
                last_state = state

            ###################### Motion planning per state ######################
            T_sent_L = None; T_sent_R = None
            if state==State.GO_HOME:
                isfinished,new_pos=go_home(start_home_pos,start_home_count,count)
                tgt_data["qpos"]["chassis"]=new_pos[SL_CHASSIS].tolist()
                tgt_data["qpos"]["torso"]=new_pos[SL_TORSO].tolist()
                tgt_data["qpos"]["left_arm"]=new_pos[SL_LEFT_ARM].tolist()
                tgt_data["qpos"]["right_arm"]=new_pos[SL_RIGHT_ARM].tolist()
                tgt_data["qpos"]["left_gripper"]=new_pos[SL_LEFT_GRIPPER].tolist()
                tgt_data["qpos"]["right_gripper"]=new_pos[SL_RIGHT_GRIPPER].tolist()
                if isfinished:
                    rbt_tgt.set_real_qpos(new_pos[:N_QPOS_TOTAL],True)
                    rbt_tgt.update_configureation_to_home()# update optimized configuration to home pose
                    # rbt_tgt.set_lock_flag(True)
                    home7 = rbt_tgt.solve_fk()
                    # print(home7)
                    target7_L = home7[0]
                    target7_R = home7[1]
                    start_adjust_count = count
                    start_adjust_pos = np.concatenate([home7[0][0:3], home7[1][0:3]], axis=0)
                    state = State.ADJUST_EE
            elif state==State.GO_INIT:
                isfinished,new_pos=go_init(start_init_pos,start_init_count,count)
                tgt_data["qpos"]["chassis"]=new_pos[SL_CHASSIS].tolist()
                tgt_data["qpos"]["torso"]=new_pos[SL_TORSO].tolist()
                tgt_data["qpos"]["left_arm"]=new_pos[SL_LEFT_ARM].tolist()
                tgt_data["qpos"]["right_arm"]=new_pos[SL_RIGHT_ARM].tolist()
                tgt_data["qpos"]["left_gripper"]=new_pos[SL_LEFT_GRIPPER].tolist()
                tgt_data["qpos"]["right_gripper"]=new_pos[SL_RIGHT_GRIPPER].tolist()
                if isfinished:
                    rbt_tgt.set_real_qpos(new_pos[:N_QPOS_TOTAL],True)
                    pause_until = time.perf_counter()
                    next_state_after_pause = None
                    state = State.WAIT
            elif state==State.SLOW_STOP:
                isfinished,new_pos=slow_stop(start_slow_stop_pos,start_slow_stop_vel,start_slow_stop_count,count)
                _info(f"slow_stop: {new_pos-start_slow_stop_pos}")
                tgt_data["qpos"]["chassis"]=new_pos[SL_CHASSIS].tolist()
                tgt_data["qpos"]["torso"]=new_pos[SL_TORSO].tolist()
                tgt_data["qpos"]["left_arm"]=new_pos[SL_LEFT_ARM].tolist()
                tgt_data["qpos"]["right_arm"]=new_pos[SL_RIGHT_ARM].tolist()
                tgt_data["qpos"]["left_gripper"]=new_pos[SL_LEFT_GRIPPER].tolist()
                tgt_data["qpos"]["right_gripper"]=new_pos[SL_RIGHT_GRIPPER].tolist()
                if isfinished:
                    rbt_tgt.set_real_qpos(new_pos[:N_QPOS_TOTAL],True)
                    state=State.IDLE
            elif state==State.ADJUST_EE:
                _adj = ADJUST_EE_DIST if adjust_ee_dist is None else adjust_ee_dist
                isfinished,new_pos=adjust_ee(start_adjust_pos,start_adjust_count,count,_adj)
                target7_L[0:3]=new_pos[0:3]
                target7_R[0:3]=new_pos[3:6]
                target7=[target7_L, target7_R]
                q_cmd=rbt_tgt.solve_ik(target7)
                q_cmd = _require_1d_float_array(q_cmd, N_QPOS_TOTAL, "solve_ik.q_cmd(arms)")
                tgt_data["qpos"]["chassis"]=q_cmd[SL_CHASSIS].tolist()
                tgt_data["qpos"]["torso"]=q_cmd[SL_TORSO].tolist()
                tgt_data["qpos"]["left_arm"]=q_cmd[SL_LEFT_ARM].tolist()
                tgt_data["qpos"]["right_arm"]=q_cmd[SL_RIGHT_ARM].tolist()
                # tgt_data["qpos"]["left_gripper"]=q_cmd[SL_LEFT_GRIPPER].tolist()
                # tgt_data["qpos"]["right_gripper"]=q_cmd[SL_RIGHT_GRIPPER].tolist()
                if isfinished:
                    pause_until = time.perf_counter() + WAIT_TIME
                    if REPLAY_TYPE==1:
                        next_state_after_pause = State.REPLAY_JNT
                    elif REPLAY_TYPE==0:
                        next_state_after_pause = State.REPLAY_POSE
                    else:
                        pass
                    state = State.WAIT
                    playback_idx = 0
            elif state == State.REPLAY_POSE:
                # Use loaded hand poses to drive IK, replacing controller inputs
                if playback_idx < n_steps:
                    # Current reference time
                    t_ref = float(timeline[playback_idx])
                    # Find nearest pose indices for L/R
                    iLp = _nearest_index(tL_pose, t_ref)
                    iRp = _nearest_index(tR_pose, t_ref)
                    # Build pose dicts matching get_pose_item format
                    def _row_to_pose(row: np.ndarray) -> Dict[str, Any]:
                        #row: [t, x, y, z, qx, qy, qz, qw]
                        return {
                            "child_frame_id": "pose_from_file",
                            "transform": {
                                "translation": {"x": float(row[1]), "y": float(row[2]), "z": float(row[3])},
                                "rotation": {"x": float(row[4]), "y": float(row[5]), "z": float(row[6]), "w": float(row[7])}
                            }
                        }
                    pose_L = _row_to_pose(traj_left_pose[iLp, :])
                    pose_R = _row_to_pose(traj_right_pose[iRp, :])

                    # Alignment on first valid poses
                    if recalib_pending and (pose_L is not None) and (pose_R is not None):
                        zero_pose = {
                            "transform": {
                                "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                            }
                        }
                        T_hand0_L = hand_pose_to_T(zero_pose) # Start from zero pose so frame-1 is also checked for discontinuity (avoid first-frame jumps)
                        T_hand0_R = hand_pose_to_T(zero_pose)
                        rbt_tgt.set_real_qpos(concat_qpos_parts(tgt_data["qpos"]), True)
                        home7 = rbt_tgt.solve_fk()
                        home_axisAngle_L = smath.pose7_to_axisAngle(home7[0])
                        home_axisAngle_R = smath.pose7_to_axisAngle(home7[1])
                        last_axisAngle_L = home_axisAngle_L[3:7].copy()
                        last_axisAngle_R = home_axisAngle_R[3:7].copy()
                        for i in range(7):
                            filt_L[i].reset(home_axisAngle_L[i])
                            filt_R[i].reset(home_axisAngle_R[i])
                        recalib_pending = False
                        last_pose_L = pose_L
                        last_pose_R = pose_R
                        _info("[STATE] Trajectory replay: first alignment completed.")
                    # Ready checks
                    left_ready  = (home_axisAngle_L is not None) and (T_hand0_L is not None) and (pose_L is not None)
                    right_ready = (home_axisAngle_R is not None) and (T_hand0_R is not None) and (pose_R is not None)

                    if left_ready:
                        T_hand_L_now = hand_pose_to_T(pose_L)
                        p_home_L  = home_axisAngle_L[0:3]
                        dp_world_L = T_hand_L_now[:3, 3] - T_hand0_L[:3, 3]
                        dp_base_L  = R_W2B_L @ dp_world_L
                        p_target_L = p_home_L + SCALE_POS_L * dp_base_L
                        R_tcp0_L  = smath.axisAngle_to_R(home_axisAngle_L[3:7])
                        dR_world_L = T_hand_L_now[:3,:3] @ T_hand0_L[:3,:3].T
                        R_target_L = dR_world_L @ R_tcp0_L
                        axisAngle_target_L = smath.R_to_axisAngle(R_target_L)
                        axisAngle_target_L=smath.get_nearest_axisAngle(last_axisAngle_L,axisAngle_target_L)
                        last_axisAngle_L=axisAngle_target_L.copy()
                        target7_axisAngle_L = np.concatenate([p_target_L, axisAngle_target_L], axis=0)
                        for i in range(7):
                            target7_axisAngle_L[i] = filt_L[i].filter_step(target7_axisAngle_L[i])
                        target7_L=smath.axisAngle_to_pose7(target7_axisAngle_L)
                    if right_ready:
                        T_hand_R_now = hand_pose_to_T(pose_R)
                        p_home_R  = home_axisAngle_R[0:3]
                        dp_world_R = T_hand_R_now[:3, 3] - T_hand0_R[:3, 3]
                        dp_base_R  = R_W2B_R @ dp_world_R
                        p_target_R = p_home_R + SCALE_POS_R * dp_base_R
                        R_tcp0_R  = smath.axisAngle_to_R(home_axisAngle_R[3:7])
                        dR_world_R = T_hand_R_now[:3,:3] @ T_hand0_R[:3,:3].T
                        R_target_R = dR_world_R @ R_tcp0_R
                        axisAngle_target_R = smath.R_to_axisAngle(R_target_R)
                        axisAngle_target_R=smath.get_nearest_axisAngle(last_axisAngle_R,axisAngle_target_R)
                        last_axisAngle_R=axisAngle_target_R.copy()
                        target7_axisAngle_R = np.concatenate([p_target_R, axisAngle_target_R], axis=0)
                        for i in range(7):
                            target7_axisAngle_R[i] = filt_R[i].filter_step(target7_axisAngle_R[i])
                        target7_R=smath.axisAngle_to_pose7(target7_axisAngle_R)

                    # IK solve
                    target7=[target7_L, target7_R]
                    q_cmd=rbt_tgt.solve_ik(target7)
                    q_cmd = _require_1d_float_array(q_cmd, N_QPOS_TOTAL, "solve_ik.q_cmd(arms)")
                    tgt_data["qpos"]["chassis"]=q_cmd[SL_CHASSIS].tolist()
                    tgt_data["qpos"]["torso"]=q_cmd[SL_TORSO].tolist()
                    tgt_data["qpos"]["left_arm"]=q_cmd[SL_LEFT_ARM].tolist()
                    tgt_data["qpos"]["right_arm"]=q_cmd[SL_RIGHT_ARM].tolist()
                    # Apply gripper qpos using nearest timestamp match
                    iLg = _nearest_index(tLG, t_ref)
                    iRg = _nearest_index(tRG, t_ref)
                    lg = float(qLG[iLg]) if qLG.size > 0 else GRIPPER_OPEN
                    rg = float(qRG[iRg]) if qRG.size > 0 else GRIPPER_OPEN
                    _lg_val = lg**3/80/80
                    _rg_val = rg**3/80/80
                    tgt_data["qpos"]["left_gripper"]=[_lg_val]*N_LEFT_GRIPPER if N_LEFT_GRIPPER > 0 else []
                    tgt_data["qpos"]["right_gripper"]=[_rg_val]*N_RIGHT_GRIPPER if N_RIGHT_GRIPPER > 0 else []
                    playback_idx += 1
                else:
                    start_init_count = count
                    start_init_pos = concat_qpos_parts(act_data["qpos"])
                    # schedule pause handled in GO_INIT finish; add immediate note
                    pause_until = time.perf_counter() + WAIT_TIME
                    next_state_after_pause = State.GO_INIT
                    state = State.WAIT
            elif state == State.REPLAY_JNT:
                # Use loaded joint poses to drive robot, replacing controller inputs
                if playback_idx < n_steps:
                    # Current reference time
                    t_ref = float(timeline[playback_idx])
                    # Find nearest pose indices for L/R
                    iLp = _nearest_index(tL_pose, t_ref)
                    iRp = _nearest_index(tR_pose, t_ref)
                    # Apply joint qpos
                    qL = traj_left_pose[iLp, 1:1+N_LEFT_ARM]
                    qR = traj_right_pose[iRp, 1:1+N_RIGHT_ARM]
                    tgt_data["qpos"]["left_arm"]=np.asarray(qL, dtype=float).reshape(-1).tolist()
                    tgt_data["qpos"]["right_arm"]=np.asarray(qR, dtype=float).reshape(-1).tolist()
                    # Apply gripper qpos using nearest timestamp match
                    iLg = _nearest_index(tLG, t_ref)
                    iRg = _nearest_index(tRG, t_ref)
                    lg = float(qLG[iLg]) if qLG.size > 0 else GRIPPER_OPEN
                    rg = float(qRG[iRg]) if qRG.size > 0 else GRIPPER_OPEN
                    _lg_val = lg**3/80/80
                    _rg_val = rg**3/80/80
                    tgt_data["qpos"]["left_gripper"]=[_lg_val]*N_LEFT_GRIPPER if N_LEFT_GRIPPER > 0 else []
                    tgt_data["qpos"]["right_gripper"]=[_rg_val]*N_RIGHT_GRIPPER if N_RIGHT_GRIPPER > 0 else []
                    playback_idx += 1
                else:
                    start_init_count = count
                    start_init_pos = concat_qpos_parts(act_data["qpos"])
                    # schedule pause handled in GO_INIT finish; add immediate note
                    pause_until = time.perf_counter() + WAIT_TIME
                    next_state_after_pause = State.GO_INIT
                    state = State.WAIT
            elif state == State.WAIT:
                # Hold last command during pause
                if last_cmd is not None and last_cmd.size >= N_QPOS_TOTAL:
                    tgt_data["qpos"]["chassis"] = last_cmd[SL_CHASSIS].tolist()
                    tgt_data["qpos"]["torso"] = last_cmd[SL_TORSO].tolist()
                    tgt_data["qpos"]["left_arm"] = last_cmd[SL_LEFT_ARM].tolist()
                    tgt_data["qpos"]["right_arm"] = last_cmd[SL_RIGHT_ARM].tolist()
                    tgt_data["qpos"]["left_gripper"] = last_cmd[SL_LEFT_GRIPPER].tolist()
                    tgt_data["qpos"]["right_gripper"] = last_cmd[SL_RIGHT_GRIPPER].tolist()
                if time.perf_counter() >= pause_until:
                    if next_state_after_pause is None:
                        break
                    else:
                        state = next_state_after_pause
                        next_state_after_pause = None
            
            ########################### Robot command dispatch #####################
            clamped=True
            try:
                tmpcmd=concat_qpos_parts(tgt_data["qpos"])
                while clamped:
                    cmd,clamped=clamp_joint_step(last_cmd,tmpcmd)
                    last_cmd=cmd.copy()
                    
                    if logger is not None:
                        if state==State.WAIT or state==State.SLOW_STOP or state==State.GO_HOME or state==State.GO_INIT:
                            rbt_tgt.set_real_qpos(concat_qpos_parts(tgt_data["qpos"]), True)
                            tgt_pose7=rbt_tgt.solve_fk()
                            tgt_data["tcp"]["left_arm"]=smath.pose7_wxyz_to_xyzw(tgt_pose7[0]).tolist()
                            tgt_data["tcp"]["right_arm"]=smath.pose7_wxyz_to_xyzw(tgt_pose7[1]).tolist()
                        elif state==State.ADJUST_EE or state==State.REPLAY_POSE:
                            tgt_data["tcp"]["left_arm"]=smath.pose7_wxyz_to_xyzw(target7_L).tolist()
                            tgt_data["tcp"]["right_arm"]=smath.pose7_wxyz_to_xyzw(target7_R).tolist()
                        else:
                            pass
                        act_vec = np.concatenate([concat_qpos_parts(act_data["qpos"]),
                                                  act_data["tcp"]["left_arm"],
                                                  act_data["tcp"]["right_arm"]],axis=0)
                        tgt_vec = np.concatenate([cmd.tolist(),tgt_data["tcp"]["left_arm"],tgt_data["tcp"]["right_arm"]],axis=0)
                        try:
                            logger.log(act_vec.tolist(), tgt_vec.tolist())
                        except Exception:
                            pass
                    # print("sending cmd:",cmd)
                    if RUN_MODE_REAL:
                        ctrl.set_latest_cmd(cmd, CONTROL_HZ) # Set latest command; background thread interpolates and publishes at 150 Hz
                        # ctrl.apply_real_qpos(cmd) # If set, background thread publishes directly. For replay we do not interpolate here.
                    else:
                        rbt_tgt.apply_mapped_qpos(cmd)
                    if clamped:
                        now_pc = time.perf_counter()
                        sleep_sec = next_deadline - now_pc
                        if RUN_MODE_REAL:
                            if sleep_sec > 0:
                                time.sleep(sleep_sec)
                            else:
                                _info(f"[LOOP] Warning: control cycle overtime {(-sleep_sec)*1000.0:.2f} ms")
                        next_deadline += dt
                        # catch-up: if we already lag many cycles, skip missed deadlines to avoid persistent negative sleep
                        now_pc2 = time.perf_counter()
                        if next_deadline < now_pc2 + 1e-3:  # 1 ms tolerance
                            missed = int((now_pc2 - next_deadline) / dt) + 1
                            next_deadline += missed * dt
                        continue
            except Exception as e:
                _info(f"[SIM] apply cmd failed: {e!r}")
            
            #################################### Trajectory score computation ##########################
            if not RUN_MODE_REAL and REPLAY_TYPE==0:
                # Score only in simulation
                # Must be here: apply_mapped_qpos has just been executed, so robot state is correct.
                # playback_idx >= 1 avoids an issue where teleop only runs in the next cycle after WAIT->teleop.
                if state==State.REPLAY_POSE and playback_idx >= 1:
                    out_ws=rbt_tgt.check_out_workspace()
                    collision=rbt_tgt.check_collision_pairs()
                    continuous=check_continuous(last_pose_L,last_pose_R,pose_L, pose_R)
                    # Update minimum score and corresponding frame for all three metrics (current frame is playback_idx-1)
                    cur_idx = max(0, playback_idx - 1)
                    try:
                        c_score = float(continuous.get("score", float("nan")))
                    except Exception:
                        c_score = float("nan")
                    try:
                        w_score = float(out_ws.get("score", float("nan")))
                    except Exception:
                        w_score = float("nan")
                    try:
                        col_score = float(collision.get("score", float("nan")))
                    except Exception:
                        col_score = float("nan")

                    if (not math.isnan(c_score)) and c_score < continuous_min_score:
                        continuous_min_score = c_score
                        continuous_min_index = cur_idx
                        continuous_min_detail = continuous
                    if (not math.isnan(w_score)) and w_score < out_ws_min_score:
                        out_ws_min_score = w_score
                        out_ws_min_index = cur_idx
                        out_ws_min_detail = out_ws
                    if (not math.isnan(col_score)) and col_score < collision_min_score:
                        collision_min_score = col_score
                        collision_min_index = cur_idx
                        collision_min_detail = collision
            
            last_pose_L=pose_L
            last_pose_R=pose_R
            count=count+1
            
            # Fixed-period sleep: keep cycle duration consistent
            now_pc = time.perf_counter()
            sleep_sec = next_deadline - now_pc
            if RUN_MODE_REAL or START_VIEWER:
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                # else:
                    # _info(f"[LOOP] Warning: control cycle overtime {(-sleep_sec)*1000.0:.2f} ms")
            next_deadline += dt
    except KeyboardInterrupt:
        _info("\n[RM] Interrupted by user, exiting.")
    finally:
        try:
            if not RUN_MODE_REAL:
                if START_VIEWER:
                    rbt_tgt.stop_viewer_thread()
            else:
                ctrl.cleanup()
            #     ros2i.stop_ros2_publisher_thread()
            #     ros2i.stop_ros2_node()
        except Exception: pass
        if logger is not None:
            try:
                logger.close()
            except Exception:
                pass
        _info("[RM] Disconnected.")

    # Return overall score result for the full trajectory
    # Total score = product of the minima of continuous/workspace/collision
    # Also return the frame index and timestamp (from unified timeline) for each metric minimum
    continuous_min_timestamp = None
    out_ws_min_timestamp = None
    collision_min_timestamp = None
    try:
        if isinstance(continuous_min_index, int) and continuous_min_index >= 0 and continuous_min_index < timeline.size:
            continuous_min_timestamp = float(timeline[int(continuous_min_index)])
        if isinstance(out_ws_min_index, int) and out_ws_min_index >= 0 and out_ws_min_index < timeline.size:
            out_ws_min_timestamp = float(timeline[int(out_ws_min_index)])
        if isinstance(collision_min_index, int) and collision_min_index >= 0 and collision_min_index < timeline.size:
            collision_min_timestamp = float(timeline[int(collision_min_index)])
    except Exception:
        continuous_min_timestamp = None
        out_ws_min_timestamp = None
        collision_min_timestamp = None

    if (math.isinf(continuous_min_score) or math.isinf(out_ws_min_score) or math.isinf(collision_min_score)):
        total_min_score_out: Optional[float] = None
    else:
        total_min_score_out = float(continuous_min_score * out_ws_min_score * collision_min_score / 1e4)

    result = {
        "total_min_score": total_min_score_out,
        # total_min_index no longer maps to a single sample point; keep key for backward compatibility.
        "total_min_index": -1,
        "total_min_timestamp": None,
        "continuous_min_score": (None if math.isinf(continuous_min_score) else float(continuous_min_score)),
        "continuous_min_index": int(continuous_min_index),
        "continuous_min_timestamp": continuous_min_timestamp,
        "out_ws_min_score": (None if math.isinf(out_ws_min_score) else float(out_ws_min_score)),
        "out_ws_min_index": int(out_ws_min_index),
        "out_ws_min_timestamp": out_ws_min_timestamp,
        "collision_min_score": (None if math.isinf(collision_min_score) else float(collision_min_score)),
        "collision_min_index": int(collision_min_index),
        "collision_min_timestamp": collision_min_timestamp,
        "detail": {
            "continuous": continuous_min_detail,
            "out_ws": out_ws_min_detail,
            "collision": collision_min_detail,
        },
    }
    return result

def score(enable_writing_txt: bool = True,
          adj: Optional[np.ndarray] = None):
    # In score mode, force viewer off and quiet mode on
    global START_VIEWER, QUIET_MODE, HOME_TIME, ADJUST_TIME, STEP_SCALE, REPLAY_TYPE, LOG_FLAG
    START_VIEWER = 0
    QUIET_MODE   = 1
    HOME_TIME    = 0.5 # Reduce impact on scoring runtime
    ADJUST_TIME  = 0.2 # Reduce impact on scoring runtime
    STEP_SCALE   = 20.0 # Avoid runtime inflation caused by speed limits; replay loop limiting must stay consistent to keep score parity
    REPLAY_TYPE  = 0  # Score mode uses pose trajectories only
    LOG_FLAG     = False  # Do not generate replay logs during search
    # Generate a timestamped prefix for this run to avoid overwriting history
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("log", exist_ok=True)
    manager = mp.Manager()
    results_list = manager.list()
    # Compute adj once in main process and pass to workers
    if adj is None:
        adj = ADJUST_EE_DIST
    try:
        adj = np.asarray(adj, dtype=float).reshape(-1)
    except Exception:
        adj = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    if adj.size < 6:
        adj = np.pad(adj, (0, 6 - adj.size), mode='constant', constant_values=0.0)
    adj = np.asarray(adj[:6], dtype=float)

    # Detect number of physical CPU cores (Linux) and recommend worker count
    def _count_physical_cores_linux() -> int:
        try:
            phys_cores = set()
            with open('/proc/cpuinfo', 'r') as f:
                phys_id = None
                core_id = None
                for line in f:
                    if line.startswith('physical id'):
                        phys_id = line.strip().split(':')[-1].strip()
                    elif line.startswith('core id'):
                        core_id = line.strip().split(':')[-1].strip()
                    elif line.strip() == '':
                        if phys_id is not None and core_id is not None:
                            phys_cores.add((phys_id, core_id))
                        phys_id = None
                        core_id = None
            return max(1, len(phys_cores))
        except Exception:
            return max(1, os.cpu_count() or 1)

    # Build a cpu_list by selecting one representative logical CPU per physical core (Linux)
    def _get_physical_core_cpu_list_linux() -> list:
        try:
            allowed = set(os.sched_getaffinity(0))
        except Exception:
            allowed = None

        cpu_root = Path("/sys/devices/system/cpu")
        if not cpu_root.is_dir():
            cpus = list(range(os.cpu_count() or 1))
            return [c for c in cpus if allowed is None or c in allowed]

        seen = set()
        physical = []
        for cpu_dir in cpu_root.glob("cpu[0-9]*"):
            try:
                cpu = int(cpu_dir.name.replace("cpu", ""))
            except ValueError:
                continue
            if allowed is not None and cpu not in allowed:
                continue
            try:
                core_id = (cpu_dir / "topology" / "core_id").read_text().strip()
                pkg_id = (cpu_dir / "topology" / "physical_package_id").read_text().strip()
            except OSError:
                continue
            key = (pkg_id, core_id)
            if key in seen:
                continue
            seen.add(key)
            physical.append(cpu)

        if physical:
            return sorted(physical)
        cpus = list(range(os.cpu_count() or 1))
        return [c for c in cpus if allowed is None or c in allowed]

    physical_cores = _count_physical_cores_linux()
    logical_cores = os.cpu_count() or physical_cores
    recommended_workers = max(1, physical_cores - 2) if physical_cores >= 4 else physical_cores
    _info(f"[CPU] physical cores={physical_cores}, logical cores={logical_cores}, recommended workers={recommended_workers}")

    physical_cpu_list = []
    try:
        physical_cpu_list = _get_physical_core_cpu_list_linux()
    except Exception:
        physical_cpu_list = []

    # Build index list to process, placed here for compatibility with non-grid score mode
    fi = None
    parquet_episode_refs = None
    if FILE_TYPE == "parquet":
        parquet_episode_refs = _list_lerobot_episode_refs(SCORE_FOLDER_PATH)
        totalcount = len(parquet_episode_refs)
        idx_list = list(range(0, totalcount))
        if totalcount <= 0:
            raise RuntimeError(
                f"No parquet episodes found under SCORE_FOLDER_PATH={SCORE_FOLDER_PATH!r}. "
                f"Expected v2 layout data/chunk-*/episode_*.parquet or v3 layout data/chunk-*/file-*.parquet."
            )
    else:
        fi = FolderIndexer(SCORE_FOLDER_PATH,FILE_SEARCH_MODE,FILE_SEARCH_LEVEL)
        totalcount=fi.count()
        idx_list = list(range(0, totalcount))
    print(f"Total trajectories to score: {totalcount}, trying end-effector adjustment=({adj[0]:.1f}, 0.0, {adj[2]:.1f}) m")
    # Allow overriding worker count via env var (e.g., SCORE_WORKERS=8)
    _env_workers = 0
    try:
        _env_workers = int(os.getenv("SCORE_WORKERS", "0"))
    except Exception:
        _env_workers = 0
    num_workers = _env_workers if _env_workers > 0 else recommended_workers
    # Cap worker count by representative physical-core CPUs to avoid process contention on the same core
    if physical_cpu_list:
        num_workers = max(1, min(num_workers, len(idx_list), len(physical_cpu_list)))
    else:
        num_workers = max(1, min(num_workers, len(idx_list), os.cpu_count() or num_workers))

    # Split idx list evenly across workers
    chunks = [[] for _ in range(num_workers)]
    for i, idx in enumerate(idx_list):
        chunks[i % num_workers].append(idx)

    def _worker_run(worker_id: int, idx_chunk: list, results_list, _adj: np.ndarray, stop_event: Any):
        # Pin worker to a specific physical-core CPU when possible
        try:
            if physical_cpu_list:
                cpu = physical_cpu_list[worker_id % len(physical_cpu_list)]
            else:
                cpu = worker_id % (os.cpu_count() or 1)
            os.sched_setaffinity(0, {cpu})
            _info(f"[CPU] Worker {worker_id} pinned to CPU set {list(os.sched_getaffinity(0))}")
        except Exception as e:
            _info(f"[CPU] Worker {worker_id} CPU pinning failed: {e!r}")
        if stop_event.is_set():
            return
        def _has_nonempty_content(path: str) -> bool:
            try:
                if (not isinstance(path, str)) or (not os.path.isfile(path)):
                    return False
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip():
                            return True
                return False
            except Exception:
                return False
        for idx in idx_chunk:
            if stop_event.is_set():
                break
            t_idx_start = time.time()
            # Evaluate adjust_ee_dist once only (no x_list/z_list traversal)
            best_score = -float('inf')
            best_detail = None
            # Preload trajectory and reuse robot instances to avoid repeated re-read/rebuild
            try:
                base_dir_for_idx = ""
                if FILE_TYPE == "parquet":
                    if not parquet_episode_refs:
                        raise RuntimeError(f"No parquet episodes found under SCORE_FOLDER_PATH={SCORE_FOLDER_PATH!r}")
                    parquet_path, ep_idx = parquet_episode_refs[int(idx)]
                    _trajL,_trajR,_trajLG,_trajRG = _load_lerobot_episode_as_trajs(parquet_path, episode_index=int(ep_idx), fps=30)
                    base_dir_for_idx = f"{parquet_path}#episode={int(ep_idx)}"
                else:
                    assert fi is not None
                    traj_path = fi.build_traj_paths(int(idx))
                    if isinstance(traj_path, tuple):
                        lpath, rpath, lgpath, rgpath = traj_path
                        # Empty file / whitespace-only content: assign 0 score and skip
                        if (not _has_nonempty_content(lpath)) or (not _has_nonempty_content(rpath)) or (not _has_nonempty_content(lgpath)) or (not _has_nonempty_content(rgpath)):
                            base_dir = fi.path_by_index(int(idx))
                            duration_sec = 0.0
                            line = f"{idx}\t{0.0:.4f}\t{0.0:.4f}\t{0.0:.4f}\t{0.0:.4f}\t{duration_sec:.4f}\t{base_dir}\n"
                            results_list.append(line)
                            elapsed = time.time() - t_idx_start
                            print(f"[SCORE] (w{worker_id}) trajectory index={idx} is empty/no content, wrote score=0, elapsed {elapsed:.2f}s")
                            continue
                        _trajL = _load_pose_traj(lpath)
                        _trajR = _load_pose_traj(rpath)
                        _trajLG = _load_gripper_traj(lgpath)
                        _trajRG = _load_gripper_traj(rgpath)
                    else:
                        _trajL,_trajR,_trajLG,_trajRG = _load_lerobot_episode_as_trajs(traj_path, episode_index=0, fps=30)
                    base_dir_for_idx = fi.path_by_index(int(idx))

                # Trajectory duration (based on union of left/right pose timestamps)
                duration_sec = float('nan')
                try:
                    tL_pose = np.asarray(_trajL[:, 0], dtype=float).reshape(-1)
                    tR_pose = np.asarray(_trajR[:, 0], dtype=float).reshape(-1)
                    timeline = np.union1d(tL_pose, tR_pose)
                    if timeline.size >= 1:
                        duration_sec = float(timeline[-1] - timeline[0])
                except Exception:
                    duration_sec = float('nan')
            except Exception as _e:
                # Loading failed (including parse errors from empty files): assign 0 score and continue
                try:
                    if FILE_TYPE == "parquet" and parquet_episode_refs:
                        parquet_path, ep_idx = parquet_episode_refs[int(idx)]
                        base_dir = f"{parquet_path}#episode={int(ep_idx)}"
                    else:
                        assert fi is not None
                        base_dir = fi.path_by_index(int(idx))
                except Exception:
                    base_dir = str(idx)
                duration_sec = 0.0
                line = f"{idx}\t{0.0:.4f}\t{0.0:.4f}\t{0.0:.4f}\t{0.0:.4f}\t{duration_sec:.4f}\t{base_dir}\n"
                results_list.append(line)
                elapsed = time.time() - t_idx_start
                _info(f"[RUN] (w{worker_id}) trajectory {idx} failed to load or is empty: {_e!r}, wrote score=0, elapsed {elapsed:.2f}s")
                continue
            _rbt_tgt = RbtKin()
            _rbt_act = RbtKin()
            try:
                adj = np.asarray(_adj, dtype=float).reshape(-1)
            except Exception:
                adj = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
            if adj.size < 6:
                adj = np.pad(adj, (0, 6 - adj.size), mode='constant', constant_values=0.0)
            _x = float(adj[0]); _z = float(adj[2])
            _info(f"\n[RUN] (w{worker_id}) trajectory index={idx}, end-effector adjustment=({_x:.2f}, 0.0, {_z:.2f}) m")
            _res = replay(trajectory_index=idx,
                          adjust_ee_dist=adj,
                          traj_left_pose=_trajL,
                          traj_right_pose=_trajR,
                          traj_left_gripper=_trajLG,
                          traj_right_gripper=_trajRG,
                          rbt_tgt_obj=_rbt_tgt,
                          rbt_act_obj=_rbt_act)
            # Parse returned score result dictionary
            try:
                total_min_score = _res.get("total_min_score", None) if isinstance(_res, dict) else None
                total_min_index = _res.get("total_min_index", -1) if isinstance(_res, dict) else -1
                total_min_detail = _res.get("detail", None) if isinstance(_res, dict) else None
            except Exception:
                total_min_score, total_min_index, total_min_detail = None, -1, None

            total_min_score_f: Optional[float] = None
            if isinstance(total_min_score, (int, float)):
                try:
                    total_min_score_f = float(total_min_score)
                except Exception:
                    total_min_score_f = None

            if total_min_score_f is not None:
                cur_score = total_min_score_f
                c_idx = _res.get("continuous_min_index", -1) if isinstance(_res, dict) else -1
                w_idx = _res.get("out_ws_min_index", -1) if isinstance(_res, dict) else -1
                col_idx = _res.get("collision_min_index", -1) if isinstance(_res, dict) else -1
                _info(
                    f"[SCORE] (w{worker_id}) trajectory total_score={cur_score:.2f}, "
                    f"continuous_min_idx={c_idx}, workspace_min_idx={w_idx}, collision_min_idx={col_idx}"
                )
                best_score = cur_score
                best_detail = total_min_detail
            else:
                _info("[SCORE] No valid total score data.")

            # Collect one result row: idx, total_score, continuity, workspace, collision, duration_sec, path (no header)
            try:
                # Extract per-metric scores
                c_score = float('nan'); w_score = float('nan'); col_score = float('nan')
                if isinstance(_res, dict):
                    try:
                        c_score = float(_res.get("continuous_min_score", float('nan')))
                    except Exception:
                        c_score = float('nan')
                    try:
                        w_score = float(_res.get("out_ws_min_score", float('nan')))
                    except Exception:
                        w_score = float('nan')
                    try:
                        col_score = float(_res.get("collision_min_score", float('nan')))
                    except Exception:
                        col_score = float('nan')
                # Extract path
                base_dir = base_dir_for_idx
                # Append to shared results list
                line = f"{idx}\t{best_score:.4f}\t{c_score:.4f}\t{w_score:.4f}\t{col_score:.4f}\t{duration_sec:.4f}\t{base_dir}\n"
                results_list.append(line)
                elapsed = time.time() - t_idx_start
                if enable_writing_txt:
                    print(f"[SCORE] (w{worker_id}) final result row collected for trajectory index={idx}, elapsed {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - t_idx_start
                print(f"[SCORE] Failed to collect score result: {e!r}, elapsed {elapsed:.2f}s")

    # Start multi-process parallel execution (supports Ctrl+C interruption and child cleanup)
    procs = []
    stop_event = mp.Event()

    for wid, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = mp.Process(target=_worker_run, args=(wid, chunk, results_list, adj, stop_event))
        p.start()
        procs.append(p)
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        _info("[MAIN] Caught Ctrl+C, aborting scoring and terminating child processes...")
        stop_event.set()
        for p in procs:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass
    try:
        raw_lines = [ln for ln in list(results_list) if ln and ln.strip()]
        # Parse rows (new format): idx, best_score, c_score, w_score, col_score, duration_sec, path
        entries = []
        for ln in raw_lines:
            parts = ln.strip().split("\t")
            if len(parts) < 7:
                continue
            try:
                idx_val = int(parts[0])
                score = float(parts[1])
                c_score = float(parts[2]) if parts[2] else float('nan')
                w_score = float(parts[3]) if parts[3] else float('nan')
                col_score = float(parts[4]) if parts[4] else float('nan')
                duration_sec = float(parts[5]) if parts[5] else float('nan')
                path = parts[6]
            except Exception:
                # Skip unparseable rows
                continue
            entries.append({
                "idx": idx_val,
                "score": score,
                "c_score": c_score,
                "w_score": w_score,
                "col_score": col_score,
                "duration_sec": duration_sec,
                "path": path,
            })
        # Sort by idx
        entries.sort(key=lambda e: e["idx"]) 

        # Build score distribution and index buckets
        buckets = {
            ">90": [],
            "80-90": [],
            "70-80": [],
            "60-70": [],
            "0-60": [],
            "0": [],
        }
        for e in entries:
            s = e["score"]
            idx_val = e["idx"]
            if s > 90.0:
                buckets[">90"].append(idx_val)
            elif 80.0 <= s <= 90.0:
                buckets["80-90"].append(idx_val)
            elif 70.0 <= s < 80.0:
                buckets["70-80"].append(idx_val)
            elif 60.0 <= s < 70.0:
                buckets["60-70"].append(idx_val)
            elif s == 0.0:
                buckets["0"].append(idx_val)
            elif 0.0 < s < 60.0:
                buckets["0-60"].append(idx_val)

        # Summary stats: total count, ratio above 90, ratio above/equal 80
        total_count = len(entries)
        gt90_count = len(buckets['>90'])
        ge80_count = gt90_count + len(buckets['80-90'])
        ratio_gt90 = (gt90_count / total_count) if total_count > 0 else 0.0
        ratio_ge80 = (ge80_count / total_count) if total_count > 0 else 0.0

        # enable_writing_txt=False: do not write files, only return ratio above 90
        if not enable_writing_txt:
            return float(ratio_gt90)

        # enable_writing_txt=True: keep existing result-writing logic (without x/z and weighted_center)
        out_path = os.path.join("log", f"{ROBOT}_score_results_{ts}.txt")
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write("idx\tbest_score\tcontinuous\tworkspace\tcollision\tduration_sec\tpath\n")
            for e in entries:
                dur = e.get('duration_sec', float('nan'))
                fout.write(f"{e['idx']}\t{e['score']:.4f}\t{e['c_score']:.4f}\t{e['w_score']:.4f}\t{e['col_score']:.4f}\t{float(dur):.4f}\t{e['path']}\n")
        _info(f"[SCORE] Results written to: {out_path}")

        # Write summary stats to a separate file
        summary_path = os.path.join("log", f"{ROBOT}_score_summary_{ts}.txt")
        with open(summary_path, "w", encoding="utf-8") as fout:
            fout.write(f"amount:\t{total_count}\n")
            fout.write(f"count( >90 )\t{len(buckets['>90'])}\tidxs\t{','.join(map(str, buckets['>90']))}\n")
            fout.write(f"count(80-90)\t{len(buckets['80-90'])}\tidxs\t{','.join(map(str, buckets['80-90']))}\n")
            fout.write(f"count(70-80)\t{len(buckets['70-80'])}\tidxs\t{','.join(map(str, buckets['70-80']))}\n")
            fout.write(f"count(60-70)\t{len(buckets['60-70'])}\tidxs\t{','.join(map(str, buckets['60-70']))}\n")
            fout.write(f"count( 0-60)\t{len(buckets['0-60'])}\tidxs\t{','.join(map(str, buckets['0-60']))}\n")
            fout.write(f"count(  0  )\t{len(buckets['0'])}\tidxs\t{','.join(map(str, buckets['0']))}\n")

            # Trajectory duration statistics (seconds): mean/median/min/max
            durs = []
            dur_idx_pairs = []
            for e in entries:
                d = e.get('duration_sec', float('nan'))
                try:
                    d = float(d)
                except Exception:
                    d = float('nan')
                if not (math.isnan(d) or math.isinf(d)):
                    durs.append(d)
                    dur_idx_pairs.append((d, int(e.get('idx', -1))))
            if len(durs) == 0:
                fout.write("duration(sec)\ttotal:0.0000\n")
            else:
                durs_np = np.asarray(durs, dtype=float)
                d_total = float(np.sum(durs_np))
                d_mean = float(np.mean(durs_np))
                d_median = float(np.median(durs_np))
                d_min = float(np.min(durs_np))
                d_max = float(np.max(durs_np))
                # Index for shortest/longest duration (if tied, take first)
                dur_idx_pairs.sort(key=lambda x: x[0])
                min_idx = dur_idx_pairs[0][1]
                max_idx = dur_idx_pairs[-1][1]
                fout.write(f"duration(sec)\ttotal:{d_total:.4f}\tmean:{d_mean:.4f}\tmedian:{d_median:.4f}\tmin:{d_min:.4f}\tmax:{d_max:.4f}\n")
        _info(f"[SCORE] Summary written to: {summary_path}")

        return {
            "ratio_gt90": float(ratio_gt90),
            "ratio_ge80": float(ratio_ge80),
            "out_path": out_path,
            "summary_path": summary_path,
        }
    except Exception as e:
        _info(f"[SCORE] Failed to write results: {e!r}")
        return (0.0 if (not enable_writing_txt) else {"ratio_gt90": 0.0, "ratio_ge80": 0.0})
    
def search_best_score(adj: Optional[np.ndarray] = None):
    # In score-search mode, force viewer off and quiet mode on
    global START_VIEWER, QUIET_MODE, HOME_TIME, ADJUST_TIME, STEP_SCALE, REPLAY_TYPE, LOG_FLAG
    START_VIEWER = 0
    QUIET_MODE   = 1
    HOME_TIME    = 0.5 # Reduce impact on scoring runtime
    ADJUST_TIME  = 0.2 # Reduce impact on scoring runtime
    STEP_SCALE   = 20.0 # Avoid runtime inflation caused by speed limits; replay loop limiting must stay consistent to keep score parity
    REPLAY_TYPE  = 0  # Score mode uses pose trajectories only
    LOG_FLAG     = False  # Do not generate replay logs during search
    
    best_adj = None
    best_ratio = -1.0
    # Three-stage search:
    # 0) test x=z=0
    # 1) if best_ratio != 1, add x,z in {-0.1, 0, 0.1}
    # 2) if best_ratio != 1, search 3x3 neighborhood around current best_adj with step 0.1
    #    (some points overlap with stage 0/1 and are skipped)
    x_list = [0.0]
    z_list = [0.0]

    evaluated = set()

    def _eval_point(x: float, z: float):
        nonlocal best_adj, best_ratio
        key = (float(x), float(z))
        if key in evaluated:
            return
        evaluated.add(key)

        cur_adj = np.array([x, 0.0, z, x, 0.0, z], dtype=float)
        ratio_gt90 = score(enable_writing_txt=False, adj=cur_adj)
        if isinstance(ratio_gt90, (int, float)):
            ratio_gt90_val = float(ratio_gt90)
            _info(f"[SEARCH] ratio above 90={ratio_gt90_val:.6f}")
            if ratio_gt90_val > best_ratio:
                best_ratio = ratio_gt90_val
                best_adj = cur_adj.copy()
            elif ratio_gt90_val == best_ratio:
                # If scores are tied, choose adjustment closer to center
                if best_adj is not None:
                    best_dist = np.linalg.norm(best_adj[[0, 2]])
                    cur_dist = np.linalg.norm(cur_adj[[0, 2]])
                    if cur_dist < best_dist:
                        best_adj = cur_adj.copy()
        else:
            _info(f"[SEARCH] ratio above 90={ratio_gt90}")

    def _is_done() -> bool:
        return bool(best_ratio >= 1.0 - 1e-12)

    # Stage 0: only (0,0)
    _eval_point(0.0, 0.0)
    # if _is_done():
    #     _info(f"\n[SEARCH] Search complete, best end-effector adjustment=({best_adj[0]:.2f}, {best_adj[1]:.2f}, {best_adj[2]:.2f}) m, ratio above 90={best_ratio:.6f}")
    #     return best_adj, best_ratio

    # # Stage 1: add {-0.1, 0, 0.1} grid
    # x_list = [-0.1, 0.0, 0.1]
    # z_list = [-0.1, 0.0, 0.1]
    # for x in x_list:
    #     for z in z_list:
    #         _eval_point(x, z)
    # if _is_done():
    #     _info(f"\n[SEARCH] Search complete, best end-effector adjustment=({best_adj[0]:.2f}, {best_adj[1]:.2f}, {best_adj[2]:.2f}) m, ratio above 90={best_ratio:.6f}")
    #     return best_adj, best_ratio

    # # Stage 2: search 3x3 neighborhood around current best_adj
    # if best_adj is not None:
    #     bx = float(best_adj[0])
    #     bz = float(best_adj[2])
    #     neigh_x = [bx - 0.1, bx, bx + 0.1]
    #     neigh_z = [bz - 0.1, bz, bz + 0.1]
    #     for x in neigh_x:
    #         for z in neigh_z:
    #             _eval_point(x, z)
    # _info(f"\n[SEARCH] Search complete, best end-effector adjustment=({best_adj[0]:.2f}, {best_adj[1]:.2f}, {best_adj[2]:.2f}) m, ratio above 90={best_ratio:.6f}")
    return best_adj, best_ratio

if __name__ == "__main__":
    # CLI: python replay.py 1
    idx = 1
    # Single-trajectory scoring
    if len(sys.argv) == 2:
        try:
            idx = int(sys.argv[1])
            adj = ADJUST_EE_DIST
            _res = replay(trajectory_index=idx, adjust_ee_dist=adj)
            _info(f"[MAIN] Replay completed for trajectory index={idx}, result: {_res}")
        except Exception as e:
            _info(f"[MAIN] Replay failed: {e!r}")
            sys.exit(2)
    # Search best end-effector adjustment for all trajectories in one task
    elif len(sys.argv) == 1:
        try:
            best_adj,best_ratio=search_best_score()
            score(enable_writing_txt=True, adj=best_adj)
        except Exception as e:
            _info(f"[MAIN] Score mode run failed: {e!r}")
            sys.exit(2)
    else:
        _info("[MAIN] Invalid arguments. Usage:")
        _info("  Replay single trajectory: python replay.py <trajectory_index>")
        _info("  Score multiple trajectories: python replay.py")
        sys.exit(2)
    
    
