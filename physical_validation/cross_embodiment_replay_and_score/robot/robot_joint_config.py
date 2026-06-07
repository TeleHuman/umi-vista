# -*- coding: utf-8 -*-

"""Robot joint config loader + strict validators.

This module is intended to be imported by multiple scripts (e.g. replay/teleop/etc)
so that joint counts, slices, and vector validation stay consistent.

Behavior matches the original inlined block:
- Loads JSON from $ROBOT_JOINT_CONFIG or configs/robot_joint_config.json
- Fails fast with SystemExit(1) when required config is missing/invalid
- Provides strict length/type checks (no pad/trunc)
"""

from __future__ import annotations
import json
import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_CONFIGS_DIR = os.path.join(_REPO_ROOT, "configs")

def _fatal(msg: str) -> None:
    # Use SystemExit so it won't be swallowed by `except Exception` blocks.
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(1)

GRIPPER_OPEN      = 100.0
GRIPPER_CLOSE     = 0.0

def _load_robot_joint_config() -> Dict[str, Any]:
    """Load robot joint config JSON.

    Path resolution order:
    1) env var ROBOT_JOINT_CONFIG (explicit path)
    2) env var ROBOT_NAME -> ./configs/robots/<ROBOT_NAME>.json
    3) ./configs/robot_joint_config.json

    Expected keys:
    - counts: chassis, torso, left_arm, right_arm, left_gripper, right_gripper
    - vectors: home_pos, exit_pos, max_step_rad
      Each vector may be:
        (a) a flat list (length ~= total joints)
        (b) a dict of per-part lists (keys same as counts)
    """

    robot_name = os.getenv("ROBOT_NAME")
    if robot_name:
        cfg_path = os.path.join(_CONFIGS_DIR, f"{robot_name}.json")
    else:
        raise RuntimeError("Environment variable ROBOT_NAME not set")
    try:
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

ROBOT_CFG: Dict[str, Any] = _load_robot_joint_config()
if not ROBOT_CFG:
    _fatal(
        "Robot joint config JSON missing/invalid. Set ROBOT_JOINT_CONFIG or create configs/robot_joint_config.json"
    )

def _require_nonneg_int_from_cfg(key: str) -> int:
    if key not in ROBOT_CFG:
        _fatal(f"Missing required int field in robot config: '{key}'")
    try:
        raw = ROBOT_CFG.get(key)
        if raw is None:
            _fatal(f"Field '{key}' must be an int, got: None")
        v = int(raw)
    except Exception:
        _fatal(f"Field '{key}' must be an int, got: {ROBOT_CFG.get(key)!r}")
    if v < 0:
        _fatal(f"Field '{key}' must be >= 0, got: {v}")
    return v

JOINT_COUNTS = {
    "chassis": _require_nonneg_int_from_cfg("chassis"),
    "torso": _require_nonneg_int_from_cfg("torso"),
    "left_arm": _require_nonneg_int_from_cfg("left_arm"),
    "right_arm": _require_nonneg_int_from_cfg("right_arm"),
    "left_gripper": _require_nonneg_int_from_cfg("left_gripper"),
    "right_gripper": _require_nonneg_int_from_cfg("right_gripper"),
}

N_CHASSIS = int(JOINT_COUNTS["chassis"])
N_TORSO = int(JOINT_COUNTS["torso"])
N_LEFT_ARM = int(JOINT_COUNTS["left_arm"])
N_RIGHT_ARM = int(JOINT_COUNTS["right_arm"])
N_LEFT_GRIPPER = int(JOINT_COUNTS["left_gripper"])
N_RIGHT_GRIPPER = int(JOINT_COUNTS["right_gripper"])
N_QPOS_TOTAL = N_CHASSIS + N_TORSO + N_LEFT_ARM + N_RIGHT_ARM + N_LEFT_GRIPPER + N_RIGHT_GRIPPER

_off = 0
SL_CHASSIS = slice(_off, _off + N_CHASSIS)
_off += N_CHASSIS
SL_TORSO = slice(_off, _off + N_TORSO)
_off += N_TORSO
SL_LEFT_ARM = slice(_off, _off + N_LEFT_ARM)
_off += N_LEFT_ARM
SL_RIGHT_ARM = slice(_off, _off + N_RIGHT_ARM)
_off += N_RIGHT_ARM
SL_LEFT_GRIPPER = slice(_off, _off + N_LEFT_GRIPPER)
_off += N_LEFT_GRIPPER
SL_RIGHT_GRIPPER = slice(_off, _off + N_RIGHT_GRIPPER)
_off += N_RIGHT_GRIPPER

def _require_1d_float_array(arr: Any, expected_len: int, ctx: str) -> np.ndarray:
    try:
        v = np.asarray(arr, dtype=float).reshape(-1)
    except Exception:
        _fatal(f"{ctx}: value must be array-like of floats, got {type(arr).__name__}: {arr!r}")
        raise

    if expected_len <= 0:
        if v.size != 0:
            _fatal(f"{ctx}: expected length 0, got {v.size}")
        return np.zeros((0,), dtype=float)

    if v.size != expected_len:
        _fatal(f"{ctx}: expected length {expected_len}, got {v.size}")

    return v


def _require_fb_vec(container: Dict[str, Any], key: str, expected_len: int, ctx: str) -> np.ndarray:
    if expected_len > 0 and key not in container:
        _fatal(f"{ctx}: missing key '{key}' (expected length {expected_len})")
    return _require_1d_float_array(container.get(key, []), expected_len, ctx)


def _load_qpos_vector_from_cfg(key: str) -> np.ndarray:
    """Load a qpos-like vector strictly from ROBOT_CFG.

    Allowed formats:
    - flat list of length N_QPOS_TOTAL
    - dict of per-part lists, each exactly matching its configured count

    No padding/truncation is performed. Any mismatch triggers a fatal error.
    """

    if key not in ROBOT_CFG:
        _fatal(f"Missing required vector field in robot config: '{key}'")

    raw = ROBOT_CFG.get(key)

    if isinstance(raw, (list, tuple, np.ndarray)):
        return _require_1d_float_array(raw, N_QPOS_TOTAL, f"config.{key}")

    if isinstance(raw, dict):

        def _part(name: str, expected: int) -> np.ndarray:
            if expected <= 0:
                # Allow missing or empty for 0-dof parts
                if name not in raw:
                    return np.zeros((0,), dtype=float)
                return _require_1d_float_array(raw.get(name, []), 0, f"config.{key}.{name}")
            if name not in raw:
                _fatal(f"config.{key}: missing part '{name}' (expected length {expected})")
            return _require_1d_float_array(raw.get(name), expected, f"config.{key}.{name}")

        parts = [
            _part("chassis", N_CHASSIS),
            _part("torso", N_TORSO),
            _part("left_arm", N_LEFT_ARM),
            _part("right_arm", N_RIGHT_ARM),
            _part("left_gripper", N_LEFT_GRIPPER),
            _part("right_gripper", N_RIGHT_GRIPPER),
        ]

        nonempty = [p for p in parts if p.size > 0]
        if not nonempty:
            return np.zeros((0,), dtype=float)

        vec = np.concatenate(nonempty, axis=0)
        if vec.size != N_QPOS_TOTAL:
            _fatal(f"config.{key}: concatenated length {vec.size} != N_QPOS_TOTAL {N_QPOS_TOTAL}")
        return vec

    _fatal(f"config.{key}: must be a list or dict, got {type(raw).__name__}")
    return np.zeros((0,), dtype=float)
