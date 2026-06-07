import numpy as np
from enum import Enum

class State(Enum):
    IDLE         = 0
    GO_HOME      = 1
    SLOW_STOP    = 2
    GO_INIT      = 3
    WAIT         = 4   # pause between transitions
    ADJUST_EE    = 5
     
    REPLAY_POSE  = 6   # playback mode
    REPLAY_JNT   = 7   # playback mode
    # REPLAY_DELTA = 8   # playback mode
    TELEOP       = 9   # teleoperation mode
    VLA_MOVE     = 10  # variable admittance mode

def concat_qpos_parts(qpos_dict) -> np.ndarray:
    parts = [
        np.asarray(qpos_dict.get("chassis", []), dtype=float),
        np.asarray(qpos_dict.get("torso", []), dtype=float),
        np.asarray(qpos_dict.get("left_arm", []), dtype=float),
        np.asarray(qpos_dict.get("right_arm", []), dtype=float),
        np.asarray(qpos_dict.get("left_gripper", []), dtype=float),
        np.asarray(qpos_dict.get("right_gripper", []), dtype=float),
    ]
    nonempty = [p for p in parts if p.size > 0]
    if not nonempty:
        return np.zeros((0,), dtype=float)
    return np.concatenate(nonempty, axis=0)