from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import math
import time
import threading
import numpy as np
import utils.self_math as smath
import utils.lowpass_filter as flt

import mink

_HERE = Path(__file__).parent.parent
_XML = _HERE / "model" / "rm75" / "front_mounted_scene.xml"
_ARM_JOINT_NAMES = [
    "joint1_left",
    "joint2_left",
    "joint3_left",
    "joint4_left",
    "joint5_left",
    "joint6_left",
    "joint7_left",
    "joint1_right",
    "joint2_right",
    "joint3_right",
    "joint4_right",
    "joint5_right",
    "joint6_right",
    "joint7_right",
]
_GRIPPER_JOINT_NAMES = {
    "left": {
        "driver": "robotiq_85_left_knuckle_joint_left",
        "knuckle": (
            "robotiq_85_left_knuckle_joint_left",
            "robotiq_85_right_knuckle_joint_left",
        ),
        "inner": (
            "robotiq_85_left_inner_knuckle_joint_left",
            "robotiq_85_right_inner_knuckle_joint_left",
        ),
        "tip": (
            "robotiq_85_left_finger_tip_joint_left",
            "robotiq_85_right_finger_tip_joint_left",
        ),
    },
    "right": {
        "driver": "robotiq_85_left_knuckle_joint_right",
        "knuckle": (
            "robotiq_85_left_knuckle_joint_right",
            "robotiq_85_right_knuckle_joint_right",
        ),
        "inner": (
            "robotiq_85_left_inner_knuckle_joint_right",
            "robotiq_85_right_inner_knuckle_joint_right",
        ),
        "tip": (
            "robotiq_85_left_finger_tip_joint_right",
            "robotiq_85_right_finger_tip_joint_right",
        ),
    },
}
_MAPPING = [
    {"mj_name": name, "robot_index": idx, "scale": 1.0, "offset": 0.0, "sign": 1.0}
    for idx, name in enumerate(_ARM_JOINT_NAMES)
] + [
    {
        "mj_name": _GRIPPER_JOINT_NAMES["left"]["driver"],
        "robot_index": 14,
        "scale": 1.0,
        "offset": 0.0,
        "sign": 1.0,
    },
    {
        "mj_name": _GRIPPER_JOINT_NAMES["right"]["driver"],
        "robot_index": 15,
        "scale": 1.0,
        "offset": 0.0,
        "sign": 1.0,
    },
]

class DoubleRm75Mink:
    def __init__(self):
        model = mujoco.MjModel.from_xml_path(_XML.as_posix())

        self.configuration = mink.Configuration(model)
        self.hands = ["left_gripper_site", "right_gripper_site"]
        self.posture_task = mink.PostureTask(model, cost=1e-1)
        self.tasks = [
            self.posture_task,
        ]
        self.hand_tasks = []
        for hand in self.hands:
            task = mink.FrameTask(
                frame_name=hand,
                frame_type="site",
                position_cost=100.0,
                orientation_cost=100.0,
                lm_damping=1.0,
            )
            self.hand_tasks.append(task)
        self.tasks.extend(self.hand_tasks)
        collision_pairs = [(["left_hand_collision"], ["left_base_collision"]),
                           (["right_hand_collision"], ["right_base_collision"])]
        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.01,
            collision_detection_distance=0.1,
        )
        self.limits = [
            mink.ConfigurationLimit(model),
            collision_avoidance_limit,
        ]
        self.hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in self.hands]

        self.model = self.configuration.model
        self.data = self.configuration.data
        # Separate render buffer to avoid viewer touching live data directly
        self._render_data = mujoco.MjData(self.model)
        # Concurrent-access protection: prevent viewer and control loop from modifying/copying mjData at the same time,
        # which can trigger mj_copyDataVisual errors. Use a reentrant lock to avoid deadlocks in nested locked calls.
        self._mj_lock = threading.RLock()
        self.solver = "daqp"
    
        # Initialize to the home keyframe.
        self.configuration.update_from_keyframe("teleop")
        self.posture_task.set_target_from_configuration(self.configuration)
        # Initialize mocap bodies at their respective sites.
        for hand in self.hands:
            mink.move_mocap_to_frame(self.model, self.data, f"{hand}_target", hand, "site")

        # Record baseline mocap positions (hands + COM) and (quaternions, MuJoCo order [w,x,y,z]).
        self.hands_base = [self.data.mocap_pos[mid].copy() for mid in self.hands_mid]
        self.hands_base_quat = [self.data.mocap_quat[mid].copy() for mid in self.hands_mid]
        # Gripper joint name list (for unit conversion mm -> m)
        self.gripper_joint_max = 0.80285 # close, 0 is open
        self._sync_gripper_from_current_locked()
        
    def _gripper_pct_to_joint(self, value_pct: float) -> float:
        pct = float(np.clip(value_pct, 0.0, 100.0))
        return (100.0-pct) / 100.0 * self.gripper_joint_max

    def _gripper_joint_to_pct(self, joint_value: float) -> float:
        if self.gripper_joint_max <= 1e-9:
            return 0.0
        pct = (1.0 - float(joint_value) / self.gripper_joint_max) * 100.0
        return float(np.clip(pct, 0.0, 100.0))
        
    def _set_gripper_side_locked(self, side: str, value_pct: float) -> None:
        joint_value = self._gripper_pct_to_joint(value_pct)
        joints = _GRIPPER_JOINT_NAMES[side]
        for joint_name in joints["knuckle"] + joints["inner"]:
            self.data.qpos[self.model.joint(joint_name).id] = joint_value
        for joint_name in joints["tip"]:
            self.data.qpos[self.model.joint(joint_name).id] = -joint_value
    
    def _get_gripper_pct_locked(self, side: str) -> float:
        driver_name = _GRIPPER_JOINT_NAMES[side]["driver"]
        joint_value = float(self.data.qpos[self.model.joint(driver_name).id])
        return self._gripper_joint_to_pct(joint_value)

    def _sync_gripper_from_current_locked(self) -> None:
        self._set_gripper_side_locked("left", self._get_gripper_pct_locked("left"))
        self._set_gripper_side_locked("right", self._get_gripper_pct_locked("right"))
    
        
    def solve_ik(self, tgt_hand_pose7):
        """Solve IK for given hand mocap targets (position + optional orientation).

        Args:
            tgt_hand_pos: list/tuple of two 3D position arrays.
            tgt_hand_quat: optional list of two quaternions [w,x,y,z]; if provided will set mocap orientations.
        Returns:
            numpy.ndarray: Updated joint positions `qpos` after integration (shape: (nq,)).
        """
        with self._mj_lock:
            left_gripper_pct = self._get_gripper_pct_locked("left")
            right_gripper_pct = self._get_gripper_pct_locked("right")
            self.data.mocap_pos[self.hands_mid[0]] = np.array(tgt_hand_pose7[0][0:3])
            self.data.mocap_pos[self.hands_mid[1]] = np.array(tgt_hand_pose7[1][0:3])
            self.data.mocap_quat[self.hands_mid[0]] = np.array(tgt_hand_pose7[0][3:7])
            self.data.mocap_quat[self.hands_mid[1]] = np.array(tgt_hand_pose7[1][3:7])

            # Dynamically adjust hands and COM costs (based on whether COM is inside/outside support triangle)
            for i, hand_task in enumerate(self.hand_tasks):
                hand_task.set_target(mink.SE3.from_mocap_id(self.data, self.hands_mid[i]))
            vel = mink.solve_ik(
                self.configuration, self.tasks, 0.005, self.solver, 1e-1, limits=self.limits
            )
            self.configuration.integrate_inplace(vel, 0.005)
            _camlight = getattr(mujoco, "mj_camlight", None)
            if _camlight is not None:
                _camlight(self.model, self.data)
            # Collect mapped qpos for return while still holding the lock
            arm_qpos = [
                float(self.data.qpos[self.model.joint(joint_name).id])
                for joint_name in _ARM_JOINT_NAMES
            ]
            qret = np.array(
                arm_qpos
                + [
                    self._get_gripper_pct_locked("left"),
                    self._get_gripper_pct_locked("right"),
                ],
                dtype=float,
            )
        return qret
    
    def forward_kinematics(self, frame_name: str, frame_type: str = "site", sync_mocap: bool = False):
        """Return current world pose of a site/body.

        Args:
            frame_name: name of the site or body in the model.
            frame_type: "site" or "body".

        Returns:
            (pos, quat): tuple of numpy arrays. pos shape (3,), quat shape (4,w,x,y,z).
        """
        # Ensure all MuJoCo access happens under the lock to avoid stack-in-use errors
        with self._mj_lock:
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            if sync_mocap:
                self._sync_mocap_targets_locked()

            if frame_type == "site":
                sid = self.model.site(frame_name).id
                pos = self.data.site_xpos[sid].copy()
                if hasattr(self.data, "site_xquat"):
                    quat = self.data.site_xquat[sid].copy()
                elif hasattr(self.data, "site_xmat"):
                    mat9 = self.data.site_xmat[sid].copy()
                    quat = np.empty(4, dtype=float)
                    try:
                        mujoco.mju_mat2Quat(quat, mat9)
                    except Exception:
                        # Fallback: identity if conversion utility is unavailable
                        quat = None
                else:
                    quat = None
                # print("*********************",pos,quat)
                return pos, quat
            elif frame_type == "body":
                bid = self.model.body(frame_name).id
                pos = self.data.xpos[bid].copy()
                quat = self.data.xquat[bid].copy()
                return pos, quat
            else:
                raise ValueError(f"Unsupported frame_type: {frame_type}. Use 'site' or 'body'.")
        
    def solve_fk(self):
        """Return current 7D poses of both hands as [x,y,z,qw,qx,qy,qz].

        Uses the actual hand sites (not the mocap target bodies) so this reflects
        the robot's current end-effector pose. Safe for seed/home capture.
        """
        poses7 = []
        for hand in self.hands:
            # site frame (not *_target) gives real TCP pose
            pos, quat = self.forward_kinematics(hand, "site", sync_mocap=False)# set_real_qpos has already synchronized this
            if quat is None:
                # Fallback: identity quaternion if missing
                quat = np.array([1.0,0.0,0.0,0.0], dtype=float)
            # MuJoCo site_xquat ordering is [w,x,y,z]; maintain same ordering used elsewhere
            pose7 = np.concatenate([pos.astype(float), quat.astype(float)])
            poses7.append(pose7)
        return poses7

    def set_real_qpos(self, q_real, sync_mocap: bool = False):
        """Set MuJoCo qpos from real robot encoders using a mapping.

        Args:
            q_real: sequence or numpy array of real robot joint positions (encoder readings).
            mapping: list of dicts, each with keys:
                - 'mj_name': MuJoCo joint name
                - 'robot_index': index in q_real
                - 'scale': unit scale (e.g., deg->rad). Default 1.0
                - 'offset': additive offset. Default 0.0
                - 'sign': +1 or -1 for direction. Default +1

        Writes mapped values into self.data.qpos in MuJoCo joint order, then runs forward update.
        """
        # Build name->id map once; perform full update in one locked block
        with self._mj_lock:
            q_mj = self.data.qpos.copy()
            for m in _MAPPING:
                mj_name = m['mj_name']
                ridx = m['robot_index']
                scale = m.get('scale', 1.0)
                offset = m.get('offset', 0.0)
                sign = m.get('sign', 1.0)
                jid = self.model.joint(mj_name).id
                # Unit conversion: gripper raw input is %, convert to m
                raw = q_real[ridx]
                val = sign * (raw * scale + offset)
                q_mj[jid] = val
            self.data.qpos[:] = q_mj
            self._set_gripper_side_locked("left", float(q_real[14]) if len(q_real) > 14 else 0.0)
            self._set_gripper_side_locked("right", float(q_real[15]) if len(q_real) > 15 else 0.0)
            has_range = getattr(self.model, 'jnt_range', None) is not None
            if has_range:
                for j in range(self.model.njnt):
                    lo, hi = self.model.jnt_range[j]
                    if lo < hi:
                        self.data.qpos[j] = float(np.clip(self.data.qpos[j], lo, hi))
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            if sync_mocap:
                self._sync_mocap_targets_locked()
    
    def update_configureation_to_home(self):
        """Update configuration from a named keyframe."""
        # with self._mj_lock:
        #     # 1. Apply keyframe
        #     self.configuration.update_from_keyframe("homepos")
        #     self._sync_gripper_from_current_locked()
        #     # 2. Forward update
        #     _fwd = getattr(mujoco, "mj_forward", None)
        #     if _fwd is not None:
        #         _fwd(self.model, self.data)
        #     else:
        #         _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
        #         if _fwd_pos is not None:
        #             _fwd_pos(self.model, self.data)
        #     # 3. Posture & torso orientation
        #     self.posture_task.set_target_from_configuration(self.configuration)
        pass
    
    def set_lock_flag(self, enabled: bool):
        pass

    def update_viewer(self, viewer=None):
        """Update MuJoCo derived data and optionally sync the viewer."""
        with self._mj_lock:
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            _sensor_pos = getattr(mujoco, "mj_sensorPos", None)
            if _sensor_pos is not None:
                _sensor_pos(self.model, self.data)
            if viewer is not None:
                viewer.sync()
            
    def launch_viewer(self, hand_amp=0.5, hand_freq=0.5):
        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            _free_cam = getattr(mujoco, "mjv_defaultFreeCamera", None)
            if _free_cam is not None:
                _free_cam(self.model, viewer.cam)
            t = 0.0
            rate = RateLimiter(frequency=200.0, warn=False)
            tgt_hand_pose7 =[np.hstack([self.hands_base[0].copy(),self.hands_base_quat[0].copy()]),
                             np.hstack([self.hands_base[1].copy(),self.hands_base_quat[1].copy()])]
            while viewer.is_running():
                # Drive simple periodic mocap motions so IK produces movement.
                t += rate.dt
                # meters, Hz (COM trajectory optional)
                # com_amp, com_freq = 0.2, 0.25

                w_hand = 2.0 * math.pi * hand_freq
                for i in range(2):
                    phase = math.pi if (i % 2 == 1) else 0.0  # alternate left/right
                    offset_x = hand_amp * math.sin(w_hand * t + phase)
                    new_pos = self.hands_base[i].copy()
                    new_pos[0] = new_pos[0] + offset_x
                    tgt_hand_pose7[i][0:3] = new_pos

                # Solve IK for current targets and update viewer
                self.solve_ik(tgt_hand_pose7)
                self.update_viewer(viewer)
                rate.sleep()
    
    # Test function for this class
    def test_traj(self, duration=5.0, viewer=False, hand_amp=0.5, hand_freq=0.5):
        """Run a simple sinusoidal hand trajectory.

        If viewer=False, only perform kinematic IK updates without rendering.
        """
        if viewer:
            return self.launch_viewer(hand_amp=hand_amp, hand_freq=hand_freq)

        # No viewer: run purely kinematic loop
        t = 0.0
        rate = RateLimiter(frequency=200.0, warn=False)
        tgt_hand_pose7 =[np.hstack([self.hands_base[0].copy(),self.hands_base_quat[0].copy()]),
                         np.hstack([self.hands_base[1].copy(),self.hands_base_quat[1].copy()])]
        steps = int(duration * 200)
        for _ in range(steps):
            t += rate.dt
            w_hand = 2.0 * math.pi * hand_freq
            for i in range(2):
                phase = math.pi if (i % 2 == 1) else 0.0
                offset_x = hand_amp * math.sin(w_hand * t + phase)
                new_pos = self.hands_base[i].copy()
                new_pos[0] = new_pos[0] + offset_x
                tgt_hand_pose7[i][0:3] = new_pos
                # Example: keep orientation constant; could add slow yaw rotation here if desired
            self.solve_ik(tgt_hand_pose7)
            self.update_viewer(viewer=None)
            rate.sleep()
        return True
    
    # Teleoperation and simulation interaction function
    def apply_mapped_qpos(self, q_mapped, sync_mocap: bool = False):
        """Apply a mapped qpos array (ordered like `_MAPPING`) into the MuJoCo sim.

        Args:
            q_mapped: iterable of floats with length == len(_MAPPING).
            sync_mocap: if True, move mocap targets (hands, COM) to match current frames
                        after updating qpos; useful for passive visualization without IK.
        """
        q = np.asarray(q_mapped, dtype=float).reshape(-1)
        if q.shape[0] < 1:
            return
        # Structure:
        # First 14 values: 7 (left arm) + 7 (right arm)
        # Optional 15th/16th values: two gripper scalars (left, right) in mm.
        # If provided, they are copied into 4 gripper joints in the model with mm->m conversion
        # (and the same /2 scaling convention as set_real_qpos).
        with self._mj_lock:
            # Write torso and arm values (no unit conversion involved)
            base_count = min(14, q.shape[0])
            for i in range(base_count):
                m = _MAPPING[i]
                jid = self.model.joint(m['mj_name']).id
                self.data.qpos[jid] = float(q[i])
            # Handle gripper (if 2 parameters are provided, [14]=left, [15]=right percentage)
            if q.shape[0] >= 16:
                left_grip_pct = float(q[14])
                right_grip_pct = float(q[15])
                self._set_gripper_side_locked("left", left_grip_pct)
                self._set_gripper_side_locked("right", right_grip_pct)
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            if sync_mocap:
                self._sync_mocap_targets_locked()

    def get_sim_feedback(self):
        """Return a minimal feedback dict from the current MuJoCo state.

        The dict contains qpos/qvel split into `torso`, `left_arm`, `right_arm`,
        plus placeholder gripper values. All joint lists are numpy arrays (radians).
        """
        # Build list of q values in the same order as _MAPPING
        with self._mj_lock:
            left_arm = np.array(
                [float(self.data.qpos[self.model.joint(name).id]) for name in _ARM_JOINT_NAMES[:7]],
                dtype=float,
            )
            right_arm = np.array(
                [float(self.data.qpos[self.model.joint(name).id]) for name in _ARM_JOINT_NAMES[7:14]],
                dtype=float,
            )
            left_arm_vel = np.array(
                [float(self.data.qvel[self.model.joint(name).id]) for name in _ARM_JOINT_NAMES[:7]],
                dtype=float,
            )
            right_arm_vel = np.array(
                [float(self.data.qvel[self.model.joint(name).id]) for name in _ARM_JOINT_NAMES[7:14]],
                dtype=float,
            )
        left_grip_fb = self._get_gripper_pct_locked("left")
        right_grip_fb = self._get_gripper_pct_locked("right")
        feedback = {
            'qpos': {
                'left_arm': left_arm,
                'right_arm': right_arm,
                'left_gripper': [left_grip_fb],
                'right_gripper': [right_grip_fb],
            },
            'qvel': {
                'left_arm': left_arm_vel,
                'right_arm': right_arm_vel,
            }
        }
        return feedback

    def _sync_mocap_targets_locked(self):
        """Assumes caller holds `self._mj_lock`.

        Move mocap bodies (hands targets and COM target) to the current frames so that
        visual markers match the robot state. This is helpful when not running IK.
        """
        for hand in self.hands:
            # Move the corresponding mocap body to the site's current pose
            mink.move_mocap_to_frame(self.model, self.data, f"{hand}_target", hand, "site")

    def start_viewer_thread(self, hand_amp=0.0, hand_freq=0.0, drive_mocap: bool = False, viewer_hz: float = 60.0, compute_hz: float = 200.0):
        """Start a background thread that runs a passive viewer.

        Call `stop_viewer_thread()` to stop it. This is a convenience wrapper
        useful when integrating simulation with an external control loop.
        """
        if hasattr(self, "_viewer_thread") and getattr(self, "_viewer_thread", None) is not None:
            return
        import threading
        self._viewer_stop = threading.Event()

        def _run():
            try:
                # Use a separate render buffer so viewer never reads live self.data
                with mujoco.viewer.launch_passive(model=self.model, data=self._render_data, show_left_ui=False, show_right_ui=False) as viewer:
                    _free_cam = getattr(mujoco, "mjv_defaultFreeCamera", None)
                    if _free_cam is not None:
                        _free_cam(self.model, viewer.cam)
                    rate = RateLimiter(frequency=max(1.0, float(compute_hz)), warn=False)
                    # viewer sync limiter (lower rate than compute)
                    viewer_interval = 1.0 / max(1.0, float(viewer_hz))
                    last_view_sync_t = time.time()
                    while viewer.is_running() and not self._viewer_stop.is_set():
                        # simple periodic mocap motion if requested
                        t = time.time()
                        w_hand = 2.0 * math.pi * hand_freq
                        # Update live data and copy to render buffer under lock
                        with self._mj_lock:
                            if drive_mocap and hand_amp != 0.0:
                                for i in range(2):
                                    phase = math.pi if (i % 2 == 1) else 0.0
                                    offset_x = hand_amp * math.sin(w_hand * t + phase)
                                    new_pos = self.hands_base[i].copy()
                                    new_pos[0] = new_pos[0] + offset_x
                                    self.data.mocap_pos[self.hands_mid[i]] = new_pos
                            # Compute derived quantities on live data (fast loop)
                            _fwd = getattr(mujoco, "mj_forward", None)
                            if _fwd is not None:
                                _fwd(self.model, self.data)
                            else:
                                _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
                                if _fwd_pos is not None:
                                    _fwd_pos(self.model, self.data)
                                _sensor_pos = getattr(mujoco, "mj_sensorPos", None)
                                if _sensor_pos is not None:
                                    _sensor_pos(self.model, self.data)
                            # Copy live data to render buffer safely
                            _copy = getattr(mujoco, "mj_copyData", None)
                            if _copy is not None:
                                _copy(self._render_data, self.model, self.data)
                            else:
                                # Fallback: minimal fields
                                self._render_data.qpos[:] = self.data.qpos
                                self._render_data.qvel[:] = self.data.qvel
                                self._render_data.mocap_pos[:] = self.data.mocap_pos
                                self._render_data.mocap_quat[:] = self.data.mocap_quat
                                _fwd2 = getattr(mujoco, "mj_forward", None)
                                if _fwd2 is not None:
                                    _fwd2(self.model, self._render_data)
                        # Sync the viewer at throttled rate, reading only render buffer
                        now_t = time.time()
                        if (now_t - last_view_sync_t) >= viewer_interval:
                            viewer.sync()
                            last_view_sync_t = now_t
                        rate.sleep()
            except Exception:
                pass

        self._viewer_thread = threading.Thread(target=_run, daemon=True)
        self._viewer_thread.start()

    def stop_viewer_thread(self):
        if hasattr(self, "_viewer_stop") and self._viewer_stop is not None:
            self._viewer_stop.set()
        if hasattr(self, "_viewer_thread") and self._viewer_thread is not None:
            self._viewer_thread.join(timeout=1.0)
            self._viewer_thread = None

    def check_collision_pairs(self):
        """Check collisions across configured geom groups.

        Returns a flat result: {"colliding": bool, "min_dist": [d0, d1, ...]}
        where each element in min_dist is the smallest contact distance for the
        corresponding configured pair (negative means penetration). The overall
        colliding is True if any pair has penetration.
        """
        min_dists = []
        scores = []
        any_colliding = False
        with self._mj_lock:
            _fwd = getattr(mujoco, "mj_forward", None)
            if _fwd is not None:
                _fwd(self.model, self.data)
            pairs = [(["left_hand_collision"], ["left_base_collision"]),
                     (["right_hand_collision"], ["right_base_collision"])]
            for groupA, groupB in pairs:
                idsA = []
                idsB = []
                for name in groupA:
                    try:
                        idsA.append(self.model.geom(name).id)
                    except Exception:
                        pass
                for name in groupB:
                    try:
                        idsB.append(self.model.geom(name).id)
                    except Exception:
                        pass
                min_dist = float("inf")
                ncon = int(self.data.ncon)
                pair_colliding = False
                for ci in range(ncon):
                    con = self.data.contact[ci]
                    g1 = int(con.geom1)
                    g2 = int(con.geom2)
                    if (g1 in idsA and g2 in idsB) or (g1 in idsB and g2 in idsA):
                        dist = float(con.dist)
                        if dist < min_dist:
                            min_dist = dist
                        if dist < 0.005:
                            pair_colliding = True
                if min_dist == float("inf"):
                    # No contact constraints matched this pair. This is normal when geoms are separated,
                    # especially if geom margins are 0. Use mj_geomDistance to get actual separation.
                    _gd = getattr(mujoco, "mj_geomDistance", None)
                    if _gd is not None and len(idsA) > 0 and len(idsB) > 0:
                        distmax = 1.0  # meters; increase if your model has larger separations
                        best = float("inf")
                        best_pair = None
                        for ga in idsA:
                            for gb in idsB:
                                try:
                                    dsep = float(_gd(self.model, self.data, int(ga), int(gb), float(distmax), None))
                                    if dsep < best:
                                        best = dsep
                                        best_pair = (int(ga), int(gb))
                                except Exception:
                                    pass
                        if best != float("inf"):
                            min_dist = best
                        else:
                            min_dist = float("nan")
                    else:
                        min_dist = float("nan")
                min_dists.append(min_dist)
                # Score mapping: >0.045 -> 100; [0.005,0.045] -> linear to 0; <0 -> 0; NaN -> 100
                if np.isnan(min_dist):
                    score = 100.0
                elif min_dist >= 0.045:
                    score = 100.0
                elif min_dist <= 0.005:
                    score = 0.0
                else:
                    score = 100.0 * ((min_dist-0.005) / 0.04)
                scores.append(float(score))
                any_colliding = any_colliding or pair_colliding
            overall_score = float(min(scores)) if len(scores) > 0 else 100.0
            return {"colliding": bool(any_colliding), "min_dist": min_dists, "score": overall_score}

    def get_hand_target_pose_error(self):
        """Compute 6D error between hand target mocaps and actual hand sites.

        Returns a list with two dicts (left, right):
        - pos_error: Euclidean distance between mocap pos and site pos (meters)
        - ori_error_rad: rotation angle between mocap quat and site quat (radians)
        - pos_delta: 3-vector difference (target - actual)
        - ori_axis_angle: 3-vector axis * angle (angle-axis) representing orientation error
        """
        errs = []
        with self._mj_lock:
            # Make sure derived quantities are current
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            for i, hand in enumerate(self.hands):
                # Target from mocap
                tpos = self.data.mocap_pos[self.hands_mid[i]].copy()
                tquat = self.data.mocap_quat[self.hands_mid[i]].copy()
                # Actual site pose
                spos, squat = self.forward_kinematics(hand, "site", sync_mocap=False)
                if squat is None:
                    squat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                # Position error
                dp = tpos - spos
                pos_err = float(np.linalg.norm(dp))
                # Orientation error: q_err = q_target * inv(q_actual)
                qw_t, qx_t, qy_t, qz_t = tquat
                qw_a, qx_a, qy_a, qz_a = squat
                # inverse actual
                q_inv_a = np.array([qw_a, -qx_a, -qy_a, -qz_a], dtype=float)
                q_err = smath.quat_multiply(np.array([qw_t, qx_t, qy_t, qz_t], dtype=float), q_inv_a)
                # angle from quaternion using shortest rotation (q and -q are equivalent)
                w = float(q_err[0])
                w = max(-1.0, min(1.0, w))
                ang = 2.0 * math.acos(abs(w))
                # normalize axis from vector part; handle near-zero rotation robustly
                v = np.array([q_err[1], q_err[2], q_err[3]], dtype=float)
                v_norm = float(np.linalg.norm(v))
                if v_norm > 1e-12:
                    axis = v / v_norm
                else:
                    axis = np.array([0.0, 0.0, 0.0], dtype=float)
                ori_axis_angle = axis * ang
                errs.append({
                    "pos_error": pos_err,
                    "ori_error_rad": float(ang),
                    "pos_delta": dp,
                    "ori_axis_angle": ori_axis_angle,
                })
        return errs
    
    def check_out_workspace(self):
        """Determine whether workspace limits are exceeded based on 6D error between target and actual hand pose.

        Rule: if either hand has `pos_error` or `ori_error_rad` > 0.03, mark as out-of-workspace.
        Returns: {"any_out": bool, "left": {...}, "right": {...}}
        left/right include: pos_error, ori_error_rad, pos_delta, ori_axis_angle, out.
        """
        errs = self.get_hand_target_pose_error()
        result_per_hand = []
        any_out = False
        overall_score = 100.0
        for e in errs:
            out = (float(e.get("pos_error", 0.0)) > 0.01) or (float(e.get("ori_error_rad", 0.0)) > 2.0/180.0*math.pi) # and not self._is_com_inside_support() ?
            # Scoring rules:
            # - Position error p: 0 -> 100, 0.045 -> 60; above threshold, use exponential decay asymptotically toward 0
            # - Angular error a_deg: 0 -> 100, 9 deg -> 60; above threshold, use exponential decay asymptotically toward 0
            p = float(e.get("pos_error", 0.0))
            a_rad = float(e.get("ori_error_rad", 0.0))
            a_deg = a_rad * 180.0 / math.pi
            if p<=0.005:
                score_pos = 100.0
            elif p <= 0.045:
                score_pos = 100.0 - 40.0 * ((p-0.005) / 0.04)
            else:
                # Progressive decay: continuous at p=0.045 with score 60, then exponentially decays toward 0
                delta_p = p - 0.045
                decay_scale_p = 0.1  # Decay scale of about 10 cm
                score_pos = 60.0 * math.exp(-max(0.0, delta_p) / max(1e-6, decay_scale_p))
            if a_deg <= 1.0:
                score_ori = 100.0
            elif a_deg <= 9.0:
                score_ori = 100.0 - 40.0 * (a_deg / 8.0)
            else:
                # Progressive decay: continuous at a=9 degrees with score 60, then exponentially decays toward 0
                delta_a = a_deg - 9.0
                decay_scale_a = 20.0  # Decay scale of about 20 degrees
                score_ori = 60.0 * math.exp(-max(0.0, delta_a) / max(1e-6, decay_scale_a))
            score = min(score_pos, score_ori)
            res = {
                "pos_error": float(e.get("pos_error", 0.0)),
                "ori_error_rad": float(e.get("ori_error_rad", 0.0)),
                "pos_delta": np.asarray(e.get("pos_delta", np.zeros(3, dtype=float)), dtype=float),
                "ori_axis_angle": np.asarray(e.get("ori_axis_angle", np.zeros(3, dtype=float)), dtype=float),
                "out": bool(out),
                "score": float(max(0.0, score)),
            }
            result_per_hand.append(res)
            any_out = any_out or out
            overall_score = min(overall_score, res["score"]) if result_per_hand else res["score"]
        left = result_per_hand[0] if len(result_per_hand) > 0 else None
        right = result_per_hand[1] if len(result_per_hand) > 1 else None
        return {"any_out": bool(any_out), "left": left, "right": right, "score": float(overall_score if result_per_hand else 100.0)}
    
                
if __name__ == "__main__":
    r1pro = DoubleRm75Mink()
    r1pro.test_traj(duration=10.0, viewer=True, hand_amp=0.05, hand_freq=0.5)
    # r1pro.replay_txt_traj(viewer=True)
