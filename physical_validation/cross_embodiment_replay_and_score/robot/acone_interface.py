# ROS2 interaction functions, running in a dedicated thread.

import os
import re
import threading
import time
from typing import Any, Optional
from utils.lowpass_filter import RealTimeButterworthFilter as Flt
import numpy as np


class _Ros2CommandFeedbackBridge:
    """Bind ROS2 subscriptions and publishers for both feedback and command paths."""

    def __init__(self, owner, node, robot_status_type, robot_cmd_type, qos: int = 10):
        self._owner = owner
        self._node = node
        self._robot_status_type = robot_status_type
        self._robot_cmd_type = robot_cmd_type

        self._node.create_subscription(robot_status_type, "/arm_slave_l_status", self.cb_arm_l, qos)
        self._node.create_subscription(robot_status_type, "/arm_slave_r_status", self.cb_arm_r, qos)

        try:
            pub_arm_l = self._node.create_publisher(robot_cmd_type, "/arm_master_l_status", qos)
        except Exception:
            pub_arm_l = None
        try:
            pub_arm_r = self._node.create_publisher(robot_cmd_type, "/arm_master_r_status", qos)
        except Exception:
            pub_arm_r = None
        with self._owner._ros2_lock:
            self._owner._ros2_pubs["arm_l"] = pub_arm_l
            self._owner._ros2_pubs["arm_r"] = pub_arm_r

    def cb_arm_l(self, msg):
        pos, vel = self._owner._extract_pos_vel(msg)
        with self._owner._ros2_lock:
            if pos.size >= 7:
                self._owner._ros2_fb["qpos"]["left_arm"] = pos[:6]
                self._owner._ros2_fb["qpos"]["left_gripper"] = [float(pos[6]) / (-3.4) * 100.0]
            if vel.size >= 7:
                self._owner._ros2_fb["qvel"]["left_arm"] = vel[:6]
            self._owner._ros2_warm_flags["arm_l"] = True

    def cb_arm_r(self, msg):
        pos, vel = self._owner._extract_pos_vel(msg)
        with self._owner._ros2_lock:
            if pos.size >= 7:
                self._owner._ros2_fb["qpos"]["right_arm"] = pos[:6]
                self._owner._ros2_fb["qpos"]["right_gripper"] = [float(pos[6]) / (-3.4) * 100.0]
            if vel.size >= 7:
                self._owner._ros2_fb["qvel"]["right_arm"] = vel[:6]
            self._owner._ros2_warm_flags["arm_r"] = True


class Dual_arm_controller:
    """acone dual-arm controller."""

    def __init__(self, high_freq_flag: bool = True, high_freq_hz: float = 200.0):
        self.publisher_thread = None
        self.last_error: Optional[Exception] = None

        self._ros2_lock = threading.Lock()
        self._ros2_node_started = False
        self._ros2_node_ref = None
        self._ros2_bridge_ref = None
        self._ros2_pubs: dict[str, Any] = {
            "arm_l": None,
            "arm_r": None,
            "grip_l": None,
            "grip_r": None,
        }
        self._ros2_thread = None
        self._ros2_stop_event = threading.Event()
        self._ros2_robotstatus = None
        self._ros2_robotcmd = None
        self.flt_q: list[Optional[Flt]] = [None] * 12
        self.flt_enabled = True

        self._ros2_fb = {
            "qpos": {
                "left_arm": np.zeros(6, dtype=float),
                "right_arm": np.zeros(6, dtype=float),
                "left_gripper": [0.0],
                "right_gripper": [0.0],
            },
            "qvel": {
                "left_arm": np.zeros(6, dtype=float),
                "right_arm": np.zeros(6, dtype=float),
            },
        }
        self._ros2_warm_flags = {"arm_l": False, "arm_r": False}

        self._pub_lock = threading.Lock()
        self._latest_cmd: Optional[np.ndarray] = None
        self._prev_cmd: Optional[np.ndarray] = None
        self._latest_slope: Optional[np.ndarray] = None
        self._prev_slope: Optional[np.ndarray] = None
        self._prev_cmd_time: float = 0.0
        self._curr_cmd_time: float = 0.0
        self._pub_stop_event = threading.Event()

        try:
            self._start_ros2_node_once()
        except Exception as e:
            self.last_error = e
            return

        if high_freq_flag:
            try:
                self.publisher_thread = self.start_high_freq_thread(hz=high_freq_hz)
            except Exception as e:
                self.last_error = e

    def _try_import_ros2(self):
        try:
            import rclpy
            from rclpy.node import Node

            RobotStatus = None
            RobotCmd = None
            try:
                from arx5_arm_msg.msg import RobotStatus as _RS, RobotCmd as _RC  # type: ignore

                RobotStatus, RobotCmd = _RS, _RC
            except Exception:
                pkg = os.environ.get("ACONE_MSG_PKG") or os.environ.get("X5_MSG_PKG")
                if pkg:
                    mod = __import__(pkg + ".msg", fromlist=["RobotStatus", "RobotCmd"])  # type: ignore
                    RobotStatus = getattr(mod, "RobotStatus", None)
                    RobotCmd = getattr(mod, "RobotCmd", None)
            return rclpy, Node, RobotStatus, RobotCmd
        except Exception as e:
            raise RuntimeError(f"[ROS2] rclpy is not installed or the ROS2 environment is not initialized: {e!r}")

    @staticmethod
    def _extract_pos_vel(msg):
        def _to_vec(val):
            try:
                if isinstance(val, np.ndarray):
                    return np.array(val, dtype=float).reshape(-1)
                if isinstance(val, (list, tuple)):
                    return np.array(val, dtype=float).reshape(-1)
                if isinstance(val, str):
                    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
                    return np.array([float(x) for x in nums], dtype=float).reshape(-1)
                if hasattr(val, "__iter__") and not isinstance(val, (bytes, bytearray)):
                    return np.array(list(val), dtype=float).reshape(-1)
                return np.array([float(val)], dtype=float)
            except Exception:
                return np.array([], dtype=float)

        if hasattr(msg, "joint_pos"):
            pos = _to_vec(getattr(msg, "joint_pos"))
            vel = _to_vec(getattr(msg, "joint_vel")) if hasattr(msg, "joint_vel") else np.zeros_like(pos)
            return pos, vel
        return np.array([], dtype=float), np.array([], dtype=float)

    def _start_ros2_node_once(self):
        with self._ros2_lock:
            if self._ros2_node_started:
                return
            rclpy, Node, RobotStatus, RobotCmd = self._try_import_ros2()
            if (RobotStatus is None) or (RobotCmd is None):
                raise RuntimeError("[ROS2] RobotStatus/RobotCmd was not found.")
            self._ros2_robotstatus = RobotStatus
            self._ros2_robotcmd = RobotCmd

            def _spin():
                try:
                    if not rclpy.ok():
                        rclpy.init(args=None)
                    node = Node("teleop_feedback_node")
                    bridge = _Ros2CommandFeedbackBridge(self, node, RobotStatus, RobotCmd)
                    executor = rclpy.executors.SingleThreadedExecutor()
                    executor.add_node(node)
                    with self._ros2_lock:
                        self._ros2_node_ref = node
                        self._ros2_bridge_ref = bridge
                    while not self._ros2_stop_event.is_set():
                        executor.spin_once(timeout_sec=0.05)
                    try:
                        executor.remove_node(node)
                    except Exception:
                        pass
                    node.destroy_node()
                    if rclpy.ok():
                        rclpy.shutdown()
                except Exception as e:
                    if e.__class__.__name__ != "ExternalShutdownException":
                        print(f"[ROS2] spin error: {e!r}")

            self._ros2_thread = threading.Thread(target=_spin, daemon=True)
            self._ros2_thread.start()
            self._ros2_warm_flags = {"arm_l": False, "arm_r": False}
            self._ros2_node_started = True

    def apply_real_qpos(self, q: np.ndarray):
        self._start_ros2_node_once()
        q = np.asarray(q, dtype=float).flatten()
        if q.size < 14:
            raise ValueError(f"apply_real_qpos requires at least 14 dimensions, got {q.size}")
        if self.flt_enabled:
            if self.flt_q[0] is None:
                for i in range(12):
                    self.flt_q[i] = Flt(lowcut=10, fs=100, btype='low', initial_value=q[i])
            else:
                for i in range(12):
                    q[i] = self.flt_q[i].filter_step(q[i])
        left = q[0:6]
        right = q[6:12]
        lgr = float(np.clip(q[12], 0, 100.0)) / 100.0 * (-3.4)
        rgr = float(np.clip(q[13], 0, 100.0)) / 100.0 * (-3.4)

        with self._ros2_lock:
            pub_l = self._ros2_pubs.get("arm_l")
            pub_r = self._ros2_pubs.get("arm_r")

        if self._ros2_robotcmd is None:
            raise RuntimeError("[ROS2] RobotCmd is not ready")

        try:
            if pub_l is not None:
                rc_l = self._ros2_robotcmd()
                if hasattr(rc_l, "joint_pos"):
                    rc_l.joint_pos = list(map(float, left))
                if hasattr(rc_l, "gripper"):
                    rc_l.gripper = float(lgr)
                if hasattr(rc_l, "mode"):
                    rc_l.mode = 5
                if hasattr(rc_l, "end_pos"):
                    rc_l.end_pos = [0.0] * 6
                pub_l.publish(rc_l)
        except Exception as e:
            print(f"[ROS2] Failed to publish arm_left: {e!r}")

        try:
            if pub_r is not None:
                rc_r = self._ros2_robotcmd()
                if hasattr(rc_r, "joint_pos"):
                    rc_r.joint_pos = list(map(float, right))
                if hasattr(rc_r, "gripper"):
                    rc_r.gripper = float(rgr)
                if hasattr(rc_r, "mode"):
                    rc_r.mode = 5
                if hasattr(rc_r, "end_pos"):
                    rc_r.end_pos = [0.0] * 6
                pub_r.publish(rc_r)
        except Exception as e:
            print(f"[ROS2] Failed to publish arm_right: {e!r}")

    def set_latest_cmd(self, cmd: np.ndarray, low_freq: float):
        now = time.time()
        with self._pub_lock:
            cmdf = np.array(cmd, dtype=float).flatten()
            if self._latest_cmd is not None and self._latest_slope is not None:
                self._prev_slope = self._latest_slope.copy()
                self._prev_cmd = self._latest_cmd.copy()
            else:
                self._prev_cmd = cmdf.copy()
                self._prev_slope = np.zeros_like(cmdf)
            self._latest_slope = (cmdf - self._prev_cmd) / max(1e-6, 1.0 / low_freq)
            self._latest_cmd = cmdf
            self._prev_cmd_time = now
            self._curr_cmd_time = now + max(1e-6, 1.0 / low_freq)

    def _publisher_loop(self, target_hz: float = 200.0):
        dt = 1.0 / max(1.0, float(target_hz))
        next_deadline = time.perf_counter() + dt
        while not self._pub_stop_event.is_set():
            with self._pub_lock:
                curr = None if self._latest_cmd is None else self._latest_cmd.copy()
                prev = None if self._prev_cmd is None else self._prev_cmd.copy()
                t0, t1 = self._prev_cmd_time, self._curr_cmd_time
            if curr is not None and prev is not None:
                now_t = time.time()
                dur = max(1e-6, t1 - t0)
                s = (now_t - t0) / dur
                if s <= 0:
                    cmd_out = prev
                elif s >= 1:
                    cmd_out = curr
                else:
                    h00 = 2 * s**3 - 3 * s**2 + 1
                    h10 = s**3 - 2 * s**2 + s
                    h01 = -2 * s**3 + 3 * s**2
                    h11 = s**3 - s**2
                    m0 = self._prev_slope.copy() if self._prev_slope is not None else np.zeros_like(curr)
                    m1 = self._latest_slope.copy() if self._latest_slope is not None else np.zeros_like(curr)
                    cmd_out = h00 * prev + h10 * (m0 * dur) + h01 * curr + h11 * (m1 * dur)
                try:
                    self.apply_real_qpos(cmd_out)
                except Exception as e:
                    print(f"[ROS2] Publisher thread error: {e!r}")
            elif curr is not None:
                try:
                    self.apply_real_qpos(curr)
                except Exception as e:
                    print(f"[ROS2] Publisher thread error: {e!r}")
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            next_deadline += dt

    def start_high_freq_thread(self, hz: float = 200.0):
        self._pub_stop_event.clear()
        self.publisher_thread = threading.Thread(target=self._publisher_loop, kwargs={"target_hz": hz}, daemon=True)
        self.publisher_thread.start()
        return self.publisher_thread

    def stop_high_freq_thread(self, timeout: float = 1.0):
        self._pub_stop_event.set()
        if self.publisher_thread is not None and self.publisher_thread.is_alive():
            self.publisher_thread.join(timeout=timeout)

    def stop_ros2_node(self, timeout: float = 1.0):
        with self._ros2_lock:
            if not self._ros2_node_started:
                return False
            self._ros2_stop_event.set()
        if self._ros2_thread is not None:
            self._ros2_thread.join(timeout=timeout)
        with self._ros2_lock:
            self._ros2_node_started = False
            self._ros2_node_ref = None
            self._ros2_bridge_ref = None
            self._ros2_thread = None
            self._ros2_stop_event.clear()
            for k in list(self._ros2_pubs.keys()):
                self._ros2_pubs[k] = None
        return True

    def ros2_is_warmed(self) -> bool:
        with self._ros2_lock:
            return all(self._ros2_warm_flags.values())

    def get_real_feedback(self):
        self._start_ros2_node_once()
        with self._ros2_lock:
            la = np.array(self._ros2_fb["qpos"]["left_arm"], dtype=float)
            ra = np.array(self._ros2_fb["qpos"]["right_arm"], dtype=float)
            lg = float(self._ros2_fb["qpos"]["left_gripper"][0]) if self._ros2_fb["qpos"]["left_gripper"] else 0.0
            rg = float(self._ros2_fb["qpos"]["right_gripper"][0]) if self._ros2_fb["qpos"]["right_gripper"] else 0.0
            vl = np.array(self._ros2_fb["qvel"]["left_arm"], dtype=float)
            vr = np.array(self._ros2_fb["qvel"]["right_arm"], dtype=float)
        return {
            "qpos": {
                "left_arm": la,
                "right_arm": ra,
                "left_gripper": [lg],
                "right_gripper": [rg],
            },
            "qvel": {
                "left_arm": vl,
                "right_arm": vr,
            },
            "fce": {
                "left_arm": [0.0] * 6,
                "right_arm": [0.0] * 6,
            },
            "warmed": self.ros2_is_warmed(),
        }

    def cleanup(self, timeout: float = 1.0):
        try:
            self.stop_high_freq_thread(timeout=timeout)
        except Exception:
            pass
        try:
            self.stop_ros2_node(timeout=timeout)
        except Exception:
            pass
