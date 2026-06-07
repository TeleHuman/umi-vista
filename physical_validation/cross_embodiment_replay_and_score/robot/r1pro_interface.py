# ROS2 interaction functions, running in a dedicated thread.

import threading
import time
from typing import Any, Optional

import numpy as np


class _Ros2CommandFeedbackBridge:
    """Bind ROS2 subscriptions and publishers for both feedback and command paths."""

    def __init__(self, owner, node, joint_state_type, qos: int = 10):
        self._owner = owner
        self._node = node

        self._node.create_subscription(joint_state_type, "/hdas/feedback_torso", self.cb_torso, qos)
        self._node.create_subscription(joint_state_type, "/hdas/feedback_arm_left", self.cb_arm_l, qos)
        self._node.create_subscription(joint_state_type, "/hdas/feedback_arm_right", self.cb_arm_r, qos)
        self._node.create_subscription(joint_state_type, "/hdas/feedback_gripper_left", self.cb_grip_l, qos)
        self._node.create_subscription(joint_state_type, "/hdas/feedback_gripper_right", self.cb_grip_r, qos)

        try:
            pub_torso = self._node.create_publisher(joint_state_type, "/motion_target/target_joint_state_torso", qos)
        except Exception:
            pub_torso = None
        try:
            pub_arm_l = self._node.create_publisher(joint_state_type, "/motion_target/target_joint_state_arm_left", qos)
        except Exception:
            pub_arm_l = None
        try:
            pub_arm_r = self._node.create_publisher(joint_state_type, "/motion_target/target_joint_state_arm_right", qos)
        except Exception:
            pub_arm_r = None
        try:
            pub_gl = self._node.create_publisher(joint_state_type, "/motion_target/target_position_gripper_left", qos)
        except Exception:
            pub_gl = None
        try:
            pub_gr = self._node.create_publisher(joint_state_type, "/motion_target/target_position_gripper_right", qos)
        except Exception:
            pub_gr = None

        with self._owner._ros2_lock:
            self._owner._ros2_pubs["torso"] = pub_torso
            self._owner._ros2_pubs["arm_l"] = pub_arm_l
            self._owner._ros2_pubs["arm_r"] = pub_arm_r
            self._owner._ros2_pubs["grip_l"] = pub_gl
            self._owner._ros2_pubs["grip_r"] = pub_gr

    def cb_torso(self, msg):
        pos, vel = self._owner._extract_pos_vel(msg)
        with self._owner._ros2_lock:
            if pos.size >= 4:
                self._owner._ros2_fb["qpos"]["torso"] = pos[:4]
            if vel.size >= 4:
                self._owner._ros2_fb["qvel"]["torso"] = vel[:4]
            self._owner._ros2_warm_flags["torso"] = True

    def cb_arm_l(self, msg):
        pos, vel = self._owner._extract_pos_vel(msg)
        with self._owner._ros2_lock:
            if pos.size >= 7:
                self._owner._ros2_fb["qpos"]["left_arm"] = pos[:7]
            if vel.size >= 7:
                self._owner._ros2_fb["qvel"]["left_arm"] = vel[:7]
            self._owner._ros2_warm_flags["arm_l"] = True

    def cb_arm_r(self, msg):
        pos, vel = self._owner._extract_pos_vel(msg)
        with self._owner._ros2_lock:
            if pos.size >= 7:
                self._owner._ros2_fb["qpos"]["right_arm"] = pos[:7]
            if vel.size >= 7:
                self._owner._ros2_fb["qvel"]["right_arm"] = vel[:7]
            self._owner._ros2_warm_flags["arm_r"] = True

    @staticmethod
    def _grip_value_m(pos_array: np.ndarray) -> float:
        if pos_array.size == 0:
            return 0.0
        if pos_array.size == 1:
            return float(pos_array[0])
        return float((pos_array[0] + pos_array[1]) * 0.5)

    def cb_grip_l(self, msg):
        pos, _ = self._owner._extract_pos_vel(msg)
        with self._owner._ros2_lock:
            self._owner._ros2_fb["qpos"]["left_gripper"] = [self._grip_value_m(pos)]
            self._owner._ros2_warm_flags["grip_l"] = True

    def cb_grip_r(self, msg):
        pos, _ = self._owner._extract_pos_vel(msg)
        with self._owner._ros2_lock:
            self._owner._ros2_fb["qpos"]["right_gripper"] = [self._grip_value_m(pos)]
            self._owner._ros2_warm_flags["grip_r"] = True


class Dual_arm_controller:
    """r1pro dual-arm controller."""

    def __init__(self, high_freq_flag: bool = True, high_freq_hz: float = 100.0):
        self.publisher_thread = None
        self.last_error: Optional[Exception] = None

        self._ros2_lock = threading.Lock()
        self._ros2_node_started = False
        self._ros2_node_ref = None
        self._ros2_bridge_ref = None
        self._ros2_pubs: dict[str, Any] = {
            "torso": None,
            "arm_l": None,
            "arm_r": None,
            "grip_l": None,
            "grip_r": None,
        }
        self._ros2_thread = None
        self._ros2_stop_event = threading.Event()
        
        # r1pro has strong built-in smoothing, so external filtering is disabled for now to avoid excessive lag.
        self.flt_q: list[Optional[Flt]] = [None] * 18
        self.flt_enabled = False

        self._ros2_fb = {
            "qpos": {
                "torso": np.zeros(4, dtype=float),
                "left_arm": np.zeros(7, dtype=float),
                "right_arm": np.zeros(7, dtype=float),
                "left_gripper": [0.0],
                "right_gripper": [0.0],
            },
            "qvel": {
                "torso": np.zeros(4, dtype=float),
                "left_arm": np.zeros(7, dtype=float),
                "right_arm": np.zeros(7, dtype=float),
            },
        }
        self._ros2_warm_flags = {
            "torso": False,
            "arm_l": False,
            "arm_r": False,
            "grip_l": False,
            "grip_r": False,
        }

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

    @staticmethod
    def _extract_pos_vel(msg):
        if hasattr(msg, "position"):
            pos = np.asarray(list(msg.position or []), dtype=float)
            vel = np.asarray(list(getattr(msg, "velocity", []) or []), dtype=float)
            if vel.size == 0:
                vel = np.zeros_like(pos)
            return pos, vel
        if hasattr(msg, "data"):
            try:
                pos = np.asarray(list(msg.data or []), dtype=float)
            except Exception:
                pos = np.asarray([float(msg.data)], dtype=float)
            return pos, np.zeros_like(pos)
        return np.array([], dtype=float), np.array([], dtype=float)

    def _try_import_ros2(self):
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState  # type: ignore

            return rclpy, Node, JointState
        except Exception as e:
            raise RuntimeError(f"[ROS2] rclpy is not installed or the ROS2 environment is not initialized: {e!r}")

    def _start_ros2_node_once(self):
        with self._ros2_lock:
            if self._ros2_node_started:
                return
            rclpy, Node, JointState = self._try_import_ros2()

            def _spin():
                try:
                    if not rclpy.ok():
                        rclpy.init(args=None)
                    node = Node("teleop_feedback_node")
                    bridge = _Ros2CommandFeedbackBridge(self, node, JointState)
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
            for k in list(self._ros2_warm_flags.keys()):
                self._ros2_warm_flags[k] = False
            self._ros2_node_started = True

    def apply_real_qpos(self, q: np.ndarray):
        self._start_ros2_node_once()
        q = np.asarray(q, dtype=float).flatten()
        if q.size < 18:
            raise ValueError(f"apply_real_qpos requires at least 18 dimensions (without grippers) or 20 (with grippers), got {q.size}")
        if self.flt_enabled:
            if self.flt_q[0] is None:
                for i in range(18):
                    self.flt_q[i] = Flt(lowcut=10, fs=100, btype='low', initial_value=q[i])
            else:
                for i in range(18):
                    q[i] = self.flt_q[i].filter_step(q[i])
        torso = q[0:4] if q.size >= 4 else np.zeros(4, dtype=float)
        left = q[4:11] if q.size >= 11 else np.zeros(7, dtype=float)
        right = q[11:18] if q.size >= 18 else np.zeros(7, dtype=float)
        lgr = float(q[18]) if q.size >= 19 else 0.0
        rgr = float(q[19]) if q.size >= 20 else 0.0

        try:
            from sensor_msgs.msg import JointState  # type: ignore
        except Exception:
            JointState = None  # type: ignore
        if JointState is None:
            return

        with self._ros2_lock:
            pub_torso = self._ros2_pubs.get("torso")
            pub_l = self._ros2_pubs.get("arm_l")
            pub_r = self._ros2_pubs.get("arm_r")
            pub_gl = self._ros2_pubs.get("grip_l")
            pub_gr = self._ros2_pubs.get("grip_r")

        try:
            if pub_torso is not None:
                msg_t = JointState()
                msg_t.position = list(map(float, torso))
                pub_torso.publish(msg_t)
        except Exception as e:
            print(f"[ROS2] Failed to publish torso: {e!r}")
        try:
            if pub_l is not None:
                msg_l = JointState()
                msg_l.position = list(map(float, left))
                pub_l.publish(msg_l)
        except Exception as e:
            print(f"[ROS2] Failed to publish arm_left: {e!r}")
        try:
            if pub_r is not None:
                msg_r = JointState()
                msg_r.position = list(map(float, right))
                pub_r.publish(msg_r)
        except Exception as e:
            print(f"[ROS2] Failed to publish arm_right: {e!r}")
        try:
            if pub_gl is not None:
                msg_gl = JointState()
                msg_gl.position = [float(np.clip(lgr, 0, 100.0))]
                pub_gl.publish(msg_gl)
        except Exception as e:
            print(f"[ROS2] Failed to publish left_gripper: {e!r}")
        try:
            if pub_gr is not None:
                msg_gr = JointState()
                msg_gr.position = [float(np.clip(rgr, 0, 100.0))]
                pub_gr.publish(msg_gr)
        except Exception as e:
            print(f"[ROS2] Failed to publish right_gripper: {e!r}")

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

    def start_high_freq_thread(self, hz: float = 100.0):
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
            torso = np.array(self._ros2_fb["qpos"]["torso"], dtype=float)
            la = np.array(self._ros2_fb["qpos"]["left_arm"], dtype=float)
            ra = np.array(self._ros2_fb["qpos"]["right_arm"], dtype=float)
            lg = float(self._ros2_fb["qpos"]["left_gripper"][0]) if self._ros2_fb["qpos"]["left_gripper"] else 0.0
            rg = float(self._ros2_fb["qpos"]["right_gripper"][0]) if self._ros2_fb["qpos"]["right_gripper"] else 0.0
            vt = np.array(self._ros2_fb["qvel"]["torso"], dtype=float)
            vl = np.array(self._ros2_fb["qvel"]["left_arm"], dtype=float)
            vr = np.array(self._ros2_fb["qvel"]["right_arm"], dtype=float)
        return {
            "qpos": {
                "torso": torso,
                "left_arm": la,
                "right_arm": ra,
                "left_gripper": [lg],
                "right_gripper": [rg],
                "chassis": [],
            },
            "qvel": {
                "torso": vt,
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
