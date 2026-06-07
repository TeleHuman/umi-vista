import numpy as np
import atexit
import sys
import time
import threading
from Robotic_Arm.rm_robot_interface import *
from utils.robotiq_interface import SafeGripperController
from typing import Any, Optional
from utils.lowpass_filter import RealTimeButterworthFilter as Flt

LEFT_ARM_IP = "192.168.1.18"
RIGHT_ARM_IP = "192.168.1.19"
ARM_PORT = 8080
LEFT_GRIPPER_PORT = "/dev/ttyUSB1"
RIGHT_GRIPPER_PORT = "/dev/ttyUSB0"
LEFT_GRIPPER_SERIAL = "DAK2KK0V"
RIGHT_GRIPPER_SERIAL = "DAK2KORB"
CANFD_FOLLOW = False
CANFD_TRAJECTORY_MODE = 2
CANFD_RADIO = 1000
UDP_TARGET_IP = "192.168.1.200"
UDP_TARGET_PORT = 8089

def _force_to_np(force_value, n: int = 6) -> np.ndarray:
    """Convert vendor ctypes/sequence force array to a stable numpy array."""
    try:
        arr = np.ctypeslib.as_array(force_value, shape=(n,))
        return np.asarray(arr, dtype=np.float32).copy()
    except Exception:
        try:
            arr = np.asarray(list(force_value), dtype=np.float32).reshape(-1)
            if arr.size >= n:
                return np.asarray(arr[:n], dtype=np.float32).copy()
            out = np.zeros(n, dtype=np.float32)
            out[: arr.size] = arr
            return out
        except Exception:
            return np.zeros(n, dtype=np.float32)

class Dual_arm_controller():
    def __init__(self):
        self._cleaned = False
        self._fb_lock = threading.Lock()
        self._have_fb = False
        # arm
        self.left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        # NOTE: do not change IPs; these must match your physical wiring.
        right_arm_handle = self.left_arm.rm_create_robot_arm(LEFT_ARM_IP, ARM_PORT)
        print(f"Connected to robot arm with ID: {right_arm_handle.id}")
        left_arm_handle = self.right_arm.rm_create_robot_arm(RIGHT_ARM_IP, ARM_PORT)
        print(f"Connected to robot arm with ID: {left_arm_handle.id}")
        self.left_gripper = SafeGripperController(portname=LEFT_GRIPPER_PORT, serial_number=LEFT_GRIPPER_SERIAL)
        print("Left gripper connected successfully!")
        self.right_gripper = SafeGripperController(portname=RIGHT_GRIPPER_PORT, serial_number=RIGHT_GRIPPER_SERIAL)
        print("Right gripper connected successfully!")

        self.left_current_joint_rad = np.zeros(7, dtype=float)
        self.right_current_joint_rad = np.zeros(7, dtype=float)
        self.left_joint_speed_rad = np.zeros(7, dtype=float)
        self.right_joint_speed_rad = np.zeros(7, dtype=float)
        self.left_fce_data = np.zeros(6, dtype=float)
        self.right_fce_data = np.zeros(6, dtype=float)

        # Initialize feedback once via polling to avoid upstream logic treating
        # the default zeros as real joint positions (which can cause a "go to zero" first).
        self._init_feedback_from_polling()

        # set realtime push config
        self.udp_enabled=True
        if(self.udp_enabled):
            custom=rm_udp_custom_config_t()
            custom.joint_speed=1
            custom.lift_state=0
            custom.expand_state=0
            config=rm_realtime_push_config_t(1,True,UDP_TARGET_PORT,1,UDP_TARGET_IP,custom)
            tag = self.left_arm.rm_set_realtime_push(config)
            if tag != 0:
                print(f"[UDP] left rm_set_realtime_push failed: tag={tag}")
            # IMPORTANT: SDK registers a *global* realtime callback (no handle). only registering once
            self._arm_state_callback = rm_realtime_arm_state_callback_ptr(self._on_arm_state)
            self.left_arm.rm_realtime_arm_state_call_back(self._arm_state_callback)

        # interpolation/high freq thread state (500Hz)
        self._high_freq_lock = threading.Lock()
        self._latest_cmd = None
        self._prev_cmd = None
        self._prev_slope = None
        self._latest_slope = None
        self._prev_cmd_time = 0.0
        self._curr_cmd_time = 0.0
        self._high_freq_stop_event = threading.Event()
        self._high_freq_thread = threading.Thread(
            target=self._high_freq_loop, 
            kwargs={"target_hz": 500.0}, 
            daemon=True)
        self._high_freq_thread.start()

        # gripper rate limit (1Hz) - independent of upstream/high-freq arm commands
        self._gripper_min_period_s = 0.2
        self._gripper_last_send_t = 0.0  # time.perf_counter()
        self.flt_q: list[Optional[Flt]] = [None] * 14
        self.flt_enabled = False

        atexit.register(self.cleanup)

    def _decode_arm_ip(self, arm_ip_field) -> str:
        try:
            if isinstance(arm_ip_field, (bytes, bytearray)):
                raw = bytes(arm_ip_field)
            else:
                raw = bytes(arm_ip_field)
            ip = raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
            return ip
        except Exception:
            return ""

    def _on_arm_state(self, arm_state: rm_realtime_arm_joint_state_t):
        """Single SDK callback; dispatch by arm_state.arm_ip."""
        ip = self._decode_arm_ip(getattr(arm_state, "arm_ip", b""))
        joint = arm_state.joint_status.joint_position
        speed = arm_state.joint_status.joint_speed
        q_rad = np.array(np.deg2rad(joint), dtype=float)
        qd_rad = np.array(np.deg2rad(speed), dtype=float)
        fce = _force_to_np(arm_state.force_sensor.zero_force)

        with self._fb_lock:
            if ip == LEFT_ARM_IP:
                self.left_current_joint_rad = q_rad
                self.left_joint_speed_rad = qd_rad
                self.left_fce_data = fce
            elif ip == RIGHT_ARM_IP:
                self.right_current_joint_rad = q_rad
                self.right_joint_speed_rad = qd_rad
                self.right_fce_data = fce
            else:
                # Unknown sender: do not clobber state; still mark feedback arrived.
                pass
            self._have_fb = True

            # if ip == self._left_arm_ip:
            #     print(f"Left Arm State - Joint (rad): {q_rad}, Force: {fce}")
            # elif ip == self._right_arm_ip:
            #     print(f"Right Arm State - Joint (rad): {q_rad}, Force: {fce}")
            # else:
            #     print(f"Arm State ({ip}) - Joint (rad): {q_rad}, Force: {fce}")

    def _init_feedback_from_polling(self):
        try:
            _, left_current_joint = self.left_arm.rm_get_current_arm_state()
            _, right_current_joint = self.right_arm.rm_get_current_arm_state()
            left_joint = left_current_joint.get("joint", None) if isinstance(left_current_joint, dict) else None
            right_joint = right_current_joint.get("joint", None) if isinstance(right_current_joint, dict) else None
            if left_joint is None or right_joint is None:
                return
            left_rad = np.array(np.deg2rad(left_joint), dtype=float)
            right_rad = np.array(np.deg2rad(right_joint), dtype=float)
            if left_rad.size >= 7 and right_rad.size >= 7:
                with self._fb_lock:
                    self.left_current_joint_rad = left_rad.reshape(-1)[:7]
                    self.right_current_joint_rad = right_rad.reshape(-1)[:7]
                    self._have_fb = True
        except Exception:
            # Leave defaults; caller can rely on warmed flag.
            pass
        
    def get_real_feedback(self):
        # Best-effort ensure we have at least one valid feedback sample.
        if not self._have_fb:
            self._init_feedback_from_polling()
        if not self.udp_enabled:
            flag, left_current_joint = self.left_arm.rm_get_current_arm_state()
            self.left_current_joint_rad = np.array(np.deg2rad(left_current_joint["joint"]), dtype=float)
            self.left_joint_speed_rad = np.zeros(7, dtype=float) # no speed info from polling
            flag, right_current_joint = self.right_arm.rm_get_current_arm_state()
            self.right_current_joint_rad = np.array(np.deg2rad(right_current_joint["joint"]), dtype=float)
            self.right_joint_speed_rad = np.zeros(7, dtype=float) # no speed info from polling
            flag, left_fce_data = self.left_arm.rm_get_force_data()
            self.left_fce_data = left_fce_data["work_zero_force_data"] if left_fce_data is not None else None
            flag, right_fce_data = self.right_arm.rm_get_force_data()
            self.right_fce_data = right_fce_data["work_zero_force_data"] if right_fce_data is not None else None
        left_gripper_pos = self.left_gripper.get_pos()/0.4  # convert 0-40 to 0-100
        right_gripper_pos = self.right_gripper.get_pos()/0.4  # convert 0-40 to 0-100

        with self._fb_lock:
            left_arm_qpos = np.array(self.left_current_joint_rad, dtype=float).reshape(-1)[:7]
            right_arm_qpos = np.array(self.right_current_joint_rad, dtype=float).reshape(-1)[:7]
            left_fce = self.left_fce_data
            right_fce = self.right_fce_data
            warmed = bool(self._have_fb)

        result = {
            'warmed': warmed,
            'qpos':{
                'left_arm': left_arm_qpos,
                'right_arm': right_arm_qpos,
                'left_gripper': np.array([left_gripper_pos], dtype=np.float32),
                'right_gripper': np.array([right_gripper_pos], dtype=np.float32)
            },
            'qvel':{
                'left_arm': self.left_joint_speed_rad if self.left_joint_speed_rad is not None else np.zeros(7, dtype=np.float32),
                'right_arm': self.right_joint_speed_rad if self.right_joint_speed_rad is not None else np.zeros(7, dtype=np.float32)
            },
            'fce': {
                'left_arm': left_fce if left_fce is not None else np.zeros(6, dtype=np.float32),
                'right_arm': right_fce if right_fce is not None else np.zeros(6, dtype=np.float32),
            }
        }
        return result
    
    def apply_real_qpos(self,cmd:np.ndarray):
        q = np.asarray(cmd, dtype=float).flatten()
        if q.size < 14:
            raise ValueError(f"apply_real_qpos requires at least 14 dimensions, got {q.size}")
        if self.flt_enabled:
            if self.flt_q[0] is None:
                for i in range(12):
                    self.flt_q[i] = Flt(lowcut=10, fs=100, btype='low', initial_value=q[i])
            else:
                for i in range(12):
                    q[i] = self.flt_q[i].filter_step(q[i])
        left_qpos = q[:7] #rad
        right_qpos = q[7:14]
        left_gripper_qpos=np.clip(q[14],11,100)*0.4 # convert 0-100 to 0-40
        right_gripper_qpos=np.clip(q[15],11,100)*0.4 # convert 0-100 to 0-40
        self.left_arm.rm_movej_canfd(
            list(np.rad2deg(left_qpos)),
            follow=CANFD_FOLLOW,
            trajectory_mode=CANFD_TRAJECTORY_MODE,
            radio=CANFD_RADIO,
        )
        self.right_arm.rm_movej_canfd(
            list(np.rad2deg(right_qpos)),
            follow=CANFD_FOLLOW,
            trajectory_mode=CANFD_TRAJECTORY_MODE,
            radio=CANFD_RADIO,
        )
        self.left_gripper.move(pos_mm=left_gripper_qpos, speed=255, force=2) # non-blocking
        self.right_gripper.move(pos_mm=right_gripper_qpos, speed=255, force=2) # non-blocking

        # Ensure gripperstep runs at 1Hz regardless of upstream command rate
        now_pc = time.perf_counter()
        should_send = False
        if (now_pc - self._gripper_last_send_t) >= self._gripper_min_period_s:
            self._gripper_last_send_t = now_pc
            should_send = True
        if should_send:
            self.left_gripper.move(pos_mm=left_gripper_qpos, speed=255, force=2) # non-blocking
            self.right_gripper.move(pos_mm=right_gripper_qpos, speed=255, force=2) # non-blocking
        
    def set_latest_cmd(self,cmd: np.ndarray, low_freq: float):
        now = time.time()
        with self._high_freq_lock:
            cmdf = np.array(cmd, dtype=float).flatten()
            # rotate segment: prev <- curr, curr <- new
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

    def _high_freq_loop(self, target_hz: float = 500.0):
        dt = 1.0 / max(1.0, float(target_hz))
        next_deadline = time.perf_counter() + dt
        while not self._high_freq_stop_event.is_set():
            with self._high_freq_lock:
                curr = None if self._latest_cmd is None else self._latest_cmd.copy()
                prev = None if self._prev_cmd is None else self._prev_cmd.copy()
                t0 = self._prev_cmd_time
                t1 = self._curr_cmd_time
            if curr is not None and prev is not None:
                now_t = time.time()
                dur = max(1e-6, t1 - t0)
                s = (now_t - t0) / dur
                if s <= 0:
                    cmd_out = prev
                elif s >= 1:
                    cmd_out = curr
                else:
                    h00 = 2*s**3 - 3*s**2 + 1
                    h10 = s**3 - 2*s**2 + s
                    h01 = -2*s**3 + 3*s**2
                    h11 = s**3 - s**2
                    m0 = self._prev_slope.copy() if self._prev_slope is not None else np.zeros_like(curr)
                    m1 = self._latest_slope.copy() if self._latest_slope is not None else np.zeros_like(curr)
                    cmd_out = h00*prev + h10*(m0*dur) + h01*curr + h11*(m1*dur)
                try:
                    self.apply_real_qpos(cmd_out)
                except Exception as e:
                    print(f"[PY] Publisher thread error: {e!r}")
            elif curr is not None:
                try:
                    self.apply_real_qpos(curr)
                except Exception as e:
                    print(f"[PY] Publisher thread error: {e!r}")
            now_pc = time.perf_counter()
            sleep_sec = next_deadline - now_pc
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            next_deadline += dt

    def _stop_high_freq_thread(self):
        self._high_freq_stop_event.set()
        try:
            if self._high_freq_thread.is_alive():
                self._high_freq_thread.join(timeout=1.0)
        except Exception:
            pass

    def cleanup(self):
        if getattr(self, "_cleaned", False):
            return
        self._cleaned = True
        print("Program terminated cleanly.")
        self._stop_high_freq_thread()
        try:
            if hasattr(self, "left_gripper") and self.left_gripper is not None:
                self.left_gripper.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "right_gripper") and self.right_gripper is not None:
                self.right_gripper.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "left_arm") and self.left_arm is not None:
                self.left_arm.rm_delete_robot_arm()
        except Exception:
            pass
        try:
            if hasattr(self, "right_arm") and self.right_arm is not None:
                self.right_arm.rm_delete_robot_arm()
        except Exception:
            pass