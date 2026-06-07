import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any
import time
import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
import h5py
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle


def make_abs_state(obs):
    """
    [x, y, z, qx, qy, qz, qw, abs_gripper_qpos]
    """
    ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)

    ee_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    ee_quat = ee_quat / (np.linalg.norm(ee_quat) + 1e-8)

    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    gripper_abs = np.array([np.mean(np.abs(gripper_qpos))], dtype=np.float32)

    return np.concatenate([ee_pos, ee_quat, gripper_abs], axis=0).astype(np.float32)


def abs_quat_action_to_libero_action(action_8d):
    """
    Policy action:
        [x, y, z, qx, qy, qz, qw, gripper]

    LIBERO / robosuite OSC_POSE action:
        [x, y, z, ax, ay, az, gripper]
    """
    action_8d = np.asarray(action_8d, dtype=np.float32)

    pos = action_8d[:3]

    quat = action_8d[3:7]
    quat = quat / (np.linalg.norm(quat) + 1e-8)

    axis_angle = quat2axisangle(quat).astype(np.float32)

    # The gripper is already -1 or +1, so keep it unchanged
    gripper = action_8d[7:8]

    return np.concatenate([pos, axis_angle, gripper], axis=0).astype(np.float32)


def apply_fisheye_sim_rectilinear_to_fisheye(img, fov_deg=150, gamma=0.6):
    """
    Simulate a fisheye image from a regular pinhole image
    Input:
        img: HxWxC
        fov_deg: Fisheye field of view, commonly 120 to 180
    Output:
        fisheye_img
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    fov = np.deg2rad(fov_deg)
    theta_max = fov / 2.0

    # Maximum output fisheye radius
    R = min(w, h) / 2.0

    # Output fisheye image grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = u - cx
    y = v - cy
    r = np.sqrt(x**2 + y**2)

    # fisheye: radius r corresponds to angle theta
    theta = r / R * theta_max * gamma

    # Avoid division by zero
    r_safe = np.where(r == 0, 1, r)

    # Unit direction
    x_unit = x / r_safe
    y_unit = y / r_safe

    # Map back to the pinhole model:
    # rectilinear projection uses r_pinhole = f * tan(theta)
    f_p = R / np.tan(theta_max)
    r_p = f_p * np.tan(theta)

    map_x = cx + x_unit * r_p
    map_y = cy + y_unit * r_p

    # Mark areas outside the circle as invalid
    valid = r <= R
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    fisheye = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    fisheye[~valid] = 0
    return fisheye

def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _get_suite(name: str) -> benchmark.Benchmark:
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    if task_ids is None:
        return list(range(total_tasks))

    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task_suite.tasks[i].problem_folder
        / task_suite.tasks[i].init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)
    return init_states


def get_libero_dummy_action():
    """
    No-op action for the delta controller:
        [0, 0, 0, 0, 0, 0, -1]
    Do not use this directly with the absolute controller.
    """
    return [0, 0, 0, 0, 0, 0, -1]


OBS_STATE_DIM = 8
ACTION_DIM = 8

AGENT_POS_LOW = -1000.0
AGENT_POS_HIGH = 1000.0
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        hdf5_path: str | None = None,
        hdf5_demo_name: str = "demo_0",
        hdf5_replay: bool = False,
        hdf5_compare_policy: bool = True,
        hdf5_debug_dir: str = "/tmp/libero_hdf5_action_debug",
        # Use HDF5 images instead of the env wrist image
        hdf5_use_obs_image: bool = False,
        hdf5_obs_image_key: str = "obs/eye_in_hand_rgb",
    ):
        super().__init__()

        self.task_suite = task_suite
        self.task_id = task_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = _parse_camera_names(camera_name)

        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "unused_image",
                "robot0_eye_in_hand_image": "wrist_image",
            }

        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index

        self._init_states = get_task_init_states(task_suite, self.task_id) if self.init_states else None
        self._init_state_id = self.episode_index

        self._env = self._make_envs_task(task_suite, self.task_id)

####
        # self.hdf5_path = hdf5_path
        # self.hdf5_demo_name = hdf5_demo_name
        # self.hdf5_replay = hdf5_replay
        # self.hdf5_compare_policy = hdf5_compare_policy

        # self._hdf5_init_state = None
        # self._hdf5_actions = None
        # self._hdf5_step = 0
        # self.hdf5_use_obs_image = hdf5_use_obs_image
        # self.hdf5_obs_image_key = hdf5_obs_image_key
        # self._hdf5_wrist_images = None

        # if self.hdf5_path is not None:
        #     with h5py.File(self.hdf5_path, "r") as f:
        #         demo = f["data"][self.hdf5_demo_name]

        #         if "states" not in demo:
        #             raise KeyError(f"Missing data/{self.hdf5_demo_name}/states in {self.hdf5_path}")

        #         if "abs_actions" not in demo:
        #             raise KeyError(f"Missing data/{self.hdf5_demo_name}/abs_actions in {self.hdf5_path}")

        #         self._hdf5_init_state = demo["states"][0]
        #         # import ipdb; ipdb.set_trace()
        #         self._hdf5_actions = demo["abs_actions"][()].astype(np.float32)

        #         if self.hdf5_use_obs_image:
        #             if self.hdf5_obs_image_key not in demo:
        #                 available = []
        #                 demo.visit(lambda name: available.append(name))
        #                 raise KeyError(
        #                     f"Missing data/{self.hdf5_demo_name}/{self.hdf5_obs_image_key} "
        #                     f"in {self.hdf5_path}. Available first 50: {available[:50]}"
        #                 )

        #             self._hdf5_wrist_images = demo[self.hdf5_obs_image_key][()]
        #             print(
        #                 f"[hdf5-debug] loaded wrist images | "
        #                 f"key={self.hdf5_obs_image_key} | "
        #                 f"shape={self._hdf5_wrist_images.shape} | "
        #                 f"dtype={self._hdf5_wrist_images.dtype}"
        #             )

        #     print(
        #         f"[hdf5-debug] loaded {self.hdf5_path} | "
        #         f"demo={self.hdf5_demo_name} | "
        #         f"actions={self._hdf5_actions.shape}"
        #     )
        # self.hdf5_debug_dir = hdf5_debug_dir

        # self._debug_policy_actions = []
        # self._debug_hdf5_actions = []
        # self._debug_action_steps = []
        # self._debug_rewards = []
        # self._action_debug_saved = False
####

        default_steps = 500
        self._max_episode_steps = TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)

        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )

        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=AGENT_POS_LOW,
                        high=AGENT_POS_HIGH,
                        shape=(OBS_STATE_DIM,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    def render(self):
        raw_obs = self._env.env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["wrist_image"]
        return image

    def _make_envs_task(self, task_suite: Any, task_id: int = 0):
        task = task_suite.get_task(task_id)

        self.task = task.name
        self.task_description = task.language

        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 1024,
            "camera_widths": 1024,
        }

        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env


    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        images = {}

        for camera_name in self.camera_name:
            if camera_name == "robot0_eye_in_hand_image":
                # hdf5_img = self._get_hdf5_wrist_image()
                # import ipdb;ipdb.set_trace()
                # if hdf5_img is not None:
                #     # Use the training HDF5 wrist image as observation.images.wrist_image
                #     eye_img = hdf5_img
                # else:
                image = raw_obs[camera_name]
                image = image[::-1, ::-1]
                # import ipdb;ipdb.set_trace()
                eye_img = apply_fisheye_sim_rectilinear_to_fisheye(
                    image,
                    fov_deg=150,
                    gamma=0.6,
                )
                # import ipdb;ipdb.set_trace()
                eye_img = cv2.resize(
                    eye_img,
                    (256, 256),
                    interpolation=cv2.INTER_AREA,
                )

                images[self.camera_name_mapping[camera_name]] = eye_img

            else:
                image = raw_obs[camera_name]
                image = image[::-1, ::-1]
                image = cv2.resize(
                    image,
                    (256, 256),
                    interpolation=cv2.INTER_AREA,
                )

                images[self.camera_name_mapping[camera_name]] = image

        # Match regenerate make_abs_state exactly：
        # [x, y, z, qx, qy, qz, qw, abs_gripper_qpos]
        agent_pos = make_abs_state(raw_obs)
        # import ipdb;ipdb.set_trace()

        if self.obs_type == "pixels":
            return {"pixels": images.copy()}

        if self.obs_type == "pixels_agent_pos":
            return {
                "pixels": images.copy(),
                "agent_pos": agent_pos,
            }

        raise NotImplementedError(
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv."
        )
    
    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._env.seed(seed)
        raw_obs = self._env.reset()
        if self.init_states and self._init_states is not None:
            raw_obs = self._env.set_init_state(self._init_states[self._init_state_id])
###
        # if self._hdf5_init_state is not None:
        #     self._env.set_init_state(self._hdf5_init_state)
        #     self._env.sim.forward()
        #     raw_obs = self._env.env._get_observations()
        #     self._hdf5_step = 0

        # elif self.init_states and self._init_states is not None:
        #     self._env.set_init_state(self._init_states[self._init_state_id])
        #     self._env.sim.forward()
        #     raw_obs = self._env.env._get_observations()
###
         #umi(6)
        # Adjust fov and wrist camera
        cam_id = self._env.sim.model.camera_name2id("robot0_eye_in_hand")
        self._env.sim.model.cam_fovy[cam_id] = 150
        hand_body_id = self._env.sim.model.cam_bodyid[cam_id]
        self._env.sim.model.cam_pos[cam_id] += np.array([0.025, 0.0, 0.0])
        self._env.sim.model.cam_targetbodyid[cam_id] = hand_body_id
        self._env.sim.forward()
        
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        # Increasing this value can improve determinism and reproducibility across resets.
        for robot in self._env.robots:
            robot.controller.use_delta = False
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D, got shape {action.shape} with ndim={action.ndim}"
            )

        if action.shape[0] != ACTION_DIM:
            raise ValueError(
                f"Expected {ACTION_DIM}D absolute EE action, got shape {action.shape}"
            )

        # The policy outputs an 8D quaternion absolute action；
        # The LIBERO absolute OSC_POSE controller accepts a 7D axis-angle action。
        libero_action = abs_quat_action_to_libero_action(action)

        raw_obs, reward, done, info = self._env.step(libero_action.tolist())

        is_success = self._env.check_success()
        terminated = done or is_success

        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )

        observation = self._format_raw_obs(raw_obs)

        if terminated:
            info["final_info"] = {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            self.reset()

        truncated = False
        return observation, reward, terminated, truncated, info
    # def _save_action_debug_plot(self, reason: str = "unknown"):
    #     if self._action_debug_saved:
    #         return

    #     if len(self._debug_policy_actions) == 0:
    #         return

    #     import os
    #     import numpy as np

    #     os.makedirs(self.hdf5_debug_dir, exist_ok=True)

    #     policy_actions = np.asarray(self._debug_policy_actions, dtype=np.float32)
    #     hdf5_actions = np.asarray(self._debug_hdf5_actions, dtype=np.float32)
    #     steps = np.asarray(self._debug_action_steps, dtype=np.int32)

    #     if policy_actions.ndim != 2 or policy_actions.shape[1] != 8:
    #         print(f"[action-plot] skip: bad policy_actions shape {policy_actions.shape}")
    #         return

    #     if hdf5_actions.ndim != 2 or hdf5_actions.shape[1] != 8:
    #         print(f"[action-plot] skip: bad hdf5_actions shape {hdf5_actions.shape}")
    #         return

    #     tag = (
    #         f"{self.task}_task{self.task_id}_"
    #         f"{self.hdf5_demo_name}_"
    #         f"{int(time.time())}"
    #     )

    #     npz_path = os.path.join(self.hdf5_debug_dir, f"{tag}_actions.npz")
    #     png_path = os.path.join(self.hdf5_debug_dir, f"{tag}_8d_action_compare.png")
    #     err_png_path = os.path.join(self.hdf5_debug_dir, f"{tag}_action_error.png")

    #     np.savez(
    #         npz_path,
    #         steps=steps,
    #         policy_actions=policy_actions,
    #         hdf5_actions=hdf5_actions,
    #         rewards=np.asarray(self._debug_rewards, dtype=np.float32),
    #     )

    #     import matplotlib
    #     matplotlib.use("Agg")
    #     import matplotlib.pyplot as plt

    #     names = [
    #         "x",
    #         "y",
    #         "z",
    #         "qx",
    #         "qy",
    #         "qz",
    #         "qw",
    #         "gripper",
    #     ]

    #     fig, axes = plt.subplots(8, 1, figsize=(14, 18), sharex=True)
    #     # import os, torch, numpy as np;
    #     # save_dir="/data/guolinzheng/umi_work_space_0324/libero_action_compare_debug"
    #     # os.makedirs(save_dir, exist_ok=True)
    #     # a=torch.as_tensor(hdf5_actions[:50]).detach().cpu().float()
    #     # torch.save(a, f"{save_dir}/hdf5_actions_image_first50.pt")
    #     for i, ax in enumerate(axes):
    #         ax.plot(steps, hdf5_actions[:, i], label="hdf5", linewidth=1.5)
    #         ax.plot(steps, policy_actions[:, i], label="policy", linewidth=1.2, alpha=0.85)
    #         ax.set_ylabel(names[i])
    #         ax.grid(True, alpha=0.3)

    #         if i == 0:
    #             ax.set_title(
    #                 f"8D action compare | task={self.task} | demo={self.hdf5_demo_name} | reason={reason}"
    #             )
    #             ax.legend(loc="best")

    #     axes[-1].set_xlabel("step")

    #     fig.tight_layout()
    #     fig.savefig(png_path, dpi=150)
    #     plt.close(fig)

    #     pos_err = np.linalg.norm(policy_actions[:, :3] - hdf5_actions[:, :3], axis=1)

    #     p_q = policy_actions[:, 3:7]
    #     g_q = hdf5_actions[:, 3:7]
    #     p_q = p_q / (np.linalg.norm(p_q, axis=1, keepdims=True) + 1e-8)
    #     g_q = g_q / (np.linalg.norm(g_q, axis=1, keepdims=True) + 1e-8)
    #     quat_err = 1.0 - np.abs(np.sum(p_q * g_q, axis=1))

    #     gripper_err = np.abs(policy_actions[:, 7] - hdf5_actions[:, 7])

    #     fig, ax = plt.subplots(figsize=(14, 5))
    #     ax.plot(steps, pos_err, label="pos_l2_err")
    #     ax.plot(steps, quat_err, label="quat_err: 1 - |dot|")
    #     ax.plot(steps, gripper_err, label="gripper_abs_err")
    #     ax.set_title(f"Action error | task={self.task} | demo={self.hdf5_demo_name}")
    #     ax.set_xlabel("step")
    #     ax.set_ylabel("error")
    #     ax.grid(True, alpha=0.3)
    #     ax.legend(loc="best")
    #     fig.tight_layout()
    #     fig.savefig(err_png_path, dpi=150)
    #     plt.close(fig)

    #     self._action_debug_saved = True

    #     print("[action-plot] saved:")
    #     print("  npz:", npz_path)
    #     print("  8d :", png_path)
    #     print("  err:", err_png_path)
        
    # def _get_hdf5_wrist_image(self) -> np.ndarray | None:
    #     """
    #     Return HDF5 wrist image for current env observation step.

    #     reset returns obs_0 when self._hdf5_step = 0
    #     step applies action_t, increments self._hdf5_step, then returns obs_{t+1}
    #     Use the current self._hdf5_step as the image index.
    #     """
    #     if not self.hdf5_use_obs_image:
    #         return None

    #     if self._hdf5_wrist_images is None:
    #         return None

    #     idx = int(self._hdf5_step)
    #     idx = max(0, min(idx, len(self._hdf5_wrist_images) - 1))

    #     img = self._hdf5_wrist_images[idx]
    #     img = np.asarray(img)

    #     # Support CHW to HWC
    #     if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
    #         img = np.transpose(img, (1, 2, 0))

    #     # Support grayscale
    #     if img.ndim == 2:
    #         img = np.repeat(img[..., None], 3, axis=-1)

    #     # float [0,1] -> uint8
    #     if img.dtype != np.uint8:
    #         img = img.astype(np.float32)
    #         if img.max() <= 1.5:
    #             img = img * 255.0
    #         img = np.clip(img, 0, 255).astype(np.uint8)

    #     # Ensure HWC with 3 channels
    #     if img.ndim != 3 or img.shape[-1] != 3:
    #         raise ValueError(
    #             f"Expected HDF5 wrist image HWC/RGB with 3 channels, got shape={img.shape}"
    #         )

    #     # If the HDF5 image is already the final training image, only resize it to the env observation size.
    #     img = cv2.resize(
    #         img,
    #         (self.observation_width, self.observation_height),
    #         interpolation=cv2.INTER_AREA,
    #     )

    #     if idx < 3:
    #         print(
    #             f"[hdf5-image-debug] idx={idx} "
    #             f"shape={img.shape} dtype={img.dtype} "
    #             f"min/max/mean={int(img.min())}/{int(img.max())}/{float(img.mean()):.3f}"
    #         )

    #     return img
    
    # def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    #     if action.ndim != 1:
    #         raise ValueError(
    #             f"Expected action to be 1-D, got shape {action.shape} with ndim={action.ndim}"
    #         )

    #     if action.shape[0] != ACTION_DIM:
    #         raise ValueError(
    #             f"Expected {ACTION_DIM}D absolute EE action, got shape {action.shape}"
    #         )

    #     policy_action = np.asarray(action, dtype=np.float32).copy()
    #     exec_action = policy_action

    #     debug_info = {}

    #     if self.hdf5_replay:
    #         if self._hdf5_actions is None:
    #             raise RuntimeError("hdf5_replay=True but no HDF5 actions were loaded.")

    #         if self._hdf5_step >= len(self._hdf5_actions):
    #             self._save_action_debug_plot(reason="hdf5_actions_exhausted")
    #             # End the episode after demo actions are exhausted
    #             raw_obs = self._env.env._get_observations()
    #             observation = self._format_raw_obs(raw_obs)
    #             info = {
    #                 "task": self.task,
    #                 "task_id": self.task_id,
    #                 "done": True,
    #                 "is_success": bool(self._env.check_success()),
    #                 "hdf5_done": True,
    #             }
    #             return observation, 0.0, True, False, info

    #         hdf5_action = self._hdf5_actions[self._hdf5_step].astype(np.float32)
    #         exec_action = hdf5_action
            

    #         self._debug_policy_actions.append(policy_action.copy())
    #         self._debug_hdf5_actions.append(hdf5_action.copy())
    #         self._debug_action_steps.append(int(self._hdf5_step))

    #         if self.hdf5_compare_policy:
    #             pred = policy_action
    #             gt = hdf5_action

    #             pred_q = pred[3:7] / (np.linalg.norm(pred[3:7]) + 1e-8)
    #             gt_q = gt[3:7] / (np.linalg.norm(gt[3:7]) + 1e-8)

    #             action_pos_err = float(np.linalg.norm(pred[:3] - gt[:3]))
    #             action_quat_err = float(1.0 - abs(np.dot(pred_q, gt_q)))
    #             action_gripper_err = float(abs(pred[7] - gt[7]))

    #             debug_info.update(
    #                 {
    #                     "hdf5_step": int(self._hdf5_step),
    #                     "policy_action": policy_action.copy(),
    #                     "hdf5_action": hdf5_action.copy(),
    #                     "action_pos_err": action_pos_err,
    #                     "action_quat_err": action_quat_err,
    #                     "action_gripper_err": action_gripper_err,
    #                     "policy_gripper": float(pred[7]),
    #                     "hdf5_gripper": float(gt[7]),
    #                 }
    #             )

    #             if self._hdf5_step < 10 or self._hdf5_step % 20 == 0:
    #                 print(
    #                     f"[hdf5-debug] step={self._hdf5_step:04d} "
    #                     f"pos_err={action_pos_err:.6f} "
    #                     f"quat_err={action_quat_err:.6f} "
    #                     f"grip_err={action_gripper_err:.3f} "
    #                     f"pred_g={pred[7]:+.3f} "
    #                     f"gt_g={gt[7]:+.3f}"
    #                 )

    #     # The env still performs the 8D to 7D conversion internally
    #     libero_action = abs_quat_action_to_libero_action(exec_action)

    #     raw_obs, reward, done, info = self._env.step(libero_action.tolist())

    #     self._hdf5_step += 1

    #     is_success = self._env.check_success()
    #     terminated = done or is_success

    #     info.update(
    #         {
    #             "task": self.task,
    #             "task_id": self.task_id,
    #             "done": done,
    #             "is_success": is_success,
    #             **debug_info,
    #         }
    #     )

    #     observation = self._format_raw_obs(raw_obs)

    #     if terminated:
    #         info["final_info"] = {
    #             "task": self.task,
    #             "task_id": self.task_id,
    #             "done": bool(done),
    #             "is_success": bool(is_success),
    #         }
    #         self._save_action_debug_plot(reason="hdf5_actions_exhausted")
    #         self.reset()

    #     truncated = False
    #     return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
) -> list[Callable[[], LiberoEnv]]:
    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_index=episode_index,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))

    return fns


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")

    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)

    camera_names = _parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]

    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    print(
        f"Creating LIBERO envs | suites={suite_names} | "
        f"n_envs(per task)={n_envs} | init_states={init_states}"
    )

    if task_ids_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)

    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter)

        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}'.")

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
            )

            out[suite_name][tid] = env_cls(fns)

            print(
                f"Built vec env | suite={suite_name} | "
                f"task_id={tid} | n_envs={n_envs}"
            )

    return {suite: dict(task_map) for suite, task_map in out.items()}


#####################################################
#####################################################
# #!/usr/bin/env python

# # Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# from __future__ import annotations

# import os
# from collections import defaultdict
# from collections.abc import Callable, Iterable, Mapping, Sequence
# from functools import partial
# from pathlib import Path
# from typing import Any

# import gymnasium as gym
# import numpy as np
# import torch
# from gymnasium import spaces
# from libero.libero import benchmark, get_libero_path
# from libero.libero.envs import OffScreenRenderEnv
# from robosuite.utils.transform_utils import quat2axisangle


# def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
#     """Normalize camera_name into a non-empty list of strings."""
#     if isinstance(camera_name, str):
#         cams = [c.strip() for c in camera_name.split(",") if c.strip()]
#     elif isinstance(camera_name, (list | tuple)):
#         cams = [str(c).strip() for c in camera_name if str(c).strip()]
#     else:
#         raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
#     if not cams:
#         raise ValueError("camera_name resolved to an empty list.")
#     return cams


# def _get_suite(name: str) -> benchmark.Benchmark:
#     """Instantiate a LIBERO suite by name with clear validation."""
#     bench = benchmark.get_benchmark_dict()
#     if name not in bench:
#         raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
#     suite = bench[name]()
#     if not getattr(suite, "tasks", None):
#         raise ValueError(f"Suite '{name}' has no tasks.")
#     return suite


# def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
#     """Validate/normalize task ids. If None → all tasks."""
#     if task_ids is None:
#         return list(range(total_tasks))
#     ids = sorted({int(t) for t in task_ids})
#     for t in ids:
#         if t < 0 or t >= total_tasks:
#             raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
#     return ids


# def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
#     init_states_path = (
#         Path(get_libero_path("init_states"))
#         / task_suite.tasks[i].problem_folder
#         / task_suite.tasks[i].init_states_file
#     )
#     init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
#     return init_states


# def get_libero_dummy_action():
#     """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
#     return [0, 0, 0, 0, 0, 0, -1]


# OBS_STATE_DIM = 8
# ACTION_DIM = 7
# AGENT_POS_LOW = -1000.0
# AGENT_POS_HIGH = 1000.0
# ACTION_LOW = -1.0
# ACTION_HIGH = 1.0
# TASK_SUITE_MAX_STEPS: dict[str, int] = {
#     "libero_spatial": 280,  # longest training demo has 193 steps
#     "libero_object": 280,  # longest training demo has 254 steps
#     "libero_goal": 300,  # longest training demo has 270 steps
#     "libero_10": 520,  # longest training demo has 505 steps
#     "libero_90": 400,  # longest training demo has 373 steps
# }


# class LiberoEnv(gym.Env):
#     metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

#     def __init__(
#         self,
#         task_suite: Any,
#         task_id: int,
#         task_suite_name: str,
#         camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
#         obs_type: str = "pixels",
#         render_mode: str = "rgb_array",
#         observation_width: int = 256,
#         observation_height: int = 256,
#         visualization_width: int = 640,
#         visualization_height: int = 480,
#         init_states: bool = True,
#         episode_index: int = 0,
#         camera_name_mapping: dict[str, str] | None = None,
#         num_steps_wait: int = 10,
#     ):
#         super().__init__()
#         self.task_id = task_id
#         self.obs_type = obs_type
#         self.render_mode = render_mode
#         self.observation_width = observation_width
#         self.observation_height = observation_height
#         self.visualization_width = visualization_width
#         self.visualization_height = visualization_height
#         self.init_states = init_states
#         self.camera_name = _parse_camera_names(
#             camera_name
#         )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)

#         # Map raw camera names to "image1" and "image2".
#         # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
#         # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
#         # This ensures the policy consistently receives observations in the
#         # expected format regardless of the original camera naming.
#         if camera_name_mapping is None:
#             camera_name_mapping = {
#                 "agentview_image": "image",
#                 "robot0_eye_in_hand_image": "image2",
#             }
#         self.camera_name_mapping = camera_name_mapping
#         self.num_steps_wait = num_steps_wait
#         self.episode_index = episode_index
#         # Load once and keep
#         self._init_states = get_task_init_states(task_suite, self.task_id) if self.init_states else None
#         self._init_state_id = self.episode_index  # tie each sub-env to a fixed init state

#         self._env = self._make_envs_task(task_suite, self.task_id)
#         default_steps = 500
#         self._max_episode_steps = TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)

#         images = {}
#         for cam in self.camera_name:
#             images[self.camera_name_mapping[cam]] = spaces.Box(
#                 low=0,
#                 high=255,
#                 shape=(self.observation_height, self.observation_width, 3),
#                 dtype=np.uint8,
#             )

#         if self.obs_type == "state":
#             raise NotImplementedError(
#                 "The 'state' observation type is not supported in LiberoEnv. "
#                 "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
#             )

#         elif self.obs_type == "pixels":
#             self.observation_space = spaces.Dict(
#                 {
#                     "pixels": spaces.Dict(images),
#                 }
#             )
#         elif self.obs_type == "pixels_agent_pos":
#             self.observation_space = spaces.Dict(
#                 {
#                     "pixels": spaces.Dict(images),
#                     "agent_pos": spaces.Box(
#                         low=AGENT_POS_LOW,
#                         high=AGENT_POS_HIGH,
#                         shape=(OBS_STATE_DIM,),
#                         dtype=np.float64,
#                     ),
#                 }
#             )

#         self.action_space = spaces.Box(
#             low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
#         )

#     def render(self):
#         raw_obs = self._env.env._get_observations()
#         image = self._format_raw_obs(raw_obs)["pixels"]["image"]
#         return image

#     def _make_envs_task(self, task_suite: Any, task_id: int = 0):
#         task = task_suite.get_task(task_id)
#         self.task = task.name
#         self.task_description = task.language
#         task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

#         env_args = {
#             "bddl_file_name": task_bddl_file,
#             "camera_heights": self.observation_height,
#             "camera_widths": self.observation_width,
#         }
#         env = OffScreenRenderEnv(**env_args)
#         env.reset()
#         return env

#     def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
#         images = {}
#         for camera_name in self.camera_name:
#             image = raw_obs[camera_name]
#             image = image[::-1, ::-1]  # rotate 180 degrees
#             images[self.camera_name_mapping[camera_name]] = image
#         state = np.concatenate(
#             (
#                 raw_obs["robot0_eef_pos"],
#                 quat2axisangle(raw_obs["robot0_eef_quat"]),
#                 raw_obs["robot0_gripper_qpos"],
#             )
#         )
#         agent_pos = state
#         if self.obs_type == "pixels":
#             return {"pixels": images.copy()}
#         if self.obs_type == "pixels_agent_pos":
#             return {
#                 "pixels": images.copy(),
#                 "agent_pos": agent_pos,
#             }
#         raise NotImplementedError(
#             f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
#             "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
#         )

#     def reset(self, seed=None, **kwargs):
#         super().reset(seed=seed)
#         self._env.seed(seed)
#         if self.init_states and self._init_states is not None:
#             self._env.set_init_state(self._init_states[self._init_state_id])
#         raw_obs = self._env.reset()

#         # After reset, objects may be unstable (slightly floating, intersecting, etc.).
#         # Step the simulator with a no-op action for a few frames so everything settles.
#         # Increasing this value can improve determinism and reproducibility across resets.
#         for _ in range(self.num_steps_wait):
#             raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
#         observation = self._format_raw_obs(raw_obs)
#         # import ipdb;ipdb.set_trace()
#         info = {"is_success": False}
#         return observation, info

#     def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
#         if action.ndim != 1:
#             raise ValueError(
#                 f"Expected action to be 1-D (shape (action_dim,)), "
#                 f"but got shape {action.shape} with ndim={action.ndim}"
#             )
#         raw_obs, reward, done, info = self._env.step(action)

#         is_success = self._env.check_success()
#         terminated = done or is_success
#         info.update(
#             {
#                 "task": self.task,
#                 "task_id": self.task_id,
#                 "done": done,
#                 "is_success": is_success,
#             }
#         )
#         observation = self._format_raw_obs(raw_obs)
#         if terminated:
#             info["final_info"] = {
#                 "task": self.task,
#                 "task_id": self.task_id,
#                 "done": bool(done),
#                 "is_success": bool(is_success),
#             }
#             self.reset()
#         truncated = False
#         return observation, reward, terminated, truncated, info

#     def close(self):
#         self._env.close()


# def _make_env_fns(
#     *,
#     suite,
#     suite_name: str,
#     task_id: int,
#     n_envs: int,
#     camera_names: list[str],
#     init_states: bool,
#     gym_kwargs: Mapping[str, Any],
# ) -> list[Callable[[], LiberoEnv]]:
#     """Build n_envs factory callables for a single (suite, task_id)."""

#     def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
#         local_kwargs = dict(kwargs)
#         return LiberoEnv(
#             task_suite=suite,
#             task_id=task_id,
#             task_suite_name=suite_name,
#             camera_name=camera_names,
#             init_states=init_states,
#             episode_index=episode_index,
#             **local_kwargs,
#         )

#     fns: list[Callable[[], LiberoEnv]] = []
#     for episode_index in range(n_envs):
#         fns.append(partial(_make_env, episode_index, **gym_kwargs))
#     return fns


# # ---- Main API ----------------------------------------------------------------


# def create_libero_envs(
#     task: str,
#     n_envs: int,
#     gym_kwargs: dict[str, Any] | None = None,
#     camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
#     init_states: bool = True,
#     env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
# ) -> dict[str, dict[int, Any]]:
#     """
#     Create vectorized LIBERO environments with a consistent return shape.

#     Returns:
#         dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
#     Notes:
#         - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
#         - `task` can be a single suite or a comma-separated list of suites.
#         - You may pass `task_ids` (list[int]) inside `gym_kwargs` to restrict tasks per suite.
#     """
#     if env_cls is None or not callable(env_cls):
#         raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
#     if not isinstance(n_envs, int) or n_envs <= 0:
#         raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

#     gym_kwargs = dict(gym_kwargs or {})
#     task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks

#     camera_names = _parse_camera_names(camera_name)
#     suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
#     if not suite_names:
#         raise ValueError("`task` must contain at least one LIBERO suite name.")

#     print(
#         f"Creating LIBERO envs | suites={suite_names} | n_envs(per task)={n_envs} | init_states={init_states}"
#     )
#     if task_ids_filter is not None:
#         print(f"Restricting to task_ids={task_ids_filter}")

#     out: dict[str, dict[int, Any]] = defaultdict(dict)

#     for suite_name in suite_names:
#         suite = _get_suite(suite_name)
#         total = len(suite.tasks)
#         selected = _select_task_ids(total, task_ids_filter)

#         if not selected:
#             raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

#         for tid in selected:
#             fns = _make_env_fns(
#                 suite=suite,
#                 suite_name=suite_name,
#                 task_id=tid,
#                 n_envs=n_envs,
#                 camera_names=camera_names,
#                 init_states=init_states,
#                 gym_kwargs=gym_kwargs,
#             )
#             out[suite_name][tid] = env_cls(fns)
#             print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

#     # return plain dicts for predictability
#     return {suite: dict(task_map) for suite, task_map in out.items()}
