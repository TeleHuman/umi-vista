#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate LingbotVLA checkpoint in LeRobot Libero envs.

This script is intentionally lightweight and does not require registering a new
LeRobot policy type. It directly loads `QwenPiInfer` from an inference python file
and adapts it to the LeRobot rollout loop.

Environment variables:
- LINGBOT_CKPT_PATH: Required. Path to LingbotVLA hf_ckpt directory.
- LINGBOT_INFER_PY: Optional. Path to inference.py that defines QwenPiInfer.
  Default: /gemini/space/users/glz/workspace/lingbot-vla-main/tasks/vla/inference.py
- LINGBOT_INFER_LENGTH: Optional. Default: 50
- LINGBOT_USE_BF16: Optional. true/false, default: true

Example:
  LINGBOT_CKPT_PATH=/path/to/hf_ckpt \
  python -m lerobot.scripts.lerobot_eval_lingbotvla \
      --env.type=libero \
      --env.task='[libero_10]' \
      --eval.batch_size=1 \
      --eval.n_episodes=10
"""

import concurrent.futures as cf
import importlib.util
import json
import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging, inside_slurm

DEFAULT_LINGBOT_INFER_PY = "/gemini/space/users/glz/workspace/lingbot-vla-main/tasks/vla/inference.py"


class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def _str_to_bool(raw: str, default: bool) -> bool:
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _load_qwen_pi_infer_class(inference_py: str):
    if not os.path.exists(inference_py):
        raise FileNotFoundError(f"Inference file not found: {inference_py}")

    spec = importlib.util.spec_from_file_location("lingbot_inference_module", inference_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {inference_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "QwenPiInfer"):
        raise AttributeError(f"QwenPiInfer not found in: {inference_py}")
    return module.QwenPiInfer


class LingbotVLAPolicyAdapter:
    """Minimal adapter exposing select_action/reset/eval for rollout loop."""

    def __init__(self, infer_model: Any):
        self.infer_model = infer_model

    def reset(self):
        # No recurrent cache to clear for current QwenPiInfer implementation.
        return None

    def eval(self):
        return self

    def _pick_image_key(self, batch: dict[str, Any]) -> str:
        preferred = [
            "observation.images.wrist_image",
            "observation.images.image2",
            "observation.images.image",
        ]
        for key in preferred:
            if key in batch:
                return key

        image_keys = [k for k in batch.keys() if k.startswith("observation.images.")]
        if not image_keys:
            raise KeyError("No image key found in batch for LingbotVLA inference.")
        return image_keys[0]

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        image_key = self._pick_image_key(batch)
        if "observation.state" not in batch:
            raise KeyError("Missing key 'observation.state' in batch.")

        images = batch[image_key]
        states = batch["observation.state"]
        tasks = batch.get("task", [""] * images.shape[0])
        if isinstance(tasks, str):
            tasks = [tasks]

        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W], got type={type(images)}")
        if not isinstance(states, torch.Tensor) or states.ndim != 2:
            raise ValueError(f"Expected state tensor [B,D], got type={type(states)}")

        bsz = images.shape[0]
        action_list: list[torch.Tensor] = []
        for i in range(bsz):
            img_chw = images[i].detach().cpu().float()
            img_hwc = img_chw.permute(1, 2, 0).numpy()
            img_u8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)

            state_np = states[i].detach().cpu().numpy().astype(np.float32)
            task_str = tasks[i] if i < len(tasks) else ""

            infer_obs = {
                "observation.images.wrist_image": img_u8,
                "observation.state": state_np,
                "task": task_str,
            }
            infer_out = self.infer_model.infer(infer_obs)
            action_chunk = np.asarray(infer_out["action"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[0] == 0:
                raise ValueError(f"Unexpected action chunk shape: {action_chunk.shape}")

            action_list.append(torch.from_numpy(action_chunk[0]))

        return torch.stack(action_list, dim=0)


def make_lingbot_policy_from_env() -> LingbotVLAPolicyAdapter:
    ckpt_path = os.environ.get("LINGBOT_CKPT_PATH", "").strip()
    if not ckpt_path:
        raise ValueError("Please set environment variable LINGBOT_CKPT_PATH to your LingbotVLA hf_ckpt path.")

    inference_py = os.environ.get("LINGBOT_INFER_PY", DEFAULT_LINGBOT_INFER_PY)
    infer_length = int(os.environ.get("LINGBOT_INFER_LENGTH", "50"))
    use_bf16 = _str_to_bool(os.environ.get("LINGBOT_USE_BF16", "true"), default=True)

    QwenPiInfer = _load_qwen_pi_infer_class(inference_py)
    infer_model = QwenPiInfer(ckpt_path, infer_length=infer_length, use_bf16=use_bf16)
    return LingbotVLAPolicyAdapter(infer_model=infer_model)


def rollout(
    env: gym.vector.VectorEnv,
    policy: Any,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    policy.reset()
    observation, _ = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),
        leave=False,
    )
    check_env_attributes_and_types(env)

    while not np.all(done) and step < max_steps:
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = add_envs_task(env, observation)
        if "observation.images.unused_image" in observation:
            del observation["observation.images.unused_image"]

        with torch.inference_mode():
            action = policy.select_action(observation)

        action_numpy: np.ndarray = action.detach().cpu().numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"
        if action_numpy.shape[1] >= 7:
            action_numpy[:, 6] = -(action_numpy[:, 6] * 2 - 1)

        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError("Unsupported `final_info` format; expected dict (Gymnasium >= 1.0).")
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    return ret


def _compile_episode_data(
    rollout_data: dict, done_indices: torch.Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)
    return data_dict


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: Any,
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")
    if not hasattr(policy, "select_action"):
        raise ValueError(f"Policy must implement select_action, got: {type(policy)}")

    start = time.time()
    policy.eval()

    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []
    n_episodes_rendered = 0

    def render_frame(env: gym.vector.VectorEnv):
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs))

        rollout_data = rollout(
            env=env,
            policy=policy,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)
            for stacked_frames, done_index in zip(batch_stacked_frames, done_indices.flatten().tolist(), strict=False):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix({"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"})

    for thread in threads:
        thread.join()

    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: Any,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> TaskMetrics:
    task_result = eval_policy(
        env=env,
        policy=policy,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    metrics = eval_one(
        env,
        policy=policy,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    start_t = time.time()

    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    def _accumulate_to(group: str, metrics: dict):
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    task_runner = partial(
        run_one,
        policy=policy,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    if max_parallel_tasks <= 1:
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    runtime_device = cfg.policy.device if cfg.policy is not None else "cuda"
    runtime_use_amp = cfg.policy.use_amp if cfg.policy is not None else False
    device = get_safe_torch_device(runtime_device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Loading LingbotVLA policy from env vars.")
    policy = make_lingbot_policy_from_env()
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if runtime_use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )

    print("Overall Aggregated Metrics:")
    print(info["overall"])
    print("Per-group Aggregated Metrics:")
    print(info["per_group"])

    close_envs(envs)

    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


def main():
    init_logging()
    eval_main()


if __name__ == "__main__":
    main()
