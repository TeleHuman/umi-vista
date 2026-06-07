#!/usr/bin/env python
import csv
import logging
import os
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.transforms import AbsoluteActionTransform
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging


EPISODE_ID = int(os.environ.get("DATASET_EPISODE_ID", "0"))
MAX_POINTS_ENV = os.environ.get("DATASET_MAX_POINTS")
MAX_POINTS = int(MAX_POINTS_ENV) if MAX_POINTS_ENV else None
SAVE_DIR = os.environ.get(
    "DATASET_CHUNK_SAVE_DIR",
    "/data/guolinzheng/umi_work_space_0324/RhodesLeRobot/visual/dataset_episode_chunk_policy_compare",
)
ACTION_DIM = 16

ACTION_NAMES_16D = [
    "left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw", "left_gripper",
    "right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw", "right_gripper",
]


def clone_batch(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = deepcopy(v)
    return out


def add_batch_dim(sample: dict) -> dict:
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v
    return batch


def ensure_btd_action(action: torch.Tensor) -> torch.Tensor:
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action)
    if action.ndim == 1:
        return action[None, None]
    if action.ndim == 2:
        return action[:, None]
    if action.ndim == 3:
        return action
    raise ValueError(f"Unsupported action shape: {tuple(action.shape)}")


def ensure_bd_state(state: torch.Tensor) -> torch.Tensor:
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state)
    if state.ndim == 1:
        return state[None]
    if state.ndim == 2:
        return state
    if state.ndim == 3:
        return state[:, 0]
    raise ValueError(f"Unsupported state shape: {tuple(state.shape)}")


def make_delta_action_mask(action_dim: int) -> torch.Tensor:
    mask = torch.ones(action_dim, dtype=torch.bool)
    if action_dim == 16:
        mask[7] = False
        mask[15] = False
    elif action_dim == 8:
        mask[7] = False
    else:
        raise ValueError(f"Unsupported action_dim={action_dim}; expected 8 or 16")
    return mask


def delta_chunk_to_absolute(delta_action: torch.Tensor, raw_state: torch.Tensor) -> torch.Tensor:
    delta_action = ensure_btd_action(delta_action).detach().cpu().clone()
    raw_state = ensure_bd_state(raw_state).detach().cpu()
    transform = AbsoluteActionTransform(make_delta_action_mask(delta_action.shape[-1]))
    return transform({"observation.state": raw_state, "action": delta_action})["action"]


def apply_postprocessor_to_action_chunk(postprocessor, pred: torch.Tensor) -> torch.Tensor:
    """Match async deployment: postprocess each [B,D] action step separately."""
    pred = ensure_btd_action(pred)
    processed_actions = []
    for i in range(pred.shape[1]):
        out = postprocessor(pred[:, i, :])
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"postprocessor returned {type(out)} at step={i}")
        processed_actions.append(out)
    return torch.stack(processed_actions, dim=1)


def tensor_to_image_np(img) -> np.ndarray | None:
    if img is None or not isinstance(img, torch.Tensor):
        return None
    img = img.detach().float().cpu()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 5:
        img = img[0, 0]
    if img.ndim != 3:
        return None
    if img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    arr = img.numpy()
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    if max_v > 1.0 or min_v < 0.0:
        arr = (arr - min_v) / (max_v - min_v + 1e-8)
    return np.clip(arr, 0.0, 1.0)


def plot_one_chunk(pred_chunk, gt_chunk, sample_name, save_path, left_wrist_img=None, right_wrist_img=None):
    t = min(pred_chunk.shape[0], gt_chunk.shape[0])
    d = min(pred_chunk.shape[1], gt_chunk.shape[1])
    pred_chunk = pred_chunk[:t, :d]
    gt_chunk = gt_chunk[:t, :d]
    left_img = tensor_to_image_np(left_wrist_img)
    right_img = tensor_to_image_np(right_wrist_img)
    has_images = left_img is not None or right_img is not None
    names = ACTION_NAMES_16D if d == 16 else [f"dim_{i}" for i in range(d)]

    if has_images:
        fig, axes = plt.subplots(
            nrows=d,
            ncols=2,
            figsize=(18, max(8, d * 1.35)),
            squeeze=False,
            sharex="col",
            gridspec_kw={"width_ratios": [1.2, 4.0]},
        )
        if left_img is not None:
            axes[0, 0].imshow(left_img)
            axes[0, 0].set_title("left_wrist", fontsize=9)
        axes[0, 0].axis("off")
        if right_img is not None and d > 1:
            axes[1, 0].imshow(right_img)
            axes[1, 0].set_title("right_wrist", fontsize=9)
        if d > 1:
            axes[1, 0].axis("off")
        for row in range(2, d):
            axes[row, 0].axis("off")
        action_axes = axes[:, 1]
    else:
        fig, axes = plt.subplots(nrows=d, ncols=1, figsize=(14, max(8, d * 1.35)), squeeze=False, sharex=True)
        action_axes = axes[:, 0]

    x = np.arange(t)
    for dim in range(d):
        ax = action_axes[dim]
        ax.plot(x, gt_chunk[:, dim], label="dataset gt abs", linewidth=1.6)
        ax.plot(x, pred_chunk[:, dim], label="policy pred abs", linewidth=1.1, alpha=0.9)
        ax.set_ylabel(names[dim])
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=9)
    mse = float(np.mean((pred_chunk - gt_chunk) ** 2))
    rmse = mse ** 0.5
    mae = float(np.mean(np.abs(pred_chunk - gt_chunk)))
    action_axes[-1].set_xlabel("action chunk step")
    fig.suptitle(f"{sample_name} | T={t}, D={d} | MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}", y=0.995)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def get_episode_bounds(dataset, episode_id: int) -> tuple[int, int]:
    start = int(dataset.meta.episodes["dataset_from_index"][episode_id])
    end = int(dataset.meta.episodes["dataset_to_index"][episode_id])
    return start, end


def get_abs_gt_chunk(dataset, dataset_idx: int, episode_end: int, target_len: int, raw_batch: dict, use_delta_action: bool) -> torch.Tensor:
    raw_action = ensure_btd_action(raw_batch[ACTION]).to(torch.float32)
    if raw_action.shape[1] > 1:
        gt = raw_action[:, :target_len]
        if use_delta_action:
            gt = delta_chunk_to_absolute(gt, raw_batch["observation.state"])
        return gt

    chunks = []
    for idx in range(dataset_idx, min(episode_end, dataset_idx + target_len)):
        sample = dataset[idx]
        b = add_batch_dim(sample)
        a = ensure_btd_action(b[ACTION]).to(torch.float32)[:, :1]
        if use_delta_action:
            a = delta_chunk_to_absolute(a, b["observation.state"])
        chunks.append(a)
    return torch.cat(chunks, dim=1)


@parser.wrap()
def eval_dataset_episode_chunks(cfg: TrainPipelineConfig):
    cfg.validate()
    init_logging()

    save_dir = Path(SAVE_DIR)
    per_point_dir = save_dir / f"episode_{EPISODE_ID:04d}_per_point_chunks"
    per_point_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    if cfg.policy.use_delta_action:
        logging.info("Using delta action stats/transform on dataset")
        dataset.load_delta_action_norm_stats()
    else:
        logging.info("Using absolute action stats on dataset")
        dataset.load_abs_action_norm_stats()
    dataset[0]

    start_idx, end_idx = get_episode_bounds(dataset, EPISODE_ID)
    if MAX_POINTS is not None:
        end_idx = min(end_idx, start_idx + MAX_POINTS)
    logging.info("Episode %d dataset range [%d, %d), points=%d", EPISODE_ID, start_idx, end_idx, end_idx - start_idx)

    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    device = torch.device(cfg.policy.device)
    policy.to(device)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    logging.info("policy.pretrained_path=%s", cfg.policy.pretrained_path)
    logging.info("policy.use_delta_action=%s", cfg.policy.use_delta_action)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    metric_rows = []
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for local_idx, dataset_idx in enumerate(range(start_idx, end_idx)):
            sample = dataset[dataset_idx]
            raw_batch = clone_batch(add_batch_dim(sample))
            batch = preprocessor(clone_batch(raw_batch))

            pred_norm = policy.predict_action_chunk(batch).to(torch.float32)
            pred_delta = apply_postprocessor_to_action_chunk(postprocessor, pred_norm).to(torch.float32)
            pred_abs = pred_delta
            if cfg.policy.use_delta_action:
                pred_abs = delta_chunk_to_absolute(pred_delta, raw_batch["observation.state"])
            pred_abs = ensure_btd_action(pred_abs).detach().cpu()

            gt_abs = get_abs_gt_chunk(
                dataset=dataset,
                dataset_idx=dataset_idx,
                episode_end=end_idx,
                target_len=pred_abs.shape[1],
                raw_batch=raw_batch,
                use_delta_action=cfg.policy.use_delta_action,
            )
            gt_abs = ensure_btd_action(gt_abs).detach().cpu()

            valid_len = min(pred_abs.shape[1], gt_abs.shape[1], end_idx - dataset_idx)
            pred_np = pred_abs[0, :valid_len, :ACTION_DIM].float().numpy().astype(np.float32)
            gt_np = gt_abs[0, :valid_len, :ACTION_DIM].float().numpy().astype(np.float32)

            diff = pred_np - gt_np
            mse = float(np.mean(diff ** 2))
            rmse = mse ** 0.5
            mae = float(np.mean(np.abs(diff)))
            max_abs = float(np.max(np.abs(diff)))

            name = f"episode_{EPISODE_ID:04d}_point_{local_idx:06d}_dataset_idx_{dataset_idx:08d}"
            png_path = per_point_dir / f"{name}_policy_vs_dataset_action.png"
            plot_one_chunk(
                pred_chunk=pred_np,
                gt_chunk=gt_np,
                sample_name=name,
                save_path=str(png_path),
                left_wrist_img=raw_batch.get("observation.images.left_wrist"),
                right_wrist_img=raw_batch.get("observation.images.right_wrist"),
            )

            metric_rows.append({
                "episode_id": EPISODE_ID,
                "local_idx": local_idx,
                "dataset_idx": dataset_idx,
                "valid_len": valid_len,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "max_abs": max_abs,
                "png": str(png_path),
            })
            all_pred.append(pred_np)
            all_gt.append(gt_np)
            logging.info("[%04d/%04d] dataset_idx=%d T=%d mse=%.8f rmse=%.8f mae=%.8f max_abs=%.8f plot=%s",
                         local_idx + 1, end_idx - start_idx, dataset_idx, valid_len, mse, rmse, mae, max_abs, png_path)

    csv_path = save_dir / f"episode_{EPISODE_ID:04d}_per_point_chunk_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    npz_path = save_dir / f"episode_{EPISODE_ID:04d}_per_point_chunks_policy_vs_dataset_action.npz"
    np.savez(
        npz_path,
        episode_id=EPISODE_ID,
        start_idx=start_idx,
        end_idx=end_idx,
        pred_chunks=np.asarray(all_pred, dtype=object),
        gt_chunks=np.asarray(all_gt, dtype=object),
    )

    logging.info("Saved per-point plots: %s", per_point_dir)
    logging.info("Saved metrics csv: %s", csv_path)
    logging.info("Saved npz: %s", npz_path)


if __name__ == "__main__":
    eval_dataset_episode_chunks()
