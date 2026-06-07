#!/usr/bin/env python
import csv
import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.transforms import AbsoluteActionTransform
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging


REAL_OBS_DIR = os.environ.get(
    "REAL_OBS_DIR",
    "/data/guolinzheng/umi_work_space_0324/RhodesLeRobot/real_obs/unpreprocessed_obs",
)
SAVE_DIR = os.environ.get(
    "REAL_OBS_SAVE_DIR",
    "/data/guolinzheng/umi_work_space_0324/RhodesLeRobot/visual/real_unpreprocessed_obs_policy_compare",
)

# The files in real_obs/unpreprocessed_obs are raw deployment observations.
# Keep this true so tokenization, normalization, batching, and device placement match deployment.
APPLY_LEROBOT_PREPROCESSOR = os.environ.get("REAL_OBS_APPLY_PREPROCESSOR", "1") == "1"

# The saved deployment action is normally the final postprocessed action.
POSTPROCESS_PRED = os.environ.get("REAL_OBS_POSTPROCESS_PRED", "1") != "0"

MAX_FILES_ENV = os.environ.get("REAL_OBS_MAX_FILES")
MAX_FILES = int(MAX_FILES_ENV) if MAX_FILES_ENV else None


ACTION_NAMES_16D = [
    "left_x",
    "left_y",
    "left_z",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_qw",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_qw",
    "right_gripper",
]


def clone_batch(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = deepcopy(v)
    return out


def move_tensors_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


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
    """Mask pose/quaternion dims, keep gripper dims absolute."""
    mask = torch.ones(action_dim, dtype=torch.bool)
    if action_dim == 16:
        mask[7] = False
        mask[15] = False
    elif action_dim == 8:
        mask[7] = False
    else:
        raise ValueError(f"Unsupported action_dim={action_dim}; expected 8 or 16")
    return mask


def delta_action_chunk_to_absolute(delta_action: torch.Tensor, raw_state: torch.Tensor) -> torch.Tensor:
    delta_action = ensure_btd_action(delta_action).detach().cpu().clone()
    raw_state = ensure_bd_state(raw_state).detach().cpu()
    transform = AbsoluteActionTransform(make_delta_action_mask(delta_action.shape[-1]))
    return transform({"observation.state": raw_state, "action": delta_action})["action"]


def apply_postprocessor_to_action_chunk(postprocessor, pred: torch.Tensor) -> torch.Tensor:
    """Match async deployment: postprocess each [B, D] action step separately."""
    pred = ensure_btd_action(pred)
    _, chunk_size, _ = pred.shape
    processed_actions = []
    for i in range(chunk_size):
        processed_action = postprocessor(pred[:, i, :])
        if not isinstance(processed_action, torch.Tensor):
            raise TypeError(f"postprocessor returned {type(processed_action)} at step {i}")
        processed_actions.append(processed_action)
    return torch.stack(processed_actions, dim=1)


def tensor_stats_line(name: str, value) -> str:
    if not isinstance(value, torch.Tensor):
        return f"{name}: {type(value).__name__}"
    msg = f"{name}: shape={tuple(value.shape)} dtype={value.dtype}"
    if value.numel() and value.dtype.is_floating_point:
        msg += f" min={float(value.min()):.6g} max={float(value.max()):.6g}"
    return msg


def format_tensor_values(value, precision: int = 6) -> str:
    if not isinstance(value, torch.Tensor):
        return repr(value)
    arr = value.detach().float().cpu().numpy()
    return np.array2string(arr, precision=precision, suppress_small=False, max_line_width=240)


def plot_first_step_timeline(pred: np.ndarray, gt: np.ndarray, save_path: str):
    """Plot action[:, first_chunk_step, dim] across saved real obs files."""
    pred0 = pred[:, 0]
    gt0 = gt[:, 0]
    n, d = pred0.shape

    names = ACTION_NAMES_16D if d == 16 else [f"dim_{i}" for i in range(d)]
    fig, axes = plt.subplots(
        nrows=d,
        ncols=1,
        figsize=(14, max(8, d * 1.35)),
        sharex=True,
        squeeze=False,
    )

    x = np.arange(n)
    for dim in range(d):
        ax = axes[dim, 0]
        ax.plot(x, gt0[:, dim], label="pt action", linewidth=1.6)
        ax.plot(x, pred0[:, dim], label="local pred abs", linewidth=1.1, alpha=0.9)
        ax.set_ylabel(names[dim])
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1, 0].set_xlabel("unpreprocessed_obs file index")
    fig.suptitle(f"Real Unpreprocessed Obs | First Action Step Compare | N={n}, D={d}", y=0.995)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_file_mse(file_mse: np.ndarray, save_path: str):
    x = np.arange(len(file_mse))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, file_mse, linewidth=1.5)
    ax.set_title("Per-file Chunk MSE")
    ax.set_xlabel("unpreprocessed_obs file index")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def tensor_to_image_np(img) -> np.ndarray | None:
    if img is None or not isinstance(img, torch.Tensor):
        return None
    img = img.detach().float().cpu()
    if img.ndim == 4:
        img = img[0]
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


def plot_one_pt_action_chunk(
    pred_chunk: np.ndarray,
    gt_chunk: np.ndarray,
    pt_name: str,
    save_path: str,
    left_wrist_img=None,
    right_wrist_img=None,
):
    """Plot one .pt file's observation images and full [T, D] action chunk."""
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
            axes[1, 0].axis("off")

        for row in range(2, d):
            axes[row, 0].axis("off")
        action_axes = axes[:, 1]
    else:
        fig, axes = plt.subplots(
            nrows=d,
            ncols=1,
            figsize=(14, max(8, d * 1.35)),
            sharex=True,
            squeeze=False,
        )
        action_axes = axes[:, 0]

    x = np.arange(t)
    for dim in range(d):
        ax = action_axes[dim]
        ax.plot(x, gt_chunk[:, dim], label="pt action", linewidth=1.6)
        ax.plot(x, pred_chunk[:, dim], label="local pred abs", linewidth=1.1, alpha=0.9)
        ax.set_ylabel(names[dim])
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=9)

    mse = float(np.mean((pred_chunk - gt_chunk) ** 2))
    rmse = mse**0.5
    mae = float(np.mean(np.abs(pred_chunk - gt_chunk)))

    action_axes[-1].set_xlabel("action chunk step")
    fig.suptitle(f"{pt_name} | T={t}, D={d} | MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}", y=0.995)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(rows: list[dict], save_path: str):
    if not rows:
        return
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@parser.wrap()
def eval_real_preprocessed_obs(cfg: TrainPipelineConfig):
    cfg.validate()
    init_logging()

    obs_dir = Path(REAL_OBS_DIR)
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(obs_dir.glob("*.pt"))
    if MAX_FILES is not None:
        files = files[:MAX_FILES]
    if not files:
        raise FileNotFoundError(f"No .pt files found in {obs_dir}")

    logging.info("Creating dataset only for metadata/config compatibility")
    dataset = make_dataset(cfg)
    dataset[0]

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    device = torch.device(cfg.policy.device)
    policy.to(device)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    logging.info("policy.pretrained_path=%s", cfg.policy.pretrained_path)
    logging.info("policy.pretrained_name_or_path=%s", getattr(cfg.policy, "pretrained_name_or_path", None))
    logging.info("real obs dir=%s", obs_dir)
    logging.info("num files=%d", len(files))
    logging.info("apply lerobot preprocessor=%s", APPLY_LEROBOT_PREPROCESSOR)
    logging.info("postprocess pred=%s", POSTPROCESS_PRED)
    logging.info("policy.use_delta_action=%s", getattr(cfg.policy, "use_delta_action", None))
    logging.info("compare space=absolute; local delta pred is converted with raw observation.state")

    processor_kwargs = {
        "preprocessor_overrides": {
            "device_processor": {"device": device.type},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
    )

    pred_chunks = []
    gt_chunks = []
    metric_rows = []
    per_pt_plot_dir = save_dir / "per_pt_plots"
    per_pt_plot_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for file_idx, pt_path in enumerate(files):
            raw_batch = torch.load(pt_path, map_location="cpu")
            if not isinstance(raw_batch, dict):
                raise TypeError(f"{pt_path} must contain a dict, got {type(raw_batch)}")
            if ACTION not in raw_batch:
                raise KeyError(f"{pt_path} has no {ACTION!r}; keys={list(raw_batch.keys())}")

            if file_idx == 0:
                logging.info("first file=%s", pt_path)
                for key in raw_batch:
                    logging.info("  %s", tensor_stats_line(key, raw_batch[key]))

            gt = ensure_btd_action(raw_batch[ACTION]).to(torch.float32)

            batch = clone_batch(raw_batch)
            if APPLY_LEROBOT_PREPROCESSOR:
                batch = preprocessor(batch)
            else:
                batch = move_tensors_to_device(batch, device)

            logging.info("[%04d/%04d] %s raw_state=%s", file_idx + 1, len(files), pt_path.name, format_tensor_values(raw_batch.get("observation.state")))
            logging.info("[%04d/%04d] %s model_state=%s", file_idx + 1, len(files), pt_path.name, format_tensor_values(batch.get("observation.state")))

            pred = policy.predict_action_chunk(batch).to(torch.float32)
            if POSTPROCESS_PRED:
                pred = apply_postprocessor_to_action_chunk(postprocessor, pred).to(torch.float32)

            pred = ensure_btd_action(pred)
            if getattr(cfg.policy, "use_delta_action", False):
                pred = delta_action_chunk_to_absolute(pred, raw_batch["observation.state"])

            pred = ensure_btd_action(pred).detach().cpu()
            gt = gt.detach().cpu()

            t = min(pred.shape[1], gt.shape[1])
            d = min(pred.shape[2], gt.shape[2])
            pred = pred[:, :t, :d]
            gt = gt[:, :t, :d]

            diff = pred - gt
            mse = float(diff.pow(2).mean())
            rmse = mse**0.5
            mae = float(diff.abs().mean())
            max_abs = float(diff.abs().max())

            metric_rows.append(
                {
                    "file": str(pt_path),
                    "index": file_idx,
                    "t": t,
                    "d": d,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "max_abs": max_abs,
                }
            )
            pred_chunks.append(pred[0].numpy().astype(np.float32))
            gt_chunks.append(gt[0].numpy().astype(np.float32))

            per_pt_plot_path = per_pt_plot_dir / f"{pt_path.stem}_policy_vs_pt_action.png"
            plot_one_pt_action_chunk(
                pred_chunk=pred_chunks[-1],
                gt_chunk=gt_chunks[-1],
                pt_name=pt_path.name,
                save_path=str(per_pt_plot_path),
                left_wrist_img=raw_batch.get("observation.images.left_wrist"),
                right_wrist_img=raw_batch.get("observation.images.right_wrist"),
            )

            logging.info(
                "[%04d/%04d] %s T=%d D=%d mse=%.8f rmse=%.8f mae=%.8f max_abs=%.8f plot=%s",
                file_idx + 1,
                len(files),
                pt_path.name,
                t,
                d,
                mse,
                rmse,
                mae,
                max_abs,
                per_pt_plot_path,
            )

    pred_np = np.stack(pred_chunks, axis=0)
    gt_np = np.stack(gt_chunks, axis=0)
    diff_np = pred_np - gt_np
    file_mse = np.asarray([row["mse"] for row in metric_rows], dtype=np.float64)

    npz_path = save_dir / "real_unpreprocessed_obs_policy_vs_pt_action.npz"
    csv_path = save_dir / "real_unpreprocessed_obs_policy_vs_pt_action_metrics.csv"
    timeline_path = save_dir / "real_unpreprocessed_obs_policy_vs_pt_action_first_step.png"
    mse_path = save_dir / "real_unpreprocessed_obs_policy_vs_pt_action_file_mse.png"

    np.savez(
        npz_path,
        files=np.asarray([str(p) for p in files]),
        pred=pred_np,
        gt=gt_np,
        diff=diff_np,
        apply_lerobot_preprocessor=APPLY_LEROBOT_PREPROCESSOR,
        postprocess_pred=POSTPROCESS_PRED,
        converted_delta_to_absolute=getattr(cfg.policy, "use_delta_action", False),
    )
    write_metrics_csv(metric_rows, str(csv_path))
    plot_first_step_timeline(pred_np, gt_np, str(timeline_path))
    plot_file_mse(file_mse, str(mse_path))

    logging.info("=" * 80)
    logging.info("Overall MSE=%.10f", float(np.mean(diff_np**2)))
    logging.info("Overall RMSE=%.10f", float(np.sqrt(np.mean(diff_np**2))))
    logging.info("Overall MAE=%.10f", float(np.mean(np.abs(diff_np))))
    logging.info("Overall max_abs=%.10f", float(np.max(np.abs(diff_np))))
    logging.info("Saved npz: %s", npz_path)
    logging.info("Saved csv: %s", csv_path)
    logging.info("Saved plot: %s", timeline_path)
    logging.info("Saved mse plot: %s", mse_path)
    logging.info("Saved per-pt plots: %s", per_pt_plot_dir)


if __name__ == "__main__":
    eval_real_preprocessed_obs()
