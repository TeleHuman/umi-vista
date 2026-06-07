# #!/usr/bin/env python
# import os
# import logging

# import torch

# from lerobot.configs import parser
# from lerobot.configs.train import TrainPipelineConfig
# from lerobot.datasets.factory import make_dataset
# from lerobot.datasets.sampler import EpisodeAwareSampler
# from lerobot.policies.factory import make_policy, make_pre_post_processors
# from lerobot.utils.constants import ACTION
# from lerobot.utils.utils import init_logging

# import math
# import torch
# import matplotlib.pyplot as plt


# def plot_action_chunk_gt_pred(
#     pred: torch.Tensor,
#     gt: torch.Tensor,
#     batch_idx: int = 0,
#     left_wrist_img: torch.Tensor | None = None,
#     right_wrist_img: torch.Tensor | None = None,
#     max_dims: int | None = 16,
#     ncols: int = 1,
#     figsize_per_row: float = 2.2,
#     save_path: str | None = None,
#     show: bool = False,
# ):
#     """
#     pred, gt: shape [B, T, D]
#     """
#     assert pred.ndim == 3 and gt.ndim == 3, f"Expect [B,T,D], got pred={pred.shape}, gt={gt.shape}"
#     assert pred.shape[:2] == gt.shape[:2], f"Batch/time mismatch: pred={pred.shape}, gt={gt.shape}"

#     b = batch_idx
#     t = min(pred.shape[1], gt.shape[1])
#     d = min(pred.shape[2], gt.shape[2])
#     if max_dims is not None:
#         d = min(d, max_dims)

#     pred_np = pred[b, :t, :d].detach().float().cpu().numpy()  # [T, d]
#     gt_np = gt[b, :t, :d].detach().float().cpu().numpy()      # [T, d]

#     def _tensor_to_image(img: torch.Tensor | None, idx: int):
#         if img is None:
#             return None

#         if img.ndim == 4:  # [B, C, H, W] or [B, H, W, C]
#             if idx >= img.shape[0]:
#                 return None
#             img = img[idx]
#         elif img.ndim == 5:  # [B, T, C, H, W]
#             if idx >= img.shape[0]:
#                 return None
#             img = img[idx, 0]

#         img = img.detach().float().cpu()
#         if img.ndim != 3:
#             return None

#         if img.shape[0] in (1, 3):
#             img = img.permute(1, 2, 0)

#         img_np = img.numpy()
#         if img_np.shape[-1] == 1:
#             img_np = img_np[..., 0]

#         min_v = float(img_np.min())
#         max_v = float(img_np.max())
#         if max_v > 1.0 or min_v < 0.0:
#             if max_v > min_v:
#                 img_np = (img_np - min_v) / (max_v - min_v)
#             else:
#                 img_np = img_np * 0.0

#         return img_np

#     left_img_np = _tensor_to_image(left_wrist_img, b)
#     right_img_np = _tensor_to_image(right_wrist_img, b)
#     has_wrist_images = left_img_np is not None and right_img_np is not None

#     nrows = math.ceil(d / ncols)

#     if has_wrist_images:
#         layout_rows = max(nrows, 2)
#         fig, axes = plt.subplots(
#             nrows=layout_rows,
#             ncols=ncols + 1,
#             figsize=(14, max(3, layout_rows * figsize_per_row)),
#             squeeze=False,
#             sharex="col",
#             gridspec_kw={"width_ratios": [1.2] + [3.0] * ncols},
#         )

#         axes[0][0].imshow(left_img_np)
#         axes[0][0].set_title("Left Wrist", fontsize=9)
#         axes[0][0].axis("off")

#         axes[1][0].imshow(right_img_np)
#         axes[1][0].set_title("Right Wrist", fontsize=9)
#         axes[1][0].axis("off")

#         for r in range(2, layout_rows):
#             axes[r][0].axis("off")
#     else:
#         layout_rows = nrows
#         fig, axes = plt.subplots(
#             nrows=nrows,
#             ncols=ncols,
#             figsize=(12, max(3, nrows * figsize_per_row)),
#             squeeze=False,
#             sharex=True,
#         )

#     x = range(t)
#     for i in range(d):
#         r, c = divmod(i, ncols)
#         ax = axes[r][c + 1] if has_wrist_images else axes[r][c]
#         ax.plot(x, gt_np[:, i], label="Ground Truth", linewidth=1.8)
#         ax.plot(x, pred_np[:, i], label="Prediction", linewidth=1.2, alpha=0.9)
#         ax.set_ylim(-1, 1)
#         ax.set_ylabel(f"Dim {i}")
#         ax.grid(True, alpha=0.25)
#         if i == 0:
#             ax.legend(loc="upper right", fontsize=8)

#     # Hide empty subplots
#     for j in range(d, layout_rows * ncols):
#         r, c = divmod(j, ncols)
#         target_ax = axes[r][c + 1] if has_wrist_images else axes[r][c]
#         target_ax.axis("off")

#     x_axis = axes[layout_rows - 1][1] if has_wrist_images else axes[nrows - 1][0]
#     x_axis.set_xlabel("Chunk Step")
#     fig.suptitle(f"Action Chunk GT vs Pred (batch={b}, T={t}, D={d})", y=0.995)
#     fig.tight_layout()

#     if save_path is not None:
#         save_dir = os.path.dirname(save_path)
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#         fig.savefig(save_path, dpi=180, bbox_inches="tight")

#     if show:
#         plt.show()

#     return fig


# @parser.wrap()
# def eval_mse_offline(cfg: TrainPipelineConfig):
#     cfg.validate()
#     init_logging()

#     logging.info("Creating dataset")
#     dataset = make_dataset(cfg)
#     dataset[0]

#     logging.info("Creating policy")
#     policy = make_policy(
#         cfg=cfg.policy,
#         ds_meta=dataset.meta,
#         rename_map=cfg.rename_map,
#     )

#     device = torch.device(cfg.policy.device)
#     policy.to(device)
#     policy.eval()

#     logging.info(f"policy.pretrained_path={cfg.policy.pretrained_path}")
#     logging.info(f"policy.pretrained_name_or_path={getattr(cfg.policy, 'pretrained_name_or_path', None)}")
#     # import ipdb;ipdb.set_trace()    
#     # === Build preprocessors aligned with training ===
#     processor_kwargs = {}
#     postprocessor_kwargs = {}

#     if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
#         processor_kwargs["dataset_stats"] = dataset.meta.stats

#     if cfg.policy.type == "sarm":
#         processor_kwargs["dataset_meta"] = dataset.meta

#     if cfg.policy.pretrained_path is not None:
#         processor_kwargs["preprocessor_overrides"] = {
#             "device_processor": {"device": device.type},
#             "normalizer_processor": {
#                 "stats": dataset.meta.stats,
#                 "features": {**policy.config.input_features, **policy.config.output_features},
#                 "norm_map": policy.config.normalization_mapping,
#             },
#             "rename_observations_processor": {"rename_map": cfg.rename_map},
#         }
#         postprocessor_kwargs["postprocessor_overrides"] = {
#             "unnormalizer_processor": {
#                 "stats": dataset.meta.stats,
#                 "features": policy.config.output_features,
#                 "norm_map": policy.config.normalization_mapping,
#             },
#         }

#     preprocessor, _ = make_pre_post_processors(
#         policy_cfg=cfg.policy,
#         pretrained_path=cfg.policy.pretrained_path,
#         **processor_kwargs,
#         **postprocessor_kwargs,
#     )

#     # === Build the dataloader aligned with training ===
#     if hasattr(cfg.policy, "drop_n_last_frames"):
#         shuffle = False
#         sampler = EpisodeAwareSampler(
#             dataset.meta.episodes["dataset_from_index"],
#             dataset.meta.episodes["dataset_to_index"],
#             episode_indices_to_use=dataset.episodes,
#             drop_n_last_frames=cfg.policy.drop_n_last_frames,
#             shuffle=True,
#         )
#     else:
#         shuffle = True
#         sampler = None
#     # import ipdb;ipdb.set_trace()
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         num_workers=2,
#         batch_size=1,
#         shuffle=shuffle and not cfg.dataset.streaming,
#         sampler=sampler,
#         pin_memory=device.type == "cuda",
#         drop_last=False,
#         prefetch_factor=2 if cfg.num_workers > 0 else None,
#     )

#     total_sq_error = 0.0
#     total_valid = 0.0
#     num_batches = 0
    
    

#     from tqdm import tqdm

#     with torch.no_grad():
#         pbar = tqdm(dataloader, desc="Evaluating", leave=False)

#         for batch_idx, batch in enumerate(pbar):
#             raw_batch = batch
#             batch = preprocessor(batch)
#             # import ipdb;ipdb.set_trace()    
#             pred = policy.predict_action_chunk(batch).to(torch.float32)
#             gt = batch[ACTION].to(pred.device, dtype=torch.float32)

#             t = min(pred.shape[1], gt.shape[1])
#             d = min(pred.shape[2], gt.shape[2])
#             pred = pred[:, :t, :d]
#             gt = gt[:, :t, :d]
#             plot_action_chunk_gt_pred(
#                 pred,
#                 gt,
#                 batch_idx=0,
#                 left_wrist_img=raw_batch.get("observation.images.left_wrist"),
#                 right_wrist_img=raw_batch.get("observation.images.right_wrist"),
#                 max_dims=16,
#                 ncols=1,
#                 save_path=f"/lumos-vePFS/teleai_manp/guolinzheng/umi_work_space_0324/RhodesLeRobot/pi05_abs_eval/compare_{batch_idx:06d}.png",
#             )
#             # import ipdb;ipdb.set_trace()
#             valid_mask = torch.isfinite(gt)

#             if "action_is_pad" in batch:
#                 pad = batch["action_is_pad"][:, :t].to(torch.bool).unsqueeze(-1)
#                 valid_mask = valid_mask & (~pad)

#             gt = torch.nan_to_num(gt, nan=0.0)

#             sq_err = (pred - gt).pow(2)
#             total_sq_error += (sq_err * valid_mask).sum().item()
#             total_valid += valid_mask.sum().item()
#             num_batches += 1

#             # 👉 Update metrics live
#             if total_valid > 0:
#                 mse = total_sq_error / total_valid
#                 pbar.set_postfix(mse=f"{mse:.6f}")

#     if total_valid == 0:
#         logging.warning("No valid action elements found. Cannot compute MSE.")
#         return

#     mse = total_sq_error / total_valid
#     rmse = mse ** 0.5
#     logging.info(f"[offline-eval] batches={num_batches}, valid_elems={int(total_valid)}")
#     logging.info(f"[offline-eval] MSE={mse:.8f}, RMSE={rmse:.8f}")


# if __name__ == "__main__":
#     eval_mse_offline()


##libero_umi
#!/usr/bin/env python
import os
import logging

import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging

import math
import torch
import matplotlib.pyplot as plt


def plot_action_chunk_gt_pred(
    pred: torch.Tensor,
    gt: torch.Tensor,
    batch_idx: int = 0,
    wrist_img: torch.Tensor | None = None,
    max_dims: int | None = None,
    ncols: int = 1,
    figsize_per_row: float = 2.2,
    save_path: str | None = None,
    show: bool = False,
):
    """
    pred, gt: shape [B, T, D]
    """
    assert pred.ndim == 3 and gt.ndim == 3, f"Expect [B,T,D], got pred={pred.shape}, gt={gt.shape}"
    assert pred.shape[:2] == gt.shape[:2], f"Batch/time mismatch: pred={pred.shape}, gt={gt.shape}"

    b = batch_idx
    t = min(pred.shape[1], gt.shape[1])
    d = min(pred.shape[2], gt.shape[2])
    if max_dims is not None:
        d = min(d, max_dims)

    pred_np = pred[b, :t, :d].detach().float().cpu().numpy()  # [T, d]
    gt_np = gt[b, :t, :d].detach().float().cpu().numpy()      # [T, d]

    def _tensor_to_image(img: torch.Tensor | None, idx: int):
        if img is None:
            return None

        if img.ndim == 4:  # [B, C, H, W] or [B, H, W, C]
            if idx >= img.shape[0]:
                return None
            img = img[idx]
        elif img.ndim == 5:  # [B, T, C, H, W]
            if idx >= img.shape[0]:
                return None
            img = img[idx, 0]

        img = img.detach().float().cpu()
        if img.ndim != 3:
            return None

        if img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)

        img_np = img.numpy()
        if img_np.shape[-1] == 1:
            img_np = img_np[..., 0]

        min_v = float(img_np.min())
        max_v = float(img_np.max())
        if max_v > 1.0 or min_v < 0.0:
            if max_v > min_v:
                img_np = (img_np - min_v) / (max_v - min_v)
            else:
                img_np = img_np * 0.0

        return img_np

    left_img_np = _tensor_to_image(wrist_img, b)
    # right_img_np = _tensor_to_image(right_wrist_img, b)
    has_wrist_images = left_img_np is not None

    nrows = math.ceil(d / ncols)

    if has_wrist_images:
        layout_rows = max(nrows, 2)
        fig, axes = plt.subplots(
            nrows=layout_rows,
            ncols=ncols + 1,
            figsize=(14, max(3, layout_rows * figsize_per_row)),
            squeeze=False,
            sharex="col",
            gridspec_kw={"width_ratios": [1.2] + [3.0] * ncols},
        )

        axes[0][0].imshow(left_img_np)
        axes[0][0].set_title("Wrist", fontsize=9)
        axes[0][0].axis("off")

        # axes[1][0].imshow(right_img_np)
        # axes[1][0].set_title("Right Wrist", fontsize=9)
        # axes[1][0].axis("off")

        for r in range(2, layout_rows):
            axes[r][0].axis("off")
    else:
        layout_rows = nrows
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(12, max(3, nrows * figsize_per_row)),
            squeeze=False,
            sharex=True,
        )

    x = range(t)
    for i in range(d):
        r, c = divmod(i, ncols)
        ax = axes[r][c + 1] if has_wrist_images else axes[r][c]
        ax.plot(x, gt_np[:, i], label="Ground Truth", linewidth=1.8)
        ax.plot(x, pred_np[:, i], label="Prediction", linewidth=1.2, alpha=0.9)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(f"Dim {i}")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Hide empty subplots
    for j in range(d, layout_rows * ncols):
        r, c = divmod(j, ncols)
        target_ax = axes[r][c + 1] if has_wrist_images else axes[r][c]
        target_ax.axis("off")

    x_axis = axes[layout_rows - 1][1] if has_wrist_images else axes[nrows - 1][0]
    x_axis.set_xlabel("Chunk Step")
    fig.suptitle(f"Action Chunk GT vs Pred (batch={b}, T={t}, D={d})", y=0.995)
    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    return fig


@parser.wrap()
def eval_mse_offline(cfg: TrainPipelineConfig):
    cfg.validate()
    init_logging()

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    if cfg.policy.use_delta_action:
        logging.info("Using delta action transform on dataset")
        dataset.load_delta_action_norm_stats()
    else:
        logging.info("Using absolute action transform on dataset")
        dataset.load_abs_action_norm_stats()
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

    logging.info(f"policy.pretrained_path={cfg.policy.pretrained_path}")
    logging.info(f"policy.pretrained_name_or_path={getattr(cfg.policy, 'pretrained_name_or_path', None)}")
    # import ipdb;ipdb.set_trace()    
    # === Build preprocessors aligned with training ===
    processor_kwargs = {}
    postprocessor_kwargs = {}

    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # === Build the dataloader aligned with training ===
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    # import ipdb;ipdb.set_trace()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    total_sq_error = 0.0
    total_valid = 0.0
    num_batches = 0
    
    

    from tqdm import tqdm

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)

        for batch_idx, batch in enumerate(pbar):
            raw_batch = batch
            batch = preprocessor(batch)
            # import ipdb;ipdb.set_trace()    
            pred = policy.predict_action_chunk(batch).to(torch.float32)
            gt = batch[ACTION].to(pred.device, dtype=torch.float32)

            t = min(pred.shape[1], gt.shape[1])
            d = min(pred.shape[2], gt.shape[2])
            pred = pred[:, :t, :d]
            gt = gt[:, :t, :d]
            plot_action_chunk_gt_pred(
                pred,
                gt,
                batch_idx=0,
                wrist_img=raw_batch.get("observation.images.wrist_image"),
                ncols=1,
                save_path=f"/data/guolinzheng/umi_work_space_0324/RhodesLeRobot/visual/pi05_libero_eval/compare_{batch_idx:06d}.png",
            )
            # import ipdb;ipdb.set_trace()
            valid_mask = torch.isfinite(gt)

            if "action_is_pad" in batch:
                pad = batch["action_is_pad"][:, :t].to(torch.bool).unsqueeze(-1)
                valid_mask = valid_mask & (~pad)

            gt = torch.nan_to_num(gt, nan=0.0)

            sq_err = (pred - gt).pow(2)
            total_sq_error += (sq_err * valid_mask).sum().item()
            total_valid += valid_mask.sum().item()
            num_batches += 1

            # 👉 Update metrics live
            if total_valid > 0:
                mse = total_sq_error / total_valid
                pbar.set_postfix(mse=f"{mse:.6f}")

    if total_valid == 0:
        logging.warning("No valid action elements found. Cannot compute MSE.")
        return

    mse = total_sq_error / total_valid
    rmse = mse ** 0.5
    logging.info(f"[offline-eval] batches={num_batches}, valid_elems={int(total_valid)}")
    logging.info(f"[offline-eval] MSE={mse:.8f}, RMSE={rmse:.8f}")


if __name__ == "__main__":
    eval_mse_offline()

##libero
#!/usr/bin/env python
# import os
# import logging

# import torch

# from lerobot.configs import parser
# from lerobot.configs.train import TrainPipelineConfig
# from lerobot.datasets.factory import make_dataset
# from lerobot.datasets.sampler import EpisodeAwareSampler
# from lerobot.policies.factory import make_policy, make_pre_post_processors
# from lerobot.utils.constants import ACTION
# from lerobot.utils.utils import init_logging

# import math
# import torch
# import matplotlib.pyplot as plt


# def plot_action_chunk_gt_pred(
#     pred: torch.Tensor,
#     gt: torch.Tensor,
#     batch_idx: int = 0,
#     left_wrist_img: torch.Tensor | None = None,
#     right_wrist_img: torch.Tensor | None = None,
#     max_dims: int | None = None,
#     ncols: int = 1,
#     figsize_per_row: float = 2.2,
#     save_path: str | None = None,
#     show: bool = False,
# ):
#     """
#     pred, gt: shape [B, T, D]
#     """
#     assert pred.ndim == 3 and gt.ndim == 3, f"Expect [B,T,D], got pred={pred.shape}, gt={gt.shape}"
#     assert pred.shape[:2] == gt.shape[:2], f"Batch/time mismatch: pred={pred.shape}, gt={gt.shape}"

#     b = batch_idx
#     t = min(pred.shape[1], gt.shape[1])
#     d = min(pred.shape[2], gt.shape[2])
#     if max_dims is not None:
#         d = min(d, max_dims)

#     pred_np = pred[b, :t, :d].detach().float().cpu().numpy()  # [T, d]
#     gt_np = gt[b, :t, :d].detach().float().cpu().numpy()      # [T, d]

#     def _tensor_to_image(img: torch.Tensor | None, idx: int):
#         if img is None:
#             return None

#         if img.ndim == 4:  # [B, C, H, W] or [B, H, W, C]
#             if idx >= img.shape[0]:
#                 return None
#             img = img[idx]
#         elif img.ndim == 5:  # [B, T, C, H, W]
#             if idx >= img.shape[0]:
#                 return None
#             img = img[idx, 0]

#         img = img.detach().float().cpu()
#         if img.ndim != 3:
#             return None

#         if img.shape[0] in (1, 3):
#             img = img.permute(1, 2, 0)

#         img_np = img.numpy()
#         if img_np.shape[-1] == 1:
#             img_np = img_np[..., 0]

#         min_v = float(img_np.min())
#         max_v = float(img_np.max())
#         if max_v > 1.0 or min_v < 0.0:
#             if max_v > min_v:
#                 img_np = (img_np - min_v) / (max_v - min_v)
#             else:
#                 img_np = img_np * 0.0

#         return img_np

#     left_img_np = _tensor_to_image(left_wrist_img, b)
#     right_img_np = _tensor_to_image(right_wrist_img, b)
#     has_wrist_images = left_img_np is not None and right_img_np is not None

#     nrows = math.ceil(d / ncols)

#     if has_wrist_images:
#         layout_rows = max(nrows, 2)
#         fig, axes = plt.subplots(
#             nrows=layout_rows,
#             ncols=ncols + 1,
#             figsize=(14, max(3, layout_rows * figsize_per_row)),
#             squeeze=False,
#             sharex="col",
#             gridspec_kw={"width_ratios": [1.2] + [3.0] * ncols},
#         )

#         axes[0][0].imshow(left_img_np)
#         axes[0][0].set_title("Wrist", fontsize=9)
#         axes[0][0].axis("off")

#         axes[1][0].imshow(right_img_np)
#         axes[1][0].set_title("Right Wrist", fontsize=9)
#         axes[1][0].axis("off")

#         for r in range(2, layout_rows):
#             axes[r][0].axis("off")
#     else:
#         layout_rows = nrows
#         fig, axes = plt.subplots(
#             nrows=nrows,
#             ncols=ncols,
#             figsize=(12, max(3, nrows * figsize_per_row)),
#             squeeze=False,
#             sharex=True,
#         )

#     x = range(t)
#     for i in range(d):
#         r, c = divmod(i, ncols)
#         ax = axes[r][c + 1] if has_wrist_images else axes[r][c]
#         ax.plot(x, gt_np[:, i], label="Ground Truth", linewidth=1.8)
#         ax.plot(x, pred_np[:, i], label="Prediction", linewidth=1.2, alpha=0.9)
#         ax.set_ylim(-1, 1)
#         ax.set_ylabel(f"Dim {i}")
#         ax.grid(True, alpha=0.25)
#         if i == 0:
#             ax.legend(loc="upper right", fontsize=8)

#     # Hide empty subplots
#     for j in range(d, layout_rows * ncols):
#         r, c = divmod(j, ncols)
#         target_ax = axes[r][c + 1] if has_wrist_images else axes[r][c]
#         target_ax.axis("off")

#     x_axis = axes[layout_rows - 1][1] if has_wrist_images else axes[nrows - 1][0]
#     x_axis.set_xlabel("Chunk Step")
#     fig.suptitle(f"Action Chunk GT vs Pred (batch={b}, T={t}, D={d})", y=0.995)
#     fig.tight_layout()

#     if save_path is not None:
#         save_dir = os.path.dirname(save_path)
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#         fig.savefig(save_path, dpi=180, bbox_inches="tight")

#     if show:
#         plt.show()

#     return fig


# @parser.wrap()
# def eval_mse_offline(cfg: TrainPipelineConfig):
#     cfg.validate()
#     init_logging()

#     logging.info("Creating dataset")
#     dataset = make_dataset(cfg)
#     if cfg.policy.use_delta_action:
#         logging.info("Using delta action transform on dataset")
#         dataset.load_delta_action_norm_stats()
#     else:
#         logging.info("Using absolute action transform on dataset")
#         dataset.load_abs_action_norm_stats()
#     dataset[0]

#     logging.info("Creating policy")
#     policy = make_policy(
#         cfg=cfg.policy,
#         ds_meta=dataset.meta,
#         rename_map=cfg.rename_map,
#     )

#     device = torch.device(cfg.policy.device)
#     policy.to(device)
#     policy.eval()

#     logging.info(f"policy.pretrained_path={cfg.policy.pretrained_path}")
#     logging.info(f"policy.pretrained_name_or_path={getattr(cfg.policy, 'pretrained_name_or_path', None)}")
#     # import ipdb;ipdb.set_trace()    
#     # === Build preprocessors aligned with training ===
#     processor_kwargs = {}
#     postprocessor_kwargs = {}

#     if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
#         processor_kwargs["dataset_stats"] = dataset.meta.stats

#     if cfg.policy.type == "sarm":
#         processor_kwargs["dataset_meta"] = dataset.meta

#     if cfg.policy.pretrained_path is not None:
#         processor_kwargs["preprocessor_overrides"] = {
#             "device_processor": {"device": device.type},
#             "normalizer_processor": {
#                 "stats": dataset.meta.stats,
#                 "features": {**policy.config.input_features, **policy.config.output_features},
#                 "norm_map": policy.config.normalization_mapping,
#             },
#             "rename_observations_processor": {"rename_map": cfg.rename_map},
#         }
#         postprocessor_kwargs["postprocessor_overrides"] = {
#             "unnormalizer_processor": {
#                 "stats": dataset.meta.stats,
#                 "features": policy.config.output_features,
#                 "norm_map": policy.config.normalization_mapping,
#             },
#         }

#     preprocessor, _ = make_pre_post_processors(
#         policy_cfg=cfg.policy,
#         pretrained_path=cfg.policy.pretrained_path,
#         **processor_kwargs,
#         **postprocessor_kwargs,
#     )

#     # === Build the dataloader aligned with training ===
#     if hasattr(cfg.policy, "drop_n_last_frames"):
#         shuffle = False
#         sampler = EpisodeAwareSampler(
#             dataset.meta.episodes["dataset_from_index"],
#             dataset.meta.episodes["dataset_to_index"],
#             episode_indices_to_use=dataset.episodes,
#             drop_n_last_frames=cfg.policy.drop_n_last_frames,
#             shuffle=True,
#         )
#     else:
#         shuffle = True
#         sampler = None
#     # import ipdb;ipdb.set_trace()
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         num_workers=2,
#         batch_size=1,
#         shuffle=False,
#         sampler=sampler,
#         pin_memory=device.type == "cuda",
#         drop_last=False,
#         prefetch_factor=2 if cfg.num_workers > 0 else None,
#     )

#     total_sq_error = 0.0
#     total_valid = 0.0
#     num_batches = 0
    
    

#     from tqdm import tqdm

#     with torch.no_grad():
#         pbar = tqdm(dataloader, desc="Evaluating", leave=False)

#         for batch_idx, batch in enumerate(pbar):
#             raw_batch = batch
#             batch = preprocessor(batch)
#             # import ipdb;ipdb.set_trace()    
#             pred = policy.predict_action_chunk(batch).to(torch.float32)
#             gt = batch[ACTION].to(pred.device, dtype=torch.float32)

#             t = min(pred.shape[1], gt.shape[1])
#             d = min(pred.shape[2], gt.shape[2])
#             pred = pred[:, :t, :d]
#             gt = gt[:, :t, :d]
#             plot_action_chunk_gt_pred(
#                 pred,
#                 gt,
#                 batch_idx=0,
#                 left_wrist_img=raw_batch.get("observation.images.left_wrist"),
#                 right_wrist_img=raw_batch.get("observation.images.right_wrist"),
#                 ncols=1,
#                 save_path=f"/data/guolinzheng/umi_work_space_0324/visual/compare_{batch_idx:06d}.png",
#             )
#             # import ipdb;ipdb.set_trace()
#             valid_mask = torch.isfinite(gt)

#             if "action_is_pad" in batch:
#                 pad = batch["action_is_pad"][:, :t].to(torch.bool).unsqueeze(-1)
#                 valid_mask = valid_mask & (~pad)

#             gt = torch.nan_to_num(gt, nan=0.0)

#             sq_err = (pred - gt).pow(2)
#             total_sq_error += (sq_err * valid_mask).sum().item()
#             total_valid += valid_mask.sum().item()
#             num_batches += 1

#             # 👉 Update metrics live
#             if total_valid > 0:
#                 mse = total_sq_error / total_valid
#                 pbar.set_postfix(mse=f"{mse:.6f}")

#     if total_valid == 0:
#         logging.warning("No valid action elements found. Cannot compute MSE.")
#         return

#     mse = total_sq_error / total_valid
#     rmse = mse ** 0.5
#     logging.info(f"[offline-eval] batches={num_batches}, valid_elems={int(total_valid)}")
#     logging.info(f"[offline-eval] MSE={mse:.8f}, RMSE={rmse:.8f}")


# if __name__ == "__main__":
#     eval_mse_offline()