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


# ##libero_umi
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
#     wrist_img: torch.Tensor | None = None,
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

#     left_img_np = _tensor_to_image(wrist_img, b)
#     # right_img_np = _tensor_to_image(right_wrist_img, b)
#     has_wrist_images = left_img_np is not None

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

#         # axes[1][0].imshow(right_img_np)
#         # axes[1][0].set_title("Right Wrist", fontsize=9)
#         # axes[1][0].axis("off")

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
#                 wrist_img=raw_batch.get("observation.images.wrist_image"),
#                 ncols=1,
#                 save_path=f"/gemini/space/users/glz/umi_work_space_0324/visual/pi05_libero_eval/compare_{batch_idx:06d}.png",
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
# import numpy as np

# def plot_episode_gt_trajectory(
#     dataset,
#     episode_id: int,
#     save_path: str,
#     max_dims: int | None = 16,
#     ncols: int = 1,
#     figsize_per_row: float = 2.2,
# ):
#     """Extract actions for all timesteps in one episode and plot the full trajectory."""
#     try:
#         start_idx = dataset.meta.episodes["dataset_from_index"][episode_id]
#         end_idx = dataset.meta.episodes["dataset_to_index"][episode_id]
#     except IndexError:
#         logging.error(f"Episode {episode_id} is out of range.")
#         return

#     logging.info(f"Extracting actions for Episode {episode_id} from {start_idx} to {end_idx - 1}...")

#     actions = []
#     for i in range(start_idx, end_idx):
#         act = dataset[i]["action"]
#         if hasattr(act, "detach"):
#             act = act.detach().cpu().numpy()
#         else:
#             act = np.asarray(act)
#         actions.append(act)

#     gt_np = np.stack(actions, axis=0)  # [T, D]
#     # ==== Additional compatibility handling ====
#     if gt_np.ndim == 3:
#         gt_np = gt_np[:, 0, :]  # If 3D, use timestep 0 of each chunk as the current-frame action
#     elif gt_np.ndim > 2:
#         # Flatten unexpected dimensions to 2D
#         gt_np = gt_np.reshape(gt_np.shape[0], -1) 
#     t, d = gt_np.shape
#     if max_dims is not None:
#         d = min(d, max_dims)

#     nrows = math.ceil(d / ncols)
#     fig, axes = plt.subplots(
#         nrows=nrows,
#         ncols=ncols,
#         figsize=(12, max(3, nrows * figsize_per_row)),
#         squeeze=False,
#         sharex=True,
#     )

#     x = range(t)
#     for i in range(d):
#         r, c = divmod(i, ncols)
#         ax = axes[r][c]
#         ax.plot(x, gt_np[:, i], label="Ground Truth Action", linewidth=1.8, color='blue')
#         ax.set_ylabel(f"Dim {i}")
#         ax.grid(True, alpha=0.25)
#         if i == 0:
#             ax.legend(loc="upper right", fontsize=8)

#     # Hide empty plots
#     for j in range(d, nrows * ncols):
#         r, c = divmod(j, ncols)
#         axes[r][c].axis("off")

#     axes[nrows - 1][0].set_xlabel("Time Step (Entire Episode)")
#     fig.suptitle(f"Episode {episode_id} GT Action Trajectory (T={t}, D={d})", y=0.995)
#     fig.tight_layout()

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     fig.savefig(save_path, dpi=180, bbox_inches="tight")
#     plt.close(fig)
#     logging.info(f"Saved episode trajectory plot to {save_path}")


# @parser.wrap()
# def eval_mse_offline(cfg: TrainPipelineConfig):
#     cfg.validate()
#     init_logging()

#     logging.info("Creating dataset")
#     dataset = make_dataset(cfg)
    
#     # === Plot the episode trajectory directly and exit ===
#     episode_to_plot = 100  # Set the episode id to extract here
#     save_plot_dir = "/data/guolinzheng/umi_work_space_0324/RhodesLeRobot/visual/episode_plots"
#     plot_save_path = os.path.join(save_plot_dir, f"episode_{episode_to_plot:04d}_action_trajectory.png")
    
#     plot_episode_gt_trajectory(dataset, episode_id=episode_to_plot, save_path=plot_save_path)
    
#     logging.info("Episode trajectory plotting completed. Exiting without model inference.")
#     return  # Return directly without loading the policy or running prediction evaluation



# if __name__ == "__main__":
#     eval_mse_offline()
import os
import logging
from copy import deepcopy

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


# =========================
# User settings
# =========================

EPISODE_TO_PLOT = 5

SAVE_DIR = (
    "/data/guolinzheng/umi_work_space_0324/"
    "RhodesLeRobot/visual/chunk_action_episode_5"
)

MAX_STEPS = None  # None = full episode

ACTION_DIM = 16

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


# =========================
# Small helpers
# =========================

def clone_batch(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = deepcopy(v)
    return out


def add_batch_dim(sample: dict) -> dict:
    """
    dataset[i] -> batch size 1.
    """
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v
    return batch


def ensure_btd_action(x: torch.Tensor) -> torch.Tensor:
    """
    Convert action tensor to [B, T, D].

    Accepts:
        [D]
        [B, D]
        [T, D]
        [B, T, D]
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.ndim == 1:
        # [D] -> [1, 1, D]
        x = x.unsqueeze(0).unsqueeze(0)

    elif x.ndim == 2:
        # In this script, after add_batch_dim, [1, D] means [B, D].
        # Treat it as [B, 1, D].
        x = x.unsqueeze(1)

    elif x.ndim == 3:
        pass

    else:
        raise ValueError(f"Unsupported action ndim={x.ndim}, shape={tuple(x.shape)}")

    return x


def ensure_state_bd(x: torch.Tensor) -> torch.Tensor:
    """
    Convert state to [B, D].
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.ndim == 1:
        x = x.unsqueeze(0)

    elif x.ndim == 2:
        pass

    elif x.ndim == 3:
        # If state is [B, T, D], use first frame.
        x = x[:, 0]

    else:
        raise ValueError(f"Unsupported state ndim={x.ndim}, shape={tuple(x.shape)}")

    return x


def find_state_key(batch: dict) -> str:
    """
    Find unnormalized absolute state key.
    Your dataset usually maps agent_pos -> observation.state.
    """
    candidates = [
        "observation.state",
        "observation.agent_pos",
        "agent_pos",
    ]

    for k in candidates:
        if k in batch:
            return k

    state_like = [k for k in batch.keys() if "state" in k or "agent_pos" in k]
    raise KeyError(
        "Cannot find state key for delta->absolute action transform. "
        f"Available state-like keys: {state_like}. "
        f"Available keys first 30: {list(batch.keys())[:30]}"
    )


def normalize_quat_np(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + eps)


def apply_postprocessor_to_action_chunk(postprocessor, pred: torch.Tensor) -> torch.Tensor:
    """
    postprocessor normally unnormalizes model output actions.

    Some processors accept [B,T,D] directly.
    If not, fall back to flatten [B*T,D].
    """
    try:
        out = postprocessor(pred)
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"postprocessor returned non-tensor type: {type(out)}")
        return out
    except Exception as exc:
        logging.warning(
            "postprocessor(pred[B,T,D]) failed; trying flattened fallback. "
            f"Original error: {repr(exc)}"
        )

    if pred.ndim != 3:
        raise ValueError(f"Expected pred [B,T,D] for fallback, got {tuple(pred.shape)}")

    b, t, d = pred.shape
    flat = pred.reshape(b * t, d)
    out = postprocessor(flat)

    if not isinstance(out, torch.Tensor):
        raise TypeError(f"postprocessor returned non-tensor type after flatten fallback: {type(out)}")

    return out.reshape(b, t, d)


def make_delta_action_mask(action_dim: int) -> torch.Tensor:
    """
    For 16D dual-arm action:
        left  gripper index = 7
        right gripper index = 15

    Gripper dimensions should not be transformed as delta pose.
    """
    mask = torch.ones(action_dim, dtype=torch.bool)

    if action_dim == 16:
        mask[7] = False
        mask[15] = False
    elif action_dim == 8:
        mask[7] = False
    else:
        raise ValueError(
            f"Unsupported action_dim={action_dim}. "
            "Expected 8D single-arm or 16D dual-arm action."
        )

    return mask


def delta_chunk_to_absolute_chunk(
    delta_actions: torch.Tensor,
    abs_state: torch.Tensor,
) -> torch.Tensor:
    """
    Convert unnormalized delta action chunk to unnormalized absolute action chunk.

    Input:
        delta_actions: [B, T, D]
        abs_state:     [B, D] or [B, T, D]

    Output:
        abs_actions:   [B, T, D]
    """
    delta_actions = ensure_btd_action(delta_actions).detach().cpu()
    abs_state = ensure_state_bd(abs_state).detach().cpu()

    action_dim = delta_actions.shape[-1]
    mask = make_delta_action_mask(action_dim)

    transform = AbsoluteActionTransform(mask)

    temp = {
        "observation.state": abs_state,
        "action": delta_actions,
    }

    out = transform(temp)
    return out["action"]


def to_abs_unnormalized_action_chunk(
    *,
    action_chunk_unnorm: torch.Tensor,
    raw_batch: dict,
    use_delta_action: bool,
) -> torch.Tensor:
    """
    Convert action chunk to absolute unnormalized action.

    If use_delta_action=False:
        action is already absolute unnormalized.

    If use_delta_action=True:
        action is unnormalized delta, convert delta -> absolute
        using raw unnormalized observation.state.
    """
    action_chunk_unnorm = ensure_btd_action(action_chunk_unnorm)

    if not use_delta_action:
        return action_chunk_unnorm.detach().cpu()

    state_key = find_state_key(raw_batch)
    abs_state = raw_batch[state_key]

    return delta_chunk_to_absolute_chunk(
        delta_actions=action_chunk_unnorm,
        abs_state=abs_state,
    )


def take_chunk_np(action_chunk: torch.Tensor) -> np.ndarray:
    """
    [B, T, D] -> [T, D], using B=0.
    """
    action_chunk = ensure_btd_action(action_chunk)
    x = action_chunk[0].detach().float().cpu().numpy()
    return x.astype(np.float32)


def infer_chunk_len_from_action(action_tensor: torch.Tensor) -> int:
    """
    Infer chunk length from [B,T,D].
    """
    action_tensor = ensure_btd_action(action_tensor)
    return int(action_tensor.shape[1])


def collect_gt_abs_chunk_from_dataset(
    *,
    dataset,
    start_dataset_idx: int,
    end_dataset_idx: int,
    target_len: int,
    use_delta_action: bool,
) -> torch.Tensor:
    """
    Fallback path when raw_batch[ACTION] only contains one-step action.

    Collect GT actions from subsequent dataset frames:
        dataset[start_dataset_idx : start_dataset_idx + target_len]

    Output:
        [1, valid_len, 16]
    """
    gt_list = []

    max_dataset_idx = min(end_dataset_idx, start_dataset_idx + target_len)

    for idx in range(start_dataset_idx, max_dataset_idx):
        sample = dataset[idx]
        raw_batch = add_batch_dim(sample)
        raw_batch = clone_batch(raw_batch)

        gt_unnorm = ensure_btd_action(raw_batch[ACTION]).to(torch.float32)

        # If a frame still returns a chunk, only take its first step here,
        # because this fallback collects one timeline step per dataset frame.
        gt_unnorm_step = gt_unnorm[:, :1, :]

        gt_abs_step = to_abs_unnormalized_action_chunk(
            action_chunk_unnorm=gt_unnorm_step,
            raw_batch=raw_batch,
            use_delta_action=use_delta_action,
        )

        gt_list.append(gt_abs_step)

    if len(gt_list) == 0:
        raise RuntimeError(
            f"Failed to collect GT chunk from dataset at idx={start_dataset_idx}"
        )

    return torch.cat(gt_list, dim=1)


# =========================
# Plotting
# =========================

def plot_episode_abs_action_compare(
    pred_abs: np.ndarray,
    gt_abs: np.ndarray,
    episode_id: int,
    save_path: str,
):
    """
    pred_abs, gt_abs: [T, 16]
    """
    assert pred_abs.ndim == 2 and pred_abs.shape[1] == 16, pred_abs.shape
    assert gt_abs.ndim == 2 and gt_abs.shape[1] == 16, gt_abs.shape

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    t = min(len(pred_abs), len(gt_abs))
    pred_abs = pred_abs[:t]
    gt_abs = gt_abs[:t]
    x = np.arange(t)

    fig, axes = plt.subplots(
        nrows=16,
        ncols=1,
        figsize=(14, 20),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for dim in range(16):
        ax = axes[dim]
        ax.plot(x, gt_abs[:, dim], label="GT abs unnorm", linewidth=1.7)
        ax.plot(x, pred_abs[:, dim], label="Policy abs unnorm", linewidth=1.2, alpha=0.9)
        ax.set_ylabel(ACTION_NAMES_16D[dim])
        ax.grid(True, alpha=0.25)

        if dim == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Episode step")
    fig.suptitle(
        f"Episode {episode_id} | Chunk-wise Absolute Unnormalized Action Compare | T={t}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved 16D absolute action comparison plot to {save_path}")


def plot_episode_abs_action_error_16d(
    pred_abs: np.ndarray,
    gt_abs: np.ndarray,
    episode_id: int,
    save_path: str,
):
    """
    16D dual-arm error plot:
        left_pos_l2
        left_quat_err = 1 - |dot|
        left_gripper_abs_err
        right_pos_l2
        right_quat_err = 1 - |dot|
        right_gripper_abs_err
    """
    assert pred_abs.ndim == 2 and pred_abs.shape[1] == 16, pred_abs.shape
    assert gt_abs.ndim == 2 and gt_abs.shape[1] == 16, gt_abs.shape

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    t = min(len(pred_abs), len(gt_abs))
    pred_abs = pred_abs[:t]
    gt_abs = gt_abs[:t]
    x = np.arange(t)

    left_pos_err = np.linalg.norm(pred_abs[:, 0:3] - gt_abs[:, 0:3], axis=1)

    pred_left_q = normalize_quat_np(pred_abs[:, 3:7])
    gt_left_q = normalize_quat_np(gt_abs[:, 3:7])
    left_quat_err = 1.0 - np.abs(np.sum(pred_left_q * gt_left_q, axis=1))

    left_gripper_err = np.abs(pred_abs[:, 7] - gt_abs[:, 7])

    right_pos_err = np.linalg.norm(pred_abs[:, 8:11] - gt_abs[:, 8:11], axis=1)

    pred_right_q = normalize_quat_np(pred_abs[:, 11:15])
    gt_right_q = normalize_quat_np(gt_abs[:, 11:15])
    right_quat_err = 1.0 - np.abs(np.sum(pred_right_q * gt_right_q, axis=1))

    right_gripper_err = np.abs(pred_abs[:, 15] - gt_abs[:, 15])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(x, left_pos_err, label=f"left_pos_l2 mean={left_pos_err.mean():.6f}")
    ax.plot(x, left_quat_err, label=f"left_quat_err mean={left_quat_err.mean():.6f}")
    ax.plot(x, left_gripper_err, label=f"left_gripper_err mean={left_gripper_err.mean():.6f}")

    ax.plot(x, right_pos_err, label=f"right_pos_l2 mean={right_pos_err.mean():.6f}")
    ax.plot(x, right_quat_err, label=f"right_quat_err mean={right_quat_err.mean():.6f}")
    ax.plot(x, right_gripper_err, label=f"right_gripper_err mean={right_gripper_err.mean():.6f}")

    ax.set_title(f"Episode {episode_id} | 16D Absolute Action Error")
    ax.set_xlabel("Episode step")
    ax.set_ylabel("error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved 16D absolute action error plot to {save_path}")


def print_episode_initial_state(dataset, episode_id: int):
    start_idx = int(dataset.meta.episodes["dataset_from_index"][episode_id])
    end_idx = int(dataset.meta.episodes["dataset_to_index"][episode_id])

    sample0 = dataset[start_idx]

    print("=" * 80)
    print(f"Episode {episode_id}")
    print(f"dataset index range: [{start_idx}, {end_idx})")
    print(f"num frames: {end_idx - start_idx}")
    print("sample0 keys:")

    for k in sample0.keys():
        v = sample0[k]
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}")

    state_key = find_state_key(sample0)
    state = sample0[state_key]

    if isinstance(state, torch.Tensor):
        state_np = state.detach().cpu().numpy()
    else:
        state_np = np.asarray(state)

    print("-" * 80)
    print(f"initial state key: {state_key}")
    print(f"initial state shape: {state_np.shape}")
    print(f"initial state dtype: {state_np.dtype}")
    print("initial state:")
    print(np.array2string(state_np, precision=8, suppress_small=False))

    if state_np.shape[-1] >= 16:
        print("-" * 80)
        print("parsed as 16D dual-arm state:")
        print(f"  left_pos      : {state_np[..., 0:3]}")
        print(f"  left_quat     : {state_np[..., 3:7]}")
        print(f"  left_gripper  : {state_np[..., 7]}")
        print(f"  right_pos     : {state_np[..., 8:11]}")
        print(f"  right_quat    : {state_np[..., 11:15]}")
        print(f"  right_gripper : {state_np[..., 15]}")

    elif state_np.shape[-1] >= 8:
        print("-" * 80)
        print("parsed as 8D single-arm state:")
        print(f"  pos     : {state_np[..., 0:3]}")
        print(f"  quat    : {state_np[..., 3:7]}")
        print(f"  gripper : {state_np[..., 7]}")

    print("=" * 80)

    return state_np


# =========================
# Core evaluation
# =========================

def eval_one_episode_abs_unnormalized_chunkwise(
    *,
    dataset,
    policy,
    preprocessor,
    postprocessor,
    episode_id: int,
    use_delta_action: bool,
    save_dir: str,
    max_steps: int | None = None,
):
    """
    Chunk-wise evaluation.

    For one episode:

        dataset_idx = start_idx

        while dataset_idx < end_idx:
            raw_batch = dataset[dataset_idx]
            batch = preprocessor(raw_batch)

            pred_norm = policy.predict_action_chunk(batch)
            pred_unnorm = postprocessor(pred_norm)
            pred_abs_chunk = delta/abs -> absolute unnormalized

            gt_abs_chunk:
                If raw_batch[ACTION] is already [1, T, 16], use it.
                If raw_batch[ACTION] is [1, 16], collect GT from future dataset frames.

            append whole chunk:
                pred_abs_chunk[0, :valid_len, :]
                gt_abs_chunk[0, :valid_len, :]

            dataset_idx += valid_len

    Output:
        pred_abs: [episode_T, 16]
        gt_abs:   [episode_T, 16]
    """
    start_idx = int(dataset.meta.episodes["dataset_from_index"][episode_id])
    end_idx = int(dataset.meta.episodes["dataset_to_index"][episode_id])

    if max_steps is not None:
        end_idx = min(end_idx, start_idx + max_steps)

    logging.info(
        f"Evaluating episode {episode_id} chunk-wise: "
        f"dataset idx {start_idx} -> {end_idx - 1}, "
        f"use_delta_action={use_delta_action}"
    )

    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    pred_abs_chunks = []
    gt_abs_chunks = []

    dataset_idx = start_idx
    local_chunk_id = 0

    with torch.no_grad():
        while dataset_idx < end_idx:
            sample = dataset[dataset_idx]

            raw_batch = add_batch_dim(sample)
            raw_batch = clone_batch(raw_batch)

            batch = clone_batch(raw_batch)
            batch = preprocessor(batch)

            pred_norm = policy.predict_action_chunk(batch).to(torch.float32)
            pred_unnorm = apply_postprocessor_to_action_chunk(postprocessor, pred_norm)
            pred_unnorm = ensure_btd_action(pred_unnorm).to(torch.float32)

            if pred_unnorm.shape[-1] != ACTION_DIM:
                raise ValueError(
                    f"Expected policy action dim={ACTION_DIM}, "
                    f"got pred_unnorm shape={tuple(pred_unnorm.shape)}"
                )

            pred_abs_chunk = to_abs_unnormalized_action_chunk(
                action_chunk_unnorm=pred_unnorm,
                raw_batch=raw_batch,
                use_delta_action=use_delta_action,
            )

            pred_chunk_len = infer_chunk_len_from_action(pred_abs_chunk)

            raw_action = ensure_btd_action(raw_batch[ACTION]).to(torch.float32)

            if raw_action.shape[-1] != ACTION_DIM:
                raise ValueError(
                    f"Expected GT action dim={ACTION_DIM}, "
                    f"got raw_batch[ACTION] shape={tuple(raw_action.shape)}"
                )

            # Case A:
            # dataset returns GT chunk directly, e.g. [1, T, 16]
            if raw_action.shape[1] > 1:
                gt_abs_chunk = to_abs_unnormalized_action_chunk(
                    action_chunk_unnorm=raw_action,
                    raw_batch=raw_batch,
                    use_delta_action=use_delta_action,
                )

            # Case B:
            # dataset returns only one-step GT, e.g. [1, 1, 16].
            # Collect future GT actions from dataset.
            else:
                gt_abs_chunk = collect_gt_abs_chunk_from_dataset(
                    dataset=dataset,
                    start_dataset_idx=dataset_idx,
                    end_dataset_idx=end_idx,
                    target_len=pred_chunk_len,
                    use_delta_action=use_delta_action,
                )

            pred_abs_chunk_np = take_chunk_np(pred_abs_chunk)
            gt_abs_chunk_np = take_chunk_np(gt_abs_chunk)

            available_len = min(
                pred_abs_chunk_np.shape[0],
                gt_abs_chunk_np.shape[0],
                end_idx - dataset_idx,
            )

            if available_len <= 0:
                raise RuntimeError(
                    f"Invalid available_len={available_len} at dataset_idx={dataset_idx}"
                )

            pred_abs_chunk_np = pred_abs_chunk_np[:available_len]
            gt_abs_chunk_np = gt_abs_chunk_np[:available_len]

            if pred_abs_chunk_np.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"Expected pred chunk dim={ACTION_DIM}, "
                    f"got {pred_abs_chunk_np.shape}"
                )

            if gt_abs_chunk_np.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"Expected GT chunk dim={ACTION_DIM}, "
                    f"got {gt_abs_chunk_np.shape}"
                )

            pred_abs_chunks.append(pred_abs_chunk_np)
            gt_abs_chunks.append(gt_abs_chunk_np)

            logging.info(
                f"[episode {episode_id}] chunk={local_chunk_id:04d} "
                f"dataset_idx={dataset_idx} "
                f"pred_chunk_len={pred_chunk_len} "
                f"gt_chunk_len={gt_abs_chunk_np.shape[0]} "
                f"valid_len={available_len}"
            )

            dataset_idx += available_len
            local_chunk_id += 1

    pred_abs = np.concatenate(pred_abs_chunks, axis=0).astype(np.float32)
    gt_abs = np.concatenate(gt_abs_chunks, axis=0).astype(np.float32)

    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(
        save_dir,
        f"episode_{episode_id:04d}_chunkwise_abs_unnorm_policy_vs_gt.npz",
    )
    plot_path = os.path.join(
        save_dir,
        f"episode_{episode_id:04d}_chunkwise_abs_unnorm_policy_vs_gt_16d.png",
    )
    err_path = os.path.join(
        save_dir,
        f"episode_{episode_id:04d}_chunkwise_abs_unnorm_policy_vs_gt_error_16d.png",
    )

    np.savez(
        npz_path,
        episode_id=episode_id,
        start_idx=start_idx,
        end_idx=end_idx,
        use_delta_action=use_delta_action,
        pred_abs_unnorm=pred_abs,
        gt_abs_unnorm=gt_abs,
    )

    plot_episode_abs_action_compare(
        pred_abs=pred_abs,
        gt_abs=gt_abs,
        episode_id=episode_id,
        save_path=plot_path,
    )

    plot_episode_abs_action_error_16d(
        pred_abs=pred_abs,
        gt_abs=gt_abs,
        episode_id=episode_id,
        save_path=err_path,
    )

    logging.info("[episode chunk-wise abs eval] saved:")
    logging.info(f"  npz : {npz_path}")
    logging.info(f"  plot: {plot_path}")
    logging.info(f"  err : {err_path}")

    return {
        "pred_abs_unnorm": pred_abs,
        "gt_abs_unnorm": gt_abs,
        "npz_path": npz_path,
        "plot_path": plot_path,
        "err_path": err_path,
    }


# =========================
# Main
# =========================

@parser.wrap()
def eval_episode_abs_action_offline(cfg: TrainPipelineConfig):
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

    # Force metadata / first sample init.
    dataset[0]
    print_episode_initial_state(dataset, EPISODE_TO_PLOT)

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
    logging.info(f"policy.use_delta_action={cfg.policy.use_delta_action}")

    # Align with training/eval_mse processor construction.
    processor_kwargs = {}
    postprocessor_kwargs = {}

    # if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
    #     processor_kwargs["dataset_stats"] = dataset.meta.stats

    # if cfg.policy.type == "sarm":
    #     processor_kwargs["dataset_meta"] = dataset.meta

    # if cfg.policy.pretrained_path is not None:
    #     processor_kwargs["preprocessor_overrides"] = {
    #         "device_processor": {"device": device.type},
    #         "normalizer_processor": {
    #             "stats": dataset.meta.stats,
    #             "features": {**policy.config.input_features, **policy.config.output_features},
    #             "norm_map": policy.config.normalization_mapping,
    #         },
    #         "rename_observations_processor": {"rename_map": cfg.rename_map},
    #     }
    #     postprocessor_kwargs["postprocessor_overrides"] = {
    #         "unnormalizer_processor": {
    #             "stats": dataset.meta.stats,
    #             "features": policy.config.output_features,
    #             "norm_map": policy.config.normalization_mapping,
    #         },
    #     }
    processor_kwargs["preprocessor_overrides"] = {
        "device_processor": {"device": device.type},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    eval_one_episode_abs_unnormalized_chunkwise(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        episode_id=EPISODE_TO_PLOT,
        use_delta_action=cfg.policy.use_delta_action,
        save_dir=SAVE_DIR,
        max_steps=MAX_STEPS,
    )

    logging.info("Done: chunk-wise episode absolute unnormalized policy-vs-GT action comparison.")


if __name__ == "__main__":
    eval_episode_abs_action_offline()