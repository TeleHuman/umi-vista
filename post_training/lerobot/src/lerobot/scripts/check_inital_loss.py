#!/usr/bin/env python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from typing import Dict

import torch
from accelerate import Accelerator
import matplotlib.pyplot as plt

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import init_logging

def get_avg_loss(cfg: TrainPipelineConfig, pretrained_path: str, num_batches: int) -> float:
    # Override with the checkpoint path currently being tested
    cfg.policy.pretrained_path = pretrained_path

    accelerator = Accelerator()
    init_logging(accelerator=accelerator)
    device = accelerator.device

    logging.info(f"Creating dataset from: {cfg.dataset.repo_id}")
    dataset = make_dataset(cfg)
    if cfg.policy.use_delta_action:
        logging.info("Using delta action transform on dataset")
        dataset.load_delta_action_norm_stats()
    else:
        logging.info("Using absolute action transform on dataset")
        dataset.load_abs_action_norm_stats()
    
    logging.info(f"Creating policy of type '{cfg.policy.type}' from: {pretrained_path}")
    # Core path: create the policy from cfg and dataset information.
    # make_policy loads and rebuilds model weights and internal config from pretrained_path.
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    processor_kwargs = {}
    if pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        if cfg.rename_map:
             processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
                "rename_map": cfg.rename_map
            }

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=pretrained_path,
        **processor_kwargs,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    policy, dataloader = accelerator.prepare(policy, dataloader)
    dl_iter = cycle(dataloader)

    policy.eval()

    total_loss = 0.0
    with torch.no_grad():
        for i in range(num_batches):
            try:
                batch = next(dl_iter)
                import ipdb; ipdb.set_trace()
                batch = preprocessor(batch)
                import ipdb; ipdb.set_trace()
                loss, _ = policy.forward(batch)
                total_loss += loss.item()
                logging.info(f"Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")
            except StopIteration:
                logging.warning(f"Dataloader exhausted after {i} batches.")
                num_batches = i
                break
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    logging.info(f"Average loss over {num_batches} batches: {avg_loss:.4f}")
    return avg_loss

def plot_losses(checkpoint_losses: Dict[str, float], output_dir: str):
    if not checkpoint_losses:
        logging.warning("No losses to plot.")
        return

    aliases = list(checkpoint_losses.keys())
    losses = list(checkpoint_losses.values())
    
    plt.figure(figsize=(12, 7))
    plt.plot(aliases, losses, marker='o', linestyle='-')
    
    plt.title('Initial Loss Comparison of Checkpoints')
    plt.xlabel('Checkpoint Alias')
    plt.ylabel('Average Initial Loss')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'initial_loss_comparison.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    logging.info(f"Plot saved to {output_path}")

@parser.wrap()
def main(cfg: TrainPipelineConfig):
    # ==================================================================
    # 1. Define all checkpoint paths to test
    # ==================================================================
    checkpoints = {
        # "pi05_base": "/data/guolinzheng/app/download/model/pi05_base_pytorch_newest",
        # "umi_40w": "/data/guolinzheng/app/download/model/umi_0312_40w",
        "pre2_2w": "/data/guolinzheng/umi_work_space_0324/ckpt/umipi_pre2/checkpoints/020000/pretrained_model",
        "pre2_8w": "/data/guolinzheng/umi_work_space_0324/ckpt/umipi_pre2/checkpoints/040000/pretrained_model",
        "pre2_12w": "/data/guolinzheng/umi_work_space_0324/ckpt/umipi_pre2/checkpoints/120000/pretrained_model",
        "pre2_16w": "/data/guolinzheng/umi_work_space_0324/ckpt/umipi_pre2/checkpoints/160000/pretrained_model",
        
        # "trained_ckpt_20k": "...",
    }

    num_batches_to_check = 100
    plot_output_dir = "/data/guolinzheng/umi_work_space_0324/RhodesLeRobot/visual"

    checkpoint_losses = {}

    for alias, ckpt_path in checkpoints.items():
        print("\n" + "="*50)
        print(f"Processing checkpoint: {alias} ({ckpt_path})")
        
        try:
            # Pass one shared cfg and only change the loaded pretrained path
            avg_loss = get_avg_loss(cfg, ckpt_path, num_batches_to_check)
            checkpoint_losses[alias] = avg_loss
        except Exception as e:
            logging.error(f"Failed to process checkpoint {alias}: {e}", exc_info=True)
            checkpoint_losses[alias] = float('nan')

    print("\n" + "="*50)
    print("Final Results:")
    for alias, loss in checkpoint_losses.items():
        print(f"  - {alias}: {loss:.4f}")
    print("="*50)

    plot_losses(checkpoint_losses, plot_output_dir)

if __name__ == "__main__":
    main()