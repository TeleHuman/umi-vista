# Post-Training

This directory contains the VISTA post-training and downstream fine-tuning code.

The main implementation lives in the vendored LeRobot tree:

```text
post_training/lerobot/
```

The VISTA policy is registered as an independent LeRobot policy type:

```text
--policy.type=vista
```

Use the repository-level [README](../README.md) for the public quick start, or [docs/installation_finetuning.md](../docs/installation_finetuning.md) for the validated fine-tuning smoke-test setup.

The most important launch wrapper is:

```text
post_training/lerobot/training_scripts/vista_finetune_smoke_test.sh
```
