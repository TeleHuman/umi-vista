# LIBERO-UMI Evaluation

This directory contains the project-owned LIBERO-UMI evaluation runner for VISTA checkpoints.

The runner evaluates each LIBERO task in the selected suites. `N_EPISODES_PER_TASK` maps directly to `--eval.n_episodes`; the default is 50 episodes for every task in each suite.

```bash
bash simulation_evaluation/libero_umi/run_libero_umi_eval.sh
```

For a quick check, explicitly override the default:

```bash
N_EPISODES_PER_TASK=5 bash simulation_evaluation/libero_umi/run_libero_umi_eval.sh
```

Important environment variables:

- `POLICY_PATH`: pretrained policy directory to evaluate.
- `OUTPUT_ROOT`: directory for logs, per-suite outputs, and the summary TSV.
- `N_EPISODES_PER_TASK`: number of episodes per task.
- `GPU_ID`: visible GPU id for evaluation.
- `SUITES`: space-separated LIBERO suites.

The runner creates `LIBERO_CONFIG_PATH/config.yaml` when it is missing so LIBERO does not stop for an interactive dataset prompt. The default dataset/cache directory is `/data_ysy/app/codex_space/share/libero_datasets/datasets` and can be overridden with `LIBERO_DATASETS_DIR`.

The default LIBERO package paths are derived from `ENV_DIR`:

- `LIBERO_PACKAGE_DIR`: `${ENV_DIR}/lib/python3.10/site-packages/libero/libero`
- `LIBERO_ASSETS_DIR`: `${LIBERO_PACKAGE_DIR}/assets`
- `LIBERO_BDDL_DIR`: `${LIBERO_PACKAGE_DIR}/bddl_files`
- `LIBERO_INIT_STATES_DIR`: `${LIBERO_PACKAGE_DIR}/init_files`

See `docs/installation_libero_umi.md` for manual `config.yaml` setup and optional LIBERO asset or official dataset download commands.

For headless MuJoCo rendering, the runner prepends `GLVND_RUNTIME_DIR` to `LD_LIBRARY_PATH`. The default runtime directory is `/data_ysy/app/codex_space/share/glvnd_runtime`.
