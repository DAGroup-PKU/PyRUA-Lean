# Vendored trainer (modified openpi)

This directory is a modified copy of part of [openpi](https://github.com/Physical-Intelligence/openpi) by Physical
Intelligence, under the Apache License 2.0 (`LICENSE`). `LICENSE_GEMMA.txt` applies to PaliGemma weights, which are not
included. To launch training, use `tools/launch_pi05_training.py` (`docs/reference/policy-training.md`).

## What is here

The JAX PI0.5 training path: `scripts/train.py` and what it imports. Upstream's examples, policy serving code, websocket
client, PI0-FAST models, the `transformers_replace` patch and documentation are not included.

Two parts stay only because hash-bound or base modules import them at load time, and no shipped config uses them:

- `src/openpi/models_pytorch/`, imported by `models/model.py`. It cannot run from this tree, because it needs the
  `transformers_replace` patch.
- `src/openpi/training/droid_rlds_dataset.py`, imported by `training/config.py` and `training/data_loader.py`.

## Changes from upstream

- **Piper training:** the `pi05_piper_dual` and `pi05_piper_dual_eef_xyz3d` configs and their data configs in
  `training/config.py`; the policies `policies/piper_policy.py` and `policies/piper_eef_xyz3d_policy.py`.
- **DataFarm path bindings:** `training/config.py` imports `datafarm.runtime.runtime_path` to expand
  `${OPENPI_CACHE}`, so `src/` of this repository must be importable when it loads.
- **Training metrics:** per-dimension loss, a JSONL metrics log and a fixed-batch `eval_step` in `scripts/train.py`.
- **Noise-prior research options:** a Gaussian-process prior (`use_gp_prior`, `gp_kernel_type`, `gp_length_scale`),
  optimal-transport coupling (`use_ot_coupling`, `ot_reg`, `ot_num_iters`) and history-conditioned noise (`use_gpr`,
  `gpr_past_horizon`). They are in `models/pi0_config.py`, `models/pi0.py`, `models/gp_utils.py`, `models/ot_utils.py`,
  `models_pytorch/gp_utils_pytorch.py`, and in `prev_action_sequence_keys` and `MergePrevActionKeys` on the data side.
  All default to off, and no shipped config or launcher sets them.
- **Extra observation fields:** `episode_index`, `frame_index`, `prev_state`, `prev_action`, `action_horizon`,
  `actions_during_latency` and `gpr_prev_action` in `models/model.py`, with matching code in `transforms.py`
  (including delta-pose handling), `training/data_loader.py`, `models/pi0.py` and `models_pytorch/pi0_pytorch.py`.
- **Dependencies:** Python below 3.12, torch 2.9.1, and no `gym-aloha` or `ml_collections`.

Smaller edits exist elsewhere, for example in `models/tokenizer.py` and `shared/download.py`. Compare with upstream for
the full difference.

`configs/training/template.json` pins the SHA-256 of `scripts/train.py`, `models/pi0.py`, `training/config.py`,
`training/data_loader.py` and `policies/piper_eef_xyz3d_policy.py`. A launch refuses a tree where they differ, so
changing one of those files also means updating the pins.
