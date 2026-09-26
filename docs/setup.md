# Setup

Everything PyRUA-Lean needs beyond this repository: the three RPent stacks
and the agent side.  All three stacks run
[RPent](https://github.com/RLinf/RPent) at commit `886b3b2`, each in its own
checkout and virtual environment (their extras conflict); `RPENT_REPO_ROOT`
names the checkout of the robot you run.  This package declares no
dependencies of its own (the RPent environments provide what it imports:
numpy, Pillow and, for `play`, mcp 1.x, uvicorn and httpx), so either
`pip install -e .` into the RPent environment or put `src/` on `PYTHONPATH`.

## The RPent stacks

**LIBERO-PRO** (Python 3.11).  `torch==2.7.1` / `torchvision==0.22.1` from
`https://download.pytorch.org/whl/cu128`, then `pip install -e ".[libero-pro]"`
in the RPent checkout.

- Assets: `liberopro-download-assets --assets-dir <dir>` (`LIBERO_PRO_ASSET_PATH`).
  liberopro looks for them inside its package, so replace
  `site-packages/liberopro/liberopro/assets` by a symlink to `<dir>`.
- Checkpoints: Pi0.5 `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT` from Hugging
  Face (`PI05_CHECKPOINT_PATH`; `optimizer.pt` is not needed) and SAM3
  `facebook/sam3` (`SAM3_CHECKPOINT_PATH`, the `sam3.pt` file).
- Environment: `LIBERO_TYPE=pro ROBOT_PLATFORM=LIBERO MUJOCO_GL=egl
  PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false`, a writable
  `LIBERO_CONFIG_PATH`, and `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
  `MKL_NUM_THREADS` (4 in the comparison).
- An episode with private servers boots Pi0.5 and SAM3 (~10 GB of GPU memory,
  ~100 s); attached to running servers (`play --vla-endpoint` /
  `--sam3-endpoint`) an episode needs ~1.5 GB and ~15 s.

**RoboTwin 2.0** (Python 3.11).  The same torch build, then
`pip install -e ".[rlinf]" rlinf-robotwin-runtime==0.1.1 rlinf-lingbotvla==0.1.1`,
`imageio-ffmpeg` (for `episode.mp4`), and cuRobo v0.7.8
(`https://github.com/NVlabs/curobo`, tag `v0.7.8`) built with
`pip install --no-build-isolation` against that torch (a CUDA 12.8 toolkit in
`CUDA_HOME`, `TORCH_CUDA_ARCH_LIST` set for your GPUs).

- Assets: `robotwin-download-assets --output <dir>` (the Hugging Face dataset
  `TianxingChen/RoboTwin2.0`, 16 GB; `ROBOTWIN_ASSETS_PATH`).
- Checkpoint: `RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500` at revision
  `e727b46cd220b66981ea4d2fd9ba84adc189e2cc` including `qwen_base/`
  (`LINGBOT_MODEL_PATH`).
- Rendering is headless SAPIEN on Vulkan: the NVIDIA ICD must be found; set
  `VK_ICD_FILENAMES` when it is not under `/usr/share/vulkan/icd.d`.
- An episode boots private env and VLA servers (~7 GB + ~9 GB of GPU memory).

**RoboCasa365** (Python 3.10, which RLDX-1 requires).  The same torch build,
then `.[rlinf]` plus the three sources of RPent's `[robocasa]` extra at
`RLinf/robocasa@2692d8fc5fd86708a1b2028dcbce892ec418a9e7`,
`RLinf/RLDX-1@ebcfd13df5177e4b3e574bdf5a3b427c7c4a1e8a` and
`RLinf/robosuite@97cfbde4b68d8ec43dad20cf4747297866a6ca2e`, installed with
`--constraint robots/robocasa/eval/target50-constraints.txt` (with uv, add
`--prerelease=if-necessary-or-explicit`: RPent's pyproject allows
pre-releases).

- Assets: `robocasa-download-assets --assets-path <dir> --no-macros`
  (`ROBOCASA_ASSETS_PATH`, 23 GB; it is the only way robocasa finds them).
- Checkpoint: `RLWRLD/RLDX-1-FT-RC365` at revision
  `587e9ecdcc5e7184fcc17f58713908edff5af041` (`RLDX_MODEL_PATH`), and the
  support files (`*.json *.txt *.jinja *.md *.png .gitattributes`, no weights)
  of `RLWRLD/RLDX-1-VLM` at `4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7` in the
  Hugging Face cache (`HF_HOME`).  Download them once more without
  `--revision`: RLDX resolves the backbone at `main`, and without `refs/main`
  in the cache the VLA server spends its whole boot budget retrying the Hub.
- Environment: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
  PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=ROBOCASA NO_ALBUMENTATIONS_UPDATE=1`, the
  Target50 execution values `RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999
  RLDX_ACTION_STEPS_PER_CHUNK=8`, and `RLDX_RESET_SEED`, `RLDX_ALLOW_RESET`
  and `CUDA_VISIBLE_DEVICES` unset (both servers pin their GPU from
  `--cuda-device`).
- Reproducible scenes: RoboCasa picks the counter region objects go on from a
  `list(set(...))` of XML elements (memory-address order), and some objects by
  Python's string hashing, so one (task, seed) gave different scenes from run to
  run.  Patch the installed file once (idempotent):
  `sed -i 's/list(set(valid_geoms))/list(dict.fromkeys(valid_geoms))/' "$(python -c 'import importlib.util as u, pathlib as p; print(p.Path(u.find_spec("robocasa").origin).parent / "models/fixtures/counter.py")')"`.
  PyRUA-Lean starts the env server with `PYTHONHASHSEED=0` itself; export it
  yourself when you run RPent's own agent.
- An episode boots private env and VLA servers (~3 GB + ~15-17 GB).

`runs/_empty-memory*/` are empty memory corpora for RPent's local memory
profile (`--memory-profile local --memory-dir runs/_empty-memory`, one per
robot); the RoboCasa365 host hands `runs/_empty-memory-robocasa` to RPent's
toolkit by default.

## The agent side

- Codex CLI 0.155.1 (`npm install -g @openai/codex@0.155.1`).  Point both arms
  at the real binary, not a wrapper that rewrites `CODEX_HOME`:
  `PYRUALEAN_CODEX_BIN` for the code arm, RPent's `CODEX_BIN` for the
  tool-calling arm.
- A clean Codex home for `play --codex-home` (or `PYRUALEAN_CODEX_HOME`): an
  empty directory with the credentials and a `config.toml` that sets
  `approval_policy = "never"` and `memories = false` under `[features]`.
  With `gpt-6-astra` Codex assumes a 272k window and compacts the context at
  a threshold derived from it; the comparison set
  `model_auto_compact_token_limit = 945000` (90% of the 1,050,000-token window
  its gateway served), so no episode was compacted.  `result.json` records the
  effective setting (`host.codex_context`) and any compactions
  (`generation.compactions`).
- No code mode for `gpt-6-astra` (guide.md, "Codex code mode"): with that model id,
  `play` and `generate` pass `-c model_catalog_json=<path>` with
  `src/pyrualean/codex-catalog-gpt-6-astra-fallback.json`, Codex 0.155.1's bundled
  catalog with only the `gpt-6-astra` entry set to the values of Codex's fallback
  metadata, so Codex's system prompt, built-in tools and settings are byte-identical
  to the comparison's (whose gateway id Codex did not know).  A one-entry catalog
  would also give direct tools but would
  keep Codex's Planning prompt sections.  The same line,
  `model_catalog_json = "<absolute path>"`, goes in the `config.toml` of the Codex
  home RPent uses.
- One OpenAI-compatible endpoint and key for both arms: `play` and `generate`
  take `--base-url` and `--api-key-env` (the name of the variable that holds
  the key), RPent's Codex planner reads `CODEX_BASE_URL` / `CODEX_API_KEY`.
  The model id is the endpoint's name for the planner model (`--model`; the
  comparison ran GPT-6 Astra, `openai/openai/gpt-6-astra` on its gateway).
  Without `--base-url` the code arm uses whatever provider the Codex home
  configures.
- Optional second runtime: `play --runtime claude` (or `PYRUALEAN_RUNTIME`)
  runs the code arm on Claude Code through the Claude Agent SDK
  (`claude-agent-sdk` in the RPent environment, a `claude` executable in
  `--claude-cli` or `PYRUALEAN_CLAUDE_CLI`; Opus 5.5 needs Claude Code
  2.1.280 or newer), with the model in `--model`.
