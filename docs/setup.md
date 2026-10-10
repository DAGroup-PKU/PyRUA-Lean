# Setup

Everything PyRUA-Lean needs beyond this repository: the RPent stacks and the
agent side.  The LIBERO-PRO, RoboTwin and RoboCasa365 stacks run
[RPent](https://github.com/RLinf/RPent) at commit `886b3b2`, the RoboDojo
stack at `068cd64` (RPent's RoboDojo backend arrived in `d08d712`), each in
its own checkout and virtual environment (their extras conflict);
`RPENT_REPO_ROOT` names the checkout of the robot you run.  This package declares no
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

**RoboDojo** (Python 3.11; experimental, as RPent labels it).  RPent
`068cd64` and its own guide, `docs/source-en/rst_source/simulators/robodojo.rst`:
`uv pip install -e ".[robodojo]" --extra-index-url https://pypi.nvidia.com
--override requirements/robodojo-override.txt` (Isaac Sim 5.1.0, torch 2.7.0;
add `--torch-backend=cu128` where the GPU needs a CUDA 12.8 PyTorch), then the
three IsaacLab packages editable (`--no-deps -e source/isaaclab -e
source/isaaclab_assets -e source/isaaclab_tasks`) from a checkout of
`yuechen0614/IsaacLab` at `afca7b09`.  Keep that checkout whole: the launcher
opens `apps/*.kit` by relative path and the kit file loads every extension
under `source/`.  With the NVIDIA index on the command line uv takes
`warp-lang` from it, and that index only publishes nightlies; the stack was
verified with the PyPI release, so pin it afterwards: `uv pip install
--index-url https://pypi.org/simple warp-lang==1.18.0`.

- Assets: `hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset
  --include 'Assets/**'`; `ROBODOJO_ASSETS_ROOT` is the directory that
  CONTAINS `Assets/`.  `Assets/Robots/<robot>/curobo.yml` must exist: RoboDojo
  generates it from `curobo_tmp.yml` by filling in `${ASSETS_PATH}`
  (`utils/update_embodiment_config_path.py` of its source tree).
- Checkpoints: RoboDojo's Pi_05 for the ARX-X5,
  `ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999` of the same dataset.
  Only `params/` (12.4 GB) and `assets/` are needed.  It is a JAX checkpoint:
  convert it with RLinf's `python -m
  rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python --checkpoint-dir
  <.../59999> --config-name pi05_aloha --output-path <out>` through a path
  that contains lowercase `pi05`, copy `59999/assets` to `<out>/assets`, and
  set `PI05_CHECKPOINT_PATH=<out>` and
  `PI05_NORM_STATS_PATH=<out>/assets/arx_x5_sim` (without the second variable
  openpi runs the policy un-normalised without saying so; the policy server's
  log must show "Loaded norm stats").  SAM3 as for LIBERO-PRO
  (`SAM3_CHECKPOINT_PATH`).
- Environment: `ROBODOJO_PLACEMENT_SETTLE_STEPS=1000` (RoboDojo's own default
  leaves objects unsettled) and `OMNI_KIT_ACCEPT_EULA=YES` (Kit otherwise
  waits for an answer on stdin and exits).
- Cells: `--task <name> --seed <layout id>` (`--suite arx_x5`); the task list
  is `task/RoboDojo/config/*.yml` of the source tree the runtime wheel ships.
- An episode boots private env, SAM3 and VLA servers (~10 GB + ~7 GB + ~8 GB
  of GPU memory, two to three minutes once Kit's shader cache is warm).
- The simulation host needs working egress, or a proxy in `http_proxy` /
  `https_proxy`: Kit's first start pulls two extensions from its online
  registry, and every scene load fetches one material and its textures from
  Omniverse's content bucket on S3.  On a route too slow for that the
  simulator server stays at "Resetting all environments" and the host gives
  up with "server did not become ready within 900s".  A model endpoint that
  must not use the proxy goes in `NO_PROXY`.
- The episode's native-action budget is the task's own `step_lim` (700 on
  `put_bottles_into_dustbin`), which the task card states.
- The host attaches to the simulator without the second reset of RPent's
  development-mode client: tasks with an articulated object do not survive
  it (`Object ... of type articulation not found in env 0`).
  `--connect-reset` (or `ROBODOJO_CONNECT_RESET=1`) brings it back.
- The benchmark's third arm.  Three tasks have an arm the benchmark drives
  itself (the opponent of `play_tic_tac_toe` and `make_kong`, the
  demonstrator of `imitate_sorting_sequence`).  RoboDojo's own evaluation
  client queues that arm's trajectory and replays it inside each action; the
  steppable client RPent's env server goes through (the bridge and collect
  client of `rlinf-robodojo-runtime`) does neither, so the opponent never
  moves.  The host therefore starts RPent's env server through
  `python -m pyrualean.hosts.robodojo_env_server`, which attaches the eval
  client's three steps to the runtime's classes in memory, only for a task
  that has such an arm, and then runs the server script unchanged; nothing
  installed is modified.  `ROBODOJO_SUPPORT_ARM=upstream` runs RPent's server
  as it is.
- What RoboDojo scores.  A task's `run_reward` registers checks that are
  latched: the task is solved at the first native action at which they all
  hold.  45 of the 54 tasks include `all_robot_back_to_origin` (both
  end-effectors within 0.15 m per axis and 20 degrees of their starting poses)
  and 31 include `is_all_gripper_open` (threshold 0.8 for most); only six
  instructions say so.  The knowledge files state both conditions.

The RoboDojo robot's primitives are this library's own, computed in the host
and registered on RPent's toolkit at boot (`robodojo_tools.register`), so
they are executed and recorded like RPent's tools; only what a program cannot
compose from the others is a primitive.

- `move_to(xyz, arm, gripper, pitch, yaw, roll, path)` drives the grasp point
  (0.145 m out from the end-effector link) to a position and an orientation.
  `hosts/robodojo_kinematics.py` carries the ARX-X5's URDF chain with both
  bases calibrated against the poses the simulator reports (its forward
  kinematics reproduces them) and solves the full pose by damped least
  squares; the motion follows a straight line of the grasp point, or swings
  the joints where the line cannot be followed, and is commanded as joint
  angles.  A pose out of reach moves nothing and the result names nearby
  reachable poses; a pose missed by at most 2 cm and 8 degrees is taken with a
  note.  The IK keeps to the joint limits the simulator enforces, measured by
  driving each joint alone (`examples/robodojo_joint_limits.py`), not the
  URDF's +-10 rad: joint5 stops at +-1.56 rad, joint3 at 0 and joint2 at
  -0.23 below the start.  Pointing straight down the grasp point then reaches
  only y = 0.0 at z = 0.80 and nowhere at z = 1.0, so tall objects are taken
  from the side; the knowledge file gives the reach.  The clearance check uses
  the table's real extent (`env_cfg/scene/default.yml`).  With an empty open
  gripper the half-turned pose (the fingers swapped) may be taken; holding
  something, never.
- `set_gripper(arm, command, opening=None)` repeats the command until the
  fingers stop, reports whether something stopped them apart and how wide it
  is, and opens part way with `opening`.  `KEEP` in a motion re-commands the
  last command, so a held object stays squeezed.
- `locate(text)` measures an object from its SAM3 mask and the depth image
  (centre, top, footprint, heading and the yaw of a top grasp across it);
  mask pixels on the robot's own arm are dropped.  No other call exposes the
  depth image.
- `home(arm)` lifts a low grasp point 6 cm, then commands the joint angles the
  episode started with.
- A twist is a `move_to` with another `roll`; a button press is a `move_to`
  1-2 cm past the surface (every command from 1 to 2.8 cm deep stops about
  7 mm down, where the button bottoms out).  The knowledge file says how.
- `pi0_pick` calls RoboDojo's Pi0.5 checkpoint, a policy for the whole task:
  interrupted by its grasp heuristic and called again, it carries on.  The
  checkpoint was trained on states and actions packed arm by arm
  (`[left_arm(6), left_gripper, right_arm(6), right_gripper]`); the policy
  wrapper RPent's `robodojo` extra installs packs them in another order, and
  with it the policy reaches but does not grasp.
- `examples/robodojo_check_primitives.py` exercises the primitives in the
  simulator without a planner (top grasps and a stack, side grasps, presses).

Measured with GPT-6 Astra (100 decisions, the task's own step budget, one
episode per cell, the 54 tasks at layouts 0-2):

| configuration | scored | solved |
|---|---|---|
| the earlier set (RPent's position-only tools and `home`), with the Pi0.5 policy | 158 | 11 (7 %) |
| the Pi0.5 policy alone | 157 | 12 (8 %) |
| this library's primitives, without the policy | 158 | 68 (43 %) |
| the same, the cells of the 18 tasks that sweep never solved re-run after the last fixes | 158 | 76 (48 %) |
| this library's primitives with the policy | 158 | 76 (48 %) |

The rows with the policy were measured with the policy wrapper feeding the
checkpoint its own channel order.  Cell by cell the policy adds nothing on
average to these primitives (76 against 75 on the 157 cells scored in both;
19 cells solved only with it, 18 only without).  Episodes that did not boot are not scored
(`imitate_sorting_sequence` never boots within 900 s).  37 of the 53 tasks
that boot were solved at least once.

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
