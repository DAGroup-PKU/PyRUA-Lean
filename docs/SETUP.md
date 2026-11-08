# Setup

Three kinds of environments are involved, each in its own Python: this repository's (the actor's
episode driver, SAM 3, the learn and eval coordinators), the simulator of each benchmark, and the
two agent CLIs. Paths go into `paths.env` (copied from [`paths.env.example`](../paths.env.example));
`source tools/env.sh` reads it before every command.

## 1. This repository's environment

Python 3.10 or later with a CUDA build of `torch` and the `torchvision` built for it (the paper
used torch 2.8.0 / CUDA 12.9), then:

```bash
python -m venv /path/to/envs/criteria-exposure && source /path/to/envs/criteria-exposure/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129   # your CUDA
pip install -e ".[dev]"            # numpy, pillow, httpx, mcp 1.x, uvicorn, pytest, ruff
pip install "sam3 @ git+https://github.com/RLinf/sam3.git@c1f6b1e83967c912f675d8975d437ccdc41e17d8" \
    einops pycocotools psutil "numpy>=1.26,<2" "opencv-python-headless<4.12" \
    msgpack-numpy "websockets>=13" pyyaml "h5py>=3.8"
```

`integrations/xpolicylab/PyRUA_Lean/install.sh` does the same inside an existing environment and
keeps its torch. Set `CE_PYTHON` in `paths.env` to this environment's `python`.

**SAM 3.** Download `sam3.pt` of the Hugging Face model `facebook/sam3` (3.4 GB;
`integrations/xpolicylab/PyRUA_Lean/download_assets.sh` fetches it) and set
`SAM3_CHECKPOINT_PATH`. The configs record its sha256; set `SAM3_SHA256` if yours differs.

Check: `bash tools/test_offline.sh "$CE_PYTHON"` (no simulator, GPU or model).

## 2. RoboDojo (main benchmark)

1. Clone [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) at commit `cb2a37d58dbb`
   with the submodules it records (`XPolicyLab` at `53c5da5a01bc`, `third_party/IsaacLab` at
   `afca7b09d60d`, `third_party/curobo` at `d17b54ce32cb`):
   ```bash
   git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git && cd RoboDojo
   git checkout cb2a37d58dbb
   git submodule update --init third_party/IsaacLab third_party/curobo XPolicyLab
   ```
2. Install the simulator environment with RoboDojo's own installer (conda env `RoboDojo`;
   Isaac Sim 5.1.0, Isaac Lab 0.54.3, cuRobo). Its submodule step runs
   `git submodule update --remote`, which moves the submodules to their newest commits, so put
   them back afterwards and rebuild cuRobo at its recorded commit:
   ```bash
   bash scripts/install.sh -i
   git submodule update --init third_party/IsaacLab third_party/curobo XPolicyLab
   conda run -n RoboDojo python -m pip uninstall -y nvidia-curobo
   bash scripts/install.sh --from curobo
   git submodule status        # 53c5da5a01bc XPolicyLab, afca7b09d60d IsaacLab, d17b54ce32cb curobo
   ```
   If `git submodule update` moved IsaacLab too (its recorded commit was no longer the newest),
   uninstall `isaaclab` as well as `nvidia-curobo` and run `bash scripts/install.sh --from isaaclab`
   instead of `--from curobo`.
3. Fetch the assets of the HF dataset `RoboDojo-Benchmark/RoboDojo` at revision
   `35efbc7dedfdbeeb6e95fb749bd885d73d483e41` into `Assets/` (39 GB). `init_assets.sh` clones
   the newest revision with depth 1 and then checks out `HF_REVISION`, which fails for an older
   one; with the clone already in place it fetches that revision instead, so clone first:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse \
       https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo .cache/robodojo_assets_repo
   HF_REVISION=35efbc7dedfdbeeb6e95fb749bd885d73d483e41 bash scripts/init_assets.sh
   conda run -n RoboDojo python utils/update_embodiment_config_path.py   # writes Assets/Robots/*/curobo.yml
   ```
   Later revisions change the assets (e.g. the handle frame of `mug/00014`). The last command
   fills the checkout's absolute path into the robots' `*_tmp.yml` templates; without its
   `curobo.yml` every episode fails at boot. Run it again if the checkout moves.
4. Set `ROBODOJO_SOURCE_ROOT` (the checkout) and `ROBODOJO_SIM_PYTHON` (the simulator
   environment's `python`). Nothing has to be linked into the checkout: each episode builds its
   own client directory from RoboDojo, XPolicyLab and `integrations/xpolicylab/PyRUA_Lean`.
5. Build the instruction-faithful criteria once (see [`criteria_v2/README.md`](../criteria_v2/README.md)):
   ```bash
   source tools/env.sh
   bash criteria_v2/build_source.sh v2-b9          # -> criteria_v2/build/v2-b9
   ROOT=criteria_v2/build/v2-b9 "$ROBODOJO_SIM_PYTHON" -m pytest -q criteria_v2/tests/test_offline.py
   ```
   and set `ROBODOJO_FAITHFUL_ROOT=$CE_ROOT/criteria_v2/build/v2-b9`.

A RoboDojo episode takes about 45 GB of GPU memory at its peak (simulator and SAM 3) and 10–60
minutes.

## 3. LIBERO-PRO (optional)

The `libero_rgb` robot runs LIBERO-PRO through [RPent](https://github.com/RLinf/RPent) (commit
`8f9d498` or `886b3b2`; its `robots/libero` and SAM 3 components are the same in both). Install
RPent's LIBERO-PRO environment as its README describes (Python 3.11, torch 2.7.1 / cu128,
`pip install -e ".[libero-pro]"`), add this repository's dependencies of section 1 to it, and
download the assets (about 615 MB). liberopro looks for them inside its package, which ships a
partial `assets/` directory of its own; `--link` replaces it by a link to the downloaded tree:

```bash
liberopro-download-assets --assets-dir <dir>
liberopro-download-assets --link <dir>      # site-packages/liberopro/liberopro/assets -> <dir>
```

No VLA checkpoint is needed: `no-vla` starts only the environment and SAM 3.

Set `RPENT_ROOT`, `LIBERO_PYTHON`, `LIBERO_PRO_ASSET_PATH` and `LIBERO_PRO_ROOT` (the
`liberopro/liberopro` directory of the package). An episode takes about 25 GB of GPU memory.

## 4. RoboCasa365 (optional)

The `robocasa_rgb` robot runs the kitchen in a Python of its own (3.10) that needs only
robocasa 1.0.1 and robosuite 1.5.2 (no RPent and no VLA). Pin MuJoCo to 3.3.1: later 3.3
releases (3.3.7, for one) no longer fill `qM` in `mj_crb`, the kitchen's controller then reads an
all-zero mass matrix and every arm motion fails with `Singular matrix`. `numpy==1.26.4` and
`opencv-python==4.11.0.86` are the versions of the paper's environment.

```bash
python3.10 -m venv /path/to/envs/robocasa && source /path/to/envs/robocasa/bin/activate
pip install "robosuite @ git+https://github.com/RLinf/robosuite.git@97cfbde4b68d8ec43dad20cf4747297866a6ca2e" \
    "rpent-robocasa365 @ git+https://github.com/RLinf/robocasa.git@2692d8fc5fd86708a1b2028dcbce892ec418a9e7" \
    "mujoco==3.3.1" "numpy==1.26.4" "opencv-python==4.11.0.86"
# deterministic scenes: robocasa orders some counter geoms by memory address, so the same seed
# gave a different kitchen from run to run; the patch keeps their first-seen order
SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
patch -d "$SITE" -p1 < eval_robocasa/patches/robocasa-counter-geom-order.patch
yes y | robocasa-download-assets --assets-path <dir> --no-macros     # ~23 GB; it asks to confirm
```

Set `ROBOCASA_SIM_PYTHON`, `ROBOCASA_SIM_PACKAGES` (its `site-packages`) and
`ROBOCASA_ASSETS_PATH`. An episode renders with EGL on the GPU it is given (about 12 GB) and runs
SAM 3 next to it.

## 5. Agent CLIs

- **Claude Code** for the learner (the paper used 2.1.303; it needs Node.js 22 or later):
  `npm install -g @anthropic-ai/claude-code@2.1.303`.
  The campaigns call it through `tools/claude-clean`, which keeps the user's own settings,
  CLAUDE.md files, memory and plugins out of the learner's session.
- **Codex CLI** only for `execution.runtime: codex` (the paper's version: 0.155.1):
  `npm install -g @openai/codex@0.155.1`, then [docs/MODELS.md](MODELS.md).

## 6. Paths and caches

`source tools/env.sh` sets `CE_ROOT`, puts `src/` on `PYTHONPATH`, reads `paths.env`
(`CE_PATHS_ENV` names another file) and keeps writable caches under `.cache/` unless
`XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME` or `LIBERO_CONFIG_PATH` are already set. It starts
nothing and sets no key. Config templates name these variables as `${NAME}` and are filled by
`python tools/render_config.py TEMPLATE OUT`, which stops at any unset name.
