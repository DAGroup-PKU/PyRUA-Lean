# Quick Start

Install PyRUA-Lean, run one LIBERO-PRO episode and look at what the agent did. In this example GPT-6 Astra writes
the code through the Codex CLI, Pi0.5 does the grasping and SAM3 finds the objects.

For RoboDojo, follow [RoboDojo](../robots/robodojo.md) instead: it runs through XPolicyLab and needs none of the
RPent steps below.

## Requirements

- Linux, an NVIDIA GPU and a CUDA 12 driver; EGL for headless rendering.
- `uv`, `git`, and Node.js for the Codex CLI.
- Access to the model: an account the Codex CLI can log in with, or an OpenAI Responses-compatible endpoint and its
  key. Model calls are billed by the provider.

Each RPent robot runs in its own RPent checkout and environment: their extras conflict, and RoboCasa365 needs
Python 3.10.

## 1. Install RPent's LIBERO-PRO stack

LIBERO-PRO runs on [RPent](https://github.com/RLinf/RPent)'s stack, at the commit the paper used:

```bash
git clone https://github.com/RLinf/RPent.git && cd RPent && git checkout 886b3b2
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[libero-pro]"
export RPENT_ROOT=$PWD
cd ..
```

## 2. Install PyRUA-Lean

Into the same environment:

```bash
git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git
uv pip install -e ./PyRUA-Lean
pyrualean api             # the robot API the agent will see; no simulator needed
```

PyRUA-Lean declares no dependencies of its own: what it imports (numpy, Pillow and, for `play`, `mcp` 1.x, uvicorn
and httpx) comes with RPent's environment.

## 3. Download Assets and Models

```bash
liberopro-download-assets --assets-dir /path/to/liberopro-assets
hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT --exclude optimizer.pt --local-dir /path/to/pi05
hf auth login                                                     # facebook/sam3 is gated: accept its licence first
hf download facebook/sam3 sam3.pt --local-dir /path/to/sam3

export LIBERO_PRO_ASSET_PATH=/path/to/liberopro-assets
export PI05_CHECKPOINT_PATH=/path/to/pi05
export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt
export LIBERO_TYPE=pro ROBOT_PLATFORM=LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export LIBERO_CONFIG_PATH=/path/to/a/writable/dir
```

liberopro looks for its assets inside its own package: replace `site-packages/liberopro/liberopro/assets` by a
symlink to the assets directory. [LIBERO-PRO](../robots/libero.md) has the details.

## 4. Configure the Agent

```bash
npm install -g @openai/codex@0.155.1
codex login                # or an endpoint: --base-url https://host/v1 --api-key-env MY_KEY
```

The Codex CLI speaks the OpenAI Responses API; a server that only serves chat completions (SGLang, vLLM) is used with
`--runtime openai` instead. [Agent Configuration](../guide/agents.md) covers the runtimes.

## 5. Run Your First Episode

Task 7 of `libero_spatial_swap` at seed 2, with at most 40 code cells and two hours:

```bash
pyrualean play --arm cells --suite libero_spatial_swap --task 7 --seed 2 \
    --model gpt-6-astra --max-turns 40 --budget-s 7200 --out runs/demo --cuda-device 0
```

Each episode boots its own simulator, Pi0.5 and SAM3 servers (about 100 s and 10 GB of GPU memory) unless it is
attached to running ones ([Running Episodes](../guide/running.md#vla-policies-guides-and-shared-servers)).

## 6. Look at the Result

```bash
pyrualean summarize runs/demo
```

`runs/demo` keeps the episode video, every cell the agent ran and what it printed, the primitives it called, the
token usage and LIBERO's verdict ([Output Files](../guide/outputs.md)).
