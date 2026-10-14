# LIBERO-PRO

!!! note "Runs on RPent"

    LIBERO-PRO runs on [RPent](https://github.com/RLinf/RPent)'s LIBERO stack: RPent boots the simulator, the Pi0.5
    VLA and SAM3, and every `robo` call goes through RPent's toolkit. New robots are added through XPolicyLab
    ([Adding a Robot](adding-a-robot.md)).

LIBERO-PRO perturbs LIBERO's tabletop tasks: swapped objects and positions, rephrased or changed instructions.
The robot is a simulated Franka arm with RPent's scripted servos, the frozen Pi0.5 VLA skills and SAM3 perception.

## Overview

| | |
|---|---|
| **Robot** | Franka arm in MuJoCo / robosuite |
| **Action models** | Pi0.5 (`pi0_pick`, `pi0_doubled`); SAM3 for segmentation |
| **Agent runtimes** | `codex` (default), `claude`, `openai` ([Agent Configuration](../guide/agents.md)) |
| **Tasks** | The LIBERO-PRO suites RPent runs, e.g. `libero_object_swap`, `libero_spatial_swap` |
| **Hardware** | Linux, NVIDIA GPU, EGL; Python 3.11 |

### Observation and Action

| Item | Description |
|---|---|
| Observation | The `agentview` and `wrist` images, their per-pixel world coordinates, the end-effector pose, the gripper opening and the scene's object names. |
| Action | The agent's cells call `robo`; each call is one RPent tool (a scripted servo or a Pi0.5 skill) that steps the simulator. |
| Success | LIBERO's own task predicate. `robo.done` turns true once it fires, and every further motion is refused. |
| Task prompt | The suite's task instruction, a task card (the step-0 images and the initial gripper pose) and the robot's knowledge text. |

## Installation and Resources

RPent at commit `886b3b2`, in its own Python 3.11 environment, with PyRUA-Lean in the same environment:

```bash
git clone https://github.com/RLinf/RPent.git && cd RPent && git checkout 886b3b2
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[libero-pro]"
uv pip install -e /path/to/PyRUA-Lean
liberopro-download-assets --assets-dir /path/to/liberopro-assets
```

liberopro looks for its assets inside its own package: replace `site-packages/liberopro/liberopro/assets` by a
symlink to the assets directory. Then the environment:

```bash
export LIBERO_PRO_ASSET_PATH=/path/to/liberopro-assets
export LIBERO_TYPE=pro ROBOT_PLATFORM=LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export LIBERO_CONFIG_PATH=/path/to/a/writable/dir
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4      # as in the paper's runs
```

## VLA Configuration

Pi0.5 `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT` from Hugging Face (`optimizer.pt` is not needed):

```bash
hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT --local-dir /path/to/pi05
export PI05_CHECKPOINT_PATH=/path/to/pi05
```

## SAM3 Configuration

```bash
hf auth login                                                     # facebook/sam3 is gated: accept its licence first
hf download facebook/sam3 sam3.pt --local-dir /path/to/sam3
export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt
```

## Task Selection

| Option | Meaning |
|---|---|
| `--suite` | A LIBERO-PRO suite, e.g. `libero_object_swap`. |
| `--task` | Zero-based task index within the suite. |
| `--seed` | Environment seed (default `0`). |
| `--libero-type` | The installed LIBERO variant (default `pro`). |

## Run a Task

```bash
export RPENT_ROOT=/path/to/rpent
pyrualean play --arm cells --suite libero_spatial_swap --task 7 --seed 2 \
    --model gpt-6-astra --max-turns 40 --budget-s 7200 --out runs/demo --cuda-device 0
pyrualean summarize runs/demo
```

An episode with private servers boots Pi0.5 and SAM3 (~10 GB of GPU memory, ~100 s); attached to running servers
(`--vla-endpoint` / `--sam3-endpoint`) it needs ~1.5 GB and ~15 s. `--guides` gives the agent RPent's three LIBERO
operating guides ([Running Episodes](../guide/running.md#vla-policies-guides-and-shared-servers)).

## View Results

The run directory keeps the episode video (`episode.mp4`), every cell the agent ran (`workspace/`) and what each
printed (the runtime's event file, e.g. `codex_events.jsonl`), the ledger of primitive calls (`calls.jsonl`), the token
usage and LIBERO's verdict
(`result.json`), and RPent's own records (`states.json`, per-step images).

## Results

GPT-6 Astra, 40 tasks at five seeds (200 task instances), at most 40 LLM calls and two hours per episode. Against RPent's tool-calling agent
with the same planner and primitives; LLM calls, input tokens and cost per solved episode on the instances both
solved:

| | Success (%) | LLM calls | Input tokens | Cost |
|---|---|---|---|---|
| Tool calling (RPent) | 83.0 | 21.4 | 1.23M | $2.49 |
| **PyRUA-Lean** | **94.0** | **8.4** | **273k** | **$0.79** |

## What Runs Where

- **Env server** (RPent) owns the LIBERO MuJoCo environment and its EGL rendering.
- **Pi0.5 server** and **SAM3 server** (RPent) hold the VLA and the segmentation model.
- **PyRUA-Lean** boots them for the episode, builds RPent's toolkit and serves the agent's cells against `robo`.
- **Agent runtime** (Codex CLI, Claude Code or the `openai` loop) talks to the model.

## Primitives the Agent Can Call

Motion:

- `move_to(xyz, gripper, ...)`: servo the gripper to a world position, holding its orientation.
- `move_pose(xyz, target_pitch=..., target_yaw=..., ...)`: servo position and wrist orientation together.
- `rotate_wrist(target_yaw= / delta_yaw=)`, `rotate_pitch(target_pitch= / delta_pitch=)`: turn or tilt the gripper.
- `set_gripper(gripper, steps)`, `release()`: drive the gripper; open it and let go.
- `pi0_pick(prompt)`: let the frozen Pi0.5 VLA perform a closed-loop grasp.
- `pi0_doubled(prompt)`: let Pi0.5 perform a contact skill that is not a pick.

Perception and state:

- `segment(prompt | point=...)`: locate an object with SAM3 and return its world position.
- `back_project(row, col)`, `region_center(row_range, col_range)`: the world point under a pixel, the centre of a
  pixel window.
- `state()`, `image(camera)`, `world_map(camera)`, `show(camera)`, `camera_meta(camera)`, `artifact(name)`, `done`.

[How It Works](../getting-started/how-it-works.md#what-robo-returns) gives the rules every call follows.

??? note "API reference: what the agent reads, generated from `pyrualean.libero` when these docs are built"

    <!-- api-reference: libero -->
