# RoboCasa365

!!! note "Runs on RPent"

    RoboCasa365 runs on [RPent](https://github.com/RLinf/RPent)'s RoboCasa stack: RPent boots the kitchen
    environment and the RLDX-1 VLA, and every `robo` call goes through RPent's toolkit. New robots are added
    through XPolicyLab ([Adding a Robot](adding-a-robot.md)).

RoboCasa365 is a household benchmark of kitchen scenes. The robot is a mobile manipulator: a mobile base and an
arm, with navigation, scripted arm motions and the frozen RLDX-1 VLA.

## Overview

| | |
|---|---|
| **Robot** | PandaOmron mobile manipulator in RoboCasa's kitchens (MuJoCo / robosuite) |
| **Action models** | RLDX-1 (`rldx_skill`, `rldx_arm`) |
| **Agent runtimes** | `codex` (default), `claude`, `openai` ([Agent Configuration](../guide/agents.md)) |
| **Tasks** | The Target50 set: 18 atomic and 32 composite tasks |
| **Hardware** | Linux, NVIDIA GPU, EGL; Python 3.10 (RLDX-1 requires it) |

### Observation and Action

| Item | Description |
|---|---|
| Observation | The `agentview`, `navview` and `wrist` images (256 × 256), their depth and per-pixel world coordinates, and the state of the base, the arm and the episode. |
| Action | The agent's cells call `robo`; each call is one RPent tool (navigation, a base or arm motion, a gripper command, an RLDX-1 skill) that steps the simulator. |
| Success | RoboCasa's own task predicate. `robo.done` turns true once it fires, and every further motion is refused; `success_criteria()` returns the predicate's source, as RPent's agent can read it too. |
| Task prompt | The task instruction, a task card (the step-0 images and the robot's initial pose) and the robot's knowledge text. |

## Installation and Resources

RPent at commit `886b3b2`, in its own environment with Python 3.10 (RLDX-1 requires it) and the same torch build as
for LIBERO-PRO, with PyRUA-Lean in the same environment. RPent's `[robocasa]` extra follows three moving branches;
install `.[rlinf]` and those three sources at the commits the paper used instead, with the Target50 constraints
(RPent's pyproject allows pre-releases):

```bash
uv pip install -e ".[rlinf]" \
    "rpent-robocasa365 @ git+https://github.com/RLinf/robocasa.git@2692d8fc5fd86708a1b2028dcbce892ec418a9e7" \
    "rlinf-rldx @ git+https://github.com/RLinf/RLDX-1.git@ebcfd13df5177e4b3e574bdf5a3b427c7c4a1e8a" \
    "robosuite @ git+https://github.com/RLinf/robosuite.git@97cfbde4b68d8ec43dad20cf4747297866a6ca2e" \
    --constraint robots/robocasa/eval/target50-constraints.txt --prerelease=if-necessary-or-explicit
uv pip install -e /path/to/PyRUA-Lean
robocasa-download-assets --assets-path /path/to/robocasa-assets --no-macros      # 23 GB
```

The environment (both servers pin their GPU from `--cuda-device`, so leave `CUDA_VISIBLE_DEVICES` unset):

```bash
export ROBOCASA_ASSETS_PATH=/path/to/robocasa-assets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=ROBOCASA NO_ALBUMENTATIONS_UPDATE=1
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8     # the Target50 execution values
unset RLDX_RESET_SEED RLDX_ALLOW_RESET CUDA_VISIBLE_DEVICES
```

!!! warning "Reproducible scenes"

    Unpatched, RoboCasa can build a different scene for the same task and seed from run to run. Patch the installed
    file once (idempotent):

    ```bash
    sed -i 's/list(set(valid_geoms))/list(dict.fromkeys(valid_geoms))/' \
        "$(python -c 'import importlib.util as u, pathlib as p; print(p.Path(u.find_spec("robocasa").origin).parent / "models/fixtures/counter.py")')"
    ```

    PyRUA-Lean also starts the env server with `PYTHONHASHSEED=0`.

## VLA Configuration

RLDX-1 at a fixed revision, and the support files (no weights) of its backbone in the Hugging Face cache, at a fixed
revision and once more at `main` (RLDX resolves the backbone at `main`; without `refs/main` in the cache the VLA server
spends its whole boot budget retrying the Hub):

```bash
hf download RLWRLD/RLDX-1-FT-RC365 --revision 587e9ecdcc5e7184fcc17f58713908edff5af041 --local-dir /path/to/rldx
export RLDX_MODEL_PATH=/path/to/rldx
hf download RLWRLD/RLDX-1-VLM --revision 4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7 \
    --include '*.json' '*.txt' '*.jinja' '*.md' '*.png' '.gitattributes'
hf download RLWRLD/RLDX-1-VLM --include '*.json' '*.txt' '*.jinja' '*.md' '*.png' '.gitattributes'
```

## Task Selection

| Option | Meaning |
|---|---|
| `--task` | A RoboCasa365 task name, e.g. `OpenDrawer`. |
| `--suite` | The data split (default `target`, the Target50 evaluation setting). |
| `--seed` | The scene seed: 1-10 for atomic tasks, 1-5 for composite ones. Always pass it: `pyrualean play` sends `0` when it is left out. |

## Run a Task

```bash
export RPENT_ROOT=/path/to/rpent
pyrualean play --robot robocasa --arm cells --task OpenDrawer --suite target --seed 1 \
    --model gpt-6-astra --max-turns 40 --budget-s 7200 --out runs/robocasa-demo --cuda-device 0
pyrualean summarize runs/robocasa-demo
```

An episode boots private env and VLA servers (~3 GB + ~15-17 GB of GPU memory); `--vla-endpoint` attaches it
to a running RLDX-1 server instead.

## View Results

The run directory keeps the episode video (`episode.mp4`), every cell the agent ran (`workspace/`) and what each
printed (the runtime's event file, e.g. `codex_events.jsonl`), the ledger of primitive calls (`calls.jsonl`), the token
usage and RoboCasa's verdict
(`result.json`), and RPent's own records (`states.json`, per-step images and world maps).

## Results

GPT-6 Astra, five seeds per task; at most 40 LLM calls on atomic tasks and 100 on composite ones, and two hours per
episode. Against RPent's
tool-calling agent with the same planner and primitives; LLM calls, input tokens and cost per solved episode on
the instances both solved:

| | Success (%) | LLM calls | Input tokens | Cost |
|---|---|---|---|---|
| Atomic (90 task instances), tool calling (RPent) | 78.9 | 10.4 | 261k | $0.62 |
| Atomic, **PyRUA-Lean** | **86.7** | **5.6** | **171k** | **$0.54** |
| Composite (160 task instances), tool calling (RPent) | 34.4 | 28.4 | 1.22M | $2.01 |
| Composite, **PyRUA-Lean** | **39.4** | **14.9** | **603k** | **$1.25** |

## What Runs Where

- **Env server** (RPent) owns the RoboCasa kitchen environment and its EGL rendering.
- **RLDX-1 server** (RPent) holds the VLA.
- **PyRUA-Lean** boots them for the episode, builds RPent's toolkit and serves the agent's cells against `robo`.
- **Agent runtime** (Codex CLI, Claude Code or the `openai` loop) talks to the model.

## Primitives the Agent Can Call

Navigation and motion:

- `navigate_to(xy)`: drive the base towards a world (x, y) and stop facing it; `move_base(forward, lateral, turn)`:
  raw base velocities.
- `move_to(xyz)`, `move_delta(dxyz)`: servo the gripper to a world position or by a displacement.
- `rotate_pitch(target_pitch)`: tilt the wrist.
- `set_gripper(gripper, steps)`, `release()`, `scripted_grasp(xyz)`: drive the fingers; let go; a coarse scripted
  grasp (open, hover, descend, close, lift).
- `rldx_skill()`: let the frozen RLDX-1 VLA drive the whole body on the task instruction; `rldx_arm()`: the arm,
  with the base clamped to small motions.

Perception and state:

- `back_project_batch(pixels)`, `query_world_map(z_min, z_max)`: world points under pixels, what lies inside a
  height band.
- `floor_overlay()`: the navigation view with the floor painted.
- `state()`, `image(camera)`, `depth(camera)`, `world_map(camera)`, `show(camera)`, `camera_meta(camera)`,
  `success_criteria()`, `artifact(name)`, `done`.

??? note "API reference: what the agent reads, generated from `pyrualean.robocasa` when these docs are built"

    <!-- api-reference: robocasa -->
