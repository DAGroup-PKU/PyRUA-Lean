# RoboTwin 2.0

!!! note "Runs on RPent"

    RoboTwin 2.0 runs on [RPent](https://github.com/RLinf/RPent)'s RoboTwin stack: RPent boots the SAPIEN
    environment and the LingBot-VLA, and every `robo` call goes through RPent's toolkit. New robots are added
    through XPolicyLab ([Adding a Robot](adding-a-robot.md)).

RoboTwin 2.0 is a dual-arm manipulation benchmark in SAPIEN. The robot has two arms with RPent's collision-aware
motion planning and the frozen LingBot-VLA.

## Overview

| | |
|---|---|
| **Robot** | Two arms in SAPIEN (headless, Vulkan) |
| **Action models** | LingBot-VLA (`lingbot_act`) |
| **Agent runtimes** | `codex` (default), `claude`, `openai` ([Agent Configuration](../guide/agents.md)) |
| **Tasks** | RoboTwin 2.0's 50 dual-arm tasks, in the `demo_randomized` evaluation setting |
| **Hardware** | Linux, NVIDIA GPU with Vulkan; Python 3.11 |

### Observation and Action

| Item | Description |
|---|---|
| Observation | The `head`, `left_wrist` and `right_wrist` images (240 × 320), their depth and per-pixel world coordinates, and the state of both arms. |
| Action | The agent's cells call `robo`; each call is one RPent tool (a planned arm motion, a gripper command or a LingBot-VLA run) that steps the simulator. |
| Success | RoboTwin's own `eval_success` predicate. `robo.done` turns true once it fires, and every further motion is refused. |
| Task prompt | The task instruction, a task card (the step-0 images and the arms' initial poses) and the robot's knowledge text. |

## Installation and Resources

RPent at commit `886b3b2`, in its own Python 3.11 environment (the same torch build as for LIBERO-PRO), with
PyRUA-Lean in the same environment:

```bash
uv pip install -e ".[rlinf]" rlinf-robotwin-runtime==0.1.1 rlinf-lingbotvla==0.1.1 imageio-ffmpeg
git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git
uv pip install --no-build-isolation ./curobo         # against that torch: CUDA_HOME (12.8) and TORCH_CUDA_ARCH_LIST set
uv pip install -e /path/to/PyRUA-Lean
robotwin-download-assets --output /path/to/robotwin-assets      # the Hugging Face dataset TianxingChen/RoboTwin2.0, 16 GB
export ROBOTWIN_ASSETS_PATH=/path/to/robotwin-assets
```

Rendering is headless SAPIEN on Vulkan: the NVIDIA ICD must be found; set `VK_ICD_FILENAMES` when it is not under
`/usr/share/vulkan/icd.d`.

## VLA Configuration

LingBot-VLA `RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500` at revision `e727b46cd220b66981ea4d2fd9ba84adc189e2cc`,
including `qwen_base/`:

```bash
hf download RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500 --revision e727b46cd220b66981ea4d2fd9ba84adc189e2cc \
    --local-dir /path/to/lingbot
export LINGBOT_MODEL_PATH=/path/to/lingbot
```

## Task Selection

| Option | Meaning |
|---|---|
| `--task` | A RoboTwin task name, e.g. `beat_block_hammer`. |
| `--suite` | The task config (default `demo_randomized`, the evaluation setting). |
| `--seed` | The exact scene seed, e.g. `100000`; RPent lists verified seeds in `robots/robotwin/eval/demo_randomized.json`. Always pass it: `pyrualean play` sends `0` when it is left out. |

## Run a Task

```bash
export RPENT_ROOT=/path/to/rpent
pyrualean play --robot robotwin --arm cells --task beat_block_hammer --suite demo_randomized --seed 100000 \
    --model gpt-6-astra --max-turns 40 --budget-s 7200 --out runs/robotwin-demo --cuda-device 0
pyrualean summarize runs/robotwin-demo
```

An episode boots private env and VLA servers (~7 GB + ~9 GB of GPU memory); `--vla-endpoint` attaches it to a
running LingBot-VLA server instead.

## View Results

The run directory keeps the episode video (`episode.mp4`), every cell the agent ran (`workspace/`) and what each
printed (the runtime's event file, e.g. `codex_events.jsonl`), the ledger of primitive calls (`calls.jsonl`), the token
usage and RoboTwin's verdict
(`result.json`), and RPent's own records (`states.json`, per-step images and world maps).

## Results

GPT-6 Astra, 50 tasks at five seeds (250 task instances), at most 40 LLM calls and two hours per episode. Against RPent's tool-calling agent
with the same planner and primitives; LLM calls, input tokens and cost per solved episode on the instances both
solved:

| | Success (%) | LLM calls | Input tokens | Cost |
|---|---|---|---|---|
| Tool calling (RPent) | 60.0 | 12.2 | 415k | $1.00 |
| **PyRUA-Lean** | **69.2** | **8.9** | **245k** | **$0.66** |

## What Runs Where

- **Env server** (RPent) owns the RoboTwin task environment and its SAPIEN rendering.
- **LingBot-VLA server** (RPent) holds the VLA.
- **PyRUA-Lean** boots them for the episode, builds RPent's toolkit and serves the agent's cells against `robo`.
- **Agent runtime** (Codex CLI, Claude Code or the `openai` loop) talks to the model.

## Primitives the Agent Can Call

Motion:

- `move_to(arm, xyz, quat=None, gripper=None)`: plan and execute a collision-aware motion of one arm to a world pose.
- `rotate_wrist(arm, delta_yaw_deg)`: rotate one end-effector about the world z axis, keeping its position.
- `set_gripper(arm, val, steps)`, `release(arm)`: drive one gripper; open it.
- `lingbot_act(chunks)`: let the frozen LingBot-VLA drive both arms for `chunks` × 50 native actions.

Perception and state:

- `sample_world_xyz(view, pixels)`, `query_world_map(view, bbox)`: world points under pixels, statistics of the
  points inside a pixel box.
- `state()`, `image(camera)`, `depth(camera)`, `world_map(camera)`, `render()`, `show(camera)`, `camera_meta(camera)`,
  `artifact(name)`, `done`.

??? note "API reference: what the agent reads, generated from `pyrualean.robotwin` when these docs are built"

    <!-- api-reference: robotwin -->
