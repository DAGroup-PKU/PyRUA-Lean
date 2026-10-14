# RoboDojo

[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) is a bimanual manipulation benchmark in Isaac Sim with a
public [leaderboard](https://robodojo-benchmark.com/LeaderBoard). PyRUA-Lean runs on it as an
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) policy: RoboDojo's own evaluation client runs the simulator
and judges every episode, and PyRUA-Lean answers it as a policy server. A language model writes Python against
`robo`, one cell at a time, while the episode runs. No VLA policy, no privileged information.

## Overview

| | |
|---|---|
| **Robot** | Two ARX-X5 arms with one gripper each; joint actions (`env_cfg_type` `arx_x5`, `action_type` `joint`) |
| **Agent runtime** | `openai`: any OpenAI-compatible chat endpoint (GPT-6 Astra in our evaluation) |
| **Models on the policy side** | SAM3 (segmentation); no VLA |
| **Tasks** | 42 simulation tasks in five dimensions: Generalization, Precision, Long-Horizon, Memory, Open |
| **Hardware** | Linux, NVIDIA GPU; SAM3 takes about 8 GB on the policy side, Isaac Sim needs its own share |

### Observation and Action

| Item | Description |
|---|---|
| Observation | RoboDojo's default observation: three colour images (`cam_head`, `cam_left_wrist`, `cam_right_wrist`), the joint and end-effector state of both arms, and the instruction. No depth image, no camera matrices, no object poses. |
| Action | The agent's cells call `robo`'s primitives; each motion is solved to joint targets in the policy server and sent to the client as native actions. |
| Success | RoboDojo's own check of the scene. The agent never sees it; when a program finishes before the task is solved, RoboDojo judges the scene as it stands. |
| Task prompt | The task's instruction, and one knowledge text for every task: how the tools behave and how RoboDojo judges an episode (for most tasks: both arms back at their start pose with the grippers open). No task recipes. |

## Installation and Resources

1. Install RoboDojo with its XPolicyLab submodule and the simulator's environment, as
   [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) describes.
2. Put the adapter into XPolicyLab: copy `integrations/xpolicylab/PyRUA_Lean` of this repository to
   `XPolicyLab/policy/PyRUA_Lean` (or link it from a PyRUA-Lean checkout).
3. In the policy environment, a conda environment (XPolicyLab's scripts `conda activate` it by name) with a CUDA
   build of `torch` and the `torchvision` built for it:

```bash
cd XPolicyLab/policy/PyRUA_Lean
bash install.sh              # XPolicyLab's server, PyRUA-Lean, the sam3 package
hf auth login                # facebook/sam3 is gated: accept its licence on Hugging Face first
bash download_assets.sh      # SAM3's checkpoint (3.4 GB) into ./checkpoints/sam3.pt
```

`install.sh` installs PyRUA-Lean from the checkout the directory is linked from, or from the public repository when
the directory was copied; it keeps the environment's `torch` and `torchvision` as they are.
`download_assets.sh modelscope` fetches the checkpoint from the ModelScope mirror, and `SAM3_CHECKPOINT_PATH` names a
checkpoint kept elsewhere. The simulator's environment needs nothing beyond RoboDojo's own installation.

## Model Configuration

The cells are written by a model behind an OpenAI-compatible chat endpoint:

```bash
export OPENAI_BASE_URL=https://YOUR_API_HOST/v1
export OPENAI_API_KEY=...
export PYRUALEAN_MODEL=gpt-6-astra          # the model's name at that endpoint
```

The adapter's `deploy.yml` sets the episode:

| Key | Default | Meaning |
|---|---|---|
| `max_turns` | `100` | decision budget: code cells per episode |
| `budget_s` | `5400` | wall-clock limit of an episode, in seconds |
| `images` | `on-demand` | the model sees a camera only when its code asks for it (`robo.show`) |
| `feedback` | `pure` | a tool call returns what the code printed, nothing else |
| `openai_options` | `max_tokens: 8192`, `nudges: 20`, `extra_body: {reasoning_effort: high}` | the longest reply, the replies without a tool call tolerated, and JSON merged into every request |

## Task Selection

| Option | Meaning |
|---|---|
| `--task` / `task_name` | A RoboDojo task, e.g. `stack_blocks`. |
| `--seed` / `seed` | The layout set: `0`, `1` or `2`, each with its own layouts of every task. |
| `--eval-num` | Episodes of the task; `native` runs RoboDojo's own number (50, or 25 for each half of a Generalization task). Start with `1`: an episode took about 25 LLM calls and 13 minutes in the paper's runs, about $2.4 at GPT-6 Astra's gateway price. |
| `--ckpt` / `ckpt_name` | Ignored: there are no weights. `no-checkpoint` is the placeholder. |

## Run a Task

=== "RoboDojo CLI"

    From the RoboDojo checkout, the official entry point:

    ```bash
    bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/PyRUA_Lean --task stack_blocks \
        --ckpt no-checkpoint --policy-env <policy env> --eval-env <simulator env> \
        --env-cfg arx_x5 --action-type joint --seed 0 --eval-num native
    ```

    `robodojo.sh server` (on the policy machine) and `robodojo.sh client` (on the simulator machine, with
    `--policy-host` and `--policy-port`) run the two sides apart; give both `--action-type joint` and the same
    `--ckpt`, since their default is `ee`.

=== "Adapter"

    ```bash
    cd XPolicyLab/policy/PyRUA_Lean
    bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
      <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

    # example
    bash eval.sh RoboDojo stack_blocks no-checkpoint arx_x5 joint 0 0 0 pyrualean RoboDojo
    ```

=== "Wiring check"

    No simulator and no model call: XPolicyLab's debug client exercises the loop, the action keys and their
    dimensions.

    ```bash
    cd XPolicyLab/policy/PyRUA_Lean
    EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 pyrualean RoboDojo
    ```

## View Results

- RoboDojo records every episode under `eval_result/` of its checkout; `bash scripts/robodojo.sh summarize` writes the
  leaderboard's table (SR and Score per dimension) to `eval_result/RoboDojo/_summary.md`. A task counts there only
  once all its episodes for that seed are in.
- Each episode's programs, tool calls and full transcript are kept under the adapter's `result_dir`:
  `./results/<task>_l<layout>_s<seed>-<time>/`.

## Results

GPT-6 Astra, seed 0, all 42 tasks (2,100 episodes):

| | Generalization | Precision | Long-Horizon | Memory | Open | **Average** |
|---|---|---|---|---|---|---|
| SR (%) | 45.83 | 37.25 | 48.00 | 63.00 | 64.00 | **51.62** |
| Score | 50.65 | 47.02 | 63.26 | 63.00 | 67.09 | **58.20** |

That is 19.4 points above the best entry (32.26) of a snapshot of the public
[leaderboard](https://robodojo-benchmark.com/LeaderBoard) taken on October 9, 2026; the board changes as entries arrive.
An episode takes 25.7 LLM calls on average.

## What Runs Where

- **Evaluation client** (RoboDojo, in the simulator's environment) owns Isaac Sim: it builds the observation,
  counts the task's action budget and gives the verdict.
- **Policy server** (XPolicyLab's server with the adapter, in the policy environment) runs the PyRUA-Lean episode:
  the agent loop, the sandboxed cells, the primitives (inverse kinematics, gripper, perception) and SAM3.
- **Model endpoint** writes the cells.

The two sides talk XPolicyLab's websocket protocol (`update_obs` / `get_action`). A model may think for minutes
between two actions, so the policy server answers "wait" until the next action is ready and the client asks again;
no request is held longer than `poll_s` (20 s).

## Primitives the Agent Can Call

Motion:

- `move_to(xyz, arm, gripper, pitch, yaw, roll, path)`: move one arm's grasp point to a world position, with an
  orientation. A pose out of reach moves nothing, and the result names reachable poses nearby.
- `set_gripper(arm, gripper, opening=None)`: close or open one gripper until the fingers stop; the result says
  whether something is held and how wide it is.
- `home(arm="both")`: return an arm to the joint angles it had when the episode started.

Perception and state:

- `locate(text_prompt, camera)`: find an object by words and measure it in 3D from the colour cameras: centre,
  top, footprint, heading and the yaw of a top grasp across it.
- `segment(text_prompt, camera)`: the SAM3 mask of an object in the current image.
- `back_project(row, col, camera)`: the world point seen at one pixel.
- `state()`, `image(camera)`, `show(camera)`, `camera_meta(camera)`, `artifact(name)`: the recorded state of both
  arms, the images, and what the model looks at.

Twists and button presses are compositions of these; the knowledge text says how.

??? note "API reference: what the agent reads, generated from `pyrualean.robodojo_xpl_rgb` when these docs are built"

    <!-- api-reference: robodojo_xpl_rgb primitives=no-vla -->
