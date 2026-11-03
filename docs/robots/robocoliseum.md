# RoboColiseum

[RoboColiseum](https://robocoliseum.ai/leaderboard) is an embodied-AI benchmark jointly initiated by nearly 20
universities, companies and research institutes worldwide, including Hugging Face, NVIDIA and Tsinghua University. It runs on
[GenieSim](https://github.com/AgibotTech/genie_sim): the G2 robot in Isaac Sim, on four boards (instruction following,
spatial reasoning, general manipulation and robustness). GenieSim's own benchmark runs the simulator and judges every
episode, and asks a policy for actions over its inference protocol; PyRUA-Lean answers it as that policy. A language
model writes Python against `robo`, one cell at a time, while the episode runs. No VLA policy, no privileged
information.

## Overview

| | |
|---|---|
| **Robot** | AgiBot G2 (`G2_omnipicker`): two 7-joint arms with one gripper each, a 5-joint waist and a wheeled base |
| **Agent runtime** | `openai`: any OpenAI-compatible chat endpoint (GPT-6.1 Sol, reasoning effort `high`, in our runs) |
| **Models on the policy side** | SAM3 (segmentation), as one HTTP service for every policy process; no VLA |
| **Boards** | `instruction`, `spatial`, `manip`, `robust` |
| **Hardware** | Linux x86_64, Python 3.11 or 3.12 (the arm IK's wheels); an NVIDIA GPU for SAM3 (about 10 GB). The simulator runs on the platform, or wherever you run GenieSim |

### Observation and Action

| Item | Description |
|---|---|
| Observation | GenieSim's inference request: three colour images (`head`, `hand_left`, `hand_right`), each camera's depth image and calibration (intrinsics and pose; the benchmark withholds the calibration while it degrades the images, as the robustness board's camera-quality cases do), the joints of both arms, the grippers and the waist, both end-effector poses, and the instruction. No object poses, and the task's progress is not read. |
| Action | One action per request: arm motions are solved in the library on the benchmark's own IK (`ik_solver`) and sent as joint targets (`JOINT_ABS`), with the gripper commands; the waist as joint targets, the base as a relative pose. |
| Success | The benchmark's own checks of the scene, stage by stage, every 30 actions. The agent never sees them; when a program finishes, the robot holds still until the benchmark ends the episode. |
| Task prompt | The instruction, and one knowledge text for every task: how the tools behave and how the benchmark judges a stage (which state counts, when it is checked). No task recipes. |

## Installation and Resources

1. The policy environment, from a PyRUA-Lean checkout:

    ```bash
    pip install -e ".[robocoliseum]"
    ```

    The extra installs the bridge's and perception's packages and the benchmark's arm IK, `ik_solver`, which is not
    on PyPI: its wheel comes from GenieSim's `3rdparty/` at a fixed commit. The robot's URDF and the solver's
    configuration are GenieSim's own, shipped in `pyrualean/hosts/robocoliseum_ik/`; `GENIESIM_IK_SDK` names another
    directory holding them (`G2_NO_GRIPPER.urdf`, `g2_solver.yaml`) when GenieSim's robot description changes.

2. SAM3, as one service for every policy process on the machine. Its environment needs a CUDA build of `torch`, the
   `sam3` package (it pins `numpy<2`, so a separate environment is simplest) and the checkpoint:

    ```bash
    pip install "sam3 @ git+https://github.com/RLinf/sam3.git@c1f6b1e83967c912f675d8975d437ccdc41e17d8" \
        einops pycocotools psutil
    bash integrations/xpolicylab/PyRUA_Lean/download_assets.sh   # sam3.pt (3.4 GB), into that directory's checkpoints/
    python src/pyrualean/hosts/sam3_http.py --port 8790 \
        --checkpoint integrations/xpolicylab/PyRUA_Lean/checkpoints/sam3.pt
    ```

    It prints `SAM3 ready on http://127.0.0.1:8790` once loaded. `PYRUA_SAM3_ENDPOINT` points the policy processes at
    another address. Without the service, finding an object by name reports that SAM3 is not running; measuring the
    object at a pixel still works.

## Model Configuration

The cells are written by a model behind an OpenAI-compatible chat endpoint, with `play`'s options:

| Option | Our runs | Meaning |
|---|---|---|
| `--runtime openai` | | the in-process agent loop |
| `--base-url`, `--api-key-env`, `--model` | the endpoint, the variable holding its key, `gpt-6.1-sol` | |
| `--openai-extra-body` | `'{"reasoning_effort": "high"}'` | JSON merged into every request |
| `--openai-max-tokens` | `32000` | the longest reply |
| `--images` | `on-demand` | the model sees a camera only when its code asks for it (`robo.show`) |
| `--budget-s` | `1800` | wall-clock limit of an episode, in seconds |

## Run a Task

=== "Online (RoboColiseum)"

    Submit a job for one board on the platform first; it gives the job's `uuid`, its tunnel endpoint, a job token
    and its `parallelism`. Each agent process dials the platform's gateway and serves one tunnel: start up to
    `parallelism` of them, and they play the cases the platform sends until it drains the job.

    ```bash
    export JOB_UUID=... TUNNEL_ENDPOINT=ws://.../api/challenge/tunnel JOB_TOKEN=...
    export OPENAI_API_KEY=...
    for i in $(seq 1 "$PARALLELISM"); do
      python -m pyrualean.hosts.geniesim_robocoliseum agent --arm cells --out "runs/$JOB_UUID/agent_$i" \
          --runtime openai --base-url https://YOUR_API_HOST/v1 --api-key-env OPENAI_API_KEY \
          --model gpt-6.1-sol --openai-extra-body '{"reasoning_effort": "high"}' \
          --openai-max-tokens 32000 --images on-demand --budget-s 1800 &
    done
    wait
    ```

    An agent exits 0 once the job is drained, 1 when its tunnel is refused or stays lost past the gateway's
    30-second reconnect window, 130 on Ctrl-C. The platform keeps the scores.

=== "Your own GenieSim run"

    Start the policy first, then the benchmark with its inference host pointing at it:

    ```bash
    python -m pyrualean.play --robot robocoliseum --arm cells --out runs/pick_block_color \
        --task g2op_if_pick_block_color --port 8999 --output-root <genie_sim>/output \
        --runtime openai --base-url https://YOUR_API_HOST/v1 --api-key-env OPENAI_API_KEY \
        --model gpt-6.1-sol --openai-extra-body '{"reasoning_effort": "high"}' \
        --openai-max-tokens 32000 --images on-demand

    # in GenieSim's simulator environment, from its checkout
    python source/geniesim_benchmark/src/geniesim_benchmark/app/app.py \
        --config source/geniesim_benchmark/src/geniesim_benchmark/config/g2op_if_pick_block_color.yaml \
        --benchmark.infer_host 127.0.0.1:8999 --benchmark.model_arc corobot
    ```

    `play` plays the run's first episode and ends the rest. `--output-root`, the benchmark's output directory, lets
    it read the episode's scores.

## View Results

- Each episode's card, prompt, programs, tool calls, transcript and every recorded step (the three images and the
  state) are kept under `--out`: `episode_NNN/` of each session for an agent, the directory itself for `play`.
- An agent appends one line per episode to `online.jsonl` (session, task, instruction, status, actions).
- With `--output-root`, `benchmark_verdict.json` holds the benchmark's scores of the episode: one per stage
  (`STEPS`) and `E2E`.

## What Runs Where

- **Benchmark** (GenieSim, in Isaac Sim: on the platform online, or in your simulator environment) owns the
  simulator: it builds every request, counts each stage's action budget and judges the episode.
- **Policy process** (this library) answers the benchmark's requests: as a WebSocket server the benchmark connects
  to, or online through the platform's tunnel, which it dials. It runs the agent loop, the sandboxed cells and the
  primitives (inverse kinematics, gripper, perception).
- **SAM3 service** segments for every policy process on the machine.
- **Model endpoint** writes the cells.

One request is answered with one action, so a primitive sees the result of every native action. The benchmark does
not step the simulator while the model thinks.

## Primitives the Agent Can Call

Motion:

- `move_to(xyz, arm, gripper, pitch, yaw, roll, path)`: move one arm's grasp point to a position, with an
  orientation. A pose out of reach moves nothing, and the result names reachable poses nearby.
- `move_both(left, right)`: both arms' grasp points at once, each to a position.
- `set_gripper(arm, command)`: close or open one gripper until the fingers stop; the result says whether something
  is held.
- `move_waist(forward, up, turn_deg, lean_deg)`, `move_base(forward, left, turn_deg)`: the waist and the wheeled
  base.
- `home(arm="both")`: return an arm to the joint angles it had when the episode started.
- `wait_for_check(checks=1)`: hold still until the benchmark's next stage check has seen the scene.

Perception and state:

- `locate(target, camera)`: find an object by words (SAM3) or at a pixel and measure it in 3D from the depth image:
  centre, top, the surface it stands on, extent, tilt, and the yaw of a top grasp across it.
- `segment(camera, text=..., row=..., col=...)`: the object's region in the current image.
- `back_project(row, col, camera)`: the point seen at one pixel.
- `state()`, `image(camera)`, `show(camera)`, `artifact(name)`: the recorded state of both arms, the images, and
  what the model looks at.

??? note "API reference: what the agent reads, generated from `pyrualean.robocoliseum` when these docs are built"

    <!-- api-reference: robocoliseum -->
