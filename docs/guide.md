# PyRUA-Lean guide

The full reference: the robots, the `robo` API, the prompt, single-shot and interactive runs, budgets, zero-shot isolation, evaluation and configuration. The [README](../README.md) is the short introduction.

A policy is one ordinary Python file:

```python
def run(robo):
    bowl = robo.segment("the black bowl on the stove")
    if not bowl.found:
        return
    x, y, _ = bowl.world_xyz
    robo.move_to([x, y + 0.045, 0.60])                    # hover above the rim
    pick = robo.pi0_pick("pick up the black bowl", max_chunks=8)
    if robo.state().gripper_opening < 0.01:                # closed on nothing
        pick = robo.pi0_pick("pick up the black bowl", max_chunks=8)
    robo.set_gripper(robo.CLOSE, steps=8)
    plate = robo.segment("the white plate")
    robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.60], gripper=robo.CLOSE)
    robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.47], gripper=robo.CLOSE, step_clip=0.012)
    robo.release()
```

The host constructs `robo`, calls `run(robo)` once, and the simulator's own
success predicate decides the outcome.  No language model runs while the
program executes.

The primitives behind `robo` are **RPent's LIBERO primitives, unchanged**: the
scripted OSC servos (`move_to`, `move_pose`, `rotate_*`, `set_gripper`,
`release`), the frozen Pi0.5 VLA skills (`pi0_pick`, `pi0_doubled`) and the
SAM3 / back-projection perception helpers.  Every call still goes through
RPent's `Toolkit.execute_tool`, so a PyRUA-Lean episode leaves the same
`states.json`, `episode.mp4` and per-step images as an RPent agent episode and
is judged by the same `toolkit.solved()`.  That is the point of the project:
the *same* primitives can be exercised in two interaction forms,

| arm | who decides each step | interaction | where |
|---|---|---|---|
| tool calling (CUA-style) | the model, one MCP/tool call per turn, observation returned every turn | RPent planner (`rpent --planner codex ...`) | [RPent](https://github.com/RLinf/RPent) |
| code (this package) | the model, writing Python against `robo`: one cell per decision, or a whole program | `pyrualean play --arm cells`, or `pyrualean generate` then `pyrualean run` | this repo |

and compared on identical `(suite, task, seed)` cells with identical success
predicates.

## Robots

`--robot libero` (default), `--robot robocasa` and `--robot robotwin` select
the robot: its policy-facing class (`pyrualean/<robot>.py`), the host that
boots RPent's stack for it (`pyrualean/hosts/rpent_<robot>.py`) and its
knowledge file.  All three share `pyrualean/_robot.py` (call ledger,
episode guard, images, world maps, `show`) and are registered through
`pyrualean.robots`; `docs/multi-robot.md` is the contract for adding one.
The sections below describe the LIBERO robot; the other two follow the
same pattern with their own primitives (a mobile base and the RLDX-1 VLA
on RoboCasa365, two arms and LingBot-VLA on RoboTwin).

## The library

`pyrualean.libero.LiberoRobot` is the only API a policy sees.  Each method
maps 1:1 onto one RPent tool, takes real Python arguments with the same
names and defaults as RPent's schema, and returns a frozen dataclass whose
fields are documented (that documentation *is* the prompt, see below).

| `robo.` | RPent tool | returns |
|---|---|---|
| `state(step=-1)` | `view_env_state` | `State` (eef pose, gripper opening, object names, flags) |
| `image(camera, resolution)` / `world_map(...)` / `artifact(name)` | recorded artifacts | numpy arrays / bytes |
| `camera_meta(camera)` | `view_camera_meta` | dict |
| `segment(prompt \| point=...)` | `segment` | `Segment` (`found`, `world_xyz`, `score`) |
| `back_project(row, col)` | `back_project` | `BackProjection` |
| `region_center(row_range, col_range)` | `back_project` (region mode) | `RegionCenter` |
| `move_to(xyz, gripper=-1, ...)` | `move_to` | `Move` (`reached`, `final_dist_m`, ...) |
| `move_pose(xyz, target_pitch=..., ...)` | `move_pose` | `MovePose` |
| `rotate_wrist(target_yaw= / delta_yaw=)` | `rotate_wrist` | `Rotation` |
| `rotate_pitch(target_pitch= / delta_pitch=)` | `rotate_pitch` | `Rotation` |
| `set_gripper(gripper, steps=5)` | `set_gripper` | `Gripper` |
| `release(max_steps=20)` | `release` | `Release` (`terminated`) |
| `pi0_pick(prompt, max_chunks=24, ...)` | `pi0_pick` | `Pick` (`success`, `peak_lift_m`, ...) |
| `pi0_doubled(prompt, max_chunks=20)` | `pi0_doubled` | `Contact` |

Library rules that a policy can rely on:

- Motion calls block and return what actually happened; `Move.reached` is
  `final_dist_m <= tol`, `Pick.success` is RPent's lift heuristic.
- Misuse raises: `ValueError` for bad arguments (including a planar move over
  0.30 m, which RPent forbids because it flips the OSC IK), `ToolError` when
  the backend rejects a call or a service fails.  Measured outcomes (a mask
  not found, a stalled servo) are results, never exceptions.
- Once the task predicate fires, `robo.done` is true and every further motion
  call raises `EpisodeFinished`, a `BaseException` that `except Exception`
  cannot swallow.  The host's watchdog uses the same mechanism on timeout.
- Every call is recorded in a ledger (`calls.jsonl`: tool, arguments, wall
  time, exact simulator steps, outcome summary).

`RESULT_TYPES` and the docstrings in `src/pyrualean/libero.py` are the
source of truth; `pyrualean api` prints the generated reference.

## The prompt

`pyrualean prompt` renders what a model is given:

1. the contract (write `policy.py` with `run(robo)`; how it is run; how it is
   scored; output format),
2. the API reference, generated with `inspect` from the live class so it can
   never drift from the injected object,
3. `knowledge/libero.md`: the manipulation lessons RPent's system prompt gives
   its agent (gripper sign, 0.30 m traversal limit, `step_clip` per object,
   VLA usage, SAM3 phrasing, agentview-vs-wrist localisation, placing), and
4. a task card (`card.json`): suite/task/seed, the task instruction, object
   names, the initial gripper pose and the step-0 `agentview`/`wrist` images.

Nothing in the card is privileged: it is what RPent's agent sees in its first
`view_env_state` call.

## Running a cell

Requirements: an RPent checkout with the `libero-pro` extra installed, the
LIBERO-PRO assets, the Pi0.5 and SAM3 checkpoints and their environment
variables (`LIBERO_PRO_ASSET_PATH`, `PI05_CHECKPOINT_PATH`,
`SAM3_CHECKPOINT_PATH`, `LIBERO_TYPE=pro`, `MUJOCO_GL=egl`), i.e. exactly what
`rpent --robot libero` needs (`docs/setup.md` lists the setup of all
three robots).  Run the commands below with RPent's interpreter and
`PYTHONPATH` pointing at `src/` (or `pip install -e .` into that environment).
`RPENT_ROOT` locates the checkout when RPent is not installed as a package.

```bash
export RPENT_ROOT=/path/to/rpent
export PYTHONPATH=/path/to/pyrualean/src

# 1. task card (boots only the simulator, ~1 min)
python -m pyrualean card --suite libero_object_swap --task 2 --seed 0 \
    --out runs/card-object_swap_t2_s0 --cuda-device 0

# 2. single-shot generation with the Codex CLI (read-only, empty scratch dir,
#    every Codex event kept so tool use during generation is auditable)
python -m pyrualean generate --card runs/card-object_swap_t2_s0/card.json \
    --out runs/gen-object_swap_t2_s0 --model gpt-6-astra --reasoning xhigh

# 3. run the generated file: one subprocess per episode, hard timeout,
#    crash / timeout / boot failure recorded as a failed episode
python -m pyrualean run --policy runs/gen-object_swap_t2_s0/policy.py \
    --suite libero_object_swap --task 2 --seed 0 \
    --out runs/code-object_swap_t2_s0 --cuda-device 0 --timeout-s 1200

# 4. tabulate / pair with RPent runs
python -m pyrualean summarize runs/code-object_swap_t2_s0 /path/to/rpent/logs/<run>
python -m pyrualean compare runs/code-object_swap_t2_s0 rpent-episodes.json --output paired.json
```

`generate --transfer` switches the card wording to transfer mode: the model
is told the program will run on other seeds of the task and must not hard-code
pixels.  Without it, generation is per-cell (the images are step 0 of the run).

A run directory contains `policy.py` (copy + sha256 in `result.json`),
`prompt.txt`, `card.json` + `card/*.png`, `calls.jsonl`, `policy_output.txt`
(the program's stdout/stderr), `result.json` and RPent's own artifacts
(`states.json`, `episode.mp4`, per-step images, server logs).  Held-out
evaluation is just steps 3-4 on seeds the policy was not generated for.

`examples/replay_object_swap_t2_s0.py` is a plumbing check: a command
sequence transcribed from one successful RPent episode on that seed.  It
proves the library drives the primitives to the same outcome and nothing
more.

## Interactive arms: `pyrualean play`

`generate` + `run` is the strict single shot: one reply, one program, no
feedback.  `play` runs the interactive protocols with the same library and the
same judge.  The agent runtime (the Codex CLI, or Claude Code through the
Claude Agent SDK with `--runtime claude`) owns the agent loop; PyRUA-Lean
serves the arm's tools through an in-process MCP server (adapted from RPent's;
it needs `mcp` 1.x, `uvicorn` and `httpx`, which RPent's environments include)
and audits the event stream.

| arm | tool the agent gets | one call executes | what persists between calls |
|---|---|---|---|
| `cells` | `python(code)` | a cell in a persistent namespace with `robo`, `np`, `math`, `time` (OpenAI python-tool style) | variables, functions, imports, workspace files |
| `program` | `run_program(code \| path)` | a complete `run(robo)` program in a fresh namespace | workspace files (importable), the agent's context |

Both arms also get `finish(status, summary)`; `--max-programs N` caps the
number of programs (`1` = single shot with the result visible afterwards).
Every tool result carries stdout/stderr, the exception if any, the ledger of
primitive calls and the robot state.  `--images` decides when camera images
are attached: `on-motion` (default) attaches the two current images after
every turn that moved the robot, so per-turn perception matches what a
tool-calling agent receives after each primitive; `on-demand` attaches an
image only when the code called `robo.show("agentview" | "wrist")`, i.e. the
program decides when the author needs to look; `none` never attaches.
`--feedback pure` returns only what the code printed (python-tool semantics);
`rich` (default) adds the robot state and the ledger.

Two budget modes.  By default the wall-clock budget (`--budget-s`, thinking
included) and the simulator step budget are the only limits and turns are
counted, not capped.  With `--max-turns N` the episode runs under a
*decision budget* instead: at most N `python` / `run_program` / `finish`
calls (look-only calls included), i.e. N model responses, since the agent
makes one call per response.  `--budget-s` is then only a safety ceiling and
the contract tells the agent that the number of calls is limited without
stating it (RPent's prompt does not state it either).  A decision budget
makes outcomes independent of provider latency, which a wall-clock budget
does not.

`--guides` copies RPent's three LIBERO operating guides (the files its
tool-calling agent reads at the start of every episode) into the workspace
and adds one sentence to the contract saying they are there and optional;
`--guides-dir <dir>` copies every `*.md` file of another directory instead,
and `result.json` records the hashes of the files the agent received.
`--vla-endpoint host:port` / `--sam3-endpoint host:port` attach the episode
to shared Pi0.5 / SAM3 servers instead of booting private ones; the
simulator is always private to the episode.

```bash
python -m pyrualean play --arm cells   --suite libero_object_swap --task 2 --seed 0 \
    --out runs/cells-object_swap_t2_s0 --cuda-device 0 --budget-s 1200 \
    --model gpt-6-astra --base-url https://gateway.example/v1 --api-key-env MY_KEY
python -m pyrualean play --arm program --max-programs 1 ...   # single shot
python -m pyrualean play --arm program --max-programs 5 ...   # multi-round
```

Isolation for zero-shot runs: pass `--codex-home` (a Codex home containing
only `config.toml` + credentials, `memories = false`, `approval_policy =
"never"`; `docs/setup.md` lists the settings the comparison used) and
`--codex-bin` pointing at the real Codex binary, not a wrapper that forces
`CODEX_HOME`.  `play` gives the agent a minimal environment and disables
Codex's shell snapshot and memories, so the sandbox shell cannot find other
Codex homes or earlier sessions.

**Codex code mode.**
- Codex CLI 0.155.1's bundled model catalog marks `gpt-6-astra` `code_mode_only`,
  which hides every tool behind one JavaScript `exec` tool.  The comparison's gateway
  id (`openai/openai/gpt-6-astra`) was unknown to Codex, so it ran on Codex's fallback
  metadata: direct function calls and Codex's generic prompt.
- For `gpt-6-astra`, `play` and `generate` pass `-c model_catalog_json=<path>` with the
  package's `codex-catalog-gpt-6-astra-fallback.json`: Codex 0.155.1's bundled catalog
  with only the `gpt-6-astra` entry set to the fallback's values, so Codex's system
  prompt, built-in tools and settings are byte-identical to the comparison's.  (A
  one-entry catalog would also give direct tools, but Codex keeps its Planning prompt
  sections whenever a catalog is set.)
  With `gpt-6-astra`, give the Codex home RPent uses the same
  `model_catalog_json = "<absolute path>"` line in its `config.toml`.
- Check a run: the first tool call in the session's rollout
  (`$CODEX_HOME/sessions/**/rollout-*.jsonl`) is a `function_call`, not a
  `custom_tool_call` named `exec`, and its base instructions start "You are a coding
  agent running in the Codex CLI".
- Pin Codex 0.155.1 (`npm install -g @openai/codex@0.155.1`).

Policy code runs in a sandbox on the code arms (`pyrualean/sandbox.py`):
imports are limited to pure-computation modules and workspace files, `open`
is confined to the workspace, `exec`/`eval`/`compile` are removed, the
`robo` object carries no handle to the backend, code that names Python's
introspection routes (`__globals__`, `__func__`, `__subclasses__`, ...) is
refused before it runs, and every cell or program is audited statically;
anything that reaches for host handles, private attributes, introspection
or paths outside the workspace is listed in `result.json`
(`generation.sandbox_flags`) and `tool_turns.json`.

The run directory adds `workspace/` (every cell / program the agent ran, plus
its own files), `codex_events.jsonl`, `tool_turns.json` and the usual
`result.json` (with `arm`, `finish`, token usage and an audit of shell
commands).  The tool-calling arm of the comparison is RPent itself, run
unchanged (`rpent --robot libero --planner codex ...`) and parsed with
`parse_rpent_run`.

## Evaluation

`pyrualean.evaluation` holds the episode record (`EpisodeMetrics`), a
read-only parser for RPent transcripts (`parse_rpent_run`), and a strict
paired comparison (`compare`) that refuses unmatched or duplicated
`(backend, task, seed)` keys.  On the code arm `environment_success` is
always a boolean and a crash, timeout or boot failure is a failed episode.
`env_steps` is counted the same way on both sides (exact frame counter on the
code arm; reconstructed from `states.json` for RPent transcripts).

## Setting up the robot stacks

[`docs/setup.md`](setup.md) covers the three RPent stacks (versions,
checkpoints, environment variables) and the agent side (Codex CLI, Codex
home, endpoint, Claude runtime).

## Configuration

Settings that the library reads from the environment are named
`PYRUALEAN_<NAME>`.

| variable | read by | meaning (default) |
|---|---|---|
| `PYRUALEAN_CODEX_BIN`, `PYRUALEAN_CODEX_HOME` | `play` | Codex executable (`codex`) and clean Codex home (none) |
| `PYRUALEAN_RUNTIME`, `PYRUALEAN_CLAUDE_CLI` | `play` | agent runtime `codex` / `claude` (`codex`), `claude` executable (`claude`) |
| `PYRUALEAN_PRIMITIVES` | `play` | primitive set `full` / `no-vla` (`full`) |
| `PYRUALEAN_CODEX_RETRIES` | `play`, `generate` | request / stream retries of a `--base-url` provider (`10`) |

`RPENT_ROOT` / `RPENT_REPO_ROOT` locate the RPent checkout; the robots'
own variables are listed in `docs/setup.md`.

## Development

```bash
pip install -e '.[dev]'
pytest          # offline tests against scripted RPent-shaped backends
ruff check .
```
