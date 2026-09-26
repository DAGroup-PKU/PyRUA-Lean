<div align="center">

# PyRUA-Lean

**Robot agents that act in code, not calls.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

**PyRUA-Lean** (*Py*thon for lean *R*obot-*U*se *A*gents, the robot counterpart of computer-use agents)
turns a robot into a Python object. A tool-calling agent spends a full LLM call on every small step of a task
and re-sends everything before it each time. A PyRUA-Lean agent writes code against `robo` instead, so one call
can find an object, move above it, grasp, check the gripper and try again.

<p align="center">
  <img src="docs/assets/results-overview.png" width="100%"
       alt="Tool calling against PyRUA-Lean on four benchmarks: a higher success rate on every one (63.1% to 71.0% overall), and per solved episode fewer LLM calls, 1.5x to 4.5x fewer prompt tokens and 1.2x to 3.1x fewer dollars at list price">
</p>

## Why code

Same model (GPT-6 Astra), same robot primitives, VLA policies and simulators, same budget of LLM calls; only the
action format differs. Success rate from tool calling to PyRUA-Lean; tokens and cost on the task instances both
agents solve:

| Benchmark | Success rate | Tokens | Cost at list price |
|---|---|---|---|
| LIBERO-PRO (40 tasks) | 83.0% → **94.0%** | **4.5×** fewer | **3.1×** cheaper |
| RoboTwin 2.0 (50 tasks) | 60.0% → **67.2%** | **1.7×** fewer | **1.5×** cheaper |
| RoboCasa365 atomic (18 tasks) | 78.9% → **86.7%** | **1.5×** fewer | **1.2×** cheaper |
| RoboCasa365 composite (32 tasks) | 34.4% → **39.4%** | **2.0×** fewer | **1.6×** cheaper |

Over all four, PyRUA-Lean solves 12% more task instances and consumes 65% fewer tokens on the ones both agents
solve.

## What an agent writes

The agent gets one tool, `python(code)`, over a persistent namespace that holds `robo`. One LLM call can do all
of this:

```python
bowl = robo.segment("the black bowl on the stove")                # SAM3 + depth: a point in the world
x, y, _ = bowl.world_xyz
robo.move_to([x, y + 0.045, 0.60])                                # scripted servo; returns what happened
pick = robo.pi0_pick("pick up the black bowl", max_chunks=8)      # a frozen Pi0.5 VLA skill
if robo.state().gripper_opening < 0.01:                           # closed on nothing: try again
    pick = robo.pi0_pick("pick up the black bowl", max_chunks=8)
print(pick.success, pick.peak_lift_m)                             # the agent reads only what it prints
robo.show("wrist")                                                # and looks only when it asks to
```

A tool-calling agent spends an LLM call on each of those steps, and every call re-sends the conversation so
far, camera images included.

## Quick start

PyRUA-Lean drives the robot stacks of [RPent](https://github.com/RLinf/RPent): install RPent with the robot you
want first ([docs/setup.md](docs/setup.md) lists versions, checkpoints and environment variables).
Agents run through the [Codex CLI](https://github.com/openai/codex) 0.155.1 or Claude Code.

```bash
git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git && cd PyRUA-Lean
pip install -e .          # into the Python environment of your RPent robot stack
pyrualean api             # the robot API the agent sees; no simulator needed

# one LIBERO-PRO episode: GPT-6 Astra writes code against robo, at most 40 LLM calls
export RPENT_ROOT=/path/to/rpent
pyrualean play --arm cells --suite libero_spatial_swap --task 7 --seed 2 \
    --model gpt-6-astra --max-turns 40 --out runs/demo --cuda-device 0
pyrualean summarize runs/demo
```

The run directory keeps the episode video, every cell the agent ran with its output, the ledger of primitive
calls, the token usage and the benchmark's own verdict.

## Robots

| `--robot` | Benchmark | Robot and skills |
|---|---|---|
| `libero` (default) | LIBERO-PRO | Franka arm; scripted servos, Pi0.5 VLA, SAM3 perception |
| `robotwin` | RoboTwin 2.0 | two arms; LingBot-VLA |
| `robocasa` | RoboCasa365 | mobile manipulator; RLDX-1 VLA, navigation |

[docs/multi-robot.md](docs/multi-robot.md) is the contract for adding a robot.

## How it works

- **Same primitives, same judge.** Each `robo` method maps one to one onto an RPent tool and runs through RPent,
  so an episode leaves the same records as an RPent episode and is scored by the benchmark's own success check.
- **Results, not exceptions.** Motion calls block and return what actually happened (`Move.reached`,
  `Pick.success`, ...); misuse raises, and a finished episode stops every further motion.
- **A prompt that cannot drift.** The API reference the agent reads is generated from the live class.
- **Three ways to run.** `play --arm cells` (one cell per decision), `play --arm program` (whole programs), and a
  single shot with `generate` + `run`.
- **Sandboxed.** Agent code runs with a fixed set of imports, file access confined to its workspace and no
  handle to the backend.

## Documentation

- [docs/guide.md](docs/guide.md): the `robo` API, the prompt, single-shot and interactive runs, budgets,
  evaluation and configuration.
- [docs/setup.md](docs/setup.md): setting up the three robot stacks and the agent runtime.

## License and acknowledgements

Apache License 2.0 ([LICENSE](LICENSE)). PyRUA-Lean builds on [RPent](https://github.com/RLinf/RPent)
(Apache-2.0): its robot stacks, services and primitive implementations do the work behind `robo`, the
tool-calling baseline is RPent itself, and parts of this repository are adapted from it; see `NOTICE`. If you use
RPent, please cite its paper, [Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via
Memory-Guided Agents](https://arxiv.org/abs/2607.08448).
