# Starting with no prior context

You are taking over an agent-driven data-production pipeline. Your input is a real robot episode of a new task from the
user. Your job includes writing the new scene and the new task's adaptation yourself, not just running existing scripts.
The repository provides general rules and executable tools. You need a coding agent that can read
and write code, run commands, view images and videos and maintain task state; a chat API without tools is not a complete executor.

## Using the repository

Start at the repository root, read AGENTS.md, and take the episode path and a request such as "following the pipeline, produce
N episodes of data for this new task". Do not expect an adapter, judge or prompt to exist. Execute
`.agents/skills/pipeline-orchestrate/SKILL.md` and prepare the task directory yourself:

```bash
python tools/pipeline_agent.py prepare \
  --episode /data/new-task/episode.mp4 \
  --request-file /data/request.txt --output /data/runs/new-task
python tools/pipeline_agent.py context \
  --work /data/runs/new-task --role orchestrator --output /data/runs/new-task/context
```

- `request.txt` is the user's request; you may write it, and it is not a specification the user is asked to write.
- `prepare` only locks the input identity and creates the state file. There is no preset task type, and it does not check
  whether an adapter exists. `--reference` (repeatable) binds extra real photos and device evidence. Never treat a credentials
  directory as input.
- A single video is enough to start observing. Discover and verify actions, calibration and robot assets that sit next to the
  episode or in the deployment configuration.
- Extract frames from the real video as appearance references. Do not make the user re-supply existing images.
- If a single video cannot resolve scale, first use robot dimensions, known objects or multiple views to fit and cross-validate.
  Raise an explicit missing item only when it truly cannot be resolved. "The user has not written the technical config" is
  not a missing input.
- To improve the general capability from work feedback, see `prompts/instructions/self-improvement.md`. Outputs of other
  tasks are not prerequisite inputs.

## Using another agent

`context` produces `context.md`, `messages.json` and `receipt.json`. The messages contain the general instructions and roles;
the receipt records every file hash and the final message hash. Any tool-capable host can pass the system and user messages to
its agent, and a CLI host can take `context.md`. The rules do not depend on the host discovering a hidden directory.

`supervise` is the continuous entry point, hosted by the run ledger; `drive` is a bounded foreground run:

```bash
python tools/pipeline_agent.py drive --work /data/runs/new-task \
  --driver /data/agent-driver.json --max-sessions 8 --timeout 1800
```

A driver file is `{"argv":["/opt/agent-wrapper"],"input":"messages"}` (`input` may also be `text`). The wrapper receives the
message JSON or text on stdin, works in the task directory, uses its own tool loop and returns an exit code. There is no shell
interpolation and no key goes in argv. A wrapper around a plain API must provide read/write, command, image and
continuous-execution tools. After exit, the harness rereads the actual state and outputs rather than trusting stdout. It runs
at most the explicit number of sessions, terminates the process group on timeout, and keeps a log and loading receipt per
session. It never loads other tasks' outputs automatically. Host long runs per the `datafarm-run-launch` skill.

`prepare` defaults to advisory cost mode. The enforced mode needs a driver that really limits all sub-calls and a
`cost_control` declaration, which an ordinary CLI cannot satisfy. See `prompts/instructions/task-cost.md`.

## Task state

`state.json` is the breakpoint and next-step note, not a success certificate. Keep `phase`, `status`, `next_action`,
`open_questions` and `external_jobs` current. Adapters, judges and media go in directories such as `task/` and `evidence/`.
`status` is only `working`, `needs-input` or `complete`; after `complete`, a `delivery.json` that passes
`python tools/pipeline_agent.py verify --work ...` must exist. Review tasks and author tasks start separately. The expert
harness still isolates the protected judge and holdout, and the orchestrator being able to write files must not let an author
read the hidden evaluation. See `prompts/instructions/task-artifact-contract.md`.

## The main line

Observe the complete episode → summarize the goal and object relations → fit and verify the scene, camera and physics →
generate the task and an independent judge → write the expert, initialization and adapter → small-budget rollouts, read logs and
videos, correct → freeze the holdout validation → complete a white-model small sample → bind real references and an explicit
mode → compile the task prompt → small-batch photographic generation and independent review → streaming expansion →
source-bound admission and data export → check count, alignment and failure records and deliver.

Every arrow has a checkable output. Code that compiles, provider success or a status string cannot skip the physical and video
review. When the user only wants data, the work ends at data delivery; training is not forced.

Ask only when a goal that cannot be derived is missing (for example, OOD or non-OOD is not stated), an external device fact truly
cannot be resolved, or access or resource authorization is insufficient. Do not re-ask when there is an explicit goal or standing
authorization, and keep doing work that does not depend on the answer while waiting.
