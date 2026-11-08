# transfer: shared libraries from learn campaigns

The learn loop (`python -m pyrualean.learn.campaign`) writes one library per task: skill modules
(`skills/*.py`, importable by the actor as `library.<module>`) and memory cards
(`lessons/*.md`, appended verbatim to the actor's prompt). This directory turns the libraries of
finished campaigns into libraries for tasks they were not learned on (paper, appendix "Library
Transfer" and "Why Cross-benchmark Libraries Help Little"):

| Stage | Directory | From | To |
|---|---|---|---|
| cross-task distillation | `robodojo/` | the libraries of N RoboDojo source tasks | one RoboDojo library (off-10 / on-10 in the paper) |
| cross-benchmark translation | `xbench/` | one distilled RoboDojo library | a LIBERO-PRO (`libero_rgb`) or RoboCasa365 (`robocasa_rgb`) library |
| in-benchmark distillation | `robocasa/` | the libraries of N RoboCasa365 source tasks | one RoboCasa365 library (off-rc10 / on-rc10) |

Every stage is the same pipeline: `prep.py` collects read-only inputs into a working directory,
`brief.py` writes the prompt `BRIEF.md`, `run_worker.sh` runs **one** headless Claude Code session
that writes `out/`, `check.py` (and a smoke test against a stub robot) accepts or rejects `out/`,
and `freeze.sh` freezes it as `library-v1/`. Both settings of the paper go through the same
scripts: w/o CE (campaigns run with `--worker-scoring off`, the learner saw only success) and
w/ CE (`--worker-scoring on`, it also read the benchmark's criteria). The two briefs of a stage
differ in one passage on what the learners saw; w/ CE tells the distiller that a rule that only
describes how one task is judged does not transfer.

Nothing here starts a simulator. `run_worker.sh` is the only script that calls a model.

## Files

| File | Role |
|---|---|
| `run_worker.sh` | one distillation session for a working directory (all stages) |
| `settings.json` | the session's Claude Code settings: no hooks, auto-memory, git instructions or plugins |
| `freeze.sh` | `DIR/out` -> read-only `DIR/library-v1` + `library-v1.sha256`, copied to a destination |
| `status.py` | progress of a session: tool calls, permission denials, `out/` files, cost when done |
| `robodojo/prep.py` | inputs from RoboDojo learn campaign roots |
| `robodojo/BRIEF.wo_ce.md`, `BRIEF.w_ce.md` | the briefs (placeholders: task list and count, dev layouts, API path, working directory) |
| `robodojo/brief.py` | fills a brief into `DIR/BRIEF.md` |
| `robodojo/check.py` | static acceptance checks of a RoboDojo library |
| `robodojo/fake_robodojo.py` | stub `robodojo_xpl_rgb` robot for smoke tests |
| `xbench/render.py` | renders a benchmark's system prompt the way its evaluation builds it, without the task card |
| `xbench/prep.py` | inputs: the source library, both system prompts, both API sources |
| `xbench/BRIEF.template.md`, `brief.py` | the translation brief (placeholders: source library, setting sentence, target, working directory) |
| `xbench/check.py` | static checks against the `libero_rgb` or `robocasa_rgb` API |
| `xbench/fake_libero_rgb.py`, `fake_robocasa_rgb.py` | stub target robots for smoke tests |
| `xbench/cards/` | one task card per robot: only its robot and native-action budget enter a rendered prompt |
| `robocasa/prep.py` | inputs from RoboCasa365 learn campaign roots, plus the `robocasa_rgb` system prompt and API |
| `robocasa/BRIEF.wo_ce.md`, `BRIEF.w_ce.md`, `brief.py` | the in-benchmark briefs |

## Setup

```bash
source tools/env.sh                          # sets CE_ROOT and PYTHONPATH=$CE_ROOT/src
export CLAUDE_BIN=$CE_ROOT/tools/claude-clean   # optional: the learner's clean config dir
```

The Python scripts need only this repository (`pyrualean` from `src/`) and the interpreter you
use for it (`$CE_PYTHON`, docs/SETUP.md); no simulator, benchmark checkout or GPU.

The session uses `$CLAUDE_BIN` (default `claude`; `tools/claude-clean` keeps your CLAUDE.md,
memory, skills and plugins out of it, as for the learner, see docs/MODELS.md) with the CLI's own
login or your `ANTHROPIC_API_KEY`, and the model `$LEARNER_MODEL` (default `claude-opus-5-5`,
effort high, as in the paper). It runs
with `--setting-sources project,local`, so your user settings do not apply, and in
`--permission-mode dontAsk` with these rules only:

- tools `Read, Glob, Grep, Write, Edit, Bash`; Bash only `ls`, `wc` and `python3 -c`;
- read `DIR/**`; write and edit only `DIR/out/**`;
- `robodojo/` (`run_worker.sh --api`): also read `$CE_ROOT/src/pyrualean/**`, which the brief
  names as the robot API source. `xbench/` and `robocasa/` copy the API files into `DIR/inputs/`.

A session writes `DIR/worker.stream.jsonl` (stream-json), `DIR/worker.stderr` and
`DIR/worker.exit`; a directory with `worker.exit` is not run again. Run sessions detached, e.g.
`setsid nohup sh transfer/run_worker.sh DIR > /dev/null 2>&1 &`, and follow them with
`python transfer/status.py DIR`.

## 1. RoboDojo: N source tasks -> one library

Inputs per source task (`DIR/inputs/<task>/`): the campaign's active revision as the actor read it
(`library/<task>/views/<rev>/<task>/{skills,lessons}`, retired cards removed), the last worker's
`FINDINGS.md` (as `FINDINGS-last.md`) and the dev successes per round from `report.json`
(`dev_rounds.txt`). `SOURCES.json` records campaign, revision and dev curve per task;
`MANIFEST.json` the campaign roots, the number of dev layouts and the sha256 of every input. A
task may be in only one of the campaigns given; give the campaigns of one setting.

```bash
# the paper's ten source tasks: groups 1 and 2, learned once per setting (docs/REPRODUCE.md §1)
W=runs/transfer/rd10-wo_ce
python transfer/robodojo/prep.py $W runs/learn/wo_ce/group1 runs/learn/wo_ce/group2
python transfer/robodojo/brief.py $W --setting wo_ce
sh transfer/run_worker.sh --api $W
python transfer/robodojo/check.py $W              # task names of $W/SOURCES.json; --names adds more
python my_smoke.py $W/out/skills/<module>.py       # a smoke test of this library (see Acceptance)
sh transfer/freeze.sh $W runs/transfer/off-10      # -> runs/transfer/off-10/library-v1

# w/ CE: the same with the campaigns run with --worker-scoring on
W=runs/transfer/rd10-w_ce
python transfer/robodojo/prep.py $W runs/learn/w_ce/group1 runs/learn/w_ce/group2
python transfer/robodojo/brief.py $W --setting w_ce
sh transfer/run_worker.sh --api $W
python transfer/robodojo/check.py $W
sh transfer/freeze.sh $W runs/transfer/on-10
```

`prep.py ... --tasks T1 T2 ...` takes a subset of the campaigns' tasks.

**Evaluate** with `eval/`: in the plan of `eval/export.py`, a task with `"library": DIR` (here
`runs/transfer/off-10/library-v1`) runs that library instead of a revision of its campaign, and
`"settings_task"` lets a target task no campaign learned run with a campaign's settings (see the
docstring of `eval/export.py`); then `python eval/run.py ROOT run` / `report`. Use target tasks
that no source task covers. The paper's plan is `configs/eval/robodojo/transfer-rest10.json`
(`TRANSFER_LIBRARY=runs/transfer/off-10/library-v1`), commands in docs/REPRODUCE.md §5.

## 2. RoboDojo library -> LIBERO-PRO or RoboCasa365

The source is a frozen library from stage 1 (a `freeze.sh` destination: `library-v1/`,
`NOTES.md`, `SOURCES.json`). `prep.py` copies it to `DIR/inputs/source/` with the RoboDojo system
prompt (`render.py`: `robodojo_xpl_rgb`, `no-vla`, on-demand images, pure feedback, 5400 s, empty
library, task card removed) and API (`robodojo_xpl_rgb.py`, `robodojo.py`, `_robot.py`), and puts
the target's system prompt and API in `DIR/inputs/target/`: LIBERO-PRO with eval_libero's settings
(`libero_rgb`, `no-vla`, on-demand, pure, 7200 s; `libero_rgb.py`, `_robot.py`) or RoboCasa365
with eval_robocasa's (`robocasa_rgb`, same settings, the horizon of the atomic task in
`cards/robocasa_rgb.json`; `robocasa_rgb.py`, `_robot.py`). `--source-card` / `--target-card`
replace the cards. The distiller gets no target task list, trajectories or judging code.

```bash
for T in libero robocasa; do
  W=runs/transfer/xbench-off-10-$T
  python transfer/xbench/prep.py runs/transfer/off-10 $W --target $T
  python transfer/xbench/brief.py runs/transfer/off-10 $W --setting wo_ce --target $T
  sh transfer/run_worker.sh $W
  python -I transfer/xbench/check.py --target $T $W
  sh transfer/freeze.sh $W runs/transfer/off-10-$T
done
# w/ CE: runs/transfer/on-10 with --setting w_ce
```

**Evaluate** with `eval_libero/` (`arms`: `{name: {library: DIR}}`; the paper's
`eval_libero/configs/xbench-*.json` take `LIBERO_LIB_WO_CE` / `LIBERO_LIB_W_CE`, e.g.
`runs/transfer/off-10-libero/library-v1`, next to the empty library as `v0`) or `eval_robocasa/`
(`eval_robocasa/configs/heldout10-*.json`: `ROBOCASA_LIB_XBENCH_WO_CE` / `_W_CE`); see their
READMEs and docs/REPRODUCE.md §6.

## 3. RoboCasa365: N source tasks -> one library

As stage 1, from RoboCasa365 learn campaigns (`configs/learn/robocasa/atomic5.json` and
`composite5.json` in the paper). `dev_rounds.txt` also lists each round's revision and the
rounds that evaluated the active one, and says when the last worker's `FINDINGS.md` describes a
revision that was not kept. `DIR/inputs/robot/` holds the `robocasa_rgb` system prompt (rendered
as in stage 2) and API. The brief adds articulated fixtures and base/torso use to the card
stages and allows kitchen fixture and object categories in `applies:`.

```bash
W=runs/transfer/rc10-wo_ce
python transfer/robocasa/prep.py $W runs/learn/wo_ce/atomic5 runs/learn/wo_ce/composite5
python transfer/robocasa/brief.py $W --setting wo_ce
sh transfer/run_worker.sh $W
python -I transfer/xbench/check.py --target robocasa --lib $W/out --prompt $W/inputs/robot/system_prompt.md
sh transfer/freeze.sh $W runs/transfer/off-rc10
# w/ CE: the campaigns run with --worker-scoring on, --setting w_ce, -> runs/transfer/on-rc10
```

**Evaluate** with `eval_robocasa/` (`arms`) on RoboCasa365 tasks disjoint from the source tasks:
`eval_robocasa/configs/heldout10-*.json` take `ROBOCASA_LIB_RC10_WO_CE` / `_W_CE`
(`runs/transfer/off-rc10/library-v1`, `runs/transfer/on-rc10/library-v1`).

## Limits the briefs set

- `out/lessons/*.md`: at most **12,000 characters** in total, cards with the fields `stage`,
  `symptom`, `applies`, `do`, `evidence` (`## id` heading); a general rule each, no task, object,
  dimension or goal configuration in `do:`; `evidence:` names the source tasks (or source cards)
  and should rest on two of them, or on one where it decided between failing and succeeding
  rounds, flagged as single-task.
- `out/skills/*.py`: optional, at most **2 modules, 15,000 characters** in total; task-agnostic
  helpers on the real robot API, a docstring per function saying when to use it.
- `out/NOTES.md` (not given to the actor): the source of every card and skill and what was left
  out (stage 2: also the source-to-target API mapping).
- At least one file in `out/lessons/` or `out/skills/`.

## Acceptance

Before a library is evaluated:

1. `check.py` passes (exit status 0).
   - `robodojo/check.py DIR` (or `--lib LIB --names SOURCES.json ...`): the size limits, the card
     fields, no source task name in a card's `do:`/`applies:` or in a skill, every
     `library.<module>[.<name>]` a card names exists, every skill parses and imports.
   - `xbench/check.py [--target libero|robocasa] DIR` (or `--lib LIB --prompt PROMPT`): the same
     limits and fields, and the target API: every `robo.<name>(...)` in skills (AST) and cards
     (text) exists on the target robot with arguments that bind to its signature, is not a VLA
     primitive, `state().<field>` exists, camera names are the target's, and no RoboDojo-only
     words (`home(`, `path=`, `arm=`, `cam_*` on LIBERO; another arm than `"right"`, RoboDojo or
     LIBERO cameras, `done`/`world_map` on RoboCasa365) appear in a card's `do:`; no RoboDojo task
     name, and no LIBERO-10 object or RoboCasa365 target task name in `do:`/`applies:`/skills
     (generic object words, and objects the target prompt itself names, are warnings only).
     Stage 3 uses it with `--target robocasa --lib`.
2. A smoke test of each skill module against the stub robot of its benchmark passes:
   `robodojo/fake_robodojo.py`, `xbench/fake_libero_rgb.py`, `xbench/fake_robocasa_rgb.py`. Each
   binds every call to the real robot class's signature (a wrong argument raises), returns the
   real result dataclasses (a wrong field raises) and moves ideally unless `block_below` stops a
   descent. A smoke script is written per library, for its own functions:

   ```python
   import sys

   sys.path.insert(0, "transfer/robodojo")
   from fake_robodojo import check, load, report

   m = load(sys.argv[1], "moves")
   check("grasp", lambda r: m.grasp_from_above(r, "left", 0.1, 0.0, 10.0, 0.80), lambda o: o["ok"])
   check(
       "grasp, blocked",
       lambda r: m.grasp_from_above(r, "left", 0.1, 0.0, 10.0, 0.80),
       lambda o: not o["ok"],
       block_below=0.82,
   )
   sys.exit(report())
   ```

   (the xbench fakes' `check` takes `expect(result, fake)`.)

Skills are only smoke-tested against these stub robots, **not in simulation**: a passing smoke
test means the calls and result fields are right, not that a skill works on the robot. Passing
the checks does not mean a library helps; that is what the evaluation measures.

## Freezing

`sh transfer/freeze.sh DIR DEST` copies `DIR/out/lessons/*.md` and `DIR/out/skills/*.py` (no
`__pycache__`, nothing else) to a read-only `DIR/library-v1`, writes `DIR/library-v1.sha256`, and
copies `library-v1`, `BRIEF.md`, `out/NOTES.md`, `SOURCES.json` and `MANIFEST.json` (when present)
to the new directory `DEST`, checking the copy against the hashes. Evaluations take
`DEST/library-v1`; stage 2 takes `DEST`.
