# Pipeline run ledger

## What it records

The ledger gives every requested invocation one **pipeline run** with a fixed item denominator and a status that is **derived**, not self-reported. An operator asks for a full pass ("generate the white episodes through to a trained checkpoint") or a partial pass ("take this cohort from render to admission"), and a supervisor's own `running` state may outlive its process.

| Question | Where the answer comes from |
|---|---|
| What was requested, by whom, over which items and stages? | `run.json` (fixed at creation) |
| Which item stopped at which stage, and why? | `matrix.json` (item × stage/substage cells) |
| Which processes/GPU leases work on it, and are they alive? | `workers.json` + live PID/lease probes + progress-file ages |
| What happened, in order? | `events.jsonl` (append-only) |

Runs live under `runs/ledger/<run-id>/` (git-ignored, like all run output). Run IDs are `PRUN-YYYYMMDD-HHMM-<slug>` in local time.

## Stages: two levels

Columns are the pipeline stages, in order:
`grounding → task → episode → render → admission → training`. A run covers a
contiguous range (`--from render --to admission`). Any stage may be split into
adapter-specific substages (`--substages render=global-style,guided-render,verify`),
which become columns `render/global-style`, … . The last column is the *exit column*.
An item is complete only when it has **passed** that column.

Cell states: `pending`, `running`, `passed`, `rejected` (verifier said no; may still
be repaired), `failed` (given up), `skipped`.

## Run states (derived by `status`)

| State | Meaning |
|---|---|
| `running` | at least one worker alive with fresh progress |
| `running_stalled` | workers alive but no progress file changed within `stale_seconds` (default 900) |
| `lost` | no live worker, and a worker died **without** an exit record: an unexpected exit |
| `exited_with_error` | a worker recorded a non-zero exit code |
| `blocked` / `failed` | the supervisor itself declared it (e.g. residual rejects need repair) |
| `idle_incomplete` | every worker ended cleanly but items remain unpassed |
| `idle_awaiting_finalize` | all items passed, no final verification recorded yet |
| `complete` / `complete_with_gaps` | finalized with a hashed verification report |
| `superseded` / `cancelled` | replaced by another run / stopped on purpose |
| `registered` | created, no worker yet |

A later worker registration (relaunch) overrides an earlier `blocked`/`failed`
declaration. Liveness of workers on other hosts is probed over SSH. A process that cannot be probed falls back to progress-file age.

## Operator workflow

```bash
cd /path/to/datafarm
# 1. Register the request before launching anything.
tools/datafarm_runs.py create --slug my-task-render100 --title "..." \
  --request "<the operator's request>" --authorization <id> --task-family my-task \
  --from render --to training --substages render=generate,dense-audit \
  --substages training=dataset,prepare,train \
  --items-file cohort-ids.json --cohort-manifest <manifest> --code-root .
# 2. Launch detached through the sentinel: it records PID/SID, heartbeats, and the exit code.
tools/datafarm_runs.py launch PRUN-... --role supervisor -- python tools/<supervisor>.py ...
# 3. Second, independent persistence check in a new shell after a short wait.
tools/datafarm_runs.py check PRUN-...
# Any time:
tools/datafarm_runs.py status            # all open runs
tools/datafarm_runs.py status PRUN-... --gaps -v   # every missing item and where it stopped
```

`launch` is the persistence-checked way to start a worker.
Its first check is recorded as a `persistence_check` event. `check` records the
second one. A repair or partial rerun of some items is a **child run**
(`--parent PRUN-...`) with its own item list, so the parent shows its children.

Supervisors started through `launch` see `DATAFARM_LEDGER_RUN_DIR` and
`DATAFARM_LEDGER_WORKER`. `datafarm.ledger.ledger.attach_from_env()` lets them write item
cells directly.

Manual interventions go into the log: `event --kind note --message ...`,
`close-worker RUN WORKER --reason ...` (e.g. a process replaced by hand),
`declare --status blocked|failed|cancelled --reason ...`, `supersede --by ...`, and
`finalize --verification <report>` (refuses unpassed items unless `--allow-gaps`).

## Native-state sync for other supervisors

A supervisor may keep its own state files. A run can declare `--sync
module:function@{json args}`. `status` then calls that function first, and it
translates native files into cells and workers. No sync functions ship; they are task code.

Sync functions only read native files. A sync error is shown in `status` and never
hides the rest of the report.

## Guarantees

- The item denominator is fixed at creation. Duplicate IDs, empty requests and updates outside the denominator or column plan are refused.
- Run status is derived: a worker that vanished without an exit record is `lost`, a non-zero exit is `exited_with_error`, and a relaunch overrides an earlier `blocked` or `failed`.
- `launch` runs the worker under a sentinel that records the exit code, and records the first persistence check.
- `finalize` needs an existing verification file and no gaps, unless `--allow-gaps`.

Code: `src/datafarm/ledger/`, `tools/datafarm_runs.py`.

## Known limits

- The ledger does not verify outputs. It records the verdicts that verifiers produce.
  `passed` means the owning verifier said so.
- Workers that were not started through `launch` have no exit code. When one
  disappears, the run shows `lost` until a native terminal state or an operator
  `close-worker` explains it.
- An unprobeable worker with stale progress is `unknown_stale` and counted as lost. Remote PID probes need
  `DATAFARM_SSH_CONFIG`; without it remote workers fall back to progress age.
- Ledger and batch locks use `flock`, whose coherence across hosts on a shared filesystem is not guaranteed.
