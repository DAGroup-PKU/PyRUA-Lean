# Batch supervisor: one bounded loop for per-item attempts

This supervisor manages execution state, retries and recovery. It does not choose the render mode
and does not judge whether real references were used; see the
[mode and real-style contract](../../prompts/instructions/mode-and-style-contract.md). A systematic early failure
pauses new submissions, and a target must not be met by unlimited retries or by shrinking the
denominator.

`src/datafarm/batch/supervisor.py` is the single loop controller for any batch that must
bring a fixed set of items (samples, episodes, shards) to an admitted result by launching
attempts, polling them, and verifying each result. Task differences (how to launch, poll and verify one attempt) are
injected; the controller owns attempts, seeds, budgets, liveness semantics, durable state and the
terminal status.

## 1. What it guarantees

- Attempt, crash, cost and zero-progress budgets, each with a terminal status.
- A failed lease query is `unknown`, never an empty set, and unknown never triggers a resubmit.
- Crashes retry the same attempt and seed. A burst backs off instead of stopping.
- Each item is verified as it finishes, and a systematic failure stops new launches early.

## 2. Words used here

- **Item**: one member of the batch, identified by a string id. The item list is the batch's
  **fixed denominator**: it is stored in `state.json` and a resume with a different list is
  refused (`DenominatorMismatch`). Every count is reported against all items.
- **Attempt**: one try of one item with one **seed**. Attempt numbers start at 1. The seed of
  attempt *n* is deterministic (`seed_for(item, n)`, default a hash bounded to int32).
- **Launch / try**: one start of an attempt. A crashed attempt is relaunched as the *same*
  attempt (same seed) with a new try number; its launch id is `<item>.aNN.tNN`.
- **Finished**: the attempt produced a result. It is verified immediately and uses up one unit
  of the item's **attempt budget**, whatever the verdict.
- **Verdict**: `pass` (admitted), `reject` (with the `gate` that failed; optionally `final`), or
  `abstain` (the verifier cannot decide; still not an admission).
- **Crash**: the attempt ended without a usable result (process died, stale or half-written
  result, launch failed). Crashes use a separate **crash budget** and never a seed. Kinds:
  `transient` (retry), `provider_down` (pause everything, no budget used), `permanent` (the
  request can never succeed as is: `rejected_final`).
- **Unknown**: the controller cannot tell whether an attempt is alive (status file unreadable,
  scheduler query failed, other host). Unknown attempts are never relaunched.
- **Round**: an attempt generation. Round *g* closes when every item that is still open is past
  attempt *g*. Crash retries do not start new rounds.

## 3. How one batch runs

The main line, one control tick at a time (`BatchSupervisor.step`, every `poll_interval_s`):

1. **Account time.** Active time grows unless the provider is marked down; then the time counts
   as provider pause instead. Supervisor downtime is never charged.
2. **Poll every attempt in flight.** `finished` → verify now (streaming verification);
   `crashed` → crash handling; `running` → alive; `unknown` → consult the optional lease probe:
   `alive` → running, `dead` on `lost_confirmations` consecutive polls → crash ("lost"),
   otherwise it stays unknown.
3. **Apply verdicts.** `pass` → `admitted`. `reject`/`abstain` → next attempt with a new seed if
   the attempt budget and `max_rounds` allow, otherwise `exhausted`; a `final` reject →
   `rejected_final`.
4. **Close rounds and check the guards**: zero progress, systematic failure, cost cap, wall
   budget, unknown-liveness bound, provider-pause bound.
5. **Decide the status.** A requested stop drains the attempts in flight (no new launches), then
   ends as `stopped_*`; a pause ends at once as `paused_*`. With nothing in flight and nothing to
   launch the batch is `complete` (all admitted) or `complete_with_gaps`.
6. **Launch** the due items (lowest attempt first, so every item's first attempt goes out before
   anyone's second one) up to `max_concurrency`, unless a crash-burst backoff or a provider
   outage is active. Before each launch the intent (item, attempt, seed, launch id) is written to
   `state.json`; the returned handle is written right after.

Example, one item: attempt 1 (seed s1) crashes during the carrier upload → relaunched as attempt
1 again (seed s1, try 2) after `crash_retry_delay_s` → finishes → the topology gate rejects it →
attempt 2 (seed s2) → finishes → passes → `admitted` with `attempts_used = 2`,
`crash_retries = 1`.

## 4. Policy

`BatchPolicy` is a frozen dataclass; `BatchPolicy.from_dict` rejects unknown keys, so a typo
fails instead of silently using a default. Its SHA-256 is stored in the state; a changed policy
on resume is recorded as a `policy_changed` event. `configs/batch/template.json` is a
starting point.

| Field | Default | Meaning |
|---|---|---|
| `max_attempts_per_item` | 8 | Finished attempts (seeds) per item. |
| `max_crash_retries_per_item` | 8 | Relaunches after transient crashes per item; seeds are kept. |
| `max_concurrency` | 10 | Attempts in flight, unknown ones included. |
| `max_rounds` | 12 | Hard ceiling on attempt numbers, also after `reopen`. |
| `zero_progress_rounds` | 3 | Stop after this many consecutive closed rounds without a newly admitted item (0 = off). |
| `max_total_cost`, `launch_cost_estimate` | off, 0 | Stop at the recorded cost cap; with an estimate, reserve it before launching. A provider can still charge more than the estimate. |
| `max_wall_seconds` | off | Active wall-clock budget (provider pauses excluded). |
| `crash_retry_delay_s`, `..._cap_s` | 30, 900 | Per-item delay before a crash retry, doubling per crash. |
| `crash_burst_items`, `crash_burst_window_s` | 5, 600 | This many *distinct* items crashing inside the window is a burst (0 = off). One item crash-looping is not a burst: its own delay and budget handle it. |
| `burst_backoff_s`, `..._cap_s` | 60, 1800 | Global launch pause after a burst, doubling while bursts repeat within the window; launches resume by themselves. |
| `provider_probe_interval_s` | 120 | While the provider is down: how often to probe (`provider_probe()`, or one half-open launch). |
| `max_provider_pause_s` | 21600 | Total provider pause before `paused_provider_down`. |
| `max_consecutive_unknown` | 20 | Polls of one attempt with unknown liveness before `paused_unknown_liveness`. |
| `lost_confirmations` | 2 | Consecutive dead-lease probes (without positive life) that make an attempt lost. |
| `systematic_min_items`, `systematic_window`, `systematic_share` | 8, 20, 0.6 | Over the latest verdict of the most recent 20 distinct items (once ≥ 8 exist), one gate failing ≥ 60% stops new launches (0 = off). |
| `max_verify_errors` | 3 | Verifier exceptions on one result before it counts as `abstain` (gate `verifier_error`). |
| `poll_interval_s`, `max_tick_gap_s` | 30, 600 | Tick period; longer gaps (supervisor down) are not charged to budgets. |
| `drain_on_stop` | true | Let in-flight attempts finish and be verified before a `stopped_*` status. |
| `seed_base`, `seed_modulus` | 0, 2147483647 | Default seeds; the modulus keeps them inside the int32 bound the provider enforces. |

## 5. States and statuses

Item states: `pending`, `launching` (intent persisted, outcome not known yet), `running`,
`unknown`, `crashed_retryable` (same attempt/seed next), `rejected_retryable` (next attempt next),
and the terminal `admitted`, `rejected_final`, `exhausted` (reason `attempt_budget`,
`crash_budget` or `max_rounds`).

Batch statuses (`running`, `stopping` while draining, then one terminal status):

| Status | When | Operator action |
|---|---|---|
| `complete` | every item admitted | none |
| `complete_with_gaps` | nothing left to try; some items exhausted or finally rejected | inspect `gaps`; `reopen` items after a fix |
| `stopped_zero_progress` | `zero_progress_rounds` rounds closed without a new admission | find the cause, then `acknowledge` |
| `stopped_systematic_failure` | one gate fails ≥ `systematic_share` of the recent window; the report lists gate, share and failing items | fix the gate/prompt/renderer, then `acknowledge` |
| `stopped_cost_cap` / `stopped_wall_budget` | budget reached (wall excludes provider pauses) | raise the budget in the policy, `acknowledge` |
| `stopped_operator` | stop requested (`request_stop`, the CLI `stop`/STOP file) | `acknowledge` |
| `paused_provider_down` | provider unavailable longer than `max_provider_pause_s` in total | `acknowledge` when it is back |
| `paused_unknown_liveness` | an attempt's liveness unknown for `max_consecutive_unknown` polls | check by hand; `acknowledge`, and `resolve_unknown(item, not_launched \| crashed)` if needed |
| `paused_cost_unknown` | a finished attempt has missing or invalid cost evidence when cost tracking is configured or a cost cap is active | restore the cost evidence, then `acknowledge`; the same result is re-read before verification, with no new provider request |

A stopped or paused batch stays stopped across restarts until someone calls `acknowledge`
(streaks and the systematic window restart; budgets do not). Other operator actions:
`reopen(items, extra_attempts)` for exhausted/final items, `revoke_admission(item)` when a later
cohort-level verifier withdraws an admission, `adopt_admitted(item)` for items admitted before the
batch existed.

## 6. Durable state and resume

`<state_dir>/`:

- `state.json`: the whole controller state, written atomically (temporary file, fsync, rename,
  directory fsync) on every change and at least every `heartbeat_s` (a liveness signal).
- `events.jsonl`: append-only history (`launch_intent`, `launched`, `verdict`, `crashed`,
  `crash_burst_backoff`, `provider_down/up`, `round_closed`, `stop_requested`, `terminal`, ...).
- `report.json`: the summary written at every terminal status.
- `supervisor.lock`: an exclusive `flock`; a second supervisor on the same directory fails with
  `SupervisorBusy` instead of racing the first one.

Resume never launches twice. The intent is on disk before `launch` runs, so after a supervisor
crash an item can be found in `launching` without a handle. On the first tick the optional
`reconcile(item, attempt, seed, launch_id)` looks the launch up (for the subprocess adapter: the
attempt directory). Found → it is polled like any running attempt. Confirmed not started → the
same attempt and seed are launched, without using any budget. Unknown, or no `reconcile` →
the attempt stays unknown, is never relaunched, and the batch pauses after
`max_consecutive_unknown` ticks for an operator decision. Attempts with a persisted handle are
simply polled again. A provisional verdict left by a runner killed while finalizing is treated as a crash
retry of the same attempt and seed.

## 7. Run ledger

When the supervisor runs under `tools/datafarm_runs.py launch` (environment `DATAFARM_LEDGER_RUN_DIR` and
`DATAFARM_LEDGER_WORKER`) and a ledger column is configured (`ledger_column`, or `ledger.column` in a
task config), every item change is mirrored through the ledger's public API
(`attach_from_env`, `RunLedger.set_cells`): running states → `running`, rejected with retries left
→ `rejected` (failure `gate:reason`), admitted → `passed` (artifact = verdict evidence), exhausted
or final → `failed`. A `stopped_*`/`paused_*` status is declared `blocked` with its reason. The
column is never guessed: without one, an event `ledger_not_mirrored` is written. Ledger errors are
recorded as events and never stop the batch.

## 8. Generic CLI: a supervisor becomes a config

`tools/supervise_batch.py` drives the controller with `src/datafarm/batch/subprocess_task.py`.
Each attempt is one command run through `batch_exec.py` in its own session and directory
`<work_dir>/attempts/<launch_id>/` (`attempt.json`, `handle.json`, `started.json`, `exit.json`,
`log.txt`), so a restarted supervisor can tell finished, crashed and running apart without being
the attempt's parent. Commands:

```text
tools/supervise_batch.py validate CONFIG        # check config and templates, print the first launch
tools/supervise_batch.py run CONFIG             # run to a terminal status; resumable
tools/supervise_batch.py status CONFIG [-v]     # read-only summary, no lock
tools/supervise_batch.py stop CONFIG --reason R # a running supervisor stops launching and drains
tools/supervise_batch.py acknowledge CONFIG --reason R
tools/supervise_batch.py reopen CONFIG --items A,B --reason R [--extra-attempts N]
tools/supervise_batch.py revoke CONFIG --items A,B --reason R [--gate dense_audit]
tools/supervise_batch.py resolve CONFIG --item A --outcome not_launched|crashed --reason R
```

`run` exits 0 (`complete`), 3 (`complete_with_gaps`), 4 (`stopped_*`), 5 (`paused_*`).

The task config (schema `datafarm.batch-task-config/1.0.0`, unknown keys rejected) holds: `items` or
`items_file`, optional per-item variables `item_vars`/`item_vars_file`, `work_dir`, the `launch`
command template, the `result` contract (`path`, `required_keys`, `match` against the attempt, and
`incomplete_values` for provisional statuses), `verify` (`result_status` mapping or a verifier
command printing `{"decision", "reason", "gate"}`), `crash_classes` (regex on the log tail →
`transient`/`provider_down`/`permanent`), optional `lease_probe`, `provider_probe` (exit 0 up,
exit 75 down), `unresolved_submission_marker`, `cost`, `seed` (`hash` or `linear`),
`ledger.column` and `policy`. Template fields: `{item} {item_index} {attempt} {seed} {launch_id}
{attempt_dir} {work_dir} {result_path} {vars[name]}`.

A verifier command must exit successfully before its JSON verdict is accepted. If it prints a
`pass` record and then exits nonzero, the supervisor records a verifier error and retries or
abstains under `max_verify_errors`; the item is never admitted from that partial output.

`launch.timeout_s` bounds one local attempt command. On expiry the wrapper stops that command's
process group and writes `exit.json` with `timed_out: true`. The supervisor treats the attempt as
unknown and pauses under `max_consecutive_unknown`; it does not automatically spend another seed
or resubmit a possibly still-running remote provider task. An operator must reconcile any remote
task ID and result before using `resolve` and `acknowledge` to continue.

An adapter that submits to a GPU scheduler should implement `lease_probe` so that a failed
`gpu-scheduler leases` query returns `unknown` (never an empty lease set), and `reconcile` by job
name, since a submit can time out after the job was queued.

For commands that submit remote paid work, set `unresolved_submission_marker` to a filename inside
each attempt directory. The command writes this marker **before** sending a request that may have
remote effects, and removes it only after persisting a definitive task ID or proving that no task
was created. If the command exits or disappears while the marker remains, the supervisor reports
`unknown` and pauses without another submit. A `dead` lease probe cannot turn an unidentifiable
remote submission into a safe retry. Reconcile the provider's task list and stored request evidence
before using `resolve`; a local process exit alone is insufficient.

### Per-item upstream readiness

`upstream_batch_state` optionally names another generic batch state JSON. The subprocess adapter defers each item until the same ID is admitted upstream, before creating an attempt directory or submitting work. Deferred items consume no attempt or crash budget and do not block ready items behind them. Workers still validate the source result and its artifact hashes; readiness is not an admission certificate. Missing/unreadable state is treated as waiting. This supports streaming dependent stages without interpreting an unfinished upstream item as a provider outage.

### Capacity refusals

`crash_classes` may map an explicit allocator refusal (for example `no matching free GPU found`) to `capacity`. Use only when the allocator confirms no job was submitted. Such waits keep the same seed, consume no quality/crash budget, use the provider probe interval as per-item backoff, and do not open the global provider breaker. Unknown submissions must still be reconciled.

## 9. Limitations

- `launch.timeout_s` bounds only a local attempt command. The controller has no per-attempt timeout of its own: an attempt that
  hangs while reporting `running` holds its slot, because relaunching without cancelling would run the same seed twice.
- A new verifier version is not re-applied to results already judged; `reopen` or `revoke` the
  affected items.
- Resubmission safety depends on the adapter. Resuming paid jobs by their recorded IDs holds
  only when the adapter supplies `reconcile` or `unresolved_submission_marker`. Without `reconcile`, a
  raising launcher is assumed not to have started anything and the attempt is relaunched.
- A dead lease can still trigger a relaunch after `lost_confirmations` probes even if the remote process
  survives on an unreachable host, unless the submission marker is used.
- Liveness of subprocess attempts is probed on the supervisor's host only; after a move to
  another host they are `unknown` until their exit record appears.

Code: `src/datafarm/batch/`, `tools/supervise_batch.py`.
