# Agent runtime

The agent runtime turns a user request and a real episode into a resumable task workspace and
runs a supplied coding agent on it in bounded sessions. It compiles each session's
instructions from the pinned capability layer, writes an execution record before launching,
dispatches queued reviews to separate reviewer sessions, waits for external jobs without
calling a model, and accepts `complete` only when the delivery verifier passes. Task state
(`state.json`) is the agent's resumable notebook, not a success certificate.

Related: [capability layer](capability-layer.md) (context compiler and source pin), [run ledger](run-ledger.md) (used to launch `supervise`), [batch supervisor](batch-supervisor.md), [cost control](cost-control.md) (budget modes, reservations, receipts), [capability evolution](capability-evolution.md) (`feedback`, post-run evolution). The handoff rules agents follow are in [`prompts/instructions/runtime-handoff.md`](../../prompts/instructions/runtime-handoff.md).

## Guarantees

1. **Inputs are locked at `prepare` and re-checked before every session.** Every episode and
   reference file is hashed into `input.json`; a changed file stops `context`/`drive`/`verify`.
2. **`complete` is accepted only through the delivery verifier.** `drive` returns
   `verify_delivery(work)`; an invalid delivery raises and the command exits 2.
3. **`needs-input` must carry concrete questions.** `open_questions` must be a non-empty list of
   non-blank strings or objects with `question`; otherwise `ValueError`. Which questions qualify
   (intent, device facts, access that cannot be derived) is a role instruction, not code.
4. **One controller per workspace.** `supervise` holds `.supervise.lock`; `drive` holds it too
   (unless called by `supervise`) plus `.driver.lock`, both non-blocking `flock`.
5. **The execution record is written before the process starts and in-flight work is
   reconciled before any new model call.** A live local worker → `waiting-worker`; a saved
   native thread ID → `interrupted` with `recovery_session_id`; a native result → status
   recovered; anything else → `recovery-required`.
6. **Executions launched on another host are not resumed here.** Records carry `host` and
   `boot_id`; an in-flight record from another host returns `recovery-required` unchanged, and
   `runtime-status` reports `supervisor_alive`/`worker_alive` as `null` off-host.
7. **Session and retry limits survive restarts.** `supervise` increments `sessions` and saves
   `dispatch_after` before calling `drive`; a dispatch with no execution record afterwards is
   `recovery-required`; transient-failure retries are bounded by `max_retries`.
8. **Review requests are frozen, source-bound and dispatched before author work.** Each
   `state.review_requests[]` entry needs a safe `id`, `subjects` (workspace files with matching
   SHA-256) and `instructions`; it is copied once to `.runtime/reviews/<id>/request.json` and
   may not change afterwards. The next session runs as `task-verifier`.
9. **A review completion is host-written and re-validated on every use.** It binds the report,
   execution record, transcript, identity file and context hash (`receipt.json`
   `messages_sha256` = execution `context_sha256`); the verdict cannot be upgraded; subjects and
   evidence must still hash the same.
10. **Reviewer and author sessions cannot share a provider identity.** Generic drivers must
    declare `datafarm.agent-identity/1` before a review is launched; identities are registered per
    `session_key` (`author` or `review-<id>`), a session ID reused across keys is refused, a
    non-fresh session must be provably the same thread, and a native review may not reuse the
    author thread.
11. **Delivered episode reviews are bound to runtime review completions.** In a runtime-managed
    workspace each episode review must cite `runtime_review: <id>` whose completion validates,
    passed, covered every delivered artifact hash, and lists this review file in its evidence;
    a hand-written execution record is refused. Outside the runtime the record is accepted and
    labelled. The result field `review_independence` is `runtime-verified` only if every episode
    is, else `declared-execution-record`.
12. **Waiting on external jobs does not call a model.** While at least one job in
    `model_wait_for` is not yet successful and none has failed (`failed`, `error`, `cancelled`,
    `needs-input`), `drive` returns `waiting-external-jobs` and `supervise` polls state;
    unreadable status files never mean failure; success with a non-zero `exit_code` becomes `failed`.
13. **Stop terminates the local process group, not external jobs.** `stop` writes
    `.runtime/stop.json`; the controller kills the session group (TERM, then KILL) and marks it
    `stopped`; external jobs are left running.
14. **Budget gates precede launch.** An enforced budget refuses the native driver and any driver
    without a descendant-wide `datafarm.session-cost/1` cap; any non-disabled budget (advisory too)
    refuses `--evolution-driver`.
15. **Published progress counts only verified deliveries.** `completed_episodes` is non-zero only
    when the last result is `valid` and `complete`; heartbeats do not change `last_substantive_at`.

Code: `tools/pipeline_agent.py`, `src/datafarm/agent/runtime.py`, `src/datafarm/agent/native_codex.py`, `src/datafarm/agent/task_wait.py`, `src/datafarm/agent/progress.py`, `src/datafarm/task/contract.py`.

## Data model

| File | Contents (as written by code) |
|---|---|
| `input.json` | `schema: datafarm.episode-input/1`, `files[{path, sha256}]`, `sha256`, `request`, `target_count`, `stop_after` (`render`/`white`), `cost_policy_sha256` |
| `state.json` | `schema: datafarm.agent-state/1`, `status` one of `working`, `needs-input`, `complete`, `phase`, `next_action`, `open_questions`, `external_jobs[{id, status, status_file?, ...}]`; optional `model_wait_for[]`, `review_requests[{id, subjects[{path, sha256}], instructions}]` |
| `capability.json`, `AGENTS.md`, `task/`, `evidence/`, `sessions/` | created by `prepare` |
| `sessions/<ns>/` | `messages.json`, `context.md`, `receipt.json`, `execution.json`, `stdout.log`, `stderr.log`; native: `native-command.json`, `native-result.json`, `native-stderr.log`; `identity.json`; `cost-receipt.json` |
| `execution.json` | `status` (starting → running → returned/failed/timeout/stopped/stop-failed; recovery: interrupted), `started`, `ended`, `input_kind`, `context_sha256`, `role`, `host`, `boot_id`, `pid`, `process_identity`, `exit_code`; optional `review_id`, `native_state`, `cost_reservation_*`, `cost`, `recovered`, `recovery_session_id` |
| `identity.json` | `schema: datafarm.agent-identity/1`, `session_id`, `session_key`, `role`, `fresh`, `evidence[{path, sha256}]` (inside the session folder), optional `origin` |
| `.runtime/` | `supervisor.json` (heartbeat, `supervisor_pid/identity/host`, `worker`), `control.json` (`sessions`, `retries`, `config_history`, `status`, `last_result`, `dispatch_after`, `retry_at`), `stop.json`, `driver.json`, `identities.json`, `author.json` and `review-<id>.json` (native thread state), `status-output.json` |
| `.runtime/reviews/<id>/` | `request.json` (frozen), `report.json` (`verdict` pass/reject/unverified, `subjects`, `evidence`, `findings`), `completion.json` (`verdict`, `report_sha256`, `session`, `execution_sha256`, `context_sha256`, `role`, `transcript_sha256`, `identity_sha256`, `native_session`) |
| `native-result.json` | `schema: datafarm.native-session/1`, `session_id`, `exit_code`, `usage` (last `turn.completed`), `cost_usd: null`, `cost_status: unknown`, `includes_descendants: false`, `errors` |

## How it works

`drive` (one foreground controller):
1. Take locks; `recover_inflight`; return early if a worker is live or recovery is needed.
2. Per session: stop check; `verify_input`; cost gate; pick a pending review (role
   `task-verifier`) or the author (`orchestrator`). `complete`/`needs-input` with no pending review
   return `verify_delivery` / `needs_input_result`; an external wait returns `waiting-external-jobs`.
3. Re-verify the pinned source; compile context into a new `sessions/<time_ns>/`; write
   `execution.json` (`starting`, host, boot ID); reserve cost; start the driver in a new session
   with the payload on stdin; record PID and start time.
4. Heartbeat each second; on stop/timeout kill the group; afterwards SIGKILL the group anyway.
5. Settle cost; record the result event; register identity (native or identity protocol);
   finalize a review (`completion.json`).

`supervise` (persistent): load config; write `control.json`, `driver.json`; loop: stop → kill
in-flight group; refresh external jobs; wait on jobs in bounded slices; for `complete`/`needs-input`
call `drive` once and finish; else reserve a session and call `drive(max_sessions=1)`;
transient failures back off `min(backoff_max_seconds, backoff_seconds * 2^(n-1))`.

Native Codex (`native_codex.py::run`): argv `codex exec [resume <id>] --json
--skip-git-repo-check [--model M] -c sandbox_mode=... [-c model_reasoning_effort=...] [-c k=v] -`;
the thread ID from `thread.started` is saved to the namespace state file at once, so a crash after
the first event is resumable by explicit ID. `messages` input is joined into one prompt.

General driver: `argv` (list, no shell); receives the payload on stdin and `DATAFARM_SESSION_DIR`,
`DATAFARM_AGENT_ROLE`, `DATAFARM_AGENT_SESSION_KEY`, `DATAFARM_AGENT_IDENTITY_RECEIPT` (and cost variables
when budgeted); with `session_identity_protocol: datafarm.agent-identity/1` it writes `identity.json`.
[`examples/runtime/agent_cli_driver.py`](../../examples/runtime/agent_cli_driver.py) does this for Claude Code and Pi.

## Entry points

```bash
python tools/pipeline_agent.py prepare --episode /data/ep.mp4 --request-file /data/request.txt --output /data/task --target-count 100
python tools/pipeline_agent.py context --work /data/task --output /data/task-ctx
python tools/pipeline_agent.py drive --work /data/task --driver /data/driver.json --max-sessions 4
python tools/pipeline_agent.py supervise --work /data/task --config examples/runtime/native-codex.json
python tools/pipeline_agent.py stop --work /data/task --reason "pause"
python tools/pipeline_agent.py runtime-status --work /data/task
python tools/pipeline_agent.py verify --work /data/task
python tools/pipeline_agent.py cost-status --work /data/task
python tools/pipeline_agent.py wait --work /data/task --timeout 600
python tools/pipeline_agent.py feedback --work /data/task --file /data/event.json
```

Configurations are JSON, with no shell interpolation and no embedded credentials. [`examples/runtime/native-codex.json`](../../examples/runtime/native-codex.json) is a general template.

`supervise` is meant to be launched detached through `tools/datafarm_runs.py launch`. Exit codes: 0;
75 for `failed`, `timeout`, `needs-input`, `session-budget-exhausted`, `cost-budget-blocked`,
`recovery-required`, `stopped`, `stop-failed`, `error`; 2 for refused input or validation errors.

## Configuration

| Key / flag | Default | Source |
|---|---|---|
| runtime `schema` | `datafarm.agent-runtime/1` (required) | `agent/runtime.py::load_config` |
| `max_sessions`, `timeout`, `poll_seconds`, `heartbeat_seconds` | 16, 1800 s, 5 s, 10 s | `load_config` |
| `max_retries`, `backoff_seconds`, `backoff_max_seconds` | 3, 5 s, 120 s | `load_config` |
| `driver` (required), `working_directory`, `log_directory`, `status_output{path, task_id}` | none | `load_config`, `drive`, `supervise` |
| driver `type: codex` options | `executable` codex, `sandbox` workspace-write (or read-only, danger-full-access), `model`, `reasoning_effort`, `settings` (`features.*`/`model_*`/`service_tier*`, scalars) | `native_codex.py::command` |
| driver `input` | `messages` (or `text`) | `drive` |
| driver `cost_control` | required when enforced: `protocol`, `includes_descendants: true`, `max_session_usd > 0` | `drive` |
| `drive --max-sessions`, `--timeout` | 1, 1800 s | `pipeline_agent.py::main` |
| `prepare --stop-after`, `--cost-policy`, `--reference` | `render`, `configs/task-cost-policy.json` (advisory), none | `main` |
| `wait --timeout`, `--interval` | 1800 s, 5 s | `main` |

Field notes:

- `max_sessions` is a cumulative cap shared by author, review and failure retries, and a restart does not reset it. `timeout` applies per driver run. `poll_seconds` checks external status and does not call the model.
- `working_directory` is where the model's tools run; the task identity and output directory stay the task. `log_directory` stores logs per session ID, with a link kept in the sessions directory.
- `status_output` (`{"path": ..., "task_id": ...}`) atomically merges this supervisor's state into a shared monitor file. `last_substantive_at` and `substantive_sha256` exclude heartbeat timestamps.
- If the model is omitted from a `codex` driver, the deployment configuration applies.
- Runtime configuration changes append to `config_history`; session and retry counts are kept.

## Failure behaviour

- Refused before launch: non-empty `prepare` output, credential-like directories in inputs,
  changed inputs or pinned source, invalid state status, argv not a list of strings, a review with
  an argv driver lacking the identity protocol, budget violations.
- Kept: every session folder, execution record, transcript, review request/report/completion,
  cost reservation and settlement, `config_history`.
- Operator needed: `recovery-required` (foreign host, launch without saved session ID, reviewer
  returned without a report, interrupted dispatch); `stop-failed`; invalid `complete` or
  `needs-input` without questions (raises; `supervise` records `status: error`); non-transient
  `failed`/`timeout` or exhausted retries. `supervise --resume` clears an explicit stop only.

## Limitations

- **One native driver.** Only Codex is built in; other agents use the argv driver and
  the identity protocol, whose provider evidence the driver itself supplies. The example CLI driver
  starts a fresh session each time and does not report cost.
- **Same-UID processes are not a security boundary.** Locks, hashes and identity files detect
  changes; they do not stop a same-user process from writing workspace files.
- **No write isolation for reviewers.** Reviewers use the same driver configuration (same
  sandbox type and working directory) as the author. Independence means a separate session and
  identity plus content binding; write permissions are not separated. Reviewer instructions forbid edits; edits to subjects are detected.
- **Runtime-managed detection is by marker.** `runtime_managed` checks `.runtime/reviews`,
  `author.json` or `control.json`; a workspace driven only by `drive` with an argv driver and
  no review requests has none, so a declared execution record is still accepted (labelled declared).
- **Single host per workspace.** Task locks are `flock`; PID liveness uses `/proc` on the
  supervisor's host. `boot_id` is recorded but not compared; only `host` decides `foreign_host`.
- **Unbounded external waits.** Jobs in `queued`, `running` or `unknown` keep `supervise`
  waiting with no wall-clock bound; `wait` returns `wait-timeout` without changing state.
- **`needs-input` categories are not checked.** Any concrete question passes the code check.
- **Native usage is not cost.** `native-result.json` has `cost_usd: null`; budgeted native
  sessions settle as unknown (see [cost-control.md](cost-control.md)).
- **Transient detection is lexical.** Only `failed` sessions whose logs contain capacity,
  rate-limit or 429/502/503-like text are retried; timeouts are never retried by `supervise`.
- **Delivery checks are contract checks.** `verify_delivery` checks bindings, membership and
  review records; physical and visual quality come from the reviews it binds.
