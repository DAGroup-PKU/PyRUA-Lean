# Task cost

All model costs of a new task, from the real input to the white-mesh data, share one persistent ledger. The default mode
is advisory, with a target of 30 USD and a reference limit of 50 USD, configurable before prepare. It covers orchestration,
scene and interface implementation, expert development, independent review, failure retries and repairs. Photographic
generation and GPU compute are recorded separately and are outside this model budget. The price basis, model/tier, caching
and unknown usage must be traceable. The target is not a promise that every new task can succeed within that amount.

The user's request governs. Never change the task definition, data count, mode or success criteria silently to save money.
A count that was not explicit is an execution assumption and must not be written as a user request. The quality and
independent-review requirements stay in force.

## How to work

- One main executor drives progress. The parent chat receives stage results; run no second scheduler and no per-minute
  model polling. Use programs for scheduling, log polling, counting and hash comparison. `pipeline_agent.py wait` waits for
  a substantive change of task state. Give progress updates the user asked for, but do not open a model subtask just to
  produce the same status.
- Load only the skills the current stage needs; do not inject all rendering, training, development and review rules every
  round. A stage handoff passes the input/code/contract hashes, a result summary, the main failures, output paths, the
  remaining budget and in-flight IDs. Raw logs stay in the outputs and are viewed on demand.
- Reuse the verified general robot interface, collection, clock, image encoding and statistics tools. The semantics,
  adapter, expert and judge of a new task are still generated from current evidence; reusing tools never copies another
  task's authorization or skips this task's verification.
- First confirm with 2-3 discriminating samples that the necessary physical actions are feasible, and only then start
  expert trial and error. Repair the largest failure group first. Do not let the expert burn budget on a low-level action
  interface known to fail.
- Deterministic checks with the same source, contract, judge and environment hashes can be reused. When something changes,
  redo only the affected checks, then the required final independent acceptance. The visual review of an earlier sample
  cannot replace the review of a new video.
- Label general infrastructure work and routine task runs separately. Infrastructure spending incurred for a task still
  counts toward that task's end-to-end total; changing the category does not remove it. Opening separate infrastructure
  work is a new scope and does not enlarge the budget.

## Modes

`prepare` pins `configs/task-cost-policy.json` and saves the policy hash and a persistent ledger.

- `disabled`: no threshold scheduling, but all sessions, known costs and unknown liabilities are still saved.
- `advisory` (default): hints from target and limit. `on_excess=warn` continues; `pause` pauses once the known cost reaches
  the limit. Unknown costs are marked, and you must not claim the total is within budget.
- `enforced`: needs a metered driver that can limit itself and all its child calls. Reservations, unsettled amounts, unknown
  receipts and overruns block continuation. A post-hoc token estimate cannot replace a hard limit.

Native Codex uses the existing login and saves JSONL token usage. Without a reliable dollar receipt, `cost_usd=null` and the
total cost is unknown. The native CLI does not satisfy enforced, and declaring `cost_control` does not make it so. Missing
usage is never zero, and the known cost is only a subtotal. `disabled` does not delete costs or cancel the physical, video,
independent-review and provenance contracts. Record an overrun as it is.

Never rewrite the policy or its hash, delete the ledger, or create a new task to remove earlier costs. Adjust the quota of a
new task before prepare. A configuration change during a run keeps the control history and does not reset the task policy,
the session count or the retry state. When model consumption outside the driver is not covered, mark the end-to-end cost as
incompletely covered. A claim of cost savings needs a measurement on the same task with the same data volume, difficulty and
judge; software tests only show budget and state behaviour.

## Waiting without a model

`pipeline_agent.py wait` only polls `state.json`. It calls no model and changes no task state. It returns when
phase/status/next_action/questions or the ID/status/error of `external_jobs` changes, not on a heartbeat. `complete` is only
a status notification; the delivery must still be verified.

When the only next step is waiting for a submitted job, write `"model_wait_for": ["job-id"]` in `state.json` and save the same
ID and the real status in `external_jobs`. Before creating a session, `drive` checks these dependencies and returns
`waiting-external-jobs` until they end. The external supervisor keeps monitoring the real process and updating the status. A
heartbeat, an unknown status or an observation timeout must not be written as a failure followed by a duplicate submission.
When independent work remains, do not set this field; clear it after the dependencies are handled.

Protocol details: `docs/reference/cost-control.md`.
