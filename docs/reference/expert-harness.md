# Task-expert harness

The harness is the shared, task-independent runtime in which a coding agent writes a task expert (a
controller program) against a frozen judge. It runs agent sessions in an OS sandbox, executes every rollout
itself through a broker, records every attempt and its cost, freezes the judge and inputs by hash, and
evaluates the submitted experts on hidden held-out seeds after the round.

Package: `src/datafarm/expert/`; command line: `tools/expert_harness.py`; adapter template:
`configs/expert-harness/template.json`. Authoring method: `.agents/skills/task-expert-author/SKILL.md`.

## 1. Role and messages

The role prompt is in `prompts/roles/task-expert-author.md` and is loaded explicitly by the launcher. The first, continue and final session messages stay separate, and backend and model come from the round's adapter. A round is a program: its budgets, sandbox and rollouts are enforced by code, every number comes from one config, and every outcome is recorded by the harness rather than by the agent.

## 2. One round in one paragraph

The operator writes (once per task) a **task adapter** (start from `configs/expert-harness/template.json`)
and a **brief template**. `prepare-round` renders the brief from the template and the
adapter, lints it (fail closed), copies the operator's hidden holdout seeds into a
harness-private directory, installs the agent tool `xh`, and seeds the workspace from the
previous round's **handoff bundle**. `run-round` then starts the agent CLI (Claude Code or
Codex, headless) inside an **OS sandbox** in which it can read only the listed inputs and
write only its workspace. The agent's only way to run anything is `xh`: `xh rollout` drops
a request that the harness validates (development seed, concurrency, budget, static audit
of the expert file) and turns into a GPU job; the job runs the expert inside a second
sandbox and the verifier outside it. The harness records every attempt, resumes the agent
session after each break, stops at the budgets, and hashes the judge, physics and holdout
files at start, after every session and before holdout evaluation. After the round,
`eval-holdout` runs every submission on the hidden seeds (a score-vs-cost curve),
`handoff` assembles the bundle and the next brief, and `library add` stores a verified
expert.

```text
adapter + template ──prepare-round──> round/ (brief.md, bin/xh, workspace/, harness/ [private])
                                         │
run-round:  agent CLI in sandbox ── xh rollout/submit/flag ──> broker (harness process)
              ▲  resume per session         │ validate, snapshot, audit, record
              │  cost from result event     ▼
              │                         job.sh: probe → sim in sandbox → judge outside
              └── status prompt ◄───── attempts ledger / results / status.json
eval-holdout ─> harness/holdout/results.json (curve)   handoff ─> bundle + next brief
library add ─> content-addressed verified experts      suite ─> pass@k, pass^k, $/verified
```

## 3. The parts

| Module | What it does |
|---|---|
| `adapter.py` | One validated JSON per task: task/IR, verifier and exact criteria, runtime facts, readable and protected paths, rollout and judge commands, dev seeds, holdout file, budgets, agent backend and model. Absolute paths live only here. |
| `brief.py` | Render from template + adapter + handoff; `lint_brief` fails if a required section is missing (task, interface, verifier with version, every judge point and "AND", runtime limits, budgets matching the adapter, tools, readable paths, forbidden actions, deliverables, submit protocol, holdout policy, abort channel), if a runtime fact or criterion token is missing, if a masked path inside a readable root is not named, if a protected path is listed as readable, or if any holdout seed value appears. |
| `isolation.py` | `bwrap` (preferred): allowlist sandbox for the agent (only system dirs, readable inputs and round files; workspace, agent home and request box writable), root-read-only sandbox for rollouts; protected paths masked; **capabilities always dropped** (bwrap started by root keeps them, and `CAP_SYS_ADMIN` can simply `umount` a mask); optional switch to an unprivileged uid with `setpriv`. `unshare -m` fallback with the same masks (denylist semantics). A probe inside the exact sandbox checks every mask (after an `umount` attempt), read-only dirs and writable dirs before anything runs; a leak aborts the round (agent sandbox) or the rollout (rollout sandbox, recorded as infrastructure). |
| `toolbox.py` + `agent_tool.py` (`xh`) | The broker validates each request (development seeds only; a holdout seed is a recorded violation; concurrency from live attempts, not leases; counted budget includes rollouts in flight; infra cap; wall budget; round state), snapshots and audits the expert, launches the job, and appends launch/end records to `ledger/attempts.jsonl`. Symlinks and path escapes are refused (the harness reads agent files without following links). `submit` snapshots experts (acks only; warns when an expert never ran); `flag-for-human` ends the round cleanly. |
| `audit.py` | AST audit on every rollout request and submission: syntax, class present, import allowlist, no dynamic code or file I/O built-ins, no introspection escape hatches, no module file I/O (`np.load`, `np.save` ...). Necessary, not sufficient. |
| `rollout.py` | One generated `job.sh` per attempt: heartbeat, environment digests, isolation probe, simulator **inside** the sandbox (run-dir root read-only), judge **outside** it (verifier code is masked from agent code), `harness-job.json` last. Liveness comes from a heartbeat file, never from scheduler leases. Outcome classes: judged / expert error (counted); invalid start, simulator start-up failure, judge failure, lost, timeout, isolation failure, submit failure, abandoned (infra, not counted). Backends: `command` (`gpu-scheduler submit`), `local`, `fake`. |
| `session.py` | Headless sessions (`claude -p --output-format stream-json --verbose --model M --session-id/--resume`, tool allowlist, `--max-budget-usd` = remaining budget; `codex exec --json` / `exec resume` with the prompt on stdin), transcript streamed to disk, per-session wall cap (SIGTERM, then SIGKILL, to the process group), cost/usage/turns/session id from the `result` event (price-table estimate, marked as such, only for killed sessions), error classification (API outage vs authentication vs other). |
| `round.py` | `prepare_round`, `RoundRunner.run`: stop on final submission, flag, wall / USD / session / counted-rollout budget (with one final "submit now" session), infra cap, idle sessions, tampering or a fatal agent error; provider outages back off 60→120→240→300 s and after 10 min pause the round (exit 75, resumable); outage and downtime are not charged; one `datafarm.agent-call/1` record per session via `call_log.append_call` (component `expert-harness/session`), including aborted and crashed sessions; drain in-flight rollouts; automatic final submission if the agent made none. |
| `manifest.py` | SHA-256 of judge/verifier code, the verifier contract package, runner, physics/material/generator files, the holdout seed list, adapter, template, brief and tool, at start, after every session and before holdout. A change invalidates the round unless recorded first with `tools/expert_harness.py amend`. |
| `holdout.py` | Harness-only: seeds drawn from a secret offset (`make-holdout-seeds`), stored privately, run after the round for the final expert and each submission (deduplicated by digest), spare seeds replace invalid starts in the same family, infra retried; results and a score-vs-cost curve stay in `harness/`. |
| `handoff.py` | Frozen expert (by digest), REPORT/NOTES (links not followed), harness attempt ledger, failure taxonomy, agent-safe holdout summary (per-family counts and failure labels only), costs, problems detected in the round; renders and lints the next brief. Refuses to write any holdout seed value. |
| `library.py` | Content-addressed store of experts that passed holdout at the adapter's threshold, with task id, brief and Task-IR digests, verifier version, holdout scores, model and cost; digests verified on read; `prepare-round --library L --references N` retrieves references. |
| `evalsuite.py` | Tasks × models × repeats; each cell is a full round; per group: pass@k and pass^k over rounds (unbiased estimators), mean holdout pass rate, cost per verified expert, wall time, counted rollouts, violations (tamper, isolation, concurrency, budget, holdout-seed and non-development requests, audit and unsafe-path rejections). Tasks without an adapter are listed with the interfaces they lack. |

## 4. Where things live in a round

```text
<round>/round.json brief.md bin/xh handoff-in/ references/     public, read-only for the agent
<round>/workspace/ agent-home/ tmp/ broker/requests/            writable by the agent (and nothing else)
<round>/broker/{responses,results,status.json} rollouts/ submissions/ ledger/   harness-written, agent-readable
<round>/harness/   state, transcripts, manifests, holdout seeds, holdout runs and results   (masked everywhere)
```

Put rounds under `$EXPERT_HARNESS_HOME/rounds/`; `prepare-round` refuses a
round directory inside a protected path. Holdout seed files belong in
`$EXPERT_HARNESS_HOME/private/` (list that directory in the adapter's `workspace.protected`).

## 5. Running a real round (operator)

All commands run from the repo root on a host that can submit GPU jobs. `DATAFARM_PYTHON` can point at any
Python >= 3.11 (the harness is standard library only).

```bash
cd /path/to/datafarm
export DATAFARM_PYTHON=/path/to/venv/bin/python
# Check that this interpreter exists and meets the version requirement before running.
H=tools/expert_harness.py; A=/data/task/adapter.json    # start from configs/expert-harness/template.json
X=$EXPERT_HARNESS_HOME                                # e.g. /data/expert-harness

# 0. checks that cost nothing
$H lint-brief --adapter $A
$H dry-run                                        # scripted agent + fake simulator, real sandboxes

# 1. hidden holdout seeds (CPU only)
mkdir -p $X/private/my-task
$H make-holdout-seeds --adapter $A --per-family 4 --spares 2 --out $X/private/my-task/holdout.json

# 2. prepare the round (add --handoff <bundle> to continue from an earlier round)
$H prepare-round --adapter $A --round-dir $X/rounds/my-task-1

# 3. the round itself (needs approval: agent API spend and GPU rollouts)
export ANTHROPIC_API_KEY=...   # or the variables your agent CLI uses; passed by name, never logged
nohup $H run-round --round-dir $X/rounds/my-task-1 \
   --approve-agent-spend --approve-gpu > $X/rounds/my-task-1.log 2>&1 &
$H status --round-dir $X/rounds/my-task-1

# 4. after the round (GPU)
$H eval-holdout --round-dir $X/rounds/my-task-1 --approve-gpu
$H handoff --round-dir $X/rounds/my-task-1 --out $X/handoff/my-task-1
$H library add --library $X/library --round-dir $X/rounds/my-task-1
```

Operational notes:

* **Before the first live round**, run one smoke rollout (one GPU job, needs approval):
  `$H smoke-rollout --adapter $A --expert <expert.py> --class <ExpertClass> --seed 11
  --out $X/smoke/1 --approve-gpu`. It uses the same job template, sandbox and judge as a round and
  prints the outcome, the job record and the isolation probe. The rollout sandbox makes `/` read-only
  and only the listed caches writable (for Isaac Sim: `isaacsim/kit/{data,logs}`, `$HOME/.cache`,
  `.nv`, `.nvidia-omniverse`, `.local/share/ov`). If the simulator needs another writable path, add
  it to `isolation.rollout.writable`; if a file owned by another user is unreadable without
  capabilities, add `"keep_caps": ["CAP_DAC_OVERRIDE"]` (never `CAP_DAC_READ_SEARCH` or
  `CAP_SYS_ADMIN`; the harness refuses them). Isaac's NVIDIA runtime may also need a private writable
  `/run/user/0`, mounted as a per-job `tmpfs` alongside `{isaac_kit}/cache`. Simulator output should
  go under `native_workdir/native` and move to `native` only after simulation finishes, because the
  simulator creates its own output directory and exits if `native` already exists.
* **Authentication.** The agent runs as `nobody` with its own `HOME` in the round; it cannot
  read `~/.claude`. Pass credentials as environment variables named in
  `agent.auth.env_passthrough`, or bind a settings file read-only with `agent.auth.ro_bind`
  (then set `isolation.agent.run_as` to `null` if the file is root-only). A failure to
  authenticate stops the round with status `error` before any work.
* **Pausing.** SIGTERM or Ctrl-C ends the current session, persists state and exits with 75;
  running `run-round` again resumes the same conversation. Downtime is not charged. A
  provider outage longer than 10 minutes pauses the round the same way. If the harness process
  dies, the agent process tree dies with it (`PR_SET_PDEATHSIG`, bwrap `--die-with-parent`); the
  next `run-round` closes the interrupted session with an `aborted` agent-call record.
* **Changing a protected file mid-round:** stop the harness,
  change the file, run `tools/expert_harness.py amend --round-dir R --reason "..."`, then resume. Without
  the amendment the round becomes `invalid_tamper` at the next session boundary.
* **Codex.** `prepare-round --backend codex --model <model>`. The session uses
  `codex exec --json` and `codex exec resume`; no price is listed, so its cost is recorded as
  unknown (not zero) unless `configs/expert-harness/prices.json` lists one. Codex CLI has no per-session USD hard cap,
  so `run-round` refuses a Codex round whose adapter declares `max_usd`; a price-table estimate
  alone would not enforce a hard cap during a long session. A Codex round requires explicit
  approval for unpriced spend and an adapter with `max_usd: null`, retaining session, wall and
  rollout bounds.

## 6. Eval suite

```bash
$H suite plan   --suite suite.json --out /data/suite
$H suite run    --suite suite.json --out /data/suite --fake                                   # plumbing only
$H suite run    --suite suite.json --out /data/suite --approve-agent-spend --approve-gpu --only <cell-id>
$H suite report --out /data/suite
```

A round is *verified* when its final expert passes at least `verified_pass_rate` (default 0.5) of
the hidden holdout seeds. pass@k = P(at least one of k independent rounds is verified),
pass^k = P(all k are verified). Re-run the suite before and after any change to a brief
template, prompt, skill, model or the harness. A suite may declare tasks without an adapter as
`unavailable`, listing the interfaces they lack (plugin loader, out-of-process judge on a recorded
trace, seeded randomized starts, protected-file list, brief template). The suite file is an output
of the evaluation you are running; no suite is shipped.

## 7. Approvals

Nothing in the harness spends money or GPU time unless told to: `run-round` refuses a
non-fake agent without `--approve-agent-spend` and the `command` rollout backend without
`--approve-gpu`; `eval-holdout` and `suite run` do the same. Approve the agent spend of a round
against the adapter's `budgets`, and its GPU time against its rollout count; holdout evaluation adds
one rollout per submission and seed.

## 8. Guarantees

1. An adapter loads only if it is complete and well-formed: schema, slug `task_id`, all sections
   (`brief`, `verifier`, `seeds`, `budgets`, `agent`, `rollout`, `workspace`, `isolation`, `expert`,
   `holdout`), verifier `version`/`name`/`judge_points`/`criteria`, a known `agent.backend` with an
   explicit `agent.model`, rollout backend fields, isolation modes, `expert.allowed_imports`, absolute
   workspace paths; an `unavailable` adapter must list `missing_interfaces`.
2. A `task_ir` with `profile: "stage-graph/1"` is validated at adapter load (stages with exit
   predicates and recovery, known roles/predicates, ordered sync points, explicit tolerances,
   forbidden shortcuts, skill evidence) and, when `verifier.criterion_ids` is given, every judge
   criterion must be covered by a terminal/transient predicate and no predicate may cite an unknown
   one. A summary-form `task_ir` is not validated.
3. Held-out seeds must be disjoint from development seeds; the round keeps a private copy (mode
   0600) and the brief is refused if it fails lint (including mentioning held-out seeds).
4. The broker runs a rollout only on a development seed; a held-out or other seed is rejected and
   recorded as a violation.
5. Concurrency is counted from live attempts, the counted-rollout budget includes rollouts in
   flight, infrastructure failures are not counted but capped, and the wall budget closes the round.
6. Every rollout and submission reads the expert as a regular, non-symlinked file inside the
   workspace, passes a static audit (import allowlist, no file I/O or dynamic code), and is
   snapshotted by SHA-256 into a harness-written ledger.
7. The judge and everything feeding a verdict are frozen by hash: verifier files, adapter manifest
   files, the private held-out seed copy, brief, `xh` tool, brief template, adapter file and the
   harness code digest are hashed at round start and re-checked after each session, on resume, at
   the end and before held-out evaluation; any change marks the round `invalid_tamper` unless an
   operator first records an amendment with a reason.
8. A prepared round runs only with the adapter it was prepared with (same SHA-256).
9. Agent sessions and rollouts run in a sandbox: `bwrap` (allowlist or read-only root) or an
   `unshare` fallback; with no usable mode the harness fails closed; an allowlist spec degrades to
   `unshare` only with `allow_degraded_fallback`; capabilities are dropped or the process switches
   uid via `setpriv`; protected paths and the private `harness/` directory are masked; secrets reach
   the child only by named environment passthrough. Mode `none` is allowed only when configured and
   is recorded as a violation.
10. Held-out evaluation runs only after the round ended without tampering; each chosen submission's
    snapshot is digest-checked and run fresh on every held-out seed with the development job
    template and isolation; infrastructure failures are retried and invalid starts are replaced by a
    spare seed of the same family.
11. No paid work starts without explicit flags: a non-fake agent needs `--approve-agent-spend`, the
    `command` rollout backend needs `--approve-gpu`, and a `codex` round may not declare `max_usd`.
12. The `codex-metered` backend requires `agent.pricing` naming the same model as `agent.model`;
    the check runs when a session starts.
13. Only the final expert of an untampered round whose held-out pass rate reaches the threshold
    enters the expert library; entries are content-addressed and re-verified on read; handoff
    bundles refuse to leak held-out seeds.
14. The audit of a recorded joint trace (`tools/audit_joint_margin_trace.py <trace.json> --output <report.json>`) reports
    per-arm, per-phase, per-joint margins and 1/5/10° bands, and `recorded_trace_passed` requires the policy's joint target, all actual
    joints and all commands within limits, complete commands and no control-step gap. The task's collector writes the trace:
    `{"binding": {"physical_joint_limits_rad", "sample_hz", "policy": {"target_joint", "target_margin_deg"}}, "records":
    [{"step", "phase", "left", "right", "commands": {"left": {"qpos_rad"}, "right": {"qpos_rad"}}}]}`. Take the physical limits
    from the intersection of the URDF and the robot vendor's SDK limits, and record where they came from.
15. Both delivery verifiers require every episode review to carry `joint_margin: pass` among its
    verdicts.

Code: `src/datafarm/expert/`, `src/datafarm/simulation/joint_margin_audit.py`, `src/datafarm/task/contract.py`.

## 9. Data model

- **Adapter** (`datafarm.expert-harness.adapter/1`, `configs/expert-harness/template.json`): `task_id`,
  `status`, `vars`, `task_ir`, `brief.template`, `verifier` (`version`, `name`, `judge_points` ⊆
  `release|settled|terminal|any`, `criteria`, `files`, optional `criterion_ids`), `seeds.dev`
  (`ranges`, `explicit`; optional `families`, `family_command`), `holdout` (`seeds_file`, `targets`,
  `max_targets`, `max_concurrent`, `retry_infra`, `verified_pass_rate`, `timeout_s`), `budgets`,
  `agent` (`backend` claude|codex|codex-metered|fake, `model`, `pricing` for codex-metered), `expert` (`default_file`,
  `default_class`, `max_bytes`, `allowed_imports`, `reject_unchanged_rollouts`), `workspace`
  (`readable`, `protected`), `isolation.{agent,rollout}` (`mode`, `fallback`, `base`, `run_as`,
  `allow_degraded_fallback`, `extra_masked`), `rollout` (`backend` command|local|fake, `sim.argv`,
  `judge`, `verdict.file`, `verdict.success`, `submit.argv`), `manifest.files`, `capabilities`.
- **Stage-graph task IR** (`task_ir.py` docstring): `demand`, `roles`, `stage_graph.{stages, sync}`,
  `predicates` (`kind` exit|guard|transient|terminal, `expr`, `tolerance`, `judge_criterion`),
  `forbidden_shortcuts`, `skill_evidence`, optional `strategy_space`.
- **Round directory** (`layout.py` docstring): public `round.json`, `brief.md`, `bin/xh`; agent-writable
  `workspace/`, `broker/requests/`; harness-written, agent-readable `broker/responses|results`,
  `rollouts/<attempt>/`, `submissions/<id>/`, `ledger/attempts.jsonl`, `ledger/events.jsonl`,
  `ledger/agent-calls.jsonl`; private `harness/` (state, transcripts, manifests, held-out seeds,
  held-out runs and `holdout/results.json`).
- **Broker schemas**: `datafarm.expert-harness.request/1` (`request_id`, `kind` rollout|submit|flag-for-human,
  `args`), `.response/1`, `.attempt/1`, `.submission/1`, `.status/1`; held-out seeds
  `datafarm.expert-harness.holdout-seeds/1` (`seed`, `family`, `spare`); manifest
  `datafarm.expert-harness.manifest/1`; library entry `datafarm.expert-harness.library-entry/1`.
- **Joint-margin trace**: `{binding: {urdf_sha256, physical_joint_limits_rad, planning_joint_limits_rad,
  policy, sample_hz}, records: [{step, phase, left, right, commands}]}`; requirements in
  `configs/joint-margin-requirements.json`.

## 10. Configuration

- Wrapper: `DATAFARM_PYTHON` (default `python3`); `PYTHONPATH` gets `src` prepended.
- Budgets (required: `wall_per_session_s`, `total_wall_s`, `max_counted_rollouts`,
  `max_concurrent_rollouts`, `max_sessions`); defaults `max_usd null`, `max_infra_failures 20`,
  `max_idle_sessions 3`, `final_grace_s 300`, `wait_cap_s 200` (≤ 240), `warn_before_wall_s 900`.
- Held-out defaults: `targets all`, `max_targets 6`, `retry_infra 2`, `max_concurrent 8`,
  `verified_pass_rate 0.5`; `make-holdout-seeds` `--per-family 4 --spares 2 --range-start 100000
  --range-size 1000000`.
- CLI: `run-round --tick 1.0`, `eval-holdout --poll 15`, `--isolation bwrap|unshare|none` overrides;
  exit code 75 means paused and resumable.
- Model and pricing have no defaults (`agent.model`; `agent.pricing` for codex-metered). Deployment
  bindings, scheduler and agent CLIs: [deployment](../deployment.md).

## 11. Failure behaviour

- Invalid adapters raise `ConfigError`; harness refusals raise `HarnessError` (CLI prints JSON, exit 1).
- Rejected requests get a response with a code (`not_dev_seed`, `holdout_seed`, `budget`,
  `concurrency`, `audit`, `unsafe_path`, `unchanged_expert`, `round_closed`, ...); violations are
  counted in the round state. Nothing is deleted: attempts, failures, transcripts and manifests stay.
- Tampering ends the round as `invalid_tamper`; held-out evaluation and library admission are then
  refused. Recovery requires an operator amendment before the change, or a new round.
- An infrastructure-failure cap, a failed isolation probe or an auth error ends the round and needs
  an operator.

## 12. Metered Codex backend

`agent.backend: "codex-metered"` connects the isolated author and the `xh` broker to a trusted local Responses gateway. Set `agent.model`, `agent.pricing`, `agent.credentials_file` (the deployment's protected credentials file) and a finite positive `budgets.max_usd`. Credentials are not copied into the author's home or mounted into its sandbox. Keep the author in the allowlist sandbox and verify the actual mounts before a paid run.

The harness starts the gateway outside the author sandbox and routes the CLI through it. The gateway serializes requests, reserves a conservative bound before contacting the provider, journals response IDs and upstream usage, and drains requests before writing a session receipt. Receipts and journals live under the protected harness sessions directory. Report their price-table basis separately from supplier billing and from the cost of any coordinating parent agent. The author receives the explicit role bundle and the first, continue and final briefs, automatic project-document loading is disabled, and the CLI compacts context at 64,000 tokens through the same gateway and budget.

Cost and transport rules:

- A locally rejected request with no upstream call costs zero. A local request-bound denial ends the round for inspection.
- Missing upstream usage stops the round with `cost_unknown`; it is not a free retry. The upstream usage receipt is the charging authority.
- The gateway buffers the provider's complete JSON response before exposing its output items and terminal status through a local SSE stream ([event reference](https://developers.openai.com/api/reference/resources/responses/streaming-events)). This is not a retry, and a nonterminal or malformed response blocks further spend.
- The backend tracks whether an HTTP write started after TLS establishment. Only a failure before any write settles at zero generation cost; failed writes and missing responses retain unknown liability. It records no headers, payload bytes or credentials and does not retry generation requests.
- `agent.provider_receipt_lookup: openai_compatible` enables read-only reconciliation after a response transport exception: the gateway looks up the exact server request ID and requires a matching model and a recognized usage schema. A matched receipt settles API-equivalent cost but reports missing output as a temporary service failure, and never invents a response. Lookup retries at most three times on connection or timeout errors; HTTP errors or unmatched receipts do not retry, and the liability stays unknown.
- `agent.upstream_read_timeout_s` sets an idle read timeout of 30–900 seconds (default 900). `agent.upstream_stream_max_s` optionally limits time spent consuming a streamed response (30–900 seconds), checked on each incoming SSE line. On expiry, the response closes and the same lookup applies.
- A final journal event proving a pre-write failure or a reconciled lost response marks a recoverable interruption, even if earlier tool calls succeeded. Costs and rollout records stay charged, and recovery resumes the conversation. For an unmetered CLI session, error text counts as an outage only if the session made no tool calls.
- After an operator audits a terminated session with missing usage, the harness can retain its full reservation in `bounded_unknown_costs` and resume only against the budget beyond it. Known spend and retained liabilities are reported separately, and new unknown sessions still stop.

Two operator controls apply to any backend:

- `expert.reject_unchanged_rollouts: true` makes the broker reject a completed, counted expert/class/seed combination before allocating another trial. `xh rollout --repeat` permits deliberate replication. Leave it off when an expert depends on mutable files outside its source hash.
- The protected file `harness/operator-diagnostic.txt` (UTF-8, at most 8192 bytes) is given to each later session after the standard instructions, with a private copy and hashes recording what was delivered. The author cannot write it, and symlinks are rejected. Never include credentials, hidden seeds or judge internals. It does not change the task, budgets, physics or verdicts. Record operator help as an engineering contribution when comparing runs.

## 13. Limitations

- **No snapshot search.** A backend such as Isaac particle cloth has no save/restore or live-query interface. The adapter template declares `capabilities.sim_snapshot_restore: false` and `query_state: false`, and the broker offers only whole seeded rollouts. Bit-exact replay of GPU cloth is impossible; fresh authoritative replay is the harness-run held-out evaluation.
- **In-process tampering is not prevented.** The expert plugin runs inside the simulator process. OS isolation hides files but cannot stop code reaching simulator objects in memory. The static audit and the verifier running outside the sandbox narrow this, and do not close it.
- **Network egress** of the agent is not blocked at the OS level, because the CLI needs its API; the tool allowlist excludes web tools.
- **Denylist rollouts.** The rollout sandbox shows the host read-only minus masks, so an unlisted sensitive path is readable by expert code (not by the agent, whose sandbox is an allowlist). Keep `workspace.protected` and `isolation.rollout.extra_masked` complete.
- **Timing of cost.** A session's cost is known when it ends, so the curve reports the cost before and through the session that made each submission.
- **Only an adapter template is shipped** (`configs/expert-harness/template.json`). Each task's adapter is an output of the run.
- **The stage-graph profile and `criterion_ids` are opt-in.** Without `criterion_ids`, criterion coverage is not checked.
- **The judge's independence is a role instruction,** stated in `prompts/roles/task-designer.md`, `prompts/roles/task-verifier.md` and the authoring skill. The code freezes whatever judge files the adapter lists, by hash. It does not check who wrote them, that they implement the contract, or that `criteria` text matches the verifier code.
- **Production episodes are not harness-produced.** The harness runs held-out evaluation. White batch collection is done by task code, and the delivery verifiers do not check harness round IDs, library entries or held-out results.
- **Per-episode scene variation is task code.** The shared code provides camera-pose and grasp randomization plans (`src/datafarm/simulation/randomization.py`); table height and tilt are in the task's scene sampler.
- **Joint-margin auditing assumes one robot layout:** six joints on dual `left` and `right` arms and a joint target, with a trace the task's collector writes. `configs/joint-margin-requirements.json` declares itself a review contract, not an automatic admission gate. Delivery checks a reviewer's `joint_margin` verdict, not the audit artifact.

## 14. Joint margin

New or replanned robot data must meet the joint-margin requirements in `configs/joint-margin-requirements.json` before scaling collection or rendering, or declaring data ready. Full executed joint trajectories, phase-specific occupancy and dwell statistics, and a validated task profile are required review evidence.
