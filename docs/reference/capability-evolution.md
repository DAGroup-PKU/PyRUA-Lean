# Capability evolution

The capability layer (skills, prompts, harness code) can be improved from recorded feedback without hand-editing a running deployment. A coding driver edits a copy of the active source. The host then evaluates the parent and the candidate with a frozen evaluator, and promotes the candidate only if it improves a declared metric without regressing any other. Promotion is a compare-and-swap of the active-version pointer. Running tasks keep their pinned source, and any version that was ever active can be restored by rollback.

The rules agents follow are in [`prompts/instructions/self-improvement.md`](../../prompts/instructions/self-improvement.md).

## Guarantees

1. **Feedback is recorded once, scoped, and kept.** Events are idempotent by content, carry selectors for where they apply, and can be superseded but not deleted. Importing the same event ID again writes nothing, and the same ID with different content is an error.
2. **A candidate may change only allowed, unprotected paths.** The protected list (`PROTECTED` in `src/datafarm/evolution/controller.py`) includes `AGENTS.md`, the normative rules in `prompts/instructions/`, standing rules and `evaluations/`, the expert-harness holdout, isolation and audit modules, the admission-certificate issuer, and the evolution package and its CLI. A policy can add `protected_paths`. Interpreter hooks (`sitecustomize.py`, `usercustomize.py`, `*.pth`) are refused, and a new top-level module under `src/` must start with `datafarm_`, because the evaluator runs the candidate's `src/` on `PYTHONPATH`. Scope is checked at proposal time and again from the stored sources before evaluation.
3. **The evaluator is frozen by content.** `freeze_policy` copies the evaluator assets into the store and names the policy by its digest. `check_policy` re-hashes them before every use.
4. **Promotion requires a paired comparison on the same denominator.** Both evaluations must be valid and the candidate's checks must pass, `case_ids` must be identical, no metric may regress beyond its tolerance, and at least one decisive metric must improve by at least `min_improvement`.
5. **No self-certification.** The driver writes only a candidate source and a proposal. It cannot write results. Failed and rejected candidates stay in the store.
6. **Activation is compare-and-swap, and rollback keeps run pins.** A candidate whose parent is no longer active becomes `stale`. A crash between writing the pointer and updating the state is recovered without re-running evaluations. Rollback only targets versions that were active, and tasks already bound to a version keep it.
7. **Lessons count only when they are evaluated explicitly.** A policy that tests lessons must set `tests_lessons` and pass `{experience}` to a check. Lessons are injected into a task's context only within their selectors.

Code: `src/datafarm/evolution/`.

## Data model

The store directory (`--store`) holds:

- `events/`: feedback records;
- `sources/`: content-addressed source snapshots;
- `versions/`: `{source, parent, lessons, evaluation}`;
- `candidates/`: `datafarm.evolution-candidate/1`: parent, source, events, policy, proposal, lessons, changed_paths;
- `states/`: `proposed`, `evaluating`, `rejected`, `verified` (shadow evaluation only), `promoted`, `stale` or `evaluation-error`;
- `policies/<digest>/`: frozen evaluator assets;
- `evaluations/`, `activations/`, `active.json`, `cycles/`.

An evaluation policy (`datafarm.evolution-policy/1`) contains `allowed_paths` and `protected_paths`, `checks` (`[{id, argv containing {repo} and {report}, timeout}]`), `metrics`, `evidence_level` (`software`, `simulator`, `new-task` or `meta-improvement`), `assets` (evaluator files, relative to the policy file) and optionally `tests_lessons`.

## Using it

Put the store outside the source directory or inside `.datafarm`. Raw videos, full outputs and models do not go into the source version.

```bash
python tools/pipeline_evolve.py --store /data/evolution init
python tools/pipeline_agent.py prepare --episode /data/episode.mp4 --request-file /data/request.txt \
  --target-count 100 --output /data/task --evolution-store /data/evolution --scope /data/task-scope.json
```

The scope is an optional exact-match selector such as `{"task_family":"my-task","hardware":"my-robot"}`. A `run_id` is added automatically. Without `--evolution-store`, `prepare` pins the invoked source in the repository's `.datafarm/evolution`; with it, the store's active version is used.

### Recording feedback

The host saves an event and imports it when a user correction, a manual patch, a check or a task ends:

```json
{
  "id": "host-session-7-message-12",
  "kind": "correction",
  "statement": "The request did not use the real reference the task requires; check the chain from the contract to the request.",
  "source": {"actor": "user", "ref": "session-7/message-12"},
  "details": {"request_receipt": "/data/task/request-receipt.json"}
}
```

```bash
python tools/pipeline_agent.py feedback --work /data/task --file /data/feedback.json
python tools/pipeline_evolve.py --store /data/evolution status
python tools/pipeline_evolve.py --store /data/evolution record-change \
  --statement "Fix the reproduced empty-input failure" --source-ref session-7/manual-fix \
  --evidence /data/test-output.txt
```

`record-change` compares the active parent with the current source and saves before and after snapshots and the changed paths. It does not promote. Use `--parent` to record a fix for an inactive development branch. Task sessions record their own results.

### Driver and evaluator

The driver uses the same argv and stdin form as the task entry point, and runs in the candidate source copy:

```json
{"argv": ["/opt/your-agent", "run"], "input": "messages"}
```

The evaluation policy and its code and data are frozen into the store first, so a candidate cannot replace them by rewriting its own tests. `check.py` below is provided by the deployment and computes metrics for the real improvement target. `argv` is not interpreted by a shell and must not contain credentials.

```json
{
  "schema": "datafarm.evolution-policy/1",
  "evidence_level": "software",
  "allowed_paths": ["src/", "tools/", "tests/", ".agents/skills/", "prompts/"],
  "assets": ["check.py", "cases.json"],
  "checks": [{
    "id": "replay",
    "argv": ["python", "{suite}/check.py", "--source", "{repo}", "--output", "{report}"],
    "timeout": 120
  }],
  "metrics": {"replay.correct_cases": "max", "replay.failed_cases": "min"}
}
```

A metric rule is `min`, `max`, or a noise-aware object such as `{"direction": "max", "min_improvement": 0.10}` or `{"direction": "min", "tolerance": 0.10, "relative": true, "decisive": false}`. `tolerance` is the maximum allowed regression (default 0), `relative` scales it by the parent value (default false), `min_improvement` is the minimum gain that counts (default 0), and a metric with `decisive: false` only guards against regression. For a noisy evaluation, first determine these thresholds by measuring the parent against itself.

An exit code of 0 from a check means the measurement completed, and a report with `passed: false` means a failure was measured. A crash is not a valid baseline failure. The report is `{"passed":true,"case_ids":["a","b"],"metrics":{...}}`: case IDs are unique, both versions run the same cases, and metrics are finite numbers. List every important metric that must not regress.

`evaluations/evolution-software/policy.json` is a runnable deterministic example. It verifies software behaviour only.

### Running a cycle

```bash
python tools/pipeline_evolve.py --store /data/evolution cycle \
  --driver /data/improver-driver.json --policy /data/evaluation/policy.json \
  --max-candidates 1 --timeout 600
python tools/pipeline_evolve.py --store /data/evolution evaluate \
  --candidate CANDIDATE_ID --timeout 600 --shadow
python tools/pipeline_evolve.py --store /data/evolution rollback --version PREVIOUS_VERSION
```

`cycle` freezes the policy, copies the active source, gives the driver the `pipeline-improver` role context with the pending feedback, reads the proposal and snapshots the candidate, evaluates the parent and the candidate, and promotes if `compare` accepts. `--timeout` is the wall budget for the whole cycle. By default it does not reprocess feedback that already produced a candidate; select it with `--event EVENT_ID`. After a restart, `status` shows where it stopped, and `evaluate` creates a new evaluation without overwriting results.

`pipeline_agent.py drive --evolution-driver D --evolution-policy P --evolution-timeout T` runs one bounded cycle after a task. It is refused for any budgeted task that is not `disabled`. It is a foreground controller; long-running operation must be hosted by a deployment service.

## Failure behaviour

- An out-of-scope change, a tampered policy or candidate, or a changed denominator is rejected and recorded. The active version does not change.
- A timeout or driver failure ends the cycle and keeps its logs. A timeout bounds the local process group; it does not revoke remote paid jobs.
- Running tasks are never switched to a newly promoted version. `rollback` does not change existing runs, delete outputs or reset the working tree.

## Limitations

- **One run per arm.** Promotion compares one baseline run with one candidate run, with no replication or confidence interval in the controller. Replicated evaluation must be part of the policy's checks when evidence above `software` level is claimed.
- **Scope is path-based.** The protected-path list does not stop a candidate from changing behaviour through an allowed module that a protected module imports.
- **Same-UID trust.** The driver runs as the same user as the store. Candidate code can read the files the evaluator passes to it, so hidden cases and holdout must be protected by a separate identity or sandbox.
- **Promotion is not evidence of real-task improvement.** It shows that the frozen evaluator measured an improvement at the policy's `evidence_level`, and nothing more. The underlying model weights are not trained.
