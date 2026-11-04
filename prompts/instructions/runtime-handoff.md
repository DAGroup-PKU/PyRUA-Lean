# Runtime handoff: reviews, recovery and external jobs

Entry point: `pipeline_agent.py supervise --work TASK --config CONFIG`; a bounded run uses `drive --driver DRIVER`. Configuration fields are in
`docs/reference/agent-runtime.md`. Sandbox permissions are decided by the deployment; if the working directory changes, the tools must still be able to
write to the task directory.

## Independent review handoff

The author atomically updates the queue in `state.json`. A status string cannot self-sign a review result.

```json
{"status":"working","review_requests":[{
  "id":"contract-001",
  "instructions":"Independently check the contract and the negative controls; keep rejection evidence, do not change the judge.",
  "subjects":[{"path":"task/contract.json","sha256":"actual file SHA256"}]
}]}
```

The program freezes the request and subject hashes and schedules a task-verifier context and a separate session. The reviewer writes
`.runtime/reviews/contract-001/report.json`: `verdict` is pass/reject/unverified, `subjects` exactly match the request, `evidence` is a list of real
file path/sha256 entries, and `findings` are concrete issues. The program verifies the report and evidence and binds the execution record, context hash and
session to `completion.json`.

- A review that does not pass returns to the author for repair. A new revision uses a new request ID, and the earlier rejection is kept.
- A review never resumes the author session.
- When reviewing a delivered episode, the reviewer also writes an episode review file containing `runtime_review: <request ID>` and lists it in the report
  evidence. Delivery validation uses this to confirm the review came from an independent execution.
- After a reviewed output is modified, remove the superseded request from the active queue and add a new ID. All history stays in `.runtime/reviews`. Versioned
  outputs avoid overwriting review provenance.
- Each time a completed review is used, its verdict, subjects, evidence, context, execution, identity and log hashes are rechecked. A stale review is rejected.
- This queue records the handoff. It does not replace the semantic, physical, image and provenance verification required by delivery.

A same-UID file is not isolation from a malicious author. The deployment's permission boundaries protect hidden judge/holdout data and production ledgers.

A general argv driver must declare `session_identity_protocol: "datafarm.agent-identity/1"` before entering the review queue. A trusted driver obtains identity and
evidence from the actual provider execution and must not copy environment variables to fake isolation. The host rejects reuse of the author thread.

## Recovery

A restart first checks the recorded process identity and waits if it is alive. If it is dead and has a native thread ID, that ID is resumed. An unclear submission
without an ID stops as `recovery-required`: check the original logs and external IDs, and never resubmit automatically.

If `state.json` written by the model is invalid, the runtime stops and saves the error; exit 0 is not success. Save the invalid file and a repair record, fix only the
control metadata, revalidate the JSON, then resume the original thread. The original actions, images and metrics must not change.

Do not delete the ledger, control file or lock to reset counts or bypass a check. `stop` ends only the local model process group; check remote jobs under their
original IDs before resuming, and do not resubmit.

## External jobs

`external_jobs` and `model_wait_for` follow `prompts/instructions/task-cost.md`. The deployed job supervisor monitors existing remote IDs and atomically updates state. The
runtime never treats an unknown remote status as completed, and never cancels or resubmits remote tasks.

An `external_jobs` entry may supply `status_file` (absolute or task-relative). The record must carry the same `id` and an explicit status; unreadable or mismatched records stay
unknown. A nonzero exit cannot become success, and `exported-unqualified` wakes the model only for inspection. No job is submitted or inferred complete from a heartbeat or
modification time. Workers write these records atomically and preserve actual scheduler IDs.

`status_output` publishes this supervisor's state into a shared monitor file. It is a status publisher, not a notifier or another controller. `completed_episodes` counts only
host-verified delivery; producer-reported progress is labelled separately.
