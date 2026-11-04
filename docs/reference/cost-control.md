# Whole-task cost control

Every model session that works on a task (orchestration, implementation, expert authoring, review, retries and repairs) is charged to one persistent per-task ledger. The policy is fixed when the task is prepared, so a later edit cannot reset or relax the budget, and a session whose cost is unknown stays unknown instead of being counted as zero. The ledger is a scheduler guard: it decides whether the next model call may start. It is not a billing firewall. A hard dollar cap exists only when the driver itself enforces the reservation across all descendant calls.

The rules agents follow are in [`prompts/instructions/task-cost.md`](../../prompts/instructions/task-cost.md).

## Guarantees

1. **A policy is valid only with `0 < target_usd <= limit_usd`, finite amounts and the whole-task scope.** Policies must also name `mode` and `on_excess`.
2. **The policy cannot change under a running task.** The ledger stores the policy digest, and `prepare` records it in the input lock. Changing or deleting the policy afterwards fails, and so does deleting the ledger after a session has run.
3. **Missing, invalid or partial usage never becomes zero.** A receipt counts only if it uses schema `datafarm.session-cost/1`, names the same reservation ID, sets `includes_descendants=true` and `complete=true`, and carries evidence and a `provider` or `api-equivalent` basis. Any other receipt marks the row `unknown`, and while a row is unknown `total_usd` is `null`.
4. **A reservation ID authorizes exactly one call and one settlement.** Reusing an ID fails, and so does settling a row twice. Concurrent drivers serialize on `.cost.lock`.
5. **Enforced mode caps the committed spend.** Each reservation is clipped to the remaining limit. An overrun is recorded as `overrun` and is not clipped. An unknown or overrun row blocks every later reservation until an operator reconciles it.
6. **Enforced mode needs a driver that can really cap all descendant calls.** The driver must declare `cost_control.protocol = datafarm.session-cost/1`, `includes_descendants = true` and a positive `max_session_usd`. The native Codex driver is refused.
7. **A budgeted task does not launch an unbudgeted post-run evolution driver.** This applies in advisory and enforced mode; only `disabled` allows it.
8. **A crash keeps the reservation.** On restart, a reservation from a session that never settled is settled from its receipt or marked unknown. It is never refunded by timeout.

Code: `src/datafarm/agent/cost_budget.py`, `tools/pipeline_agent.py`.

## Data model

`cost-policy.json` (copied into the task workspace at `prepare`; the default is `configs/task-cost-policy.json`):

| Field | Values |
|---|---|
| `schema` | `datafarm.task-cost-policy/2` |
| `mode` | `disabled` \| `advisory` (default) \| `enforced` |
| `on_excess` | `warn` (default) \| `pause` (advisory only: stop once known spend reaches the limit) |
| `target_usd`, `limit_usd` | finite, `0 < target <= limit`; default 30 / 50 |
| `scope` | must be `all-model-work-through-white-data` |
| `includes`, `excludes`, `currency_basis` | descriptive; the default excludes render-stage and GPU compute |

| Mode | Scheduling and evidence |
|---|---|
| `disabled` | Does not block on thresholds; saves existing cost, unknown items and usage |
| `advisory` | `warn` continues, `pause` pauses at the known limit; not a guarantee of actual total cost |
| `enforced` | Caps all child calls; unsettled, unknown or over-limit items block; an ordinary native CLI cannot be used |

`cost-ledger.json` (`datafarm.task-cost-ledger/1`) holds `policy_sha256`, `blocked` and `reservations`, a map from ID to `{status: reserved|settled|unknown|overrun, reserved_microusd, charged_microusd, actual_microusd, receipt?, usage_evidence?}`. Amounts are integer micro-USD, rounded up.

`cost-status` reports `mode`, `coverage` (`complete` or `incomplete`), `unknown_sessions`, `total_usd` (`null` while anything is unknown), `known_usd`, `committed_usd`, `remaining_usd`, `target_reached`, `warning`, `stop_required`, `blocked`, `sessions` and `history_entries`.

## How it works

1. `prepare` copies the policy, hashes it into `input.json` and starts an empty ledger.
2. Before every model session, `drive` calls `budget.status()`. It stops with `cost-budget-blocked` if `stop_required` is set, or, in enforced mode, if any row is unresolved.
3. The session reserves an amount under a fresh ID. Enforced mode clips the reservation to the remaining limit; advisory and disabled reserve 0.
4. When the session ends, `settle` applies its `cost-receipt.json`. Without a valid receipt the row becomes `unknown`.
5. External jobs that cost money outside model sessions (for example USDCraft builds) reserve and settle the same way, and record `cost_usd: null` when the cost cannot be measured.

## Entry points

```bash
python tools/pipeline_agent.py prepare --episode /data/episode.mp4 --request-file /data/request.txt \
  --target-count 5 --output /data/task --cost-policy examples/runtime/budget-disabled.json
python tools/pipeline_agent.py cost-status --work /data/task
python tools/pipeline_agent.py wait --work /data/task --timeout 1800 --interval 5
```

Example policies: `examples/runtime/budget-disabled.json`, `examples/runtime/budget-enforced.json`.

`wait` only polls `state.json`. It is not a health check of remote processes, and it does not resume tasks or submit paid calls. For `model_wait_for`, `drive` lets the model handle results once all dependencies are finished, completed, complete or succeeded, or any dependency is failed, error, cancelled or needs-input. A dependency that does not exist, is duplicated or lacks a status raises an error. This covers only calls made through `drive`, not the host's native chat.

## Enforced driver protocol

A driver declaration, which must be wired to a real metering and limiting implementation:

```json
{
  "argv": ["/opt/deployment/metered-agent"],
  "input": "messages",
  "cost_control": {
    "protocol": "datafarm.session-cost/1",
    "includes_descendants": true,
    "max_session_usd": 5
  }
}
```

`drive` passes three environment variables: `DATAFARM_SESSION_MAX_USD` (the total quota for this session and all its child calls, at most the configured value and not above what remains), `DATAFARM_COST_RESERVATION_ID` (the unique session ID) and `DATAFARM_COST_RECEIPT` (the receipt path).

Before every provider request the driver reserves the worst-case amount for the input, the maximum output, cache writes and the applicable long-context or tier rates, including retries and format fallbacks. If the quota is insufficient it must not send, and the same total must not be copied to every sub-agent. Before returning, the driver waits for all child calls to end or keeps an unknown liability; it must not return a complete receipt and leave an orphan that is billed separately.

```json
{
  "schema": "datafarm.session-cost/1",
  "reservation_id": "ID from the environment variable",
  "includes_descendants": true,
  "complete": true,
  "cost_usd": 2.31,
  "cost_basis": "api-equivalent",
  "evidence": "per-request usage persisted on the deployment side, with a price version index"
}
```

`cost_basis` may also be `provider`. Even if the session exits non-zero, the cost already incurred must be reported. A missing or invalid receipt blocks resume and retains the whole reserved amount; this does not mean the amount was paid. A reservation left by a crash also blocks the next session. A receipt above the reservation is recorded as an overrun and later calls stop.

[`examples/runtime/metered_codex_driver.py`](../../examples/runtime/metered_codex_driver.py) is an example of such a driver: it runs `codex exec` through a local metered OpenAI-compatible gateway (`src/datafarm/agent/metered_provider.py`). The model, endpoint and price table must be configured explicitly with `--pricing` or `DATAFARM_METERED_PRICING`; there is no default gateway.

## Failure behaviour

- An unknown or overrun row blocks enforced tasks until an operator reconciles it. There is no automatic refund and no command that clears unknown costs; the operator verifies the bill and the terminal states of all tasks first.
- A changed policy, a missing ledger after execution, or a reused reservation ID raises an error and never silently resets the budget.
- Advisory tasks keep running past the limit. `warning` is set, and with `on_excess: pause` `stop_required` becomes true once `known_usd >= limit_usd`.

## Limitations

- **Advisory is a report, not a cap.** The native Codex CLI reports tokens, not dollars, so sessions settle as `unknown` and `known_usd` can stay 0. An advisory `pause` only triggers on costs that were actually measured. The plain Codex CLI has no hard dollar cap.
- **The ledger is not a security boundary.** A same-UID process can edit `cost-ledger.json`. In production, keep the ledger and the metering entry point outside the author's write access.
- **Not every cost is in the ledger.** Rendering and GPU compute are excluded by the default policy, and the cost of an external chat that started the task is not captured. Neither enters `total_usd` unless the deployment records it.
