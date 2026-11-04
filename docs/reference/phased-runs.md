# Phased runs

`tools/pipeline_phases.py` (`src/datafarm/agent/phases.py`) creates a task one phase at a time. Each phase is one fresh agent session on a brief from [`prompts/phases/`](../../prompts/phases/), with a USD cap that the host enforces. Producers and independent reviewers never share a session, and capture attempts run with no model. A short handoff file carries the work from phase to phase, so no session has to hold the whole task.

Unlike [`supervise`](agent-runtime.md), this flow does not decide what runs next. The host chooses each phase from the previous result. The host can be you or a coding agent following this note. The flow ends at the white delivery; rendering and training are separate stages.

## Phases

| Phase | Role | What it does | Example cap |
|---|---|---|---|
| `P1` | Producer | Observes the episode; proposes the contract; grounds the scene; builds a headless runner | $25 |
| `R1` | Reviewer | Freezes the contract, judge and hidden admission seeds | $10 |
| `P2` | Producer | Writes the expert on dev seeds; runs a pre-registered held-out list once | $35 |
| `R2` | Reviewer | Tests the expert independently; denies with at most 5 blocking items, or freezes the capture batch | $15 |
| `P2r` | Producer | Repairs exactly the blocking items after a denial; then `R2` runs again | $30 |
| capture | No model | Runs frozen attempts in order until the deterministic gate stops | none |
| `RI` | Reviewer | Only after `halt_infra_investigate`: decides whether capture may resume | $8 |
| `R3` | Reviewer | Admits or rejects each candidate episode | $10 |
| `P4` | Producer | Builds the white delivery over the admitted files | $8 |
| `R3b` | Reviewer | Binds the review files to the delivery manifests; runs the verifier | $6 |

The general rules every phase follows are in [`protocol.md`](../../prompts/phases/protocol.md).

## Run a task

```bash
python tools/pipeline_agent.py prepare --episode /data/episode.mp4 --request-file /data/request.txt \
  --target-count 5 --stop-after white --output /data/task/work
RUN="python tools/pipeline_phases.py run --task-dir /data/task --config examples/runtime/phases-solpi.json"

$RUN --phase P1
$RUN --phase R1 --quote-final P1
$RUN --phase P2 --quote-final R1
$RUN --phase R2 --quote-final P2
python tools/pipeline_phases.py capture --task-dir /data/task --review R2
$RUN --phase R3 --summary-file /data/capture-notes.md
$RUN --phase P4
$RUN --phase R3b
python tools/pipeline_agent.py verify --work /data/task/work
```

Before the first run, copy a config from [`examples/runtime/`](../../examples/runtime/README.md) and fill in `environment`: the simulator python, how to reach a GPU, the path of the robot assets (this repository does not include them), and how many seeds may run at once. Install the harness first; the [README](../../README.md#quickstart) has the SoL-Pi steps.

Read `phases/<label>/final.md` and the review's `report.json` before starting the next phase. A later phase reads its input from files, and `--quote-final` adds the earlier final message as unverified claims. After a denial, use a new label for each session:

```bash
$RUN --phase P2b --template P2r --quote-final R2 --set DENIAL=R2 --set PREVIOUS_HANDOFF=P2-handoff.md
$RUN --phase R2b --template R2 --quote-final P2b --set REQUEST=P2b-source-request.json --set HANDOFF=P2b-handoff.md
```

If `R2b` froze the batch, pass `--review R2b` to `capture` and `--set BATCH_REVIEW=R2b` to `R3` and `P4`. `--print-brief` shows a brief without running it, and `--brief FILE` runs your own, which is how you give a producer extra instructions. Record such help, because it is part of the cost and the result.

After an infrastructure failure, continue the session under a new label with `--resume-from`:

```bash
$RUN --phase P2-retry --template P2 --resume-from P2 --brief /data/continue.md
```

`pipeline_phases.py cost --task-dir /data/task` prints the spend per phase, including the running phase.

## Guarantees

1. **A label runs once.** A second run of a label is refused, so a phase cannot be rerun over its own record.
2. **Caps stop the session.** The host polls usage every `poll_seconds` and stops the whole process group at the phase cap, the task cap or `max_phase_seconds`. A phase does not start once the task cap is reached.
3. **Usage without a price stops the phase.** A cap cannot be enforced on usage it cannot price.
4. **Provider errors count as failures.** Pi exits 0 after a provider error, so the host reads the outcome from the last assistant message. Claude Code and Codex are read from their result events.
5. **Sessions stay separate.** The ledger records each session ID. A session reused by another phase is marked `session_reused`, unless `--resume-from` asked for it.
6. **A finished review gets a host-written execution record.** `work/reviews/executions/<label>/` holds a copy of the session's transcript and an `execution.json` that binds it. `verify` checks this record, and reports it as `declared-execution-record`.
7. **The gate decides capture and only the reviewer admits.** `capture` runs the reviewer-frozen `capture_attempt.sh` one attempt at a time and stops at any `decision` other than `continue`, including a missing one.

## Records

| Path | Content |
|---|---|
| `phase-usage.jsonl` | One row per phase: `phase`, `template`, `role`, `harness`, `model`, `exit`, `status` (`done`, `failed`, `stopped`, `session_reused`), `stop_reason`, `over_cap`, `session_id`, `resumed_from`, `unpriced_records`, `phase_usd`, `task_usd_before`, `task_usd_after`, `wall_s` |
| `phases/<label>/` | `brief.md`, `prompt.md` (`AGENTS.md` and the brief), `command.json`, `stream.jsonl`, `stderr.log`, `final.md`; for Pi also `pi-sessions/` |
| `phases/current-cost.json` | Spend of the running phase; removed when it ends |
| `work/reviews/<label>/` | Review outputs. Keep them out of `work/.runtime/`: that folder marks a task as driven by `supervise`, and `verify` would then require runtime reviews |
| `work/capture/` | `driver.log`, and the gate's ledger, attempts and admissions |

## Configuration

Configs use schema `datafarm.phase-run/1`. Examples: [`phases-solpi.json`](../../examples/runtime/phases-solpi.json), [`phases-claude-code.json`](../../examples/runtime/phases-claude-code.json), [`phases-codex.json`](../../examples/runtime/phases-codex.json).

| Key | Meaning |
|---|---|
| `harness` | `pi`, `claude` or `codex` |
| `model` | Passed to the CLI; required for `codex`, whose stream does not name it |
| `extra_args`, `env` | Added to the CLI's arguments and environment. Do not put credentials here |
| `prices_usd_per_mtok` | Per model: `input`, `cache_read`, `cache_write_5m`, `output`, optional `long_prompt` (`above_tokens`, `input_multiplier`, `output_multiplier`). A key matches any model ID that contains it. Check these against your provider |
| `task_cap_usd`, `phase_caps_usd` | Required. A cap is looked up by label, then by template |
| `poll_seconds`, `max_phase_seconds` | Usage polling, 30 s by default, and an optional time limit |
| `environment` | Deployment notes shown to every phase: simulator python, GPU access, robot assets, parallel seed limit |
| `freshness_forbidden` | Further paths agents must not open, such as earlier task outputs |

## Limitations

- **Little real use so far.** One complete task, stack cups on SoL-Pi, has run through this tool. Codex has run it with short prompts only, and Claude Code only as a stub. One run does not show that another task will succeed; that takes its own run and independent review.
- **A cap can overrun by one model response.** Pi and Codex record usage when a response ends, and Claude Code's stream does the same per message, so the overrun is visible as `over_cap` but not prevented.
- **Costs are API-equivalent estimates from your price table.** Claude Code's reported total replaces the estimate when its run ends.
- **These costs are outside the task's cost ledger.** `pipeline_agent.py cost-status` does not include them.
- **Review independence is declared, not proven.** Agents run as the same user and only instructions and file permissions hide the holdout seeds. For a stronger check, use `supervise`, whose reviews `verify` reports as `runtime-verified`.
- **Hosts decide.** Nothing here chooses the next phase or judges a review.
