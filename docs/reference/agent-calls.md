# Agent calls: call ledger and reviewer calibration

Pipeline code runs coding agents and vision-language model (VLM) reviewers in a few places. None of these
models is an authority: a model may only *propose* or *screen*, and simulator physics and local verifiers decide
what is admitted. That rule is only meaningful if every model call can be counted and checked.

How system and role prompts are assembled is described in [capability-layer](capability-layer.md). A new `prompts/roles` file must be loaded explicitly; it is not in effect merely because the file exists.

| Piece | Code | What it does |
|---|---|---|
| Call ledger | `src/datafarm/agent/call_log.py` | One JSON line per model call attempt (schema `datafarm.agent-call/1`) |
| Ledger report | `src/datafarm/agent/call_report.py`, `tools/agent_calls_report.py` | Calls, tokens, known and unknown cost, latency per component, backend, model and status |
| Reviewer calibration | `src/datafarm/verify/reviewer_calibration.py`, `tools/calibrate_vlm_reviewer.py`, `configs/standing-rules.json` | VLM vs human agreement; audit against standing rules |

## 1. Call ledger (`datafarm.agent-call/1`)

The controller that issues a call writes the record, never the agent. A record is
appended (under `flock`, fsynced) after each attempt and *before* the controller acts on
it (retry, fall back, return), so failed, retried and abandoned attempts stay in the
denominator. Prompts, responses and credentials are never stored, only SHA-256 digests.
The default file is `<run dir>/agent-calls.jsonl`.

| Field | Meaning |
|---|---|
| `schema` | `datafarm.agent-call/1` |
| `call_id` | unique per attempt (`call-<hex>`) |
| `component` | controller that issued the call, e.g. `expert-session` |
| `backend` | the caller's backend, for example `claude-cli` or `codex-exec` |
| `model` | requested model |
| `started_at`, `ended_at` | UTC timestamps |
| `status` | see below |
| `request_sha256` | canonical JSON of the request body (no headers) |
| `prompt_sha256` | canonical JSON of the `messages` list |
| `response_sha256` | canonical JSON of the parsed response body (raw bytes if not JSON) |
| `response_id` | provider id, e.g. `chatcmpl-…` |
| `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens` | provider usage; for OpenAI-style providers `input_tokens` includes cached tokens |
| `cost_usd`, `cost_source` | `price_table` when a caller-supplied table priced the call, otherwise `null` = unknown |
| `turns` | 1 for a chat completion that returned a 2xx body |
| `latency_s` | wall time of the attempt |
| `error_type` | e.g. `ReadTimeout`, `http_429`, `schema_violation` |
| `run_id`, `attempt` | pipeline run id; 1-based live attempt number within the logical call |
| `extra` | caller-specific details |

Statuses: `ok` (answer parsed and schema-valid), `invalid_response` (an answer arrived
but was not usable), `error` (transport or provider error), `timeout`, `aborted` and `budget_exhausted`
(controller stops; used by the harness).

### Ledger report

```bash
tools/agent_calls_report.py runs/                      # every agent-calls.jsonl below runs/
tools/agent_calls_report.py a.jsonl b.jsonl --json report.json
tools/agent_calls_report.py runs/ --group-by model,status --json -
```

Rows are grouped by `component, backend, model, status` (or `--group-by` any of
`component, backend, model, status, run_id, error_type, cost_source`). Each row and the
total report `calls` (every record), `logical_calls`, tokens, `cost_usd_known`,
`calls_with_unknown_cost` and latency sum/mean/p50/p95/max. A malformed line
fails the report unless `--skip-invalid` (then it is listed under `problems`); a
`call_id` seen twice (the same ledger passed twice) is counted once and listed.

## 2. Calibrating a VLM reviewer against human labels

A VLM screening verdict is only usable once its agreement with humans is known, per
criterion. `tools/calibrate_vlm_reviewer.py` pairs per-sample, per-criterion verdicts with
human labels.

**VLM inputs** (`--vlm`, any mix): run directories (every
`episodes/<id>/visual-review/assessment.json`), a summary file, or
generic `{"id", "criteria": {name: {"status", "evidence"}}}` JSON. A
focused recheck assessment overrides the criteria it names, but only when its `original_assessment_sha256` equals the hash of
the first-pass file; otherwise it is ignored and reported. `--no-rechecks` evaluates the
first pass alone. The overall outcome is re-derived: any `fail` gives `flagged`, else any
`uncertain` gives `uncertain`, else `sampled_pass`.

**Human labels** (`--human`): JSON `{"labels": {sample: {criterion: label}}}`, a list of
`{sample_id, criterion, label}` rows, or CSV in long (`sample_id,criterion,label`) or wide
(`sample_id,<criterion>,…`) form. `--id-map` renames human ids, either from an object
`{human_id: review_id}` or from a blind key such as
`<run>/blind-key.json` with `--id-template` (for example
`sample-{sample}`).

**Mapping** (`--mapping`, optional; default: each human criterion maps to the VLM
criterion of the same name):

```json
{"schema": "datafarm.vlm-calibration-mapping/1",
 "criteria": {"geometry": {"vlm": ["object_geometry", "camera_layout"]},
              "overall": {"vlm": ["@outcome"]}},
 "human_vocabulary": {"pass": ["acceptable", "none"], "fail": ["unacceptable", "obvious"],
                      "abstain": ["unsure"]}}
```

Several VLM criteria combine conservatively: any `fail` gives `fail`; otherwise any
`uncertain` or missing criterion gives `abstain`. `@outcome` is the overall verdict. An
unknown label is an error, not a guess.

**Metrics** (per human criterion and pooled). `fail` means "defect present". Human
`abstain` labels are excluded and counted.

- confusion matrix: human {pass, fail} × VLM {pass, fail, abstain};
- accuracy and Cohen's kappa over *decided* pairs (VLM pass/fail); kappa is `null` when
  undefined (both raters used one identical category);
- false-accept rate = VLM pass on human fail ÷ all human fails; false-reject rate = VLM
  fail on human pass ÷ all human passes; both also over decided pairs only;
- abstention rate; the "abstain as reject" variant (the operational gate, where
  `uncertain` blocks admission): accuracy, kappa and false-reject rate;
- Wilson 95% intervals for accuracy and both error rates.

**Standing rules** (`--standing-rules`, optional; `--prompt-file` adds prompt sources).
Each rule has an id, statement, source, `forbidden_criteria` (VLM criteria that must not
count as defects), `forbidden_phrases`/`forbidden_patterns`, `scopes` (`criterion`,
`evidence`, `prompt`) and `exempt_criteria`. A `fail` verdict on a forbidden criterion
contradicts the rule by construction; a phrase found in fail evidence or in a reviewer
prompt is a *candidate* for a human to confirm. The report counts flagged verdicts and
prompts per rule and shows how outcome counts would change if the flagged verdicts were
ignored. `configs/standing-rules.json` encodes the project's
rules: soft light is not required and hard shadows are not a defect; exposure is only a gross
guard. Lighting that changes over time within a
video remains a defect, so `temporal_appearance` is exempt.

**Procedure.** Export the labels, map ids, run, and read the per-criterion rows before
the pooled one:

```bash
tools/calibrate_vlm_reviewer.py --vlm runs/<run> --human labels.json \
  --id-map runs/<run>/blind-key.json --id-template 'sample-{sample}' \
  --mapping mapping.json --standing-rules configs/standing-rules.json \
  --out calibration.json
```

With `--human` but zero pairs the tool exits 3 (unless `--allow-empty`): nothing was
calibrated. Without `--human` it reports only the verdict distribution and the rule
audit, and says that agreement was not measured.
