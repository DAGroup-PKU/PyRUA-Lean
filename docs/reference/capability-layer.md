# Capability layer and per-task artifacts

The release separates what is shared by every task (the capability layer K) from what an
agent generates for one request (the task artifacts A_rho). K is repository source: the role
catalog, role prompts, stage skills, runtime instructions, universal standing rules, render
templates and the harness, ledger and verifier code. A_rho = {G, T, S, C, J, R, B} (grounding
record, task contract, scene/initial-state sampler, expert controller, judge specification,
render contract, batch/dataset configurations) is written by agents into the task workspace.
This document covers the code that keeps the two apart: the role catalog and context compiler
that load K explicitly, the instruction-layer audit, the standing-rules lint, and the source
pin that fixes the version of K a run uses.

Related: [capability evolution](capability-evolution.md) (how K changes and is activated), [agent runtime](agent-runtime.md) (who calls the compiler), [cost control](cost-control.md).

## Guarantees

1. **A role's context is built only from the files the catalog names for it.** The compiler
   loads `common` documents, `prompts/roles/<role>.md`, the role's `documents` and
   `.agents/skills/<skill>/SKILL.md` for each listed skill, in that order with duplicates removed.
   It never scans `runs/` or task workspaces; `task_products_auto_loaded` is always `false`.
2. **A missing instruction file is an error, not a silent omission.** Every path is resolved
   with `strict=True`.
3. **No instruction file may resolve outside the repository root** (for example through a
   symlink).
4. **Every loaded file and the final message list are hashed.** The receipt lists
   `{path, sha256}` per file and `messages_sha256` over the canonical JSON of the messages.
5. **Only universal standing rules enter a context, without their provenance fields.** Rules
   whose `scope` is not `universal` (for example `task:<family>`) are excluded; `waivers`,
   `known_violations`, `source` and `globs` are stripped.
6. **A context is never written over an existing one.** A non-empty destination is refused.
7. **A prepared task is bound to a content-addressed snapshot of the source.** `prepare` writes
   `capability.json` naming a capability version whose `source` is the SHA-256 of a file manifest.
8. **A changed snapshot or version record is detected before use.** Resolving a pin re-hashes
   every file of the snapshot and the version record.
9. **Snapshots contain no symlinks and no runtime/credential directories.** Symlinked files or
   directories are refused; `EXCLUDED` names (`.git`, `.datafarm`, `runs`, `secrets`, `keys`, `.env`,
   `data`, `external`, ...) and `*.pyc`/`*.log` are skipped.
10. **Activating a new version does not move existing pins.** A run keeps the version written
    in its `capability.json`; activation and rollback only change `active.json`.
11. **Pinned task commands execute the pinned code.** `context`, `drive`, `verify`,
    `cost-status`, `wait` and `supervise` with `--work` re-execute
    `<pinned>/tools/pipeline_agent.py` when the pin differs from the invoked checkout.
12. **New user requirements reach a pinned run; learned lessons stay pinned.** The experience
    block is selected per session with the pinned version: all matching unsuperseded
    `user-requirement` events, plus at most 8 lessons of that version.
13. **The standing-rules lint fails on any unlisted violation.** Known violations (with owner
    and reason) are reported but pass; stale known entries are reported.
14. **The instruction audit fails when a universal instruction carries machine-local authority.**

Code: `src/datafarm/agent/context.py`, `src/datafarm/evolution/integration.py`, `src/datafarm/evolution/versions.py`, `src/datafarm/evolution/events.py`, `tools/pipeline_agent.py`, `tools/lint_standing_rules.py`, `tools/audit_instruction_layers.py`.

## Data model

| File | Schema / required fields |
|---|---|
| `configs/agent-instructions.json` | `schema: datafarm.agent-instructions/1`; `common` (list of repo paths); `roles.<name>.skills`, `roles.<name>.documents`; `task_products` (informational string, never read by code) |
| `<out>/messages.json` | `[{role: system, content}, {role: user, content}]` |
| `<out>/context.md` | the same messages as text |
| `<out>/receipt.json` | `schema: datafarm.agent-context/1`, `role`, `files[{path, sha256}]` (last entry is `configs/standing-rules.json` with a `selection` note), `messages_sha256`, `task_products_auto_loaded: false`, `experience` |
| `configs/standing-rules.json` | `schema: datafarm.standing-rules/1`; per rule `id` (unique), `statement`, `globs` (all required), optional `scope`, `forbidden[{regex, why}]`, `numeric_min[{key, min, why}]`, `required_json[{key, allowed}]`, `waivers[{path}]`, `known_violations[{path, ...}]`, `review_audit` |
| `<work>/capability.json` | `schema: datafarm.run-capability/1`, `store`, `version`, `scope` (includes `run_id`), `managed`, `experience` |
| store `versions/<id>` | `schema: datafarm.capability-version/1`, `source`, `parent`, `lessons`, `evaluation`, `id` = digest of the rest |
| store `sources/<digest>/` + record | the copied files; record `{id, files: {path: {sha256, executable}}}`, `id` = digest of `files` |

Current catalog: 5 `common` documents and 7 roles (`orchestrator`, `task-designer`,
`task-expert-author`, `task-verifier`, `render-reviewer`, `verifier-auditor`,
`pipeline-improver`). Callers: `tools/pipeline_agent.py` (orchestrator, task-verifier),
`src/datafarm/expert/round.py` (task-expert-author), `src/datafarm/evolution/controller.py`
(pipeline-improver). `task-designer`, `render-reviewer` and `verifier-auditor` are loaded only through `context --role <role>`.

Inventory layers (path-rule classification in `audit_instruction_layers.py::inventory`):
`universal-instruction`, `universal-runtime-schema`,
`deployment-or-device-profile`, `runtime-implementation`, `task-product-or-upstream-reference`,
`task-product-or-tool-schema`, `task-product-template-or-example`, `test-evidence`,
`implementation-reference`.

## How it works

Context compilation (`build_context`):
1. Read the catalog; reject an unknown role.
2. Build the path list (common, role prompt, documents, skills); de-duplicate preserving order.
3. For each path: resolve strictly, refuse escapes, hash the bytes, append a
   `--- PROJECT INSTRUCTION: <path> ---` section.
4. Append the universal standing rules (filtered fields) and hash the rules file.
5. Prepend a fixed preamble (repository-owned instructions, role limits duties, task inputs are
   evidence, repository path).
6. User message: the caller's text (for task sessions, `pipeline_agent.py::user_context`: request,
   input manifest, current state, cost status) plus the scoped experience block.
7. Hash the messages; `write_context` writes `messages.json`, `context.md`, `receipt.json`.

Source pinning:
1. `prepare` calls `bind(work, ROOT, store_path=--evolution-store)`.
2. Without `--evolution-store`, the store is `<checkout>/.datafarm/evolution`; the invoked checkout
   is snapshotted and a version is created (not activated). With it, the store's active
   version is used (or the checkout becomes the initial active version).
3. `snapshot` lists files (`git ls-files --cached --others --exclude-standard` in a Git checkout,
   a filtered walk otherwise), copies them to a temporary directory, re-hashes, and renames it to
   `sources/<manifest digest>`; an existing digest is reused.
4. Later commands with `--work` call `source_for`, which verifies the snapshot and re-execs the
   pinned `tools/pipeline_agent.py`; inside the snapshot `ROOT` equals the pin, so contexts and
   the native driver come from the pinned files.

Lint: for each rule, glob the files (rules file itself excluded), apply `forbidden` regexes
(case-insensitive), `numeric_min` (JSON values or `key: number` text), `required_json`; tag each
finding `waived`, `known` or `new`; exit 1 if any `new`.

Audit: build every role's context to get the loaded set, list repository files, classify each
`.md/.txt/.json/.yaml/.yml/.py` file by path rules, record Python assignments whose name matches
`prompt|system_message`, and flag universal-instruction files that contain a site-specific filesystem
prefix, an `ssh -o` command or a host alias.

## Prompt assembly

The project controls message bodies and the assembly process. It cannot treat a host's hidden system prompt as a repository asset.

- Each policy has one source. `configs/standing-rules.json` registers rule IDs, `prompts/instructions/` explains the boundaries, render templates implement text requirements, a task overlay only adds specifics, skills carry procedures, and verifier specs implement testable judges. A change updates each consumer and passes the lint.
- `prompts/render/` templates use `${UPPER_SNAKE_CASE}` slots. A missing or unknown variable, a leftover slot or a conflicting overlay rule blocks submission. The substituter only assembles text; it does not run a shell or eval. Assembly order: common, one explicit mode template, the task overlay, then frozen event and continuity content.
- The first, continue and final messages of an expert round live in `prompts/sessions/` and are not interchangeable. Role bodies are in `prompts/roles/` and are injected by the launcher. A task adapter declares a specific author backend and model.
- `pipeline_agent.py context` and `drive` and the expert-harness round load roles explicitly. Prompts inside other tools are not injected as general instructions.

## Entry points

```bash
python tools/pipeline_agent.py context --role orchestrator --output /data/ctx          # bare role context
python tools/pipeline_agent.py context --work /data/task --role task-verifier --output /data/ctx2
python tools/lint_standing_rules.py --json
python tools/audit_instruction_layers.py
```

## Configuration

| Item | Default | Source |
|---|---|---|
| `context --role` | `orchestrator` | `tools/pipeline_agent.py::main` |
| `prepare --evolution-store` | unset: private store `<checkout>/.datafarm/evolution`, `managed: false` | `integration.py::bind` |
| `prepare --scope` | JSON selectors merged with an automatic `run_id` | `integration.py::bind` |
| lessons per context | 8 (`limit`) | `events.py::select` |
| `lint_standing_rules.py --root`, `--rules`, `--json` | repo root, `<root>/configs/standing-rules.json`, text | `lint_standing_rules.py::main` |

## Failure behaviour

- Unknown role → `ValueError`; missing file → `FileNotFoundError`; escaping path or non-empty
  destination → `ValueError`. `pipeline_agent.py` prints the error and exits 2.
- A changed pinned snapshot or version → `ValueError('pinned source changed' / 'capability
  version changed')` on every command with `--work`; nothing is re-pinned automatically. An
  operator must restore the snapshot or start a new workspace.
- Changed input files → `context`/`drive` fail in `task/contract.py::verify_input` before any model call.
- A snapshot that changes while being copied is discarded (`source changed while creating snapshot`).
- Lint: exit 1 on new violations; the rules file is validated first (`ValueError` on duplicate
  ids, missing fields, bad regex).
- Audit: exit 1 when `errors` is non-empty.

## Limitations

- **Pin scope.** The pin covers repository files only, not the Python interpreter, installed
  packages, external CLIs (for example the Codex executable), provider models or GPU images.
- **Commands not re-executed.** `prepare`, `stop`, `runtime-status`, `feedback` and `context`
  without `--work` run the invoked checkout. Workspaces without `capability.json` fall
  back to the invoked checkout (`integration.py::source_for`).
- **Other entry points are unpinned.** The run ledger, the batch supervisor
  and the expert harness do not bind a capability version; the ledger records a Git fingerprint
  only (`ledger.py::git_fingerprint`).
- **Changing K under tests is a process rule.** Code protects pinned runs from later edits and the evolution
  controller evaluates candidates, but direct source edits are not gated by tests.
- **Reproducibility of messages.** `messages_sha256` covers the repository path, workspace path,
  current `state.json` and cost status, so it changes per workspace and per session. Each
  session's `messages.json` is saved; regenerating identical bytes later is not guaranteed.
- **Runtime reviews use one role.** The reviewer dispatched by `drive` always uses `task-verifier`, including for rendered-episode
  reviews. `render-reviewer` and `verifier-auditor` load only through `context --role`.
- **Audit is path-based.** Layers come from path prefixes, not from content; the machine-local check is a fixed regex applied only to the
  `universal-instruction` layer.
- **Lint is lexical.** Regex and numeric-key checks over the configured globs; it cannot detect
  a paraphrased contradiction or a rule enforced only at runtime. Only rule SR-005 (literal
  credentials) covers `docs/**/*.md`.
- **Render templates** are protected by `src/datafarm/render/policy.py::load_contract`, which recompiles the contract from the current
  templates; see [render contract](render-contract.md).
