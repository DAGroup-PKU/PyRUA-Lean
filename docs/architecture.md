# Architecture

DataFarm turns one real calibration episode and a data request into action-labelled, photographic
robot data. A coding agent does the task-specific work; this repository supplies the shared
instructions, the checks the agent cannot sign for itself, and the records that keep every attempt.
Everything written for a task (scene, expert, judge, prompts, configs) goes into the task's workspace,
not into this repository.

## The flow

```text
episode + request
  │  tools/pipeline_agent.py prepare     pin source, input and cost policy (datafarm.agent)
  │  tools/pipeline_agent.py supervise   run agent sessions until delivery or a stop (datafarm.agent)
  ▼
grounding              scene, cameras, references, labelled physical quantities (datafarm.core.grounding)
  ▼
simulator qualification  a variant certified for the task's skills (datafarm.simulation)
  ▼
task contract, expert  frozen judge, sandboxed author, held-out seeds (datafarm.task.contract, datafarm.expert)
  ▼
rendering              render contract and provider payload checks (datafarm.render.policy),
  │                    H3 control hybrid or Seedance (datafarm.render)
  ▼
verification, review   render verifier and independent review (datafarm.verify, datafarm.task.contract)
  │                    bounded per-item batches around all of the above (datafarm.batch)
  ▼
delivery, admission    tools/pipeline_agent.py verify; admission certificate (datafarm.core.admission)
  ▼
training (on request)  certificate-gated pi0.5 launch (datafarm.training)

the run ledger (datafarm.ledger) records requests, items, workers and verdicts throughout
```

A rejection returns to the stage that caused it. New attempts create new records, and earlier attempts and
failures are kept.

## Who may sign what

The producer of a stage never certifies it.

| Stage | Producer | Checked by |
|---|---|---|
| Grounding | agent | independent review; `validate_quantities` at delivery |
| Task contract | agent | independent task verifier |
| Episode | expert and simulator | frozen judge run by the expert harness, on held-out seeds |
| Render | video model | render payload checks, render verifier, independent reviewer |
| Delivery | agent | `verify_delivery` and `verify_white_delivery`, which require content-bound reviews |
| Admission | certificate compiler | admission root, re-derived from stored evidence bytes |

Receipts are records, not cryptographic signatures: anyone who can write the store can write one.
Independence comes from separate executions, content binding, and re-running verification where the
evidence is available. Process boundaries are not a security boundary between agents running as the
same user.

## Artifacts and admission

`datafarm.core` defines typed documents (`ArtifactDocument`, `GateReceipt`) stored by content address
in an `ArtifactStore`. Documents cannot contain absolute host paths, credential fields or mutable signed
URLs. `datafarm validate <file>` checks one document or receipt.

A training admission certificate binds the admitted member ids and per-member media digests, the
allowlist digest, the audit reports, the dataset provenance and a digest manifest of every dataset file,
and an explicit waiver policy. An independent admission root re-derives the claim from the stored bytes
and signs a gate receipt. `tools/launch_pi05_training.py train-command` verifies the certificate before a launch
manifest exists, and a launch without a verified certificate is refused. Operator steps are in
[delivery and admission](reference/delivery-and-admission.md).

## Packages in `src/datafarm`

| Package | Job |
|---|---|
| `core` | Artifact documents, receipts, schema checks, content-addressed store, file binding, source locks, grounding ledger, admission certificates |
| `agent` | Compile role context, run and resume agent sessions, meter usage, keep the cost ledger and call log |
| `task` | Task contract and delivery check, white delivery |
| `expert` | Expert harness: sandboxed author sessions, rollout broker, frozen judge, held-out evaluation, expert library |
| `simulation` | Simulator-qualification decision logic, joint-margin audit, camera and grasp randomization plans, USDCraft adapter |
| `render` | Render contract (`render/policy.py`) and renderers: the H3 control hybrid (`render/h3`) and the Seedance client (`render/seedance`) |
| `verify` | Spec-driven render verifier, counterfactual calibration, reviewer calibration |
| `batch` | Batch supervisor: bounded per-item launch, poll, verify and retry |
| `ledger` | Run ledger: fixed item denominator, derived run status, launch sentinel |
| `training` | Certificate-gated pi0.5 launch, dataset and action-mode contracts, cohort auditors |
| `evolution` | Feedback store and controller that evaluates and activates capability-layer changes |
| `runtime.py`, `doctor.py`, `cli.py`, `common.py` | Runtime bindings, prerequisite checks, the `datafarm` command, shared helpers |

## Top-level directories

| Directory | Job |
|---|---|
| `src/` | The `datafarm` package |
| `tools/` | Command-line entry points: `pipeline_agent.py`, `pipeline_phases.py`, `expert_harness.py`, `supervise_batch.py`, `verify_render.py`, `launch_pi05_training.py`, ... |
| `.agents/skills/` | Stage skills the agents load |
| `prompts/` | Role instructions, runtime instructions, expert session messages, render prompt templates |
| `configs/` | General policy files, render recipes, one template per task-artifact kind |
| `evaluations/` | Frozen evaluators: render EVAL-v1 and the software check for capability evolution |
| `examples/` | Runtime and budget examples, USDCraft example |
| `vendor/` | openpi trainer (Apache-2.0) and the VideoX-Fun patch |
| `assets/` | Figures for the README |
| `tests/` | CPU test suite |
| `docs/` | This documentation; [index](README.md) |

External components (simulators, video models, datasets, credentials, GPU scheduler) are runtime bindings;
see [deployment](deployment.md). Heavy runtime imports stay out of the core packages.

## Further reading

- [`AGENTS.md`](../AGENTS.md), [`prompts/instructions/`](../prompts/instructions), [`configs/agent-instructions.json`](../configs/agent-instructions.json): what the agent reads and produces.
- [Reference notes](README.md#reference): one note per component, with guarantees, records and limits.
