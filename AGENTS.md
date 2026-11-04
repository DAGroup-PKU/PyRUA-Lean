# DataFarm: instructions for any coding agent

Your job is to adapt a NEW real robot task/scene episode into independently verified,
action-labelled synthetic data. You are the general task compiler, not a selector of
prebuilt task adapters. Read prompts/instructions/agent-start.md and follow .agents/skills/pipeline-orchestrate/SKILL.md.
The absence of a matching task name, adapter, task brief or video prompt is expected:
observe, derive, implement, test and generate them. Never ask the user to write these for you.

## Authority and instruction loading

1. Read prompts/instructions/instruction-layers.md, authority-and-scope.md and mode-and-style-contract.md.
2. Read prompts/roles/orchestrator.md and the stage skills listed in configs/agent-instructions.json.
3. Use `python tools/pipeline_agent.py context --role orchestrator --output /tmp/datafarm-context`
   for a fully expanded, hash-recorded context. This explicitly loads instructions even when your
   host does not discover .agents/skills. Follow its file routing.
4. The user's current instructions take priority. Preserve valid in-scope authorization. Ask only for
   genuinely missing intent, inaccessible inputs/resources or permissions, after inspecting what
   is available. Infer and validate technical choices yourself. Unknown OOD versus sim2real intent
   must be resolved before rendering; it does not block grounding or local implementation.

## Invariants

- General rules/skills/roles are first-layer source. A task's adapter, brief, judge spec, scene,
  camera fit, prompts, run configs and reports are second-layer OUTPUTS. Outputs of other tasks are
  not prerequisites, global rules or authorization for a new task.
- Use full episodes as the unit; splitting for video transport does not change action semantics.
- Both render modes require real-world image conditioning and photographic output. Extract real
  references from the input episode yourself. Source frames/continuity pictures do not replace them.
- The producer must not self-certify success. Separate task-contract review, expert development,
  simulator/judge evaluation, and video review. Independent means a separate review execution with
  evidence; a different string in a JSON field is not proof of isolation.
- Keep all attempts, failures, provider IDs, seeds, budgets, code/prompt/input hashes and replacements.
  Resume existing IDs before submitting again. Confirm background persistence twice with progress.
- Do not change raw actions/images/metrics to pass a gate. Replanning requires matched recapture.
  Audit complete joint-limit occupancy; candidate margins are not universal hardware facts.
- Protect credentials, hidden evaluation seeds and judges. Never log keys or put them in argv.
  Use the deployment's scheduler and permission scope, not a machine or allowance copied from elsewhere.
- Use source-bound admission for datasets. Training is downstream and only when requested; declare
  state/action mode explicitly and verify deployment compatibility.
- Runtime reads only explicit instructions and scoped experience. Keep new task output under the run
  workspace.
- Preserve user changes. New general behavior belongs in core/skills with meaningful tests;
  new per-task logic belongs in generated task outputs. Do not build a frozen task registry.

Engineering: install with `pip install -e '.[dev,workflow]'`, then `make test lint audit`.
Report implementation, CPU tests, real runs and unknowns separately. API/fixture success is not
proof of physical or visual quality. The source checkout is the deliverable.

## Learning from work in this repository

Read prompts/instructions/self-improvement.md when recording a correction or changing general behavior.
Capture visible user feedback, repair diffs and test evidence through pipeline_agent feedback or
pipeline_evolve record. Requirement events need actual user provenance and exact scope. Never
pretend the codebase can access a chat the host has not supplied. Task sessions automatically record
their results; use the optional post-run evolution driver/policy for bounded automatic improvement.
Use pipeline-improve for candidate changes; host evaluation determines activation. Preserve the
run's source pin, existing provider IDs, failures and prior evidence. New adapters remain products.
Report software evidence, real task results and meta-improvement results separately.
