# General capability and task outputs

DataFarm takes a robot task, scene and real episode it has not seen before, and the agent autonomously
completes understanding, scene adaptation, expert development, verification, batch collection and photographic data
production. The input episode is a complete real-robot record: a single video, or multi-camera videos with
actions/states, camera parameters and device information. The output is a dataset that keeps its correspondence to the
simulated actions, with provenance and independent review records. Training is an optional downstream step.

## First layer: the shared capability

- `.agents/skills/`: how to observe, propose verifiable hypotheses, generate implementations, trial-run, diagnose,
  correct, scale the batch and resume.
- `prompts/roles/`: the responsibilities of each role.
- `prompts/render/`: the real-reference, action-fidelity and style requirements that a task cannot override.
- `prompts/instructions/`: authority, evidence and the handoffs between stages.
- `src/datafarm/agent/context.py` and `src/datafarm/task/contract.py`: load these instructions explicitly and verify task outputs.
- The expert harness, run ledger, batch supervisor, providers and admission: execution tools.

A new task with no adapter in the repository is not a capability gap: the agent generates a new adapter, brief, judge
and initial-state sampler from the new task. Judge a missing general capability by whether the agent knows how to
generate it, can call tools to verify it, and will diagnose and recover from a failure.

## Second layer: the products of one task

| Output | Who generates it, based on what | Purpose |
|---|---|---|
| grounding.json / grounding.md | The orchestrator observes the complete real episode, reads device/calibration evidence and fits | Geometry, cameras, objects, actions, the identifiable range and uncertainty |
| task.json / requirements.md | The task designer, from the observations and the user's goal | Success relations, complete action phases, randomization axes, physical and collection criteria |
| scene / init / expert / adapter.json | The agent, using the simulator API and general interfaces | Randomized initial states, observation-to-action control, the rollout command for the broker |
| judge / verifier spec / brief | Independent contract-review and verification roles, frozen afterwards | Success criteria and negative controls; the expert author cannot edit the judge |
| render-overlay.json / render prompt | The orchestrator, from the task, cameras and real references | The objects, events, views and continuity of this task; they cannot cancel the general rules |
| batch configs / dataset spec / receipts | The orchestrator and an independent verifier | Batch scaling, review, admission, data export and reproduction |

These files go in `runs/<run-id>/task/` or the workspace the user names, with the manifest and hashes registered. The
templates under `configs/` are starting points, not task results. Never load the thresholds, mode authorizations, model
choices or prompts of an unrelated task automatically. File extensions do not tell the layers apart.
`tools/audit_instruction_layers.py` classifies files by layer; the loading manifest is `configs/agent-instructions.json`.

## Changing the layers

Requirements specific to a new task go only in the second layer. Change the first layer only for a verifiable defect
that spans tasks, and update the skill, prompt, verifier and tests together. Success on one task does not authorize OOD
on another, and a numerical threshold of one task is not a general truth. Keep earlier outputs and failure evidence, and
never manufacture compliance with a new policy by rewriting existing certificates.
