# Task expert authoring and isolated evaluation

A task expert is the control program that lets the simulated robot complete the task. The author works through `tools/expert_harness.py` and must not start hidden
rollouts, modify the judge or view holdout seeds.

The operator here is by default the orchestrating agent: it generates the adapter from the new episode, and the user is not required to fill it in. In the adapter the
orchestrator declares the interface, physical and camera facts, development seeds, budget, concurrency, allowed and protected paths, verifier version, acceptance conditions and
backend/model. The harness generates and lints the brief. The author uses the broker tool `xh` to request rollout, submit and flag. The harness accounts attempts, cost, versions
and status; the author's self-report cannot replace them.

Development reads only approved inputs and changes only the designated working directory. Restrictions are inherited as they are, and suggestions in a report are not permissions.
Each round keeps the failure classification, the best expert hash, the test evidence and a safe handoff summary. The harness executes the holdout on the isolated side, and the
handoff must not leak seeds or per-item identifying information.

The first message contains the task and the complete brief. The continue message carries the current session, rollout, budget, in-flight and state information. When the budget is
exhausted, the final message allows only freezing the submission and reporting. One general system prompt cannot replace these three messages. Whether the environment really
implements isolation, complete cost accounting or holdout leak protection depends on code tests and this round's probes, not on the design description.

## Support randomization

Simulator collection by default randomizes the support surface per episode: table height uniform within ±20 mm, table tilt with magnitude uniform in 1-2° and arbitrary direction,
implemented in the task's scene sampler. The tabletop and the objects on it tilt rigidly about the tabletop origin; the robot base and cameras stay fixed. The plan records
`table_tilt` and `table_tilt_policy`, the materialized alignment records `support_plane_tilt` (a plane point and normal), and `support_z_at` gives the support height at any xy. Turning
off the tilt requires a recorded reason. Tilted samples need a new full physics rollout and an independent judge; actions or images must not be modified offline.

## Boundaries

Confirming the rendering mode and the real references is the orchestrator's job. An expert author producing a geometric randomization scheme does not obtain OOD appearance authorization.
Training and rendering go to their own flows.

The creation and debugging method is in `.agents/skills/task-expert-author/SKILL.md`. Expert runtime interface (any simulator implements the same semantics): stage markers, predicate records
(summary only), planning-only reachability probes, bounded per-setup policy search (expert self-check, fresh reset and attempt counting) and sweep parameters. Single-run variance is large, so
adopt a change to the author method only after a paired comparison with repeats, including held-out tasks.

Use existing stage digests and handoff artifacts rather than inspecting session or provider logs, and do not launch another orchestrator. An expert passing software checks is not evidence that a
task can be completed within the target cost. Backend and transport details are in `docs/reference/expert-harness.md`.
