---
name: task-expert-author
description: Generate a new task adapter and expert, then improve the observation-driven controller through isolated, budgeted simulator trials and independent evaluation.
---

# Write and iterate a new task expert

The orchestration/task-design role first generates the adapter, brief, seeded init and an independent judge; only after
review and freezing is the expert-harness started for the restricted author. The author role text is in prompts/roles/task-expert-author.md.
The orchestrating agent configures the adapter; no human fills it in by hand.

1. Against this real task, define the observations, actions, state/units/clock and success conditions; it is forbidden to read future target trajectories
   or to replay a fixed animation and pass it off as a closed-loop expert. Short action sequences may be used for verifiable sub-skills, but state judgments must come from actual observations.
2. Use the generic expert-harness adapter schema to wire up the rollout command, result fields, development seeds, budget,
   readable code/assets and protected paths. Write the observation→action plugin and the seed→initial-state plugin; implement missing interfaces proactively.
   Simulator-collected expert trajectories by default apply a constant support-surface randomization to every episode: table height
   (default ±20 mm uniform) and table tilt (default magnitude 1–2° uniform, direction 0–360° uniform; the tabletop and the objects on it tilt
   rigidly about the tabletop origin, while the robot base and cameras stay fixed), implemented in the task's scene sampler. Initial-state objects must land on the tilted plane; the expert reads the actual support
   plane and the live object poses from the observations and must not assume a level table. Turn it off only with a recorded reason (such as a paired level control or a hardware fact);
   write the reason into the plan. This is geometric randomization, needs a new full physics rollout to validate, and does not authorize appearance/OOD changes.
3. First run lint-brief, prepare-round, the built-in preflight of run-round, and a minimal complete round. Confirm the output media/actions/state, the independent judge receipts,
   the failure classification and the cost accounting. When isolation is unavailable, fix the environment or report the block honestly; never silently switch to none and claim independent pass.
4. The author runs development seeds only through the xh broker; read the actual rollout failures, event frames, joint readbacks and physics logs,
   and modify grasp, approach, transport, release/stabilization or recovery control; retries are limited, and all quality and infrastructure failures are kept.
5. An independent verifier checks the success semantics and the negative controls; the author must not edit or read the protected judge, holdout or budget ledger.
   When an exploitable loophole is found, report it and fix the general contract rather than exploiting the judge loophole.
6. Submit at every scoreable improvement; when the budget is exhausted, freeze the best expert. The harness runs the hidden holdout, records all denominators,
   and delivers the expert/hash/adapter/brief/qualifications; do not present a cherry-picked subset as a 100% first-attempt success.
7. After a holdout failure, the orchestrator decides further development based on a summary that does not leak hidden seeds; a new round keeps test isolation.
   Once qualified, move on to white-model batch collection; the code author must not sign admission itself.

Keep the three messages FIRST, CONTINUE and FINAL together with the budget/in-flight fields; the role text is an additional constraint and does not replace
these runtime messages. See docs/reference/expert-harness.md and prompts/instructions/expert-authoring.md.

## Efficient creation and debugging

Execute in order:

1. Get a first valid signal quickly: the shortest end-to-end draft (demonstrated order, annotated points, planner) + a stage table of ≤8 lines (stage, arm, exit predicate and tolerance) + stage markers,
   then immediately run a multi-seed rollout; write no alternative draft or recovery logic before the first rollout. Write the Task IR with the stage-graph profile, short, and refine it with evidence.
2. Escalate by evidence: only for a structural failure write one alternative and compare in batch; otherwise fix the first failing stage of the largest failure group, one hypothesis per round, at most 3 rounds per group.
3. Measure uncertain quantities first with a 2–3 seed probe; before an irreversible action, run a planning-only reachability probe and pick a feasible candidate.
4. Numeric margins (offset/height/angle) are found by a parameter sweep, not by guessing.
5. For setup-dependent failures use a bounded per-setup retry: the expert's self-check uses only its own terminal predicates (never reads the judge), the first attempt is numbered 0, and all attempts count in the denominator.
6. Stop only at ≥95% on 20 unseen dev seeds; between 85% and 95% keep fixing the largest failure group; submit the best measured version and report first-attempt success and the number of attempts.
7. Debug with stage summaries rather than raw logs; request images only when the summary cannot explain the failure.
8. Waiting for rollouts consumes model calls: keep rollouts fast (resident hot workers, enough GPU capacity), block once while waiting, and do not poll at short intervals.

## Low-cost checks before white-model batch expansion

Before the first complete white-model collection, finish the sim-grounding real-image check of a few poses from three views (or the views the contract specifies),
and save the independent review receipt and the camera/source hashes; a successful full physical action cannot compensate for a wrong view. Pure physics development probes
may run first, but must not be treated as camera-qualified or as a complete white-model pass.

Before the first complete rollout, check the real-source home, the simulated reset and all bound joint limits; record the SDK/URDF
intersection, the planning margin and the actual readback separately. Home carries no implicit tolerance, and one joint passing does not mean the others pass.
If a new reset is needed, keep the source and record the change explicitly, and recapture matching images/actions; never crop old labels.
Rehearse the complete transport and retreat path, including returning to home, from the actual grasp/release state; a single endpoint IK success is not enough to expand a batch.
From the very first episode the collector saves the state the contract requires (including the actual EEF), the initial frame, every control step and the final state,
and completes the all-joint/clock/field coverage check before scaling up to the user-specified count. For the SR-006 joint-margin
review, write the trace described in docs/reference/expert-harness.md and run `tools/audit_joint_margin_trace.py <trace.json> --output <report.json>`. The audit's
recorded_trace_passed only proves the given record and cannot replace review of the initial frame and complete episode coverage.

In the place/release phase, the existence of a collision manifold must not be equated with already bearing load: first check the surface gap of the actual collision geometry,
the contact impulse, or other observations that prove support, and only then release. On the first development trajectory record
the whole settling window before and after release, and distinguish an origin change caused by rigid-body rotation from an actual drop; do not compare only two adjacent
frames. Numeric display/lighting requirements must state the image region, the pixel algorithm and the stage interval, and be measured from the media;
display-state metadata cannot replace pixel evidence. The low-cost order is to locate the physical failure first, then collect the complete white model,
keeping independent review; never reduce cost by relying on missed detections.
