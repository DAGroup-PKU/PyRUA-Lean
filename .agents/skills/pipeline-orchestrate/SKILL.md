---
name: pipeline-orchestrate
description: Adapt any new real robot task/scene episode into verified batch data using repository-owned instructions and generated task products.
---

# From a new task to data delivery

Follow prompts/instructions/task-cost.md: all model roles share a persistent per-task cost ledger; the default is advisory, with a target of 30 and a reference limit of 50 USD, configurable manually; the semantics of disabled/advisory/enforced are in that file,
and photorealistic generation services and GPU are excluded. Orchestration loads only the skill of the current stage and does not inject all stage context every time; before entering the steps below, actually read the corresponding .agents/skills/<name>/SKILL.md.
The parent chat does not schedule redundantly, and waiting is handled by programs.
If the only next step is to wait for a submitted external job, write its external_jobs.id into state.model_wait_for;
drive refuses to start the next model session until the dependency ends. The supervisor verifies the process and updates the status; a model must not stand in as a poller
by querying repeatedly. When independent work remains, do not set a wait dependency. The protocol is in docs/reference/cost-control.md.

First read AGENTS.md, prompts/instructions/agent-start.md and prompts/instructions/instruction-layers.md.
You are the general orchestration capability of this pipeline; do not treat the absence of a task adapter as a blocker.

1. Save the user's original request, lock the input episode (video/record directory) and the target count, and run
   `python tools/pipeline_agent.py prepare`. Read the deployment configuration, existing resources and authorizations the user provides;
   inspect the actual files, the full video metadata and the key actions, and only then decide what is missing. Technical defaults are chosen and validated by you;
   do not make the user plan for you. An unknown mode blocks only paid rendering, not observation and coding.
2. Generate the new task products following the new-task-adaptation skill; build the scene following sim-grounding,
   and build observable control and an independent judge following task-expert-author.
   You must generate the task's input/output boundary, the complete success conditions, randomized initial states and an executable adapter.
3. Expert trial and error: record hypothesis, minimal discriminating experiment, actual trajectory/images, failure classification, correction. Distinguish environment startup errors,
   perception/calibration errors, physical-model errors, control errors and verifier misjudgments. Proactively write missing interfaces and general fixes,
   rather than only reporting problems. Keep at least one alternative explanation; it is forbidden to "raise the success rate" by changing the success conditions or deleting the failure denominator.
4. After the small-sample target is reached, freeze the code/contract and validate on independently held-out full episodes; check that action/state/image share the same source,
   left and right wrist identity, coordinates/units, clocks, joint-limit margin, release/terminal state/reset. Decide when to scale the batch only on explicit evidence.
5. Call render-batch: generate traceable real references from the real-robot input, compile the core plus task slots following render-policy,
   and check the actual provider payload. The agent must personally decode and view the real original image/episode and the generated video and compare them side by side,
   covering the real robot body, the attached cameras/cables and objects that actually exist, the table/background, and the photographic feel of every
   view declared by the current task (all three for a three-view episode); similarity of a local texture must not
   count as adopting the whole scene. OOD makes only reasonable real changes along approved axes relative to the real baseline. Run a representative small sample first, then scale the batch; stream the review,
   retry a limited number of times, and stop early on systematic failure. Real references provide appearance only; do not copy the action/layout in them or add entities.
   For camera or reference leakage, run a single-variable comparison; do not solve it by removing the reference. After a rejection, call verifier-rejection-recheck first.
6. Call dataset-delivery: after the complete review, an independent role issues the source-bound evidence, exports the aligned data, and checks reading.
   Count against the fixed target, and also give all attempts/failures/replacements; if the count is short, continue the authorized recovery and re-collection.
7. Continue the existing drive persistently through `pipeline_agent supervise --config`; native Codex stores an explicit session ID, and unknown usage is not treated as zero or a hard limit. Independent review writes state.review_requests (unique ID, instructions, path/sha256 of subjects), and the program actually schedules a separate reviewer session; the review result and execution provenance are in .runtime/reviews, with no self-signing. A user stop keeps the original ID, and resume does not reset the cost/session count. Throughout, keep accounts, double-check persistence and reuse remote IDs following datafarm-run-launch. Before a session is interrupted, maintain state.json
   and next_action, and on resume first verify the current status of the jobs/files. Do not stop the real task just because the chat/command returned.
8. Output delivery.json and a self-contained report, and run verify. The report distinguishes real verification, simulated fixtures and unverified items.
   Proceed to training only when the user requests it (dataset-delivery, step 6), and do not widen the goal.

State transitions rest on evidence. The new task's adapter, requirements document and video prompts are your deliverables and cannot be
required from the user in advance. After an independent review fails, fix the products and re-verify; a generated script that was never actually executed is not complete.
