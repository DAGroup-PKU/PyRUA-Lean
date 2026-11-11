---
name: new-task-adaptation
description: Infer a previously unseen task from a real episode and generate its scene, adapter, criteria, brief, render overlay and batch contracts.
---

# Generate task outputs from a new episode

Input: a complete real-robot episode, the current user goal, and the available robot/simulator deployment. Output: the code and contracts under this run/task,
never depending on the task name appearing in a registry. Read prompts/instructions/task-artifact-contract.md.

1. Observe the whole video, not just the final frame. Record each camera's identity, the robot/gripper, objects and counts, the grasp/contact/transfer/
   release/final-state time windows, occlusions and the initial state. Extract representative real-robot frames, and keep the original video SHA, frame numbers/times and crop
   mappings. When action/state files are available, read the schema, units and clock; if there are none, acknowledge the uncertainty of the motion estimate.
   An imported dataset episode carries `episode.json` (camera mapping, state/action fields, fps, task text): start there.
2. Write "what was done" as an appearance-independent object-relation / event contract; distinguish the incidental trajectory of the demonstration from the conditions the task requires.
   Infer the stage order, completion conditions, the randomizable range (by default including table height and a 1–2° table tilt, and the object placement mode `local_robustness` or `broad_table`, see task-expert-author), invariants and failure types. Tie every fact to an observed frame or
   device information; mark fitted parameters as inference. Ask only when the task goal is unclear; you do the technical adaptation.
3. Generate requirements.md, task.json, the scene/init code, adapter.json, brief, verifier spec,
   render-overlay.json and the batch/dataset config. It need not be perfect in one pass: first validate the interface with a minimal complete episode,
   then iterate on failure evidence. Examples must not carry another task's numbers or authorizations into the new task.
4. The independent task-contract reviewer compares the real robot with the predefined success conditions, and designs positive/negative controls, for example not grasped, only approached, not released,
   wrong object identity, wrong timing, not settled. The judge must decide from the actual external trace/images, not by returning the expert's
   check_success(). First verify with failure traces that it cannot pass falsely, then freeze the judge and let the author start development.
5. The new adapter uses the existing generic JSON/command interface of docs/reference/expert-harness.md. When a robot plugin is needed,
   write the observation→action interface and seed→initial_state; results land in a trace and an independent verdict.
   Validate the interface with `tools/expert_harness.py --help` / lint-brief / prepare-round / the built-in preflight of run-round, not merely JSON parseability.
   When a task-level interface is missing, implement it yourself; mark an external block only when a simulator/hardware fact or access permission is truly missing.
6. Whenever the scene changes, re-verify the cameras, object sizes/contacts, random initial-state coverage and dynamics, rather than just swapping the task text.
   The render overlay states object identity, events, views/clock and continuity; it does not cancel the real images, ignore actions or choose an OOD on its own.
7. Before batch expansion, freeze the hashes of this contract/code/config; the reviewer and producer execute separately, and the evidence names the version explicitly.
   When revising a judge, record the general reason, review it independently and create a new version; keep the earlier denominators and results.

See sim-grounding, task-expert-author, render-batch, dataset-delivery. All knowledge needed at runtime
is written into this task's outputs and must not remain only in the chat; generic experience is written back to the generic layer after testing.
