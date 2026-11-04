# Rendering: fix the target, then verify the references and the final video

Rendering turns the action-authoritative white-model video into a real-world-style video. Never rewrite actions to improve appearance. First apply
`prompts/instructions/mode-and-style-contract.md`.

## Before submission

- Confirm the mode and the approved scope, and register the real image sources and hashes.
- Check the paths in the configuration against the actual request item by item. Record the provider reference slot mapping and the final expanded prompt.
- For a Seedance task, put the full-scene real images into `job.real_references` in authority order, and use `real_scene_reference` for each item in `references`; source and continuity images may only come after them.
- After the payload is built and before the POST, check that the actual image hash prefixes match the render intent exactly.
- An ordinary source first frame may only supplement geometric anchoring.
- If a real reference is unreadable or overridden by source-only, stop. Automatic downgrade is forbidden.
- Read protected credentials at runtime only.

## Canary

For Seedance, choose a representative small batch according to task length and variation. H3 first runs 2 complete episode prefixes, stops if any real appearance has not been adopted, and then runs the
fixed 8-item canary. H3 defaults to the control hybrid route (`docs/reference/rendering.md`). Compare the white model, the real references and the generated video item by item, covering
approach, grasp, move, release, final state and seams. First, middle and last frames can only serve as a preliminary screen.

Record structural normality and style normality separately. Both the executing agent and the independent reviewer personally decode the real images or episode and view them beside the generated video. Save
comparison images and frame numbers for the real baseline, the robot body, the attached cameras/cables and objects that actually exist, the table/background, and the photographic feel of all declared
views. A component that does not exist uses `not_applicable` with an evidence-backed reason and must not be invented. Whole-scene conclusions use full-view evidence; local texture, the prompt or a hash
cannot substitute.

A real object's outline improving toward the real reference is not a deviation from the white model, but object counts, contacts, camera identity and events must be preserved. When an extra gripper leaks in from a
reference, fix the cropping or role of the real image, the way it is referenced, or the prompt, and verify with a paired small sample first. Never delete all the real images. Any cropping keeps the source mapping and
must still provide valid real-appearance evidence. The real reference supplies appearance only and must not copy its actions or layout or add entities.

## Render verifier

Every task has a render verifier spec. Copy `configs/render-verifier/template.json` into the task workspace, write one entity per Task IR role and declare panels, events, seams and the realism mode
(`docs/reference/render-verifier.md`). After independent review has accepted a canary render, calibrate the spec with `tools/calibrate_render_verifier.py`: the positive control must pass and every constructible
counterfactual must be caught. Then run `tools/verify_render.py` on each generated episode and keep its `receipt.json`. A machine pass does not replace the independent review, and on the H3 route it does not replace EVAL-v1.

## Batch expansion and stopping

Verify each item as it completes. When a high proportion of the same kind of problem appears in the first 5-10 items, pause new submissions and look for the systemic cause rather than waiting for the whole batch. The
actual batch policy sets thresholds, and they must not be lowered to reach the target. Upload or service crashes and quality rejections are separate: an infrastructure failure does not consume the quality seed budget. Use
bounded retries, backoff, and a stop on consecutive zero progress. Measured throughput is not a general concurrency rule.

## Review standards

The real references and the approved changes together define the target; an OOD change must still be a plausible real change from the real baseline. Hard shadows and directional light are not defects in themselves, and exposure
drift of 60/255 by default is only a coarse guard. Focus on added or missing objects and arms, object and container topology, wrist-view swaps, reference layout leakage, stage order, final state, seams and unexplained temporal changes.
Tell a pre-defined layout gutter apart from added black borders in the frame.

Every automatic rejection is first independently rechecked per `verifier-rejection-recheck`; do not report it to the user as a certain defect. If the full evidence has not been seen, it is unverified. Fix a false positive by repairing the
general rule and recalibrating, not by changing a threshold for one sample. Output a per-sample table and a side-by-side review page so the user can recheck; the review page does not replace a certificate.

## Completion

Fully record the intent, payload, reference sources, final video, and independent style and structure evidence. The target count, candidate attempts, crashes, each kind of rejection, admitted count and cost must be traceable. Technical
success, a machine gate pass or a full file count must not be used to claim the real style is qualified.

Task-specific facts stay in the task's contract: a real device's shape should match its real references and must not copy a simplified simulated proxy. Object counts, colours and left/right arm order come from the current Task and Episode
contract, and no single task may be hard-coded as a general rule.
