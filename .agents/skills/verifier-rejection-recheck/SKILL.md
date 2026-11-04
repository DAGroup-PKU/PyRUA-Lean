---
name: verifier-rejection-recheck
description: Independently recheck render verifier/VLM rejections against source motion, actual real references and authorized style.
---

# verifier-rejection-recheck

Read `prompts/instructions/render.md` and the task's verifier spec. Automated rejection is a hypothesis to investigate, not a fact to relay unchanged.

1. Record gate, threshold, measured value, sample, frames/timestamps and verifier version. Compare source and generated measurements; locate visible evidence.
2. For appearance, require actual real-image references and in-scope mode/axes. OOD axes permit only reasonable real-world changes from the baseline. Missing references or evidence remains unverified and cannot be promoted. A pure-white source frame is not real appearance evidence.
3. Personally open/decode and compare source, real images and generated video across relevant phases; export bound side-by-side evidence/contact sheets or review pages. Cover the reference baseline, robot body, mounted cameras/cables and task objects when present, table/background, and photographic consistency in every task-declared view. Record absent components as evidenced not-applicable rather than inventing them. A local texture/crop cannot establish full-scene realism. Do not claim whole-video validity from sparse frames, prompt text, hashes or another reviewer's prose.
4. Distinguish confirmed defect, verifier false positive and unverified. Explain measurement and visibility. Correct false-positive general rules and recalibrate, rather than adjusting one sample's threshold.
5. Approved OOD variation, realistic object shape relative to crude proxies, hard shadows and directional light are not inherently defects. OOD must actually be authorized for this run. Exposure default 60/255 is only a gross guard. Task-specific contact semantics come from the current task contract, never a different task example.
6. Prioritize hallucinated arms/objects, wrong counts/topology, wrist identity, action order, terminal relation, seam jumps and unexplained temporal appearance changes. Distinguish expected panel gutters from generated black artifacts.
7. Report sample | gate | measurement | threshold | visibility | verdict | evidence. Real references supply appearance, never action/layout authority or permission to add entities. Never rerender solely on an unexamined VLM rejection; missing required reference provenance or direct visual-comparison evidence still blocks admission.
