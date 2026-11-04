# Phased task-creation protocol

This applies to every agent in a phased run. It encodes process rules, not task answers.

## Freshness: what you must not read

Use only the original episode input (the files in `work/input.json`), the general code, skills and hardware assets of the pinned source, and this task directory's own outputs. Do not open outputs of any earlier task, sibling tasks, or task-specific product files. Your brief may list further paths.

## Gates before GPU

1. **Measurement validity.** Evaluate every acceptance quantity on a no-action, unheld or resting baseline, and on a hand-made known-good and known-bad case, before it gates anything. Prefer geometry and pose-derived quantities. Any raw solver quantity (velocity, contact impulse) needs this baseline check.
2. **Satisfiability.** Check each threshold against hardware limits (joint ranges, gripper aperture and force, reach) and modelled object dimensions.
3. **Asset plausibility.** Compare object dimensions, mass, density and friction with the real episode (pixels with a camera fit, robot readings such as the gripper opening while holding) and with physical norms.
4. **Headless CPU first.** Build a headless simulator runner on day one and confirm it matches an image-enabled run. Develop on dev seeds. Pre-register a held-out seed list with a timestamp before repair work, and run it once after freezing source hashes. Observed seeds are never fresh again.
5. **Infrastructure.** Confirm which GPU hosts render correctly before batch capture. Every judge and evaluator has an internal timeout below the worker deadline, set from a run measured on a busy node.

## Agent economy

- One fresh agent per phase, with a compact handoff file instead of a resumed long-context agent. Summarize tool outputs; never dump large files into context.
- Independent reviewers are separate fresh sessions, one per decision point. Review several episodes in one pass.
- A deterministic checklist script gates capture attempts, not a model. Models review only completed candidate episodes.
- Each phase has a hard USD cap, given in its brief. The host kills the phase at that cap. If you are near it, write the handoff and stop.
- A handoff is at most 1 page. Evidence is a per-seed table, logs and a source-freeze sha manifest. Do not build integrity frameworks.

## Independence and admission

Producers never certify success. Independent review decides the contract freeze and episode acceptance. Admission is rejection sampling: pre-registered seeds in order, every attempt in the denominator, success rate and selection bias disclosed. Criteria are never relaxed to pass.

## Contract scope

The deliverable is in `work/input.json` (`request`): complete, independently reviewed white-model, action-labelled episodes. Rendering and training are not part of this flow.

A requirement may block admission only if failing it could admit an episode that is not a correct, physically valid, action-labelled instance of the real task. Such requirements cover:

- the correct task steps, order and outcome compared with the real episode;
- valid contacts, forces and joint limits;
- no attachment or teleport;
- labels and images bound to the logged states;
- the user's explicit task requirements.

The following are advisory by default. Record and disclose them, but they never block:

- evidence-infrastructure demands: OS-level isolation proofs, exhaustive artifact-binding audits, sensitivity or plausibility matrices beyond the frozen gates, and cross-render parity beyond one same-seed headless-versus-image state check;
- numeric camera-reprojection budgets. For white data, camera fit blocks only if a task-relevant object or contact is missing from a view in which the real reference shows it, or if a view is grossly wrong.

Gate thresholds that are kept are never loosened. A gate that cannot separate a known-good case from a known-bad case is a broken measurement. Fix it as a new judge version, independently reviewed against those cases. Never silently relax it or work around it.

## Repair loop

- A review that denies lists at most 5 blocking items. Each item needs an exact, executable acceptance check: a command plus the expected result.
- Every other concern goes in an advisory list.
- Later reviews may not add blocking items without new evidence of a correctness defect.
- The producer keeps the best-known source pin, the one with the highest dev full-pass rate. A change that lowers the dev full-pass rate is reverted before any held-out run.
- Held-out seeds are spent once observed. Use a fresh pre-registered range for each frozen source.
