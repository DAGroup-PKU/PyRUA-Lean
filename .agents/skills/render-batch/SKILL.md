---
name: render-batch
description: Run or resume a reference-grounded, explicitly scoped DataFarm render batch with canary and independent review.
---

# render-batch

Before any render/resume/scale decision, read `prompts/instructions/mode-and-style-contract.md` and `prompts/instructions/render.md`.

H3 means the control hybrid (MiniMax-H3 Ref2VA + ControlNet-Union soft-HED, independent chunks, 8 seeds + per-chunk VLM screen, first-chunk exposure lock): read `docs/reference/rendering.md`, start from `configs/render/h3-control-hybrid-v1.json` (OOD tabletop/light: `configs/render/h3-ood-restyle-v1.json`) and run `src/datafarm/render/h3/`; accept only through the independent EVAL-v1 blind review (`evaluations/render-h3/`). Bundle, references and recipe copies are task artifacts you derive for the current task.

1. Resolve mode from an explicit in-scope user instruction, or ask and wait. Both OOD and non-OOD require real-world images and real-world output style across the whole scene. OOD permits only approved, physically plausible real-world changes from the recorded baseline. Never infer OOD from provider, task name, geometry randomization or another task's recipe.
2. Bind real-image provenance/hash and verify actual provider payload; reject source-only and missing/structural-only references. Do not diagnose leakage by dropping every real image.
3. Register/launch via datafarm-run-launch. Stay within the GPU and provider permissions of this deployment; control costs and retries.
4. Canary first: use a representative small Seedance sample; H3 two prefixes then the eight-sample canary. Personally decode and view the actual real references/episode and generated videos side by side. Independently inspect the real baseline, robot body, mounted cameras/cables and task objects when present, table/background and photographic consistency in every task-declared view through all task phases. Record absent components as evidenced not-applicable rather than inventing them. A local texture/crop is not full-scene coverage; prompt text and file hashes are not visible-adoption evidence. Do not scale before canary evidence.
5. Stream checks, pause systematic early failures, use the shared batch supervisor for bounded retries/backoff. Separate infrastructure failure from quality rejection; retain fixed denominators and replacement lineage.
6. Write this task's render verifier spec from `configs/render-verifier/template.json` into the task workspace, following `docs/reference/render-verifier.md`. Calibrate it on an accepted canary render with `tools/calibrate_render_verifier.py`, and run `tools/verify_render.py` on every generated episode as it completes. A spec that is not calibrated is not evidence. On the H3 route the verifier runs alongside EVAL-v1; neither replaces the independent review.
7. Recheck verifier rejections independently before reporting defects. Allow real-world hard shadows and approved OOD variations; gross exposure default 60/255 is not a realism verdict.
8. Report style, reference adherence, structure, coverage and unverified items separately. Produce bound side-by-side comparison images and generated frame indices for every required appearance area, plus ledger completion records. Real references govern appearance only: never copy their action/layout or add entities absent from the source carrier. Task-specific appearance contracts belong in the task's own configs and docs.

Deployment scope: GPU and paid-provider authorization comes from the current deployment and session. Obtaining the repository grants none, and permission already given is not asked for again.
