# Render verifier

This document describes how a generated video is judged by machine before it can be admitted. Every task has a spec.
The code is in `src/datafarm/verify/`, the entry point is `tools/verify_render.py`, and calibration is `tools/calibrate_render_verifier.py`.

## 1. Fixed part and task part

The Render layer repaints the simulated white carrier into a target-domain video. The judgment has two parts:

- **Fixed part (code, does not change with the task):** the check families, the decision logic, the persistence rule, fail-closed behaviour and the report format.
- **Task part (data, written by the agent):** a JSON spec with the panel layout, the colour keys and geometric primitives of each role in the Task IR, the object names in the state, the real-robot references and the OOD perturbation axes.

A new task only writes a spec, not a new verifier.

## 2. Inputs and authority

| Input | Authoritative for | Source |
|---|---|---|
| carrier (white-model video) | geometry, entity visibility, timing | Episode IR |
| Simulation state (object poses, camera K/extrinsics, gripper commands) | full object outlines, event instants | Episode IR |
| Real-robot video (same layout) | appearance, camera exposure and flicker behavior | Grounding IR |
| Appearance reference image | only used to check layout leakage | Render IR |
| Generated video | the object being judged, with no authority whatsoever | provider |

## 3. Nine check families

Each result is one of four values:

- `pass`;
- `fail`;
- `abstain`: the authority needed for the judgment is missing or it cannot be measured; **blocks admission**;
- `not_applicable`: objectively unmeasurable, for example a moving wrist camera has no static background; does not block admission.

A sample passes if and only if all hard checks are pass or not_applicable. A defect must persist for `persist_samples` (default 3 sampled frames) before it is judged a failure.

1. **transport**: frame count, fps, aspect ratio, black frames, sha256.
2. **structure**:
   - for each panel and each sampled frame, compare the cosine of the projected-geometry descriptor (blurred grayscale + gradient);
   - estimate the global translation with phase correlation (this can catch squashing or misalignment);
   - black borders at the panel edge: a uniform near-black band (>= 4 layout px) appears at a panel edge of the generated picture that the white model does not have, and persists; this is judged a panel crop or offset error.
3. **entity** (one item per Task IR role):
   - *missing*: an instance visible in the white model has no pixels of that entity near it in the generated video;
   - *unsupported*: an entity-sized connected blob appears in the generated video, but nearby there is no white-model instance, no simulated projected outline, and no other entity;
   - *extent*: the area of the generated entity lying outside the simulated projected outline is too large. It is treated as unmeasurable when it touches other entities or is truncated by the panel edge.
   - Generated-side segmentation first learns the entity appearance **of this video itself** (fitting a CIELAB model on pixels authorized by the white model; multi-color objects are learned part by part), and then adds a chroma floor. In this way the entity checks judge only geometry and presence, and whether the appearance looks like the real robot is left to realism. In OOD lighting mode only this adaptive model is used, not the fixed target keys.
   - If more than 2% of the background far from any simulated object is segmented as that entity (for example, a new table material with the same colour as a task object), the unsupported / extent items of that panel are judged abstain: color can no longer serve as evidence.
   - A sub-check can be restricted to a panel, written as `unsupported:head`.
4. **event**:
   - for each event (gripper open/close, or frames given by the spec), compare the motion signal within the ROI of the entities involved;
   - a task may use `event_local_motion_min` to require that the white-model ROI within four frames before and after the event has measurable motion; if the motion exists only in other parts of the window, the correlation peak is not used to claim this event is misaligned. The default 0 disables this requirement;
   - search the lag, requiring |lag| <= 6 frames and correlation >= 0.3.
5. **temporal**: three things are checked per panel at full frame rate (jumps and brightness steps are first normalized by the contrast of the two videos respectively):
   - brightness steps that cannot be explained by the white model: the default upper limit is 10/255; the real camera's own step statistics can only raise it, never lower it;
   - jumps;
   - freezes.
6. **continuity**: two kinds of problems are judged separately at segment seams:
   - photometric jump: same threshold as flicker;
   - structural jump: first do gain/bias matching, then require the change to exceed both a multiple of the local motion and an absolute floor.
   - Abstain when segments are rendered but no seam is declared.
7. **background** (fixed cameras only): on pixels where the white model does not change, the generated picture should not change either.
   - A shadow margin is excluded near objects;
   - darkening is tolerated within a wider "shadow reachable range";
   - for each frame, first compute the fraction of static pixels that are still measurable after excluding moving objects and margins; if it is below the spec's `background_min_static_fraction` the frame is not judged. The default is 3%; a task with little static background may raise it. If fewer than 30% of the sampled frames in the whole segment are measurable, the panel returns not applicable, and no pass or reject conclusion is drawn from a few pixels mixed with moving shadows;
   - if a large dark shadow of a moving object covers most of the judged pixels, almost no static background lies far from the motion, and the upper quartile of the judged pixels stays close to the first-frame exposure, the exposure is taken from that upper quartile, so the shadow is not fitted as an exposure drop. A true global exposure drop has no such upper quartile;
   - brightening is not tolerated, and the area of connected bright-spot blobs is checked separately;
   - overall exposure drift is only a coarse guard (default 60/255).
8. **realism**: per panel and per region (each entity + background), compare six features against the real-robot picture:
   - color, contrast, relative chroma, texture, sharpness, noise.
   - Each is divided by a fixed tolerance (about one just-noticeable difference), then the real robot's own first-half/second-half fluctuation is allowed. The hard threshold is 6.0, and 3.0 and above is only a reference warning. A direct copy of the white model and a recoloured one still fail.
   - Regions with too little coverage in the real-robot video are recorded as unmeasurable.
9. **reference_leak**: if an entity in the appearance reference image is not at its white-model position but is filled in at the first generated frame, it is judged layout leakage.

## 4. sim2real and OOD

- **sim2real**: all features of all regions must be close to the real robot. No fixed lighting style is imposed: if the real footage has hard shadows, the render should too.
- **OOD**: first bind this task's explicit authorization, the real baseline and real images; the spec declares only the perturbation axes within the authorized scope (`lighting`, `background`, `table_material`, `distractors`, `camera_exposure`).
  - Authorized perturbation axes do not require pixels/features to equal the real baseline; the numeric gates skip colour and contrast for lighting and the background region for background axes, but this does not skip the real-style review. Changes beyond the authorization must still be rejected.
  - The robot and the task objects are invariant entities and must still look like the real robot.
  - Geometry, event, temporal and entity checks are exactly the same as in sim2real.

## 5. Workflow for writing the spec of a new task (executed by the agent)

Copy `configs/render-verifier/template.json` into the task workspace; the spec is a task artifact.

1. List all visible roles from the Task IR, and write one entity per role:
   - `carrier_keys` segment on the white model;
   - `target_keys` loosely cover the target-domain appearance;
   - if the real reference object is dark, strict `target_keys` will report an object that is still visible as missing; a separate `presence_keys` can be set, used only for the presence check at the white-model expected position; the new-object and volume checks still use `target_keys`. Before enabling it, both the missing-object and the same-color added-object counterexamples must be replayed; passing the missing-object case alone does not establish entity-count admission.
   - Multi-color objects list keys part by part, with the white-model keys and target keys matched one-to-one by index;
   - when state exists, also write `state_objects` and the geometric primitives.
2. Write the panels, camera names and native resolution, then visually confirm the projection alignment with the projection overlay image.
3. Bind the real-robot reference (same layout) and the source/role of the real appearance image actually submitted, and record the mode confirmed by the user and the authorized perturbation axes; ask first if the mode is not confirmed. The existence of a reference used for verification cannot replace evidence that the generation request carried the real image.
4. Run `validate_spec`:
   - every Task IR role must have an entity check;
   - segmented rendering must declare seams;
   - OOD mode must declare perturbation axes;
   - exemptions must state a reason.
5. **Calibration**: pick a human-approved render as the positive example and run the counterfactual set, requiring that the positive example passes and that every class of counterfactual is caught by the corresponding check family. The counterfactuals include:
   - removing an object, adding an object, enlarging by 1.5x;
   - time shift, freeze, flicker;
   - swapping the left and right wrists, horizontal squashing;
   - background relighting, hallucinated light spots;
   - seam frame jump, recoloring, direct copy of the white model.
6. Run on the whole cohort, and spot-check the failure evidence and partially passed samples. When a false positive is found, fix the general rule and rerun calibration; do not tune thresholds to a single sample.

## 6. Commands

```bash
python tools/verify_render.py --spec <task-workspace>/verifier-spec.json \
  --carrier white.mp4 --generated render.mp4 --real real-tlayout.mp4 \
  --state frame-aligned-robot-state.json --seams <frame> --out <dir>
python tools/calibrate_render_verifier.py --spec <task-workspace>/verifier-spec.json --carrier white.mp4 \
  --generated accepted.mp4 --real real-tlayout.mp4 --out <dir> [--waive robot_ghost="no robot entity in this spec"]
```

Both tools exit 0 only on `pass` / `calibrated`, and 2 otherwise. The output is `receipt.json` (machine-readable) and `review.html`: every failed or abstained check carries "white model | generated | overlay" evidence images, and temporal evidence shows the frames before and after.

## 7. Guarantees

1. **A verifier spec that breaks the authoring rules cannot pass anything.** `validate_spec`
   checks the following rules:
   - P0: panels are unique and lie inside the layout.
   - P1: every Task IR role has an entity with a `missing` check and carrier/target colour
     keys.
   - P2: events name known roles.
   - P3: a chunked render declares its seams.
   - P4: the realism mode is explicit; `sim2real` has no perturbed axes; `ood` declares
     axes from `OOD_AXES`.
   - P5: waivers name a known family and give a reason; thresholds are known.

   Any violation returns verdict `abstain` with `spec_problems`.
2. **The verdict fails closed over fixed check families.** The families are `transport`,
   `structure`, `entity` (presence, missing, extra/unsupported, scale), `event`, `temporal`,
   `continuity` (seams), `background`, `realism` and `reference_leak`. Any hard `fail` makes
   the verdict `fail`. Otherwise any hard `abstain` makes it `abstain`. A family can be
   switched off only by a reasoned spec waiver, which yields `not_applicable`.
3. **Realism is judged against real footage, never assumed.** Without a real clip,
   `realism_vs_real_world` abstains. A render that is just the carrier fails. In `ood`
   mode, only the declared axes relax features. Background axes drop the background
   region, and `lighting` or `camera_exposure` drop the colour and contrast features.
4. **A spec counts as calibrated only after a complete, clean mutation run.**
   `calibrate_render_verifier.py` verifies the accepted render (positive control), then
   injects each of the 14 defects in `counterfactual.SUITE`. These are entity
   delete/duplicate/scale, time shift, freeze, flicker, panel swap, squeeze, relight drift,
   light spot, seam jump, recolour, white (carrier) copy, and robot ghost. `summarize` sets
   `calibrated` only when all of these hold:
   - the run is not partial (`--only` runs are diagnostics);
   - the positive control's verdict is `pass`;
   - every constructed defect fails in its expected check;
   - every not-constructible defect has an explicit `--waive NAME=REASON`.

   Waivers naming unknown defects raise, and a waiver never excuses a missed defect.
   `calibration.json` (`datafarm.render-verifier-calibration/2`) records `spec_sha256` and
   `generated_sha256`. The tool exits 0 only when `calibrated` is true.

Code: `src/datafarm/verify/`, `tools/verify_render.py`, `tools/calibrate_render_verifier.py`.

## 8. Data model

- **Verifier spec** (`datafarm.render-verifier-spec/1.0.0`): layout and panels, entities
  (role, carrier/target colour keys, checks, panels), events, `realism {mode,
  perturbed_axes}`, thresholds, waivers, seams. Start from `configs/render-verifier/template.json`.
- **Verifier receipt** (`receipt.json`): `verifier` (`datafarm-render-verifier-v2`), `verdict`,
  `failed`, `abstained`, `thresholds` with provenance, input paths and sha256, `checks[]`
  (`name`, `family`, `verdict`, `hard`, `reason`, `metrics`, `evidence`), and
  `episode_binding`.
- **Calibration** (`calibration.json`, `datafarm.render-verifier-calibration/2`): `spec_id`,
  `spec_sha256`, `generated_sha256`, `positive_control_passes`, `caught`, `total`,
  `constructed`, `missed`, `not_constructible`, `unwaived_not_constructible`, `waivers`,
  `partial`, `calibrated`, `results`.

## 9. Configuration

| Flag / key | Default | Effect |
|---|---|---|
| `verify_render.py --families` | all | Restricts families. A restricted run can still return `pass`. |
| `--mode`, `--perturbed-axes` | from spec | Override the realism target for this run. |
| `--single-chunk`, `--seams`, `--expected-frames` | spec | Episode binding for continuity and transport. |
| `calibrate --only` | all | Diagnostic subset; always `partial`, never calibrated. |
| `calibrate --waive NAME=REASON` | none | Accepts one not-constructible counterfactual. |
| spec `waivers {family: reason}` | none | Turns a family into `not_applicable`. |

## 10. Known limitations

- Entity segmentation is based on color keys and the adaptive appearance model; no learned segmentation or tracking is used. Sizes at occlusion boundaries are unmeasurable.
- "Appearing from nowhere" for the robot is not judged by color: black links, dark projections and dark walls cannot be separated by color. Instead `semantic_unsupported` is used (`semantic.py`, requires a GPU and the environment variable `DATAFARM_DINOV2`):
  - prototypes are learned from the DINOv2 features of the white-model arm region in this video;
  - if somewhere looks like an arm semantically while the white model has no simulated content there (the white-model foreground is obtained from the temporal-median background and temporal differencing, which can cover light gray links), it is judged a hallucination;
  - if the feature distributions of the arm and the background are themselves inseparable, it is judged abstain.
  - The "arm ghosting" counterfactual is part of calibration.
- Shadow jumps in the white model itself (simulated shadow-map artifacts) will also make the generated video fail. This is a real visible defect, but it should be fixed back in the Episode or white-model capture stage; the report does not label it as originating from the white model.
- The hard realism threshold is a default, not a per-task value. Run `tools/calibrate_render_verifier.py` for each new task.
- There is no semantic-level judgment, for example whether the action was truly completed or whether an object was truly placed. This part still relies on terminal-state entity checks and human review.
- **Calibration covers no reference-leak mutation.** `counterfactual.SUITE` has no
  reference-layout-leak defect. `reference_layout_leak` is therefore never calibrated. It is
  `not_applicable` when no reference image is passed, and it inspects only the first five
  sampled frames, for entities that have an `unsupported` check.
- **Spec waivers cannot be calibrated.** A family waived in the spec is `not_applicable`.
  Counterfactuals that expect that family are then reported missed, and calibration fails.
  `--waive` covers only defects that cannot be constructed.
- **No task spec ships.** Only `configs/render-verifier/template.json` is included; each task's spec is
  an output of the run.
