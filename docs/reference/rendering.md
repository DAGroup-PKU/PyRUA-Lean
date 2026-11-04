# Rendering: H3 control hybrid (default) and Seedance

The renderers turn a simulated white carrier video (actions, objects and timing already fixed) into photographic multi-camera video. The default renderer is the **H3 control hybrid**: MiniMax-H3 Ref2VA with an H3-Fun ControlNet-Union branch driven by gated soft-HED edges of the carrier and a white inpaint source. It renders independent chunks on a tiled canvas, with 8 seeds per chunk selected by a VLM, minimum-difference seams and a first-chunk exposure lock. The second renderer is the hosted **Seedance** API. Each request is bound to the task's [render contract](render-contract.md). The agent's rules for canaries and review are in [`prompts/instructions/render.md`](../../prompts/instructions/render.md).

## Rules shared by both renderers

Renderers only produce candidate observations. They cannot decide the action, physical state, camera, object count, contact, task success or training admission. A successful provider return, a decodable MP4, an equal frame count or a realistic picture is not a pass.

| Content | Final authority | May the provider change it |
|---|---|---|
| Action, state, contact, task events | Simulated episode | No |
| Camera identity, intrinsics, extrinsics, per-frame pose, panel role | Grounding record and episode | No |
| Object and robot count, identity, geometry, pose, topology, occlusion | Simulated episode | No |
| Canonical frame index and action mapping | Simulated episode | No |
| Allowed changes to material, colour, texture, lighting and the task-external scene | Render contract | Only within the declared scope |
| Purpose and usage rights of a reference | Grounding record and render contract | May not widen on its own |
| Whether the output is acceptable | Verifier, independent review, admission | The provider may not sign |

A **canonical frame** is the action and state authority frame; its index does not change with the provider. A **transport frame** is encoded to meet a provider's FPS, duration or frame-grid requirements and must map back to a canonical frame. A **recipe** is a frozen combination of provider profile, prompt, reference bundle, frame and chunk plan, model and verification version.

The following is procedure, carried out by the agent and its reviewers:

- **Freeze inputs.** A batch freezes the denominator, sample identity, train and held-out split, and recipe. Repeating a task key resumes or returns the original attempt and never silently creates a second paid task.
- **Fail closed on the profile.** A profile declares the provider, model, route, input FPS, frame or duration envelope, frame grid, concurrency and qualification scope. One provider's limits are not inherited by another. When a profile is missing, the envelope is exceeded or the recipe is not qualified, stop instead of guessing.
- **Give each reference one narrow role**, checked before the GPU or API call. A scene plate covers only the task-external scene, tabletop, exposure and lighting. An object appearance reference is position-neutral where possible. A robot appearance reference must not carry an incompatible full arm pose. A structure guide must agree with the camera, object projection, robot occupancy, interaction state and global frame. A continuity guide comes only from a selected parent segment of the same episode and is bound to its hash.
- **Keep frames exact.** Save the canonical clock, the transport FPS and canonical index of each output frame, the reversible chain of panel order, size, crop, pad and resize, and each chunk's source range and keep range. Each canonical index appears exactly once, with no interpolation, deletion, insertion or crossfade in the action region.
- **Scale only after a full-episode canary.** Deterministically select at least 8 setups covering both ends and the middle of the layouts, each a full episode with all key events, the terminal state and every seam. The fixed recipe must pass at least 75%, and no common failing check may cover half or more of the canaries. Attribute failures to `temporal_control`, `geometry_reference`, `appearance_reference`, `continuation`, `provider` or `verifier`. Any change of prompt, reference, guide, segmentation, profile or verifier invalidates the canary. Picking the best of several seeds, or running only the first chunk, is not production authorization.
- **Resume, do not resubmit.** Keep the request digest, task ID, model, seed, status updates, raw response, output hash, cost and error class. With a task ID, only poll and download. With an unclear create, keep the ambiguous attempt and quarantine it. On a definite non-retryable 4xx, mark the input blocked. Raw output is never overwritten.
- **Gate in order:** transport; camera and panel; entity count and identity; event timing; gripper, object and container topology; terminal state; appearance (reference adoption and source preservation are separate hard gates); continuity at every seam; provenance, rights, cost and the complete denominator. A high correlation or realism score cannot override a hard error in entity, contact, phase or terminal state. Insufficient evidence is `abstain`, never a pass.
- **Repair at the right level.** For an isolated stochastic rejection, freeze accepted samples and chunks, redraw only the implicated unit with a deterministic new seed, and rerun the unchanged verifier. For a systemic failure or low-yield prefix, stop expansion, repair at the earliest owner, and rerun the same canary. The denominator does not change, failed samples and their cost are kept, and a full pass must not come from relaxing thresholds, replacing the carrier, changing actions or dropping hard setups.

## H3 control hybrid

| Location | Content |
|---|---|
| `src/datafarm/render/h3/` | Render and post-processing tools; see its `README.md` for a step-by-tool table |
| `src/datafarm/render/h3/BUNDLE-SCHEMA.md` | Task bundle format |
| `vendor/videox_fun/` | Patch from upstream VideoX-Fun that adds `MiniMaxH3ControlPipeline` |
| `configs/render/h3-control-hybrid-v1.json` | Task-independent default recipe, including the reference policy |
| `configs/render/h3-ood-restyle-v1.json` | OOD edit of tabletop material and light colour |
| `evaluations/render-h3/` | Frozen EVAL-v1 (automatic scoring, independent blind review instructions) and seed-selection prompts |
| `src/datafarm/render/policy.py` | `lower_h3_prompt`, `chunk_visibility_sentences`, `check_h3_payload` |
| `prompts/render/h3-caption-*.txt` | H3 caption-style prompt templates |

### Flow

```text
White carrier (one per episode, T layout on a 1024x768 canvas with gutters)
  │ h3_enqueue.py --seed-offset 0..7 --per-chunk  →  h3_package.py build
  │   · compile the contract: h3_bundle.py → tools/render_policy.py compile (mode and axes only from bundle.mode_confirmations)
  │   · independent chunks: 192 frames, advance 120, the first 72 frames are guidance
  │   · soft-HED control with a luma-step shadow gate; active repaint of the whole region
  │   · chunk prompt = lower_h3_prompt(contract, h3-caption-chunkvis-notask) + facts measured on the white clip
  │   · references: a real scene frame matching the carrier's state, a robot reference keeping the real arm appearance
  │   · check_h3_payload runs when building and again before the worker renders
  ▼
launch_control_worker.sh [memory_mode]  (one resident worker per GPU) → h3_control_worker.py
  ▼
h3_screen_prep.py composites → a VLM scores each chunk (evaluations/render-h3/prompts/seed-screen.txt) → choice.json
  (parts from several agents are merged with h3_screen_prep.py merge)
  ▼
h3_finalize.py select: h3_select.py --choice-file → h3_assemble.py (min-diff seams, min_lead 56) → EVAL-v1 scoring
  → h3_exposure_lock.py (gain toward the first chunk, pixel positions untouched) → final-t3-<fps>fps.mp4
  ▼
h3_score_and_pack.py --packages <pkg> (scores, writes <pkg>/pack) → h3_review_prep.py blind-review jobs
  → independent review per reviewer-instructions-v1.md (the only acceptance)
```

### Recipe rationale

| Practice | Reason |
|---|---|
| Use the ControlNet hybrid, not native Ref2VA | Native Ref2VA copies the reference image's pose. The hybrid uses the white model's edge structure as the control row, so the motion follows the simulation |
| Independent chunks with guidance frames, never chained | Guiding a chunk from the previous output accumulates artifacts, vignetting and colour cast chunk by chunk. Independent chunks are joined with a minimum-difference hard cut, no crossfade |
| First-chunk exposure gain lock (proportional gain through the origin) | Removes brightness jumps between chunks. A straight-line fit diverges in a static wrist view. Do not calibrate toward the real camera's colour, which shifts the whole picture |
| White model as a 39-frame video reference | Suppresses content the model would otherwise invent, such as extra grippers or objects under an empty gripper. All 192 frames would overflow GPU memory |
| Robot reference keeps the real arm appearance | With a white video reference present, if the real links cannot be seen in the real reference, the model copies the white arm |
| Scene reference matches the carrier's task state | A state difference between the reference and the carrier flickers between chunks |
| No isolated object reference images | An isolated object image makes the object reappear in the picture after it has been put away |
| Per-chunk visibility prompt, no whole-episode task sentence | A whole-episode description makes the model draw future states early. `-notask` keeps only "the action follows Video 1" |
| 8 seeds per chunk with per-chunk VLM screening | Pixel metrics cannot catch content errors such as cables, extra objects or a white arm. Screening is only selection; acceptance is the independent blind review |
| OOD: same seed across chunks, change material and light at the output | Mixing seeds makes the table colour jump. Changing appearance at the output does not touch content, while generation-side OOD can fall back to a white table in the wrist view |

### Using it for a new task

The bundle, reference images, episode list, OOD variants, recipe copy and seed-selection results are outputs of the task, derived and verified by the agent from the current real episode.

1. Confirm the mode. sim2real or OOD (and its axes) can come only from the current user's in-scope instruction. Write the exact words and source into the bundle's `mode_confirmations`. When missing, `h3_bundle.py` refuses to compile.
2. Create the run directory (`tasks/ recipes/ queue/ eval/realpool/`) and write `tasks/<task>/bundle.json`: the complete white carrier, the real references (a state-matched scene frame and a cable-free robot reference, no isolated object images), the five overlay slots and the real baseline. `h3_clean_references.py` can generate clean references from the real episode. Run `h3_bundle.py check <bundle> --run-dir <run>`.
3. Copy `configs/render/h3-control-hybrid-v1.json` to `<run>/recipes/<name>.json`, changing only `name`, `bundle_variant` and per-task switches backed by evidence (`prompt_lowering`, `chunk.event_aware`).
4. Enqueue 8 seeds and start workers on the hosts the user and deployment allow (`<run>/queue/ALLOWED_HOSTS`).
5. Select seeds, assemble, score, lock, pack and review. Production first runs 2 complete-episode prefixes and then a fixed set of 8 canaries, with a fixed denominator and all failures kept.

```bash
export H3_RUN=/data/runs/<run> ; H=<datafarm>/src/datafarm/render/h3
for s in 0 1 2 3 4 5 6 7; do
  python $H/h3_enqueue.py --recipe $H3_RUN/recipes/<name>.json --tasks <task> \
    --episodes <episode> --modes sim2real --seed-offset $s --priority 001 --per-chunk
done
bash $H/launch_control_worker.sh full     # submit once per GPU
B=pkgs/<name>/<task>/<episode>-sim2real
cd $H3_RUN && python $H/h3_screen_prep.py composites --cands $B $B-s1 $B-s2 $B-s3 $B-s4 $B-s5 $B-s6 $B-s7 --out screen/<task>/<ep>
# the VLM writes screen/<task>/<ep>/choice.json per the seed-screen prompt
python $H/h3_finalize.py select --render-recipe <name> --task <task> --episode-tag <episode>-sim2real --name <name>v-vlm \
  --choice-file screen/<task>/<ep>/choice.json --seeds s0 s1 s2 s3 s4 s5 s6 s7
P=$H3_RUN/pkgs/exposure-lock-<name>v-vlm/<task>/<episode>-sim2real    # holds final-t3-<fps>fps.mp4
python $H/h3_score_and_pack.py --packages $P
python $H/h3_review_prep.py --subjects $P
```

Single-package commands:

```bash
H3_RUN=<run> python src/datafarm/render/h3/h3_package.py build --bundle <run>/tasks/<task>/bundle.json \
  --recipe <run>/recipes/<name>.json --episode <ep> --mode sim2real --out <run>/pkgs/<name>/<task>/<ep>-sim2real
python src/datafarm/render/h3/h3_package.py check <package>
python src/datafarm/render/h3/h3_paths.py show
python evaluations/render-h3/h3eval.py score --generated <final.mp4> --carrier <carrier.mp4> --out <score.json>
```

`h3_requeue_orphans.py` returns the tickets of dead workers to the queue.

### OOD (tabletop material and light colour)

See `configs/render/h3-ood-restyle-v1.json`. The recommended route changes the appearance at the output, on a sim2real result that has passed blind review. `h3_sam2_table_masks.py` tracks the table per view on the generated video (re-seeding points every 96 frames). `h3_ood_restyle.py` subtracts the objects known from the white model, adds the projected shadows, and falls back to the table measured from the white model when tracking is lost. Table pixels become the target material times the generated brightness over the smoothed table brightness, preserving shadows, contacts and noise. The light colour only applies a mean-preserving colour-temperature gain. Prefer uniform materials (`--material-bgr`), because the wrist cameras move and a texture cannot follow.

The alternative renders the sim2real recipe with an OOD contract, using the same seed for all chunks. A textured material requires texturing the table in simulation and re-exporting the carrier. Any OOD needs axes approved by the current user.

### Dependencies

- **GPU:** each worker takes one GPU with free memory of at least 172 GB by default (`H3_MIN_FREE_MB`) for a 1024x768 canvas with 192-frame chunks fully resident.
- **Python** (`H3_PYTHON`): torch, diffusers, transformers, omegaconf, safetensors, opencv-python, numpy; system `ffmpeg` and `ffprobe`. Packaging, seed selection and the exposure lock need only numpy and opencv.
- **External assets** (`configs/external-assets.json`, doctor profile `h3-control`): the MiniMax-H3 Ref2VA directory, the H3-Fun ControlNet-Union weights, the MiniMax-H3 Turbo 4-step LoRA, the ControlNetHED annotator weights, the VideoX-Fun repository with `videox-fun-datafarm.patch` applied (see `vendor/videox_fun/README.md`), the ControlNet v1.1 annotator source, DINOv2-base (EVAL-v1) and SAM2.1-hiera-large (OOD only).
- **Bindings** (environment variable, then `DATAFARM_<KEY>`, then the bindings file; no built-in default, and a missing key or path fails): `H3_RUN` (or `--run-dir`), `H3_PYTHON` (default: the current interpreter), `H3_MODELS`, `H3_VIDEOX_FUN`, `H3_TURBO_LORA`, `GPU_SCHEDULER`, `H3_EVAL_DINO`, `H3_SAM2`, `H3_HED_ANNOTATOR_REPO`. `python3 src/datafarm/render/h3/h3_paths.py show` prints what resolves.
- **Scheduling:** `launch_control_worker.sh` calls `$GPU_SCHEDULER submit`, with hosts from `<run>/queue/ALLOWED_HOSTS`. The interface is in [deployment](../deployment.md#gpu-scheduling).

### Guarantees

1. **Every package passes the render-policy gate twice.** `build` hides the tickets until all control inputs exist, then runs `audit`. A failed audit puts the tickets back to `.partial`. The worker runs `audit` again before rendering and writes `policy-receipt.json`. `audit` calls `check_h3_payload`, and requires the manifest schema `h3.control-hybrid-manifest/1`, `controlnet: true` and `reference_order: images-first`. Every chunk must have a legal 17n+5 length (124–362 frames), and every control, inpaint and mask input must be a canvas-sized file whose sha256 matches.
2. **The mode comes only from the bundle's recorded confirmation.** `h3_bundle.py` builds the render intent from `bundle.mode_confirmations.<mode>`, and refuses to compile without one.
3. **Each chunk prompt is a deterministic lowering of the contract plus closed-schema facts.** For `h3-caption-chunkvis-*` lowerings, the builder measures `chunk_facts` from the white clip (`head_absent_arms`, `empty_wrists`, `head_max_coloured_objects`, `wrist_large_objects`), writes `lower_h3_prompt(contract, lowering, chunk_facts)` and records `chunk_facts_detector`. Free text cannot enter: the schema is closed, and the gate recomputes the prompt.
4. **A pale object is not reported as an "empty" wrist view.** `wrist-structure-v1` counts a wrist view as showing an object in a sampled frame when saturated pixels exceed 0.4 % of the view, or when `wrist_structure()` is at least 0.75. `wrist_structure()` is the Canny edge length left after removing the dark robot, up to 4 straight room seams and specks under 12 px, in view widths. A view is "empty" only if at most 10 % of sampled frames show an object.
5. **Control edges ignore simulator cast shadows.** The soft-HED map is multiplied by a mask of strong local luminance steps in the carrier (max−min of the blurred grey image over a `window` of 7 px > `tau` 25, dilated by 5 px). The result must have exactly `generated_frames` frames.
6. **The default recipe is pinned:** backend `control-hybrid`, canvas `t3-guttered-1024x768`, profile `control-turbo`, 4 denoise forwards; chunks of 192 frames, `advance` 120, `keep_head` 72; soft-HED with the luma-step gate, `white_video_reference` 39; lowering `h3-caption-chunkvis-notask`, `min-diff` seams with `min_lead` 56; 8 seeds (offsets 0–7); a first-chunk exposure lock with no real-camera calibration. Model paths are `${H3_*}` placeholders.
7. **Deployment paths fail closed.** A missing or nonexistent value raises `MissingBinding`.
8. **Capability evolution cannot change the evaluator.** `evaluations/` is in the evolution controller's protected list.

Code: `src/datafarm/render/h3/`, `src/datafarm/render/policy.py`.

### Data model

- **Bundle** `<run>/tasks/<task>/bundle.json`: carrier, real references, the five overlay slots, real baseline, `mode_confirmations`, and a flat list of episodes.
- **Recipe** `<run>/recipes/<name>.json` (`datafarm.h3-render-recipe/1`). Fields used by code: `backend`, `canvas`, `worker_profile`, `denoise_forwards`, `chunk {frames, advance, keep_head, clip_frames, clip_offset, event_aware?}`, `seed_base`, `control {...}`, `prompt_lowering`, `assembly {seam, min_lead}`, optional `chunk_facts_wrist`.
- **Package** (`h3.package/1`) with `job-manifest.json` (`h3.control-hybrid-manifest/1`). Each job has `chunk_id`, `chunk_index`, `seed`, `source_frame_range`, `generated_frames`, `white`, `control_video`, `inpaint_video`, `inpaint_mask` (with `_sha256`), `reference_images`, `prompt`, `chunk_facts` and `chunk_facts_detector`. The package also holds `ticket.json`. Worker output: `chunks/<chunk_id>/chunk-render.mp4`, `segment-manifest.json`, `policy-receipt.json`.
- **Seed choice** `screen/<task>/<episode>/choice.json` (`h3.seed-screen/1`): `chunks [{chunk, choice}]`, `screener`. Written by an agent, not by code.

### Configuration

| Key | Default | Effect |
|---|---|---|
| recipe `chunk_facts_wrist` | `wrist-structure-v1` | The only detector; unknown names exit |
| recipe `chunk.event_aware` | absent | `{before, min_advance}` moves chunk starts ahead of arm entry and exit events |
| recipe `seeds.count` / `seed_offsets` | 8 / 0–7 | Not read by code; the operator runs `h3_enqueue.py --seed-offset` once per offset. Seed = `seed_base` (1000) + offset |
| `h3_enqueue.py --episodes` | all | Bundle episode IDs |
| `build --hed-device` / `--hed-jobs` / `--hed-threads` | `cpu` / 4 / 16 | Where and how HED runs |
| `launch_control_worker.sh [memory_mode]` | `full` | Worker memory mode; `H3_IDLE_EXIT` and `H3_MAX_TICKETS` bound its lifetime |

### Failure behaviour

Any binding, audit or shape failure stops the build or worker before rendering, and tickets of a failed build stay `.partial`. The worker resumes per chunk: a chunk with both output and segment manifest is skipped.

### Limitations

- **No code-enforced canary gate.** The tools are standalone scripts; the canary and blind review are required by the [render-batch skill](../../.agents/skills/render-batch/SKILL.md).
- **No automatic re-render path.** An agent-run VLM writes `choice.json`; no code calls a VLM, and flagged chunks are not re-rendered. Without `--choice-file`, `h3_select.py` uses a pixel occupancy screen.
- **The default recipe is a starting point, not a qualification.** A new task needs its own complete-episode canary and independent review, and OOD is not automatic.
- **Artifacts to check in review.** Shadows can look hard or smudged, with slight colour cast, vignetting and garbled small digits.
- **Reference policy is advice, not code.** The text in `references_policy` is not enforced; code enforces only hashes and ordering.
- **Chunk facts are heuristics** (HSV thresholds, Canny, Hough). A miss produces a wrong prompt sentence. The gate checks that prompt and facts agree, not that the facts are right.
- **Each later chunk loses a lead-in.** The cut lies at least `min_lead` frames into the incoming chunk and at least 4 frames before the outgoing chunk ends, so with the default chunking each later chunk loses its first 56–67 frames.
- **Weights are external.** MiniMax-H3 and the turbo LoRA are external bindings (`NOTICE.md`).

### Licence

MiniMax-H3 LICENSE §V.3 prohibits using MiniMax-H3 Works or their Outputs to improve other AI models (except MiniMax-H3 and its derivatives), and it has Excluded Territories. Training a robot policy with H3 outputs is a use that requires a legal basis. This document gives no legal interpretation, and this repository grants no rights to that model or its outputs; confirm the rights and territorial restrictions before rendering training data with H3. VideoX-Fun and the ControlNet v1.1 HED annotator are Apache-2.0.

## Seedance

`src/datafarm/render/seedance/` executes edit requests, source upload with remote SHA verification, rate limiting, task-ID query and download, and single-shot create with a fixed source identity.

- **Route and profile.** Direct video-to-video appearance rewriting uses an explicit `edit` route. Ordinary and high-throughput endpoints are separate profiles; a high-throughput profile's concurrency affects scheduling only and inherits no quality qualification. A change of endpoint, model or quota needs a new profile. The 697-frame limit below belongs to Seedance and does not limit H3.
- **Clock.** Canonical frame `i` is simulator frame `i`. The same frames are sent at 24 FPS, and the 30 FPS training clock is restored with the same index afterwards. Frames added to fit the transport grid may only copy a terminal raster the simulator proved static.
- **Segmentation and references.** A long episode may be split into provider-valid phases that overlap in simulator-annotated non-contact, released regions; every canonical frame is kept exactly once. Splitting at simulator events is task code. A recipe may declare `VIDEO 1` (the synchronized white layout), a phase-matched style image, a phase-local structure image, and a previous-output anchor that only constrains continuity between adjacent phases. Multi-panel output uses a T layout: head 1280×720 on top, wrists 640×360 below.
- **Asynchronous tasks.** The seed is an experimental control, not a byte-exact replay; returned bytes are saved. A recent `submitting` record without a task ID stays unresolved, and after the stale window it is quarantined. Provider success, technical pass, semantic pass and dataset admission are counted separately.
- **Repair.** An ordinary repair redraws only the rejected identity or phase, then goes through the same seam and semantic gates. A panel may be repaired by copying the last frame before the cut only when the simulator's camera pose, robot state and per-frame images jointly prove it static over the whole tail. Using Seedance output to train another renderer needs explicit written authorization and a recorded rights basis.

```bash
python -m datafarm.render.seedance.batch --run <seedance-run> --phase canary --workers 8
```

### Guarantees

1. **Never two POSTs for one source.** `create_once` runs `check_seedance_payload`, persists `{status: submitting}`, then POSTs. An existing `task.json` is returned, not re-sent, and a changed contract raises. A record without a provider ID (`create_rejected`, `ambiguous_no_id`, `ambiguous_transport`) stops the job with "one-create limit".
2. **The real-media upload host is opt-in.** The carrier goes only to `DATAFARM_SEEDANCE_UPLOAD_HOST` or the batch config's `upload_host`. There is no default, and the host must be `https://`. The check runs before any rate slot or attempt is spent. Each upload is bounded to 90 s, re-downloaded, and must match the local sha256.
3. **Bulk submission requires a canary review bound to this batch.** `--phase canary` (the default) submits only `canary_ids`. `--phase all` requires `canary-review.json` with a truthy `continue_exploration` and `batch_sha256` equal to the sha256 of the current `batch.json`.
4. **Every source frame is kept, and nothing is retimed.** With `clock` from `batch.json`, a source row must satisfy `0 < frames <= transport_frames <= max_transport_frames` (697 in the supported profile), `transport_frames % 8 == 1`, and fewer than 8 padding frames. The provider output must have exactly `transport_frames` frames. `finalize` trims the padding and writes `render-24fps.mp4` and `render-30fps.mp4` with `setpts=N/(fps*TB)`, so output frame *i* is source frame *i*.

Code: `src/datafarm/render/seedance/`.

### Data model and configuration

A run holds `batch.json` (`model`, `jobs`, `canary_ids`, `upload_interval_seconds`, `create_interval_seconds`, `clock {transport_fps, canonical_fps, max_transport_frames}`, optional `upload_host`), `canary-review.json` (`continue_exploration`, `batch_sha256`), and per episode `input-contract.json`, `upload.json`, `task.json`, `raw.mp4`, `result.json` and `failure.json`. `status-<phase>.json` records the fixed denominator and failures.

| Key | Default | Effect |
|---|---|---|
| `ARK_API_KEY` (env) | required | Credential, sent only in the Authorization header |
| `SEEDANCE_API_PROXY` (env) | unset | Proxy for API and upload; passed to curl through stdin config, not argv |
| `batch --workers` | 8 (1–24) | Concurrent jobs |

`run_one` re-hashes the inputs, uploads the carrier (rate-limited), builds the payload (prompt, carrier `video_url`, inline images in the order checked by `references.py::images`), calls `create_once` and polls for up to 7200 s. `finalize` marks the result `technical_complete_pending_visual_review` and `training_admitted: false`.

### Failure behaviour

Worker exceptions are written to `episodes/<id>/failure.json`, and the batch continues. An episode with `failure.json` and no `result.json` is not retried automatically. A frame-count mismatch raises, and `raw.mp4` is kept. `poll_download` raises on a terminal provider failure and on a success with no video URL. Upload makes 6 bounded attempts with back-off, each recorded in `upload-error.json`.

### Limitations

- **No pause on recurring failure in the adapter.** `datafarm.render.seedance.batch` records failures and keeps going. The stop lives in the [batch supervisor](batch-supervisor.md), so run through it for that behaviour.
- **Upload precedes the policy check.** The carrier is uploaded before `check_seedance_payload` runs, and the uploaded carrier is public until it expires (24 h).

## Comparing the routes

| Dimension | H3 control hybrid | Seedance |
|---|---|---|
| Execution | Local GPU workers | Remote asynchronous edit API |
| Clock | 24 FPS, profile-level frame limit | 24 FPS transport, `8n+1`, 697 frames |
| Source authority | White carrier with structural control | White layout reference video |
| Reference risk | Pose, layout and entity projection contamination | Phase references can reset state or appearance |
| Continuity | Independent chunks with hard-cut seams | Overlapping phases with a non-contact seam |
| Resume key | Local job or chunk plus output digest | Remote task ID plus immutable attempt |
| Main failures | Extra entities, event lag, reference attraction, false style accept | Frame-count drift, phase reset, low-motion wrist seam, ambiguous create |
