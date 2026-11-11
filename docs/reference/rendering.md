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
Captured white attempt + the task's real episode
  │ datafarm.simulation.white_replay  (render-only: real robot colour, object shapes, scene dressing, dense image
  │   rate, wide wrist cameras; the state log must stay identical)  →  h3_carrier.py compose / check
  │ h3_episode_references.py build  (f0, robot-visible frame, robot / object / wrist sheets within the memory budget)
  ▼
White carrier (one per episode, T layout on a 1024x768 canvas with gutters)
  │ h3_enqueue.py --seed-offset 0..7 --per-chunk  →  h3_package.py build
  │   · compile the contract: h3_bundle.py → tools/render_policy.py compile (mode and axes only from bundle.mode_confirmations)
  │   · the last carrier frame held 48 frames past the end (chunk.tail_pad), cut back after assembly
  │   · independent chunks: 192 frames, advance 120, the first 72 frames are guidance
  │   · soft-HED control with a luma-step shadow gate; active repaint of the whole region
  │   · keep regions (optional): real pixels kept where the white shows a task state
  │   · chunk prompt = lower_h3_prompt(contract, h3-caption-chunkvis-notask) + facts measured on the white clip
  │   · references: a real scene frame matching the carrier's state, a robot reference keeping the real arm appearance
  │   · check_h3_payload runs when building and again before the worker renders
  ▼
h3_pool.py run (resident + elastic workers) or launch_control_worker.sh [memory_mode] per GPU → h3_control_worker.py
  ▼
h3_screen_prep.py composites → a VLM scores each chunk (evaluations/render-h3/prompts/seed-screen.txt) → choice.json
  (parts from several agents are merged with h3_screen_prep.py merge)
  ▼
h3_finalize.py select: h3_select.py --choice-file (the occupancy screen when the VLM gave no usable choice.json)
  → h3_assemble.py (min-diff seams, min_lead 56) → EVAL-v1 scoring
  → h3_exposure_lock.py (gain toward the first chunk, pixel positions untouched) → final-t3-<fps>fps.mp4
  → h3_wrist_crop.py when the episode has wide_wrist (wrist views cropped back to the white's own camera model)
  ▼
h3_score_and_pack.py --packages <pkg> (scores, writes <pkg>/pack) → h3_review_prep.py blind-review jobs
  → independent review per reviewer-instructions-v1.md (the only acceptance)
  ▼
h3_gate.py record/status: 2 prefixes → canary (≥ 75% accepted, no blocking kind in half of it) → the rest;
  h3_enqueue.py --gate refuses episodes the gate has not released
```

### Recipe rationale

| Practice | Reason |
|---|---|
| Use the ControlNet hybrid, not native Ref2VA | Native Ref2VA copies the reference image's pose. The hybrid uses the white model's edge structure as the control row, so the motion follows the simulation |
| Independent chunks with guidance frames, never chained | Guiding a chunk from the previous output accumulates artifacts, vignetting and colour cast chunk by chunk. Independent chunks are joined with a minimum-difference hard cut, no crossfade |
| First-chunk exposure gain lock (proportional gain through the origin) | Removes brightness jumps between chunks. A straight-line fit diverges in a static wrist view. Do not calibrate toward the real camera's colour, which shifts the whole picture |
| White model as a 39-frame video reference | Suppresses content the model would otherwise invent, such as extra grippers or objects under an empty gripper. All 192 frames would overflow GPU memory |
| Robot reference keeps the real arm appearance | With a white video reference present, if the real links cannot be seen in the real reference, the model copies the white arm |
| Robot links in the white tinted to the real robot colour | H3 copies the carrier: pale simulator links stay white or silver even with a real robot reference |
| Scene reference matches the carrier's task state | A state difference between the reference and the carrier flickers between chunks |
| No isolated object images for objects that leave the view | An isolated object image makes the object reappear in the picture after it has been put away |
| An object close-up sheet for objects that stay in view | Wrist cameras see objects closer than the head camera ever does; without a close-up they get proxy-looking objects |
| References budgeted by encoded area (≤ 25 MP) | H3 encodes every image at a 2048-pixel short edge, so a 1280x360 strip costs ~15 MP; 31.9 MP ran out of memory |
| Render-only object shapes and scene dressing in the white | H3 keeps the white's shapes and background: a cylinder proxy stays a plain cylinder, a dim or missing wall becomes a charcoal wall |
| Wide wrist white, cropped back (optional) | The wide view gives H3 the scene context around the gripper; cropping restores the white's own intrinsics. Cost: the original view is only 0.65 of the rendered panel and is scaled back up, so wrist detail drops (stack-cups canary: about half the Laplacian variance of the narrow-wrist render). Use it when a narrow wrist view makes H3 invent walls or table edges, not by default |
| A white below 20 fps re-rendered densely, never interpolated | Motion interpolation warps and drags fast objects; H3 copies the drift |
| Last carrier frame held 48 frames past the end | The end of the last chunk is where H3 invents motion or extra objects; the padding is cut off again |
| Real pixels kept for a task state the model does not follow | A small state change (a display lighting up) is rendered at the wrong time; a keep region copies it from real pixels |
| Per-chunk visibility prompt, no whole-episode task sentence | A whole-episode description makes the model draw future states early. `-notask` keeps only "the action follows Video 1" |
| 8 seeds per chunk with per-chunk VLM screening | Pixel metrics cannot catch content errors such as cables, extra objects or a white arm. Screening is only selection; acceptance is the independent blind review |
| OOD: same seed across chunks, change material and light at the output | Mixing seeds makes the table colour jump. Changing appearance at the output does not touch content, while generation-side OOD can fall back to a white table in the wrist view |

### Preparing the white carrier and references

The renderer copies what the carrier shows, so most visible render failures are fixed in the white, not in the prompt. Evidence for every step below is independent blind review of the 2026-11 production renders (dual-arm Piper drum, weigh-apple and stack-cups; RealBot-01 scoop-cereal), where each step turned a rejection into an acceptance or removed a reviewer-named defect.

1. **Real references.** Write `tasks/<task>/episode-references.json` and run `h3_episode_references.py build`. It derives the start frame, the frame where the real robot is most visible, the robot sheet, the object close-up sheet (with a second tile for an object whose appearance changes) and the stacked wrist sheet from the task's own real episode, within the 25 MP budget. The robot pixels come from a dark-robot heuristic; for a light-coloured robot or dark furniture, first make projected label masks with `h3_robot_masks.py` from the task's validated kinematics and set `robot_masks`. For white-on-white or split objects, give `object_boxes`. Leave `object_sheet` off when an object leaves the view during the task.
2. **Replay spec.** `h3_episode_references.py robot-colors` writes `@robot` (and `@robot_links` with masks) for the white replay. Add render-only object shapes where the simulator proxy differs from the real object, render-only `extras` where the white lacks the real wall, counter or props (a bright real wall the simulator lights dimly needs `emission`), `image_stride` for a white captured below 20 fps, and `wrist_fov_factor` 0.65 for wide wrist views (only when the narrow wrist view makes H3 invent room edges; it softens wrist detail). Validate with `python -m datafarm.simulation.white_replay --check-spec`.
3. **Replay.** `python -m datafarm.simulation.white_replay <attempt> <out> --spec <spec>` on a host that can render. It refuses success when the state log differs from the attempt's or a requested change was not applied. Watch for coplanar visual parts (z-fighting) and seams between dressing parts, which H3 draws as sticks.
4. **Carrier.** `h3_carrier.py compose --replay <out> --attempt <attempt> --out <carrier>`, then `h3_carrier.py check <carrier>`. Record `wide_wrist` (factor and the wrist principal points from the replay report) on the bundle episode.
5. **Keep regions** (only when a task state is a small detail the model renders at the wrong time): write a `h3.keep-regions/1` spec, put its `states` into the replay colours as `@states`, and name it as the episode's `keep_regions`.

The replay is a render input only. It never replaces the attempt, its judge verdict or its admission, and it may not change actions, timing or physics.

### Using it for a new task

The bundle, reference images, episode list, OOD variants, recipe copy and seed-selection results are outputs of the task, derived and verified by the agent from the current real episode.

1. Confirm the mode. sim2real or OOD (and its axes) can come only from the current user's in-scope instruction. Write the exact words and source into the bundle's `mode_confirmations`. When missing, `h3_bundle.py` refuses to compile.
2. Create the run directory (`tasks/ recipes/ queue/ eval/realpool/`) and write `tasks/<task>/bundle.json`: the complete white carrier and the real references prepared as above, the five overlay slots and the real baseline. `h3_clean_references.py` can instead generate an object-free scene plate and cable-free sheets. Run `h3_bundle.py check <bundle> --run-dir <run>`; it fails on a carrier below 20 fps and warns when the head view has no wall or table seam the control can see. Write the text fields once per task from the real references and the carrier itself (object identity, colour, size and count; phases, events and terminal relations); every episode of the task then reuses that bundle text. A generic template text loses object identity: with "the objects follow the source video" the drum was drawn as a plain cylinder.
3. Copy `configs/render/h3-control-hybrid-v1.json` to `<run>/recipes/<name>.json`, changing only `name`, `bundle_variant` and per-task switches backed by evidence (`prompt_lowering`, `chunk.event_aware`).
4. Enqueue 8 seeds in one call and start workers on the hosts the user and deployment allow (`<run>/queue/ALLOWED_HOSTS`). The first seed's packages are built first and fill the HED cache the other seeds reuse; builds run at nice 10 and share `--build-slots` (default 2) run-wide slots, because each build runs 4 HED processes of 16 threads. For more than one episode, keep workers with `h3_pool.py`: `resident` workers stay loaded (a resident worker's model load took about 1000 s in the 2026-11 canary), and elastic workers are added up to `max_workers` while tickets are pending.
5. Select seeds, assemble, score, lock, pack and review. Production first runs 2 complete-episode prefixes and then a fixed set of 8 canaries, with a fixed denominator and all failures kept. `h3_gate.py init` fixes the three lists before the first render; enqueue with `--gate`, and `record` every final package (or a render failure) as it finishes. The gate reads only the blind-review verdicts. When the seed-screen session fails or leaves a partial `choice.json`, `h3_finalize.py select` falls back to the occupancy screen and records why; the episode still goes to blind review.

```bash
export H3_RUN=/data/runs/<run> ; H=<datafarm>/src/datafarm/render/h3
python $H/h3_gate.py init --gate <name>-g1 --recipe <name> --task <task> --prefix <e1> <e2> \
  --canary <e3> <e4> <e5> <e6> <e7> <e8> <e9> <e10> --rest <e11> ...
python $H/h3_enqueue.py --recipe $H3_RUN/recipes/<name>.json --tasks <task> --gate <name>-g1 \
  --episodes <episode> --modes sim2real --seed-offset 0 1 2 3 4 5 6 7 --priority 001 --per-chunk
bash $H/launch_control_worker.sh full     # submit once per GPU, or keep a pool:
echo '{"schema": "h3.worker-pool/1", "resident": 1, "max_workers": 4}' > $H3_RUN/queue/pool.json
setsid nohup python $H/h3_pool.py run > $H3_RUN/queue/pool.out 2>&1 &   # touch $H3_RUN/queue/STOP to stop all
B=pkgs/<name>/<task>/<episode>-sim2real
cd $H3_RUN && python $H/h3_screen_prep.py composites --cands $B $B-s1 $B-s2 $B-s3 $B-s4 $B-s5 $B-s6 $B-s7 --out screen/<task>/<ep>
# the VLM writes screen/<task>/<ep>/choice.json per the seed-screen prompt
python $H/h3_finalize.py select --render-recipe <name> --task <task> --episode-tag <episode>-sim2real --name <name>v-vlm \
  --choice-file screen/<task>/<ep>/choice.json --seeds s0 s1 s2 s3 s4 s5 s6 s7
P=$H3_RUN/pkgs/exposure-lock-<name>v-vlm/<task>/<episode>-sim2real    # holds final-t3-<fps>fps.mp4
python $H/h3_score_and_pack.py --packages $P
python $H/h3_review_prep.py --subjects $P
# the independent reviewer writes reviews/<rid>/review/review.json
python $H/h3_gate.py record --gate <name>-g1 --episode <episode> --package $P   # or --failed "<reason>"
python $H/h3_gate.py status --gate <name>-g1      # stage, verdicts, the episodes released for rendering
```

Single-package commands:

```bash
H3_RUN=<run> python src/datafarm/render/h3/h3_package.py build --bundle <run>/tasks/<task>/bundle.json \
  --recipe <run>/recipes/<name>.json --episode <ep> --mode sim2real --out <run>/pkgs/<name>/<task>/<ep>-sim2real
python src/datafarm/render/h3/h3_package.py check <package>
python src/datafarm/render/h3/h3_paths.py show
python evaluations/render-h3/h3eval.py score --generated <final.mp4> --carrier <carrier.mp4> --out <score.json>
```

`h3_requeue_orphans.py` returns the tickets of dead scheduler workers to the queue; in local mode `h3_pool.py` does it for the workers it started. `h3_pool.py status` prints the pool state (pending, running, alive workers, last error).

### OOD (tabletop material and light colour)

See `configs/render/h3-ood-restyle-v1.json`. The recommended route changes the appearance at the output, on a sim2real result that has passed blind review. `h3_sam2_table_masks.py` tracks the table per view on the generated video (re-seeding points every 96 frames). `h3_ood_restyle.py` subtracts the objects known from the white model, adds the projected shadows, and falls back to the table measured from the white model when tracking is lost. Table pixels become the target material times the generated brightness over the smoothed table brightness, preserving shadows, contacts and noise. The light colour only applies a mean-preserving colour-temperature gain. Prefer uniform materials (`--material-bgr`), because the wrist cameras move and a texture cannot follow.

The alternative renders the sim2real recipe with an OOD contract, using the same seed for all chunks. A textured material requires texturing the table in simulation and re-exporting the carrier. Any OOD needs axes approved by the current user.

### Dependencies

- **GPU:** each worker takes one GPU with free memory of at least 172 GB by default (`H3_MIN_FREE_MB`) for a 1024x768 canvas with 192-frame chunks fully resident.
- **Python** (`H3_PYTHON`): torch, diffusers, transformers, omegaconf, safetensors, opencv-python, numpy; system `ffmpeg` and `ffprobe`. Packaging, seed selection and the exposure lock need only numpy and opencv.
- **External assets** (`configs/external-assets.json`, doctor profile `h3-control`): the MiniMax-H3 Ref2VA directory, the H3-Fun ControlNet-Union weights, the MiniMax-H3 Turbo 4-step LoRA, the ControlNetHED annotator weights, the VideoX-Fun repository with `videox-fun-datafarm.patch` applied (see `vendor/videox_fun/README.md`), the ControlNet v1.1 annotator source, DINOv2-base (EVAL-v1) and SAM2.1-hiera-large (OOD only).
- **Bindings** (environment variable, then `DATAFARM_<KEY>`, then the bindings file; no built-in default, and a missing key or path fails): `H3_RUN` (or `--run-dir`), `H3_PYTHON` (default: the current interpreter), `H3_MODELS`, `H3_VIDEOX_FUN`, `H3_TURBO_LORA`, `GPU_SCHEDULER`, `H3_EVAL_DINO`, `H3_SAM2`, `H3_HED_ANNOTATOR_REPO`. `python3 src/datafarm/render/h3/h3_paths.py show` prints what resolves.
- **Scheduling:** `launch_control_worker.sh` calls `$GPU_SCHEDULER submit`, with hosts from `<run>/queue/ALLOWED_HOSTS`. The interface is in [deployment](../deployment.md#gpu-scheduling). Local mode (`H3_LOCAL_GPU=<index>`, or `local_gpus` + `local_host` in `pool.json`) runs the same worker on a GPU of a dedicated node without the scheduler; nothing then stops another user's job from taking that GPU, so use it only on a node the run owns.

### Guarantees

1. **Every package passes the render-policy gate twice.** `build` hides the tickets until all control inputs exist, then runs `audit`. A failed audit puts the tickets back to `.partial`. The worker runs `audit` again before rendering and writes `policy-receipt.json`. `audit` calls `check_h3_payload`, and requires the manifest schema `h3.control-hybrid-manifest/1`, `controlnet: true` and `reference_order: images-first`. Every chunk must have a legal 17n+5 length (124–362 frames), and every control, inpaint and mask input must be a canvas-sized file whose sha256 matches.
2. **The mode comes only from the bundle's recorded confirmation.** `h3_bundle.py` builds the render intent from `bundle.mode_confirmations.<mode>`, and refuses to compile without one.
3. **Each chunk prompt is a deterministic lowering of the contract plus closed-schema facts.** For `h3-caption-chunkvis-*` lowerings, the builder measures `chunk_facts` from the white clip (`head_absent_arms`, `empty_wrists`, `head_max_coloured_objects`, `wrist_large_objects`), writes `lower_h3_prompt(contract, lowering, chunk_facts)` and records `chunk_facts_detector`. Free text cannot enter: the schema is closed, and the gate recomputes the prompt.
4. **A pale object is not reported as an "empty" wrist view.** `wrist-structure-v1` counts a wrist view as showing an object in a sampled frame when saturated pixels exceed 0.4 % of the view, or when `wrist_structure()` is at least 0.75. `wrist_structure()` is the Canny edge length left after removing the dark robot, up to 4 straight room seams and specks under 12 px, in view widths. A view is "empty" only if at most 10 % of sampled frames show an object.
5. **Control edges ignore simulator cast shadows.** The soft-HED map is multiplied by a mask of strong local luminance steps in the carrier (max−min of the blurred grey image over a `window` of 7 px > `tau` 25, dilated by 5 px). The result must have exactly `generated_frames` frames.
6. **The default recipe is pinned:** backend `control-hybrid`, canvas `t3-guttered-1024x768`, profile `control-turbo`, 4 denoise forwards; chunks of 192 frames, `advance` 120, `keep_head` 72, `tail_pad` 48; soft-HED with the luma-step gate, `white_video_reference` 39; lowering `h3-caption-chunkvis-notask`, `min-diff` seams with `min_lead` 56; 8 seeds (offsets 0–7); a first-chunk exposure lock with no real-camera calibration. Model paths are `${H3_*}` placeholders.
7. **Deployment paths fail closed.** A missing or nonexistent value raises `MissingBinding`.
8. **Capability evolution cannot change the evaluator.** `evaluations/` is in the evolution controller's protected list.
9. **Padding never reaches the output.** With `tail_pad`, the canvas carrier and chunk plan cover the episode plus the held frames; `h3_assemble.py` and `h3_exposure_lock.py` cut the output back to `frames`, so output frame *i* is carrier frame *i* and the counts match the carrier clip.
10. **A white replay is accepted only with an identical trajectory.** `white_replay` compares the replay's state log with the attempt's (every field except the PhysX contact report, or every array of `log.npz`) and exits 2 when it differs or when a requested camera, colour, stride, shape or dressing change was not applied.
11. **Keep regions patch before the audit.** `h3_package.py build` patches the chunk inpaint video and repaint mask when the episode names `keep_regions`, records the frames kept per region, and stops when the regions appear in no white frame; the audit then hashes the patched inputs.

Code: `src/datafarm/render/h3/`, `src/datafarm/render/policy.py`.

### Data model

- **Bundle** `<run>/tasks/<task>/bundle.json`: carrier, real references, the five overlay slots, real baseline, `mode_confirmations`, and a flat list of episodes.
- **Recipe** `<run>/recipes/<name>.json` (`datafarm.h3-render-recipe/1`). Fields used by code: `backend`, `canvas`, `worker_profile`, `denoise_forwards`, `chunk {frames, advance, keep_head, clip_frames, clip_offset, event_aware?}`, `seed_base`, `control {...}`, `prompt_lowering`, `assembly {seam, min_lead}`, optional `chunk_facts_wrist`.
- **Package** (`h3.package/1`) with `job-manifest.json` (`h3.control-hybrid-manifest/1`). Each job has `chunk_id`, `chunk_index`, `seed`, `source_frame_range`, `generated_frames`, `white`, `control_video`, `inpaint_video`, `inpaint_mask` (with `_sha256`), `reference_images`, `prompt`, `chunk_facts` and `chunk_facts_detector`. The package also holds `ticket.json`. Worker output: `chunks/<chunk_id>/chunk-render.mp4`, `segment-manifest.json`, `policy-receipt.json`.
- **Seed choice** `screen/<task>/<episode>/choice.json` (`h3.seed-screen/1`): `chunks [{chunk, choice, scores}]`, `screener`. Written by an agent, not by code. The selected package's `selection` records `mode` (`screen` or `occupancy`), the choice, and `fallback_reason` when the occupancy screen stood in.
- **Gate** `gates/<name>.json` (`h3.render-gate/1`): `recipe`, `task`, `prefix`, `canary`, `rest` (fixed at `init`) and `records {episode: {package} | {failed}}`. Verdicts are read from `reviews/` at `status`, never stored by hand.

### Configuration

| Key | Default | Effect |
|---|---|---|
| recipe `chunk_facts_wrist` | `wrist-structure-v1` | The only detector; unknown names exit |
| recipe `chunk.event_aware` | absent | `{before, min_advance}` moves chunk starts ahead of arm entry and exit events |
| recipe `chunk.tail_pad` | 48 | Frames the last carrier frame is held past the end before chunking; cut back after assembly |
| bundle episode `wide_wrist` | absent | `{factor, principal_points}` of a wide-FOV wrist white; the final is cropped back |
| bundle episode `keep_regions` | absent | `h3.keep-regions/1` spec path; real pixels kept where the white shows the state |
| `h3_episode_references.py` spec `budget_mpix` | 25 | Encoded reference area per ticket |
| recipe `seeds.count` / `seed_offsets` | 8 / 0–7 | Not read by code; the operator passes the offsets to `h3_enqueue.py --seed-offset 0 1 … 7`. Seed = `seed_base` (1000) + offset |
| `h3_enqueue.py --build-slots` | env `H3_BUILD_SLOTS` or 2 | Concurrent package builds across the whole run (`<run>/cache/build-slots`) |
| `<run>/queue/pool.json` | absent | `h3_pool.py`: `resident`, `max_workers`, `resident_idle_min` (720), `elastic_idle_min` (20), `min_free_mb` (172000), `profile`, optional `local_gpus` + `local_host`; re-read every cycle |
| `h3_enqueue.py --episodes` | all | Bundle episode IDs |
| `build --hed-device` / `--hed-jobs` / `--hed-threads` | `cpu` / 4 / 16 | Where and how HED runs |
| `launch_control_worker.sh [memory_mode]` | `full` | Worker memory mode; `H3_IDLE_EXIT` and `H3_MAX_TICKETS` bound its lifetime |

### Failure behaviour

Any binding, audit or shape failure stops the build or worker before rendering, and tickets of a failed build stay `.partial`. The worker resumes per chunk: a chunk with both output and segment manifest is skipped.

### Limitations

- **The canary gate binds only through `--gate`.** `h3_enqueue.py` without `--gate` renders anything; the [render-batch skill](../../.agents/skills/render-batch/SKILL.md) requires the gate for production. The 75% rule and the common-kind rule count blind-review verdicts only, so the gate is as good as the reviews, and it does not judge whether the canary episodes cover the layouts.
- **No automatic re-render path.** An agent-run VLM writes `choice.json`; no code calls a VLM, and flagged chunks are not re-rendered. Without a usable choice file the pixel occupancy screen picks the seed. It misses content errors a VLM sees (cables, a white arm) and assumes a dark robot in the carrier.
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
