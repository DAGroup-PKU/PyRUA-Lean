# Render contract

Every render request is bound to a per-task *render contract*. The agent writes a render
intent (mode, authorized appearance changes, real reference images, appearance requirements)
and a five-slot task overlay. `render_policy` validates both and expands them through fixed
prompt templates the overlay cannot replace. The result is one compiled contract file. Before
a provider request leaves the process, the actual payload is checked against that file:
the prompt text, the ordered real-image bytes, and the presence of the source video. The
module makes no network calls. It proves *bindings*. It does not prove that an image is
really a photograph or that the model followed the prompt.

## Guarantees

1. **A render cannot be compiled with an unresolved mode.** The intent must use schema
   `datafarm.render-intent/1`, `mode` must be `sim2real` or `ood`, and `run_id`, `task_id` and
   `real_baseline` must be nonempty strings.
2. **The mode confirmation is scoped to this run, task and mode.** `mode_confirmation` must
   repeat the same `run_id`, `task_id` and `mode`, and carry a nonempty `quote` and `source`.
   A confirmation copied from another run fails.
3. **Mode and OOD axes agree, and the axes are the confirmed ones.** `ood` needs at least one
   entry in `authorized_ood_axes`; `sim2real` needs none. The list must equal
   `mode_confirmation.authorized_ood_axes` exactly. Each axis needs a nonempty name (a string,
   or a dict with `axis`).
4. **An OOD axis may change appearance only.** The JSON text of every axis (name and any other
   fields) is matched against `PROTECTED_OOD_AXIS`. A match on the robot (robot, arm, gripper,
   end-effector), cameras (camera, viewpoint, camera/view layout), objects (object count,
   number of objects, added/removed/extra objects, distractors) or motion (motion, trajectory,
   timing, action, pose) rejects the intent, as do the Chinese terms in the same pattern.
   This is a lexical, fail-closed guard on axis names, not a semantic check.
5. **Every real-scene area is classified, and three are always required.**
   `appearance_requirements` must have exactly the keys of `REFERENCE_AREAS` (`robot_body`,
   `attached_cameras_and_cables`, `task_objects`, `table_and_background`,
   `photographic_character`), each with a boolean `required` and a nonempty `description`.
   `robot_body`, `table_and_background` and `photographic_character` must be `required: true`.
   `appearance_views` must be distinct nonempty names. `appearance_invariants` must be
   nonempty.
6. **Every real reference is a decodable, hash-bound image with declared provenance.** Each
   entry must match its `sha256` (`bound_file`, absolute paths allowed), decode as PNG, JPEG
   or WEBP, and declare `provenance.kind` `real-episode-frame` or `real-photo`. The kind
   `source-frame` and any other structural kind are refused. `provenance.source` is itself
   hash-bound, and an episode frame needs a non-negative integer `frame`. Each entry also
   declares `scope` (`full-view` or `local-detail`), `views` (a subset of
   `appearance_views`), an in-bounds integer `[x, y, w, h]` region per claimed view in
   `view_regions`, and `coverage` (a subset of `REFERENCE_AREAS` plus `full_scene`).
7. **Every declared view has a full-scene real reference, and the references cover every
   required area.** For each view in `appearance_views`, at least one reference has
   `scope: full-view`, names that view, and lists `full_scene`. The union of `coverage` must
   include every area marked `required`. A single local wrist texture cannot satisfy this.
8. **The task overlay fills exactly five slots and cannot reach the mode or reference policy.**
   The overlay keys must equal `VARIABLES` (`TASK_DESCRIPTION`, `SOURCE_VIDEO_ROLE`,
   `CAMERA_TIMING_CONTRACT`, `TASK_EVENT_CONTRACT`, `CONTINUITY_CONTRACT`), all nonempty.
   `MODE`, the reference map, views, baseline, axes and invariants come only from the
   intent. A lexical backstop (`CONFLICT`) rejects slot text that tries to switch off the
   real references (for example "ignore real references", "source-only").
9. **The prompt text comes from the repository templates.** The compiled prompt is
   `prompts/render/common-v1.txt` followed by `prompts/render/<mode>-v1.txt`, expanded with
   `string.Template.substitute`. Unexpanded `${` or `{{` is refused. The path and sha256 of
   each template are recorded.
10. **A compiled contract is only usable while it still recompiles to itself.** Loading
    re-validates the intent (so every reference is re-hashed) and checks `intent_sha256` and
    `prompt_sha256`. It then recompiles with the *current* repository templates and requires
    byte-equal output, so an edited template, reference or intent invalidates it. If
    `DATAFARM_RENDER_MODE` is set, it must equal the confirmed mode.
11. **A Seedance request is checked before it is persisted or sent.** The payload must contain
    exactly one text item, equal to the compiled prompt. Every `image_url` must be an inline
    `data:image/...;base64,` URL; opaque URLs are refused because their bytes cannot be
    verified. The decoded image hashes must start with the ordered real-reference hashes
    (missing, reordered or replaced images fail), and a nonempty `video_url` must be present.
    `create_once` runs this check before it writes `task.json` and before the POST. With no
    contract path, it raises before any call.
12. **A resumed Seedance task reuses its provider ID only under the same verified policy.**
    The stored `render_policy` receipt must match the current contract file hash, intent and
    prompt hashes, and real-image prefix. Otherwise the run fails rather than re-posting.
13. **An H3 manifest is checked globally and per job.** The `reference_images` files must
    hash to the ordered real references (as a prefix). The manifest prompt must equal the
    compiled prompt, or, when `prompt_lowering` is set, `lower_h3_prompt(contract, lowering)`.
    Every job must use the same reference list. A chunk-visibility job's prompt is recomputed
    from its closed-schema `chunk_facts`; any other job must reuse the manifest prompt.

Code: `src/datafarm/render/policy.py`, `src/datafarm/render/seedance/client.py`, `src/datafarm/render/seedance/batch.py`, `tools/render_policy.py`.

## Data model

- **Render intent** (`datafarm.render-intent/1`): `run_id`, `task_id`, `mode`,
  `mode_confirmation {run_id, task_id, mode, quote, source, authorized_ood_axes}`,
  `real_baseline`, `authorized_ood_axes`, `appearance_views`, `appearance_requirements`
  (five areas, `{required, description}`), `appearance_invariants`, `real_references`.
  The template `configs/render/render-intent.template.json` is intentionally
  uncompilable (`mode: unresolved`, nulls).
- **Real reference**: `path`, `sha256`, `provenance {kind, source {path, sha256},
  description, frame?}`, `scope`, `views`, `view_regions {view: [x, y, w, h]}`, `coverage`.
  Relative paths resolve against the intent's directory.
- **Overlay**: a JSON object with exactly the five `VARIABLES` keys.
- **Compiled contract** (`datafarm.render-contract/1`): `intent`, `intent_root` (absolute),
  `overlay`, `prompt`, `prompt_sha256`, `templates [{path, sha256}]`, `intent_sha256`
  (`json_sha` of the intent with sorted keys).
- **Submission receipt** (`datafarm.render-submission/1`, returned by the two checks):
  `contract_sha256`, `intent_sha256`, `prompt_sha256`, `actual_image_sha256s`,
  `payload_sha256`, `provider` (`seedance` or `h3`), plus `model` (Seedance) or
  `prompt_lowering` and `submitted_prompt_sha256` (H3). Seedance stores it in `task.json`
  under `contract.render_policy`. The H3 worker writes it to `policy-receipt.json`.

## Reference order in a request

A Seedance job holds the authoritative, ordered list of real scene images in `real_references`, and repeats `role=real_scene_reference` for each image in `references`, with `path` and `sha256` matching item by item in the same order. Only `source_frame_zero`, `continuity_same_state` or both may follow. The payload builder places all real images before the auxiliary images, and the policy check compares the actual payload prefix with the ordered hashes, so a missing, reordered or substituted image fails before the POST. Multiple jobs use their own contract paths and must not change the process environment concurrently. Request evidence stores hashes, not signed URLs or credentials.

A compiled contract does not prove that an image comes from the real robot: the source videos and photos and the extraction mapping still need independent review. Querying an existing task ID makes no new request and does not admit that record.

## How it works

1. The agent derives the intent and overlay from the current episode and the user's
   scoped instruction (see [`prompts/instructions/mode-and-style-contract.md`](../../prompts/instructions/mode-and-style-contract.md)).
2. `tools/render_policy.py compile` validates the intent, checks the overlay, expands the
   templates and writes the contract. The file is opened in exclusive mode, so an existing
   contract is never overwritten.
3. Seedance: `batch.run_one` builds the payload. `create_once` calls `check_seedance_payload`
   with the job's `render_contract` (or `DATAFARM_RENDER_CONTRACT`), then persists the record,
   then POSTs.
4. H3: `h3_package.py build` writes the manifest and per-chunk prompts, and its
   `audit` calls `check_h3_payload`. The worker repeats `audit` before rendering
   (see [rendering](rendering.md)).
5. Delivery verification reloads the contract and compares it with the stored submission
   receipt (see [delivery and admission](delivery-and-admission.md)).

## Entry points

```bash
python tools/render_policy.py compile --intent <task>/render-intent.json \
  --overlay <task>/render-overlay.json --output <task>/render-contract.json
python tools/render_policy.py check-seedance --contract <task>/render-contract.json --payload <body.json>
python tools/render_policy.py check-h3 --contract <task>/render-contract.json --payload <job-manifest.json>
```

Each command prints `{"schema": ..., "valid": true}` and exits 0, or prints
`render-policy: <reason>` to stderr and exits 2.

## Configuration

| Key | Default | Effect |
|---|---|---|
| `DATAFARM_RENDER_CONTRACT` (env) | unset | Contract path used by `create_once` when the job contract has no `render_contract`. Unset and absent raises `ContractError`. |
| `DATAFARM_RENDER_MODE` (env) | unset | If set, must equal the contract's mode, or `load_contract` fails. It cannot select a mode. |
| `prompts/render/common-v1.txt`, `sim2real-v1.txt`, `ood-v1.txt` | repository files | Core prompt body. Editing one invalidates every contract compiled from the earlier text. |
| `prompts/render/h3-caption-*.txt` | repository files | H3 lowerings (`H3_LOWERINGS`). |

## Failure behaviour

- All rule violations raise `task_contract.ContractError` (a `ValueError`) with a message
  naming the rule. Missing files raise `FileNotFoundError` from `Path.resolve(strict=True)`.
- `create_once` raises before writing `task.json`, so a refused payload leaves no attempt
  record and makes no HTTP call.
- A contract whose templates, references or intent changed after compilation fails at
  every later load: at resume, at delivery verification and in the H3 worker.

## Limitations

- **Lexical guards only.** `PROTECTED_OOD_AXIS` and `CONFLICT` are word lists. A paraphrase
  that avoids them ("manipulator colour", "a cluttered tabletop") passes compilation. Some
  harmless wording ("camera exposure", "lighting timing") is refused. The render reviewer
  must still check rendered content against the axes.
- **Inconsistent axis vocabulary across layers.** `PROTECTED_OOD_AXIS` refuses an axis that
  mentions `distractors`. The render verifier spec accepts `distractors` as a perturbed OOD
  axis (`src/datafarm/verify/spec.py::OOD_AXES`, and `src/datafarm/verify/realism.py::BACKGROUND_AXES`).
- **Provenance is declared, not proven.** The code checks hashes, decodability and region
  bounds. It cannot tell a real photo from a rendered image that is labelled `real-photo`.
- **The Seedance source video is only checked for presence.** `check_seedance_payload` does
  not hash the `video_url` target. The carrier bytes are bound separately: the upload is
  re-downloaded and hashed (`src/datafarm/render/seedance/upload_transport.py::exchange`), and the recipe pins
  `source_sha256`.
- **In `batch.run_one`, the carrier upload happens before the policy check.** A job with a
  broken contract has already published its carrier to the opt-in upload host when
  `create_once` refuses it.
- **Images after the real-reference prefix are not checked here.** Structure and continuity
  images are validated by `src/datafarm/render/seedance/references.py::images`, not by
  `render_policy`.
- **H3 chunk facts are trusted as written.** `check_h3_payload` recomputes each chunk prompt
  from the manifest's `chunk_facts`. It does not re-measure those facts from the carrier and
  does not check `chunk_facts_detector`.
- **Templates are protected by hashing, not access control.** `prompts/render/` is not in the
  evolution controller's protected list (`src/datafarm/evolution/controller.py::PROTECTED`).
  An edit is detected because earlier contracts stop loading; nothing prevents a new contract
  from compiling against edited templates.
- **The contract is tied to its location.** It stores an absolute `intent_root`, so moving a
  run directory invalidates it.
- **No quality claim.** A valid contract and payload say nothing about whether the
  generated video follows the prompt. That is the job of verification and review.
