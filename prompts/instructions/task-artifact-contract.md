# New-task outputs and handoff contract

Generate these files rather than asking the user for them: grounding, task/requirements, scene and randomized initialization, expert, independent judge, expert-harness adapter and brief, render overlay, batch and data
encoding configs. Directory names do not restrict the task type. An implementation may span several files; `task/manifest.json` registers them with path, sha256 and a role that states their duty. Iterate on working copies, then
freeze the version that passes validation.

`tools/pipeline_agent.py` handles input locking, instruction injection, session boundaries, and state and delivery integrity checks; it does not treat generated files as physical success. Generated code actually executes through the
expert harness, simulator, render provider, run ledger, batch supervisor and admission. For API details read each tool's `--help`.

## Expert adapter

A new adapter uses `datafarm.expert-harness.adapter/1` (`src/datafarm/expert/adapter.py`). Derive it from the interface and schema, not from a completed task. You may consult the fake adapter that `tools/expert_harness.py dry-run`
generates (for protocol demonstration only) and `configs/expert-harness/template.json`, but rebuild the current scene, command, judge, paths, seeds and budget. Validate with `lint-brief`, `prepare-round` and the preflight of `run-round`.
Real rollouts use the available simulator command; the fake is for software tests only. When an unknown device lacks a plugin, write the plugin.

## delivery.json (the boundary of data completion)

Schema `datafarm.agent-delivery/1`. File references are workspace-relative paths plus sha256. Large external files are recorded in the dataset manifest with absolute locations and checksums; the validator reads them without copying.

- `input_sha256`: the input-lock digest from prepare.
- `producer_id`, `target_count`.
- `attempts`: file reference binding all attempt JSONs. A non-empty array of items with `id` and `outcome` (`admitted`, `rejected` or `infrastructure_failure`). The admitted IDs must equal the delivery episodes. Never remove failures.
- `dataset`: file reference to the data manifest, schema `datafarm.dataset-files/1`: `episode_ids` and `files` (workspace-relative paths or absolute external paths). Files must be non-empty and hashes must match real files.
- `grounding`: file reference to the `datafarm.grounding/1` physical-quantity ledger (see the `sim-grounding` skill).
- `episodes`: unique admitted episodes, each with `id`, `artifacts` (file references: `action`, `source`, `video`, `prompt`, `references`), `review`, `render_contract` (the compiled contract) and `submission` (the pre-submission verification record).

For white-only delivery, see `docs/reference/delivery-and-admission.md`.

## Review

Schema `datafarm.task-review/2`:

- `producer_id`, `reviewer_id` (must differ).
- `execution`: file reference to the independent execution evidence, recording `role=reviewer`, `actor_id`, `producer_id` and a file reference to the transcript of the actual review.
- `subjects`: the SHAs of the five artifacts.
- `verdicts`: `task`, `motion`, `reference`, `style`, `alignment` and `joint_margin` must all be pass.
- `evidence`: a non-empty list of file references.
- `visual_review`.

The validator checks role, source and content binding. It **cannot prove from JSON alone that an external host really isolated the identities**; the host is responsible for independent sessions and write permissions.

For tasks run by `drive` or `supervise` (a `.runtime` record exists), a hand-written execution is not accepted. The review writes `runtime_review` (the request ID in `state.review_requests`). The validator re-verifies that review's
completion, `verdict=pass`, that the request subjects cover all artifacts, and that the report evidence binds this review file. `review_independence` in the result is `runtime-verified`, otherwise `declared-execution-record`.
Never construct a reviewer execution as the producer. The expert harness remains responsible for isolated expert evaluation.

`visual_review` records `inspection_method=decoded-real-reference-and-generated-video`, binds the final generated video and all real reference SHAs, and records item by item `reference_baseline`,
`robot_body_and_attached_hardware`, `task_objects`, `table_and_background`, `declared_view_photography` and `overall_photographic_realism`.

- Each item has a real-reference-vs-generated pass, the generated frame number, the actual reference SHA, the view, and a decodable PNG/JPEG/WebP comparison image in the review evidence.
- The baseline, table/background, declared view and overall realism items require full-view evidence, and the view item covers every view the render intent declares. Text alone or local detail cannot prove the real style of the whole scene.
- Attached hardware or task objects that do not exist may be `not_applicable` with a reason and comparison evidence. Never invent an entity to satisfy the schema. `not_applicable` must agree with `appearance_requirements.required=false` in the render intent; a reason cannot waive a required item.
- The validator decodes comparison images and verifies structure and content binding. It cannot prove the reviewer's subjective conclusion or that media was really viewed; the independent execution record and the reviewer's accountability cover that.

This machine check verifies delivery source and count. It does not replace the video, physical judge or data-loader checks of the independent review.

Delivery validation also checks that the render contract, submission, final prompt and real references are consistent in content, and that reference sources belong to the input locked by prepare. Bind extra real photos with `prepare --reference`, and do not re-ask the user for images that exist in the episode.

## Resuming in a new session

`state.json` has schema `datafarm.agent-state/1`; the orchestrator continuously updates `phase`, `status`, `next_action`, `open_questions` and `external_jobs`. Save actual remote IDs, logs and breakpoints. On restart, read and query them, and take over existing work before considering a new submission.

`needs-input` is only for a missing fact, goal or access that cannot be derived. Write the specific question, the evidence checked and the independent work that can continue in `open_questions`; `drive` rejects an empty list. A missing adapter never qualifies. `complete` must be confirmed by a successful `verify`, not by the agent's last sentence.
