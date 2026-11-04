# Data mode and real-world style

Video integrity and action alignment do not show that the data is what the user wants. Table-height randomization is a
simulation geometry design and cannot by itself authorize changing the table texture, the wall or the lighting. The
rendering service's model does not choose the data mode.

## The mode must be explicit

- `sim2real` means non-OOD: restore the appearance of this task's real target scene.
- `ood` means a change, relative to the recorded real baseline, that the user explicitly approved.
- `unresolved` is a state awaiting a question; submitting a render is forbidden.

If the user's current or standing instruction makes the mode of this task explicit, record the exact words and scope and
continue. Otherwise ask: "Should this batch be non-OOD (match the real scene) or OOD (please state the scope of change
explicitly)? Both will reference real images and keep real-world style." While waiting, continue observing, scene and
expert coding, and authorized small-sample validation that does not depend on the mode. Do not pick a mode and run it first.

An OOD run records the baseline, the axes of change, the value ranges, and the identities and relations that stay unchanged.
Allowing a change of table material does not allow a change of arm colour, object count or camera layout. An approved
axis allows only physically plausible changes that still look like real footage starting from the real baseline, not an
appearance detached from the real episode. Lighting diversity and hard shadows are not defects in themselves, but style
axes that were not approved must not be added. The absence of a general lighting constraint does not extend this task's scope.

## The inputs have different roles

- The white-mesh video provides the action, camera, timing, object presence, contact and terminal-state relations.
- The real images provide the whole-scene appearance baseline: the robot body, the attached cameras/cables and task
  objects that actually exist, the table/background, real materials and shapes, and the photographic feel of every view
  the task declares (all three for a head/left-wrist/right-wrist episode). A reasonable shape correction of a real object
  relative to its simulation proxy is not an action error.
- Pasting one wrist texture onto a scene that still looks like a white mesh does not count as adopting the real reference.
- A reference image has appearance authority only. It provides no object positions, action progress or camera layout.
  Never copy the action or layout from a reference, copy its arm or gripper into an empty area, or add entities absent from the source.

Each real reference has a source type, an acquisition note, a file hash, readability and a role. The white-mesh first
frame, generated images and continuity frames may be added but cannot pass for real images. A real-source claim needs an
auditable provenance record; a hash only proves the content is unchanged, not that the photo comes from the real world.

## Submission and review

Before submission, freeze the mode, the confirmation basis, the reference mapping, the prompt version and the provider.
The actual request must carry at least one valid real reference consistent with the frozen mapping; a path in a local
config does not count. Save credential-free submission metadata, the mapping from asset hashes to provider resource
identifiers, the final prompt and the request hash. Never save URLs with signed credentials.

After submission, review separately:

1. whether the real references were really submitted with the request;
2. whether the final video adopts the real appearance and has real-world style;
3. whether the action, topology and timing are preserved.

Both the executing agent and the independent reviewer personally open or decode the real image or episode frames and the
generated video, and compare them side by side. The prompt, hashes, the producer's summary or an earlier VLM text cannot
replace looking. Write `unverified` for any item lacking evidence, and do not borrow a pass from another item. Keep the
reviewer, frame ranges, evidence and judge version. A static three-frame preview is only a first screen and cannot show
that the timing of the whole video is normal.

The style check covers at least the real baseline, the material and shape of the arm body, the attached cameras/cables
and task objects that actually exist, the table/background, the photographic consistency of all declared views, and
reference-layout leakage. A component that does not exist is `not_applicable` with evidence and a reason, and must not be
invented. Every style pass saves a comparison image bound to the real reference and the generated video, the view and the
generated frame numbers. Whole-scene items use full-view evidence; a local texture or crop cannot claim full-scene
coverage. Judge against the real reference and the approved OOD changes: a reasonable real size, hard shadow or directional
light that deviates from the white mesh is not a defect. The default exposure threshold of 60/255 is a coarse guard and
cannot declare "real enough".

## No automatic downgrade

A missing mode, an OOD beyond authorization, a missing, unreadable or not-uploaded real reference, a source-only switch, a
reference replaced by a white-mesh frame, or an unverified review must never let production continue silently. Diagnostics
cannot waive the real reference either. If the user explicitly changes this requirement, first update the specification,
configuration, prompts and verification, and only then run.

Never backfill authorization into existing data: it keeps its evidence, a missing contract is unknown or needs-review, and
a re-check or re-capture produces a new version and certificate.
