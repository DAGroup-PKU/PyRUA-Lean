# Render template usage

common-v1.txt plus one confirmed mode template form the core text. Task parameters come from the reviewed Task/Episode contracts; every `${...}` is a declared template slot that must be expanded and validated before a run. Sending a raw template to an API is forbidden.

| Slot | Authoritative input |
|---|---|
| MODE | Confirmed run intent whose scope matches |
| TASK_DESCRIPTION | Current task brief |
| REAL_REFERENCE_MAP | Real-image slots and source roles in the actual payload |
| SOURCE_VIDEO_ROLE | White-model video slot in the actual payload |
| CAMERA_TIMING_CONTRACT | Camera, panel, FPS, segment start/end, output timing |
| TASK_EVENT_CONTRACT | Object identity/count, phases, events and final state for this segment |
| CONTINUITY_CONTRACT | Relation of this segment to the previous one and continuity references; does not replace the real images |
| REAL_BASELINE_DESCRIPTION | Bound real target and photo description |
| APPEARANCE_INVARIANTS | Identity, material/color and similar properties that must not change |
| AUTHORIZED_OOD_AXES | Explicitly approved OOD axes and ranges; used only in OOD |

H3 control hybrid additionally has caption-style lowering templates `h3-caption-noevents-v1.txt` and `h3-caption-notask-v1.txt`: `render_policy.lower_h3_prompt` expands them deterministically from the compiled contract (intent, overlay) only. `-noevents` / `-notask` drop the event contract / the whole-segment task sentence respectively, and the `h3-caption-chunkvis-*` lowerings append closed-schema in-segment visibility sentences measured from the white model. `check_h3_payload` recomputes each job per the manifest's prompt_lowering and compares them one by one; no free text may be appended.

Seedance/H3 differ in slot numbering, image preprocessing and context capability: the adapter must generate the mapping from the actual payload, and the core templates must not assume IMAGE1/IMAGE2. Object counts, phases and continuity constraints belong in a task overlay and are verified against the run contract. A `source-only` recipe is not executable, and the authorization of one OOD run must not be copied to another.

These templates are wired into tools/render_policy.py and the Seedance/H3 submission checks. Correct static rules do not mean the model will comply; verify with a canary and an independent video review.
