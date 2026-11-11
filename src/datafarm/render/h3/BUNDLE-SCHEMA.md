# h3.task-bundle/1 — per-task input bundle for the H3 control-hybrid renderer

One file per task: `<run>/tasks/<task>/bundle.json` (`bundle-<variant>.json` for reference variants).
The bundle is a TASK ARTIFACT the agent derives for the current task (second layer); it is consumed by
`h3_bundle.py` (contract compile + checks), the H3 package builder, the evaluator and the review packer.
All paths are absolute deployment paths. Bundles are read-only inputs; never copy large media,
reference them by path (sha256 is computed by the tools).

The render mode and any OOD axes are NOT supplied by the
harness. `run_id` and `mode_confirmations` record the current user's in-scope instruction (verbatim quote +
where it was given); `h3_bundle.py` copies them into the render intent and stops when they are missing.
A quote from another run or user is not authorization for this one.

```json
{
  "schema": "h3.task-bundle/1",
  "task": "my-task",                         // short id, also the directory name
  "run_id": "RUN-...",                       // the run that owns this render (required)
  "task_id": "my-task-h3-control-hybrid",    // optional; default "<task>-h3-control-hybrid"
  "mode_confirmations": {                    // required; only the modes the current user confirmed
    "sim2real": {"quote": "<user words>", "source": "<where/when the user said it>"},
    "ood": {"quote": "<user words>", "source": "...",
            "authorized_ood_axes": ["<axis 1 exactly as authorized>", "..."]}
  },
  "layout": "t3",                            // carrier layout: 1280x1080, head (0,0,1280,720),
                                             //   left_wrist (0,720,640,360), right_wrist (640,720,640,360)
  "carrier_fps": 24,                         // native fps of the carrier files below (24 or 30)
  "appearance_views": ["head", "left_wrist", "right_wrist"],
  "episodes": [EPISODE, ...],
  "real_references": [REFERENCE, ...],       // default real refs (>=1 full-view full_scene per view)
  "per_episode_references": {"<episode id>": [REFERENCE, ...]},   // optional override
  "overlay": {                                // exactly these 5 slots (render-policy VARIABLES)
    "TASK_DESCRIPTION": "...", "SOURCE_VIDEO_ROLE": "...", "CAMERA_TIMING_CONTRACT": "...",
    "TASK_EVENT_CONTRACT": "...", "CONTINUITY_CONTRACT": "..."
  },
  "real_baseline": "one paragraph describing the real scene appearance",
  "appearance_invariants": ["...", "..."],
  "appearance_requirements": {               // exactly these 5 areas; robot_body,
    "robot_body": {"required": true, "description": "..."},          // table_and_background and
    "attached_cameras_and_cables": {"required": true|false, "description": "..."},  // photographic_character
    "task_objects": {"required": true|false, "description": "..."},   // must be required=true
    "table_and_background": {"required": true, "description": "..."},
    "photographic_character": {"required": true, "description": "..."}
  },
  "ood_variants": [                          // >= 2 when ood is confirmed; only the authorized axes may change
    {"name": "slate-laminate-cool", "tabletop": "matte slate-gray laminate", "lighting": "cool 6500K daylight"}
  ],
  "realpool": {"sources": [                  // real camera frames for realism metrics, per view
    {"video": "/abs/real/cam_high.mp4", "view": "head", "stride": 15},
    {"image": "/abs/real-tlayout.png", "view": "head", "region": [0, 0, 1280, 720]}
  ]},
  "notes": "anything the renderer/evaluator must know (e.g. render window, padding, fps relabel)"
}
```

EPISODE:
```json
{"id": "episode-000",
 "carrier": "/abs/white-authority-3view-<fps>fps.mp4",   // canonical 1280x1080 T-layout white video, >= 20 fps
 "frame_range": [100, 900] | null,        // half-open window of carrier frames to render (null = all)
 "wide_wrist": {"factor": 0.65,           // optional: the carrier's wrist views are a wide-FOV white replay;
                "principal_points": {"left": [cx, cy], "right": [cx, cy]}},  // 1280x720 wrist pixels; cropped back
 "keep_regions": "/abs/keep.json",        // optional: h3.keep-regions/1 spec (h3_keep_regions.py)
 "notes": "..."}
```

REFERENCE (render-policy real reference; sha256 is added by the tools):
```json
{"path": "/abs/real-3view.png",
 "provenance": {"kind": "real-episode-frame" | "real-photo", "source": "/abs/real/source.mp4 or image",
                "frame": 240, "description": "real episode X frame 240, cam_high|left|right tiled"},
 "scope": "full-view" | "local-detail",
 "views": ["head", "left_wrist", "right_wrist"],
 "view_regions": {"head": [0, 0, 1280, 720], "left_wrist": [0, 720, 640, 360], "right_wrist": [640, 720, 640, 360]},
 "coverage": ["full_scene", "robot_body", "attached_cameras_and_cables", "task_objects",
              "table_and_background", "photographic_character"]}
```

Rules
- Real references MUST be real camera frames (never white, simulated or generated frames). Every declared view
  needs a full-view reference with `full_scene` coverage; all required areas must be covered.
- The overlay text states object identity/count, phases, events and terminal relations provider-neutrally;
  it does not mention a provider, IMAGE numbers or chunk windows. CAMERA_TIMING_CONTRACT describes the
  T-layout (head on top, left wrist bottom-left, right wrist bottom-right), 24 fps playback, and that
  output frame i corresponds to source frame i.
- Check with: `python src/datafarm/render/h3/h3_bundle.py check <run>/tasks/<task>/bundle.json --run-dir <run>`
  (must print no problems).
- Reference policy of the default recipe (configs/render/h3-control-hybrid-v1.json, docs/reference/rendering.md):
  a real scene frame whose task state matches the carrier at the render start + a real robot sheet that keeps the
  real arm links; an object close-up sheet only for objects that stay in view (none for objects that are put away).
  `h3_episode_references.py` derives the production set (start frame, robot-visible frame, robot / object / wrist
  sheets) within the 25 MP memory budget; `h3_clean_references.py` derives an object-free plate and cable-free sheets.
- `h3_bundle.py check` fails on a carrier below 20 fps or a carrier rate that differs from `carrier_fps`, validates
  `wide_wrist` and `keep_regions`, and warns when the head view has no wall/table seam above the control gate.
