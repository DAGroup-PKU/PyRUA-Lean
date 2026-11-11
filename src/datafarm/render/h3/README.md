# H3 control hybrid harness

Production H3 render route of the pipeline: MiniMax-H3 Ref2VA + H3-Fun ControlNet-Union (soft-HED) + white inpaint,
independent 192-frame chunks, Sol-H3 turbo 4-step LoRA, 8 seeds + VLM per-chunk seed screen, min-diff seams,
first-chunk exposure-gain lock. Flow, rationale, evidence and limits: [docs/reference/rendering.md](../../../../docs/reference/rendering.md).
Default recipe: `configs/render/h3-control-hybrid-v1.json`; OOD: `configs/render/h3-ood-restyle-v1.json`;
evaluator: `evaluations/render-h3/`; VideoX-Fun changes: `vendor/videox_fun/`.

The tools run by path with `H3_PYTHON`, which need not have this package installed; `h3_paths.py` locates the checkout.

| Step | Tool |
|---|---|
| render-only replay of a white attempt (robot colour, shapes, dressing, dense rate, wide wrists) | `python -m datafarm.simulation.white_replay` |
| compose and check the T-layout white carrier | `h3_carrier.py` |
| real references from the task's episode (production set, memory budget, robot colours) | `h3_episode_references.py` |
| projected robot masks from the task's kinematics (light robots, dark furniture) | `h3_robot_masks.py` |
| keep real pixels for a task state (validated here, applied by the builder) | `h3_keep_regions.py` |
| validate a bundle, compile its contract | `h3_bundle.py` |
| clean real references from an episode (optional) | `h3_clean_references.py` |
| build one package | `h3_package.py` (usually through `h3_enqueue.py`) |
| enqueue packages or chunks | `h3_enqueue.py` |
| render | `launch_control_worker.sh` -> `h3_control_worker.py` |
| return tickets of dead workers to the queue | `h3_requeue_orphans.py` |
| seed screen composites, merge of answers | `h3_screen_prep.py` |
| seed choice, assembly, exposure lock, wide-wrist crop back, scores | `h3_finalize.py` (`h3_select.py`, `h3_assemble.py`, `h3_exposure_lock.py`, `h3_wrist_crop.py`) |
| score and write the review pack | `h3_score_and_pack.py` |
| blind review jobs | `h3_review_prep.py` |
| OOD edit of an accepted render | `h3_sam2_table_masks.py`, `h3_ood_restyle.py` |

Deployment bindings (fail closed when absent; `h3_paths.py`):

| Value | Source |
|---|---|
| run directory (tasks/, recipes/, pkgs/, queue/, cache/, screen/, reviews/, eval/realpool/) | `--run-dir` or env `H3_RUN` |
| `H3_PYTHON` | interpreter with torch/diffusers/VideoX-Fun deps; default: current interpreter |
| `H3_MODELS` | contains `MiniMax-H3/Ref2VA` and `MiniMax-H3-Fun-Controlnet-Union/` |
| `H3_VIDEOX_FUN` | VideoX-Fun checkout with `vendor/videox_fun/videox-fun-datafarm.patch` applied (pipeline, HED tool, `models/annotators/ControlNetHED.pth`) |
| `H3_TURBO_LORA` | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` (turbo profiles) |
| `GPU_SCHEDULER` | scheduler executable (`submit`/`leases`); hosts from `<run>/queue/ALLOWED_HOSTS` |
| `H3_EVAL_DINO` | DINOv2-base directory passed as `--dino` to the frozen EVAL-v1 scorer |
| `H3_SAM2` | `sam2.1-hiera-large` directory (OOD tools only) |
| `H3_HED_ANNOTATOR_REPO` | ControlNet v1.1 annotator source for the soft-HED tool `<checkout>/tools/prepare_hed_control.py` in the patched checkout |

Each key is read from the environment, then `DATAFARM_<KEY>`, then an explicit entry of the runtime bindings file.
`python3 h3_paths.py show` prints what resolves.

The task bundle, references, recipes and screen choices are task artifacts in the run directory; nothing here is a
task registry or an authorization. The mode and OOD axes come only from the bundle's `mode_confirmations`.
