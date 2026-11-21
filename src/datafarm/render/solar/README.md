# SolarRenderer chained bidirectional route

SolarRenderer is MiniMax-H3 with a control LoRA and a 4-step LoRA. It turns a whole white-carrier episode into
photographic three-camera video, with no per-chunk seeds, seed screening or hard seams. One bidirectional window renders
up to about 22 s cleanly, while a whole-episode block-causal stream drifts after 5–8 s. So an episode is rendered as
chained bidirectional windows. Each window is conditioned on the previous window's rendered frame, its static background
is re-anchored to the real frame 0, and the 16 overlapping frames are cross-faded.

For the flow, rationale, evidence and limits, see [docs/reference/rendering.md](../../../../docs/reference/rendering.md#solarrenderer-chained-bidirectional).
The default recipe is `configs/render/solar-chain-bidir-v1.json`. The flash-attn-free deployment is in
`vendor/solarrenderer/`.

The tools run by path. The GPU part runs with `SOLAR_PYTHON`; the rest needs only numpy, opencv and ffmpeg.

| Step | Tool |
|---|---|
| plain T-layout white carrier at the episode frame rate | `src/datafarm/render/h3/h3_carrier.py compose` (the H3 carrier; replay the white without wide wrists) |
| first frame: real frame 0 in the T layout and on the Solar canvas | `solar_first_frame.py` |
| prompt: six-part caption, state-neutral, checked by SolarRenderer's caption skill | `solar_prompt.py` (with `SOLAR_PYTHON`) |
| optional: lit displays filled with a flat colour in the carrier | `solar_carrier_displays.py` |
| plan the windows (no GPU) | `solar_chain.py --plan-only` |
| render one episode: windows, anchor, hand-over, stitch | `solar_chain.py` → `solar_clip.py` + `solar_window.py` |
| several episodes on the GPUs of a node the run owns | `solar_batch_local.py` |
| geometry, window plan, anchor, hand-over and stitch schedule (pure functions) | `solar_canvas.py` |

```bash
python solar_first_frame.py --head real/head.mp4 --left real/left.mp4 --right real/right.mp4 --carrier ep/carrier-t3.mp4 --out ep
$SOLAR_PYTHON solar_prompt.py ep/free.json ep/prompt.txt --duration 22
python solar_chain.py --carrier ep/carrier-t3.mp4 --plan-only --win 532
CUDA_VISIBLE_DEVICES=0 python solar_chain.py --carrier ep/carrier-t3.mp4 --first ep/first-canvas.png \
    --real ep/real-f0-t3.png --prompt ep/prompt.txt --root ep/render --fps 30 --win 532
```

The output is `ep/render/stitched-t3.mp4`: 1280x1080 at the carrier frame rate, with output frame *i* = carrier frame
*i*. `chain.json` records the plan, the per-window timings and peak memory, the hand-over edits and the seam frames.
Re-running the same command resumes from the finished windows.

## Appearance that changes mid-episode

A window keeps the first-frame look of every fixed object. A scale display that lights up halfway through a window stays
dark, and neither the prompt nor recolouring the carrier changes that. To fix it, end a window a few frames after the
change and give the next window a first frame that already shows the new state:

```bash
python solar_chain.py ... --wins 141,447 --handover 124:accepted-render-f124-t3.png \
    --handover-mask-carrier ep/carrier-white-t3.mp4
```

In this example the display lights at carrier frame 117. With `--wins 141` window 0 ends so that window 1 starts at frame
124 (141 − 1 − 16). The lit-display pixels detected in the white carrier at frame 124 are taken from an accepted render
of the same frame, here an H3 render; an explicit mask can be passed as `124:SRC.png:MASK.png`. Only that conditioning
frame is edited. On the weigh-apple task, hand-over at frame 124 (7 frames after the change) was accepted by blind
review on 5 of 5 episodes. Hand-over at frame 192 (75 frames late) was rejected on 4 of 5 for the state mismatch.
A short extra last window can show concentric banding in the head view, so choose `--wins` to avoid one.

## Deployment bindings

These fail closed when absent; see `solar_paths.py`.

| Value | Source |
|---|---|
| `SOLAR_PYTHON` | interpreter with SolarRenderer's dependencies; default: the current interpreter |
| `SOLAR_REPO` | SolarRenderer checkout at the pinned commit (`scripts/render.py`, `skills/solarrenderer-caption/`) |
| `SOLAR_H3` | MiniMax-H3 directory prepared by SolarRenderer `scripts/prepare_h3.py` (`transformer_ref/`, text encoder, VAE) |
| `SOLAR_LORA` | directory with `solarrenderer_control_lora.safetensors` and `solarrenderer_4step_lora.safetensors` (and the causal pair for `--mode causal` in `solar_window.py`) |

Each key is read from the environment, then `DATAFARM_<KEY>`, then an explicit entry of the runtime bindings file.
`python3 solar_paths.py show` prints what resolves.

Nothing here is a task registry or an authorization. The carrier, first frame, prompt, hand-over sources and chain
outputs are task artifacts in the run directory. Accept a render only through an independent blind review.
