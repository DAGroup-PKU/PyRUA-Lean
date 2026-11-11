#!/usr/bin/env python3
"""Build one Ref2VA + H3-Fun ControlNet (soft-HED) + inpaint hybrid render package ("backend": "control-hybrid").

One package is one task episode x mode x recipe x seed: contract (render-policy), guttered 1024x768 canvas carrier,
independent 17n+5-frame chunks with keep ranges, job manifest and ticket.  Every chunk gets:

  control_video  soft-HED map of the white chunk clip, computed by the VideoX-Fun tool
                 (${H3_VIDEOX_FUN}/tools/prepare_hed_control.py, ControlNetHED annotator, run with
                 ${H3_PYTHON}; --min-edge-intensity from the recipe; 0 = the soft map), on the same canvas;
  inpaint_video  the white chunk clip itself (white RGB inpaint source on the 80+1280+80 padded canvas);
  inpaint_mask   active-panel repaint mask: 1 (regenerate) on canvas x=[57,967), 0 (keep) on the black
                 gutters (regenerate fraction 910/1024 = 0.888672).

Render it with h3_control_worker.py and assemble with h3_assemble.py.  The render-policy binding is checked at build
time with datafarm.render.policy.check_h3_payload (prompt file == compiled contract prompt, contract real
references first) and again by the worker before rendering.  ticket.json is written last, so an interrupted or
rejected build can never be enqueued.

  python h3_package.py build --run-dir <run> --bundle <run>/tasks/<task>/bundle.json \
      --recipe <run>/recipes/<recipe>.json --episode <episode> --mode sim2real \
      --out <run>/pkgs/<recipe>/<task>/<episode>-sim2real [--hed-jobs 4 --hed-threads 16]
  python h3_package.py check <pkg>      # re-run the binding/input audit only

The run directory (--run-dir or env H3_RUN) holds the shared HED caches; model, tool and checkpoint paths come from the
recipe or the deployment bindings (h3_paths.py), never from built-in defaults.

Recipe JSON fields: name, backend, worker_profile, canvas ("t3-guttered-1024x768"), denoise_forwards,
chunk {frames, advance, keep_head, event_aware?, tail_pad?}, seed_base, control {...}, prompt_lowering?, assembly {...},
bundle_variant?.

chunk.tail_pad holds the carrier's last frame for that many extra frames before chunking, so the episode's real last
frame is never the end of the last chunk, where the generator tends to invent motion or objects; assembly and the
exposure lock cut the output back to the episode length (frame i of the output is still carrier frame i).  The bundle
episode may name `keep_regions` (h3_keep_regions.py: real pixels kept where the white shows a task state) and
`wide_wrist` (wide-FOV wrist white, cropped back by h3_wrist_crop.py after the exposure lock).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import h3_bundle
import h3_keep_regions
import h3_paths
from h3_media import link_or_copy, probe, sha256
from h3_spatial import SPATIAL

h3_paths.datafarm_src()
from datafarm.render.policy import check_h3_payload, lower_h3_prompt  # noqa: E402

CANVAS = "t3-guttered-1024x768"
MANIFEST_SCHEMA = "h3.control-hybrid-manifest/1"
LEGAL = [17 * n + 5 for n in range(7, 22)]  # 124 .. 362
# chunkvis wrist emptiness.  Saturation alone misses white or pale task objects, so a wrist view also counts as showing
# an object when it has edge structure the empty room cannot explain (wrist_structure()).
CHUNK_FACTS_DETECTORS = ("wrist-structure-v1",)
WRIST_EDGE_TAU = 12          # Canny low threshold on the blurred grey wrist view (high = 3x)
WRIST_SEAMS = 4              # straight room seams peeled per frame (wall/table horizon, table edges, wall corners)
WRIST_MIN_STRUCTURE = 0.75   # residual edge length, in wrist-view widths, that marks a task object in a frame
# control paths a recipe may name; ${KEY} placeholders (configs/render/h3-control-hybrid-v1.json) are expanded from the
# deployment bindings, absent keys default to h3_paths.default_model_paths() (also from the bindings)
CONTROL_PATH_KEYS = ("hed_tool", "hed_checkpoint", "base_model", "control_checkpoint", "control_config",
                     "videox_fun_repo")
PROFILES = ("control-turbo",)


def run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def plan_chunks(total: int, frames: int, advance: int, keep_head: int, events: list | tuple = (), before: int = 24,
                min_advance: int = 48) -> list[dict]:
    """Chunks of `frames` (legal 17n+5), advancing `advance`; the last chunk is end-aligned.

    Keep ranges are [start_k + keep_head, start_{k+1} + keep_head).  With `events` (frames where an arm enters or leaves
    the head view) a chunk never starts within keep_head frames after an event: such a start moves to `before` frames
    ahead of the event, within [min_advance, advance], so the chunk generates the entry or exit itself instead of
    starting into its aftermath, which tends to draw a ghost gripper."""
    if total < LEGAL[0]:
        raise ValueError(f"episode shorter than {LEGAL[0]} frames: {total}")
    if frames not in LEGAL:
        raise ValueError(f"chunk frames {frames} not legal")
    starts, moved = [0], []
    while starts[-1] + frames < total:
        nxt = starts[-1] + advance
        for e in events:
            if e < nxt <= e + keep_head and e - before >= starts[-1] + min_advance:
                moved.append([nxt, e - before, e])
                nxt = e - before
                break
        if nxt + frames >= total:
            nxt = max(total - frames, starts[-1] + 1)
        starts.append(nxt)
    chunks = []
    for k, s in enumerate(starts):
        length = min(frames, total - s) if k == len(starts) - 1 else frames
        if length not in LEGAL:
            length = next(v for v in LEGAL if v >= length)
        chunks.append({"index": k, "start": s, "frames": length})
    for k, c in enumerate(chunks):
        a = 0 if k == 0 else chunks[k - 1]["keep_global"][1]
        b = total if k == len(chunks) - 1 else min(chunks[k + 1]["start"] + keep_head, total)
        if b <= a or b > c["start"] + c["frames"]:
            raise ValueError(f"chunk {k}: invalid keep range {a}..{b} for start {c['start']}")
        c["keep_global"] = [a, b]
        c["keep_local"] = [a - c["start"], b - c["start"]]
    if moved:
        chunks[0]["moved_starts"] = moved
    return chunks


def head_arm_sides(lab, st, hw: int, hy: int) -> set:
    """Which arms a head-view dark-robot labelling shows.  A component touching the left (right) border is the left
    (right) arm.  A component touching only the top edge (a distal link or gripper detached from its arm in the mask)
    belongs to the arm that does touch a side border when exactly one side does; otherwise its side is the mean
    column of its top-edge contact."""
    import numpy as np
    big = [k for k in range(1, len(st)) if st[k][4] >= 0.003 * hw * hy]
    left = any(st[k][0] <= 6 for k in big)
    right = any(st[k][0] + st[k][2] >= hw - 6 for k in big)
    sides = set(["left"] if left else []) | set(["right"] if right else [])
    for k in big:
        bx, by, bw, bh, area = st[k]
        if bx <= 6 or bx + bw >= hw - 6:
            continue
        top = np.nonzero(lab[:6] == k)[1]
        if not top.size:
            continue
        if left != right:
            sides.add("left" if left else "right")
        else:
            sides.add("left" if top.mean() < hw / 2 else "right")
    return sides


def arm_events(canvas: Path, total: int, min_gap: int = 24) -> list:
    """Frames where a robot arm enters or leaves the head view of the white carrier (either side).

    Presence per side and frame = a dark robot component (V<110, S<90, >= 0.3 % of the head area) touching that side's
    border (or the top edge in that half), as in chunk_facts; 9-frame majority smoothing; toggles closer than min_gap
    frames to the previous kept event are merged."""
    import cv2
    import numpy as np
    x0, x1, hy = SPATIAL.canvas_active_x0, SPATIAL.canvas_active_x1, SPATIAL.canvas_head_height
    cap = cv2.VideoCapture(str(canvas))
    pres, i = [], 0
    while i < total:
        ok, fr = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(fr[:hy, x0:x1], cv2.COLOR_BGR2HSV)
        dark = ((hsv[..., 2] < 110) & (hsv[..., 1] < 90)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        _, lab, st, _ = cv2.connectedComponentsWithStats(dark)
        ss = head_arm_sides(lab, st, x1 - x0, hy)
        pres.append([int("left" in ss), int("right" in ss)])
        i += 1
    cap.release()
    p = np.array(pres, float)
    if len(p) == 0:
        return []
    k = 9
    pad = np.pad(p, ((k // 2, k // 2), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, c], np.ones(k) / k, mode="valid") for c in range(2)], 1) > 0.5
    out = []
    for t in range(1, len(sm)):
        if (sm[t] != sm[t - 1]).any() and (not out or t - out[-1] >= min_gap):
            out.append(int(t))
    return out


def control_settings(recipe: dict) -> dict:
    c = h3_paths.expand(dict(recipe.get("control") or {}))
    c.setdefault("type", "soft-hed")
    c.setdefault("hed_min_intensity", 0)
    c.setdefault("context_scale", 1.0)
    c.setdefault("inpaint_source", "white")
    c.setdefault("mask", "active-repaint")
    c.setdefault("guidance_scale", 1.0)
    c.setdefault("white_video_reference", False)
    c.setdefault("edge_gate", None)
    if c["type"] == "soft-hed":
        for key in ("hed_tool", "hed_checkpoint"):
            if key not in c:
                c[key] = h3_paths.default_model_path(key)
    for key in CONTROL_PATH_KEYS:  # normalized so the worker's loaded-model comparison is exact
        if c.get(key):
            c[key] = str(Path(c[key]))
    if c["type"] != "soft-hed":
        raise SystemExit(f"unsupported control type {c['type']!r} (soft-hed only)")
    if (c["inpaint_source"], c["mask"]) != ("white", "active-repaint"):
        raise SystemExit("inpaint_source/mask must be white/active-repaint")
    if float(c["guidance_scale"]) > 1.0:
        raise SystemExit("Ref2VA is guidance-distilled: guidance_scale must be <= 1")
    if not 0.0 <= float(c["context_scale"]) <= 1.0:
        raise SystemExit("control context scale must be in [0, 1] (model card maximum is 1.0)")
    return c


def make_mask(path: Path, frames: int) -> dict:
    """Active-panel repaint mask on the H3 canvas: white (regenerate) on x=[57,967) at full height, black gutters."""
    w, h, x0, x1 = SPATIAL.canvas_width, SPATIAL.canvas_height, SPATIAL.canvas_active_x0, SPATIAL.canvas_active_x1
    if not (path.is_file() and probe(path)[:3] == (w, h, frames)):
        path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg("-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=24", "-vf",
               f"drawbox=x={x0}:y=0:w={x1 - x0}:h={h}:color=white:t=fill,format=yuv420p",
               "-frames:v", str(frames), "-an", "-r", "24", "-c:v", "libx264", "-preset", "medium",
               "-crf", "0", "-pix_fmt", "yuv420p", str(path))
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot decode mask {path}")
    m = frame[:, :, 2] > 127  # R channel after BGR decode, thresholded like the pipeline (> 0.5)
    expected = np.zeros_like(m)
    expected[:, x0:x1] = True
    if not np.array_equal(m, expected):
        raise SystemExit(f"mask {path} does not match the active-panel contract")
    return {"path": str(path), "sha256": sha256(path), "frames": frames, "regenerate_fraction": float(m.mean()),
            "active_x": [x0, x1]}


def make_hed(job: dict, out_dir: Path, ctl: dict, threads: int, device: str) -> dict:
    w, h = SPATIAL.canvas_width, SPATIAL.canvas_height
    white = Path(job["white"])
    frames = int(job["generated_frames"])
    mi = int(ctl["hed_min_intensity"])
    tag = "soft" if mi == 0 else f"min{mi}"
    target = out_dir / f"{job['chunk_id']}-hed-{tag}.mp4"
    sidecar = target.with_suffix(".h3.json")
    if target.is_file() and sidecar.is_file():
        rec = json.loads(sidecar.read_text())
        if rec.get("input_sha256") == job["white_sha256"] and rec.get("frames") == frames \
                and rec.get("min_edge_intensity") == mi and probe(target)[:3] == (w, h, frames):
            return rec
    # global cache: identical white chunk clip + HED settings -> identical control video (shared across recipes)
    cache = h3_paths.run_dir() / "cache" / "hed" / f"{job['white_sha256']}-{mi}-{frames}"
    if (cache.with_suffix(".mp4")).is_file() and (cache.with_suffix(".h3.json")).is_file():
        rec = json.loads(cache.with_suffix(".h3.json").read_text())
        target.unlink(missing_ok=True)
        link_or_copy(cache.with_suffix(".mp4"), target)
        rec = {**rec, "path": str(target), "input": str(white), "cache": str(cache.with_suffix(".mp4"))}
        sidecar.write_text(json.dumps(rec, indent=2))
        return rec
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    annotator = h3_paths.binding("H3_HED_ANNOTATOR_REPO", required=False, must_exist=False)
    if annotator:  # read by the soft-HED tool of the patched VideoX-Fun
        env["H3_HED_ANNOTATOR_REPO"] = annotator
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.time()
    r = run([h3_paths.python_exe(), ctl["hed_tool"], "--input", str(white), "--output", str(target), "--checkpoint",
             ctl["hed_checkpoint"], "--width", str(w), "--height", str(h),
             "--max-frames", str(frames), "--min-edge-intensity", str(mi)], env=env)
    meta = json.loads(target.with_suffix(".json").read_text())
    got = probe(target)
    if got[:3] != (w, h, frames) or meta["frames"] != frames:
        raise SystemExit(f"{target}: HED contract {got[:3]} != {w}x{h}x{frames}")
    rec = {"path": str(target), "sha256": sha256(target), "input": str(white),
           "input_sha256": job["white_sha256"], "frames": frames, "min_edge_intensity": mi,
           "tool": ctl["hed_tool"], "tool_sha256": sha256(Path(ctl["hed_tool"])),
           "checkpoint": ctl["hed_checkpoint"], "device": meta.get("device"), "rate": got.rate,
           "mean_active_fraction_gt20": meta["mean_active_fraction_gt20"],
           "mean_normalized_intensity": meta["mean_normalized_intensity"], "seconds": round(time.time() - t0, 1),
           "tool_stdout_tail": r.stdout[-300:]}
    sidecar.write_text(json.dumps(rec, indent=2))
    cache.parent.mkdir(parents=True, exist_ok=True)
    link_or_copy(target, cache.with_suffix(".mp4"))
    cache.with_suffix(".h3.json").write_text(json.dumps(rec, indent=2))
    return rec


def wrist_structure(panel) -> float:
    """Edge structure of one BGR wrist view that the empty room cannot explain, in view widths of edge length.

    An empty white-mesh wrist view is the plain wall and tabletop, a few straight seams between them (horizon, table
    edges, wall corners) and dark robot pixels (its own fingers, the other arm, dark cast shadows).  Canny edges of the
    blurred grey view outside the dilated dark-robot mask, minus up to WRIST_SEAMS straight seams peeled longest first
    (each removed along its whole line across the view) and minus specks < 12 px, are what a task object adds whatever
    its colour: outlines and buttons of a white device, cabinet fronts and handles, cloth folds, a white cube.
    Empty views measure ~0-0.5, such objects ~0.8-7."""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    g = cv2.GaussianBlur(cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    h, w = g.shape
    robot = cv2.dilate(((hsv[..., 2] < 110) & (hsv[..., 1] < 90)).astype(np.uint8), np.ones((11, 11), np.uint8))
    edges = cv2.Canny(g, WRIST_EDGE_TAU, 3 * WRIST_EDGE_TAU)
    edges[robot > 0] = 0
    edges[:3] = 0
    edges[-3:] = 0
    edges[:, :3] = 0
    edges[:, -3:] = 0
    reach = 2 * (w + h)
    for _ in range(WRIST_SEAMS):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=20, minLineLength=int(0.1 * w), maxLineGap=4)
        if lines is None:
            break
        a, b, c, d = max(lines[:, 0].tolist(), key=lambda s: math.hypot(s[2] - s[0], s[3] - s[1]))
        n = max(1e-6, math.hypot(c - a, d - b))
        ux, uy = (c - a) / n * reach, (d - b) / n * reach
        band = np.zeros_like(edges)
        cv2.line(band, (int(a - ux), int(b - uy)), (int(c + ux), int(d + uy)), 1, 7)
        edges[band > 0] = 0
    _, lab, st, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= 12
    keep[0] = False
    return float(keep[lab].sum()) / w


def chunk_facts(job: dict, cfg: dict | None = None) -> dict:
    """Measure chunk-local visibility facts from the chunk's white carrier clip on the guttered canvas
    (T-layout: head = top 2/3, left wrist = bottom-left, right wrist = bottom-right of the active area).
    head arm side present  <=> a dark robot component (V<110, S<90, >=0.3% of the head area) touches that side's
    border (left/right edge, or the top edge in that half) in >= 15% of sampled frames;
    wrist empty            <=> in >= 90% of sampled frames that wrist view shows no task object: saturated pixels
    (S>=70, V>=40) cover <= 0.4% of it AND wrist_structure() < 0.75, so a white or pale object is not measured as absent;
    head_max_coloured_objects = max over sampled frames of the number of separate saturated components (S>=70, V>=40,
    opened 3x3, closed 9x9, >= 0.15% of the head view) in the head view;
    wrist_large_objects    <=> a non-empty wrist view has saturated pixels on >= 6% of it in >= 25% of sampled frames."""
    import cv2
    import numpy as np
    cfg = cfg or {}
    detector = cfg.get("detector", CHUNK_FACTS_DETECTORS[0])
    if detector not in CHUNK_FACTS_DETECTORS:
        raise SystemExit(f"unknown chunk facts detector {detector!r}; one of {CHUNK_FACTS_DETECTORS}")
    x0, x1, hy = SPATIAL.canvas_active_x0, SPATIAL.canvas_active_x1, SPATIAL.canvas_head_height
    xm = (x0 + x1) // 2
    cap = cv2.VideoCapture(job["white"])
    i, arm, max_obj = 0, {"left": 0, "right": 0}, 0
    wobj = {"left_wrist": 0, "right_wrist": 0}
    wbig = {"left_wrist": 0, "right_wrist": 0}
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % 4 == 0:
            n += 1
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            head = hsv[:hy, x0:x1]
            dark = ((head[..., 2] < 110) & (head[..., 1] < 90)).astype(np.uint8)
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            nl, lab, st, cen = cv2.connectedComponentsWithStats(dark)
            hw = x1 - x0
            seen = head_arm_sides(lab, st, hw, hy)
            for side in seen:
                arm[side] += 1
            col = ((head[..., 1] >= 70) & (head[..., 2] >= 40)).astype(np.uint8)
            col = cv2.morphologyEx(col, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            col = cv2.morphologyEx(col, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            nc, _, stc, _ = cv2.connectedComponentsWithStats(col)
            max_obj = max(max_obj, int((stc[1:, cv2.CC_STAT_AREA] >= 0.0015 * hw * hy).sum()))
            for name, (a, b) in (("left_wrist", (x0, xm)), ("right_wrist", (xm, x1))):
                v = hsv[hy:, a:b]
                sat = (v[..., 1] >= 70) & (v[..., 2] >= 40)
                if sat.mean() > 0.004 or wrist_structure(fr[hy:, a:b]) >= WRIST_MIN_STRUCTURE:
                    wobj[name] += 1
                if sat.mean() >= 0.06:  # a large, very close task object at the fingers
                    wbig[name] += 1
        i += 1
    cap.release()
    absent = [s for s in ("left", "right") if arm[s] < 0.15 * max(1, n)]
    empty = [w for w in ("left_wrist", "right_wrist") if wobj[w] <= 0.10 * max(1, n)]
    return {"head_absent_arms": absent, "empty_wrists": empty, "head_max_coloured_objects": min(max_obj, 9),
            "wrist_large_objects": [w for w in ("left_wrist", "right_wrist")
                                    if w not in empty and wbig[w] >= 0.25 * max(1, n)]}


def make_gated(job: dict, hed: dict, gate: dict) -> dict:
    """Shadow-suppressed control: keep HED edges only where the white carrier itself has a strong local
    luminance step (max-min of the blurred gray image in a window > tau).  Simulator cast-shadow boundaries on
    the white table are low-contrast steps (~10-30 levels) and are removed; object/arm silhouettes and white
    rims on the table (>= ~25-40 levels) are kept.  Deterministic, cached by (white sha, HED sha, gate)."""
    import cv2
    import numpy as np
    tau, win, dil = float(gate.get("tau", 25)), int(gate.get("window", 7)), int(gate.get("dilate", 5))
    key = f"{job['white_sha256']}-{hed['sha256'][:16]}-gate{int(tau)}w{win}d{dil}"
    cache = h3_paths.run_dir() / "cache" / "hedgate" / key
    target = Path(hed["path"]).with_name(Path(hed["path"]).stem + f"-gate{int(tau)}.mp4")
    if cache.with_suffix(".mp4").is_file():
        target.unlink(missing_ok=True)
        link_or_copy(cache.with_suffix(".mp4"), target)
    else:
        white, control = cv2.VideoCapture(job["white"]), cv2.VideoCapture(hed["path"])
        tmp = target.with_suffix(".tmp.mp4")
        enc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                                "bgr24", "-s", f"{SPATIAL.canvas_width}x{SPATIAL.canvas_height}", "-r", "24", "-i", "-",
                                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p",
                                str(tmp)], stdin=subprocess.PIPE)
        n = 0
        kern_w, kern_d = np.ones((win, win), np.uint8), np.ones((dil, dil), np.uint8)
        while True:
            okw, fw = white.read()
            okh, fh = control.read()
            if not (okw and okh):
                break
            g = cv2.GaussianBlur(cv2.cvtColor(fw, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
            step = cv2.dilate(g, kern_w) - cv2.erode(g, kern_w)
            m = cv2.dilate((step > tau).astype(np.uint8), kern_d)
            enc.stdin.write(np.ascontiguousarray(fh * m[..., None]).tobytes())
            n += 1
        enc.stdin.close()
        if enc.wait() or n != int(job["generated_frames"]):
            raise SystemExit(f"gated control failed for {job['chunk_id']}: {n} frames")
        tmp.replace(target)
        cache.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(target, cache.with_suffix(".mp4"))
    return {**hed, "path": str(target), "sha256": sha256(target), "ungated_path": hed["path"],
            "ungated_sha256": hed["sha256"], "edge_gate": {"type": "luma-step", "tau": tau, "window": win, "dilate": dil}}


def audit(pkg_dir: Path) -> dict:
    """Render-policy binding + control-input contract; used at build time and by the worker."""
    manifest = json.loads((pkg_dir / "job-manifest.json").read_text())
    policy = check_h3_payload(manifest, pkg_dir / "contract" / "contract.json")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("controlnet") is not True:
        raise ValueError("not a control-hybrid manifest")
    if manifest.get("reference_order") != "images-first":
        raise ValueError("reference_order must be images-first")
    problems = []
    canvas = (SPATIAL.canvas_width, SPATIAL.canvas_height)
    for job in manifest["jobs"]:
        n = int(job["generated_frames"])
        if n < 124 or n > 362 or (n - 5) % 17:
            problems.append(f"{job['chunk_id']}: {n} frames is not a legal 17n+5 H3 shape")
        for key in ("white", "control_video", "inpaint_video", "inpaint_mask"):
            p = Path(job[key])
            if not p.is_file():
                problems.append(f"{job['chunk_id']}: missing {key} {p}")
                continue
            sha = job.get(f"{key}_sha256")
            if sha and sha256(p) != sha:
                problems.append(f"{job['chunk_id']}: {key} sha256 mismatch")
            info = probe(p)
            if info[:2] != canvas or info.frames < n:
                problems.append(f"{job['chunk_id']}: {key} is {info.width}x{info.height}x{info.frames}, "
                                f"need canvas x {n}")
    if problems:
        raise ValueError("control package audit failed:\n" + "\n".join(problems))
    return policy


def build_chunk_jobs(out: Path, task: str, ep_id: str, contract: dict, canvas: Path, chunks: list, total: int,
                     seed: int, steps: int, prompt: str) -> list[dict]:
    """White chunk clips (padded by clone at the episode end when the legal shape runs past it) and their job records."""
    refs = [r["path"] for r in contract["intent"]["real_references"]]
    jobs = []
    for c in chunks:
        cid = f"{task}-{ep_id}-c{c['index']:02d}"
        clip = out / "inputs" / f"{cid}-white.mp4"
        pad = max(0, c["start"] + c["frames"] - total)
        vf = f"trim=start_frame={c['start']}:end_frame={min(total, c['start'] + c['frames'])},setpts=PTS-STARTPTS"
        if pad:
            vf += f",tpad=stop_mode=clone:stop={pad}"
        ffmpeg("-i", str(canvas), "-vf", vf, "-fps_mode", "cfr", "-an", "-r", "24", "-c:v", "libx264", "-preset", "fast",
               "-crf", "10", "-pix_fmt", "yuv420p", str(clip))
        if probe(clip).frames != c["frames"]:
            raise SystemExit(f"{cid}: clip frames {probe(clip).frames} != {c['frames']}")
        jobs.append({"chunk_id": cid, "chunk_index": c["index"], "sample_id": ep_id,
                     "source_frame_range": [c["start"], c["start"] + c["frames"]],
                     "generated_frames": c["frames"], "keep_local_range": c["keep_local"],
                     "keep_global_range": c["keep_global"], "white": str(clip), "white_sha256": sha256(clip),
                     "reference_images": refs, "prompt": prompt, "seed": seed, "steps": steps})
    return jobs


def add_control_inputs(jobs: list[dict], ctl: dict, inputs: Path, args) -> tuple[dict, list]:
    """Add the control inputs (soft-HED, optionally gated, and the repaint mask) to the chunk jobs."""
    masks = {n: make_mask(inputs / f"repaint-mask-active-{n}f.mp4", n)
             for n in sorted({int(j["generated_frames"]) for j in jobs})}
    with cf.ThreadPoolExecutor(max(1, args.hed_jobs)) as ex:
        heds = list(ex.map(lambda j: make_hed(j, inputs, ctl, args.hed_threads, args.hed_device), jobs))
    if ctl.get("edge_gate"):
        with cf.ThreadPoolExecutor(4) as ex:
            heds = list(ex.map(lambda jh: make_gated(jh[0], jh[1], ctl["edge_gate"]), zip(jobs, heds)))
    for job, hed in zip(jobs, heds):
        mask = masks[int(job["generated_frames"])]
        job.update({"guidance_scale": float(ctl["guidance_scale"]),
                    "control_context_scale": float(ctl["context_scale"]),
                    "control_video": hed["path"], "control_video_sha256": hed["sha256"],
                    "control_meta": {k: hed[k] for k in ("min_edge_intensity", "mean_active_fraction_gt20",
                                                         "mean_normalized_intensity", "device", "tool_sha256")},
                    "inpaint_video": job["white"], "inpaint_video_sha256": job["white_sha256"],
                    "inpaint_mask": mask["path"], "inpaint_mask_sha256": mask["sha256"],
                    "inpaint_regenerate_fraction": mask["regenerate_fraction"]})
    return masks, heds


def add_chunk_prompts(jobs: list[dict], contract: dict, lowering: str, contract_dir: Path, detector: str) -> None:
    """Chunk-local lowered prompts from the facts measured on each white clip."""
    all_facts = [chunk_facts(job, {"detector": detector}) for job in jobs]
    if all(f.get("head_max_coloured_objects", 0) == 0 for f in all_facts):
        # the carrier never shows a measurable coloured object (e.g. a pale, small block): no count is stated at
        # all, so a measurement miss can never tell the generator to drop a real object
        for f in all_facts:
            f.pop("head_max_coloured_objects", None)
    for job, facts in zip(jobs, all_facts):
        pf = contract_dir / f"prompt-chunkvis-{job['chunk_id']}.txt"
        pf.write_text(lower_h3_prompt(contract, lowering, assets=h3_paths.REPO, chunk_facts=facts))
        job.update(chunk_facts=facts, chunk_facts_detector=detector, prompt=str(pf))


def cmd_build(args) -> int:
    h3_paths.run_dir()  # fail closed before any work: HED caches live in the run directory
    recipe_path = Path(args.recipe).resolve()
    recipe = json.loads(recipe_path.read_text())
    if recipe.get("backend") != "control-hybrid":
        raise SystemExit("recipe backend must be 'control-hybrid'")
    if recipe.get("worker_profile") not in PROFILES:
        raise SystemExit(f"control-hybrid recipes must use a worker_profile in {PROFILES}")
    if recipe.get("canvas", CANVAS) != CANVAS:
        raise SystemExit(f"unknown canvas {recipe['canvas']}")
    ctl = control_settings(recipe)
    bundle = h3_bundle.load(args.bundle)
    ep = next(e for e in bundle["episodes"] if e["id"] == args.episode)
    out = Path(args.out).resolve()
    if (out / "package.json").exists() and not args.force:
        raise SystemExit(f"package exists: {out}")
    t0 = time.time()
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    carrier = Path(ep["carrier"])
    fr = ep.get("frame_range") or [0, probe(carrier).frames]
    total = fr[1] - fr[0]
    ch = recipe["chunk"]
    tail_pad = int(ch.get("tail_pad", 0))
    if tail_pad < 0:
        raise SystemExit("chunk.tail_pad must be >= 0")
    render_total = total + tail_pad
    # contract + canvas carrier (the last frame held for tail_pad extra frames)
    contract = h3_bundle.compile_contract(bundle, args.mode, ep["id"], args.variant, out / "contract")
    canvas = inputs / "carrier-canvas-24fps.mp4"
    ffmpeg("-i", str(carrier), "-vf", f"trim=start_frame={fr[0]}:end_frame={fr[1]},setpts=N/(24*TB),"
           f"{SPATIAL.ffmpeg_to_canvas()}" + (f",tpad=stop_mode=clone:stop={tail_pad}" if tail_pad else ""),
           "-fps_mode", "cfr", "-an", "-r", "24", "-c:v", "libx264",
           "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p", str(canvas))
    if probe(canvas).frames != render_total:
        raise SystemExit(f"canvas carrier frame count {probe(canvas).frames} != {render_total}")
    # chunk plan (over the padded carrier)
    events = ()
    if ch.get("event_aware"):
        events = arm_events(canvas, render_total)
    chunks = plan_chunks(render_total, ch["frames"], ch["advance"], ch["keep_head"], events,
                         before=int((ch.get("event_aware") or {}).get("before", 24)),
                         min_advance=int((ch.get("event_aware") or {}).get("min_advance", 48)))
    if ch.get("event_aware"):
        (out / "chunk-plan-events.json").write_text(json.dumps(
            {"arm_events": events, "moved_starts": chunks[0].get("moved_starts", []),
             "starts": [c["start"] for c in chunks]}, indent=1))
    # jobs: base prompt, chunk clips, control inputs, chunk-local prompts
    prompt = out / "contract" / "prompt.txt"
    lowering = recipe.get("prompt_lowering")
    if lowering:
        prompt = out / "contract" / f"prompt-{lowering}.txt"
        prompt.write_text(lower_h3_prompt(contract, lowering, assets=h3_paths.REPO))
    seed = int(recipe.get("seed_base", 1000)) + int(args.seed_offset)
    steps = int(recipe["denoise_forwards"])
    jobs = build_chunk_jobs(out, bundle["task"], ep["id"], contract, canvas, chunks, render_total, seed, steps,
                            str(prompt))
    masks, heds = add_control_inputs(jobs, ctl, inputs, args)
    keep = None
    if ep.get("keep_regions"):   # real pixels kept where the white shows a task state (h3_keep_regions.py)
        keep = h3_keep_regions.apply(jobs, ep["keep_regions"], h3_paths.run_dir() / "cache" / "keep")
    if lowering:
        add_chunk_prompts(jobs, contract, lowering, out / "contract",
                          recipe.get("chunk_facts_wrist", CHUNK_FACTS_DETECTORS[0]))
    manifest = {"schema": MANIFEST_SCHEMA, "backend": "control-hybrid", "controlnet": True, "prompt": str(prompt),
                "reference_images": [r["path"] for r in contract["intent"]["real_references"]],
                "reference_order": "images-first", "control": ctl, "denoise_forwards": steps, "canvas": CANVAS,
                "jobs": jobs}
    if lowering:
        manifest["prompt_lowering"] = lowering
    (out / "job-manifest.json").write_text(json.dumps(manifest, indent=2))
    policy = audit(out)
    render_root = out / "render"
    package = {"schema": "h3.package/1", "backend": "control-hybrid", "task": bundle["task"], "episode": ep["id"],
               "mode": args.mode, "variant": args.variant, "recipe": recipe, "recipe_path": str(recipe_path),
               "bundle": str(Path(args.bundle).resolve()), "carrier": str(carrier), "carrier_sha256": sha256(carrier),
               "carrier_fps": bundle["carrier_fps"], "frame_range": fr, "frames": total, "canvas": CANVAS,
               "seed_offset": int(args.seed_offset), "chunks": chunks, "canvas_carrier": str(canvas),
               "tail_pad": tail_pad, "render_frames": render_total,
               "contract": str(out / "contract" / "contract.json"), "prompt_sha256": contract["prompt_sha256"],
               "render_root": str(render_root), "spatial": SPATIAL.to_dict(), "control": ctl,
               "control_inputs": {"masks": list(masks.values()), "hed": heds, "jobs": len(jobs)},
               "build_seconds": round(time.time() - t0, 1)}
    if keep:
        package["keep_regions"] = keep
    if ep.get("wide_wrist"):
        package["wide_wrist"] = ep["wide_wrist"]
    (out / "package.json").write_text(json.dumps(package, indent=2, ensure_ascii=False))
    ticket = {"package": str(out), "manifest": str(out / "job-manifest.json"), "contract": package["contract"],
              "output_root": str(render_root), "profile": recipe["worker_profile"]}
    (out / "ticket.json").write_text(json.dumps(ticket, indent=2))
    print(json.dumps({"package": str(out), "chunks": len(jobs), "frames": total, "profile": recipe["worker_profile"],
                      "prompt_sha256": policy["prompt_sha256"],
                      "hed_active_fraction": [round(h["mean_active_fraction_gt20"] or 0, 4) for h in heds],
                      "build_seconds": package["build_seconds"]}))
    return 0


def cmd_check(args) -> int:
    print(json.dumps(audit(Path(args.package).resolve()), indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bundle", required=True)
    b.add_argument("--recipe", required=True)
    b.add_argument("--episode", required=True)
    b.add_argument("--mode", choices=["sim2real", "ood"], required=True)
    b.add_argument("--variant")
    b.add_argument("--seed-offset", type=int, default=0)
    b.add_argument("--out", required=True)
    b.add_argument("--force", action="store_true")
    b.add_argument("--hed-jobs", type=int, default=4, help="chunks whose HED runs in parallel")
    b.add_argument("--hed-threads", type=int, default=16, help="CPU threads per HED process")
    b.add_argument("--hed-device", choices=["cpu", "cuda"], default="cpu",
                   help="cpu outside a GPU lease (default); cuda only inside a scheduler GPU lease")
    h3_paths.add_run_dir_arg(b)
    c = sub.add_parser("check")
    c.add_argument("package")
    args = ap.parse_args()
    if getattr(args, "run_dir", None) is not None:
        h3_paths.set_run_dir(args.run_dir)
    return {"build": cmd_build, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
