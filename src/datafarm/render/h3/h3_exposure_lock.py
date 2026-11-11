#!/usr/bin/env python3
"""First-chunk exposure lock of every chunk output before assembly (derived, no GPU).

Chunks render independently, so every seam can show an exposure step.  This tool applies a frame-wise per panel/channel
gain to every chunk's full output, toward the first chunk's exposure (median output / white-carrier brightness ratio)
at the frame's own carrier brightness, then assembles with the package's recorded seams (assembled/t3-seams.json) and
the same crop/scale as h3_assemble.py.  Gains only: no pixel is moved, blended with another frame or retimed.

  python h3_exposure_lock.py --src <run>/pkgs/<recipe>/<task>/<tag> --recipe <run>/recipes/exposure-lock-<recipe>.json
  (output: <run>/pkgs/<lock recipe>/<task>/<tag>; run directory = --run-dir or env H3_RUN)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import h3_paths
from h3_media import link_or_copy, read_all, sha256
from h3_spatial import SPATIAL

OUTPUT_NAME = "chunk-render.mp4"
CANVAS_PANELS = SPATIAL.canvas_views()


def lock(frames: np.ndarray, white: np.ndarray, ratios: dict) -> tuple[np.ndarray, dict]:
    out = frames.astype(np.float32)
    report = {}
    for name, (y0, y1, x0, x1) in CANVAS_PANELS.items():
        wm = white[:, y0:y1, x0:x1].mean(axis=(1, 2, 3)).astype(np.float64)
        gm = frames[:, y0:y1, x0:x1].reshape(len(frames), -1, 3).mean(1).astype(np.float64)
        gains = []
        for c in range(3):
            g = np.clip(ratios[name][c] * wm / (gm[:, c] + 1e-3), 0.7, 1.45)
            out[:, y0:y1, x0:x1, c] *= g[:, None, None]
            gains.append([float(g.min()), float(g.max())])
        report[name] = gains
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), report


def lock_package(src: Path, recipe: dict, run: Path) -> Path:
    src = Path(src).resolve()
    pkg = json.loads((src / "package.json").read_text())
    manifest = json.loads((src / "job-manifest.json").read_text())
    seams = json.loads((src / "assembled" / "t3-seams.json").read_text())
    assembled = json.loads((src / "assembled.json").read_text())
    dst = run / "pkgs" / recipe["name"] / src.parts[-2] / src.name
    dst.mkdir(parents=True, exist_ok=True)
    render = src / "render" / "chunks"
    jobs = sorted(manifest["jobs"], key=lambda j: int(j["chunk_index"]))
    reference = read_all(render / jobs[0]["chunk_id"] / OUTPUT_NAME)
    reference_white = read_all(jobs[0]["white"])[:len(reference)]
    ratios = {}
    for name, (y0, y1, x0, x1) in CANVAS_PANELS.items():
        wm = reference_white[:, y0:y1, x0:x1].mean(axis=(1, 2, 3)).astype(np.float64)
        gm = reference[:, y0:y1, x0:x1].reshape(len(reference), -1, 3).mean(1).astype(np.float64)
        ratios[name] = [float(np.median(gm[:, c] / np.maximum(wm, 1.0))) for c in range(3)]
    parts, report = [], {}
    for job, seam in zip(jobs, seams):
        if seam["chunk"] != job["chunk_id"]:
            raise SystemExit(f"seam {seam['chunk']} does not match chunk {job['chunk_id']}")
        video = read_all(render / job["chunk_id"] / OUTPUT_NAME)
        white = read_all(job["white"])[:len(video)]
        locked, report[job["chunk_id"]] = lock(video, white, ratios)
        start = int(job["source_frame_range"][0])
        a, b = seam["keep_global"]
        parts.append(locked[a - start:b - start])
    canvas = np.concatenate(parts)
    pad = int(pkg.get("tail_pad", 0))
    if pad:   # tail-padded package: the seams cover the padded carrier; keep the episode's own frames only
        if len(canvas) != int(assembled["frames"]) + pad:
            raise SystemExit(f"padded canvas has {len(canvas)} frames, expected {assembled['frames']} + {pad}")
        canvas = canvas[:int(assembled["frames"])]
    fps = int(assembled["fps"])
    final = dst / f"final-t3-{fps}fps.mp4"
    tmp = dst / "final.partial.mp4"
    enc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{canvas.shape[2]}x{canvas.shape[1]}", "-r", str(fps), "-i", "-", "-an", "-vf",
                            SPATIAL.ffmpeg_canvas_to_active(), "-c:v", "libx264", "-preset", "fast", "-crf", "14",
                            "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    enc.stdin.write(np.ascontiguousarray(canvas).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")
    os.replace(tmp, final)
    if len(canvas) != int(assembled["frames"]):
        raise SystemExit(f"frame mismatch {len(canvas)} != {assembled['frames']}")
    carrier = dst / f"carrier-t3-{fps}fps.mp4"
    link_or_copy(assembled["carrier_clip"], carrier)
    (dst / "job-manifest.json").write_text((src / "job-manifest.json").read_text())
    if not (dst / "render").exists():
        (dst / "render").symlink_to(src / "render")
    (dst / "assembled.json").write_text(json.dumps({"final": str(final), "final_sha256": sha256(final),
                                                    "carrier_clip": str(carrier), "frames": len(canvas), "fps": fps},
                                                   indent=2))
    (dst / "exposure-lock.json").write_text(json.dumps({
        "schema_version": "h3.chunk-photometric-lock/1",
        "method": "per chunk, per panel/channel frame-wise gain toward the FIRST chunk's exposure gain (median output / "
                  "white-carrier brightness ratio), then hard-cut assembly at the recorded seams",
        "first_chunk_ratios": ratios, "gains": report, "seams": seams, "source_final_sha256": assembled["final_sha256"],
        "geometry_changed": False, "retimed": False}, indent=2))
    pkg["recipe"] = {**recipe, "source_recipe": pkg["recipe"]["name"]}
    pkg["derived_from"] = [str(src)]
    (dst / "package.json").write_text(json.dumps(pkg, indent=2, ensure_ascii=False))
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--recipe", required=True)
    h3_paths.add_run_dir_arg(ap)
    args = ap.parse_args()
    recipe = json.loads(Path(args.recipe).read_text())
    print(json.dumps({"dst": str(lock_package(args.src, recipe, h3_paths.run_dir(args.run_dir)))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
