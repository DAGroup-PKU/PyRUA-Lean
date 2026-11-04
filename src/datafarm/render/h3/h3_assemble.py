#!/usr/bin/env python3
"""Assemble one rendered H3 package into an episode video.

Chunk outputs -> one episode on the H3 canvas -> active crop -> 1280x1080 T-layout at the carrier's fps
(frame i == carrier frame frame_range[0]+i), plus the matching carrier clip, so the frozen evaluator can score it 1:1.

  python h3_assemble.py <package dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from h3_media import nframes, read_frames, sha256
from h3_spatial import SPATIAL

OUTPUT_NAME = "chunk-render.mp4"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _gray_frames(path: Path, lo: int, hi: int) -> dict:
    import cv2
    import numpy as np

    def gray(frame):
        small = cv2.resize(frame, (256, 192), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return read_frames(path, lo, hi, transform=gray)


def seam_keep_ranges(jobs: list, render: Path, total: int, guard: int = 4, min_lead: int = 0) -> list:
    """Min-diff seams: within each chunk overlap pick the global frame c where the outgoing and incoming chunk
    renders differ least (mean |A-B| on gray 256x192 + mean |dA-dB| motion term), cut there.  Hard cut, no blending."""
    import numpy as np
    ranges, keep_start = [], 0
    for k, job in enumerate(jobs):
        s_k = job["source_frame_range"][0]
        if k == len(jobs) - 1:
            ranges.append((keep_start, total, None))
            break
        nxt = jobs[k + 1]
        s_n = nxt["source_frame_range"][0]
        e_k = min(job["source_frame_range"][1], total)
        lo, hi = max(s_n + guard, keep_start + guard), e_k - guard
        lo = max(lo, s_n + min_lead)  # the incoming chunk must have generated >= min_lead frames of its own lead-in
        if hi <= lo:
            c = job["keep_global_range"][1]
            cost = None
        else:
            fa = _gray_frames(render / "chunks" / job["chunk_id"] / OUTPUT_NAME, lo - 1 - s_k, hi + 1 - s_k)
            fb = _gray_frames(render / "chunks" / nxt["chunk_id"] / OUTPUT_NAME, lo - 1 - s_n, hi + 1 - s_n)
            best = (None, 1e18)
            for c in range(lo, hi):
                a, b = fa[c - s_k], fb[c - s_n]
                da, db = a - fa[c - 1 - s_k], fb[c + 1 - s_n] - b
                cost = float(np.abs(a - b).mean() + np.abs(da - db).mean())
                if cost < best[1]:
                    best = (c, cost)
            c, cost = best
        ranges.append((keep_start, c, cost))
        keep_start = c
    return ranges


def assemble_canvas(pkg_dir: Path, work: Path, seam: str, min_lead: int) -> Path:
    manifest = json.loads((pkg_dir / "job-manifest.json").read_text())
    render = pkg_dir / "render"
    if seam == "min-diff":
        total = max(j["keep_global_range"][1] for j in manifest["jobs"])
        ranges = seam_keep_ranges(manifest["jobs"], render, total, min_lead=min_lead)
        (work / "t3-seams.json").write_text(json.dumps(
            [{"chunk": j["chunk_id"], "keep_global": [a, b], "cut_cost": c}
             for j, (a, b, c) in zip(manifest["jobs"], ranges)], indent=1))
        for j, (a, b, _) in zip(manifest["jobs"], ranges):
            s_k = j["source_frame_range"][0]
            j["keep_local_range"] = [a - s_k, b - s_k]
    parts = []
    for job in manifest["jobs"]:
        src = render / "chunks" / job["chunk_id"] / OUTPUT_NAME
        if not src.is_file():
            raise SystemExit(f"missing chunk output {src}")
        a, b = job["keep_local_range"]
        part = work / f"t3-part-{job['chunk_index']:02d}.mkv"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vf",
             f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS", "-an", "-fps_mode", "passthrough",
             "-c:v", "ffv1", str(part)])
        if nframes(part) != b - a:
            raise SystemExit(f"{part}: {nframes(part)} != {b - a}")
        parts.append(part)
    listing = work / "t3-parts.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    canvas = work / "t3-canvas-24fps.mkv"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(canvas)])
    return canvas


def assemble(pkg_dir: Path) -> dict:
    pkg_dir = Path(pkg_dir).resolve()
    pkg = json.loads((pkg_dir / "package.json").read_text())
    work = pkg_dir / "assembled"
    work.mkdir(exist_ok=True)
    fps = int(pkg["carrier_fps"])
    final = pkg_dir / f"final-t3-{fps}fps.mp4"
    assembly = pkg["recipe"].get("assembly") or {}
    canvas = assemble_canvas(pkg_dir, work, assembly.get("seam", "fixed"), int(assembly.get("min_lead", 0)))
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(canvas), "-vf",
         f"{SPATIAL.ffmpeg_canvas_to_active()},setpts=N/({fps}*TB)", "-fps_mode", "cfr", "-r", str(fps), "-an",
         "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p", str(final)])
    carrier_clip = pkg_dir / f"carrier-t3-{fps}fps.mp4"
    fr = pkg["frame_range"]
    if not carrier_clip.exists():
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", pkg["carrier"], "-vf",
             f"trim=start_frame={fr[0]}:end_frame={fr[1]},setpts=PTS-STARTPTS,scale=1280:1080", "-fps_mode",
             "passthrough", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p",
             str(carrier_clip)])
    n_final, n_car = nframes(final), nframes(carrier_clip)
    if n_final != pkg["frames"] or n_car != pkg["frames"]:
        raise SystemExit(f"frame mismatch final={n_final} carrier={n_car} expected={pkg['frames']}")
    record = {"final": str(final), "final_sha256": sha256(final), "carrier_clip": str(carrier_clip),
              "frames": n_final, "fps": fps}
    (pkg_dir / "assembled.json").write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("package", type=Path)
    print(json.dumps(assemble(ap.parse_args().package)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
