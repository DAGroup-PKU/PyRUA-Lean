#!/usr/bin/env python3
"""White carrier preparation and checks for the H3 control hybrid.

  compose  three synchronized white views (head 1280x720, wrists 640x360) -> one 1280x1080 T-layout carrier at the
           views' own rate (24 or 30 fps).  Views may be videos or the replay's ffmpeg concat lists of PNG frames
           (`<camera>.concat`, rewritten from the attempt to the replay directory).  A white below MIN_WHITE_FPS is
           refused: re-render it densely (datafarm.simulation.white_replay with image_stride), never interpolate it.
           Motion interpolation warps and drags fast-moving objects, and H3 copies that drift into the output.
  check    measurements of a carrier that predict H3 failures, reported as warnings by `h3_bundle.py check`:
           frame rate, and whether the head view has a room seam (wall/table edge) the control can see.  The
           soft-HED control keeps only edges with a strong local luminance step (edge_gate tau 25): on a flat white
           scene where table and wall differ by a few grey levels the control has no table edge, and H3 invents a wall
           or table edge in the wrist views.  A shaded white (e.g. table 237, wall 186) keeps the seam.

  python h3_carrier.py compose --head cam_high.mp4 --left cam_left_wrist.mp4 --right cam_right_wrist.mp4 --out t3.mp4
  python h3_carrier.py compose --replay <replay dir> --attempt <attempt dir> --out t3.mp4
  python h3_carrier.py check <carrier.mp4>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from h3_media import probe

MIN_WHITE_FPS = 20.0
CARRIER_FPS = (24, 30)
CAMERAS = {"head": "cam_high", "left": "cam_left_wrist", "right": "cam_right_wrist"}
# room-seam check: the same luma-step gate as the default control (h3-control-hybrid-v1.json control.edge_gate)
GATE_TAU, GATE_WINDOW = 25.0, 7
SEAM_MIN_LENGTH = 0.25        # of the head-view width
SEAM_MAX_TILT_DEG = 35.0      # a wall/table seam or table edge, not a vertical robot or object edge


def fps_of(path) -> float:
    return float(Fraction(probe(path).rate))


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _image_rate(replay: Path, attempt: Path | None, cam: str) -> str | None:
    """Frame rate of a replay's PNG frames: its image-manifest.json (fps, or physics_hz / stride), else the attempt's
    own video of that camera (an undensified replay keeps the attempt's image rate)."""
    for d in (replay, attempt):
        man = Path(d) / "image-manifest.json" if d else None
        if man and man.is_file():
            m = json.loads(man.read_text())
            if m.get("fps"):
                return str(m["fps"])
            if m.get("physics_hz") and m.get("stride"):
                return str(Fraction(str(m["physics_hz"])) / int(m["stride"]))
        if d is attempt:
            break
    if attempt is not None and (Path(attempt) / f"{cam}.mp4").is_file():
        return probe(Path(attempt) / f"{cam}.mp4").rate
    return None


def replay_views(replay: Path, attempt: Path | None = None) -> dict:
    """{head, left, right} videos of a white replay: <camera>.mp4 in the replay directory, else the PNG frames listed in
    its <camera>.concat (or in the attempt's list, with paths moved to the replay directory), one frame per listed PNG
    in order, encoded at the replay's image rate."""
    replay = Path(replay)
    views = {}
    for side, cam in CAMERAS.items():
        mp4 = replay / f"{cam}.mp4"
        if mp4.is_file():
            views[side] = mp4
            continue
        names = (f"{cam}.concat", f"{cam}-frames.txt")   # ffmpeg concat lists of the captured PNG frames
        conc = next((replay / n for n in names if (replay / n).is_file()), None)
        if conc is None and attempt is not None:
            src = next((Path(attempt) / n for n in names if (Path(attempt) / n).is_file()), None)
            if src is not None:
                conc = replay / src.name
                conc.write_text(src.read_text().replace(str(Path(attempt)) + "/", str(replay) + "/"))
        if conc is None:
            raise SystemExit(f"replay {replay} has neither {cam}.mp4 nor a frame list ({' / '.join(names)})")
        pngs = [line.split("'")[1] for line in conc.read_text().splitlines() if line.startswith("file ")]
        missing = [p for p in pngs if not Path(p).is_file()]
        if not pngs or missing:
            raise SystemExit(f"{conc}: {len(missing)} of {len(pngs)} listed frames missing")
        rate = _image_rate(replay, attempt, cam)
        if rate is None:
            raise SystemExit(f"{conc}: frame rate unknown (no image-manifest.json and no original {cam}.mp4)")
        seq = replay / f"{cam}-seq"
        seq.mkdir(exist_ok=True)
        for old in seq.glob("*.png"):
            old.unlink()
        for i, p in enumerate(pngs):
            os.symlink(p, seq / f"{i:06d}.png")
        tmp = mp4.with_suffix(".tmp.mp4")
        ffmpeg("-framerate", str(rate), "-i", str(seq / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "12", str(tmp))
        os.replace(tmp, mp4)
        if probe(mp4).frames != len(pngs):
            raise SystemExit(f"{mp4}: {probe(mp4).frames} frames, {len(pngs)} listed")
        views[side] = mp4
    return views


def compose(views: dict, out: Path) -> dict:
    """T-layout carrier of three synchronized views; frame i of every view is carrier frame i (no retiming)."""
    rates = {side: fps_of(v) for side, v in views.items()}
    frames = {side: probe(v).frames for side, v in views.items()}
    if len(set(frames.values())) != 1:
        raise SystemExit(f"views have different frame counts {frames}")
    fps = rates["head"]
    if any(abs(r - fps) > 0.01 for r in rates.values()):
        raise SystemExit(f"views have different frame rates {rates}")
    if fps < MIN_WHITE_FPS:
        raise SystemExit(f"white at {fps:g} fps is below {MIN_WHITE_FPS:g}: re-render it densely (white_replay "
                         "image_stride); a motion-interpolated carrier makes H3 copy warped, drifting objects")
    rate = min(CARRIER_FPS, key=lambda r: abs(r - fps))
    if abs(rate - fps) > 0.01:
        raise SystemExit(f"white at {fps:g} fps: the carrier rate must be one of {CARRIER_FPS} without resampling")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    fc = ("[0:v]scale=1280:720:flags=lanczos,setsar=1[h];[1:v]scale=640:360:flags=lanczos,setsar=1[l];"
          "[2:v]scale=640:360:flags=lanczos,setsar=1[r];[l][r]hstack=inputs=2[b];[h][b]vstack=inputs=2,format=yuv420p[o]")
    ffmpeg("-i", str(views["head"]), "-i", str(views["left"]), "-i", str(views["right"]), "-filter_complex", fc,
           "-map", "[o]", "-r", str(rate), "-fps_mode", "cfr", "-an", "-c:v", "libx264", "-preset", "fast",
           "-crf", "10", str(tmp))
    os.replace(tmp, out)
    n = probe(out).frames
    if n != frames["head"]:
        raise SystemExit(f"composed carrier has {n} frames, views have {frames['head']}")
    meta = {"carrier": str(out), "fps": rate, "frames": n, "views": {k: str(v) for k, v in views.items()}}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def room_seams(head_bgr) -> list[dict]:
    """Long, roughly horizontal edges of a head-view frame that survive the control's luma-step gate, outside the dark
    robot.  Each: {length (of the view width), tilt_deg, step (grey levels across it)}."""
    import cv2
    import numpy as np
    h, w = head_bgr.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(head_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
    kern = np.ones((GATE_WINDOW, GATE_WINDOW), np.uint8)
    step = cv2.dilate(g, kern) - cv2.erode(g, kern)
    hsv = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2HSV)
    robot = cv2.dilate(((hsv[..., 2] < 110) & (hsv[..., 1] < 90)).astype(np.uint8), np.ones((15, 15), np.uint8))
    edges = cv2.Canny(g.astype(np.uint8), 8, 24)
    edges[(step <= GATE_TAU) | (robot > 0)] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40, minLineLength=int(SEAM_MIN_LENGTH * w), maxLineGap=12)
    out = []
    for a, b, c, d in ([] if lines is None else lines[:, 0].tolist()):
        tilt = abs(math.degrees(math.atan2(d - b, c - a)))
        tilt = min(tilt, 180 - tilt)
        if tilt <= SEAM_MAX_TILT_DEG:
            out.append({"length": round(math.hypot(c - a, d - b) / w, 3), "tilt_deg": round(tilt, 1),
                        "step": round(float(step[(b + d) // 2, (a + c) // 2]), 1)})
    return sorted(out, key=lambda s: -s["length"])


def check(carrier: Path, samples: int = 5) -> dict:
    """Carrier measurements and warnings (not a gate: a camera that never sees a wall or table edge is legitimate)."""
    import cv2
    info = probe(carrier)
    fps = float(Fraction(info.rate))
    warnings = []
    if fps < MIN_WHITE_FPS:
        warnings.append(f"carrier at {fps:g} fps: re-render the white densely (white_replay image_stride)")
    cap = cv2.VideoCapture(str(carrier))
    seams = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * max(1, info.frames - 1) / max(1, samples - 1)))
        ok, f = cap.read()
        if not ok:
            continue
        head = cv2.resize(f, (1280, 1080))[:720]
        seams.append(room_seams(head))
    cap.release()
    with_seam = sum(1 for s in seams if s)
    if seams and with_seam == 0:
        warnings.append("head view has no wall/table seam with a luminance step above the control gate (tau 25): "
                        "H3 tends to invent a wall or table edge; shade the white scene (distinct table and wall "
                        "brightness) or dress it with white_replay extras")
    return {"carrier": str(carrier), "fps": fps, "frames": info.frames, "sampled_frames": len(seams),
            "frames_with_room_seam": with_seam, "longest_seam": max((s[0] for s in seams if s),
                                                                    key=lambda x: x["length"], default=None),
            "warnings": warnings}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compose")
    c.add_argument("--head")
    c.add_argument("--left")
    c.add_argument("--right")
    c.add_argument("--replay", type=Path, help="white replay directory (datafarm.simulation.white_replay OUT_DIR)")
    c.add_argument("--attempt", type=Path, help="the replayed attempt (for its concat lists and frame rate)")
    c.add_argument("--out", required=True, type=Path)
    k = sub.add_parser("check")
    k.add_argument("carrier", type=Path)
    args = ap.parse_args()
    if args.cmd == "check":
        rep = check(args.carrier)
        print(json.dumps(rep, indent=1))
        return 0
    if args.replay:
        views = replay_views(args.replay, args.attempt)
    elif args.head and args.left and args.right:
        views = {"head": Path(args.head), "left": Path(args.left), "right": Path(args.right)}
    else:
        ap.error("give --head/--left/--right or --replay")
    print(json.dumps(compose(views, args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
