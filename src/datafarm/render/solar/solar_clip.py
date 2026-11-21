#!/usr/bin/env python3
"""Build one SolarRenderer input clip from a 1280x1080 T-layout white carrier.

    python solar_clip.py --carrier T3.mp4 --start S --n N --out CLIP [--layout t3|head] [--pad17]

CLIP/structure.mp4: carrier frames S..S+N-1 written 1:1 with a 24 fps label (clip frame k <-> carrier frame S+k) on the
1344x768 canvas. Layout t3: the whole T layout in the 910x768 active box (solar_canvas). Layout head: the head panel
only, 1280x720 -> 1344x756 with 6 px bars top and bottom. --pad17 repeats the last frame up to the next 17k+5 length
(the padding is recorded and never written to the output). CLIP/clip.json records the mapping.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_canvas import ACTIVE_W, ACTIVE_X0, CANVAS_H, CANVAS_W, SOLAR_FPS, pad_to_legal, t3_to_canvas  # noqa: E402


def head_to_canvas(t3: np.ndarray) -> np.ndarray:
    c = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    c[6:762] = cv2.resize(t3[:720], (CANVAS_W, 756), interpolation=cv2.INTER_CUBIC)
    return c


def write_clip(carrier: str, start: int, n: int, out: Path, layout: str = "t3", pad17: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(carrier)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(SOLAR_FPS), "-i", "-", "-c:v", "libx264", "-crf", "10",
                           "-preset", "slow", "-pix_fmt", "yuv420p", str(out / "structure.mp4")], stdin=subprocess.PIPE)
    got, fr = 0, None
    for k in range(n):
        ok, im = cap.read()
        if not ok:
            break
        fr = t3_to_canvas(im) if layout == "t3" else head_to_canvas(im)
        if k == 0:
            cv2.imwrite(str(out / "structure-f0.png"), fr)
        ff.stdin.write(fr.tobytes())
        got += 1
    if fr is None:
        ff.stdin.close()
        ff.wait()
        raise SystemExit(f"{carrier}: no frame at {start}")
    padded = (pad_to_legal(got) - got) if pad17 else 0
    for _ in range(padded):
        ff.stdin.write(fr.tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("solar_clip: ffmpeg failed")
    meta = {"carrier": carrier, "start": start, "frames": got, "padded_frames": padded, "layout": layout,
            "canvas": [CANVAS_W, CANVAS_H], "fps_label": SOLAR_FPS, "mapping": "clip frame k <-> carrier frame start+k",
            "active_box_xywh": [ACTIVE_X0, 0, ACTIVE_W, CANVAS_H] if layout == "t3" else [0, 6, CANVAS_W, 756]}
    (out / "clip.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--carrier", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--layout", choices=["t3", "head"], default="t3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad17", action="store_true")
    a = ap.parse_args(argv)
    meta = write_clip(a.carrier, a.start, a.n, Path(a.out), a.layout, a.pad17)
    print(a.out, meta["frames"], "padded", meta["padded_frames"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
