#!/usr/bin/env python3
"""First frame of a SolarRenderer chain: the real episode's frame 0 in the carrier's T layout.

    python solar_first_frame.py --head HEAD.mp4|png [--left LEFT.mp4|png] [--right RIGHT.mp4|png] --carrier T3.mp4 --out DIR

SolarRenderer takes the appearance of every object from the first frame and largely ignores colour in the structure
video, so the first frame must be a photograph of the same scene from the same cameras: the real frame 0 when the
simulation was fitted to it. Writes DIR/real-f0-t3.png (1280x1080: head 1280x720 on top, left and right wrist 640x360
below), DIR/first-canvas.png (the same on the 1344x768 Solar canvas) and DIR/first.json. A wrist view missing from the
input is taken from the carrier's frame 0 and recorded in first.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_canvas import T3_H, T3_W, t3_to_canvas  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def frame0(path: str, size: tuple[int, int]) -> np.ndarray:
    if Path(path).suffix.lower() in IMAGE_EXT:
        im = cv2.imread(path)
    else:
        ok, im = cv2.VideoCapture(path).read()
        im = im if ok else None
    if im is None:
        raise SystemExit(f"cannot read frame 0 of {path}")
    return cv2.resize(im, size, interpolation=cv2.INTER_AREA)


def build(head: str, carrier: str, left: str | None = None, right: str | None = None) -> tuple[np.ndarray, list]:
    t3 = np.zeros((T3_H, T3_W, 3), np.uint8)
    t3[:720] = frame0(head, (T3_W, 720))
    from_carrier, car0 = [], None
    for key, src, x in (("left", left, 0), ("right", right, 640)):
        if src:
            t3[720:, x:x + 640] = frame0(src, (640, 360))
        else:
            if car0 is None:
                car0 = frame0(carrier, (T3_W, T3_H))
            t3[720:, x:x + 640] = car0[720:, x:x + 640]
            from_carrier.append(key)
    return t3, from_carrier


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--head", required=True)
    ap.add_argument("--left")
    ap.add_argument("--right")
    ap.add_argument("--carrier", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t3, from_carrier = build(a.head, a.carrier, a.left, a.right)
    cv2.imwrite(str(out / "real-f0-t3.png"), t3)
    cv2.imwrite(str(out / "first-canvas.png"), t3_to_canvas(t3))
    meta = {"real_t3": str(out / "real-f0-t3.png"), "canvas": str(out / "first-canvas.png"),
            "source": {"head": a.head, "left": a.left, "right": a.right}, "wrist_from_carrier": from_carrier}
    (out / "first.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
