#!/usr/bin/env python3
"""Fill the lit displays of a white carrier with a flat backlight colour.

    python solar_carrier_displays.py IN_T3.mp4 OUT_T3.mp4 [--bgr 255,178,92]

The simulator draws a display black when off and plain white when lit, which Solar renders as a dim grey panel. This
fills every lit-display component (solar_canvas.lit_display_mask, with its 3 px rim) with one flat colour, in all three
views, so the prompt can name the lit state by the colour the carrier shows. Nothing else changes. On its own this does
not make Solar light a display mid-window (a window keeps its first-frame look): pair it with a window hand-over just
after the display lights (solar_chain.py --wins / --handover). Prints frames and frames with a lit display as JSON.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solar_canvas import fill_lit_displays  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bgr", default="255,178,92")
    a = ap.parse_args(argv)
    bgr = tuple(int(x) for x in a.bgr.split(","))
    cap = cv2.VideoCapture(a.src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    enc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "10", "-pix_fmt", "yuv420p", a.dst], stdin=subprocess.PIPE)
    n = lit = 0
    first_lit = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f, k = fill_lit_displays(f, bgr)
        if k and first_lit is None:
            first_lit = n
        lit += bool(k)
        n += 1
        enc.stdin.write(f.tobytes())
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")
    print(json.dumps({"src": a.src, "dst": a.dst, "frames": n, "frames_with_lit_display": lit, "first_lit_frame": first_lit,
                      "bgr": bgr}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
