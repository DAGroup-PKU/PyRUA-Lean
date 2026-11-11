#!/usr/bin/env python3
"""Crop rendered wide-FOV wrist views back to the white episode's own wrist camera model.

With a wide wrist white (datafarm.simulation.white_replay wrist_fov_factor F < 1: same camera pose and principal point,
focal length x F) H3 sees more of the scene around the gripper and renders the wrist views with real context; this
step crops both wrist panels of the final T-layout video back by F around each wrist camera's principal point and
scales them to 640x360, so the output again has the white's own wrist intrinsics.  The carrier clip is cropped the same
way (frame i still corresponds) and the result is rescored with EVAL-v1.

The bundle episode records `wide_wrist: {"factor": F, "principal_points": {"left": [cx, cy], "right": [cx, cy]}}`
(pixels of the 1280x720 wrist camera, from the replay report); `h3_package.py build` copies it into package.json and
`h3_finalize.py` runs this step after the exposure lock.

  python h3_wrist_crop.py --package <exposure-locked package> [--run-dir <run>]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h3_paths
from h3_media import link_or_copy, nframes, sha256

PANEL_W, PANEL_H = 640, 360
PANELS = (("left", 0), ("right", 640))


def crop_window(factor: float, pp, cam_w: int = 1280, cam_h: int = 720) -> tuple[float, float, float, float]:
    """(x0, y0, w, h) in a 640x360 panel: the wide view scaled by `factor` about the camera's principal point, i.e. the
    part of the wide image the original camera saw, clamped inside the panel."""
    if not 0.3 <= float(factor) < 1.0:
        raise ValueError(f"wide wrist factor {factor} outside [0.3, 1)")
    sx, sy = PANEL_W / cam_w, PANEL_H / cam_h
    cx, cy = float(pp[0]) * sx, float(pp[1]) * sy
    w, h = PANEL_W * factor, PANEL_H * factor
    x0, y0 = cx * (1 - factor), cy * (1 - factor)
    return min(max(0.0, x0), PANEL_W - w), min(max(0.0, y0), PANEL_H - h), w, h


def filter_graph(factor: float, pps: dict) -> str:
    parts = []
    for i, (side, x) in enumerate(PANELS):
        x0, y0, w, h = crop_window(factor, pps[side])
        parts.append(f"[0:v]crop={PANEL_W}:{PANEL_H}:{x}:720,crop={w:.3f}:{h:.3f}:{x0:.3f}:{y0:.3f},"
                     f"scale={PANEL_W}:{PANEL_H}:flags=lanczos,setsar=1[w{i}]")
    return ";".join(parts) + ";[0:v]crop=1280:720:0:0[hd];[w0][w1]hstack=inputs=2[b];[hd][b]vstack=inputs=2[o]"


def crop_back(src: Path, run: Path) -> Path | None:
    """<run>/pkgs/<src recipe>-wristcrop/<task>/<tag> from an assembled package whose package.json has wide_wrist;
    None when it has none."""
    src = Path(src).resolve()
    pkg = json.loads((src / "package.json").read_text())
    ww = pkg.get("wide_wrist")
    if not ww:
        return None
    asm = json.loads((src / "assembled.json").read_text())
    fps = int(asm["fps"])
    dst = run / "pkgs" / f"{src.parts[-3]}-wristcrop" / src.parts[-2] / src.name
    dst.mkdir(parents=True, exist_ok=True)
    fc = filter_graph(float(ww["factor"]), ww["principal_points"])
    out = {}
    for key, name in (("final", f"final-t3-{fps}fps.mp4"), ("carrier_clip", f"carrier-t3-{fps}fps.mp4")):
        target = dst / name
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", asm[key], "-filter_complex", fc,
                        "-map", "[o]", "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p", "-an",
                        str(target)], check=True)
        if nframes(target) != int(asm["frames"]):
            raise SystemExit(f"{target}: {nframes(target)} frames, expected {asm['frames']}")
        out[key] = target
    for name in ("job-manifest.json", "exposure-lock.json"):
        if (src / name).exists() and not (dst / name).exists():
            link_or_copy(src / name, dst / name)
    if not (dst / "render").exists() and (src / "render").exists():
        (dst / "render").symlink_to((src / "render").resolve())
    pkg["recipe"] = {**pkg["recipe"], "name": f"{src.parts[-3]}-wristcrop", "source_recipe": pkg["recipe"].get("name")}
    pkg["derived_from"] = [str(src)]
    pkg["wrist_crop_back"] = {"factor": float(ww["factor"]), "principal_points": ww["principal_points"],
                              "filter": fc}
    (dst / "package.json").write_text(json.dumps(pkg, indent=2, ensure_ascii=False))
    (dst / "assembled.json").write_text(json.dumps({"final": str(out["final"]), "final_sha256": sha256(out["final"]),
                                                    "carrier_clip": str(out["carrier_clip"]),
                                                    "frames": int(asm["frames"]), "fps": fps}, indent=2))
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", required=True, type=Path)
    h3_paths.add_run_dir_arg(ap)
    args = ap.parse_args()
    run = h3_paths.run_dir(args.run_dir)
    dst = crop_back((run / args.package).resolve(), run)
    if dst is None:
        raise SystemExit("package.json has no wide_wrist record: nothing to crop back")
    print(json.dumps({"package": str(dst)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
