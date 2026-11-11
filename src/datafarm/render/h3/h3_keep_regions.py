#!/usr/bin/env python3
"""Keep regions: task states the video model cannot infer from the white are kept from real pixels, not regenerated.

Some task states are visible only as a small detail the generator does not follow from the white carrier, e.g. a
kitchen scale's display that the task switches on: H3 renders it lit too early, too late or not at all.  A keep-region
spec (a task product, schema h3.keep-regions/1) declares

  states    render-only state colours for the white replay ("@states" of datafarm.simulation.white_replay), so the
            white carrier shows the state as a distinctive near-white rectangle exactly in the frames it holds;
  regions   per region: how to find it in a white chunk frame (`white`: a bright, unsaturated, wide rectangle) and
            which real pixels to keep there (`texture`: the largest component matching `rule` inside `box` of real
            head frame `frame` of `source`, a real video whose sha256 is `source_sha256`).

Wherever a chunk's white shows the region, the real texture is written into that chunk's inpaint video and the repaint
mask is 0 (keep) there; everything else is untouched.  `h3_package.py build` applies it when the bundle episode names
`keep_regions`, before the manifest is written and audited, so no worker can render an unpatched chunk.  Patched inputs
are cached by (spec sha256, white chunk sha256) under <run>/cache/keep, so every seed of an episode shares them.

Measured (weigh-apple, 2026-11-09/10): the display was lit in 100 % of the frames the white shows it lit, and the
render passed independent blind review 4/4/5/4; the kept display renders whiter than the real one and loses its digits.

  python h3_keep_regions.py check <spec.json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from h3_media import read_all, sha256

SPEC_SCHEMA = "h3.keep-regions/1"
WHITE_DEFAULTS = {"v_min": 245, "s_max": 40, "area": [100, 6000], "min_h": 5, "aspect_min": 1.6, "fill_min": 0.6}


def validate(spec: dict) -> list[str]:
    problems = []
    if spec.get("schema") != SPEC_SCHEMA:
        problems.append(f"schema must be {SPEC_SCHEMA}")
    regions = spec.get("regions")
    if not isinstance(regions, list) or not regions:
        problems.append("regions must be a non-empty list")
        regions = []
    names = set()
    for k, r in enumerate(regions):
        name = r.get("name") if isinstance(r, dict) else None
        if not name or name in names:
            problems.append(f"regions[{k}]: a unique name is required")
        names.add(name)
        t = (r or {}).get("texture") or {}
        for key in ("source", "source_sha256", "frame", "box"):
            if key not in t:
                problems.append(f"regions[{k}].texture needs {key}")
        if "box" in t and not (isinstance(t["box"], list) and len(t["box"]) == 4 and t["box"][0] < t["box"][2]
                               and t["box"][1] < t["box"][3]):
            problems.append(f"regions[{k}].texture.box must be [x0, y0, x1, y1]")
        unknown = set((r or {}).get("white") or {}) - set(WHITE_DEFAULTS)
        if unknown:
            problems.append(f"regions[{k}].white: unknown keys {sorted(unknown)}")
    if not isinstance(spec.get("states", {}), dict):
        problems.append("states must map entity names to white-replay @states rules")
    return problems


def load(path) -> dict:
    spec = json.loads(Path(path).read_text())
    problems = validate(spec)
    if problems:
        raise SystemExit(f"invalid keep-region spec {path}:\n" + "\n".join(problems))
    return spec


def texture(region: dict, out_dir: Path):
    """Real texture of a region (BGR array), cached as <out_dir>/keep-<name>-texture.png."""
    import cv2
    import numpy as np
    t = region["texture"]
    out = Path(out_dir) / f"keep-{region['name']}-texture.png"
    if out.is_file():
        return cv2.imread(str(out))
    if sha256(t["source"]) != t["source_sha256"]:
        raise SystemExit(f"keep {region['name']}: {t['source']} is not the real video the texture frame and box were "
                         "chosen on (sha256 mismatch); re-derive the spec for this real episode")
    cap = cv2.VideoCapture(str(t["source"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t["frame"]))
    ok, rf = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"keep {region['name']}: cannot read real frame {t['frame']}")
    x0, y0, x1, y1 = t["box"]
    crop = rf[y0:y1, x0:x1]
    b, r = crop[..., 0].astype(int), crop[..., 2].astype(int)
    rule = t.get("rule") or {}
    m = ((b - r > int(rule.get("b_minus_r", 30))) & (b > int(rule.get("b_min", 150)))).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        raise SystemExit(f"keep {region['name']}: no texture pixels in real frame {t['frame']} box {t['box']}")
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    lx, ly, lw, lh = st[k][:4]
    if lw * lh < 200:
        raise SystemExit(f"keep {region['name']}: texture too small ({lw}x{lh})")
    out.parent.mkdir(parents=True, exist_ok=True)
    tex = crop[ly:ly + lh, lx:lx + lw]
    cv2.imwrite(str(out), tex)
    return tex


def white_rects(frame, white: dict | None) -> list[tuple[int, int, int, int]]:
    """Rectangles (x, y, w, h) of a white frame where the replayed state shows: bright, unsaturated, wide, filled."""
    import cv2
    w = {**WHITE_DEFAULTS, **(white or {})}
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > int(w["v_min"])) & (hsv[..., 1] < int(w["s_max"]))).astype("uint8")
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    a0, a1 = w["area"]
    out = []
    for c in range(1, n):
        x, y, ww, hh, a = (int(v) for v in st[c])
        if a0 <= a <= a1 and hh >= int(w["min_h"]) and ww > float(w["aspect_min"]) * hh and a > float(w["fill_min"]) * ww * hh:
            out.append((x, y, ww, hh))
    return out


def encode(path: Path, frames, gray: bool = False, crf: int = 12) -> None:
    import numpy as np
    h, w = frames[0].shape[:2]
    tmp = Path(str(path) + ".tmp.mp4")
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                          "gray" if gray else "bgr24", "-s", f"{w}x{h}", "-r", "24", "-i", "-", "-an", "-c:v", "libx264",
                          "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        raise SystemExit(f"ffmpeg failed for {path}")
    tmp.replace(path)


def patch_job(job: dict, spec: dict, spec_sha: str, textures: dict, cache: Path) -> dict:
    """Patched inpaint video + repaint mask of one chunk job; returns the job fields to update."""
    import cv2
    key = f"{spec_sha[:16]}-{job['white_sha256'][:16]}"
    rec_path = cache / f"{key}.json"
    if rec_path.is_file():
        rec = json.loads(rec_path.read_text())
        if Path(rec["inpaint_video"]).is_file() and Path(rec["inpaint_mask"]).is_file():
            return rec
    white = read_all(job["white"])
    inp = read_all(job["inpaint_video"])
    mask = [cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) for m in read_all(job["inpaint_mask"])]
    n = int(job["generated_frames"])
    if not (len(white) >= n and len(inp) >= n and len(mask) >= n):
        raise SystemExit(f"keep {job['chunk_id']}: white/inpaint/mask have {len(white)}/{len(inp)}/{len(mask)} frames, "
                         f"need {n}")
    inp, mask = list(inp[:n]), mask[:n]
    kept = {r["name"]: 0 for r in spec["regions"]}
    for i in range(n):
        for r in spec["regions"]:
            rects = white_rects(white[i], r.get("white"))
            for (x, y, w, h) in rects:
                inp[i][y:y + h, x:x + w] = cv2.resize(textures[r["name"]], (w, h), interpolation=cv2.INTER_AREA)
                mask[i][max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1] = 0
            kept[r["name"]] += bool(rects)
    cache.mkdir(parents=True, exist_ok=True)
    iv, mv = cache / f"{key}-inpaint.mp4", cache / f"{key}-mask.mp4"
    encode(iv, inp)
    encode(mv, mask, gray=True, crf=0)
    rec = {"inpaint_video": str(iv), "inpaint_video_sha256": sha256(iv), "inpaint_mask": str(mv),
           "inpaint_mask_sha256": sha256(mv),
           "inpaint_regenerate_fraction": round(float(sum(m.mean() for m in mask) / len(mask) / 255.0), 6),
           "keep_regions": {"spec_sha256": spec_sha, "frames": kept, "chunk_frames": n}}
    rec_path.write_text(json.dumps(rec, indent=1))
    return rec


def apply(jobs: list, spec_path, cache: Path) -> dict:
    """Patch every chunk job in place.  Raises when the regions are found in no white frame at all (is the state
    replayed into the white?).  Returns {spec, spec_sha256, frames per region}."""
    spec_path = Path(spec_path).resolve()
    spec = load(spec_path)
    spec_sha = sha256(spec_path)
    textures = {r["name"]: texture(r, cache / "textures" / spec_sha[:16]) for r in spec["regions"]}
    total: dict = {}
    for job in jobs:
        rec = patch_job(job, spec, spec_sha, textures, cache)
        job.update(rec)
        for k, v in rec["keep_regions"]["frames"].items():
            total[k] = total.get(k, 0) + v
    if not any(total.values()):
        raise SystemExit(f"keep regions found nothing in any white chunk ({spec_path}); is the state colour replayed "
                         "into the white (spec states -> white_replay colors @states)?")
    return {"spec": str(spec_path), "spec_sha256": spec_sha, "frames": total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("spec")
    args = ap.parse_args()
    problems = validate(json.loads(Path(args.spec).read_text()))
    print(json.dumps({"spec": args.spec, "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
