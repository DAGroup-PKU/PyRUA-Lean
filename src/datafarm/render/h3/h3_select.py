#!/usr/bin/env python3
"""Per-chunk seed selection (assembly-only, no GPU): a screening model's choice.json, else a pixel occupancy screen.

Candidates are packages of the same recipe/episode rendered with different seed offsets.  Chunks render independently, so
a chunk may come from any seed.  The choice file (schema h3.seed-screen/1, see h3_screen_prep.py) names one candidate per
chunk; the derived package links those chunk outputs.  The screen is not the acceptance review.

Without a choice file the chunk goes to the candidate with the lowest carrier-grounded occupancy penalty (the fallback
when the VLM screen produced no valid answer).  Per view, against the white clip of the chunk:
  invented = generated colourful pixels (S>=70, V>=40) farther than 15 px from any carrier colourful pixel
             + 0.5 x generated near-black pixels (V<50) farther than 25 px from any carrier robot/colourful pixel
  missing  = carrier colourful pixels (eroded) with no generated colourful pixel within 9 px
penalty = mean of the worst quarter of (frame, view) values of invented + 0.5 x missing over the chunk's keep range,
plus 0.01 x the mean per-view Lab distance to the first full-view real reference (global drift, e.g. a black table).
The robot term assumes a dark robot in the carrier (recolour the white robot first, white_replay `@robot`).  Pixel
penalties miss content errors a VLM sees (cables, a white arm); the fallback is recorded in the package's selection.

  python h3_select.py --cands <pkg_s0> <pkg_s1> ... --recipe <run>/recipes/<name>.json [--choice-file <choice.json>]
  (the derived package goes to <run>/pkgs/<recipe name>/<task>/<tag>; run directory = --run-dir or env H3_RUN)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h3_paths
from h3_media import read_frames
from h3_spatial import SPATIAL

SCREEN_SCHEMA = "h3.seed-screen/1"
OUTPUT_NAME = "chunk-render.mp4"


def choice_problem(screen, chunks: int, cands: int) -> str | None:
    """Why a choice.json cannot be used for `chunks` chunks and `cands` candidates (None when it can)."""
    try:
        if screen.get("schema") != SCREEN_SCHEMA:
            return f"schema is {screen.get('schema')!r}, not {SCREEN_SCHEMA}"
        rows = sorted(screen["chunks"], key=lambda c: int(c["chunk"]))
        if [int(c["chunk"]) for c in rows] != list(range(chunks)):
            return f"covers chunks {[c['chunk'] for c in rows]}, expected 0..{chunks - 1}"
        for c in rows:
            if len(c["scores"]) != cands:
                return f"chunk {c['chunk']} has {len(c['scores'])} scores for {cands} candidates"
            if not 0 <= int(c["choice"]) < cands:
                return f"chunk {c['chunk']} chooses candidate {c['choice']} of {cands}"
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return f"malformed ({type(e).__name__}: {e})"
    return None


def _masks(frame):
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colourful = ((hsv[..., 1] >= 70) & (hsv[..., 2] >= 40)).astype("uint8")
    dark = (hsv[..., 2] < 50).astype("uint8")
    robot = ((hsv[..., 2] < 110) & (hsv[..., 1] < 90)).astype("uint8")
    return colourful, dark, robot


def reference_lab(pkg_dir: Path) -> dict | None:
    """Mean Lab per view of the package's first full-view real reference (a generator input, not the eval pool)."""
    import cv2
    intent = Path(pkg_dir) / "contract" / "intent.json"
    if not intent.is_file():
        return None
    ref = next((r for r in json.loads(intent.read_text()).get("real_references", [])
                if r.get("scope") == "full-view" and r.get("view_regions")), None)
    im = cv2.imread(ref["path"]) if ref else None
    if im is None:
        return None
    out = {}
    for view, (x, y, w, h) in ref["view_regions"].items():
        lab = cv2.cvtColor(cv2.resize(im[y:y + h, x:x + w], (160, 90)), cv2.COLOR_BGR2LAB)
        out[view] = lab.reshape(-1, 3).astype(float).mean(0)
    return out


def occupancy_penalty(white: str, generated: str, lo: int, hi: int, ref_lab: dict | None = None) -> float:
    """Occupancy (+ appearance drift) penalty of one generated chunk against its white clip, frames [lo, hi) step 3."""
    import cv2
    import numpy as np
    cw, gw = read_frames(white, lo, hi, 3), read_frames(generated, lo, hi, 3)
    views = SPATIAL.canvas_views()
    k15, k25, k9, k5 = (np.ones((n, n), np.uint8) for n in (31, 51, 19, 5))
    pens = []
    for i in sorted(set(cw) & set(gw)):
        c_col, _, c_rob = _masks(cw[i])
        g_col, g_dark, _ = _masks(gw[i])
        c_near, c_robot_near, g_near = cv2.dilate(c_col, k15), cv2.dilate(c_rob | c_col, k25), cv2.dilate(g_col, k9)
        c_core = cv2.erode(c_col, k5)
        for y0, y1, x0, x1 in views.values():
            ys, xs = slice(y0, y1), slice(x0, x1)
            area = c_col[ys, xs].size
            invented = float((g_col[ys, xs] & (1 - c_near[ys, xs])).sum()) / area
            invented += 0.5 * float((g_dark[ys, xs] & (1 - c_robot_near[ys, xs])).sum()) / area
            core = float(c_core[ys, xs].sum())
            missing = float((c_core[ys, xs] & (1 - g_near[ys, xs])).sum()) / core if core > 200 else 0.0
            pens.append(invented + 0.5 * missing)
    if not pens:
        return 1e9
    pens = np.sort(np.array(pens))[::-1]
    penalty = float(pens[: max(1, len(pens) // 4)].mean())
    if ref_lab:
        drift = []
        for view, (y0, y1, x0, x1) in views.items():
            if view in ref_lab:
                labs = [cv2.cvtColor(cv2.resize(gw[i][y0:y1, x0:x1], (160, 90)), cv2.COLOR_BGR2LAB)
                        .reshape(-1, 3).astype(float).mean(0) for i in sorted(gw)]
                drift.append(float(np.linalg.norm(np.mean(labs, 0) - ref_lab[view])))
        penalty += 0.01 * float(np.mean(drift)) if drift else 0.0
    return penalty


def occupancy_table(cands: list[Path], manifests: list[dict]) -> list[list[float]]:
    """Penalty per chunk (rows) and candidate (columns); a missing chunk output scores 1e9."""
    ref_lab = reference_lab(cands[0])
    table = []
    for k, job in enumerate(manifests[0]["jobs"]):
        a, b = job["keep_local_range"]
        row = []
        for c, m in zip(cands, manifests):
            j = m["jobs"][k]
            out = c / "render/chunks" / j["chunk_id"] / OUTPUT_NAME
            row.append(occupancy_penalty(j["white"], str(out), max(0, a - 12), min(int(j["generated_frames"]), b + 12),
                                         ref_lab) if out.is_file() else 1e9)
        table.append(row)
    return table


def select_seeds(cands: list[Path], recipe: dict, choice_file: Path | None, run: Path,
                 fallback_reason: str | None = None) -> dict:
    cands = [Path(c).resolve() for c in cands]
    manifests = [json.loads((c / "job-manifest.json").read_text()) for c in cands]
    base = manifests[0]
    for manifest in manifests[1:]:
        for job, other in zip(base["jobs"], manifest["jobs"]):
            if (job["chunk_id"], job["white_sha256"]) != (other["chunk_id"], other["white_sha256"]):
                raise SystemExit(f"candidates differ at chunk {job['chunk_id']}")
    if choice_file is not None:
        screen = json.loads(Path(choice_file).read_text())
        problem = choice_problem(screen, len(base["jobs"]), len(cands))
        if problem:
            raise SystemExit(f"{choice_file}: {problem}")
        rows = sorted(screen["chunks"], key=lambda c: int(c["chunk"]))
        choice = [int(c["choice"]) for c in rows]
        selection = {"mode": "screen", "scores": [c["scores"] for c in rows], "choice": choice,
                     "screen_file": str(Path(choice_file).resolve()), "screener": screen.get("screener")}
    else:
        table = occupancy_table(cands, manifests)
        choice = [min(range(len(row)), key=row.__getitem__) for row in table]
        selection = {"mode": "occupancy", "penalties": [[round(p, 5) for p in row] for row in table],
                     "choice": choice, "screen": "carrier-grounded occupancy + reference drift (h3_select.py)",
                     **({"fallback_reason": fallback_reason} if fallback_reason else {})}
    pkg = json.loads((cands[0] / "package.json").read_text())
    tag = re.sub(r"-s\d+$", "", cands[0].name)
    dst = run / "pkgs" / recipe["name"] / cands[0].parts[-2] / tag
    (dst / "render/chunks").mkdir(parents=True, exist_ok=True)
    for job, picked in zip(base["jobs"], choice):
        link = dst / "render/chunks" / job["chunk_id"]
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(cands[picked] / "render/chunks" / job["chunk_id"])
    (dst / "job-manifest.json").write_text(json.dumps(base, indent=2))
    pkg["recipe"] = recipe
    pkg["derived_from"] = [str(c) for c in cands]
    pkg["selection"] = selection
    (dst / "package.json").write_text(json.dumps(pkg, indent=2, ensure_ascii=False))
    return {"dst": str(dst), "choice": choice, "mode": selection["mode"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cands", nargs="+", required=True, type=Path)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--choice-file", type=Path, help="h3.seed-screen/1 answers; omitted: occupancy screen")
    h3_paths.add_run_dir_arg(ap)
    args = ap.parse_args()
    recipe = json.loads(Path(args.recipe).read_text())
    print(json.dumps(select_seeds(args.cands, recipe, args.choice_file, h3_paths.run_dir(args.run_dir))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
