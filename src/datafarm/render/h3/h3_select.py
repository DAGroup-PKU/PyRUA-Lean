#!/usr/bin/env python3
"""Per-chunk seed selection from a screening model's choice.json (assembly-only, no GPU).

Candidates are packages of the same recipe/episode rendered with different seed offsets.  Chunks render independently, so
a chunk may come from any seed.  The choice file (schema h3.seed-screen/1, see h3_screen_prep.py) names one candidate per
chunk; the derived package links those chunk outputs.  The screen is not the acceptance review.

  python h3_select.py --cands <pkg_s0> <pkg_s1> ... --recipe <run>/recipes/<name>.json --choice-file <choice.json>
  (the derived package goes to <run>/pkgs/<recipe name>/<task>/<tag>; run directory = --run-dir or env H3_RUN)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h3_paths


def select_seeds(cands: list[Path], recipe: dict, choice_file: Path, run: Path) -> dict:
    cands = [Path(c).resolve() for c in cands]
    manifests = [json.loads((c / "job-manifest.json").read_text()) for c in cands]
    base = manifests[0]
    for manifest in manifests[1:]:
        for job, other in zip(base["jobs"], manifest["jobs"]):
            if (job["chunk_id"], job["white_sha256"]) != (other["chunk_id"], other["white_sha256"]):
                raise SystemExit(f"candidates differ at chunk {job['chunk_id']}")
    screen = json.loads(Path(choice_file).read_text())
    if len(screen["chunks"]) != len(base["jobs"]):
        raise SystemExit("screen does not cover every chunk")
    rows = sorted(screen["chunks"], key=lambda c: c["chunk"])
    choice = [int(c["choice"]) for c in rows]
    if not all(len(c["scores"]) == len(cands) for c in rows) or not all(0 <= c < len(cands) for c in choice):
        raise SystemExit("screen does not match the candidates")
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
    pkg["selection"] = {"mode": "screen", "scores": [c["scores"] for c in rows], "choice": choice,
                        "screen_file": str(Path(choice_file).resolve()), "screener": screen.get("screener")}
    (dst / "package.json").write_text(json.dumps(pkg, indent=2, ensure_ascii=False))
    return {"dst": str(dst), "choice": choice}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cands", nargs="+", required=True, type=Path)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--choice-file", required=True, type=Path)
    h3_paths.add_run_dir_arg(ap)
    args = ap.parse_args()
    recipe = json.loads(Path(args.recipe).read_text())
    print(json.dumps(select_seeds(args.cands, recipe, args.choice_file, h3_paths.run_dir(args.run_dir))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
