#!/usr/bin/env python3
"""Create blind, anonymized review jobs for EVAL-v1 independent reviews.

For each subject (an H3 package dir, or any dir with pack/ + score.json + a subject.json descriptor) we copy the
lite pack into <run>/reviews/<rid>/pack, link generated.mp4 / carrier.mp4, and write <run>/reviews/<rid>/task.json
(mode + OOD description only). The subject identity lives in <run>/reviews/.mapping.json, which reviewers
are forbidden to read. Re-running is idempotent per subject.

  python h3_review_prep.py [--run-dir <run>] --subjects <dir> [<dir> ...]   # prints rid per subject
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
from pathlib import Path

import h3_paths


def subject_info(d: Path) -> dict:
    if (d / "package.json").exists():
        pkg = json.loads((d / "package.json").read_text())
        asm = json.loads((d / "assembled.json").read_text())
        bundle = json.loads(Path(pkg["bundle"]).read_text())
        var = next((v for v in bundle.get("ood_variants", []) if v["name"] == pkg.get("variant")), None)
        return {"kind": "h3", "recipe": pkg["recipe"]["name"], "task": pkg["task"], "episode": pkg["episode"],
                "mode": pkg["mode"], "variant": var, "generated": asm["final"], "carrier": asm["carrier_clip"]}
    rec = json.loads((d / "subject.json").read_text())  # descriptor of a subject that is not a package
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    h3_paths.add_run_dir_arg(ap)
    args = ap.parse_args()
    RUN = h3_paths.run_dir(args.run_dir)
    MAP = RUN / "reviews" / ".mapping.json"
    MAP.parent.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(MAP.read_text()) if MAP.exists() else {}
    by_subject = {v["subject"]: k for k, v in mapping.items()}
    for s in args.subjects:
        d = Path(s).resolve()
        if str(d) in by_subject:
            print(by_subject[str(d)], d)
            continue
        info = subject_info(d)
        rid = "r" + secrets.token_hex(4)
        rd = RUN / "reviews" / rid
        shutil.copytree(d / "pack", rd / "pack")
        for name, src in (("generated.mp4", info["generated"]), ("carrier.mp4", info["carrier"])):
            try:  # hard link / copy, never a symlink: `ls -l` must not reveal the subject's identity
                os.link(src, rd / name)
            except OSError:
                shutil.copy(src, rd / name)
        task = {"review_id": rid, "mode": info["mode"], "task": info["task"]}
        if info["mode"] == "ood" and info.get("variant"):
            task["ood_change"] = {"tabletop": info["variant"]["tabletop"], "lighting": info["variant"]["lighting"]}
        (rd / "task.json").write_text(json.dumps(task, indent=2))
        (rd / "review").mkdir()
        mapping[rid] = {"subject": str(d), **{k: v for k, v in info.items() if k not in ("generated", "carrier")}}
        print(rid, d)
    MAP.write_text(json.dumps(mapping, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
