#!/usr/bin/env python3
"""Check that simulator runs used exactly the files a simulator-qualification certificate certifies.

usage: check_sim_binding.py <certificate.json> --file NAME=PATH [--file NAME=PATH ...] [--runs RUN_DIR ...]

Each ``--file`` names a certified file (``material.json``, ``scene.json`` or a runtime-flag file such as
``grasp_model_v2.py``) and the file a run configuration uses for it; JSON files are compared by canonical
content, other files by their bytes.  With ``--runs``, every run directory must also have recorded, in its
``environment.txt`` (lines ``<sha256>  <path>``), the same bytes for each of those paths that it recorded at
all, and it must have recorded at least one of them.  Prints a JSON report; exit 0 only if everything binds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datafarm.common import sha256_file as raw_sha  # noqa: E402
from datafarm.simulation.sim_qualification import canonical_json_sha256, verify_run_binding  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--file", action="append", default=[], required=True, metavar="NAME=PATH")
    parser.add_argument("--runs", nargs="*", default=[], type=Path, metavar="RUN_DIR")
    args = parser.parse_args()
    certificate = json.loads(args.certificate.read_text())
    files: dict[str, Path] = {}
    for item in args.file:
        name, _, path = item.partition("=")
        files[name] = Path(path).resolve(strict=True)
    used = {name: canonical_json_sha256(json.loads(p.read_text())) if name.endswith(".json") else raw_sha(p)
            for name, p in files.items()}
    report = {"certificate": str(args.certificate), "selected": (certificate.get("selected_variant") or {}).get("name"),
              "files": {n: str(p) for n, p in files.items()}, "binding": list(verify_run_binding(certificate, used)),
              "runs": {}}
    current = {str(p): raw_sha(p) for p in files.values()}
    for run in args.runs:
        recorded = {}
        env = run / "environment.txt"
        for line in (env.read_text().splitlines() if env.exists() else []):
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) == 64:
                recorded[str(Path(parts[1]).resolve())] = parts[0]
        seen = [p for p in current if p in recorded]
        problems = [f"changed_since_run:{p}" for p in seen if recorded[p] != current[p]]
        if not seen:
            problems.append("no_recorded_digest")
        report["runs"][str(run)] = problems
    ok = not report["binding"] and all(not v for v in report["runs"].values())
    report["ok"] = ok
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
