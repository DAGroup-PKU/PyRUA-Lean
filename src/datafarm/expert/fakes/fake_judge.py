#!/usr/bin/env python3
"""Stand-in judge for dry runs (runs outside the rollout sandbox, like the real verifier).

Reads ``<run>/native/trace.json`` and writes ``<run>/verdict.json``: success iff the
deterministic score is below the expert's quality. Protected paths the simulator could
read are reported in ``leaked_paths`` (the dry run checks them separately).
"""

import json
import os
import sys


def main():
    run = sys.argv[1]
    with open(os.path.join(run, "native", "trace.json"), encoding="utf-8") as handle:
        trace = json.load(handle)
    leaked = [p for p, v in trace.get("probe", {}).items() if v == "readable"]
    success = trace["score"] < trace["quality"]
    error = round(20 + 100 * trace["score"], 1)
    verdict = {"schema": "fake.verdict/1", "m1": success, "m2": success,
               "fold_max_xy_mm": [error, error], "leaked_paths": leaked, "family": trace["family"]}
    with open(os.path.join(run, "verdict.json"), "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=1)
    print(json.dumps({"M2": success}))
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
