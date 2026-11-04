#!/usr/bin/env python3
"""Independent full-control trace audit, with no physical rollout or label mutation."""

import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datafarm.simulation.joint_margin_audit import audit  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.trace)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(a.output)
    print(
        json.dumps(
            {
                "samples": result["samples"],
                "joint_target_met": result["joint_target_met"],
                "min_margin_deg": {k: v["actual"]["min_margin_deg"] for k, v in result["arms"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
