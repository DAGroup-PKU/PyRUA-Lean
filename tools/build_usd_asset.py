#!/usr/bin/env python3
"""Plan or execute an optional native Codex USDCraft asset build."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datafarm.simulation.usdcraft import plan, run, run_for_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", required=True)
    p.add_argument("--observation", required=True)
    p.add_argument("--output", required=True)
    r = sub.add_parser("run")
    r.add_argument("--work", required=True)
    r.add_argument("--task-work", help="parent task workspace: bind shared budget and external job lifecycle")
    args = parser.parse_args()
    result = (
        plan(args.config, args.observation, args.output)
        if args.command == "plan"
        else (run_for_task(args.work, args.task_work) if args.task_work else run(args.work))
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(args.command == "run" and result["status"] != "exported-unqualified")


if __name__ == "__main__":
    raise SystemExit(main())
