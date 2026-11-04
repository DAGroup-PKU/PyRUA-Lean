#!/usr/bin/env python3
"""Stand-in simulator for dry runs: loads the expert plugin like the real runner
(``file.py:Class`` built as ``Class(pose_bank, home=home)``), drives ``next_plan`` /
``complete`` for a few primitives, probes that protected paths are hidden, and writes
an initial-state record plus a state "trace" for the out-of-sandbox judge.

Deterministic behaviour: family = seed % 4; seed 13 is an invalid start; ``FAKE_RAISE``
in the expert raises inside the plugin; ``FAKE_STEPS`` above ``--trace-capacity`` hits the
trace capacity like the real runner.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

FAMILIES = ("family_a", "family_b", "family_c", "family_d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe", action="append", default=[])
    parser.add_argument("--trace-capacity", type=int, default=7200)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seen = {}
    for path in args.probe:
        try:
            if os.path.isdir(path):
                seen[path] = "readable" if os.listdir(path) else "hidden"
            else:
                with open(path, "rb") as handle:
                    seen[path] = "readable" if handle.read(1) else "hidden"
        except OSError as exc:
            seen[path] = "hidden (%s)" % exc.strerror
    with open(os.path.join(args.output_dir, "probe.json"), "w") as handle:
        json.dump(seen, handle, indent=1)
    family = FAMILIES[args.seed % len(FAMILIES)]
    valid = args.seed != 13
    with open(os.path.join(args.output_dir, "init.json"), "w") as handle:
        json.dump({"requested_seed": args.seed, "family": family, "valid": valid}, handle)
    if not valid:
        print("initial state invalid after settling; no episode")
        return 3
    path, _, cls = args.expert.rpartition(":")
    spec = importlib.util.spec_from_file_location("expert_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expert = getattr(module, cls)(None, home=None)
    steps = int(getattr(module, "FAKE_STEPS", 3))
    if steps > args.trace_capacity:
        print("visualization state trace exceeded its fixed device buffer capacity "
              "(%d samples)" % args.trace_capacity)
        return 1
    for _ in range(min(steps, 50)):
        plan = expert.next_plan({"seed": args.seed})
        if plan is None:
            break
        expert.complete(plan, {"seed": args.seed}, {"ok": True})
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    quality = float(getattr(module, "FAKE_QUALITY", 0.0))
    score = int(hashlib.sha256(("%d:%s" % (args.seed, digest)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    with open(os.path.join(args.output_dir, "trace.json"), "w") as handle:
        json.dump({"seed": args.seed, "quality": quality, "score": score, "family": family,
                   "probe": seen}, handle)
    print("episode finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
