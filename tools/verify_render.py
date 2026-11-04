#!/usr/bin/env python3
"""Run the spec-driven render verifier on one generated episode.

Example:
  PYTHONPATH=src python tools/verify_render.py \
    --spec configs/render-verifier/template.json \
    --carrier white.mp4 --generated generated.mp4 --real real-tlayout.mp4 \
    --state frame-aligned-robot-state.json --seams <frame> --out runs/.../verify/<id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datafarm.verify.episode import frame_events, gripper_events, load_state  # noqa: E402
from datafarm.verify.report import write_report  # noqa: E402
from datafarm.verify.spec import load_spec  # noqa: E402
from datafarm.verify.verifier import verify_paths  # noqa: E402
from dataclasses import replace  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--carrier", required=True, type=Path)
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--real", type=Path, help="real-world footage in the same panel layout")
    parser.add_argument("--reference", action="append", default=[], type=Path,
                        help="appearance reference image(s) sent to the provider (layout leak check)")
    parser.add_argument("--state", type=Path, help="frame-aligned simulator state (gripper events)")
    parser.add_argument("--event-roles", default="", help="comma list of roles for gripper events")
    parser.add_argument("--event-frames", default="", help="comma list of explicit event frames")
    parser.add_argument("--seams", default="", help="comma list of chunk seam frames")
    parser.add_argument("--single-chunk", action="store_true")
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--mode", choices=("sim2real", "ood"))
    parser.add_argument("--perturbed-axes", default="")
    parser.add_argument("--families", default="", help="comma list to restrict check families")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-html", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_spec(args.spec)
    roles = [r for r in args.event_roles.split(",") if r] or [e.role for e in spec.entities]
    events = []
    state = load_state(args.state) if args.state else None
    if state is not None:
        events += gripper_events(state, roles=roles)
    if args.event_frames:
        events += frame_events([int(f) for f in args.event_frames.split(",")], roles=roles)
    seams = [int(s) for s in args.seams.split(",") if s]
    spec = spec.with_episode(events=events or None, seams=seams,
                             expected_frames=args.expected_frames,
                             chunked=False if args.single_chunk else None)
    if args.mode or args.perturbed_axes:
        axes = tuple(a for a in args.perturbed_axes.split(",") if a)
        spec = replace(spec, realism=replace(spec.realism, mode=args.mode or spec.realism.mode,
                                             perturbed_axes=axes))
    families = [f for f in args.families.split(",") if f] or None
    receipt = verify_paths(spec, args.carrier, args.generated, args.real, args.reference, families, state)
    receipt["episode_binding"] = {"events": [e.__dict__ for e in spec.events], "seams": list(spec.seams)}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.no_html:
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))
    else:
        write_report(spec, receipt, args.out, title=f"{args.generated.parent.parent.name}")
    print(json.dumps({"verdict": receipt["verdict"], "failed": receipt.get("failed"),
                      "abstained": receipt.get("abstained"), "timings_s": receipt.get("timings_s"),
                      "out": str(args.out)}))
    return 0 if receipt["verdict"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
