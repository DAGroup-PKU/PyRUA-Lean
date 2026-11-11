#!/usr/bin/env python3
"""Per-episode initial object placements for expert collection (datafarm.object-placement-plan/1).

  python tools/prepare_object_placement_plan.py --scene W/task/scene.json --mode local_robustness \
      --count 5 --seed 42 [--jitter-mm 30] --output W/task/placement-plan.json
  python tools/prepare_object_placement_plan.py --scene W/task/scene.json --mode broad_table \
      --grounding W/task/grounding.json --count 100 --seed 42 --output W/task/placement-plan.json

Nominal positions come from the scene's `objects_initial` (--key to change); broad_table needs the reachable tabletop
region (`reachable_tabletop_xy` in --grounding, or --reachable-json).  Without --mode the default follows the target
count and request text: broad_table for >= 50 episodes or a training goal, otherwise local_robustness.
Refuses to overwrite --output.  Prints the digest and the per-object coverage span.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from datafarm.simulation.placement import MODES, default_mode, generate_plan  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--mode', choices=MODES)
    ap.add_argument('--request-file', type=Path, help='for the default mode: the user request text')
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--jitter-mm', type=float, default=30.0)
    ap.add_argument('--min-dist-mm', type=float, default=80.0)
    ap.add_argument('--grounding', type=Path, help='grounding JSON holding reachable_tabletop_xy')
    ap.add_argument('--reachable-json', type=Path, help='JSON file that is the reachable_tabletop_xy record')
    ap.add_argument('--key', default='objects_initial', help='scene key of the nominal object positions')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        print(f'error: {args.output} exists', file=sys.stderr)
        return 2
    mode = args.mode or default_mode(args.count, args.request_file.read_text() if args.request_file else '')
    reachable = None
    if args.reachable_json:
        reachable = json.loads(args.reachable_json.read_text())
    elif args.grounding:
        reachable = json.loads(args.grounding.read_text()).get('reachable_tabletop_xy')
    try:
        plan = generate_plan(json.loads(args.scene.read_text()), mode, args.count, args.seed,
                             jitter_m=args.jitter_mm / 1000, reachable=reachable, min_dist_m=args.min_dist_mm / 1000,
                             key=args.key)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=1) + '\n')
    print(json.dumps({'output': str(args.output), 'mode': mode, 'episodes': args.count, 'digest': plan['digest'],
                      'coverage': plan['coverage']}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
