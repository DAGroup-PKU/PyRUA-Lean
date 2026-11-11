#!/usr/bin/env python3
"""Make a task input folder from one episode of a LeRobot v2.x dataset.

  python tools/import_lerobot_episode.py --dataset <lerobot root> --episode 0 --out <folder> \
      [--camera cam_high=observation.images.head_camera --camera cam_left_wrist=/path/left.mp4 ...] \
      [--robot-package <urdf folder or zip>]

Copies (hard-links when possible) the episode's videos, its parquet log, meta/ and the robot package into <folder>
and writes <folder>/episode.json (sources, sha256, camera mapping, fps, task text, state and action fields).  Then:
  python tools/pipeline_agent.py prepare --episode <folder> --request-file ... --output ...
--camera NAME=SOURCE names an output camera after a dataset video key or an existing video file (default: every video
key, named by its last component).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from datafarm.task.lerobot_episode import EpisodeImportError, import_episode  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--camera', action='append', default=[], metavar='NAME=SOURCE')
    ap.add_argument('--robot-package', type=Path)
    args = ap.parse_args(argv)
    cameras = None
    if args.camera:
        cameras = {}
        for item in args.camera:
            name, sep, source = item.partition('=')
            if not sep or not name or not source:
                ap.error(f'--camera needs NAME=SOURCE, got {item!r}')
            cameras[name] = source
    try:
        record = import_episode(args.dataset, args.episode, args.out, cameras, args.robot_package)
    except (EpisodeImportError, KeyError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    print(json.dumps({k: record[k] for k in ('episode_index', 'fps', 'tasks', 'cameras', 'log', 'robot_package')},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
