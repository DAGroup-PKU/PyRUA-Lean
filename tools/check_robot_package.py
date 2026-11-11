#!/usr/bin/env python3
"""Check a robot description package supplied with a task input (URDF + meshes, optional calibration).

  python tools/check_robot_package.py <package.zip|folder|robot.urdf> [--extract-to <folder>]

A .zip is unpacked into --extract-to without trusting it (no absolute or parent paths, no symlinks, bounded size).
Prints the main URDF (most links among the URDFs without problems), links, joints, actuated joints, the root link,
joints without limits, unresolved meshes and calibration files.  Exit 0 only when the main URDF has no problems.
The report describes the package; grounding still verifies the robot against the real views.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from datafarm.simulation.robot_package import PackageError, check_package  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('package', type=Path)
    ap.add_argument('--extract-to', type=Path, help='folder a .zip package is unpacked into')
    args = ap.parse_args(argv)
    try:
        report = check_package(args.package, args.extract_to)
    except PackageError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
