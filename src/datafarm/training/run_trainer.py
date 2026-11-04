#!/usr/bin/env python3
"""Run the vendored openpi trainer with explicit target-mesh checkpoint restoration.

``--action-adapter`` installs the config extensions of one action representation; all other
arguments go to the trainer.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys

# Started by path: replace this directory on sys.path, whose modules must not shadow the trainer's top-level imports.
sys.path[0] = str(Path(__file__).resolve().parents[2])


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--action-adapter", choices=("none", "joint", "delta-eef"), default="none")
    args, trainer_argv = parser.parse_known_args(argv)
    sys.argv = [sys.argv[0], *trainer_argv]
    if args.action_adapter == "delta-eef":
        from datafarm.training.delta_eef import install
        install()
    if args.action_adapter == "joint":
        from datafarm.training.joint_actions import install
        install()
    from datafarm.training.checkpoint_restore import install
    namespace = runpy.run_path(str(Path(os.environ["DATAFARM_TRAINER_ROOT"]) / "scripts/train.py"))
    if args.action_adapter == "joint" and "ACTION_GROUPS" in namespace:
        from datafarm.training.joint_actions import action_groups
        namespace["ACTION_GROUPS"] = action_groups()
    trainer = namespace["main"]
    install(trainer.__globals__)
    trainer(namespace["_config"].cli())


if __name__ == "__main__":
    main()
