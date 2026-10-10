# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Let the support (opponent / demonstrator) arm of RoboDojo's interactive tasks move.

    python -m pyrualean.hosts.robodojo_support_arm_patch            # patch (idempotent)
    python -m pyrualean.hosts.robodojo_support_arm_patch --check    # report only
    python -m pyrualean.hosts.robodojo_support_arm_patch --revert   # restore the runtime's file

Run it with the interpreter of RPent's ``robodojo`` environment.

Three RoboDojo tasks have a third arm the benchmark drives itself (``interact = True``):
``play_tic_tac_toe`` and ``make_kong`` (an opponent that plays its moves), and
``imitate_sorting_sequence`` (a demonstrator whose order the robot must copy).  RoboDojo's own
evaluation client (``src/eval_client/eval_env.py``) queues that arm's trajectory after the
episode's reward checks are registered and after every step (``query_support_arm_traj``,
``check_support_arm_stable``), and splices its queued actions into each step's control
sequence (``process_control_info``).  The client RPent drives the simulator through
(``src/collect_client/collect_env.py``) does neither: its control expansion skips every arm
that is not a target arm, and nothing queues the support arm.  So the opponent never plays
(``make_kong``: "no discard after 341-376 actions" in 3 of 3 episodes; ``play_tic_tac_toe``:
the agent placed every piece itself) and no task with such an arm can be solved.

The patch adds the eval client's three steps to ``collect_env.py``, guarded by ``interact``
like the original, so every other task runs exactly as before.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

#: Present in a file patched by this module.
TAG = "pyrualean robodojo support-arm patch v1"
COLLECT_FILE = Path("_source") / "src" / "collect_client" / "collect_env.py"


def _queue(indent: str) -> str:
    """Queue the support arm's first trajectory, as ``eval_env.run_eval`` does after the checks."""

    lines = (
        f"# {TAG} (python -m pyrualean.hosts.robodojo_support_arm_patch):",
        "# queue the support / opponent arm's first trajectory, as eval_env.run_eval does",
        'if getattr(self, "interact", False) and hasattr(self, "query_support_arm_traj"):',
        "    for env_idx in range(self.num_envs):",
        "        self.query_support_arm_traj(env_idx=env_idx)",
    )
    return "".join(indent + line + "\n" for line in lines)


EDITS: tuple[tuple[str, str], ...] = (
    (
        "    for robot in env.robot_manager.robot_list:\n"
        '        if robot.type != "target":\n'
        "            continue\n",
        "    for robot in env.robot_manager.robot_list:\n"
        '        if robot.type != "target":\n'
        f"            # {TAG}: the support / opponent arm replays its queued\n"
        "            # trajectory, one entry per substep (eval_env.process_control_info)\n"
        '            queue = getattr(env, "support_arm_action", None)\n'
        "            if queue is not None and len(queue[env_idx]) > 0:\n"
        "                for i in range(interpolation_nums):\n"
        "                    if len(queue[env_idx]) == 0:\n"
        "                        break\n"
        "                    control_info_list[i].update(queue[env_idx][0])\n"
        "                    queue[env_idx].pop(0)\n"
        "            continue\n",
    ),
    (
        "            self.reward_manager.step([env_idx])\n",
        "            self.reward_manager.step([env_idx])\n"
        f"            # {TAG}: the support arm takes its turn, as in eval_env.step\n"
        '            if getattr(self, "interact", False):\n'
        '                if hasattr(self, "query_support_arm_traj"):\n'
        "                    self.query_support_arm_traj(env_idx=env_idx)\n"
        '                if hasattr(self, "check_support_arm_stable"):\n'
        "                    self.check_support_arm_stable(env_idx=env_idx)\n",
    ),
    (
        "                self.reward_manager.init_state()\n"
        "                self.run_reward()\n"
        "                return\n",
        "                self.reward_manager.init_state()\n"
        "                self.run_reward()\n" + _queue(" " * 16) + "                return\n",
    ),
    (
        "            # this, get_reward() sees a non-empty check list and reports 0.\n"
        "            self.run_reward()\n",
        "            # this, get_reward() sees a non-empty check list and reports 0.\n"
        "            self.run_reward()\n" + _queue(" " * 12),
    ),
)


class PatchError(RuntimeError):
    """The runtime's file is not the one this patch was written for."""


def runtime_root() -> Path | None:
    """The installed ``robodojo_runtime`` package of this interpreter, without importing it."""

    try:
        spec = importlib.util.find_spec("robodojo_runtime")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(next(iter(spec.submodule_search_locations)))
    return root if (root / COLLECT_FILE).is_file() else None


def _orig(path: Path) -> Path:
    return path.with_suffix(".py.orig")


def status(root: Path | None = None) -> str:
    """``"corrected"``, ``"upstream"`` or ``"missing"``."""

    root = runtime_root() if root is None else root
    if root is None:
        return "missing"
    return "corrected" if TAG in (root / COLLECT_FILE).read_text() else "upstream"


def revert(root: Path) -> list[Path]:
    path = root / COLLECT_FILE
    if _orig(path).exists():
        shutil.copyfile(_orig(path), path)
        return [path]
    return []


def apply(root: Path) -> list[Path]:
    """Patch ``collect_env.py`` (idempotent); returns the files changed."""

    if status(root) == "corrected":
        return []
    path = root / COLLECT_FILE
    text = path.read_text()
    for old, new in EDITS:
        if text.count(old) != 1:
            raise PatchError(
                f"{path.name}: expected exactly one occurrence of:\n{old}\nthe runtime changed"
            )
        text = text.replace(old, new)
    compile(text, str(path), "exec")
    if not _orig(path).exists():
        shutil.copyfile(path, _orig(path))
    tmp = path.with_suffix(".py.new")
    tmp.write_text(text)
    tmp.replace(path)
    return [path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="report the state and change nothing")
    parser.add_argument("--revert", action="store_true", help="restore the runtime's own file")
    ns = parser.parse_args(argv)
    root = runtime_root()
    if root is None:
        print("no robodojo_runtime with a collect client for this interpreter", file=sys.stderr)
        return 1
    if ns.check:
        print(f"{root}: {status(root)}")
        return 0
    try:
        changed = revert(root) if ns.revert else apply(root)
    except PatchError as exc:
        print(exc, file=sys.stderr)
        return 1
    for path in changed:
        print("restored" if ns.revert else "patched", path)
    print(f"{root}: {status(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
