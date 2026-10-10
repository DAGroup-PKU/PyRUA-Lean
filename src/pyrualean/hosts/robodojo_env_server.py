# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RPent's RoboDojo env server, with the benchmark's third arm driven as RoboDojo drives it.

    python -m pyrualean.hosts.robodojo_env_server <RPent's robots/robodojo/env_server.py> [its args]

The RoboDojo host starts the simulator through this module (``rpent_robodojo.boot``); nothing
installed is changed.  This process attaches the steps below to the runtime's classes in memory,
then runs RPent's env server script as it is.

Three RoboDojo tasks have a third arm the benchmark drives itself (``interact = True``): the
opponent of ``play_tic_tac_toe`` and ``make_kong``, the demonstrator of
``imitate_sorting_sequence``.  RoboDojo's evaluation client (``src/eval_client/eval_env.py``)
queues that arm's trajectory once the episode's checks are registered and after every action
(``query_support_arm_traj``, ``check_support_arm_stable``) and replays the queue inside each
action's control sequence.  RPent's env server steps the simulator through the client
``rlinf-robodojo-runtime`` adds for that purpose (``robodojo_runtime.bridge`` and
``src/collect_client/collect_env.py``), which does none of this: the opponent never plays and
those tasks cannot be solved.  The same three steps are attached here, each only for a task that
has such an arm, so every other task runs exactly as before.
"""

from __future__ import annotations

import runpy
import sys
from typing import Any

#: Set on what this module attached, so a second :func:`install` changes nothing.
MARK = "_pyrualean_support_arm"


def _query(env: Any, env_idx: int) -> None:
    queue = getattr(env, "support_arm_action", None)
    before = len(queue[env_idx]) if queue is not None else 0
    env.query_support_arm_traj(env_idx=env_idx)
    after = len(queue[env_idx]) if queue is not None else 0
    if after > before:  # in the env server's log: the arm has a move to make
        print(f"[pyrualean] third arm: {after - before} steps queued", file=sys.stderr, flush=True)


def _queue_first(env: Any) -> None:
    """The arm's first trajectory, as ``eval_env.run_eval`` queues it after the checks."""

    if getattr(env, "interact", False) and hasattr(env, "query_support_arm_traj"):
        for env_idx in range(int(getattr(env, "num_envs", 1))):
            _query(env, env_idx)


def _take_turn(env: Any, env_idx: int = 0) -> None:
    """After an action: the arm's next move when it is its turn, then its check (``eval_env``)."""

    if not getattr(env, "interact", False):
        return
    if hasattr(env, "query_support_arm_traj"):
        _query(env, env_idx)
    if hasattr(env, "check_support_arm_stable"):
        env.check_support_arm_stable(env_idx=env_idx)


def _replaying(expand: Any) -> Any:
    """The control expansion, plus one queued step of the arm per substep.

    As ``eval_env.process_control_info`` splices the arm's queue into each action's sequence.
    """

    def expand_with_support_arm(env: Any, control_info: dict, env_idx: int) -> list[dict]:
        sequence = expand(env, control_info, env_idx)
        queue = getattr(env, "support_arm_action", None)
        if getattr(env, "interact", False) and queue is not None:
            for step in sequence:
                if not queue[env_idx]:
                    break
                step.update(queue[env_idx].pop(0))
        return sequence

    setattr(expand_with_support_arm, MARK, True)
    return expand_with_support_arm


def install(bridge: Any = None) -> None:
    """Attach the three steps to ``robodojo_runtime.bridge`` in this process (idempotent).

    The bridge imports no simulator module at load; the collect client's module is wrapped once
    the bridge has built its environment (it imports the client only after Isaac has started).
    """

    if bridge is None:
        from robodojo_runtime import bridge
    if getattr(bridge, MARK, False):
        return
    native = bridge.NativeEnv
    init, apply_action = native.__init__, native.apply_action

    def __init__(self: Any, task_config: dict) -> None:
        init(self, task_config)
        module = sys.modules.get(type(self.env).__module__)
        expand = getattr(module, "_expand_control_info", None)
        if expand is not None and not getattr(expand, MARK, False):
            module._expand_control_info = _replaying(expand)

    def apply_with_turn(self: Any, action: dict, action_type: str = "joint", **kwargs: Any) -> None:
        apply_action(self, action, action_type, **kwargs)
        _take_turn(self.env)

    native.__init__ = __init__
    native.apply_action = apply_with_turn
    for name in ("_standard_reset", "_random_reset"):
        reset = getattr(bridge, name)

        def reset_and_queue(env: Any, *args: Any, _reset: Any = reset, **kwargs: Any) -> None:
            _reset(env, *args, **kwargs)
            _queue_first(env)

        setattr(bridge, name, reset_and_queue)
    setattr(bridge, MARK, True)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "usage: python -m pyrualean.hosts.robodojo_env_server <env_server.py> [args]"
        )
    install()
    sys.argv = [args[0], *args[1:]]
    runpy.run_path(args[0], run_name="__main__")


if __name__ == "__main__":
    main()
