# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The G2 arm's IK: the solver the benchmark itself runs, loaded from its own files.

The benchmark turns an ``EEF_ABS`` action into joint targets with the vendor ``ik_solver``
(a relaxed IK: NLopt over a Pinocchio model of ``G2_NO_GRIPPER.urdf``, solved from
``arm_base_link`` to ``arm_{l,r}_end_link``; ``utils/ikfk_utils.py`` and ``utils/IK-SDK`` in
geniesim_benchmark).  Its cost has velocity and acceleration terms, so one solve takes one
small step towards a target and never reports failure.

:mod:`.robocoliseum_motion` solves whole poses here instead, with the same solver and the same
files (copied into ``robocoliseum_ik/``), and sends the joint targets itself: the GOAL solver
is the benchmark's configuration without the velocity and acceleration terms
(``g2_goal.yaml``), so a few solves from the measured joints find the reachable pose nearest
a target; the benchmark's own configuration (``g2_solver.yaml``) keeps consecutive waypoints
of a line on one IK branch.  Measured: this FK agrees with the ``arm_base_link`` end pose the
benchmark reports to 0.0 mm and 0.00 degrees.

Only the arm is modelled.  Nothing here knows the table or the objects: a pose that is
reachable can still be blocked.
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from . import geniesim_bridge as bridge_mod

#: The benchmark's IK files, copied from genie_sim (``utils/IK-SDK`` of geniesim_benchmark:
#: ``G2_NO_GRIPPER.urdf`` and ``g2_solver.yaml``, unchanged) beside the goal configuration,
#: which is ours.  ``GENIESIM_IK_SDK`` names another directory holding the benchmark's two,
#: e.g. a genie_sim checkout's, when its robot description has changed.
IK_SDK = Path(os.environ.get("GENIESIM_IK_SDK") or Path(__file__).with_name("robocoliseum_ik"))
URDF = IK_SDK / "G2_NO_GRIPPER.urdf"
GOAL_CONFIG = Path(__file__).with_name("robocoliseum_ik") / "g2_goal.yaml"
#: The benchmark's own configuration, velocity and acceleration costs included.
STEP_CONFIG = IK_SDK / "g2_solver.yaml"

_LOCK = threading.Lock()
_SOLVERS: dict[str, Any] = {}


@contextlib.contextmanager
def _quiet_fd1():
    """The solver's C++ constructor prints its configuration to file descriptor 1."""

    try:
        saved = os.dup(1)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def available() -> bool:
    try:
        import ik_solver  # noqa: F401
    except ImportError:
        return False
    return URDF.is_file() and GOAL_CONFIG.is_file() and STEP_CONFIG.is_file()


def _solver(arm: str, config: Path = GOAL_CONFIG) -> Any:
    key = f"{arm}:{config.name}"
    with _LOCK:
        if key not in _SOLVERS:
            import ik_solver

            part = ik_solver.RobotPart.LEFT_ARM if arm == "left" else ik_solver.RobotPart.RIGHT_ARM
            with _quiet_fd1():
                solver = ik_solver.Solver(part=part, urdf_path=str(URDF), config_path=str(config))
                solver.set_debug_mode(False)
            _SOLVERS[key] = solver
        return _SOLVERS[key]


def _pose(position: Any, quat_xyzw: Any) -> np.ndarray:
    from .robocoliseum_tools import quat_xyzw_matrix

    out = np.eye(4)
    out[:3, :3] = quat_xyzw_matrix(quat_xyzw)
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def base_from_arm_base(obs: bridge_mod.Observation) -> np.ndarray:
    """``T_base_link<-arm_base_link`` now, from the end pose reported in both frames.

    It depends on the waist joints only, and the benchmark reports each flange in both
    frames, so one arm's two poses give it exactly.
    """

    poses = obs.end_pose or {}
    left_base = poses["base_link"]["left_arm"]
    left_arm = poses["arm_base_link"]["left_arm"]
    t_base = _pose(left_base["position"], left_base["orientation"])
    t_arm = _pose(left_arm["position"], left_arm["orientation"])
    return t_base @ np.linalg.inv(t_arm)


__all__ = [
    "available",
    "base_from_arm_base",
]
