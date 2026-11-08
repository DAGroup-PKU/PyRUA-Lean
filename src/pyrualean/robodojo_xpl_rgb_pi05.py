# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboDojo under its own evaluation client, colour images only, with RoboDojo's Pi_05.

The robot of ``pyrualean.robodojo_xpl_rgb`` (the same API, the same knowledge files) with its
``pi0_pick`` served: the host (:mod:`pyrualean.hosts.xpolicylab_robodojo_rgb_pi05`) runs
XPolicyLab's own ``policy/Pi_05``, the policy RoboDojo's leaderboard evaluates, on the
published checkpoint (``RoboDojo-sim-arx_x5-joint-0``), in a process of its own.  Run it with
``--primitives full``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from . import robodojo_xpl_rgb as _rgb
from .robodojo_xpl_rgb import RobodojoOfficialColourRobot


class RobodojoOfficialColourPi05Robot(RobodojoOfficialColourRobot):
    __doc__ = RobodojoOfficialColourRobot.__doc__

    NAME: ClassVar[str] = "robodojo_xpl_rgb_pi05"


ADAPTER = replace(
    _rgb.ADAPTER,
    name="robodojo_xpl_rgb_pi05",
    robot_cls=RobodojoOfficialColourPi05Robot,
    host="pyrualean.hosts.xpolicylab_robodojo_rgb_pi05",
)


__all__ = ["ADAPTER", "RobodojoOfficialColourPi05Robot"]
