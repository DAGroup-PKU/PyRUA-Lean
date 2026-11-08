# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The RoboDojo robot under RoboDojo's own evaluation client, through XPolicyLab.

The same robot as ``pyrualean.robodojo`` (the same API, the same tools, the same knowledge
file), served to RoboDojo's evaluation client as an XPolicyLab policy instead of being run on
RPent's simulator bridge: the client owns the simulator and gives the verdict
(:mod:`pyrualean.hosts.xpolicylab_robodojo`).

Its perception reads what ``pyrualean.robodojo`` reads: a depth image and the camera matrices.
RoboDojo's simulator emits both, its released data is RGB-D, and the client's configuration
shipped with them on; they have been off by default since RoboDojo PR 2.  The host switches
them on in the configuration of the client it starts.  Under a client somebody else starts
they have to be on there (``observation.vision.depth`` or ``approximate_depth``,
``intrinsic_matrix``, ``extrinsic_matrix`` in ``env_cfg/<config>.yml`` and the
``distance_to_image_plane`` capture in its camera file); where they are not,
``pyrualean.robodojo_xpl_rgb`` is the robot that needs none of it.

There is no VLA in this host: run the robot with ``--primitives no-vla``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from . import robodojo as _base
from .robodojo import RobodojoRobot


class RobodojoOfficialRobot(RobodojoRobot):
    __doc__ = RobodojoRobot.__doc__

    NAME: ClassVar[str] = "robodojo_xpl"

    def _reported_success(self) -> bool:
        # RoboDojo's client ends the episode itself the moment the task is solved, and reports
        # that verdict; by then no action is taken any more, so the hosts may stop there too
        return bool(self._backend.solved())


ADAPTER = replace(
    _base.ADAPTER,
    name="robodojo_xpl",
    robot_cls=RobodojoOfficialRobot,
    host="pyrualean.hosts.xpolicylab_robodojo",
)


__all__ = ["ADAPTER", "RobodojoOfficialRobot"]
