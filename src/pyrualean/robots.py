# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Registry of the robots PyRUA-Lean can drive.

Each robot module (``pyrualean.libero``, ``pyrualean.robocasa``,
``pyrualean.robotwin``) defines a module-level ``ADAPTER`` describing the
robot class, the host module that boots RPent's stack for it, the knowledge
file and the words the contract uses for it.  ``get_adapter(name)`` imports
the module lazily so a robot whose dependencies are missing does not break
the others.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

ROBOTS: tuple[str, ...] = ("libero", "robocasa", "robotwin")


@dataclass(frozen=True)
class RobotAdapter:
    """What the host and the prompt need to know about one robot."""

    #: RPent robot name and the stem of ``knowledge/<name>.md``.
    name: str
    #: The policy-facing robot class (a :class:`pyrualean._robot.RobotBase`).
    robot_cls: type
    #: Module exposing ``Cell``, ``add_cell_args``, ``cell_from_args``,
    #: ``boot_options``, ``boot`` and ``dump_card`` (see ``hosts/rpent_libero``).
    host: str
    #: Knowledge file stem (``knowledge/<knowledge>.md``).
    knowledge: str
    #: How the contract names the robot: "a simulated Franka arm in the LIBERO benchmark".
    blurb: str
    #: RPent's operating guides for this robot, relative to the RPent checkout.
    guides_subdir: str = ""
    guide_names: tuple[str, ...] = ()
    #: Example ``robo.show(...)`` calls quoted by the on-demand image note.
    show_examples: str = "`robo.show(<camera>)`"
    #: Services the robot object is bound to besides the simulator, as a
    #: phrase: "a frozen Pi0.5 VLA and a SAM3 segmentation service".
    services: str = ""
    #: The same phrase under the ``no-vla`` primitive set (the VLA left out;
    #: empty when the VLA was the only service).
    services_novla: str = ""

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(getattr(self.robot_cls, "CAMERAS", ()))

    def host_module(self) -> Any:
        return importlib.import_module(self.host)


def get_adapter(name: str) -> RobotAdapter:
    """Return the adapter of robot ``name`` (``libero``, ``robocasa``, ``robotwin``)."""

    if name not in ROBOTS:
        raise ValueError(f"unknown robot {name!r}; known: {ROBOTS}")
    module = importlib.import_module(f"pyrualean.{name}")
    adapter = getattr(module, "ADAPTER", None)
    if not isinstance(adapter, RobotAdapter):
        raise ImportError(f"pyrualean.{name} defines no ADAPTER")
    return adapter


__all__ = ["ROBOTS", "RobotAdapter", "get_adapter"]
