# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboDojo's evaluation client as it is configured: colour images and the robot's state.

The host of ``--robot robodojo_xpl_rgb``: everything :mod:`.xpolicylab_robodojo` does, with
the client's configuration left alone and nothing read that it does not send by default (no
depth image, no camera matrices).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import xpolicylab_robodojo as _base
from .xpolicylab_robodojo import (  # noqa: F401 - the host interface
    Cell,
    add_boot_args,
    add_cell_args,
    boot_options,
    cell_from_args,
    make_backend,
)

ROBOT = "robodojo_xpl_rgb"


def boot(cell: Cell, out_dir: Path, **options: Any) -> tuple[Any, list[Any]]:
    """:func:`.xpolicylab_robodojo.boot` on the client's default observation."""

    options["depth"] = False
    return _base.boot(cell, out_dir, **options)


def dump_card(toolkit: Any, cell: Cell, out_dir: Path) -> Any:
    return _base.dump_card(toolkit, cell, out_dir, robot=ROBOT)


__all__ = [
    "Cell",
    "add_boot_args",
    "add_cell_args",
    "boot",
    "boot_options",
    "cell_from_args",
    "dump_card",
    "make_backend",
]
