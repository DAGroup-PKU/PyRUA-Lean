# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The host of ``--robot libero_rgb``: RPent's LIBERO stack with this library's colour tools.

Everything :mod:`.rpent_libero` does (RPent's env, Pi0.5 and SAM3 servers, its toolkit and its
records), plus the tools of :mod:`.libero_tools` registered on the toolkit after boot.  The task
card carries the colour images only.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..prompt import TaskCard
from . import libero_tools as tools
from . import rpent_libero as _base
from .rpent_libero import (  # noqa: F401 - the host interface
    Cell,
    add_boot_args,
    add_cell_args,
    boot_options,
    cell_from_args,
)


def boot(cell: Cell, out_dir: Path, **options: Any) -> tuple[Any, list[Any]]:
    """:func:`.rpent_libero.boot`, then this library's colour tools on the toolkit."""

    toolkit, daemons = _base.boot(cell, out_dir, **options)
    try:
        record = tools.register(toolkit)
    except BaseException:
        _base_stop(toolkit, daemons)
        raise
    (Path(out_dir) / "libero_rgb_boot.json").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return toolkit, daemons


def _base_stop(toolkit: Any, daemons: list[Any]) -> None:
    from .common import stop_all

    stop_all(toolkit, daemons)


def dump_card(toolkit: Any, cell: Cell, out_dir: Path) -> TaskCard:
    """The base card as this robot's: its words for the start pose, the table's height."""

    base = _base.dump_card(toolkit, cell, out_dir)
    primitives = toolkit._primitives
    raw = primitives.env.raw_obs()
    pos, rot = tools._pose(raw)
    pitch, yaw, roll = tools.angles(rot)
    facts = {
        "initial grasp point [x, y, z]": "[" + ", ".join(f"{v:.3f}" for v in pos) + "] m",
        "initial gripper": (
            f"pitch {pitch:.0f}, yaw {yaw:.0f}, roll {roll:.0f} deg; "
            f"opening {tools.opening_share(raw):.2f} (open)"
        ),
        "table top height (z)": f"{float(primitives._rgb_table_z):.3f} m",
    }
    card = replace(base, robot="libero_rgb", eef_pos=None, eef_quat=None, facts=facts)
    (Path(out_dir) / "card.json").write_text(card.to_json() + "\n", encoding="utf-8")
    return card


__all__ = [
    "Cell",
    "add_boot_args",
    "add_cell_args",
    "boot",
    "boot_options",
    "cell_from_args",
    "dump_card",
]
