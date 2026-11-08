# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboDojo's evaluation client as it is configured, with RoboDojo's Pi_05 beside the tools.

The host of ``--robot robodojo_xpl_rgb_pi05``: everything :mod:`.xpolicylab_robodojo_rgb`
does, and ``pi0_pick`` on the toolkit, served by XPolicyLab's own ``policy/Pi_05``
(:mod:`.pi05_xpolicylab`) in a process of its own.  The policy reads what the client sends by
default (the three colour images, the arms' joints, the instruction), as the robot does.

Run the robot with ``--primitives full``: under ``no-vla`` the VLA is refused at call time,
and the server is started all the same.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from . import pi05_xpolicylab as pi05
from . import xpolicylab_robodojo as _base
from .xpolicylab_robodojo import (  # noqa: F401 - the host interface
    Cell,
    add_cell_args,
    cell_from_args,
    make_backend,
)

ROBOT = "robodojo_xpl_rgb_pi05"


def add_boot_args(parser: argparse.ArgumentParser) -> None:
    _base.add_boot_args(parser)
    parser.add_argument(
        "--pi05-checkpoint",
        default=os.environ.get("ROBODOJO_PI05_CHECKPOINT"),
        help="step directory of RoboDojo's Pi_05 (.../RoboDojo-sim-arx_x5-joint-0/59999)",
    )
    parser.add_argument(
        "--pi05-python",
        default=os.environ.get("ROBODOJO_PI05_PYTHON"),
        help="interpreter of policy/Pi_05's uv environment (default: XPolicyLab's openpi/.venv)",
    )
    parser.add_argument(
        "--pi05-endpoint",
        default=os.environ.get("ROBODOJO_PI05_ENDPOINT"),
        help="host:port of a Pi_05 server already running (nothing is started)",
    )


def boot_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **_base.boot_options(args),
        "pi05_checkpoint": getattr(args, "pi05_checkpoint", None),
        "pi05_python": getattr(args, "pi05_python", None),
        "pi05_endpoint": getattr(args, "pi05_endpoint", None),
    }


def boot(
    cell: Cell,
    out_dir: Path,
    *,
    env_only: bool = False,
    pi05_checkpoint: str | None = None,
    pi05_python: str | None = None,
    pi05_endpoint: str | None = None,
    **options: Any,
) -> tuple[Any, list[Any]]:
    """:func:`.xpolicylab_robodojo.boot` on the client's default observation, with Pi_05.

    The policy server is started first (it loads in about a minute, the simulator in
    several), on the episode's card, and serves the later episodes of this process as well
    (:func:`.pi05_xpolicylab.shared`; an XPolicyLab policy server plays one episode after
    another).  ``env_only`` starts neither it nor SAM3.
    """

    options["depth"] = False
    if env_only:
        return _base.boot(cell, out_dir, env_only=True, **options)
    client = None
    try:
        endpoint = pi05_endpoint or os.environ.get("ROBODOJO_PI05_ENDPOINT")
        if not endpoint:
            checkpoint = pi05_checkpoint or os.environ.get("ROBODOJO_PI05_CHECKPOINT")
            if not checkpoint:
                raise ValueError(
                    "robot robodojo_xpl_rgb_pi05 needs RoboDojo's Pi_05: --pi05-checkpoint or "
                    "ROBODOJO_PI05_CHECKPOINT (or --pi05-endpoint of a running server)"
                )
            server = pi05.shared(
                _xpolicylab(options),
                checkpoint,
                out_dir,
                python=pi05_python or os.environ.get("ROBODOJO_PI05_PYTHON"),
                cuda_device=cell.cuda_device,
            )
            endpoint = server.endpoint
        client = pi05.Pi05Client(endpoint)
        toolkit, more = _base.boot(cell, out_dir, vla_client=client, **options)
    except BaseException:
        if client is not None:
            client.close()
        raise
    return toolkit, [_Closing(client), *more]


def _xpolicylab(options: dict[str, Any]) -> Path:
    """The XPolicyLab checkout whose ``policy/Pi_05`` is served.

    Under XPolicyLab's own policy server (the adapter) it is the ``XPolicyLab`` package that
    server runs from; otherwise the one :func:`.xpolicylab_robodojo.boot` would use.
    """

    explicit = options.get("xpolicylab_root") or os.environ.get("ROBODOJO_XPOLICYLAB_ROOT")
    if not explicit and options.get("bridge") is not None:
        import sys

        package = sys.modules.get("XPolicyLab")
        if package is not None and getattr(package, "__file__", None):
            return Path(package.__file__).resolve().parent
    source = _base._source_root(options.get("source_root"))
    return _base._xpolicylab_root(explicit, source)


class _Closing:
    """The policy client, closed with the daemons."""

    def __init__(self, client: pi05.Pi05Client) -> None:
        self.client = client

    def stop(self) -> None:
        self.client.close()


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
