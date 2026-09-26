# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""What every RPent-backed host shares: locating the checkout and adapting
an RPent ``Toolkit`` to the :class:`pyrualean.Backend` contract.

A host module (``rpent_libero``, ``rpent_robocasa``, ``rpent_robotwin``)
exposes ``Cell``, ``add_cell_args(parser)``, ``add_boot_args(parser)``,
``cell_from_args(args)``, ``boot_options(args)``, ``boot(cell, out_dir,
*, rpent_root=None, env_only=False, **options)`` and
``dump_card(toolkit, cell, out_dir)``; ``pyrualean.play`` drives any of
them through the robot's adapter.  A host may also define
``make_backend(toolkit)`` returning a :class:`RpentBackend` subclass when
RPent's toolkit lacks ``solved()`` / ``primitives`` or counts steps in
another unit than frames.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

RESULT_SCHEMA = "pyrualean-episode-v2"


def resolve_rpent_root(explicit: str | None = None) -> Path:
    """Locate the RPent checkout and make it importable."""

    candidates = [explicit, os.environ.get("RPENT_ROOT"), os.environ.get("RPENT_REPO_ROOT")]
    for value in candidates:
        if value:
            root = Path(value).expanduser().resolve()
            if (root / "rpent").is_dir() and (root / "robots").is_dir():
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                return root
            raise FileNotFoundError(f"not an RPent checkout: {root}")
    try:
        import rpent  # type: ignore[import-not-found]

        root = Path(rpent.__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise FileNotFoundError(
            "RPent not found: set RPENT_ROOT or run inside RPent's environment"
        ) from exc


class RpentBackend:
    """Adapt an RPent ``Toolkit`` to the :class:`pyrualean.Backend` contract."""

    def __init__(self, toolkit: Any):
        self._toolkit = toolkit

    @property
    def toolkit(self) -> Any:
        return self._toolkit

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        return self._toolkit.execute_tool(name, input_dict)

    def solved(self) -> bool:
        """The environment's own success flag (never the agent's claim).

        RPent toolkits normally implement ``solved()``; one that leaves the
        base method unimplemented (RoboTwin) keeps the latest
        ``episode_status`` in ``_latest_status``, whose ``eval_success`` is
        the same flag.  Cheap on purpose: ``robo.done`` asks every turn.
        """
        try:
            return bool(self._toolkit.solved())
        except NotImplementedError:
            status = getattr(self._toolkit, "_latest_status", None) or {}
            return bool(status.get("eval_success"))

    def artifact(self, name: str, step: int = -1) -> bytes:
        return self._toolkit.state.load_bytes(name, step=step)

    def env_steps(self) -> int | None:
        """Frames recorded so far, from ``toolkit.primitives`` (or RPent's
        private ``_primitives``); ``None`` when the toolkit offers neither.

        This is a *frame* count.  A host whose step unit differs (RoboTwin
        counts native VLA actions) returns its own backend from
        ``make_backend(toolkit)`` and overrides this method.
        """
        primitives = getattr(self._toolkit, "primitives", None)
        if primitives is None:
            primitives = getattr(self._toolkit, "_primitives", None)
        try:
            return int(primitives.recorded_frame_count())
        except Exception:  # noqa: BLE001 - optional metadata
            return None

    def cancel(self) -> None:
        self._toolkit.cancel_active_and_wait()


def stop_all(toolkit: Any, daemons: list[Any]) -> None:
    """Close the toolkit and stop its servers, ignoring errors."""

    import contextlib

    if toolkit is not None:
        with contextlib.suppress(Exception):
            toolkit.close()
    for daemon in reversed(daemons):
        with contextlib.suppress(Exception):
            daemon.stop()


__all__ = ["RESULT_SCHEMA", "RpentBackend", "resolve_rpent_root", "stop_all"]
