# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Robot-independent core of a PyRUA-Lean robot class.

A robot class (``LiberoRobot``, ``RobocasaRobot``, ``RobotwinRobot``) is the
only object a policy sees.  Every public method maps 1:1 onto one RPent tool
of that robot and returns a frozen dataclass; this module holds what all of
them share: the backend dispatch with the call ledger, the episode guard
(``EpisodeFinished`` once the task predicate fired or the host stopped the
episode), image / world-map / artifact access and the ``show`` request used
by the ``on-demand`` image policy.

Subclasses set the class attributes below and implement
:meth:`_state_from_envelope` (turn RPent's observation envelope into their
``State`` dataclass) and :meth:`_summary_dict` (a small dict the tool
feedback shows after every turn).  Public methods are rendered into the model-facing
API reference from their signatures and docstrings, so every attribute or
method that is not meant for the policy must start with an underscore.
"""

from __future__ import annotations

import functools
import io
import time
from collections.abc import Callable
from typing import Any, ClassVar

from ._backend import Backend, EpisodeFinished, Ledger, ToolError, raise_for_error, unwrap

#: Primitive sets a run can expose to the policy: ``full`` (every public
#: method) or ``no-vla`` (the learned VLA primitives named by
#: ``VLA_PRIMITIVES`` are hidden from the API reference and refused at call
#: time, so the planner solves the task with the analytic primitives only).
PRIMITIVE_SETS = ("full", "no-vla")
#: What a hidden primitive raises (and what the cell output shows).
NO_VLA_MESSAGE = "{name} is not available in this run (no-VLA primitive set)"


class RobotBase:
    """Shared machinery of every robot class (see the module docstring)."""

    #: RPent robot name (``rpent --robot <NAME>``) and the knowledge file stem.
    NAME: ClassVar[str] = ""
    #: Public methods that run a learned VLA policy.  Under the ``no-vla``
    #: primitive set they are left out of the API reference (with the result
    #: types only they return) and every call raises :data:`NO_VLA_MESSAGE`.
    VLA_PRIMITIVES: ClassVar[tuple[str, ...]] = ()
    #: ``(old, new)`` edits applied to the rendered API reference under
    #: ``no-vla``: the sentences of the *remaining* docstrings that describe
    #: the VLA.  Each ``old`` must occur exactly once, so a docstring change
    #: fails loudly instead of leaking a VLA mention.
    VLA_REFERENCE_EDITS: ClassVar[tuple[tuple[str, str], ...]] = ()
    #: Camera names accepted by :meth:`image`, :meth:`world_map` and :meth:`show`.
    CAMERAS: ClassVar[tuple[str, ...]] = ()
    #: ``(camera, resolution) -> recorded image artifact name``.
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {}
    #: ``(camera, resolution) -> recorded world-map artifact name`` (``.npz``).
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {}
    #: Artifacts attached to a tool result after a turn that moved the robot
    #: (the ``on-motion`` image policy) and the resolution ``show`` uses.
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = ()
    SHOW_RESOLUTION: ClassVar[str] = "high"
    #: Tools that advance the simulator (recorded as stateful in the ledger).
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = frozenset()
    #: Payload keys copied into the ledger summary of a call.
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        "terminated",
        "truncated",
        "success",
        "final_dist_m",
        "final_eef_pos",
        "steps_used",
        "chunks_used",
        "found",
        "score",
        "world_xyz",
        "center_xyz",
        "step",
    )
    #: Read-only tool that returns the observation envelope of a recorded step.
    STATE_TOOL: ClassVar[str] = "view_env_state"

    def __init__(self, backend: Backend, *, ledger: Ledger | None = None, primitives: str = "full"):
        if primitives not in PRIMITIVE_SETS:
            raise ValueError(f"primitives must be one of {PRIMITIVE_SETS}, got {primitives!r}")
        self._backend = backend
        self._ledger = ledger if ledger is not None else Ledger()
        self._last_state: Any = None
        self._done = False
        self._truncated = False
        self._stop_reason: str | None = None
        self._show_requests: list[str] = []
        self._primitives = primitives
        self._hidden: frozenset[str] = (
            frozenset(self.VLA_PRIMITIVES) if primitives == "no-vla" else frozenset()
        )
        # Instance attributes shadow the class methods, so a hidden primitive
        # is refused before any argument validation, whatever it was called with.
        for name in self._hidden:
            setattr(self, name, functools.partial(self._refuse_hidden, name))

    # -- host-facing helpers (not part of the policy API) -------------------

    def _refuse_hidden(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Record and refuse a call to a primitive outside the run's primitive set."""
        message = NO_VLA_MESSAGE.format(name=name)
        record = self._ledger.open(
            name, {"args": list(args), **kwargs} if args else kwargs, stateful=False
        )
        record.error = f"RuntimeError: {message}"
        raise RuntimeError(message)

    def _take_show_requests(self) -> list[str]:
        """Return and clear the cameras requested through :meth:`show`."""
        requests, self._show_requests = self._show_requests, []
        return requests

    def _show_artifacts(self, cameras: list[str]) -> list[str]:
        """Artifact names to attach for ``show`` requests."""
        return [
            self.IMAGE_ARTIFACTS[(camera, self.SHOW_RESOLUTION)]
            for camera in cameras
            if (camera, self.SHOW_RESOLUTION) in self.IMAGE_ARTIFACTS
        ]

    def _stop_episode(self, reason: str) -> None:
        """Make every further motion call raise ``EpisodeFinished``."""
        self._stop_reason = reason
        try:
            self._backend.cancel()
        except Exception:  # noqa: BLE001 - best effort during shutdown
            pass

    # -- to be provided by each robot ---------------------------------------

    def _state_from_envelope(self, envelope: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _summary_dict(self) -> dict[str, Any]:
        """A few numbers describing the current state (host-facing, shown per turn)."""
        raise NotImplementedError

    # -- observation shared by every robot ----------------------------------

    @property
    def task(self) -> str:
        """The task instruction given by the benchmark (authoritative)."""
        return str(getattr(self._current_state(), "task", ""))

    @property
    def done(self) -> bool:
        """True once the benchmark's success predicate has fired (sticky)."""
        return self._done or bool(self._backend.solved())

    def state(self, step: int = -1) -> Any:
        """Read the recorded proprioceptive state of step ``step`` (``-1`` = latest)."""
        payload = self._readonly(self.STATE_TOOL, {"step": int(step)})
        state = self._state_from_envelope(payload)
        if step == -1:
            self._last_state = state
        return state

    def image(self, camera: str, *, resolution: str = "high", step: int = -1) -> Any:
        """Load a recorded camera image as a ``numpy`` uint8 array (H, W, 3), RGB.

        Pixel coordinates are (row, col) with row 0 at the top and col 0 at the
        left; pass the same ``camera`` and ``resolution`` to the back-projection
        helpers.  ``step`` selects the recorded step (``-1`` = latest).
        """
        data = self.artifact(self._image_name(camera, resolution), step=step)
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"))

    def world_map(self, camera: str, *, resolution: str = "high", step: int = -1) -> Any:
        """Load the per-pixel world coordinates of a recorded image.

        Returns a ``numpy`` float array (H, W, 3): ``world_map[row, col]`` is
        the [x, y, z] of the surface seen at that pixel (zeros / non-finite
        where depth is invalid).  Load it to reason about many pixels at once.
        """
        key = (camera, resolution)
        if key not in self.WORLD_MAP_ARTIFACTS:
            raise ValueError(
                f"no world map for camera={camera!r} resolution={resolution!r}; "
                f"available: {sorted(self.WORLD_MAP_ARTIFACTS)}"
            )
        data = self.artifact(self.WORLD_MAP_ARTIFACTS[key], step=step)
        import numpy as np

        loaded = np.load(io.BytesIO(data))
        if isinstance(loaded, np.ndarray):  # a bare .npy array
            return loaded
        with loaded as archive:  # an .npz archive: "array" or its first member
            name = "array" if "array" in archive.files else archive.files[0]
            return np.asarray(archive[name])

    def show(self, camera: str) -> None:
        """Attach the current image of ``camera`` to the feedback of this turn.

        Images are not returned automatically: call ``show(<camera>)`` when
        *you* (the author) need to look at the scene before writing the next
        turn.  Costs tokens: look only when a decision depends on it.
        """
        if camera not in self.CAMERAS:
            raise ValueError(f"camera must be one of {self.CAMERAS}")
        if camera not in self._show_requests:
            self._show_requests.append(camera)

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return the raw bytes of a recorded artifact by name (advanced)."""
        record = self._ledger.open("artifact", {"name": name, "step": step}, stateful=False)
        started = time.perf_counter()
        try:
            data = self._backend.artifact(name, step=int(step))
            record.ok = True
            record.summary = {"bytes": len(data)}
            return data
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.elapsed_s = time.perf_counter() - started

    # -- internals ----------------------------------------------------------

    @classmethod
    def _image_name(cls, camera: str, resolution: str) -> str:
        key = (camera, resolution)
        if key not in cls.IMAGE_ARTIFACTS:
            raise ValueError(
                f"camera must be one of {cls.CAMERAS} and resolution one of "
                f"{sorted({r for _, r in cls.IMAGE_ARTIFACTS})}"
            )
        return cls.IMAGE_ARTIFACTS[key]

    def _current_state(self) -> Any:
        if self._last_state is None:
            return self.state()
        return self._last_state

    def _guard(self) -> None:
        if self._stop_reason is not None:
            raise EpisodeFinished(self._stop_reason)
        if self._done:
            raise EpisodeFinished("terminated")
        if self._truncated:
            raise EpisodeFinished("truncated")

    def _absorb(self, envelope: dict[str, Any] | None) -> None:
        """Record the latest state and latch the episode flags.

        ``terminated`` / ``truncated`` come from the envelope and from the
        parsed state when it has attributes of those names, so a robot whose
        success flag RPent names differently (``eval_success``) or whose step
        budget RPent never reports maps them in ``_state_from_envelope``
        instead of overriding this method.
        """
        if envelope is None:
            return
        state = self._state_from_envelope(envelope)
        self._last_state = state
        if envelope.get("terminated") or getattr(state, "terminated", False):
            self._done = True
        if envelope.get("truncated") or getattr(state, "truncated", False):
            self._truncated = True

    def _summary(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {key: payload[key] for key in self.SUMMARY_KEYS if key in payload}

    def _readonly(
        self, tool: str, kwargs: dict[str, Any], *, allow_error_key: str | None = None
    ) -> dict[str, Any]:
        record = self._ledger.open(tool, kwargs, stateful=False)
        started = time.perf_counter()
        try:
            payload, envelope = unwrap(self._backend.execute_tool(tool, kwargs), tool)
            if envelope is not None:
                # The state tool returns the observation envelope itself; for a
                # read-only tool that envelope *is* the payload.
                self._absorb(envelope)
                payload = envelope
            if allow_error_key is None or allow_error_key not in payload:
                raise_for_error(tool, payload)
            record.ok = True
            record.summary = self._summary(tool, payload)
            return payload
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.elapsed_s = time.perf_counter() - started

    def _stateful(
        self,
        tool: str,
        kwargs: dict[str, Any],
        build: Callable[[dict[str, Any]], Any],
    ) -> Any:
        self._guard()
        record = self._ledger.open(tool, kwargs, stateful=True)
        steps_before = self._backend.env_steps()
        started = time.perf_counter()
        try:
            payload, envelope = unwrap(self._backend.execute_tool(tool, kwargs), tool)
            self._absorb(envelope)
            if payload.get("interrupted") or payload.get("code") == "tool_cancelled":
                raise EpisodeFinished(self._stop_reason or "cancelled")
            raise_for_error(tool, payload)
            result = build(payload)
            record.ok = True
            record.summary = self._summary(tool, payload)
            return result
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.elapsed_s = time.perf_counter() - started
            steps_after = self._backend.env_steps()
            if steps_before is not None and steps_after is not None:
                record.env_steps = max(0, steps_after - steps_before)


__all__ = ["NO_VLA_MESSAGE", "PRIMITIVE_SETS", "RobotBase", "ToolError"]
