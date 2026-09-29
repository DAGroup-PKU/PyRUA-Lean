# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboCasa365 robot API for code policies.

Every public method of :class:`RobocasaRobot` maps 1:1 onto one RPent RoboCasa
tool (the same OSC arm servos and base controller, the same frozen RLDX-1 VLA
calls and the same world-map queries that RPent's tool-calling agent uses),
with RPent's argument names and defaults.  The difference is the calling
convention: a policy is an ordinary Python program that calls these methods
and branches on their typed return values.

The docstrings in this module are the model-facing documentation: the prompt
is generated from them (see :mod:`pyrualean.prompt`), so keep them accurate.
"""

from __future__ import annotations

import io
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar

from ._backend import ToolError
from ._robot import RobotBase
from .robots import RobotAdapter

#: Largest end-effector displacement RPent allows in one servo call.  A longer
#: traversal flips the OSC controller's IK, so the library refuses it.
MAX_MOVE_M = 0.30
#: Env steps executed per RLDX-1 action chunk (``n_action_steps``; the
#: evaluation protocol may pin it through ``RLDX_ACTION_STEPS_PER_CHUNK``).
VLA_CHUNK_STEPS = 8
#: Pixels one ``back_project_batch`` call accepts.
MAX_PIXELS = 50
#: Side of every recorded camera image and world map, in pixels.
IMAGE_SIZE = 256
#: Recorded steps whose heavy world / depth maps RPent keeps on disk.
WORLD_MAP_WINDOW = 25


# ---------------------------------------------------------------------------
# Result types (returned to the policy; all fields are plain Python values)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """Snapshot of the robot and of the episode at one recorded step.

    step: index of the recorded step (0 = initial scene).  A new step is
        recorded after every motion call.
    task: the task instruction given by the benchmark (authoritative; the VLA
        acts on exactly this text).
    eef_pos: gripper (end-effector) position [x, y, z] in the world frame,
        metres.  This is what ``move_to`` servos and what ``back_project_batch``
        coordinates are directly comparable to.
    eef_quat: gripper orientation as a unit quaternion [x, y, z, w].
    gripper_qpos: the two finger joint positions (q0 >= 0, q1 <= 0).
    gripper_opening: finger separation proxy |q0| + |q1|: about 0.04 at the
        start (fingers half open), about 0.08 fully open, about 0.0 closed on
        nothing; a value that stays in between after closing means the
        fingers stopped on something.
    base_pos: mobile-base position [x, y, z] in the world frame, metres.
    base_quat: base orientation [x, y, z, w]; ``base_yaw`` is its heading.
    success: the benchmark's own success predicate (``_check_success``) has
        fired.  This is the only definition of success; it equals
        ``terminated`` and ``robo.done``.
    task_progress: the live values the success predicate computes (its
        counters, flags and sub-predicates, e.g. a joint fraction, a
        ``success_time`` counter, ``is_open`` flags).  The keys are
        task-specific: read them at step 0, then compare before and after an
        action to see whether the action moved the criterion.  May be empty.
    vla_desync: a non-VLA primitive ran since the last VLA call, so the next
        ``rldx_skill`` / ``rldx_arm`` starts from a fresh frame history.
    terminated: same as ``success``.
    truncated: RPent never reports a step budget for this robot (always
        False); the host's wall-clock and decision budgets end an episode.
    """

    step: int
    task: str
    eef_pos: tuple[float, float, float]
    eef_quat: tuple[float, float, float, float]
    gripper_qpos: tuple[float, float]
    gripper_opening: float
    base_pos: tuple[float, float, float]
    base_quat: tuple[float, float, float, float]
    success: bool
    task_progress: dict[str, Any]
    vla_desync: bool
    terminated: bool
    truncated: bool

    @property
    def base_yaw(self) -> float:
        """Heading of the base about world z in radians (0 = facing +x, pi/2 = facing +y)."""
        x, y, z, w = self.base_quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @property
    def eef_approach(self) -> tuple[float, float, float]:
        """Unit vector the fingers point along, world frame ((0, 0, -1) = straight down)."""
        x, y, z, w = self.eef_quat
        return (2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))

    @property
    def eef_tilt(self) -> float:
        """Angle between the approach axis and straight down, radians (0 = down, pi/2 = level)."""
        ax, ay, az = self.eef_approach
        return math.atan2(math.hypot(ax, ay), -az)


@dataclass(frozen=True)
class Move:
    """Outcome of ``move_to`` / ``move_delta`` (a closed-loop OSC servo).

    target_xyz: the commanded world target.
    final_eef_pos: where the gripper actually stopped (metres).
    final_dist_m: distance between the two.
    reached: ``final_dist_m < tol``.  False means the servo used its whole
        step budget without converging (out of reach, contact, joint limit):
        re-plan (navigate closer, change the approach) instead of repeating it.
    steps_used / max_steps: env steps consumed / allowed.
    gripper_q0: first finger joint after the move (RPent's gripper width
        proxy: about 0.02 half open, about 0.0 closed on nothing).
    terminated: the benchmark's success predicate after the move
        (same as ``robo.done``).
    """

    target_xyz: tuple[float, float, float]
    final_eef_pos: tuple[float, float, float]
    final_dist_m: float
    reached: bool
    steps_used: int
    max_steps: int
    gripper_q0: float
    terminated: bool


@dataclass(frozen=True)
class Pitch:
    """Outcome of ``rotate_pitch``.

    No measured angle is reported by the primitive: read
    ``state().eef_tilt`` / ``state().eef_approach`` afterwards.
    """

    target_pitch: float
    final_eef_pos: tuple[float, float, float]
    steps_used: int
    terminated: bool


@dataclass(frozen=True)
class Gripper:
    """Outcome of ``set_gripper`` / ``release``.

    gripper: the command that was held (+1 close, -1 open); steps: for how
        many env steps.  ``gripper_qpos`` / ``gripper_opening`` are the finger
        positions afterwards (a position, not proof of a hold: verify a
        grasp from the wrist image and from the object moving with the
        gripper).
    """

    gripper: float
    steps: int
    gripper_qpos: tuple[float, float]
    gripper_opening: float
    terminated: bool


@dataclass(frozen=True)
class Grasp:
    """Outcome of ``scripted_grasp`` (open, hover, descend, close, lift).

    ok: every stage servoed to its target.  ``stage`` names the stage that
        stalled (``approach`` | ``descent`` | ``lift``) or is None when ok;
        ``final_dist_m`` is that stage's remaining distance (0 when ok).
    final_eef_pos: where the gripper ended (after the lift when ok).
    gripper_opening: finger separation afterwards; a value clearly above 0
        after the close means the fingers stopped on something.
    """

    ok: bool
    stage: str | None
    final_eef_pos: tuple[float, float, float]
    final_dist_m: float
    gripper_qpos: tuple[float, float]
    gripper_opening: float
    terminated: bool


@dataclass(frozen=True)
class Navigation:
    """Outcome of ``navigate_to`` (drive the base towards a world x/y).

    reached: the base came within ``tol`` of the target and faces it.
    stuck: it ran out of steps having moved less than 0.12 m (it rammed a
        fixture: there is no path planning).  Back off with ``move_base``
        and approach from another side.
    final_dist_m: remaining distance to the target; moved_m: distance
        travelled from ``start_xy``; base_pos: the base afterwards.
    steps_used / max_steps: env steps consumed / allowed.
    """

    target_xy: tuple[float, float]
    reached: bool
    stuck: bool
    final_dist_m: float
    moved_m: float
    steps_used: int
    max_steps: int
    start_xy: tuple[float, float]
    base_pos: tuple[float, float, float]
    terminated: bool


@dataclass(frozen=True)
class BaseMove:
    """Outcome of ``move_base`` (raw base velocities in the robot's local frame).

    base_moved: world displacement [dx, dy, dz] of the base over the call;
    base_pos: the base afterwards; steps: env steps executed.
    """

    base_moved: tuple[float, float, float]
    base_pos: tuple[float, float, float]
    steps: int
    terminated: bool


@dataclass(frozen=True)
class Skill:
    """Outcome of ``rldx_skill`` / ``rldx_arm`` (a closed-loop RLDX-1 run).

    status: why the run stopped: ``success`` (the benchmark predicate fired),
        ``settled`` (arm and gripper stopped changing for ``settle_patience``
        chunks) or ``cap`` (``max_chunks`` reached; the VLA was still acting:
        call again to continue with its frame history intact).
    chunks_used / steps_applied: action chunks and env steps executed.
    max_chunks: the chunk cap that applied (the evaluation protocol may pin
        it through the environment, overriding the argument).
    grasped: holding something at the end of the call (carry-ready).
    grasp_detected: held something at some chunk (it may have been placed
        or dropped since).
    grasp_contact: both fingerpads touch the same task object right now (the
        benchmark's own grasp check; works for any grasp direction).
    held_apart: the VLA commanded a close and the fingers stopped apart, i.e.
        something is between them (also a fixture handle, which
        ``grasp_contact`` cannot see).  grasp_obj: the contacted object's
        name, or None.
    gripper_q0: first finger joint at the end; peak_lift_m: highest rise of
        the gripper above its lowest point during the run; base_drift_m: how
        far the base moved; base_clip: the base-motion cap that applied.
    instruction: the task text the policy received (always ``robo.task``).
    terminated: the benchmark's success predicate after the run.
    """

    status: str
    chunks_used: int
    steps_applied: int
    max_chunks: int
    grasped: bool
    grasp_detected: bool
    grasp_contact: bool
    held_apart: bool
    grasp_obj: str | None
    gripper_q0: float
    peak_lift_m: float
    base_drift_m: float
    base_clip: float | None
    instruction: str
    terminated: bool


@dataclass(frozen=True)
class Point:
    """One pixel of ``back_project_batch``: its world coordinate or why it has none.

    world_xyz: [x, y, z] of the surface seen at ``pixel`` (metres), None when
        the pixel is out of bounds or has no valid depth (``error`` says
        which).
    """

    pixel: tuple[int, int]
    world_xyz: tuple[float, float, float] | None
    error: str | None

    @property
    def valid(self) -> bool:
        """True when ``world_xyz`` is available."""
        return self.world_xyz is not None


@dataclass(frozen=True)
class BackProjection:
    """Outcome of ``back_project_batch``: one ``Point`` per requested pixel, in order.

    median_xyz: per-axis median of the valid points (None when none are
        valid): the robust object coordinate when the pixels lie on one
        object's top surface.  valid_count / total_count: how many pixels
        had a world coordinate.  step / camera: the world map that was read.
    """

    points: tuple[Point, ...]
    median_xyz: tuple[float, float, float] | None
    valid_count: int
    total_count: int
    camera: str
    resolution: str
    step: int

    @property
    def xyz(self) -> tuple[tuple[float, float, float], ...]:
        """The valid world coordinates only, in request order."""
        return tuple(p.world_xyz for p in self.points if p.world_xyz is not None)


@dataclass(frozen=True)
class Cluster:
    """One cluster of ``query_world_map``: pixels of one grid cell inside the height band.

    center_xyz: median world point of the cluster; bbox_min / bbox_max: its
        world extent; pixel_count: matched pixels; sample_pixel: one
        ``(row, col)`` inside it (look there in the image to see what it is).
    """

    center_xyz: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    pixel_count: int
    sample_pixel: tuple[int, int]


@dataclass(frozen=True)
class HeightQuery:
    """Outcome of ``query_world_map``: clusters of pixels inside a world-z band.

    clusters: largest first (at most 20); total_pixels: pixels that matched
        the band and the optional x/y ranges before clustering.  Clustering
        is a coarse image grid, not connected components: one object can
        span two clusters and two touching objects can share one.
    """

    clusters: tuple[Cluster, ...]
    total_pixels: int
    camera: str
    resolution: str


STATEFUL_TOOLS = frozenset(
    {
        "move_to",
        "move_delta",
        "rotate_pitch",
        "set_gripper",
        "release",
        "scripted_grasp",
        "navigate_to",
        "move_base",
        "rldx_skill",
        "rldx_arm",
    }
)


def _floats(value: Any, n: int, name: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {n} numbers, got {value!r}") from exc
    if len(vals) != n or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} must be {n} finite numbers, got {value!r}")
    return vals


def _tuple_n(value: Any, n: int) -> tuple[float, ...] | None:
    """``value`` as an n-tuple of floats, None when absent or malformed (numpy-safe)."""
    if value is None:
        return None
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return vals if len(vals) == n else None


def _tuple3(value: Any) -> tuple[float, float, float] | None:
    return _tuple_n(value, 3)  # type: ignore[return-value]


def _gripper(value: Any) -> float | str:
    """RPent's tri-state gripper argument: +1 close, -1 open or ``"hold"``."""
    if value is None or value == "hold":
        return "hold"
    if isinstance(value, str):
        raise ValueError("gripper must be +1 (close), -1 (open) or 'hold'")
    return _gripper_command(value)


def _gripper_command(value: Any) -> float:
    try:
        g = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gripper must be a number in [-1, +1]: +1 closes, -1 opens") from exc
    if not -1.0 <= g <= 1.0 or not math.isfinite(g):
        raise ValueError("gripper must be within [-1, +1]: +1 closes, -1 opens")
    return g


def _pixels(pixels: Sequence[Sequence[int]]) -> list[list[int]]:
    if len(pixels) == 2 and all(isinstance(v, Real) for v in pixels):
        pixels = [pixels]  # a single [row, col] pair
    if not 1 <= len(pixels) <= MAX_PIXELS:
        raise ValueError(f"pixels must contain between 1 and {MAX_PIXELS} [row, col] pairs")
    out = []
    for pixel in pixels:
        if len(pixel) != 2:
            raise ValueError(f"every pixel must be a [row, col] pair, got {pixel!r}")
        out.append([int(pixel[0]), int(pixel[1])])
    return out


# ---------------------------------------------------------------------------
# The robot
# ---------------------------------------------------------------------------


class RobocasaRobot(RobotBase):
    """A mobile-base PandaOmron (Franka arm with a two-finger gripper on an
    omnidirectional base) in a RoboCasa365 kitchen.

    World frame: x/y span the kitchen (coordinates run over several metres),
    z is up; metres and radians; quaternions are ``[x, y, z, w]``.
    ``state().eef_pos`` is the gripper position in this frame,
    ``state().base_pos`` the base position and ``state().base_yaw`` its
    heading.  Counters are at about z = 0.9 m; read every height from the
    world maps, never assume one.  The arm reaches about 0.8 m from the
    base: a target farther away in x/y needs ``navigate_to`` first.  Every
    base motion (``navigate_to``, ``move_base``, a VLA run) moves the arm and
    all three cameras with it, so re-localise before the next arm target.

    Gripper commands: ``robo.CLOSE`` (+1) closes and keeps squeezing,
    ``robo.OPEN`` (-1) opens, ``robo.HOLD`` (``"hold"``) servos the fingers to
    the width they had when the motion began.  ``"hold"`` is the default of
    every motion call and the carry-safe choice: a sustained +1 squeezes a
    small object out of the fingers, and carrying with -1 drops it.

    Perception is camera based: three 256x256 cameras ride on the robot,
    ``agentview`` (shoulder view: what is where), ``wrist`` (on the gripper:
    close-range geometry) and ``navview`` (forward-down floor view: where
    to drive), each with a per-pixel world map.  Object poses are never
    given.

    The frozen RLDX-1 VLA (``rldx_skill``, ``rldx_arm``) always receives the
    benchmark's own task instruction (``robo.task``); there is no prompt to
    write.  Success is the benchmark's own predicate (``state().success``,
    ``robo.done``); ``state().task_progress`` shows the values that predicate
    computes and ``success_criteria()`` its source.  Motion calls block until
    the primitive finishes and return a typed result; nothing is queued.
    """

    OPEN: float = -1.0
    CLOSE: float = 1.0
    HOLD: str = "hold"
    MAX_MOVE_M: float = MAX_MOVE_M

    #: Rendered as the result-type section of the API reference (not a constant).
    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        Move,
        Pitch,
        Gripper,
        Grasp,
        Navigation,
        BaseMove,
        Skill,
        BackProjection,
        Point,
        HeightQuery,
        Cluster,
    )
    NAME: ClassVar[str] = "robocasa"
    CAMERAS: ClassVar[tuple[str, ...]] = ("agentview", "navview", "wrist")
    #: RPent records one 256x256 resolution ("low") per camera; the optional
    #: hi-res agentview needs RPent's ``--hi-res`` and is not booted here.
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("agentview", "low"): "agentview.png",
        ("navview", "low"): "navview.png",
        ("wrist", "low"): "wrist.png",
    }
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("agentview", "low"): "agentview_world.npz",
        ("navview", "low"): "navview_world.npz",
        ("wrist", "low"): "wrist_world.npz",
    }
    #: Only agentview and wrist have depth maps and calibration; navview has neither.
    DEPTH_ARTIFACTS: ClassVar[dict[str, str]] = {
        "agentview": "agentview_depth.npz",
        "wrist": "wrist_depth.npz",
    }
    CAMERA_META_ARTIFACTS: ClassVar[dict[str, str]] = {
        "agentview": "agentview_metadata.json",
        "wrist": "wrist_metadata.json",
    }
    # RPent embeds all three views after every state-changing tool; keep parity.
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = ("agentview.png", "navview.png", "wrist.png")
    SHOW_RESOLUTION: ClassVar[str] = "low"
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = STATEFUL_TOOLS
    #: The frozen RLDX-1 skills (hidden and refused under the no-VLA primitive set).
    VLA_PRIMITIVES: ClassVar[tuple[str, ...]] = ("rldx_skill", "rldx_arm")
    #: VLA sentences of the remaining docstrings, removed from the no-VLA reference.
    VLA_REFERENCE_EDITS: ClassVar[tuple[tuple[str, str], ...]] = (
        (
            "    base motion (``navigate_to``, ``move_base``, a VLA run) moves the arm and\n",
            "    base motion (``navigate_to``, ``move_base``) moves the arm and\n",
        ),
        (
            "\n\n    The frozen RLDX-1 VLA (``rldx_skill``, ``rldx_arm``) always receives the\n"
            "    benchmark's own task instruction (``robo.task``); there is no prompt to\n"
            "    write.  Success is the benchmark's own predicate",
            "\n\n    Success is the benchmark's own predicate",
        ),
        (
            "    It is authoritative and the VLA acts on exactly this text.",
            "    It is authoritative.",
        ),
        (
            "        success flag, ``task_progress`` and ``vla_desync``.",
            "        success flag and ``task_progress``.",
        ),
        (
            "    A fallback for simple, well-localised objects; prefer ``rldx_arm``\n"
            "    for anything hard.  ``grasp_z_offset`` shifts",
            "    A fallback for simple, well-localised objects.  ``grasp_z_offset`` shifts",
        ),
        ("    vla_desync: bool,\n", ""),
        (
            "    task: the task instruction given by the benchmark (authoritative; the VLA\n"
            "        acts on exactly this text).",
            "    task: the task instruction given by the benchmark (authoritative).",
        ),
        (
            "    vla_desync: a non-VLA primitive ran since the last VLA call, so the next\n"
            "        ``rldx_skill`` / ``rldx_arm`` starts from a fresh frame history.\n",
            "",
        ),
    )
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        "ok",
        "steps",
        "final_dist",
        "moved",
        "stuck",
        "stage",
        "status",
        "chunks",
        "steps_applied",
        "grasped",
        "grasp_detected",
        "grasp_obj",
        "peak_lift",
        "base_drift",
        "prompt_overridden",
        "state_capture_error",
        "step",
    )

    # -- observation --------------------------------------------------------

    @property
    def task(self) -> str:
        """The task instruction, e.g. ``"Open the left drawer."``.

        It is authoritative and the VLA acts on exactly this text.
        """
        return super().task

    @property
    def done(self) -> bool:
        """True once the benchmark's success predicate (``state().success``) has fired (sticky)."""
        return super().done

    def state(self, step: int = -1) -> State:
        """Read the recorded state of the robot and of the episode.

        Args:
            step: recorded step to read; ``-1`` is the latest, ``0`` the
                initial scene.  A new step is recorded after every motion call.
        Returns:
            State with the gripper pose, finger positions, base pose, the
            success flag, ``task_progress`` and ``vla_desync``.
        """
        return super().state(step)

    def image(self, camera: str = "agentview", *, resolution: str = "low", step: int = -1) -> Any:
        """Load a recorded camera image as a ``numpy`` uint8 array (256, 256, 3), RGB.

        Args:
            camera: ``"agentview"`` (shoulder camera: identify objects,
                fixtures and the layout; its pixels back-project directly),
                ``"wrist"`` (on the gripper: close-range geometry, best within
                20 cm of a target) or ``"navview"`` (forward-down view of the
                floor ahead of the base: where driving is possible).  All
                three move with the robot.
            resolution: there is one resolution, ``"low"`` (256x256).  Pixel
                coordinates are (row, col) with row 0 at the top and col 0 at
                the left; pass the same camera and step to
                ``back_project_batch``.
            step: recorded step, ``-1`` = latest.
        """
        return super().image(camera, resolution=resolution, step=step)

    def world_map(
        self, camera: str = "agentview", *, resolution: str = "low", step: int = -1
    ) -> Any:
        """Load the per-pixel world coordinates of a recorded image.

        Returns a ``numpy`` float32 array (256, 256, 3): ``world_map[row, col]``
        is the [x, y, z] of the surface seen at that pixel (zeros where the
        camera saw no depth).  This is what ``back_project_batch`` and
        ``query_world_map`` read; load it to reason about many pixels at once
        (e.g. ``np.median(world_map[r0:r1, c0:c1, 2])`` is a surface height).
        Maps older than 25 recorded steps are deleted; a missing map raises
        ``ToolError``.
        """
        if (camera, resolution) not in self.WORLD_MAP_ARTIFACTS:
            raise ValueError(
                f"camera must be one of {self.CAMERAS} and resolution 'low'; "
                f"got {(camera, resolution)!r}"
            )
        try:
            return super().world_map(camera, resolution=resolution, step=step)
        except FileNotFoundError as exc:
            raise ToolError(
                "world_map",
                f"no {camera} world map recorded for step {step} (maps older than "
                f"{WORLD_MAP_WINDOW} steps are deleted)",
            ) from exc

    def depth(self, camera: str = "agentview", *, step: int = -1) -> Any:
        """Load the metric depth of a recorded ``agentview`` or ``wrist`` image.

        Returns a ``numpy`` float32 array (256, 256) in metres.  ``navview``
        has no depth map (use its world map).
        """
        if camera not in self.DEPTH_ARTIFACTS:
            raise ValueError(f"depth is recorded for {sorted(self.DEPTH_ARTIFACTS)} only")
        import numpy as np

        data = self.artifact(self.DEPTH_ARTIFACTS[camera], step=step)
        with np.load(io.BytesIO(data)) as archive:
            return np.asarray(archive["array"])

    def floor_overlay(self, step: int = -1) -> Any:
        """Load the navview image with the floor painted green: uint8 (256, 256, 3), RGB.

        Floor = pixels whose world z lies between -0.2 and 0.12 m; green
        area ahead of the base is where it can drive.
        """
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(self.artifact("navview_floor.png", step=step))) as img:
            return np.asarray(img.convert("RGB"))

    def camera_meta(self, camera: str = "agentview", *, step: int = -1) -> dict[str, Any]:
        """Return the camera calibration of a recorded step as a dict (``agentview`` / ``wrist``).

        Keys: ``intrinsic`` (3x3), ``extrinsic_cam2world`` (4x4), ``height``,
        ``width``, ``depth_near`` / ``depth_far``.  Both cameras move with the
        robot (the wrist camera with the gripper), so read the meta of the
        same step as the image.
        """
        if camera not in self.CAMERA_META_ARTIFACTS:
            raise ValueError(
                f"calibration is recorded for {sorted(self.CAMERA_META_ARTIFACTS)} only"
            )
        return json.loads(self.artifact(self.CAMERA_META_ARTIFACTS[camera], step=step))

    def success_criteria(self) -> str:
        """Return the source code of this task's success predicate (``_check_success``).

        It names the objects and fixtures that matter and the thresholds
        (e.g. a drawer joint fraction of 0.95, a ``success_time`` counter)
        that ``state().task_progress`` reports live.  It contains no object
        coordinates.
        """
        return self._session_artifact("success_criteria.md").decode("utf-8", errors="replace")

    def show(self, camera: str = "agentview") -> None:
        """Attach the current image of ``camera`` to the feedback of this turn.

        Images are not returned automatically; call ``show("agentview")``,
        ``show("wrist")`` or ``show("navview")`` when *you* (the author) need
        to look at the scene before writing the next turn.  Costs tokens:
        look only when a decision depends on it.
        """
        super().show(camera)

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return the raw bytes of a recorded artifact (advanced).

        Names are ``agentview.png``, ``wrist.png``, ``navview.png``,
        ``navview_floor.png``, ``<camera>_world.npz`` (all three cameras),
        ``<camera>_depth.npz`` and ``<camera>_metadata.json`` (agentview and
        wrist only).
        """
        return super().artifact(name, step=step)

    # -- perception -------------------------------------------------------

    def back_project_batch(
        self,
        pixels: Sequence[Sequence[int]],
        *,
        step: int = -1,
        camera: str = "agentview",
        resolution: str = "low",
    ) -> BackProjection:
        """Return the world [x, y, z] under image pixels of a recorded view.

        Args:
            pixels: 1 to 50 ``[row, col]`` pairs read off ``image(camera,
                step=step)`` (a single pair is accepted too).
            step: recorded step whose world map to read, ``-1`` = latest.
            camera: ``"agentview"``, ``"wrist"`` or ``"navview"``: the pixel
                coordinate space, so use the view the pixels came from.
            resolution: ``"low"`` (the only one recorded).
        Returns:
            BackProjection; ``median_xyz`` is the robust coordinate of what
            the pixels cover.  Every point is a visible SURFACE point (the
            top of an object, not its centre): sample 3-8 pixels firmly on
            the object's top surface and use the median; pixels on thin rims,
            edges, shadows or the gap to a counter hit the background.
        Raises:
            ToolError: no world map recorded for that step / camera.
        """
        if camera not in self.CAMERAS:
            raise ValueError(f"camera must be one of {self.CAMERAS}")
        if resolution != "low":
            raise ValueError("resolution must be 'low' (the only recorded resolution)")
        kwargs: dict[str, Any] = {
            "pixels": _pixels(pixels),
            "step": int(step),
            "camera": camera,
            "resolution": resolution,
        }
        payload = self._readonly("back_project_batch", kwargs)
        points = tuple(
            Point(
                pixel=(int(r["pixel"][0]), int(r["pixel"][1])),
                world_xyz=_tuple3(r.get("world_xyz")),
                error=(str(r["error"]) if r.get("error") else None),
            )
            for r in payload.get("results") or ()
        )
        summary = payload.get("summary") or {}
        return BackProjection(
            points=points,
            median_xyz=_tuple3(summary.get("median_xyz")),
            valid_count=int(summary.get("valid_count", 0)),
            total_count=int(summary.get("total_count", len(points))),
            camera=str(payload.get("camera", camera)),
            resolution=str(payload.get("resolution", resolution)),
            step=int(payload.get("step", step)),
        )

    def query_world_map(
        self,
        z_min: float = 0.85,
        z_max: float = 0.95,
        *,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        camera: str = "agentview",
        resolution: str = "low",
        min_cluster_size: int = 10,
    ) -> HeightQuery:
        """Find what lies inside a world-z band of the LATEST world map, as pixel clusters.

        Args:
            z_min / z_max: height band in metres.  0.85-0.95 finds objects on
                a counter; ``camera="navview"`` with 0.0-0.12 finds walkable
                floor.
            x_range / y_range: optional ``[min, max]`` world limits.
            camera: which camera's world map (``"agentview"``, ``"wrist"``,
                ``"navview"``).
            resolution: ``"low"`` (the only one recorded).
            min_cluster_size: smallest cluster reported (pixels).
        Returns:
            HeightQuery; each ``Cluster.center_xyz`` is a candidate location,
            ``sample_pixel`` a place to look in the image to identify it.
            Always reads the latest recorded step.
        """
        if camera not in self.CAMERAS:
            raise ValueError(f"camera must be one of {self.CAMERAS}")
        if resolution != "low":
            raise ValueError("resolution must be 'low' (the only recorded resolution)")
        if not float(z_min) <= float(z_max):
            raise ValueError("z_min must not exceed z_max")
        if int(min_cluster_size) < 1:
            raise ValueError("min_cluster_size must be at least 1")
        kwargs: dict[str, Any] = {
            "z_min": float(z_min),
            "z_max": float(z_max),
            "x_range": list(_floats(x_range, 2, "x_range")) if x_range is not None else None,
            "y_range": list(_floats(y_range, 2, "y_range")) if y_range is not None else None,
            "camera": camera,
            "resolution": resolution,
            "min_cluster_size": int(min_cluster_size),
        }
        payload = self._readonly("query_world_map", kwargs)
        nan3 = (math.nan, math.nan, math.nan)
        clusters = []
        for c in payload.get("clusters") or ():
            bbox = c.get("bbox_xyz") or {}
            samples = c.get("sample_pixels") or [[0, 0]]
            clusters.append(
                Cluster(
                    center_xyz=_tuple3(c.get("center_xyz")) or nan3,
                    bbox_min=_tuple3(bbox.get("min")) or nan3,
                    bbox_max=_tuple3(bbox.get("max")) or nan3,
                    pixel_count=int(c.get("pixel_count", 0)),
                    sample_pixel=(int(samples[0][0]), int(samples[0][1])),
                )
            )
        summary = payload.get("summary") or {}
        return HeightQuery(
            clusters=tuple(clusters),
            total_pixels=int(summary.get("total_pixels_matched", 0)),
            camera=camera,
            resolution=resolution,
        )

    # -- the VLA ------------------------------------------------------------

    def rldx_skill(
        self,
        *,
        base_clip: float | None = None,
        max_chunks: int = 70,
        force_reset: bool = False,
        n_action_steps: int = 8,
        settle_patience: int = 999,
        settle_eps: float = 0.012,
    ) -> Skill:
        """Let the frozen RLDX-1 VLA drive the whole body (arm AND base) on the task instruction.

        The policy always receives the benchmark's own instruction
        (``robo.task``) and the three current camera images, executes
        ``n_action_steps`` env steps per chunk and stops when the task
        succeeds, when it settles or at ``max_chunks``.  It is the tool for
        grasps, re-grasps, pulling / pushing fixtures, pressing controls and
        other contact-rich motion, and it may reposition the base by itself.

        Args:
            base_clip: cap on the base velocity commands (None = full base
                motion; ``rldx_arm`` is the same skill with 0.1).
            max_chunks: action-chunk cap.  The evaluation protocol may pin it
                (and ``n_action_steps`` / ``settle_patience``) through the
                environment; ``Skill.max_chunks`` reports what applied.
            force_reset: start from a fresh frame history even after a
                previous VLA call (the runtime does this by itself whenever
                a non-VLA primitive ran in between: see
                ``state().vla_desync``).
            n_action_steps: env steps executed per predicted chunk.
            settle_patience: chunks without arm / gripper motion after which
                the run stops as ``settled``; leave the default (999 disables
                it: the VLA is expected to run until success or the cap).
            settle_eps: motion threshold of that test, metres.
        Returns:
            Skill; check ``status``, the grasp signals and ``robo.done``.
            ``cap`` means the VLA was still acting: call again (its frame
            history continues) rather than staging by hand.
        """
        kwargs = self._skill_kwargs(
            base_clip, max_chunks, force_reset, n_action_steps, settle_patience, settle_eps
        )
        return self._stateful("rldx_skill", kwargs, lambda p: self._skill(p, kwargs))

    def rldx_arm(
        self,
        *,
        base_clip: float | None = 0.1,
        max_chunks: int = 70,
        force_reset: bool = False,
        n_action_steps: int = 8,
        settle_patience: int = 999,
        settle_eps: float = 0.012,
    ) -> Skill:
        """Let the frozen RLDX-1 VLA drive the arm with the base clamped to small motions.

        Same skill and arguments as ``rldx_skill`` with ``base_clip=0.1``: the
        VLA can micro-align the gripper for a grasp or a contact but cannot
        drive away.  Use it once the base already stands at the right place;
        use ``rldx_skill`` when the VLA has to reposition the base as well.
        """
        kwargs = self._skill_kwargs(
            base_clip, max_chunks, force_reset, n_action_steps, settle_patience, settle_eps
        )
        return self._stateful("rldx_arm", kwargs, lambda p: self._skill(p, kwargs))

    # -- scripted arm motion --------------------------------------------------

    def move_to(
        self,
        xyz: Sequence[float],
        gripper: float | str = "hold",
        *,
        step_clip: float = 0.02,
        max_steps: int = 200,
        tol: float = 0.012,
    ) -> Move:
        """Servo the gripper to a world-frame position, holding its orientation.

        A closed-loop OSC servo: each env step moves the gripper at most
        ``step_clip`` towards the target until it is within ``tol`` or
        ``max_steps`` are spent.  The first arm servo after a base motion
        spends a few extra steps calibrating.

        Args:
            xyz: target gripper position [x, y, z] in metres.  One call may
                move at most 0.30 m (``ValueError`` beyond); split longer
                traversals into 2-3 waypoints at carrying height.
            gripper: ``"hold"`` (default: keep the finger width, carry-safe),
                +1 (close and keep squeezing) or -1 (open); held during the
                whole motion.
            step_clip: per-step travel cap (m): 0.02 in free space, 0.012 for
                a fine approach or a vertical retreat.
            max_steps: env step budget.
            tol: stop when within this distance (m).
        Returns:
            Move; check ``reached`` and compare ``final_eef_pos`` with the
            target: a stalled servo returns normally.
        """
        target = _floats(xyz, 3, "xyz")
        self._check_reach(target)
        kwargs: dict[str, Any] = {
            "xyz": list(target),
            "gripper": _gripper(gripper),
            "step_clip": float(step_clip),
            "max_steps": int(max_steps),
            "tol": float(tol),
        }
        self._check_servo(kwargs)
        return self._stateful("move_to", kwargs, lambda p: self._move(target, kwargs, p))

    def move_delta(
        self,
        dxyz: Sequence[float],
        gripper: float | str = "hold",
        *,
        step_clip: float = 0.02,
        max_steps: int = 80,
    ) -> Move:
        """Servo the gripper by a relative displacement from its current position.

        ``target = current eef_pos + dxyz`` (tolerance 0.012 m); for small
        adjustments such as a final approach or a lift.  ``|dxyz|`` may not
        exceed 0.30 m.  ``gripper`` as in ``move_to`` (default ``"hold"``).
        """
        delta = _floats(dxyz, 3, "dxyz")
        if math.hypot(*delta) > MAX_MOVE_M + 1e-6:
            raise ValueError(
                f"|dxyz| = {math.hypot(*delta):.3f} m exceeds {MAX_MOVE_M} m; "
                "split the motion into waypoints"
            )
        current = self._current_state().eef_pos
        target = tuple(c + d for c, d in zip(current, delta, strict=True))
        kwargs: dict[str, Any] = {
            "dxyz": list(delta),
            "gripper": _gripper(gripper),
            "step_clip": float(step_clip),
            "max_steps": int(max_steps),
        }
        self._check_servo(kwargs)
        return self._stateful(
            "move_delta", kwargs, lambda p: self._move(target, {**kwargs, "tol": 0.012}, p)
        )

    def rotate_pitch(
        self, target_pitch: float = 0.6, *, gripper: float = 1.0, n: int = 12
    ) -> Pitch:
        """Tilt the wrist by ``target_pitch`` radians (relative, about the controller's x axis).

        Pitches the gripper forward / back over ``n`` env steps while the
        position is held by the controller (it is not actively servoed:
        expect a few mm of drift).  Use it before threading the gripper into
        an opening whose front face points along world +/-y.  ``gripper`` is
        a numeric command held during the rotation (+1 keep closed, -1 open;
        ``"hold"`` is not accepted here).  Read ``state().eef_tilt``
        afterwards; the primitive reports no angle.
        """
        pitch = float(target_pitch)
        if not math.isfinite(pitch):
            raise ValueError("target_pitch must be finite (radians; clamped to +/-1.5)")
        if int(n) < 1:
            raise ValueError("n must be at least 1")
        kwargs = {"target_pitch": pitch, "gripper": _gripper_command(gripper), "n": int(n)}
        return self._stateful(
            "rotate_pitch",
            kwargs,
            lambda p: Pitch(
                target_pitch=pitch,
                final_eef_pos=_tuple3(p.get("eef")) or self._latest_state().eef_pos,
                steps_used=int(n),
                terminated=self._latest_state().success,
            ),
        )

    def set_gripper(self, gripper: float = 1.0, steps: int = 10) -> Gripper:
        """Hold the gripper pose and drive the fingers with ``gripper`` for ``steps`` env steps.

        ``set_gripper(+1, steps=10)`` closes (or firms a grip before a
        carry), ``set_gripper(-1)`` opens.  Numeric only: ``"hold"`` is not a
        command here.  Verify a grasp afterwards from
        ``Gripper.gripper_opening`` (clearly above 0 = something between the
        fingers) and the wrist image.
        """
        command = _gripper_command(gripper)
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        kwargs = {"gripper": command, "steps": int(steps)}
        return self._stateful("set_gripper", kwargs, lambda p: self._gripper(command, steps, p))

    def release(self, steps: int = 10) -> Gripper:
        """Open the gripper in place for ``steps`` env steps (``set_gripper(-1, steps)``).

        Release only when the object is supported at its destination, then
        retreat upwards and check that it stayed put.  ``terminated``
        reports whether letting go satisfied the task.
        """
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        kwargs = {"steps": int(steps)}
        return self._stateful("release", kwargs, lambda p: self._gripper(-1.0, steps, p))

    def scripted_grasp(
        self,
        xyz: Sequence[float],
        *,
        approach_z: float = 0.10,
        grasp_z_offset: float = 0.0,
        step_clip: float = 0.02,
    ) -> Grasp:
        """Coarse scripted grasp: open, hover ``approach_z`` above ``xyz``, descend, close, lift.

        A fallback for simple, well-localised objects; prefer ``rldx_arm``
        for anything hard.  ``grasp_z_offset`` shifts the closing height
        (negative = below the sampled surface point); ``step_clip`` caps the
        descent speed.  The hover point must be within 0.30 m of the gripper.
        Returns Grasp; ``ok`` False names the ``stage`` that stalled.
        """
        target = _floats(xyz, 3, "xyz")
        hover = (target[0], target[1], target[2] + float(approach_z))
        self._check_reach(hover)
        kwargs: dict[str, Any] = {
            "xyz": list(target),
            "approach_z": float(approach_z),
            "grasp_z_offset": float(grasp_z_offset),
            "step_clip": float(step_clip),
        }
        if kwargs["step_clip"] <= 0:
            raise ValueError("step_clip must be positive")
        return self._stateful("scripted_grasp", kwargs, self._grasp)

    # -- base motion --------------------------------------------------------

    def navigate_to(
        self,
        xy: Sequence[float],
        *,
        tol: float = 0.20,
        max_steps: int = 300,
        gripper: float | str = "hold",
    ) -> Navigation:
        """Drive the base towards a world (x, y) and stop facing it.

        The base first turns to face the target, then drives forward at full
        speed (about 2.3 mm per env step, so 300 steps cover roughly 0.7 m)
        with closed-loop steering until it is within ``tol``; the first call
        spends a few steps calibrating the heading.  There is no path
        planning: choose a target with a clear line ahead (``floor_overlay``,
        ``query_world_map`` on the navview) and expect ``stuck`` when a
        fixture is in the way.  The arm is held in place and its world
        position changes with the base.

        Args:
            xy: world target [x, y] in metres (a third value is ignored).
            tol: stopping distance (m): the standoff you want in front of a
                fixture plus the object's half depth (``tol=0.6`` stops about
                0.6 m in front of the target).
            max_steps: env step budget; increase for a long drive or call
                again.
            gripper: ``"hold"`` (default, carry-safe), +1 or -1 while driving.
        Returns:
            Navigation; check ``reached`` / ``stuck`` and re-localise
            everything afterwards.
        """
        vals = _floats(xy, 3, "xy") if len(xy) == 3 else _floats(xy, 2, "xy")
        target = (vals[0], vals[1])
        if float(tol) <= 0:
            raise ValueError("tol must be positive")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be at least 1")
        kwargs: dict[str, Any] = {
            "xy": list(target),
            "tol": float(tol),
            "max_steps": int(max_steps),
            "gripper": _gripper(gripper),
        }
        return self._stateful("navigate_to", kwargs, lambda p: self._navigation(target, kwargs, p))

    def move_base(
        self,
        forward: float = 0.0,
        lateral: float = 0.0,
        turn: float = 0.0,
        *,
        steps: int = 10,
        gripper: float | str = "hold",
    ) -> BaseMove:
        """Drive the base with raw velocities in its own frame for ``steps`` env steps.

        ``forward`` (+ = ahead), ``lateral`` (+ = strafe right) and ``turn``
        (+ = rotate counter-clockwise) are clipped to [-1, 1].  For fine
        adjustments near a fixture (a short nudge, a small turn): keep
        ``forward <= 0.4``, ``turn <= 0.3`` and ``steps <= 20`` per call,
        then re-observe.  ``gripper`` as in ``navigate_to``.
        Returns BaseMove with the measured world displacement of the base.
        """
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        for name, value in (("forward", forward), ("lateral", lateral), ("turn", turn)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number in [-1, 1]")
        kwargs: dict[str, Any] = {
            "forward": float(forward),
            "lateral": float(lateral),
            "turn": float(turn),
            "steps": int(steps),
            "gripper": _gripper(gripper),
        }
        return self._stateful(
            "move_base",
            kwargs,
            lambda p: BaseMove(
                base_moved=_tuple3(p.get("base_moved")) or (0.0, 0.0, 0.0),
                base_pos=_tuple3(p.get("base_pos")) or self._latest_state().base_pos,
                steps=int(steps),
                terminated=self._latest_state().success,
            ),
        )

    # -- internals ----------------------------------------------------------

    def _latest_state(self) -> State:
        if self._last_state is None:
            return self.state()
        return self._last_state

    def _session_artifact(self, name: str) -> bytes:
        """A session-level artifact (no step), recorded in the ledger like ``artifact``."""
        record = self._ledger.open("artifact", {"name": name, "step": None}, stateful=False)
        started = time.perf_counter()
        try:
            data = self._backend.artifact(name, step=None)  # type: ignore[arg-type]
            record.ok = True
            record.summary = {"bytes": len(data)}
            return data
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.elapsed_s = time.perf_counter() - started

    def _check_reach(self, target: tuple[float, ...]) -> None:
        current = self._current_state().eef_pos
        dist = math.dist(target[:3], current)
        if dist > MAX_MOVE_M + 1e-6:
            raise ValueError(
                f"a move of {dist:.3f} m exceeds {MAX_MOVE_M} m in one call; "
                "split it into waypoints (navigate first when the target is out of reach)"
            )

    @staticmethod
    def _check_servo(kwargs: dict[str, Any]) -> None:
        if kwargs["step_clip"] <= 0:
            raise ValueError("step_clip must be positive")
        if kwargs["max_steps"] < 1:
            raise ValueError("max_steps must be at least 1")
        if "tol" in kwargs and kwargs["tol"] <= 0:
            raise ValueError("tol must be positive")

    def _skill_kwargs(
        self,
        base_clip: float | None,
        max_chunks: int,
        force_reset: bool,
        n_action_steps: int,
        settle_patience: int,
        settle_eps: float,
    ) -> dict[str, Any]:
        if base_clip is not None and not float(base_clip) > 0:
            raise ValueError("base_clip must be positive or None")
        for name, value in (
            ("max_chunks", max_chunks),
            ("n_action_steps", n_action_steps),
            ("settle_patience", settle_patience),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not float(settle_eps) > 0:
            raise ValueError("settle_eps must be positive")
        return {
            # The runtime always uses the live task language; sending it keeps
            # the recorded command identical to the tool-calling agent's.
            "prompt": self._current_state().task,
            "base_clip": float(base_clip) if base_clip is not None else None,
            "max_chunks": int(max_chunks),
            "force_reset": bool(force_reset),
            "n_action_steps": int(n_action_steps),
            "settle_patience": int(settle_patience),
            "settle_eps": float(settle_eps),
        }

    def _skill(self, p: dict[str, Any], kwargs: dict[str, Any]) -> Skill:
        state = self._latest_state()
        clip = p.get("base_clip", kwargs.get("base_clip"))
        return Skill(
            status=str(p.get("status", "")),
            chunks_used=int(p.get("chunks", 0)),
            steps_applied=int(p.get("steps_applied", 0)),
            max_chunks=int(p.get("effective_max_chunks", kwargs["max_chunks"])),
            grasped=bool(p.get("grasped")),
            grasp_detected=bool(p.get("grasp_detected")),
            grasp_contact=bool(p.get("grasp_contact")),
            held_apart=bool(p.get("held_apart")),
            grasp_obj=(str(p["grasp_obj"]) if p.get("grasp_obj") else None),
            gripper_q0=float(p.get("gripper_qpos", state.gripper_qpos[0])),
            peak_lift_m=float(p.get("peak_lift", 0.0)),
            base_drift_m=float(p.get("base_drift", 0.0)),
            base_clip=(float(clip) if clip is not None else None),
            instruction=str(p.get("effective_prompt") or state.task),
            terminated=state.success,
        )

    def _move(self, target: tuple[float, ...], kwargs: dict[str, Any], p: dict[str, Any]) -> Move:
        state = self._latest_state()
        final = _tuple3(p.get("eef")) or state.eef_pos
        dist = float(p.get("final_dist", math.dist(final, target[:3])))
        return Move(
            target_xyz=(target[0], target[1], target[2]),
            final_eef_pos=final,
            final_dist_m=dist,
            reached=bool(p.get("ok", dist < float(kwargs["tol"]))),
            steps_used=int(p.get("steps", 0)),
            max_steps=int(kwargs["max_steps"]),
            gripper_q0=float(p.get("gripper_qpos", state.gripper_qpos[0])),
            terminated=state.success,
        )

    def _gripper(self, command: float, steps: int, p: dict[str, Any]) -> Gripper:
        state = self._latest_state()
        qpos = _tuple_n(p.get("gripper_qpos"), 2) or state.gripper_qpos
        return Gripper(
            gripper=float(command),
            steps=int(steps),
            gripper_qpos=(qpos[0], qpos[1]),
            gripper_opening=abs(qpos[0]) + abs(qpos[1]),
            terminated=state.success,
        )

    def _grasp(self, p: dict[str, Any]) -> Grasp:
        state = self._latest_state()
        ok = bool(p.get("ok"))
        qpos = _tuple_n(p.get("gripper_qpos"), 2) or state.gripper_qpos
        return Grasp(
            ok=ok,
            stage=(str(p["stage"]) if p.get("stage") else None),
            final_eef_pos=_tuple3(p.get("eef")) or state.eef_pos,
            final_dist_m=float(p.get("final_dist", 0.0 if ok else math.inf)),
            gripper_qpos=(qpos[0], qpos[1]),
            gripper_opening=abs(qpos[0]) + abs(qpos[1]),
            terminated=state.success,
        )

    def _navigation(
        self, target: tuple[float, float], kwargs: dict[str, Any], p: dict[str, Any]
    ) -> Navigation:
        state = self._latest_state()
        start = _tuple_n(p.get("start_pos"), 2) or state.base_pos[:2]
        return Navigation(
            target_xy=target,
            reached=bool(p.get("ok")),
            stuck=bool(p.get("stuck")),
            final_dist_m=float(p.get("final_dist", math.inf)),
            moved_m=float(p.get("moved", 0.0)),
            steps_used=int(p.get("steps", 0)),
            max_steps=int(kwargs["max_steps"]),
            start_xy=(start[0], start[1]),
            base_pos=_tuple3(p.get("base_pos")) or state.base_pos,
            terminated=state.success,
        )

    def _state_from_envelope(self, envelope: dict[str, Any]) -> State:
        raw = envelope.get("state") or {}
        eef_pos = _tuple3(raw.get("robot0_eef_pos")) or (0.0, 0.0, 0.0)
        eef_quat = _tuple_n(raw.get("robot0_eef_quat"), 4) or (0.0, 0.0, 0.0, 1.0)
        qpos = _tuple_n(raw.get("robot0_gripper_qpos"), 2) or (0.0, 0.0)
        base_pos = _tuple3(raw.get("robot0_base_pos")) or (0.0, 0.0, 0.0)
        base_quat = _tuple_n(raw.get("robot0_base_quat"), 4) or (0.0, 0.0, 0.0, 1.0)
        success = bool(envelope.get("success")) or bool(envelope.get("robocasa_terminated"))
        progress = envelope.get("task_progress")
        return State(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or ""),
            eef_pos=eef_pos,
            eef_quat=eef_quat,  # type: ignore[arg-type]
            gripper_qpos=(qpos[0], qpos[1]),
            gripper_opening=abs(qpos[0]) + abs(qpos[1]),
            base_pos=base_pos,
            base_quat=base_quat,  # type: ignore[arg-type]
            success=success,
            task_progress=dict(progress) if isinstance(progress, dict) else {},
            vla_desync=bool(envelope.get("vla_desync")),
            terminated=bool(envelope.get("terminated")) or success,
            truncated=bool(envelope.get("truncated")),
        )

    def _summary_dict(self) -> dict[str, Any]:
        """Compact state for the per-turn tool feedback (host-facing)."""
        state = self.state()
        return {
            "step": state.step,
            "eef_pos": [round(v, 4) for v in state.eef_pos],
            "eef_tilt": round(state.eef_tilt, 3),
            "gripper_opening": round(state.gripper_opening, 4),
            "base_xy": [round(v, 4) for v in state.base_pos[:2]],
            "base_yaw": round(state.base_yaw, 3),
            "success": state.success,
            "task_progress": state.task_progress,
            **({} if self._hidden else {"vla_desync": state.vla_desync}),
        }

    def _summary(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        out = super()._summary(tool, payload)
        if tool == "view_env_state":
            state = payload.get("state") or {}
            out["eef_pos"] = state.get("robot0_eef_pos")
            out["success"] = payload.get("success")
            out["vla_desync"] = payload.get("vla_desync")
            out["task_progress"] = payload.get("task_progress")
        elif tool == "back_project_batch":
            summary = payload.get("summary") or {}
            out["valid_count"] = summary.get("valid_count")
            out["median_xyz"] = summary.get("median_xyz")
        elif tool == "query_world_map":
            summary = payload.get("summary") or {}
            out["total_clusters"] = summary.get("total_clusters")
            out["total_pixels_matched"] = summary.get("total_pixels_matched")
            clusters = payload.get("clusters") or []
            if clusters:
                out["center_xyz"] = clusters[0].get("center_xyz")
        elif tool in self.STATEFUL_TOOLS and isinstance(self._last_state, State):
            out["success"] = self._last_state.success
        return out


ADAPTER = RobotAdapter(
    name="robocasa",
    robot_cls=RobocasaRobot,
    host="pyrualean.hosts.rpent_robocasa",
    knowledge="robocasa",
    blurb="a mobile-base PandaOmron robot in the RoboCasa365 kitchen benchmark",
    guides_subdir="",  # RPent ships no guides for this robot
    guide_names=(),
    show_examples="`robo.show('agentview')`, `robo.show('navview')` or `robo.show('wrist')`",
    services="a frozen RLDX-1 VLA",
    services_novla="",
)


__all__ = [
    "ADAPTER",
    "BackProjection",
    "BaseMove",
    "Cluster",
    "Grasp",
    "Gripper",
    "HeightQuery",
    "IMAGE_SIZE",
    "MAX_MOVE_M",
    "MAX_PIXELS",
    "Move",
    "Navigation",
    "Pitch",
    "Point",
    "RobocasaRobot",
    "STATEFUL_TOOLS",
    "Skill",
    "State",
    "VLA_CHUNK_STEPS",
    "WORLD_MAP_WINDOW",
]
