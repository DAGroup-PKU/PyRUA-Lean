# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboTwin robot API for code policies.

Every public method of :class:`RobotwinRobot` maps 1:1 onto one RPent RoboTwin
tool (the same cuRobo-planned arm motions, the same frozen LingBot-VLA calls
and the same world-map queries that RPent's tool-calling agent uses), with
RPent's argument names and defaults.  The difference is the calling
convention: a policy is an ordinary Python program that calls these methods
and branches on their typed return values.

The docstrings in this module are the model-facing documentation: the prompt
is generated from them (see :mod:`pyrualean.prompt`), so keep them accurate.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar

from ._backend import ToolError
from ._robot import RobotBase
from .robots import RobotAdapter

#: Distance from the EEF (what ``move_to`` targets) to the TCP (gripper centre),
#: along the gripper's approach axis, as reported by RPent's ``robot_state``.
EEF_TO_TCP_M = 0.12
#: Native actions executed per LingBot-VLA chunk (``use_length``, fixed by the runtime).
VLA_CHUNK_ACTIONS = 50
#: ``Move.reached`` tolerance: the planner's final EEF position is usually within 3 mm.
_REACH_TOL_M = 0.01


# ---------------------------------------------------------------------------
# Result types (returned to the policy; all fields are plain Python values)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmState:
    """One arm at a recorded step.

    eef_pos: end-effector position [x, y, z] in the world frame, metres.  The
        EEF is the wrist link that ``move_to`` targets.
    eef_quat: EEF orientation as a unit quaternion [w, x, y, z].
    tcp_pos: tool centre point (the gripper centre, where an object is held):
        0.12 m beyond the EEF along the gripper's approach axis.
    tcp_quat: TCP orientation [w, x, y, z] (same as the EEF).
    gripper: normalised gripper command, 0.0 = closed ... 1.0 = open.  It is
        the position the fingers were told to hold, not a contact sensor.
    """

    eef_pos: tuple[float, float, float]
    eef_quat: tuple[float, float, float, float]
    tcp_pos: tuple[float, float, float]
    tcp_quat: tuple[float, float, float, float]
    gripper: float

    @property
    def approach(self) -> tuple[float, float, float]:
        """Unit vector from the EEF to the TCP: the direction the gripper points (world frame)."""
        d = [t - e for t, e in zip(self.tcp_pos, self.eef_pos, strict=True)]
        n = math.sqrt(sum(v * v for v in d))
        if n < 1e-9:
            return (0.0, 1.0, 0.0)
        return (d[0] / n, d[1] / n, d[2] / n)

    @property
    def eef_yaw(self) -> float:
        """Rotation of the EEF about the world z axis, radians (what ``rotate_wrist`` changes)."""
        w, x, y, z = self.eef_quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class State:
    """Snapshot of both arms and of the episode at one recorded step.

    step: index of the recorded step (0 = initial scene).  A new step is
        recorded after every motion call and after ``render``.
    task: the task instruction given by the benchmark (authoritative; the VLA
        acts on exactly this text).
    left / right: the two arms (``state.arm("left")`` picks one by name).
    eval_success: the benchmark's success predicate has fired (sticky).  This
        is the only definition of success; it equals ``terminated``.
    take_action_cnt: native actions executed so far.  Every ``move_to``
        substep, every gripper step and every VLA action is one native action.
    step_lim: native-action budget of the episode.
    terminated: same as ``eval_success``.
    truncated: the native-action budget is exhausted (``take_action_cnt >=
        step_lim``); no further motion is possible.
    """

    step: int
    task: str
    left: ArmState
    right: ArmState
    eval_success: bool
    take_action_cnt: int
    step_lim: int
    terminated: bool
    truncated: bool

    def arm(self, name: str) -> ArmState:
        if name == "left":
            return self.left
        if name == "right":
            return self.right
        raise ValueError("arm must be 'left' or 'right'")

    @property
    def actions_left(self) -> int:
        """Native actions still available before ``truncated`` fires."""
        return max(0, self.step_lim - self.take_action_cnt)


@dataclass(frozen=True)
class Move:
    """Outcome of ``move_to`` / ``rotate_wrist``.

    planned: the motion planner found a collision-free joint path.  False
        means NOTHING moved (target unreachable, in collision, or too close to
        the other arm / the table): change the target, do not repeat it.
    final_eef_pos: where the EEF actually stopped (metres), None when nothing
        moved.
    final_dist_m: distance between the requested xyz and ``final_eef_pos``
        (``rotate_wrist``: distance from the xyz it was told to hold).
    reached: ``planned`` and ``final_dist_m <= 0.01``.  A planned motion can
        still stop short when the episode ended or the budget ran out.
    waypoints: planner waypoints that were executed.
    actions_used: native actions consumed (one per waypoint).
    stop_reason: ``completed`` | ``native_success`` | ``budget_exhausted`` |
        ``plan_failed`` | ``runtime_failure``.
    terminated / truncated: episode flags after the motion.
    """

    arm: str
    target_xyz: tuple[float, float, float]
    final_eef_pos: tuple[float, float, float] | None
    final_dist_m: float
    planned: bool
    reached: bool
    waypoints: int
    actions_used: int
    stop_reason: str
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Gripper:
    """Outcome of ``set_gripper`` / ``release``.

    target: the commanded gripper value; final: the value reported after the
        motion (a commanded position, not proof of a hold: verify a grasp from
        the images and from the object moving with the TCP).
    actions_used: native actions consumed (one per interpolation step).
    """

    arm: str
    target: float
    final: float
    actions_used: int
    stop_reason: str
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class VlaRun:
    """Outcome of ``lingbot_act``.

    chunks_requested / actions_requested: what was asked (50 actions per chunk).
    actions_used: native actions actually executed; fewer than requested when
        the task succeeded (``native_success``) or the budget ran out.
    stop_reason: ``completed`` | ``native_success`` | ``budget_exhausted`` |
        ``runtime_failure``.
    instruction: the task text the policy received (always the benchmark's
        own instruction, i.e. ``robo.task``).
    terminated / truncated: episode flags after the run.
    """

    chunks_requested: int
    actions_requested: int
    actions_used: int
    stop_reason: str
    instruction: str
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class WorldSample:
    """World coordinate of one queried pixel (``sample_world_xyz``).

    xyz: median [x, y, z] (metres) of the finite world points in the pixel's
        neighbourhood; valid_points: how many neighbourhood pixels had depth.
    """

    pixel: tuple[int, int]
    xyz: tuple[float, float, float]
    valid_points: int


@dataclass(frozen=True)
class WorldSamples:
    """Outcome of ``sample_world_xyz``: one ``WorldSample`` per requested pixel, in order."""

    samples: tuple[WorldSample, ...]
    view: str
    step: int
    image_shape: tuple[int, int]

    @property
    def xyz(self) -> tuple[tuple[float, float, float], ...]:
        """The sampled coordinates only, in request order."""
        return tuple(s.xyz for s in self.samples)


@dataclass(frozen=True)
class WorldRegion:
    """Outcome of ``query_world_map``: statistics of the world points inside a pixel box.

    xyz_min / xyz_max / xyz_median: per-axis extremes and median of the
        finite points (metres); valid_points: how many pixels had depth.
    points: up to ``max_points`` ``(pixel, xyz)`` samples spread over the box
        in row-major order, for your own geometry.
    """

    xyz_min: tuple[float, float, float]
    xyz_max: tuple[float, float, float]
    xyz_median: tuple[float, float, float]
    valid_points: int
    points: tuple[tuple[tuple[int, int], tuple[float, float, float]], ...]
    view: str
    step: int


STATEFUL_TOOLS = frozenset(
    {"render", "lingbot_act", "move_to", "rotate_wrist", "set_gripper", "release"}
)


def _floats(value: Any, n: int, name: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {n} numbers, got {value!r}") from exc
    if len(vals) != n or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} must be {n} finite numbers, got {value!r}")
    return vals


def _tuple3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    vals = tuple(float(v) for v in value)
    if len(vals) != 3:
        return None
    return vals  # type: ignore[return-value]


def _arm(name: Any) -> str:
    if name not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {name!r}")
    return str(name)


def _gripper(value: Any) -> float:
    try:
        g = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gripper must be a number in [0, 1]: 0 closed, 1 open") from exc
    if not 0.0 <= g <= 1.0:
        raise ValueError("gripper must be within [0, 1]: 0 closes, 1 opens")
    return g


def _pose7(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Split a 7-vector pose into (xyz, wxyz); ``value`` may be a list or a numpy array."""
    try:
        vals = tuple(float(v) for v in value) if value is not None else ()
    except (TypeError, ValueError):
        vals = ()
    if len(vals) != 7:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    return vals[:3], vals[3:]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# The robot
# ---------------------------------------------------------------------------


class RobotwinRobot(RobotBase):
    """A RoboTwin dual-arm robot (aloha-agilex: two 6-DoF arms with parallel grippers) at a table.

    World frame: x runs across the table (the robot's LEFT arm is at negative
    x and appears on the LEFT of the head image), y points away from the robot
    towards the far edge of the table, z is up; metres and radians.
    Orientations are unit quaternions ordered [w, x, y, z].  The arms are
    ``"left"`` and ``"right"`` (``robo.LEFT`` / ``robo.RIGHT``); every motion
    call names one arm and leaves the other where it is.

    Two frames per arm: ``move_to`` targets the EEF (wrist link); the TCP
    (gripper centre, where an object is held) lies ``EEF_TO_TCP_M`` = 0.12 m
    beyond the EEF along the gripper's approach axis (``ArmState.approach``),
    so to bring the gripper centre onto a point p command
    ``p - 0.12 * approach``.  Never send an object's surface point as an EEF
    target.  Both frames are reported by ``state()``.

    Gripper commands are floats in [0, 1]: ``robo.OPEN`` (1.0) is fully open,
    ``robo.CLOSE`` (0.0) fully closed; ``state().left.gripper`` reports the
    commanded value, not a contact measurement.

    Every native action (one ``move_to`` substep, one gripper step, one VLA
    action) advances ``take_action_cnt`` towards ``step_lim``.  Motion calls
    block until the primitive finishes and return a typed result; nothing is
    queued.

    Perception is camera based: a fixed ``head`` camera (whole table, 320x240)
    and one wrist camera per arm (``left_wrist``, ``right_wrist``); every
    recorded step also carries per-pixel world coordinates for each view
    (``sample_world_xyz``, ``query_world_map``, ``world_map``).  Object poses
    are never given.

    The frozen LingBot-VLA (``lingbot_act``) is a learned dual-arm policy for
    this benchmark.  It always receives the benchmark's own task instruction
    (``robo.task``), never a prompt of yours.
    """

    ARMS: tuple[str, str] = ("left", "right")
    LEFT: str = "left"
    RIGHT: str = "right"
    OPEN: float = 1.0
    CLOSE: float = 0.0
    EEF_TO_TCP_M: float = EEF_TO_TCP_M

    #: Rendered as the result-type section of the API reference (not a constant).
    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        ArmState,
        Move,
        Gripper,
        VlaRun,
        WorldSamples,
        WorldSample,
        WorldRegion,
    )
    NAME: ClassVar[str] = "robotwin"
    CAMERAS: ClassVar[tuple[str, ...]] = ("head", "left_wrist", "right_wrist")
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("head", "high"): "head_rgb.png",
        ("left_wrist", "high"): "left_wrist_rgb.png",
        ("right_wrist", "high"): "right_wrist_rgb.png",
    }
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("head", "high"): "head_world_xyz.npy",
        ("left_wrist", "high"): "left_wrist_world_xyz.npy",
        ("right_wrist", "high"): "right_wrist_world_xyz.npy",
    }
    # RPent embeds all three views after every state-changing tool; keep parity.
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = (
        "head_rgb.png",
        "left_wrist_rgb.png",
        "right_wrist_rgb.png",
    )
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = STATEFUL_TOOLS
    #: The frozen LingBot-VLA (hidden and refused under the no-VLA primitive set).
    VLA_PRIMITIVES: ClassVar[tuple[str, ...]] = ("lingbot_act",)
    #: VLA sentences of the remaining docstrings, removed from the no-VLA reference.
    VLA_REFERENCE_EDITS: ClassVar[tuple[tuple[str, str], ...]] = (
        (
            "    Every native action (one ``move_to`` substep, one gripper step, one VLA\n"
            "    action) advances ``take_action_cnt`` towards ``step_lim``.",
            "    Every native action (one ``move_to`` substep, one gripper step)\n"
            "    advances ``take_action_cnt`` towards ``step_lim``.",
        ),
        (
            "\n\n    The frozen LingBot-VLA (``lingbot_act``) is a learned dual-arm policy for\n"
            "    this benchmark.  It always receives the benchmark's own task instruction\n"
            "    (``robo.task``), never a prompt of yours.",
            "",
        ),
        (
            "    It is authoritative: it names the arm when one is required and the VLA\n"
            "    acts on exactly this text.",
            "    It is authoritative: it names the arm when one is required.",
        ),
        (
            "    task: the task instruction given by the benchmark (authoritative; the VLA\n"
            "        acts on exactly this text).",
            "    task: the task instruction given by the benchmark (authoritative).",
        ),
        (
            "    take_action_cnt: native actions executed so far.  Every ``move_to``\n"
            "        substep, every gripper step and every VLA action is one native action.",
            "    take_action_cnt: native actions executed so far.  Every ``move_to``\n"
            "        substep and every gripper step is one native action.",
        ),
    )
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        "success",
        "stop_reason",
        "executed_steps",
        "plan_status",
        "final_dist_m",
        "final_eef_xyz",
        "gripper_val",
        "valid_points",
        "xyz_median",
        "step_idx",
        "terminated",
        "truncated",
        "step",
    )

    # -- observation --------------------------------------------------------

    @property
    def task(self) -> str:
        """The task instruction, e.g. ``"Grab the hammer with the left arm and hit the block"``.

        It is authoritative: it names the arm when one is required and the VLA
        acts on exactly this text.
        """
        return super().task

    @property
    def done(self) -> bool:
        """True once the benchmark's success predicate (``eval_success``) has fired (sticky)."""
        return super().done

    def state(self, step: int = -1) -> State:
        """Read the recorded state of both arms and the episode status.

        Args:
            step: recorded step to read; ``-1`` is the latest, ``0`` the
                initial scene.  A new step is recorded after every motion call
                and after ``render``.
        Returns:
            State with ``left`` / ``right`` arm poses and gripper values,
            ``take_action_cnt`` / ``step_lim`` and the episode flags.
        """
        return super().state(step)

    def image(self, camera: str = "head", *, resolution: str = "high", step: int = -1) -> Any:
        """Load a recorded camera image as a ``numpy`` uint8 array (240, 320, 3), RGB.

        Args:
            camera: ``"head"`` (fixed view of the whole table: identify objects,
                destinations and the global layout), ``"left_wrist"`` or
                ``"right_wrist"`` (on the gripper: close-range geometry of the
                object that arm is working on).
            resolution: there is one resolution per camera, ``"high"`` (320x240).
                Pixel coordinates are (row, col) with row 0 at the top and col 0
                at the left; pass the same camera and step to
                ``sample_world_xyz`` / ``query_world_map``.
            step: recorded step, ``-1`` = latest (call ``render()`` first to
                get a fresh picture after time has passed without motion).
        """
        return super().image(camera, resolution=resolution, step=step)

    def world_map(self, camera: str = "head", *, resolution: str = "high", step: int = -1) -> Any:
        """Load the per-pixel world coordinates of a recorded image.

        Returns a ``numpy`` float32 array (240, 320, 3): ``world_map[row, col]``
        is the [x, y, z] of the surface seen at that pixel, NaN where the
        camera saw no depth.  This is what ``sample_world_xyz`` and
        ``query_world_map`` read; load it to reason about many pixels at once
        (e.g. ``np.nanmedian(world_map[..., 2])`` is the table height).
        """
        key = (camera, resolution)
        if key not in self.WORLD_MAP_ARTIFACTS:
            raise ValueError(
                f"camera must be one of {self.CAMERAS} and resolution 'high'; got {key!r}"
            )
        return super().world_map(camera, resolution=resolution, step=step)

    def depth(self, camera: str = "head", *, step: int = -1) -> Any:
        """Load the metric depth of a recorded image: float32 (240, 320) metres, NaN if invalid."""
        if camera not in self.CAMERAS:
            raise ValueError(f"camera must be one of {self.CAMERAS}")
        import numpy as np

        return np.load(io.BytesIO(self.artifact(f"{camera}_depth.npy", step=step)))

    def camera_meta(self, camera: str = "head", *, step: int = -1) -> dict[str, Any]:
        """Return the camera calibration of a recorded step as a dict.

        Keys: ``intrinsic_K`` (3x3), ``extrinsic_cv`` (3x4), ``cam2world_gl``
        (4x4, OpenGL convention: the camera looks down its -z axis), ``width``,
        ``height``.  The wrist cameras move with their arm.
        """
        if camera not in self.CAMERAS:
            raise ValueError(f"camera must be one of {self.CAMERAS}")
        return json.loads(self.artifact(f"{camera}_camera_meta.json", step=step))

    def show(self, camera: str = "head") -> None:
        """Attach the current image of ``camera`` to the feedback of this turn.

        Images are not returned automatically; call ``show("head")``,
        ``show("left_wrist")`` or ``show("right_wrist")`` when *you* (the
        author) need to look at the scene before writing the next turn.  Costs
        tokens: look only when a decision depends on it.
        """
        super().show(camera)

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return the raw bytes of a recorded artifact (advanced).

        Names are ``<camera>_rgb.png``, ``<camera>_depth.npy``,
        ``<camera>_world_xyz.npy`` and ``<camera>_camera_meta.json`` for the
        three cameras.
        """
        return super().artifact(name, step=step)

    def render(self) -> State:
        """Record a fresh observation (new step with images and world maps) without moving.

        Use it before ``image`` / ``sample_world_xyz`` when the scene may have
        changed since the last recorded step (an object settling after a
        release, for instance).  Costs no native action.
        """
        return self._stateful("render", {}, lambda payload: self._latest_state())

    def sample_world_xyz(
        self,
        view: str,
        pixels: Sequence[Sequence[int]],
        *,
        step: int = -1,
        neighborhood: int = 1,
    ) -> WorldSamples:
        """Return the world [x, y, z] under image pixels of a recorded view.

        Args:
            view: ``"head"``, ``"left_wrist"`` or ``"right_wrist"``; it is also
                the pixel coordinate space, so use the view whose image the
                pixels were read from (same ``step``).
            pixels: 1 to 256 ``[row, col]`` pairs (a single pair is accepted too).
            step: recorded step, ``-1`` = latest.
            neighborhood: half-size of the square window (0..32) whose finite
                points are median-filtered; 1 = a 3x3 window.
        Returns:
            WorldSamples; each ``xyz`` is a visible SURFACE point (the top of an
            object, not its centre): sample several interior pixels and take
            the median; pixels on edges, shadows or the gap to the table hit
            the background.
        Raises:
            ToolError: pixel out of bounds (images are 240 rows x 320 cols),
            no finite depth in the window, or no world map for that step.
        """
        if view not in self.CAMERAS:
            raise ValueError(f"view must be one of {self.CAMERAS}")
        if len(pixels) == 2 and all(isinstance(v, Real) for v in pixels):
            pixels = [pixels]  # a single [row, col] pair
        if not 1 <= len(pixels) <= 256:
            raise ValueError("pixels must contain between 1 and 256 [row, col] pairs")
        rows_cols = []
        for pixel in pixels:
            if len(pixel) != 2:
                raise ValueError(f"every pixel must be a [row, col] pair, got {pixel!r}")
            rows_cols.append([int(pixel[0]), int(pixel[1])])
        if not 0 <= int(neighborhood) <= 32:
            raise ValueError("neighborhood must be between 0 and 32")
        kwargs: dict[str, Any] = {
            "view": view,
            "pixels": rows_cols,
            "step": int(step),
            "neighborhood": int(neighborhood),
        }
        payload = self._readonly("sample_world_xyz", kwargs, allow_error_key="success")
        self._raise_world_error("sample_world_xyz", payload)
        samples = tuple(
            WorldSample(
                pixel=(int(s["pixel"][0]), int(s["pixel"][1])),
                xyz=_tuple3(s.get("xyz")) or (math.nan, math.nan, math.nan),
                valid_points=int(s.get("valid_points", 0)),
            )
            for s in payload.get("samples") or ()
        )
        shape = payload.get("image_shape") or [0, 0]
        return WorldSamples(
            samples=samples,
            view=str(payload.get("view", view)),
            step=int(payload.get("step_idx", step)),
            image_shape=(int(shape[0]), int(shape[1])),
        )

    def query_world_map(
        self,
        view: str,
        bbox: Sequence[int],
        *,
        step: int = -1,
        max_points: int = 256,
    ) -> WorldRegion:
        """Return statistics of the world points inside a pixel box of a recorded view.

        Args:
            view: ``"head"``, ``"left_wrist"`` or ``"right_wrist"`` (the box's
                pixel space; same step as the image you read it from).
            bbox: half-open ``[row_start, col_start, row_end, col_end]``.
            step: recorded step, ``-1`` = latest.
            max_points: how many ``(pixel, xyz)`` samples to return (1..4096).
        Returns:
            WorldRegion; ``xyz_median`` is the robust surface height / position
            of what fills the box, ``xyz_min`` / ``xyz_max`` its extent.
        Raises:
            ToolError: box outside the image or no finite depth inside it.
        """
        if view not in self.CAMERAS:
            raise ValueError(f"view must be one of {self.CAMERAS}")
        box = [int(v) for v in bbox]
        if len(box) != 4:
            raise ValueError("bbox must be [row_start, col_start, row_end, col_end]")
        if not 1 <= int(max_points) <= 4096:
            raise ValueError("max_points must be between 1 and 4096")
        kwargs: dict[str, Any] = {
            "view": view,
            "bbox": box,
            "step": int(step),
            "max_points": int(max_points),
        }
        payload = self._readonly("query_world_map", kwargs, allow_error_key="success")
        self._raise_world_error("query_world_map", payload)
        points = tuple(
            (
                (int(p["pixel"][0]), int(p["pixel"][1])),
                _tuple3(p.get("xyz")) or (math.nan, math.nan, math.nan),
            )
            for p in payload.get("points") or ()
        )
        nan3 = (math.nan, math.nan, math.nan)
        return WorldRegion(
            xyz_min=_tuple3(payload.get("xyz_min")) or nan3,
            xyz_max=_tuple3(payload.get("xyz_max")) or nan3,
            xyz_median=_tuple3(payload.get("xyz_median")) or nan3,
            valid_points=int(payload.get("valid_points", 0)),
            points=points,
            view=str(payload.get("view", view)),
            step=int(payload.get("step_idx", step)),
        )

    # -- the VLA ------------------------------------------------------------

    def lingbot_act(
        self, chunks: int = 4, *, use_length: int = 50, prompt: str | None = None
    ) -> VlaRun:
        """Let the frozen LingBot-VLA drive both arms for ``chunks`` x 50 native actions.

        The policy runs on the benchmark's own task instruction (``robo.task``)
        and the three current camera images; it stops early when the task
        succeeds or the budget runs out.  It is the tool for grasps and
        re-grasps, bimanual coordination, insertion, hanging, tool use and
        other contact-rich motion; script free-space transport, staging and
        release yourself with the primitives after verifying the state.

        Args:
            chunks: action chunks to execute (>= 1).  One chunk near contact,
                near success or for a small correction; two for ordinary
                stable progress; three only for a continuity-sensitive phase
                that is already moving correctly.
            use_length: actions per chunk; the runtime fixes it at 50.
            prompt: recorded with the call but NOT sent to the policy (the
                runtime always uses the task instruction); leave it None.
        Returns:
            VlaRun; check ``terminated`` / ``robo.done`` and the images
            afterwards - a completed run is not a solved task.
        """
        if int(chunks) < 1:
            raise ValueError("chunks must be at least 1")
        if int(use_length) != VLA_CHUNK_ACTIONS:
            raise ValueError(f"use_length must be {VLA_CHUNK_ACTIONS} for this VLA")
        kwargs: dict[str, Any] = {"chunks": int(chunks), "use_length": int(use_length)}
        if prompt is not None:
            kwargs["prompt"] = str(prompt)
        requested = int(chunks) * VLA_CHUNK_ACTIONS

        def build(p: dict[str, Any]) -> VlaRun:
            return VlaRun(
                chunks_requested=int(chunks),
                actions_requested=int(p.get("requested_steps", requested)),
                actions_used=int(p.get("executed_steps", 0)),
                stop_reason=str(p.get("stop_reason", "")),
                instruction=str(p.get("prompt") or self._latest_state().task),
                terminated=self._latest_state().terminated,
                truncated=self._latest_state().truncated,
            )

        return self._stateful("lingbot_act", kwargs, build)

    # -- scripted motion ----------------------------------------------------

    def move_to(
        self,
        arm: str,
        xyz: Sequence[float],
        *,
        quat: Sequence[float] | None = None,
        gripper: float | None = None,
        substeps: int = 25,
    ) -> Move:
        """Plan and execute a collision-aware arm motion to a world-frame EEF pose.

        The motion planner computes a joint path from the current pose to the
        target; the path is executed as ``substeps`` native actions.  If no
        path exists (unreachable, in collision, through the other arm or the
        table) nothing moves and ``Move.planned`` is False.

        Args:
            arm: ``"left"`` or ``"right"``.
            xyz: target EEF position [x, y, z] in metres (NOT the gripper centre:
                subtract 0.12 m along ``ArmState.approach``).
            quat: target orientation [w, x, y, z]; None keeps the current one.
            gripper: gripper value in [0, 1] to hold during the motion; None
                keeps the current command (so a held object stays held).
            substeps: waypoints executed (1 = jump to the final pose in one
                action, 0 = every planner waypoint).  Near the table, a rim or
                the other arm use small motions (z changes of 5-10 mm) with at
                most 8 substeps and re-observe between them.
        Returns:
            Move; always check ``planned`` and ``reached`` and compare
            ``final_eef_pos`` with the target: a returned call is not proof of
            arrival.
        """
        arm = _arm(arm)
        target = _floats(xyz, 3, "xyz")
        kwargs: dict[str, Any] = {"arm": arm, "xyz": list(target), "substeps": int(substeps)}
        if int(substeps) < 0:
            raise ValueError("substeps must be non-negative")
        if quat is not None:
            kwargs["quat"] = list(_floats(quat, 4, "quat"))
        if gripper is not None:
            kwargs["gripper"] = _gripper(gripper)
        return self._stateful("move_to", kwargs, lambda p: self._move(arm, target, p))

    def rotate_wrist(
        self,
        arm: str,
        delta_yaw_deg: float,
        *,
        gripper: float | None = None,
        substeps: int = 25,
    ) -> Move:
        """Rotate one EEF about the world z axis by a relative angle, keeping its position.

        Because the TCP sits 0.12 m from the EEF, the gripper (and a held
        object) sweeps an arc: rotate only with clearance around the gripper,
        at a safe height, in small increments, after a verified hold.

        Args:
            arm: ``"left"`` or ``"right"``.
            delta_yaw_deg: relative yaw in degrees (``state().arm(arm).eef_yaw``
                reports the current yaw in radians).
            gripper: value to hold during the motion; None keeps the current.
            substeps: waypoints executed (see ``move_to``).
        Returns:
            Move (``target_xyz`` is the position it was told to keep).
        """
        arm = _arm(arm)
        yaw = float(delta_yaw_deg)
        if not math.isfinite(yaw):
            raise ValueError("delta_yaw_deg must be finite")
        if int(substeps) < 0:
            raise ValueError("substeps must be non-negative")
        kwargs: dict[str, Any] = {"arm": arm, "delta_yaw_deg": yaw, "substeps": int(substeps)}
        if gripper is not None:
            kwargs["gripper"] = _gripper(gripper)
        target = self._current_state().arm(arm).eef_pos
        return self._stateful("rotate_wrist", kwargs, lambda p: self._move(arm, target, p))

    def set_gripper(self, arm: str, val: float, steps: int = 10) -> Gripper:
        """Drive one gripper linearly to ``val`` over ``steps`` native actions.

        ``set_gripper("left", robo.CLOSE)`` closes the left gripper; verify a
        grasp afterwards from the wrist image and from the object moving with
        the TCP - the reported value is a command, not a contact measurement.
        """
        arm = _arm(arm)
        target = _gripper(val)
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        kwargs = {"arm": arm, "val": target, "steps": int(steps)}
        return self._stateful("set_gripper", kwargs, lambda p: self._gripper(arm, target, p))

    def release(self, arm: str, val: float = 1.0, steps: int = 10) -> Gripper:
        """Open one gripper to ``val`` (default fully open) over ``steps`` native actions.

        Release only when the object is supported at its destination; then
        retreat upwards and check that it stayed put.  Returns
        ``terminated=True`` when letting go satisfied the task.
        """
        arm = _arm(arm)
        target = _gripper(val)
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        kwargs = {"arm": arm, "val": target, "steps": int(steps)}
        return self._stateful("release", kwargs, lambda p: self._gripper(arm, target, p))

    # -- internals ----------------------------------------------------------

    def _latest_state(self) -> State:
        if self._last_state is None:
            return self.state()
        return self._last_state

    def _move(self, arm: str, target: tuple[float, ...], p: dict[str, Any]) -> Move:
        planned = str(p.get("plan_status", "")) == "Success" and p.get("success") is not False
        final = _tuple3(p.get("final_eef_xyz"))
        dist = float(p.get("final_dist_m", math.inf)) if final is not None else math.inf
        state = self._latest_state()
        return Move(
            arm=arm,
            target_xyz=(target[0], target[1], target[2]),
            final_eef_pos=final,
            final_dist_m=dist,
            planned=planned,
            reached=planned and dist <= _REACH_TOL_M,
            waypoints=int(p.get("waypoints", 0)),
            actions_used=int(p.get("executed_steps", 0)),
            stop_reason=str(p.get("stop_reason", "")),
            terminated=state.terminated,
            truncated=state.truncated,
        )

    def _gripper(self, arm: str, target: float, p: dict[str, Any]) -> Gripper:
        state = self._latest_state()
        return Gripper(
            arm=arm,
            target=target,
            final=float(p.get("gripper_val", state.arm(arm).gripper)),
            actions_used=int(p.get("executed_steps", 0)),
            stop_reason=str(p.get("stop_reason", "")),
            terminated=state.terminated,
            truncated=state.truncated,
        )

    @staticmethod
    def _raise_world_error(tool: str, payload: dict[str, Any]) -> None:
        if payload.get("success") is False or "error" in payload:
            error = payload.get("error")
            if isinstance(error, dict):
                message = f"{error.get('code', 'error')}: {error.get('message', '')}".strip()
            else:
                message = str(error)
            raise ToolError(tool, message, payload)

    def _state_from_envelope(self, envelope: dict[str, Any]) -> State:
        raw = envelope.get("state") or {}
        robot = raw.get("robot_state") or {}
        status = raw.get("episode_status") or {}
        arms = {}
        for name in ("left", "right"):
            eef_pos, eef_quat = _pose7(robot.get(f"{name}_eef_pose"))
            tcp_pos, tcp_quat = _pose7(robot.get(f"{name}_tcp_pose"))
            arms[name] = ArmState(
                eef_pos=eef_pos,
                eef_quat=eef_quat,
                tcp_pos=tcp_pos,
                tcp_quat=tcp_quat,
                gripper=float(robot.get(f"{name}_gripper", 0.0)),
            )
        eval_success = bool(status.get("eval_success"))
        count = int(status.get("take_action_cnt", 0) or 0)
        limit = status.get("step_lim")
        step_lim = int(limit) if limit is not None else 0
        truncated = bool(envelope.get("truncated")) or (limit is not None and count >= step_lim)
        return State(
            step=int(envelope.get("step", raw.get("step_idx", -1))),
            task=str(envelope.get("task_language") or raw.get("task_language") or ""),
            left=arms["left"],
            right=arms["right"],
            eval_success=eval_success,
            take_action_cnt=count,
            step_lim=step_lim,
            terminated=bool(envelope.get("terminated")) or eval_success,
            truncated=truncated,
        )

    def _summary_dict(self) -> dict[str, Any]:
        """Compact state for the per-turn tool feedback (host-facing)."""
        state = self.state()
        return {
            "step": state.step,
            "left_eef": [round(v, 4) for v in state.left.eef_pos],
            "right_eef": [round(v, 4) for v in state.right.eef_pos],
            "left_gripper": round(state.left.gripper, 3),
            "right_gripper": round(state.right.gripper, 3),
            "actions": f"{state.take_action_cnt}/{state.step_lim}",
            "eval_success": state.eval_success,
            "terminated": state.terminated,
            "truncated": state.truncated,
        }

    def _summary(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        out = super()._summary(tool, payload)
        if tool == "view_env_state":
            status = (payload.get("state") or {}).get("episode_status") or {}
            out["take_action_cnt"] = status.get("take_action_cnt")
        return out


ADAPTER = RobotAdapter(
    name="robotwin",
    robot_cls=RobotwinRobot,
    host="pyrualean.hosts.rpent_robotwin",
    knowledge="robotwin",
    blurb="a dual-arm robot in the RoboTwin benchmark",
    guides_subdir="robots/robotwin/guides",
    guide_names=("GUIDE_RPENT.md",),
    show_examples="`robo.show('head')`, `robo.show('left_wrist')` or `robo.show('right_wrist')`",
    services="a frozen LingBot-VLA",
    services_novla="",
)


__all__ = [
    "ADAPTER",
    "ArmState",
    "EEF_TO_TCP_M",
    "Gripper",
    "Move",
    "RobotwinRobot",
    "STATEFUL_TOOLS",
    "State",
    "VLA_CHUNK_ACTIONS",
    "VlaRun",
    "WorldRegion",
    "WorldSample",
    "WorldSamples",
]
