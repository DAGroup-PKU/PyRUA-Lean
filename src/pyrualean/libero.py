# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LIBERO robot API for code policies.

Every public method of :class:`LiberoRobot` maps 1:1 onto one RPent LIBERO
primitive (the same scripted OSC servos, the same frozen Pi0.5 VLA calls and
the same SAM3 / back-projection helpers that RPent's tool-calling agent uses).
The difference is the calling convention: a policy is an ordinary Python
program that calls these methods and branches on their typed return values,
instead of a language model emitting one tool call per turn.

The docstrings in this module are the model-facing documentation: the prompt
is generated from them (see :mod:`pyrualean.prompt`), so keep them accurate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ._backend import ToolError
from ._robot import RobotBase
from .robots import RobotAdapter

# Largest planar move RPent allows in one servo call; longer traversals flip
# the OSC controller's IK and corrupt the run, so the library refuses them.
MAX_PLANAR_MOVE_M = 0.30


# ---------------------------------------------------------------------------
# Result types (returned to the policy; all fields are plain Python values)
# ---------------------------------------------------------------------------


def _quat_to_matrix(q: Sequence[float]) -> list[list[float]]:
    x, y, z, w = (float(v) for v in q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


@dataclass(frozen=True)
class State:
    """Proprioceptive snapshot of the robot at one recorded step.

    step: index of the recorded step (0 = initial scene).
    task: the task instruction given by the benchmark (authoritative).
    eef_pos: end-effector (gripper) position [x, y, z] in the world frame, metres.
    eef_quat: end-effector orientation quaternion [x, y, z, w].
    gripper_qpos: the two finger joint positions.
    gripper_opening: finger separation proxy, |q0| + |q1|.  About 0.08 when
        fully open, about 0.0 when closed on nothing; roughly 0.01-0.05 means
        the fingers closed on an object.
    object_names: names of the movable objects in the scene (no poses).
    terminated: the benchmark's success predicate has fired.
    truncated: the simulator step budget is exhausted.
    """

    step: int
    task: str
    eef_pos: tuple[float, float, float]
    eef_quat: tuple[float, float, float, float]
    gripper_qpos: tuple[float, float]
    gripper_opening: float
    object_names: tuple[str, ...]
    terminated: bool
    truncated: bool

    @property
    def eef_yaw(self) -> float:
        """World-frame yaw of the gripper in radians (as used by rotate_wrist)."""
        m = _quat_to_matrix(self.eef_quat)
        return math.atan2(m[1][0], m[0][0])

    @property
    def eef_pitch(self) -> float:
        """Gripper tilt in radians (0 = pointing straight down; see rotate_pitch)."""
        m = _quat_to_matrix(self.eef_quat)
        return math.atan2(m[1][2], -m[2][2])


@dataclass(frozen=True)
class Move:
    """Outcome of ``move_to``.

    target_xyz: the commanded target.
    final_eef_pos: where the gripper actually stopped.
    final_dist_m: distance between the two, metres.
    reached: ``final_dist_m <= tol``.  False means the servo stalled (IK limit,
        collision, or step budget) - re-plan instead of assuming you arrived.
    steps_used / max_steps: simulator steps consumed / allowed.
    terminated / truncated: episode flags after the move.
    """

    target_xyz: tuple[float, float, float]
    final_eef_pos: tuple[float, float, float]
    final_dist_m: float
    reached: bool
    steps_used: int
    max_steps: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class MovePose:
    """Outcome of ``move_pose`` (position and orientation servo).

    reached: position within ``tol`` (orientation errors are not folded in).
    final_pitch: gripper pitch after the move, radians.
    """

    final_eef_pos: tuple[float, float, float]
    final_dist_m: float
    final_pitch: float
    reached: bool
    steps_used: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Rotation:
    """Outcome of ``rotate_wrist`` / ``rotate_pitch``; angles in radians.

    reached: ``abs(final_err) <= tol``.
    """

    start: float
    target: float
    final: float
    final_err: float
    reached: bool
    steps_used: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Gripper:
    """Outcome of ``set_gripper``: the command that was held and for how long."""

    gripper: float
    steps: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Release:
    """Outcome of ``release``.

    peak_gripper_opening: widest finger separation seen while opening.
    terminated: True if letting go satisfied the task predicate.
    """

    steps_used: int
    start_gripper_opening: float
    peak_gripper_opening: float
    final_gripper_opening: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Pick:
    """Outcome of ``pi0_pick``.

    success: the VLA's lift heuristic fired (descended, then rose by
        ``lift_thresh`` with the fingers closed on something).  It is a hint,
        not proof: confirm with ``state().gripper_opening`` and the wrist image.
    chunks_used / max_chunks: VLA action chunks consumed / allowed.
    peak_lift_m: how far the gripper rose after its lowest point.
    min_gripper_opening / final_gripper_opening: finger separation statistics.
    terminated: the task predicate fired during the pick.
    diagnostics: raw threshold bookkeeping from the primitive.
    """

    success: bool
    chunks_used: int
    max_chunks: int
    peak_lift_m: float
    min_gripper_opening: float
    final_gripper_opening: float
    terminated: bool
    truncated: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Contact:
    """Outcome of ``pi0_doubled``.

    success: mirrors ``terminated`` only.  For an intermediate contact skill
        (knob, drawer, door) ``False`` does not mean the contact failed;
        inspect the state or an image.
    """

    success: bool
    chunks_used: int
    max_chunks: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Segment:
    """Outcome of ``segment``.

    found: a mask above ``min_score`` was produced.
    world_xyz: median world coordinate of the mask (surface point, metres), or
        None when not found / no valid depth.
    score: SAM3 confidence in [0, 1] (brand nouns score ~0.05, colour+shape
        phrases ~0.7).
    box: mask bounding box as reported by SAM3, or None.
    overlay_artifact: name of the saved overlay image (``artifact()`` loads it).
    error: why nothing was found, or None.
    """

    found: bool
    world_xyz: tuple[float, float, float] | None
    score: float | None
    box: Any
    camera: str
    step: int
    overlay_artifact: str | None
    error: str | None


@dataclass(frozen=True)
class BackProjection:
    """Outcome of ``back_project``: the world point under one pixel."""

    world_xyz: tuple[float, float, float]
    pixel: tuple[int, int]
    camera: str
    resolution: str
    step: int
    image_size: tuple[int, int]


@dataclass(frozen=True)
class RegionCenter:
    """Outcome of ``region_center``.

    center_xyz: midpoint of the region's world x/y extent with median z.
    median_xyz: per-axis median of the same pixels.
    n_valid: pixels that survived the depth and z-band filters.
    """

    center_xyz: tuple[float, float, float]
    median_xyz: tuple[float, float, float]
    n_valid: int
    camera: str
    step: int


RESULT_TYPES: tuple[type, ...] = (
    State,
    Move,
    MovePose,
    Rotation,
    Gripper,
    Release,
    Pick,
    Contact,
    Segment,
    BackProjection,
    RegionCenter,
)

STATEFUL_TOOLS = frozenset(
    {
        "move_to",
        "move_pose",
        "rotate_wrist",
        "rotate_pitch",
        "set_gripper",
        "release",
        "pi0_pick",
        "pi0_doubled",
    }
)


def _xyz(value: Sequence[float], name: str = "xyz") -> tuple[float, float, float]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be three numbers, got {value!r}") from exc
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} must be three finite numbers, got {value!r}")
    return vals  # type: ignore[return-value]


def _tuple3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    vals = tuple(float(v) for v in value)
    if len(vals) != 3:
        return None
    return vals  # type: ignore[return-value]


def _gripper(value: float) -> float:
    try:
        g = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gripper must be -1 (open) or +1 (close)") from exc
    if not -1.0 <= g <= 1.0:
        raise ValueError("gripper must be within [-1, +1]: -1 opens, +1 closes")
    return g


# ---------------------------------------------------------------------------
# The robot
# ---------------------------------------------------------------------------


class LiberoRobot(RobotBase):
    """A LIBERO tabletop Franka arm with a two-finger gripper.

    World frame: x/y span the table, z is up, units are metres and radians.
    The arm starts at a home pose above the table, gripper open and pointing
    down.  Motion calls block until the primitive finishes and return a typed
    result describing what actually happened; nothing is queued.

    Gripper commands are floats: ``LiberoRobot.OPEN`` (-1) opens,
    ``LiberoRobot.CLOSE`` (+1) closes and keeps holding.  Every motion call
    takes a ``gripper`` argument that is *held for the whole motion*; carrying
    an object with the default ``gripper=-1`` silently drops it.

    Perception is camera based: the scene is observed through a fixed
    ``agentview`` camera (global layout) and a ``wrist`` camera (close range).
    Object poses are never given; use ``segment`` / ``back_project`` on the
    recorded images to obtain world coordinates.
    """

    OPEN: float = -1.0
    CLOSE: float = 1.0

    NAME: ClassVar[str] = "libero"
    CAMERAS: ClassVar[tuple[str, ...]] = ("agentview", "wrist")
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("agentview", "high"): "agentview_high.png",
        ("agentview", "low"): "agentview.png",
        ("wrist", "high"): "wrist_high.png",
        ("wrist", "low"): "wrist.png",
    }
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("agentview", "high"): "agentview_world_high.npz",
        ("agentview", "low"): "agentview_world.npz",
        ("wrist", "high"): "wrist_world_high.npz",
        ("wrist", "low"): "wrist_world.npz",
    }
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = ("agentview_high.png", "wrist_high.png")
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = STATEFUL_TOOLS
    #: The frozen Pi0.5 skills (hidden and refused under the no-VLA primitive set).
    VLA_PRIMITIVES: ClassVar[tuple[str, ...]] = ("pi0_pick", "pi0_doubled")

    # -- observation --------------------------------------------------------

    @property
    def task(self) -> str:
        """The task instruction, e.g. ``"put the bowl on the plate"``."""
        return self._current_state().task

    @property
    def done(self) -> bool:
        """True once the benchmark's success predicate has fired (sticky)."""
        return self._done or bool(self._backend.solved())

    def state(self, step: int = -1) -> State:
        """Read the recorded proprioceptive state.

        Args:
            step: recorded step to read; ``-1`` is the latest, ``0`` the
                initial scene.  A new step is recorded after every motion call.
        Returns:
            State with gripper position, orientation, finger opening, the
            object names and the episode flags.
        """
        payload = self._readonly("view_env_state", {"step": int(step)})
        state = self._state_from_envelope(payload)
        if step == -1:
            self._last_state = state
        return state

    def image(self, camera: str = "agentview", *, resolution: str = "high", step: int = -1) -> Any:
        """Load a recorded camera image as a ``numpy`` uint8 array (H, W, 3), RGB.

        Args:
            camera: ``"agentview"`` (fixed global view, best for identifying
                objects and layout) or ``"wrist"`` (camera on the gripper,
                best for close-range geometry).
            resolution: ``"high"`` (1024x1024) or ``"low"`` (256x256).  Pixel
                coordinates are (row, col) with row 0 at the top and col 0 at
                the left; pass the same resolution to ``back_project``.
            step: recorded step, ``-1`` = latest.
        """
        return super().image(camera, resolution=resolution, step=step)

    def world_map(
        self, camera: str = "agentview", *, resolution: str = "high", step: int = -1
    ) -> Any:
        """Load the per-pixel world coordinates of a recorded image.

        Returns a ``numpy`` float array (H, W, 3): ``world_map[row, col]`` is
        the [x, y, z] of the surface seen at that pixel (zeros / non-finite
        where depth is invalid).  This is what ``back_project`` reads; load it
        when you want to reason about many pixels at once.
        """
        return super().world_map(camera, resolution=resolution, step=step)

    def show(self, camera: str = "agentview") -> None:
        """Attach the current image of ``camera`` to the feedback of this turn.

        Images are not returned automatically; call ``show("agentview")`` or
        ``show("wrist")`` when *you* (the author) need to look at the scene
        before writing the next turn.  Costs tokens: look only when a
        decision depends on it.
        """
        super().show(camera)

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return the raw bytes of a recorded artifact (advanced).

        Names include ``agentview_high.png``, ``wrist_high.png``,
        ``agentview_depth.npz`` and the ``overlay_artifact`` of a ``segment``.
        """
        return super().artifact(name, step=step)

    def camera_meta(self, camera: str = "agentview", *, step: int = -1) -> dict[str, Any]:
        """Return the camera calibration metadata (intrinsics/extrinsics) as a dict."""
        payload = self._readonly("view_camera_meta", {"camera": camera, "step": int(step)})
        meta = payload.get("camera_meta")
        return dict(meta) if isinstance(meta, dict) else {}

    def segment(
        self,
        prompt: str | None = None,
        *,
        camera: str = "agentview",
        point: tuple[int, int] | None = None,
        min_score: float = 0.2,
        step: int = -1,
    ) -> Segment:
        """Locate an object with SAM3 on a recorded image and return its world position.

        Give exactly one of ``prompt`` (a short visual description) or
        ``point`` (a positive pixel ``(row, col)`` in the high-resolution
        image).  The top-ranked mask is projected through the world map, so
        ``world_xyz`` is the object's visible *surface* median: use its x/y for
        a target and pick z from the object's resting height.

        Args:
            prompt: colour + shape + relation, e.g. ``"the black bowl on the
                stove"``.  Brand or internal names (``"akita"``,
                ``"alphabet soup"``) score badly; describe what it looks like.
            camera: ``"agentview"`` for global identity, ``"wrist"`` for
                refinement once the gripper is 15-20 cm above the target.
            point: alternative to ``prompt``; a pixel on the object.
            min_score: reject masks below this confidence.
            step: recorded step whose image to use (``-1`` = latest).
        Returns:
            Segment; check ``found`` before using ``world_xyz``.
        """
        kwargs: dict[str, Any] = {
            "camera": camera,
            "step": int(step),
            "min_score": float(min_score),
        }
        if prompt is not None:
            kwargs["prompt"] = str(prompt)
        if point is not None:
            kwargs["point"] = [int(point[0]), int(point[1])]
        payload = self._readonly("segment", kwargs, allow_error_key="found")
        return Segment(
            found=bool(payload.get("found")),
            world_xyz=_tuple3(payload.get("world_xyz")),
            score=(float(payload["score"]) if payload.get("score") is not None else None),
            box=payload.get("box"),
            camera=str(payload.get("camera", camera)),
            step=int(payload.get("step", step)),
            overlay_artifact=payload.get("overlay_artifact"),
            error=(str(payload["error"]) if payload.get("error") else None)
            or (str(payload["world_error"]) if payload.get("world_error") else None),
        )

    def back_project(
        self,
        row: int,
        col: int,
        *,
        camera: str = "agentview",
        resolution: str = "high",
        step: int = -1,
    ) -> BackProjection:
        """Return the world [x, y, z] of the surface seen at pixel ``(row, col)``.

        The pixel must come from the same camera and resolution as requested
        (``image(camera, resolution=...)``).  Sample several pixels firmly on
        an object's top surface and take the median; pixels on thin rims,
        edges or the gap to the table hit the background metres away.

        Raises:
            ToolError: pixel out of bounds or no valid depth at that pixel.
        """
        payload = self._readonly(
            "back_project",
            {
                "row": int(row),
                "col": int(col),
                "camera": camera,
                "resolution": resolution,
                "step": int(step),
            },
        )
        xyz = _tuple3(payload.get("world_xyz"))
        if xyz is None:
            raise ToolError("back_project", "no world_xyz in result", payload)
        pixel = payload.get("pixel", [row, col])
        size = payload.get("image_size", [0, 0])
        return BackProjection(
            world_xyz=xyz,
            pixel=(int(pixel[0]), int(pixel[1])),
            camera=str(payload.get("camera", camera)),
            resolution=str(payload.get("resolution", resolution)),
            step=int(payload.get("step", step)),
            image_size=(int(size[0]), int(size[1])),
        )

    def region_center(
        self,
        row_range: tuple[int, int],
        col_range: tuple[int, int],
        *,
        camera: str = "agentview",
        resolution: str = "high",
        z_min: float | None = None,
        z_max: float | None = None,
        step: int = -1,
    ) -> RegionCenter:
        """Return the world centre of a pixel window (e.g. a container cavity).

        Uses the midpoint of the window's world x/y extent rather than a
        median, which is biased towards a rim or edge.  ``z_min`` / ``z_max``
        keep only pixels inside a world-z band (e.g. the basket floor).

        Raises:
            ToolError: empty window or too few valid pixels.
        """
        kwargs: dict[str, Any] = {
            "row_range": [int(row_range[0]), int(row_range[1])],
            "col_range": [int(col_range[0]), int(col_range[1])],
            "camera": camera,
            "resolution": resolution,
            "step": int(step),
        }
        if z_min is not None:
            kwargs["z_min"] = float(z_min)
        if z_max is not None:
            kwargs["z_max"] = float(z_max)
        payload = self._readonly("back_project", kwargs)
        center = _tuple3(payload.get("center_xyz"))
        median = _tuple3(payload.get("median_xyz"))
        if center is None or median is None:
            raise ToolError("back_project", "no region centre in result", payload)
        return RegionCenter(
            center_xyz=center,
            median_xyz=median,
            n_valid=int(payload.get("n_valid", 0)),
            camera=str(payload.get("camera", camera)),
            step=int(payload.get("step", step)),
        )

    # -- scripted motion ----------------------------------------------------

    def move_to(
        self,
        xyz: list[float],
        gripper: float = -1.0,
        *,
        tol: float = 0.012,
        step_clip: float = 0.025,
        max_steps: int = 80,
        target_yaw: float | None = None,
        yaw_step_clip: float = 0.10,
        action_scale: float = 0.05,
    ) -> Move:
        """Servo the gripper to a world-frame position, holding its orientation.

        Args:
            xyz: target [x, y, z] in metres.  One call may move at most
                0.30 m in the x/y plane; longer traversals raise ValueError -
                split them into 2-3 waypoints at carrying height.
            gripper: -1 open, +1 close; held during the whole motion.
            tol: stop when within this distance (m).
            step_clip: per-step travel cap (m): 0.025 empty/box, 0.015 cans,
                0.012 tall bottles or fine approaches.
            max_steps: simulator step budget.
            target_yaw: optional world yaw (rad) to servo simultaneously.
            yaw_step_clip: per-step yaw cap (rad) when ``target_yaw`` is set.
            action_scale: OSC action scale; leave at the default.
        Returns:
            Move; check ``reached`` - a stalled servo returns normally.
        """
        target = _xyz(xyz)
        self._check_planar_move(target)
        kwargs: dict[str, Any] = {
            "xyz": list(target),
            "gripper": _gripper(gripper),
            "tol": float(tol),
            "step_clip": float(step_clip),
            "max_steps": int(max_steps),
            "action_scale": float(action_scale),
        }
        if target_yaw is not None:
            kwargs["target_yaw"] = float(target_yaw)
            kwargs["yaw_step_clip"] = float(yaw_step_clip)

        def build(p: dict[str, Any]) -> Move:
            dist = float(p.get("final_dist_m", math.inf))
            return Move(
                target_xyz=_tuple3(p.get("target_xyz")) or target,
                final_eef_pos=_tuple3(p.get("final_eef_pos")) or target,
                final_dist_m=dist,
                reached=dist <= float(tol),
                steps_used=int(p.get("steps_used", 0)),
                max_steps=int(p.get("max_steps", max_steps)),
                terminated=bool(p.get("terminated")),
                truncated=bool(p.get("truncated")),
            )

        return self._stateful("move_to", kwargs, build)

    def move_pose(
        self,
        xyz: list[float],
        *,
        target_pitch: float | None = None,
        target_yaw: float | None = None,
        gripper: float = -1.0,
        step_clip: float = 0.02,
        pitch_step: float = 0.08,
        yaw_step: float = 0.08,
        tol: float = 0.012,
        ori_tol: float = 0.05,
        max_steps: int = 150,
        action_scale: float = 0.05,
    ) -> MovePose:
        """Servo position and wrist orientation (pitch/yaw) at the same time.

        Use it where ``move_to`` stalls short of a low or recessed target
        (cabinet fronts, microwave cavities): co-varying tilt and position
        threads the IK singularity that a fixed gripper-down servo hits.
        ``gripper`` defaults to open; pass ``+1`` when holding an object.
        """
        target = _xyz(xyz)
        self._check_planar_move(target)
        kwargs: dict[str, Any] = {
            "xyz": list(target),
            "gripper": _gripper(gripper),
            "step_clip": float(step_clip),
            "pitch_step": float(pitch_step),
            "yaw_step": float(yaw_step),
            "tol": float(tol),
            "ori_tol": float(ori_tol),
            "max_steps": int(max_steps),
            "action_scale": float(action_scale),
        }
        if target_pitch is not None:
            kwargs["target_pitch"] = float(target_pitch)
        if target_yaw is not None:
            kwargs["target_yaw"] = float(target_yaw)

        def build(p: dict[str, Any]) -> MovePose:
            dist = float(p.get("final_dist_m", math.inf))
            return MovePose(
                final_eef_pos=_tuple3(p.get("final_eef_pos")) or target,
                final_dist_m=dist,
                final_pitch=float(p.get("final_pitch", 0.0)),
                reached=dist <= float(tol),
                steps_used=int(p.get("steps_used", 0)),
                terminated=bool(p.get("terminated")),
                truncated=bool(p.get("truncated")),
            )

        return self._stateful("move_pose", kwargs, build)

    def rotate_wrist(
        self,
        *,
        target_yaw: float | None = None,
        delta_yaw: float | None = None,
        gripper: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> Rotation:
        """Rotate the gripper about the world z axis, keeping its position.

        Give exactly one of ``target_yaw`` (absolute, rad) or ``delta_yaw``
        (relative, rad).  ``state().eef_yaw`` reports the current yaw.
        ``gripper`` defaults to +1 (keep holding).
        """
        if (target_yaw is None) == (delta_yaw is None):
            raise ValueError("give exactly one of target_yaw or delta_yaw")
        kwargs: dict[str, Any] = {
            "gripper": _gripper(gripper),
            "max_steps": int(max_steps),
            "tol": float(tol),
            "step_clip": float(step_clip),
        }
        if target_yaw is not None:
            kwargs["target_yaw"] = float(target_yaw)
        else:
            kwargs["delta_yaw"] = float(delta_yaw)  # type: ignore[arg-type]
        return self._stateful("rotate_wrist", kwargs, lambda p: self._rotation(p, "yaw", tol))

    def rotate_pitch(
        self,
        *,
        target_pitch: float | None = None,
        delta_pitch: float | None = None,
        gripper: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> Rotation:
        """Tilt the gripper about the world x axis, keeping position and yaw.

        Pitch 0 points the gripper straight down; +pi/2 points it along world
        +y, -pi/2 along -y.  Use before threading into an opening whose front
        faces along y.  Give exactly one of ``target_pitch`` or ``delta_pitch``.
        """
        if (target_pitch is None) == (delta_pitch is None):
            raise ValueError("give exactly one of target_pitch or delta_pitch")
        kwargs: dict[str, Any] = {
            "gripper": _gripper(gripper),
            "max_steps": int(max_steps),
            "tol": float(tol),
            "step_clip": float(step_clip),
        }
        if target_pitch is not None:
            kwargs["target_pitch"] = float(target_pitch)
        else:
            kwargs["delta_pitch"] = float(delta_pitch)  # type: ignore[arg-type]
        return self._stateful("rotate_pitch", kwargs, lambda p: self._rotation(p, "pitch", tol))

    def set_gripper(self, gripper: float, steps: int = 5) -> Gripper:
        """Hold the current pose and drive the gripper for ``steps`` simulator steps.

        ``set_gripper(+1, steps=8)`` right after a successful pick firms the
        grip before carrying (use ``steps<=5`` for a laterally weak can).
        """
        kwargs = {"gripper": _gripper(gripper), "steps": int(steps)}
        return self._stateful(
            "set_gripper",
            kwargs,
            lambda p: Gripper(
                gripper=float(p.get("gripper", kwargs["gripper"])),
                steps=int(p.get("steps", steps)),
                terminated=bool(p.get("terminated")),
                truncated=bool(p.get("truncated")),
            ),
        )

    def release(self, max_steps: int = 20) -> Release:
        """Open the gripper in place for up to ``max_steps`` steps and let go.

        Returns as soon as the task predicate fires (``terminated``), so a
        successful placement usually ends the episode here.
        """
        kwargs = {"max_steps": int(max_steps)}
        return self._stateful(
            "release",
            kwargs,
            lambda p: Release(
                steps_used=int(p.get("steps_used", 0)),
                start_gripper_opening=float(p.get("start_gripper_opening", 0.0)),
                peak_gripper_opening=float(p.get("peak_gripper_opening", 0.0)),
                final_gripper_opening=float(p.get("final_gripper_opening", 0.0)),
                terminated=bool(p.get("terminated")),
                truncated=bool(p.get("truncated")),
            ),
        )

    # -- frozen VLA skills --------------------------------------------------

    def pi0_pick(
        self,
        prompt: str,
        *,
        max_chunks: int = 24,
        lift_thresh: float = 0.05,
        gripper_closed_thresh: float = 0.06,
        gripper_open_thresh: float = 0.0,
        descent_thresh: float = 0.10,
    ) -> Pick:
        """Let the frozen Pi0.5 VLA perform a closed-loop grasp.

        The VLA runs ``prompt`` for up to ``max_chunks`` action chunks (each a
        short burst of steps) and stops early once the gripper has descended by
        ``descent_thresh``, closed on something (opening in
        ``[gripper_open_thresh, gripper_closed_thresh)``) and risen by
        ``lift_thresh``.  Use it for the grasp: pre-position the gripper above
        the target, keep the prompt short (``"pick up the red mug"``), use a
        modest chunk budget (20 is the usual choice) and verify the grasp from
        the lift, the gripper closure and the images; then script the carry
        and release yourself.

        Returns:
            Pick; ``success`` is a heuristic - verify with ``state()`` and the
            wrist image before carrying.
        """
        kwargs: dict[str, Any] = {
            "prompt": str(prompt),
            "max_chunks": int(max_chunks),
            "lift_thresh": float(lift_thresh),
            "gripper_closed_thresh": float(gripper_closed_thresh),
            "gripper_open_thresh": float(gripper_open_thresh),
            "descent_thresh": float(descent_thresh),
        }
        return self._stateful(
            "pi0_pick",
            kwargs,
            lambda p: Pick(
                success=bool(p.get("success")),
                chunks_used=int(p.get("chunks_used", 0)),
                max_chunks=int(p.get("max_chunks", max_chunks)),
                peak_lift_m=float(p.get("peak_lift_m", 0.0)),
                min_gripper_opening=float(p.get("min_gripper_opening", 0.0)),
                final_gripper_opening=float(p.get("final_gripper_opening", 0.0)),
                terminated=bool(p.get("terminated")),
                truncated=bool(p.get("truncated")),
                diagnostics=dict(p.get("diagnostics") or {}),
            ),
        )

    def pi0_doubled(self, prompt: str, *, max_chunks: int = 20) -> Contact:
        """Let the frozen Pi0.5 VLA perform a contact skill that is not a pick.

        For knobs, buttons, drawers, doors and short pushes (``"turn on the
        stove"``, ``"open the top drawer"``).  Runs up to ``max_chunks`` chunks
        or until the task predicate fires.  ``success`` only mirrors that
        predicate; check the state or an image to judge an intermediate step.
        """
        kwargs = {"prompt": str(prompt), "max_chunks": int(max_chunks)}
        return self._stateful(
            "pi0_doubled",
            kwargs,
            lambda p: Contact(
                success=bool(p.get("success")),
                chunks_used=int(p.get("chunks_used", 0)),
                max_chunks=int(p.get("max_chunks", max_chunks)),
                terminated=bool(p.get("terminated")),
                truncated=bool(p.get("truncated")),
            ),
        )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _rotation(p: dict[str, Any], axis: str, tol: float) -> Rotation:
        err = float(p.get("final_err", math.inf))
        return Rotation(
            start=float(p.get(f"start_{axis}", 0.0)),
            target=float(p.get(f"target_{axis}", 0.0)),
            final=float(p.get(f"final_{axis}", 0.0)),
            final_err=err,
            reached=abs(err) <= float(tol),
            steps_used=int(p.get("steps_used", 0)),
            terminated=bool(p.get("terminated")),
            truncated=bool(p.get("truncated")),
        )

    def _state_from_envelope(self, envelope: dict[str, Any]) -> State:
        raw = envelope.get("state") or {}
        pos = _tuple3(raw.get("robot0_eef_pos")) or (0.0, 0.0, 0.0)
        quat_raw = raw.get("robot0_eef_quat") or (0.0, 0.0, 0.0, 1.0)
        quat = tuple(float(v) for v in quat_raw)
        if len(quat) != 4:
            quat = (0.0, 0.0, 0.0, 1.0)
        qpos_raw = raw.get("robot0_gripper_qpos") or (0.0, 0.0)
        qpos = tuple(float(v) for v in qpos_raw)[:2]
        if len(qpos) != 2:
            qpos = (0.0, 0.0)
        return State(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or ""),
            eef_pos=pos,
            eef_quat=quat,  # type: ignore[arg-type]
            gripper_qpos=qpos,  # type: ignore[arg-type]
            gripper_opening=abs(qpos[0]) + abs(qpos[1]),
            object_names=tuple(str(n) for n in raw.get("object_names") or ()),
            terminated=bool(envelope.get("terminated")),
            truncated=bool(envelope.get("truncated")),
        )

    def _summary_dict(self) -> dict[str, Any]:
        """Compact state for the per-turn tool feedback (host-facing)."""
        state = self.state()
        return {
            "step": state.step,
            "eef_pos": [round(v, 4) for v in state.eef_pos],
            "eef_yaw": round(state.eef_yaw, 3),
            "eef_pitch": round(state.eef_pitch, 3),
            "gripper_opening": round(state.gripper_opening, 4),
            "terminated": state.terminated,
            "truncated": state.truncated,
        }

    def _summary(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        out = super()._summary(tool, payload)
        if tool == "view_env_state":
            state = payload.get("state") or {}
            out["eef_pos"] = state.get("robot0_eef_pos")
        return out

    def _check_planar_move(self, target: tuple[float, float, float]) -> None:
        current = self._current_state().eef_pos
        planar = math.hypot(target[0] - current[0], target[1] - current[1])
        if planar > MAX_PLANAR_MOVE_M + 1e-6:
            raise ValueError(
                f"planar move of {planar:.3f} m exceeds {MAX_PLANAR_MOVE_M} m; "
                "split it into waypoints at carrying height"
            )


ADAPTER = RobotAdapter(
    name="libero",
    robot_cls=LiberoRobot,
    host="pyrualean.hosts.rpent_libero",
    knowledge="libero",
    blurb="a simulated Franka arm in the LIBERO benchmark",
    guides_subdir="robots/libero/guides",
    guide_names=("strict_hybrid_guide.md", "pro_hybrid_guide.md", "env_calibration.md"),
    show_examples="`robo.show('agentview')` or `robo.show('wrist')`",
    services="a frozen Pi0.5 VLA and a SAM3 segmentation service",
    services_novla="a SAM3 segmentation service",
)


__all__ = [
    "ADAPTER",
    "BackProjection",
    "Contact",
    "Gripper",
    "LiberoRobot",
    "MAX_PLANAR_MOVE_M",
    "Move",
    "MovePose",
    "Pick",
    "RESULT_TYPES",
    "RegionCenter",
    "Release",
    "Rotation",
    "STATEFUL_TOOLS",
    "Segment",
    "State",
]
