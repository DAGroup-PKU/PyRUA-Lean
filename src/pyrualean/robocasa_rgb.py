# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboCasa365 with RoboDojo's primitives: the robot API of ``robodojo_xpl_rgb``, carried over.

What an agent learned on RoboDojo (``robodojo_xpl_rgb`` under the ``no-vla`` primitive set) is
written against that robot's API: ``move_to(xyz, arm, gripper, pitch, yaw, roll, path)``,
``set_gripper``, ``home``, ``locate`` and its ``ObjectSight``, ``back_project``, colour images
only, and no success flag while the episode runs.  This robot is RoboCasa365's PandaOmron with
that API: the same names, arguments, result fields and conventions wherever the two robots
agree, and three additions where RoboCasa needs what RoboDojo never did (``move_base``,
``set_torso`` and the surface normal of ``back_project``).

The benchmark is run as RoboCasa365 runs it (RPent's Target50 scenes, the robot's default
controller, the official horizon of each task as the native-action budget); the success
predicate is checked after every native action and ends the episode, and no result reports it,
as on RoboDojo.  The host is :mod:`pyrualean.hosts.robocasa_rgb`; run with ``--primitives
no-vla`` (there is no VLA).

The docstrings in this module are the model-facing documentation: the prompt is generated
from them (see :mod:`pyrualean.prompt`), so keep them accurate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ._robot import RobotBase
from .robots import RobotAdapter

# ---------------------------------------------------------------------------
# Result types (returned to the policy; all fields are plain Python values)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmState:
    """The arm at a recorded step.

    tcp_pos: the grasp point [x, y, z], metres: the middle of the finger pads.
        ``move_to`` drives this point.
    pitch / yaw / roll: the gripper's orientation, degrees (see the class
        docstring): ``pitch`` 0 points level and 90 straight down, ``yaw`` is
        the heading of the pointing direction (0 straight ahead of the robot,
        +90 towards its left), ``roll`` the turn of the fingers about the
        pointing direction.  Pointing straight down, ``yaw`` alone says how the
        fingers lie and ``roll`` reads 0.
    eef_pos: position of the hand (the wrist flange behind the fingers), metres.
    eef_quat: the gripper's orientation as a unit quaternion [w, x, y, z].
    gripper: measured gripper opening, normalised: about 0.97 fully open, 0.0
        closed on nothing; closed on an object, 0.08 x this reading is the
        object's width in metres.
        (Gripper COMMANDS use the opposite sign convention: see ``set_gripper``.)
    joints: the seven arm joint angles, radians (the last turns the fingers
        about the pointing direction; it stops at about +-166 degrees).
    """

    tcp_pos: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    eef_pos: tuple[float, float, float]
    eef_quat: tuple[float, float, float, float]
    gripper: float
    joints: tuple[float, ...]


@dataclass(frozen=True)
class State:
    """Snapshot of the robot at one recorded step.

    step: index of the recorded step (0 = initial scene).  A new step is
        recorded after every motion call.
    task: the task instruction given by the benchmark (authoritative).
    right: the arm (``state.arm("right")`` too).
    base: where the base stands, [x, y, heading in degrees], in the robot
        frame the episode STARTED in (its odometry: [0, 0, 0] at the start).
    torso: the torso lift, metres (0 .. 0.34; 0 at the start).
    truncated: the episode is over: its native-action budget is exhausted,
        or (with budget left) the benchmark ended it.  No further motion is
        possible.
    """

    step: int
    task: str
    right: ArmState
    base: tuple[float, float, float]
    torso: float
    truncated: bool

    def arm(self, name: str = "right") -> ArmState:
        if name in ("right", "both"):
            return self.right
        raise ValueError("this robot has one arm, 'right'")


@dataclass(frozen=True)
class Move:
    """Outcome of ``move_to``.

    target_xyz: the requested grasp-point position.
    start_xyz / final_xyz: where the grasp point was before and after.
    dist_to_target_m: distance from ``final_xyz`` to ``target_xyz``.
    orientation_error_deg: how far the final orientation is from the one asked
        (the angles passed, the others as they were).
    pitch / yaw / roll: the orientation the gripper ended with, degrees.
    reached: within 0.01 m and 5 degrees of the target.
    path: ``"line"`` (the grasp point moved along a straight line),
        ``"joint"`` (it went round, by way of the gripper's starting pose) or
        None (nothing moved).
    ik_error: why nothing moved (the pose is out of reach, or the way there
        would turn a joint into its stop), or None.  An out-of-reach target
        names the nearby poses that ARE reachable (a less steep pitch here,
        this orientation higher or lower, or nearer the robot).
    note: set when the exact pose was out of reach but one within 2 cm and
        8 degrees was not (the arm went there instead), or when the empty
        gripper arrived turned half a turn, or went round.
    blocked: the arm stopped short of its waypoints: something is in the way,
        or a held object is pushing against something.
    steps_used: native actions consumed.
    truncated: the episode is over (budget spent, or ended by the benchmark).
    """

    arm: str
    target_xyz: tuple[float, float, float]
    start_xyz: tuple[float, float, float]
    final_xyz: tuple[float, float, float]
    dist_to_target_m: float
    orientation_error_deg: float
    pitch: float
    yaw: float
    roll: float
    reached: bool
    path: str | None
    ik_error: str | None
    note: str | None
    blocked: bool
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Home:
    """Outcome of ``home``.

    arm: ``"right"``.
    reached: the gripper is back at the pose it had on the torso when the
        episode started, within 0.01 m and 5 degrees.
    max_joint_error_rad: the largest difference that remains between the
        joints and their starting angles (the arm may stand differently with
        the gripper at the same pose).
    steps_used: native actions consumed.
    truncated: the episode is over (budget spent, or ended by the benchmark).
    """

    arm: str
    reached: bool
    max_joint_error_rad: float
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Gripper:
    """Outcome of ``set_gripper``.

    commanded: ``"closed"`` or ``"open"``.
    opening: the opening once the fingers stopped (about 0.97 open, 0.0 closed
        on nothing).
    holding: after a close, the fingers stopped more than 5 mm apart:
        something is between them.  It does not say what: check that the
        intended object moves with the arm.
    width_m: after a close, the width of what is held, across the fingers
        (0.0 when nothing is), else None.
    steps_used: native actions consumed.
    step / step_limit: native actions executed so far and the episode budget.
    truncated: the episode is over (budget spent, or ended by the benchmark).
    """

    arm: str
    commanded: str
    opening: float
    holding: bool
    width_m: float | None
    steps_used: int
    step: int | None
    step_limit: int | None
    truncated: bool


@dataclass(frozen=True)
class BaseMove:
    """Outcome of ``move_base``.

    moved: [dx, dy] the base actually moved, in the robot frame it had when
        the call began, metres.
    turned_deg: how far it turned (counter-clockwise +).
    dist_to_target_m / yaw_error_deg: what is left of the asked motion.
    reached: within 2 cm and 2 degrees of it.
    blocked: the base stopped making progress: something is in the way
        (there is no path planning).
    steps_used: native actions consumed.
    truncated: the episode is over (budget spent, or ended by the benchmark).
    """

    moved: tuple[float, float]
    turned_deg: float
    dist_to_target_m: float
    yaw_error_deg: float
    reached: bool
    blocked: bool
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Torso:
    """Outcome of ``set_torso``.

    height_m: the torso lift afterwards (0 .. 0.34).
    reached: within 5 mm of the height asked.
    steps_used: native actions consumed.
    truncated: the episode is over (budget spent, or ended by the benchmark).
    """

    height_m: float
    reached: bool
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Segment:
    """Outcome of ``segment``.

    found: SAM3 produced a mask scoring at least ``min_score``.
    centroid: (row, col) of the mask centroid, or None.  Pass exactly this
        pixel, with the same camera, to ``back_project``.
    area_px: number of mask pixels (0 when not found).
    score: SAM3 confidence in [0, 1], None when not found.
    box: mask bounding box as reported by SAM3, or None.
    """

    found: bool
    centroid: tuple[int, int] | None
    area_px: int
    score: float | None
    box: Any
    camera: str


@dataclass(frozen=True)
class ObjectSight:
    """Outcome of ``locate``: an object measured from its masks in the colour cameras.

    found: SAM3 produced a mask scoring at least ``min_score``.
    center_xyz: [x, y, z] of the middle of the smallest rectangle around the
        object's footprint, halfway between its bottom and its top, metres.
        A handle, a spout or a tail is part of the footprint and moves the
        middle away from the body's.  None when no 3D measurement was
        possible (see ``note``).
    top_z: height of the object's top.  Seen from behind and above only, it
        may be reported HIGH: ``note`` then gives the range it lies in.
    height_m: ``top_z`` above what the object stands on.
    length_m / width_m: the long and the short side of the smallest rectangle
        around the footprint.
    long_axis_deg: direction of the long side, degrees from +x (-90 .. 90); of
        a square, one of its sides.
    grasp_yaw: the ``yaw`` for ``move_to(..., pitch=90, yaw=grasp_yaw)``:
        pointing straight down, the fingers then close across the short side.
    views: how many cameras agreed on the measurement (1: only the camera
        asked saw it, or the others showed something else: no position).
    score: SAM3 confidence.
    centroid: (row, col) of the mask in the image, as ``Segment.centroid``.
    area_px: mask pixels.
    camera: the camera measured from.
    note: what limits the measurement (one camera only; the top may be
        lower, and how much; the height is one remembered from before; the
        mask is the robot's own body), else None.
    """

    found: bool
    center_xyz: tuple[float, float, float] | None
    top_z: float | None
    height_m: float | None
    length_m: float | None
    width_m: float | None
    long_axis_deg: float | None
    grasp_yaw: float | None
    score: float | None
    centroid: tuple[int, int] | None
    area_px: int
    camera: str
    note: str | None
    views: int = 1


@dataclass(frozen=True)
class SeenPoint:
    """Outcome of ``back_project``: the point seen at one pixel.

    xyz: [x, y, z] where the pixel's ray first meets the thing under the
        pixel, metres; NaN when it could not be measured (``surface``).
    depth_m: distance of that point along the camera's optical axis.
    surface: ``"object"`` when a second camera confirmed the thing under the
        pixel, ``"unmeasured"`` when none did (``xyz`` is NaN then).
    normal: unit vector out of the surface there, towards the camera, or None
        when the neighbourhood was too small to tell.
    views: how many cameras agreed.
    """

    pixel: tuple[int, int]
    xyz: tuple[float, float, float]
    depth_m: float
    camera: str
    surface: str = "unmeasured"
    normal: tuple[float, float, float] | None = None
    views: int = 1


STATEFUL_TOOLS = frozenset({"move_pose", "grip", "home", "move_base", "set_torso"})


def _floats(value: Any, n: int, name: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {n} numbers, got {value!r}") from exc
    if len(vals) != n or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} must be {n} finite numbers, got {value!r}")
    return vals


def _tuple3(value: Any) -> tuple[float, float, float]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        vals = ()
    if len(vals) != 3:
        return (math.nan, math.nan, math.nan)
    return vals  # type: ignore[return-value]


def _arm(name: Any) -> str:
    if name not in ("right", "both"):
        raise ValueError(f"this robot has one arm, 'right'; got {name!r}")
    return "right"


def _camera(name: Any, cameras: tuple[str, ...]) -> str:
    if name not in cameras:
        raise ValueError(f"camera must be one of {cameras}, got {name!r}")
    return str(name)


def _command(value: Any, *, allow_keep: bool) -> int:
    allowed = (-1, 0, 1) if allow_keep else (-1, 1)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if number not in allowed:
        words = "1 (close), -1 (open) or 0 (keep)" if allow_keep else "1 (close) or -1 (open)"
        raise ValueError(f"gripper must be {words}, got {value!r}")
    return int(number)


def _angle(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number of degrees or None, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _arm_state(raw: dict[str, Any]) -> ArmState:
    return ArmState(
        tcp_pos=_tuple3(raw.get("tcp_pos")),
        pitch=float(raw.get("pitch", math.nan)),
        yaw=float(raw.get("yaw", math.nan)),
        roll=float(raw.get("roll", math.nan)),
        eef_pos=_tuple3(raw.get("eef_pos")),
        eef_quat=tuple(float(v) for v in raw.get("eef_quat") or (1.0, 0.0, 0.0, 0.0)),  # type: ignore[arg-type]
        gripper=float(raw.get("gripper", math.nan)),
        joints=tuple(float(v) for v in raw.get("joints") or ()),
    )


# ---------------------------------------------------------------------------
# The robot
# ---------------------------------------------------------------------------


class RobocasaColourRobot(RobotBase):
    """A PandaOmron in a RoboCasa365 kitchen: one 7-DoF Franka arm (``"right"``) with a
    parallel gripper, on a torso lift, on an omnidirectional mobile base.

    ROBOT frame: x to the robot's right, y straight ahead of it, z up from
    the floor; metres.  It is attached to the BASE (its origin on the floor
    under the base's turning axis): every position, from
    ``state()``, ``locate`` or ``back_project``, is measured from where the
    base stands now, and a ``move_base`` makes all earlier positions stale
    (``state().base`` says where the base is relative to where it started).
    Kitchen counters stand about 0.9 m high; read every height from the
    scene, never assume one.  Orientations are unit quaternions [w, x, y, z].
    The robot has ONE arm, ``"right"`` (``robo.RIGHT``); motion calls take
    ``arm`` like RoboDojo's, and refuse any other.

    ``move_to`` drives the arm's GRASP POINT, the middle of the finger pads
    (``state().right.tcp_pos``), to a position, with an orientation given in
    degrees: ``pitch`` is how far the gripper points below the horizontal (0
    level, 90 straight down), ``yaw`` the heading of the pointing direction
    (0 straight ahead, +90 towards the robot's left, -90 towards its right)
    and ``roll`` the turn of the fingers about the pointing direction (0:
    side by side).  Pointing straight down, the fingers close along the line
    at ``yaw`` degrees from +x (counter-clockwise seen from above):
    ``locate``'s ``grasp_yaw`` is that line across an object's narrow side.
    At the start the arm is folded up in front of the robot, the gripper
    pointing ahead and down.

    Two gripper conventions, do not mix them up.  COMMANDS are signed:
    ``robo.CLOSE`` (1) closes, ``robo.OPEN`` (-1) opens, ``robo.KEEP`` (0)
    leaves the gripper as it is.  MEASUREMENTS (``state().right.gripper``,
    ``Gripper.opening``) are openings: about 0.97 open, 0.0 closed on nothing.

    Every native action (one step of any motion, of the gripper, the base or
    the torso) counts against the episode's native-action budget, which is
    RoboCasa's own horizon for the task; when it is spent, or when the
    benchmark has ended the episode (it checks its conditions after every
    native action), the state reports ``truncated`` and motion calls raise
    ``EpisodeFinished``.  Motion calls block until the primitive finishes
    and return a typed result; nothing is queued.

    Perception is from colour images alone; there is no depth image.  Three
    cameras ride on the robot, 640x480 each: ``agentview_left`` and
    ``agentview_right`` (on its shoulders, 0.7 m apart, looking ahead and
    down past the arm: the layout, fixtures and objects in front) and
    ``eye_in_hand`` (on the gripper: what it is about to grasp or holds).
    Where each one is, is known from the robot's state.  ``locate`` finds an
    object by a text prompt and measures it in 3D from the cameras that see
    it: it takes two of them to measure anything.  ``segment`` /
    ``back_project`` give a mask centroid and the point seen at one pixel.
    Object poses are never given.
    """

    ARMS: tuple[str, ...] = ("right",)
    RIGHT: str = "right"
    CLOSE: int = 1
    OPEN: int = -1
    KEEP: int = 0

    NAME: ClassVar[str] = "robocasa_rgb"
    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        ArmState,
        Move,
        Gripper,
        Home,
        BaseMove,
        Torso,
        ObjectSight,
        Segment,
        SeenPoint,
    )
    #: No success flag while the episode runs (as on RoboDojo) and no world maps.
    ABSENT_MEMBERS: ClassVar[frozenset[str]] = frozenset({"done", "world_map"})
    CAMERAS: ClassVar[tuple[str, ...]] = ("agentview_left", "agentview_right", "eye_in_hand")
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("agentview_left", "high"): "agentview_left.png",
        ("agentview_right", "high"): "agentview_right.png",
        ("eye_in_hand", "high"): "eye_in_hand.png",
    }
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {}
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = (
        "agentview_left.png",
        "agentview_right.png",
        "eye_in_hand.png",
    )
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = STATEFUL_TOOLS
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        "reached",
        "dist_to_target_m",
        "orientation_error_deg",
        "final_xyz",
        "path",
        "blocked",
        "steps_used",
        "moved",
        "turned_deg",
        "height_m",
        "max_joint_error_rad",
        "holding",
        "opening",
        "width_m",
        "ik_error",
        "found",
        "score",
        "views",
        "center_xyz",
        "top_z",
        "grasp_yaw",
        "world_xyz",
        "surface",
        "truncated",
        "step",
    )

    # -- observation --------------------------------------------------------

    @property
    def task(self) -> str:
        """The task instruction the benchmark generated for this scene (authoritative).

        Read it before planning.
        """
        return super().task

    @property
    def done(self) -> bool:
        """Absent (not in the API reference): there is no success flag to read."""
        raise AttributeError(
            "this benchmark reports no task success flag while the episode runs: there is no "
            "robo.done; it ends the episode once its conditions hold (`truncated` with budget "
            "left); judge progress from the images and the robot state"
        )

    def _reported_success(self) -> bool:
        # the benchmark ends the episode the moment the task is solved; the hosts may stop there
        return bool(self._backend.solved())

    def state(self, step: int = -1) -> State:
        """Read the recorded state of the robot.

        Args:
            step: recorded step to read; ``-1`` is the latest, ``0`` the
                initial scene.  A new step is recorded after every motion call.
        Returns:
            State with the arm's grasp point, gripper angles and opening,
            joint angles, the base's odometry, the torso lift and the
            ``truncated`` flag.
        """
        return super().state(step)

    def image(
        self, camera: str = "agentview_left", *, resolution: str = "high", step: int = -1
    ) -> Any:
        """Load a recorded camera image as a ``numpy`` uint8 array (480, 640, 3), RGB.

        Args:
            camera: ``"agentview_left"`` / ``"agentview_right"`` (shoulders:
                identify objects, fixtures and the layout ahead) or
                ``"eye_in_hand"`` (on the gripper: what it is about to grasp
                or is holding).
            resolution: there is one resolution per camera, ``"high"``.  Pixel
                coordinates are (row, col) with row 0 at the top and col 0 at
                the left.
            step: recorded step, ``-1`` = latest.  ``segment``, ``locate`` and
                ``back_project`` always read the CURRENT observation, which is
                the latest recorded step.
        """
        return super().image(camera, resolution=resolution, step=step)

    def world_map(
        self, camera: str = "agentview_left", *, resolution: str = "high", step: int = -1
    ) -> Any:
        """Absent (not in the API reference): there is no depth, so no world maps."""
        raise ValueError("there is no depth image; use locate / back_project")

    def camera_meta(self, camera: str = "agentview_left", *, step: int = -1) -> dict[str, Any]:
        """Return the calibration of one camera at a recorded step as a dict.

        Keys: ``shape`` (image shape), ``intrinsic_matrix`` (3x3) and
        ``extrinsic_matrix`` (4x4 camera-to-ROBOT-frame; the camera looks down
        its own -z axis with +y up).  All three cameras move with the robot
        (the shoulder ones with the torso, the wrist one with the gripper), so
        read the meta of the same step as the image.
        """
        camera = _camera(camera, self.CAMERAS)
        payload = self._readonly(self.STATE_TOOL, {"step": int(step)})
        cameras = (payload.get("state") or {}).get("cameras") or {}
        return dict(cameras.get(camera) or {})

    def show(self, camera: str = "agentview_left") -> None:
        """Attach the current image of ``camera`` to the feedback of this turn.

        Images are not returned automatically; call ``show("agentview_left")``,
        ``show("agentview_right")`` or ``show("eye_in_hand")`` when *you* (the
        author) need to look at the scene before writing the next turn.  Costs
        tokens: look only when a decision depends on it.
        """
        super().show(camera)

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return the raw bytes of a recorded artifact (advanced).

        Names are ``agentview_left.png``, ``agentview_right.png`` and
        ``eye_in_hand.png``.
        """
        return super().artifact(name, step=step)

    # -- perception ---------------------------------------------------------

    def segment(
        self, text_prompt: str, camera: str = "agentview_left", min_score: float = 0.2
    ) -> Segment:
        """Segment an object in the current image of ``camera`` by a text prompt (SAM3).

        Args:
            text_prompt: what to find, e.g. ``"the red mug"``.  Name one object
                by its visible properties.
            camera: ``"agentview_left"`` (default), ``"agentview_right"`` or
                ``"eye_in_hand"``.
            min_score: masks scoring below this are reported as not found.
        Returns:
            Segment; when ``found``, pass ``centroid`` (row, col) to
            ``back_project`` with the same camera.  The mask is the best match
            for the text, not a guarantee of identity: check where it is
            against the image.
        Raises:
            ToolError: the camera image or the SAM3 service is unavailable.
        """
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        kwargs = {"text_prompt": prompt, "camera": camera, "min_score": float(min_score)}
        p = self._readonly("segment", kwargs)
        if not p.get("found"):
            return Segment(False, None, 0, None, None, str(p.get("camera", camera)))
        centroid = p.get("centroid_rc") or ()
        return Segment(
            found=True,
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            score=float(p["score"]) if p.get("score") is not None else None,
            box=p.get("box_px"),
            camera=str(p.get("camera", camera)),
        )

    def back_project(self, row: int, col: int, camera: str = "agentview_left") -> SeenPoint:
        """Return the point seen at one pixel of the current image of ``camera``.

        There is no depth image, so a pixel alone is only a ray.  SAM3
        outlines the thing under the pixel; the same thing is looked for in
        the other cameras along that ray, and the answer is where the ray
        first meets what they agree on, with the surface's normal there.
        Small things with an outline of their own (a handle, a knob, a button,
        an object) are measured to a few cm; a pixel on a large plain surface
        (a counter, a wall, a cabinet front) is not measured reliably: its
        outline is everywhere.  Prefer ``locate`` with words for the thing.

        Args:
            row / col: pixel, row 0 at the top, col 0 at the left (images are
                480 rows x 640 cols).
            camera: the camera the pixel was read from (default
                ``"agentview_left"``).
        Returns:
            SeenPoint; check ``surface``: ``"object"`` (``xyz`` and ``normal``
            are measured) or ``"unmeasured"`` (only one camera shows it:
            ``xyz`` is NaN).
        Raises:
            ToolError: pixel out of range.
        """
        camera = _camera(camera, self.CAMERAS)
        kwargs: dict[str, Any] = {"row": int(row), "col": int(col), "camera": camera}
        p = self._readonly("back_project", kwargs)
        normal = p.get("normal")
        return SeenPoint(
            pixel=(int(row), int(col)),
            xyz=_tuple3(p.get("world_xyz")),
            depth_m=float(p.get("depth_m", math.nan)),
            camera=str(p.get("camera", camera)),
            surface=str(p.get("surface") or "unmeasured"),
            normal=_tuple3(normal) if normal is not None else None,
            views=int(p.get("views") or 1),
        )

    def locate(
        self, text_prompt: str, camera: str = "agentview_left", min_score: float = 0.2
    ) -> ObjectSight:
        """Find an object by a text prompt and measure it in 3D from the colour cameras.

        SAM3 segments the current image of ``camera`` (as ``segment``); the
        same object is looked for in the other two cameras (of several alike
        things, the one whose outline agrees with the first in 3D), and the
        object is the space inside its outline in every camera that sees it
        (outlines and geometry only: no pixel is matched between cameras).
        ``views`` says how many cameras agreed; it takes two to measure
        anything.  Nothing moves.

        Args:
            text_prompt: one object by its visible properties, e.g. ``"the
                white mug on the left"``, ``"the handle of the right drawer"``.
            camera: ``"agentview_left"`` (default), ``"agentview_right"`` or
                ``"eye_in_hand"`` (close up, from the gripper).
            min_score: masks scoring below this are reported as not found.
        Returns:
            ObjectSight.  Measured from the starting pose, a knob, a handle or
            a mug comes out within 1 cm in x and y, a bottle or a carton
            within about 4 cm; ``top_z`` may be several cm HIGH (``note`` gives
            the range): to measure a height, look from the side (the wrist
            camera about level with the object) and ``locate`` again.  The mask
            is the best match for the text, not proof of identity: check it
            against the image.
        Raises:
            ToolError: the camera image or the SAM3 service is unavailable.
        """
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        kwargs = {"text_prompt": prompt, "camera": camera, "min_score": float(min_score)}
        p = self._readonly("locate", kwargs)
        centroid = p.get("centroid_rc") or ()

        def num(key: str) -> float | None:
            return float(p[key]) if p.get(key) is not None else None

        center = p.get("center_xyz")
        note = p.get("geometry_error") or p.get("height_guess")
        return ObjectSight(
            found=bool(p.get("found")),
            center_xyz=_tuple3(center) if center is not None else None,
            top_z=num("top_z"),
            height_m=num("height_m"),
            length_m=num("length_m"),
            width_m=num("width_m"),
            long_axis_deg=num("long_axis_deg"),
            grasp_yaw=num("grasp_yaw"),
            score=num("score"),
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            camera=str(p.get("camera", camera)),
            note=str(note) if note else None,
            views=int(p.get("views") or 1),
        )

    # -- motion -------------------------------------------------------------

    def move_to(
        self,
        xyz: Sequence[float],
        arm: str = "right",
        gripper: int = 0,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
        path: str = "auto",
    ) -> Move:
        """Move the arm's grasp point to a position, with an orientation.

        The grasp point is the middle of the finger pads (``tcp_pos``).  The
        library checks the pose against the arm's reach and, with
        ``path="auto"``, moves the grasp point along a straight line to it, at
        about 1-2 cm or 3-4 degrees per native action; where the line would
        turn a joint into its stop, the gripper turns the other way round, and
        where no line works, the arm goes round by way of the gripper's
        starting pose.  Nothing moves when the pose is out of reach: ``ik_error``
        says why.

        Args:
            xyz: target grasp-point position [x, y, z], metres (robot frame).
            arm: ``"right"``, the one arm.
            gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1) to command the
                gripper on every action of the motion, ``robo.KEEP`` (0) to keep
                commanding what it was last told (a held object stays
                squeezed).  Keep it closed (1) while carrying an object.
            pitch / yaw / roll: the orientation to arrive with, degrees (the
                class docstring defines them).  An angle left None keeps its
                current value, so a plain position move does not turn the
                gripper; pointing straight down (``pitch=90``) a roll left
                None is 0, so ``yaw`` alone sets the finger line, and a yaw
                left None keeps the line the fingers close along now.  With
                an EMPTY open gripper the fingers are
                interchangeable, so the pose turned half a turn about the
                pointing direction may be taken when it is easier (``note``
                says so; yaw / roll then read 180 off); holding something, the
                pose is taken exactly as asked.
            path: ``"auto"`` (default), ``"line"`` (straight line or nothing)
                or ``"joint"`` (go round by way of the starting pose: not near
                objects).
        Returns:
            Move; test ``reached`` (within 0.01 m and 5 degrees).  ``blocked``
            means the arm was stopped by something on the way.
        Raises:
            ValueError: a malformed argument (nothing is sent).
        """
        _arm(arm)
        target = _floats(xyz, 3, "xyz")
        command = _command(gripper, allow_keep=True)
        if path not in ("auto", "line", "joint"):
            raise ValueError(f"path must be 'auto', 'line' or 'joint', got {path!r}")
        kwargs: dict[str, Any] = {"xyz": list(target), "gripper": command, "path": path}
        for name, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
            angle = _angle(value, name)
            if angle is not None:
                kwargs[name] = angle
        return self._stateful("move_pose", kwargs, lambda p: self._move(target, p))

    def set_gripper(self, arm: str, gripper: int, opening: float | None = None) -> Gripper:
        """Close or open the gripper, holding the arm where it is.

        The command is repeated until the fingers stop (about 10 native
        actions to close, 15 to open fully).  After a close, ``holding`` tells
        whether something stopped the fingers apart and ``width_m`` how wide
        it is.  Later motions with ``robo.KEEP`` keep commanding what was
        commanded last, so a held object stays squeezed and a partial opening
        stays as it is.

        Args:
            arm: ``"right"``, the one arm.
            gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1).
            opening: with ``robo.OPEN`` only: open to this reading
                (0 < opening <= 1) instead of all the way, e.g. to reach
                between close neighbours.  Something 0.08 x opening metres wide
                then just fits between the fingers.
        Returns:
            Gripper.  ``holding`` is not proof that the INTENDED object is
            held: confirm it moves with the arm (wrist camera, or ``locate``
            it again after a lift).
        """
        _arm(arm)
        command = _command(gripper, allow_keep=False)
        kwargs: dict[str, Any] = {"gripper": command}
        if opening is not None:
            try:
                fraction = float(opening)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"opening must be a number in (0, 1], got {opening!r}") from exc
            if command != -1 or not 0.0 < fraction <= 1.0:
                raise ValueError("opening goes with robo.OPEN and must be in (0, 1]")
            kwargs["opening"] = fraction

        def build(p: dict[str, Any]) -> Gripper:
            status = p.get("status") or {}
            state = self._latest_state()
            width = p.get("width_m")
            return Gripper(
                arm="right",
                commanded=str(p.get("gripper") or ("closed" if command > 0 else "open")),
                opening=float(p.get("opening", state.right.gripper)),
                holding=bool(p.get("holding")),
                width_m=float(width) if width is not None else None,
                steps_used=int(p.get("steps_used", 0)),
                step=int(status["step"]) if status.get("step") is not None else None,
                step_limit=(
                    int(status["step_limit"]) if status.get("step_limit") is not None else None
                ),
                truncated=state.truncated,
            )

        return self._stateful("grip", kwargs, build)

    def home(self, arm: str = "right") -> Home:
        """Bring the gripper back to the pose it had on the torso when the episode started.

        A grasp point away from there first goes straight up by up to 6 cm,
        then along a straight line (turning the other way round where needed)
        to the starting pose; the base and the torso stay where they are, and
        the gripper keeps the opening it has.  It plans no way round things:
        clear tall things yourself.  Many tasks count as done only once the
        gripper is well away from what it handled: ``home`` after releasing
        is the way to get there.

        Args:
            arm: ``"right"`` (``"both"`` is accepted: there is one arm).
        Returns:
            Home; test ``reached`` and call again when it is False.
        """
        _arm(arm)

        def build(p: dict[str, Any]) -> Home:
            return Home(
                arm="right",
                reached=bool(p.get("reached")),
                max_joint_error_rad=float(p.get("max_joint_error_rad", math.inf)),
                steps_used=int(p.get("steps_used", 0)),
                truncated=self._latest_state().truncated,
            )

        return self._stateful("home", {}, build)

    def move_base(self, dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0) -> BaseMove:
        """Drive the mobile base: ``dx`` to its right, ``dy`` ahead (metres), then turned ``dyaw``.

        The base is omnidirectional: it drives straight to the point (also
        sideways) while turning by ``dyaw`` degrees (counter-clockwise +), at
        up to about 2.5 cm or 3 degrees per native action, slowing down near
        the goal, under closed-loop odometry.  There is no path planning: it
        stops when it is stuck (``blocked``).  The arm keeps its place on the
        base.  Afterwards every position measured before is in the OLD frame:
        locate again (or convert with ``moved`` / ``turned_deg``).

        Args:
            dx / dy: metres, in the robot frame as it is now (x right, y
                ahead).
            dyaw: degrees, counter-clockwise seen from above.
        Returns:
            BaseMove; test ``reached`` and ``blocked``.
        """
        vals = _floats((dx, dy, dyaw), 3, "dx, dy, dyaw")
        kwargs = {"dx": vals[0], "dy": vals[1], "dyaw": vals[2]}

        def build(p: dict[str, Any]) -> BaseMove:
            moved = p.get("moved") or (0.0, 0.0)
            return BaseMove(
                moved=(float(moved[0]), float(moved[1])),
                turned_deg=float(p.get("turned_deg", 0.0)),
                dist_to_target_m=float(p.get("dist_to_target_m", math.nan)),
                yaw_error_deg=float(p.get("yaw_error_deg", math.nan)),
                reached=bool(p.get("reached")),
                blocked=bool(p.get("blocked")),
                steps_used=int(p.get("steps_used", 0)),
                truncated=self._latest_state().truncated,
            )

        return self._stateful("move_base", kwargs, build)

    def set_torso(self, height: float) -> Torso:
        """Raise or lower the torso lift to ``height`` metres (0 .. 0.34; it starts at 0).

        The arm, the gripper and the shoulder cameras ride up and down with
        it (the gripper keeps its pose on the torso): it adds up to 0.34 m to
        the arm's reach upwards (a high shelf, a cabinet above the counter).
        About 2.5 cm per native action.

        Args:
            height: metres.
        Returns:
            Torso.
        """
        h = float(height)
        if not math.isfinite(h) or not 0.0 <= h <= 0.34:
            raise ValueError("height must be within 0 .. 0.34 m")

        def build(p: dict[str, Any]) -> Torso:
            return Torso(
                height_m=float(p.get("height_m", math.nan)),
                reached=bool(p.get("reached")),
                steps_used=int(p.get("steps_used", 0)),
                truncated=self._latest_state().truncated,
            )

        return self._stateful("set_torso", {"height": h}, build)

    # -- internals ----------------------------------------------------------

    def _latest_state(self) -> State:
        if self._last_state is None:
            return self.state()
        return self._last_state

    def _move(self, target: tuple[float, ...], p: dict[str, Any]) -> Move:
        commanded = p.get("target_xyz")
        angles = p.get("final_angles") or {}
        return Move(
            arm="right",
            target_xyz=_tuple3(commanded if commanded is not None else target),
            start_xyz=_tuple3(p.get("start_xyz")),
            final_xyz=_tuple3(p.get("final_xyz")),
            dist_to_target_m=float(p.get("dist_to_target_m", math.inf)),
            orientation_error_deg=float(p.get("orientation_error_deg", math.nan)),
            pitch=float(angles.get("pitch", math.nan)),
            yaw=float(angles.get("yaw", math.nan)),
            roll=float(angles.get("roll", math.nan)),
            reached=bool(p.get("reached")),
            path=str(p["path"]) if p.get("path") else None,
            ik_error=str(p["ik_error"]) if p.get("ik_error") else None,
            note=str(p["note"]) if p.get("note") else None,
            blocked=bool(p.get("blocked")),
            steps_used=int(p.get("steps_used", 0)),
            truncated=self._latest_state().truncated,
        )

    def _state_from_envelope(self, envelope: dict[str, Any]) -> State:
        summary = envelope.get("state") or {}
        raw = summary.get("state") or {}
        base = raw.get("base") or (0.0, 0.0, 0.0)
        return State(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or summary.get("instruction") or ""),
            right=_arm_state(raw.get("arm") or {}),
            base=(float(base[0]), float(base[1]), float(base[2])),
            torso=float(raw.get("torso_m", 0.0)),
            truncated=bool(envelope.get("truncated")),
        )

    def _absorb(self, envelope: dict[str, Any] | None) -> None:
        """Latch ``truncated`` only: nothing an envelope says may end the episode as a success."""
        if envelope is None:
            return
        state = self._state_from_envelope(envelope)
        self._last_state = state
        if state.truncated:
            self._truncated = True

    def _summary_dict(self) -> dict[str, Any]:
        """Compact state for the per-turn tool feedback (host-facing)."""
        state = self.state()
        arm = state.right
        return {
            "step": state.step,
            "right_tcp": [round(v, 4) for v in arm.tcp_pos],
            "right_pitch_yaw_roll": [round(arm.pitch, 1), round(arm.yaw, 1), round(arm.roll, 1)],
            "right_gripper": round(arm.gripper, 3),
            "base": [round(v, 3) for v in state.base],
            "torso": round(state.torso, 3),
            "truncated": state.truncated,
        }


ADAPTER = RobotAdapter(
    name="robocasa_rgb",
    robot_cls=RobocasaColourRobot,
    host="pyrualean.hosts.robocasa_rgb",
    knowledge="robocasa_rgb",
    blurb="a mobile Franka arm (PandaOmron) in the RoboCasa365 kitchen benchmark",
    guides_subdir="",
    guide_names=(),
    show_examples=(
        "`robo.show('agentview_left')`, `robo.show('agentview_right')` or "
        "`robo.show('eye_in_hand')`"
    ),
    services="a SAM3 segmentation service",
    services_novla="a SAM3 segmentation service",
)


__all__ = [
    "ADAPTER",
    "ArmState",
    "BaseMove",
    "Gripper",
    "Home",
    "Move",
    "ObjectSight",
    "RobocasaColourRobot",
    "STATEFUL_TOOLS",
    "Segment",
    "SeenPoint",
    "State",
    "Torso",
]
