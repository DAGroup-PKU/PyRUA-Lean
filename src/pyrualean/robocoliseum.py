# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The G2_omnipicker robot under the GenieSim benchmark, through a bridge.

The benchmark owns the simulator: it sends an observation, this library answers with one
native action, and the benchmark steps the scene and sends the next observation.  What a
program sees is therefore the scene after every action it asked for.

Primitives: ``move_to`` / ``move_both`` (the tool centre to a point, pointing one way),
``set_gripper`` (open or close, reporting what stopped the fingers), ``move_waist`` /
``move_base``, ``home`` (back to the start pose), ``wait_for_check`` (hold still until the
benchmark's next check) and ``locate`` / ``back_project`` / ``segment`` (measuring what the
cameras see).

The docstrings in this module are the model-facing documentation: the prompt is generated
from them (see :mod:`pyrualean.prompt`), so keep them accurate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ._robot import RobotBase
from .hosts import robocoliseum_motion as motion
from .hosts import robocoliseum_sight as sight
from .hosts import robocoliseum_tools as tools
from .robots import RobotAdapter

#: Native actions one ``move_to`` may spend before it reports where it stopped.
MOVE_MAX_ACTIONS = tools.MOVE_MAX_ACTIONS
#: A motion that arrives within this and this much pointing error is ``reached``.
MOVE_TOL_M = tools.ARRIVE_M
MOVE_TOL_DEG = tools.ARRIVE_DEG
#: Native actions ``home`` may spend.
HOME_MAX_ACTIONS = 25


# ---------------------------------------------------------------------------
# Result types (returned to the policy; all fields are plain Python values)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmState:
    """One arm at a recorded step.

    tool_centre: [x, y, z] of the middle of the jaw plane, metres, in ``robo.FRAME``
        (``base_link``): this is what ``move_to`` drives.
    pointing: the unit direction the tool points, in ``robo.FRAME``.
    finger_axis: the unit line the fingers close along.
    pitch / yaw / roll: the tool's orientation in degrees, as ``move_to`` takes it.
    flange: position of the end-effector link (the wrist flange), metres.
    opening: measured gripper opening, NORMALISED here: 1.0 = open, 0.0 = closed.
        (``set_gripper``'s COMMANDS use the library's convention, see the class
        docstring.)
    """

    tool_centre: tuple[float, float, float]
    pointing: tuple[float, float, float]
    finger_axis: tuple[float, float, float]
    flange: tuple[float, float, float]
    opening: float
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None


@dataclass(frozen=True)
class State:
    """Snapshot of both arms at one recorded step.

    step: index of the recorded step (0 = the scene the episode started in).  A new step
        is recorded after every tool call that acted.
    task: the task instruction the benchmark generated (authoritative).
    left / right: the two arms.
    torso_heading_deg / torso_forward: where the upper body faces.  ``robo.FRAME`` is fixed
        to the robot's base and some tasks start with the torso TURNED (the warehouse ones
        by 90 degrees): the table is then along ``torso_forward``, not along +x.
        ``point="forward"`` / ``"left"`` / ``"right"`` follow the torso.
    """

    step: int
    task: str
    left: ArmState
    right: ArmState
    torso_heading_deg: float = 0.0
    torso_forward: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def arm(self, name: str) -> ArmState:
        if name == "left":
            return self.left
        if name == "right":
            return self.right
        raise ValueError("arm must be 'left' or 'right'")


@dataclass(frozen=True)
class Move:
    """Outcome of ``move_to``.

    target_tool_centre / final_tool_centre: where the tool centre was sent, and where it
        ended.
    dist_to_target_m: distance between the two.
    pointing_error_deg: how far the final pointing direction is from the one asked (0
        when none was asked).
    rotation_error_deg: how far the whole final orientation is from the one asked — the
        pointing direction AND the fingers' turn about it.
    reached: within 0.015 m and 5 degrees of the whole orientation.
    steps_used: native actions consumed; ``steps_planned`` what the subdividing asked for
        (a motion stopped early because it arrived).
    truncated: it used every action ``max_actions`` allowed and had not arrived.
    stop_reason: why a motion that did not arrive stopped: ``"out_of_reach"`` (the arm
        kept moving but no longer got closer — the edge of its reach from this waist
        pose), ``"blocked"`` (the arm stopped moving: something is in the way, it pushes
        against what it holds — at the end of a push, the end of the travel — or a joint
        is at its end), ``"budget"`` (still closing in when the actions ran out: call
        again), or None when it arrived.
    blocked: ``stop_reason == "blocked"``.
    reach_hint: set with ``"out_of_reach"``.  Only then consider ``move_waist``.
    alternatives: when the target was refused before moving (``steps_used == 0`` and
        ``"out_of_reach"``: the benchmark's own IK, asked first, found no pose of this arm
        within 3 cm and 10 degrees of it), nearby requests that the same IK says ARE
        reachable — each a dict naming the change and the call(s) that make it, e.g.
        ``{"change": "pointing tilted 15 deg toward vertical", "move_to": {"xyz": [...],
        "point": [...]}}`` or one with ``"move_waist": {...}`` to run first.
    """

    arm: str
    target_tool_centre: tuple[float, float, float]
    final_tool_centre: tuple[float, float, float]
    dist_to_target_m: float
    pointing_error_deg: float
    reached: bool
    steps_used: int
    steps_planned: int
    truncated: bool
    reach_hint: str | None = None
    rotation_error_deg: float = math.nan
    stop_reason: str | None = None
    blocked: bool = False
    alternatives: tuple[dict[str, Any], ...] = ()
    angles: dict[str, float] | None = None
    path: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class BothMove:
    """Outcome of ``move_both``: ``left`` / ``right`` are each arm's ``final_tool_centre``,
    ``dist_to_target_m``, ``reached`` and ``stop_reason`` (None for an arm that held)."""

    reached: bool
    left: dict[str, Any] | None
    right: dict[str, Any] | None
    steps_used: int


@dataclass(frozen=True)
class BaseMove:
    """Outcome of ``move_base``.  OPEN LOOP: ``commanded`` is what was sent, not a measured
    displacement; look with the head camera to see where the base went."""

    commanded: dict[str, float]
    steps_used: int


@dataclass(frozen=True)
class Gripper:
    """Outcome of ``set_gripper``.

    commanded: ``"closed"`` or ``"open"``.
    opening: the measured opening once the fingers stopped, normalised: 1.0 = open,
        0.0 = closed on nothing.
    holding: after a close, the fingers stopped apart: something is between them.  It
        does not say what: check that the intended object moves with the arm.
    steps_used: native actions consumed.
    """

    arm: str
    commanded: str
    opening: float
    holding: bool
    steps_used: int


@dataclass(frozen=True)
class Waist:
    """Outcome of ``move_waist``.

    moved: the waist was commanded; False when the asked pose is outside the joints' range
        (``error_reason`` says which joint), and nothing moved.
    reached: the waist joints ended where they were sent.
    chest_moved_m: how far the chest went [forward, up], metres, in ``robo.FRAME``.  The
        hands moved by about the same amount: the arms keep their joint angles.
    torso_pitch_deg: how far the torso leans forward from upright (it starts near 28).
    turn_deg: how far the upper body is turned (+ to the robot's left).
    steps_used: native actions consumed.
    """

    moved: bool
    reached: bool
    chest_moved_m: tuple[float, float]
    torso_pitch_deg: float | None
    turn_deg: float | None
    error_reason: str | None
    steps_used: int


@dataclass(frozen=True)
class Home:
    """Outcome of ``home``.

    reached: every joint of the returned arm(s) ended within 0.02 rad of the angle it had
        when the episode started.
    max_joint_error_rad: the largest difference that remains.
    steps_used: native actions consumed.
    """

    arm: str
    reached: bool
    max_joint_error_rad: float
    steps_used: int


@dataclass(frozen=True)
class WorldPoint:
    """Outcome of ``back_project``: the surface point seen at one pixel.

    xyz: [x, y, z] in ``robo.FRAME``, metres.  It is a visible SURFACE point, not a
        centre.
    depth_m: distance of that point along the camera's optical axis.
    """

    pixel: tuple[int, int]
    xyz: tuple[float, float, float]
    depth_m: float
    camera: str


@dataclass(frozen=True)
class ObjectPose:
    """Outcome of ``locate``: one colour region of the scene, measured in 3D.

    found: a region of contiguous colour was found in the head camera.
    centre_xyz: [x, y, z] of the middle of its footprint, at half its height above the
        surface it stands on, metres.
    top_z: the highest coordinate seen along ``robo.FRAME`` z.
    height_m: ``top_z`` above that surface.
    size_xyz: the region's extent along each axis of ``robo.FRAME``.
    pixel_uv: (row, col) of the region in the image it was measured in.
    area_px: pixels of the region.
    colour: mean colour of the region as a hex string, so a program can say which.
    camera: the camera measured from.
    note: why there is no 3D measurement, else None.
    tilt_deg: how far the object's upward face is tilted from horizontal, degrees (0 = lying
        flat / standing upright).  Tasks that check "upright" usually allow 5-15 degrees:
        re-``locate`` after placing and fix it if this is larger.
    long_axis_deg: direction of the long side of that face on the table, degrees from
        ``robo.FRAME`` +x (-90..90).  To close across the SHORT side, pointing down, turn the
        fingers along this line.
    support_z: height of the surface it stands on (the table top), metres.
    method: how the object was cut out of the image: ``"sam3_text"`` (SAM3 found the text
        you named), ``"geometry"`` (what rises above its surface at the pixel),
        ``"sam3_point"`` (SAM3's mask at the pixel), ``"colour"`` (the pixel's colour region).
    score: SAM3's confidence in its mask (0..1), when SAM3 made it.
    grasp_fingers: for an elongated object (a bottle, a pen, a box lying down), the
        horizontal direction ACROSS it in ``robo.FRAME``: the fingers close across the short
        side when they close along it (``grasp_yaw`` is the yaw that does so).  None for a
        round or square one (any direction closes on it).
    grasp_yaw: the ``yaw`` that, pointing straight down (``pitch=90``), closes the fingers
        across the object's narrow side: ``move_to([x, y, z], arm, pitch=90,
        yaw=pose.grasp_yaw)``.  None for a round or square object.
    width_m / length_m: the object's horizontal extent across and along its long axis —
        ``width_m`` is what the fingers must span.
    candidates: for a named object, every instance SAM3 matched to the name, best first:
        ``{"pixel_uv": (row, col), "score", "area_px"}``.  The measured one is the best
        scoring unless ``near`` chose another; when several match, check which is meant.
    """

    found: bool
    centre_xyz: tuple[float, float, float] | None
    top_z: float | None
    height_m: float | None
    size_xyz: tuple[float, float, float] | None
    pixel_uv: tuple[int, int] | None
    area_px: int
    colour: str | None
    camera: str
    note: str | None
    tilt_deg: float | None = None
    long_axis_deg: float | None = None
    support_z: float | None = None
    method: str | None = None
    score: float | None = None
    candidates: tuple[dict[str, Any], ...] = ()
    grasp_fingers: tuple[float, float, float] | None = None
    grasp_yaw: float | None = None
    width_m: float | None = None
    length_m: float | None = None


@dataclass(frozen=True)
class Segment:
    """Outcome of ``segment``: a colour region of one camera.

    found: a region of contiguous colour was found.
    centroid: (row, col) of the region's centroid; pass it to ``back_project``.
    area_px: pixels of the region.
    colour: mean colour of the region as a hex string.
    box: (row0, col0, row1, col1) around the region.
    """

    found: bool
    centroid: tuple[int, int] | None
    area_px: int
    colour: str | None
    box: tuple[int, int, int, int] | None
    camera: str
    method: str | None = None
    score: float | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _normalised(opening: float) -> float:
    """The benchmark's signed opening (open is negative) as 0 (closed) .. 1 (open)."""

    return max(0.0, min(1.0, float(opening) / tools.GRIP_OPEN))


def _arm_state(raw: dict[str, Any]) -> ArmState:
    rotation = tools.quat_xyzw_matrix(raw.get("flange_quat_xyzw") or [0.0, 0.0, 0.0, 1.0])
    flange = np.asarray(raw.get("flange") or [0.0, 0.0, 0.0], dtype=np.float64)
    return ArmState(
        tool_centre=tuple(round(float(v), 4) for v in tools.tool_centre(flange, rotation)),
        pointing=tuple(round(float(v), 3) for v in rotation[:, 2]),
        finger_axis=tuple(round(float(v), 3) for v in rotation[:, 0]),
        flange=tuple(round(float(v), 4) for v in flange),
        opening=round(_normalised(raw.get("opening") or 0.0), 3),
        **dict(zip(("pitch", "yaw", "roll"), motion.angles(rotation), strict=False)),
    )


def _stop_reason(p: dict[str, Any], done: bool) -> str | None:
    """The tool's stop reason, with ``"budget"`` for a motion that ran out of actions."""

    if done:
        return None
    return p.get("stop_reason") or "budget"


def _arm(name: str) -> str:
    if name not in tools.ARMS:
        raise ValueError(f"arm must be 'left' or 'right', got {name!r}")
    return name


# ---------------------------------------------------------------------------
# the robot
# ---------------------------------------------------------------------------


class RobocoliseumRobot(RobotBase):
    """The G2 dual-arm humanoid with two-finger grippers, in the GenieSim benchmark.

    Frame: ``robo.FRAME`` is ``base_link``, fixed to the robot's base: +x forward, +y the
    robot's left, +z up; metres and degrees.  The upper body may start turned on its waist:
    ``state().torso_heading_deg`` says where the chest (and the work surface) faces.  The
    arms are ``"left"`` and ``"right"``; a motion names one arm and the other holds.

    Orientation is three angles (``state().right.pitch``, ``.yaw``, ``.roll``): ``pitch`` 0
    points level, 90 straight down; ``yaw`` is the heading of the pointing direction, 0 along
    +x, +90 towards +y; ``roll`` turns the fingers about the pointing direction (0: they close
    horizontally).  Pointing straight down, ``yaw = locate(...).grasp_yaw`` closes the fingers
    across an object's narrow side.

    ``move_to`` drives the GRASP POINT (the gripper's centre, 0.143 m out from the wrist
    flange): it solves the arm's inverse kinematics for the whole pose, refuses one out of
    reach without moving (and names reachable ones), and follows a straight line
    (``path="line"``) or a joint swing.

    Gripper: COMMANDS are ``robo.CLOSE``, ``robo.OPEN`` and ``robo.KEEP`` (None);
    MEASUREMENTS (``state().right.opening``, ``Gripper.opening``) are openings, 1.0 open and
    0.0 closed on nothing.

    Perception: ``locate("words")`` finds an object by description and measures it in 3D from
    the depth image; ``locate(row, col)`` measures the object at a pixel.  Object poses are
    never given.  Every native action counts against the stage's budget (on the card when
    known).
    """

    ARMS: tuple[str, str] = ("left", "right")
    LEFT: str = "left"
    RIGHT: str = "right"
    CLOSE: float = tools.GRIP_CLOSE
    OPEN: float = tools.GRIP_OPEN
    KEEP: None = None
    #: The frame every position and direction is expressed in.
    FRAME: str = tools.FRAME

    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        ArmState,
        Move,
        Gripper,
        Waist,
        Home,
        ObjectPose,
        Segment,
        WorldPoint,
        BothMove,
        BaseMove,
    )
    NAME: ClassVar[str] = "robocoliseum"
    #: The benchmark judges the final scene: no tool result carries success.
    ABSENT_MEMBERS: ClassVar[frozenset[str]] = frozenset({"done", "world_map"})
    CAMERAS: ClassVar[tuple[str, ...]] = ("head", "hand_left", "hand_right")
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("head", "high"): "head.png",
        ("hand_left", "high"): "hand_left.png",
        ("hand_right", "high"): "hand_right.png",
    }
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {}
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = ()
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = frozenset(
        {"move_to", "move_both", "move_base", "move_waist", "set_gripper", "home", "wait_for_check"}
    )
    VLA_PRIMITIVES: ClassVar[tuple[str, ...]] = ()
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        "reached",
        "dist_to_target_m",
        "pointing_error_deg",
        "final_tool_centre",
        "steps_used",
        "steps_planned",
        "holding",
        "opening",
        "found",
        "centre_xyz",
        "top_z",
        "tilt_deg",
        "grasp_yaw",
        "width_m",
        "note",
        "path",
        "reach_hint",
        "stop_reason",
        "alternatives",
        "rotation_error_deg",
        "completed",
        "waypoints_passed",
        "success",
        "truncated",
    )

    # -- state ---------------------------------------------------------------------

    def _state_from_envelope(self, envelope: dict[str, Any]) -> State:
        summary = envelope.get("state") or {}
        raw = summary.get("state") or {}
        stamp = raw.get("arm_state") or {}
        torso = raw.get("torso") or {}
        return State(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or raw.get("instruction") or ""),
            left=_arm_state(stamp.get("left") or {}),
            right=_arm_state(stamp.get("right") or {}),
            torso_heading_deg=float(torso.get("heading_deg") or 0.0),
            torso_forward=tuple(torso.get("forward") or (1.0, 0.0, 0.0)),
        )

    def _absorb(self, envelope: dict[str, Any] | None) -> None:
        """Record the latest state; nothing an envelope says ends the episode."""

        if envelope is None:
            return
        self._last_state = self._state_from_envelope(envelope)

    def _summary_dict(self) -> dict[str, Any]:
        state = self._current_state()
        out: dict[str, Any] = {"step": state.step}
        for name in self.ARMS:
            arm = state.arm(name)
            out[f"{name}_tool_centre"] = [round(v, 4) for v in arm.tool_centre]
            out[f"{name}_pointing"] = [round(v, 2) for v in arm.pointing]
            out[f"{name}_gripper"] = round(arm.opening, 3)
        return out

    def state(self, step: int = -1) -> State:
        """Read the recorded state of both arms.

        Args:
            step: recorded step to read; ``-1`` is the latest, ``0`` the initial scene.
        Returns:
            State with both arms' tool centres, pointing directions and gripper openings.
        """
        return super().state(step)

    def image(self, camera: str = "head", *, resolution: str = "high", step: int = -1) -> Any:
        """Load a camera image as a ``numpy`` uint8 array (H, W, 3), RGB.

        Args:
            camera: ``"head"`` (the whole scene), ``"hand_left"`` or ``"hand_right"`` (on
                the gripper). Pixel coordinates are (row, col), row 0 at the top.
            resolution: there is one resolution per camera, ``"high"``.
            step: recorded step, ``-1`` = latest.
        """
        return super().image(camera, resolution=resolution, step=step)

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return the raw bytes of a recorded artifact (advanced).

        Names are ``head.png``, ``hand_left.png`` and ``hand_right.png``.
        """
        return super().artifact(name, step=step)

    # -- motion --------------------------------------------------------------------

    def move_to(
        self,
        xyz: Sequence[float] | None = None,
        arm: str = "right",
        gripper: float | None = None,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
        path: str = "auto",
        max_actions: int = MOVE_MAX_ACTIONS,
    ) -> Move:
        """Move one arm's grasp point to a position in ``robo.FRAME``, with an orientation.

        The arm's inverse kinematics is solved for the whole pose first (the same solver
        the benchmark uses), and the arm follows a planned path commanded as joint angles.
        Nothing moves when the pose is out of reach: ``stop_reason == "out_of_reach"``,
        ``steps_used == 0``, and ``alternatives`` names nearby poses that ARE reachable.
        A pose missed by at most 2 cm / 8 degrees is taken anyway: the arm goes to the
        closest one and ``note`` says how far off it is.

        Args:
            xyz: target grasp point [x, y, z], metres; None keeps the position.
            arm: ``"left"`` or ``"right"``.  The other arm holds its joints.
            gripper: ``robo.CLOSE`` / ``robo.OPEN`` to command the moving arm's gripper on
                every action, ``robo.KEEP`` (None) to keep commanding what it was last told
                (a held object stays squeezed).
            pitch / yaw / roll: the orientation to arrive with, degrees (the class
                docstring defines them).  An angle left None keeps its current value, so
                a plain position move does not turn the gripper.
            path: ``"auto"`` (a straight line of the grasp point where it can be followed,
                else a joint swing), ``"line"`` (the line or nothing: every approach,
                descent and lift near objects) or ``"joint"`` (swing the joints: fewer
                actions, the grasp point does not follow a line — away from objects only).
            max_actions: native actions this call may spend.
        Returns:
            Move; test ``reached`` (within 0.01 m and 5 degrees).
        Raises:
            ValueError: a malformed argument (nothing is sent).
        """
        arm = _arm(arm)
        target = (
            None if xyz is None else [float(v) for v in np.asarray(xyz, dtype=float).reshape(-1)]
        )
        if target is not None and len(target) != 3:
            raise ValueError("xyz must be three numbers")
        if path not in ("auto", "line", "joint"):
            raise ValueError(f"path must be 'auto', 'line' or 'joint', got {path!r}")
        kwargs: dict[str, Any] = {
            "arm": arm,
            "xyz": target,
            "grip": gripper,
            "path": path,
            "max_actions": int(max_actions),
        }
        for name, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
            if value is not None:
                kwargs[name] = float(value)
        return self._stateful("move_to", kwargs, lambda p: self._move(arm, p))

    def _move(self, arm: str, p: dict[str, Any]) -> Move:
        return Move(
            arm=str(p.get("arm", arm)),
            target_tool_centre=tuple(p.get("target_tool_centre") or (0.0, 0.0, 0.0)),
            final_tool_centre=tuple(p.get("final_tool_centre") or (0.0, 0.0, 0.0)),
            dist_to_target_m=float(p.get("dist_to_target_m", math.nan)),
            pointing_error_deg=float(p.get("pointing_error_deg", math.nan)),
            reached=bool(p.get("reached")),
            steps_used=int(p.get("steps_used", 0)),
            steps_planned=int(p.get("steps_planned", 0)),
            truncated=bool(p.get("truncated")),
            reach_hint=p.get("reach_hint"),
            rotation_error_deg=float(p.get("rotation_error_deg", math.nan)),
            stop_reason=_stop_reason(p, bool(p.get("reached"))),
            blocked=bool(p.get("blocked")),
            alternatives=tuple(p.get("alternatives") or ()),
            angles=p.get("angles"),
            path=p.get("path"),
            note=p.get("note"),
        )

    def set_gripper(
        self, arm: str = "right", command: float | None = None, wait_for_check: bool = True
    ) -> Gripper:
        """Close or open one gripper, without moving the arms.

        The command is repeated for a few actions until the fingers stop.  After a close,
        ``holding`` tells whether something stopped them apart.  A close that holds something
        then keeps the robot still for up to 30 more actions, until the benchmark's progress
        check has seen the grasp: a lift only counts when it starts after that check.  Later
        motions with ``robo.KEEP`` keep commanding what was commanded last, so a held object stays
        squeezed.

        Args:
            arm: ``"left"`` or ``"right"``.
            command: ``robo.CLOSE`` or ``robo.OPEN``.  ``robo.KEEP`` is refused: this
                call exists to change the gripper.
            wait_for_check: after a close that holds something, keep still until the
                progress check has seen it (up to 30 actions).  Keep True when the next
                stage is graded on LIFTING the object (a pick-up).  False skips the wait:
                for a re-grasp, or a grasp whose next stage is not judged on the lift.
        Returns:
            Gripper.  ``holding`` is not proof that the INTENDED object is held: lift,
            then look at the wrist camera or ``locate`` it again.
        Raises:
            ValueError: a malformed argument (nothing is sent).
        """
        arm = _arm(arm)
        if command is None:
            raise ValueError("command must be robo.CLOSE or robo.OPEN")
        # The library's commands are CLOSE 0.0 and OPEN -0.785; anything at or above 0
        # closes, anything below opens.
        close = float(command) >= 0.0
        return self._stateful(
            "set_gripper",
            {"arm": arm, "close": bool(close), "wait_for_check": bool(wait_for_check)},
            lambda p: self._gripped(arm, p),
        )

    def _gripped(self, arm: str, p: dict[str, Any]) -> Gripper:
        return Gripper(
            arm=str(p.get("arm", arm)),
            commanded=str(p.get("commanded", "")),
            opening=round(_normalised(p.get("opening") or 0.0), 3),
            holding=bool(p.get("holding")),
            steps_used=int(p.get("steps_used", 0)),
        )

    def move_both(
        self,
        left: Sequence[float] | None = None,
        right: Sequence[float] | None = None,
        max_actions: int = MOVE_MAX_ACTIONS,
    ) -> BothMove:
        """Move both grasp points at the same time — to carry one object with two hands.

        Each tool keeps its orientation and each gripper its last command.  The two hands
        cover the same fraction of their remaining way on every action, so they start and
        arrive together and keep their offset in between (move them by the same vector to
        carry something rigid).  None holds that arm.

        Args:
            left / right: target grasp point [x, y, z] in ``robo.FRAME``, or None.
            max_actions: native actions this call may spend.
        Returns:
            BothMove; ``reached`` when every moved arm arrived (1.5 cm).
        Raises:
            ValueError: a malformed argument (nothing is sent).
        """

        def target(v: Sequence[float] | None) -> list[float] | None:
            if v is None:
                return None
            out = [float(x) for x in np.asarray(v, dtype=float).reshape(-1)]
            if len(out) != 3:
                raise ValueError("a target must be three numbers")
            return out

        goals = {"left": target(left), "right": target(right)}
        if goals["left"] is None and goals["right"] is None:
            raise ValueError("give left, right or both")

        def build(p: dict[str, Any]) -> BothMove:
            return BothMove(
                reached=bool(p.get("reached")),
                left=p.get("left"),
                right=p.get("right"),
                steps_used=int(p.get("steps_used", 0)),
            )

        return self._stateful("move_both", {**goals, "max_actions": int(max_actions)}, build)

    def move_base(self, forward: float = 0.0, left: float = 0.0, turn_deg: float = 0.0) -> BaseMove:
        """Drive the wheeled base — only in tasks where the base is free (the ``agent_*``
        scenes); elsewhere it is fixed to the floor and nothing moves.

        ``forward`` / ``left`` are metres along the base's own axes (``robo.FRAME`` x / y),
        ``turn_deg`` counter-clockwise.  At most 2 m and 180 degrees per call.  The arms keep
        their joints.  ``robo.FRAME`` travels with the base, so arm poses stay valid, but
        every object position measured before the move is stale: look again.

        Returns:
            BaseMove.  Open loop — the benchmark reports no base pose.
        """
        values = {"forward": float(forward), "left": float(left), "turn_deg": float(turn_deg)}

        def build(p: dict[str, Any]) -> BaseMove:
            return BaseMove(
                commanded=dict(p.get("commanded") or {}), steps_used=int(p.get("steps_used", 0))
            )

        return self._stateful("move_base", values, build)

    def move_waist(
        self,
        forward: float = 0.0,
        up: float = 0.0,
        turn_deg: float = 0.0,
        lean_deg: float = 0.0,
        reset: bool = False,
        max_actions: int = 60,
    ) -> Waist:
        """Move the upper body: the chest forward/back and up/down, or turn it.

        The G2 stands on a knee-hip-waist chain.  This moves the chest by ``forward`` and
        ``up`` metres in ``robo.FRAME`` while keeping the torso's lean (``lean_deg`` adds to
        it), and turns the upper body by ``turn_deg`` (+ to the robot's left).  The arms keep
        their joint angles, so BOTH hands — and anything held — move along with the chest.
        Keep the waist still by default and reach with the arms: move it only when an arm
        cannot reach (``move_to`` returned ``reach_hint``), because every hand position and
        every earlier measurement relative to the hands changes, and a waist move costs 10-30
        actions.  ``forward`` and ``up`` move the chest along robo.FRAME +x and +z — the plane
        the legs bend in — whichever way the torso faces; when the torso is turned (see
        ``state().torso_forward``), ``forward`` moves the chest sideways relative to it.
        ``reset=True`` returns to the starting waist.

        Args:
            forward / up: chest displacement, metres (the chest can come forward about
                0.3 m; going straight down is limited, forward-and-down goes further).
            turn_deg: upper-body turn, degrees.
            lean_deg: extra forward lean of the torso, degrees (tilts both arms with it).
            reset: back to the waist angles the episode started with.
            max_actions: native actions this call may spend (a move takes 10-30).
        Returns:
            Waist; when ``moved`` is False the pose is out of range and nothing moved.
            Re-read ``state()`` afterwards: every hand position has changed.
        """
        kwargs: dict[str, Any] = {
            "forward": float(forward),
            "up": float(up),
            "turn_deg": float(turn_deg),
            "lean_deg": float(lean_deg),
            "reset": bool(reset),
            "max_actions": int(max_actions),
        }

        def build(p: dict[str, Any]) -> Waist:
            moved = p.get("chest_moved_m") or (0.0, 0.0)
            return Waist(
                moved=bool(p.get("moved")),
                reached=bool(p.get("reached")),
                chest_moved_m=(float(moved[0]), float(moved[1])),
                torso_pitch_deg=p.get("torso_pitch_deg"),
                turn_deg=p.get("turn_deg"),
                error_reason=p.get("error_reason"),
                steps_used=int(p.get("steps_used", 0)),
            )

        return self._stateful("move_waist", kwargs, build)

    def wait_for_check(self, checks: int = 1) -> int:
        """Hold both arms and grippers still until the benchmark's stage check has run.

        The benchmark looks at the scene only once every 30 native actions, and a stage
        passes only on what it sees at such a check (some stages need two checks in a row).
        A state that is reached and then left between two checks — an object lifted and put
        down again — is never seen.  Call this right after reaching what a stage asks, before
        the next motion changes it: it spends at most 30 actions per check.

        Args:
            checks: how many checks to wait for (1-5); 2 for a stage that needs the state
                on two checks in a row (putting inside, stacking, beside).
        Returns:
            the native actions spent.
        """
        if not isinstance(checks, int) or not 1 <= checks <= 5:
            raise ValueError("checks must be an integer from 1 to 5")
        p = self._stateful("wait_for_check", {"checks": int(checks)}, lambda p: p)
        return int(p.get("steps_used", 0)) if isinstance(p, dict) else 0

    def home(self, arm: str = "both", max_actions: int = HOME_MAX_ACTIONS) -> Home:
        """Return one arm or both to the joint angles the episode started with.

        Args:
            arm: ``"left"``, ``"right"`` or ``"both"``.
            max_actions: native actions this call may spend.
        Returns:
            Home; ``reached`` says whether every joint came within 0.02 rad of its start
            angle, ``max_joint_error_rad`` what remains.
        Raises:
            ValueError: a malformed argument (nothing is sent).
        """
        if arm not in (*self.ARMS, "both"):
            raise ValueError("arm must be 'left', 'right' or 'both'")
        return self._stateful("home", {"arm": arm, "max_actions": int(max_actions)}, self._homed)

    def _homed(self, p: dict[str, Any]) -> Home:
        return Home(
            arm=str(p.get("arm", "both")),
            reached=bool(p.get("reached")),
            max_joint_error_rad=float(p.get("max_joint_error_rad", math.nan)),
            steps_used=int(p.get("steps_used", 0)),
        )

    # -- perception ----------------------------------------------------------------

    def segment(
        self,
        camera: str = "head",
        *,
        row: int | None = None,
        col: int | None = None,
        text: str | None = None,
        min_score: float = 0.2,
    ) -> Segment:
        """Find one object's region in a camera image: by name, or by a pixel inside it.

        Args:
            camera: ``"head"``, ``"hand_left"`` or ``"hand_right"``.
            text: what to find, in a few words ("the black pen", "orange juice bottle"):
                SAM3 segments it, and its best mask scoring at least ``min_score`` is
                returned.  Name one object; a colour or position in the words helps.
            row / col: a pixel INSIDE the region to find; both None (and no text) takes
                the largest colour region in the image.
        Returns:
            Segment; when ``found``, pass ``centroid`` to ``back_project`` with the same
            camera.
        Raises:
            ValueError: an unknown camera, or only one of row / col.
        """
        camera = sight.camera_name(camera, self.CAMERAS)
        if (row is None) != (col is None):
            raise ValueError("give both row and col, or neither")
        if text is not None and not str(text).strip():
            raise ValueError("text must name something")
        payload = self._readonly(
            "segment",
            {"camera": camera, "row": row, "col": col, "text": text, "min_score": float(min_score)},
        )
        centroid = payload.get("centroid_rc")
        box = payload.get("box_rc")
        return Segment(
            found=bool(payload.get("found")),
            centroid=(int(centroid[0]), int(centroid[1])) if centroid else None,
            area_px=int(payload.get("area_px") or 0),
            colour=payload.get("colour"),
            box=tuple(int(v) for v in box) if box else None,
            camera=str(payload.get("camera", camera)),
            method=payload.get("method"),
            score=payload.get("score"),
        )

    def back_project(self, row: int, col: int, camera: str = "head") -> WorldPoint:
        """Return the point seen at one pixel of the current image of ``camera``.

        Args:
            row / col: pixel, row 0 at the top, col 0 at the left.
            camera: the camera the pixel was read from (default ``"head"``).
        Returns:
            WorldPoint; ``xyz`` is where the camera ray meets the first surface.  A pixel
            on an edge or a thin part lands behind the object: prefer a region's centroid.
        Raises:
            ToolError: pixel out of range, no valid depth there, or no calibration.
        """
        camera = sight.camera_name(camera, self.CAMERAS)
        payload = self._readonly(
            "back_project", {"row": int(row), "col": int(col), "camera": camera}
        )
        return WorldPoint(
            pixel=(int(row), int(col)),
            xyz=tuple(float(v) for v in payload.get("xyz") or (0.0, 0.0, 0.0)),
            depth_m=float(payload.get("depth_m", math.nan)),
            camera=str(payload.get("camera", camera)),
        )

    def locate(
        self,
        target: str | int | None = None,
        col: int | None = None,
        camera: str = "head",
        *,
        row: int | None = None,
        min_score: float = 0.2,
        near: tuple[int, int] | None = None,
    ) -> ObjectPose:
        """Measure one object in 3D: name it, or point at a pixel inside it.

        ``robo.locate("the black pen")`` — SAM3 finds what the words describe in the image
        and the depth image measures it.  ``robo.locate(row, col)`` — the object at that
        pixel, cut out by its shape above the surface it stands on (then SAM3's mask at the
        pixel, then the pixel's colour region).  Naming is the way for thin, small, or
        same-coloured objects and for objects inside containers; check ``method`` and
        ``score`` and the returned ``pixel_uv`` against the image.

        Args:
            target: what to find, in a few words (a string), or the pixel's row (an int,
                with ``col``).  None with no pixel takes the largest colour region.
            col: the pixel's column, with an int ``target`` (or ``row=``).
            camera: ``"head"`` (default) or a wrist camera; the head camera sees the
                whole scene, a wrist camera is close up.
            min_score: the lowest SAM3 confidence accepted for a named object.
            near: (row, col) — with a name, measure the instance of it at (or nearest)
                that pixel rather than the best-scoring one.  SAM3 ranks by what a thing
                is, not where: "the juice in the basket" can score a bottle on the shelf
                highest.  Read ``candidates``, or the image, and pass the one you mean.
        Returns:
            ObjectPose.  To grasp: hop to ``[x, y, top_z + 0.06]`` with
            ``gripper=robo.OPEN`` (keeping the orientation), descend to
            ``top_z - 0.02``, then ``set_gripper(arm, robo.CLOSE)``.
        Raises:
            ValueError: an unknown camera, or only one of row / col.
        """
        camera = sight.camera_name(camera, self.CAMERAS)
        text = None
        if isinstance(target, str):
            if not target.strip():
                raise ValueError("text must name something")
            text = target
        elif target is not None:
            if row is not None:
                raise ValueError("give the row once")
            row = int(target)
        if (row is None) != (col is None):
            raise ValueError("give both row and col, or neither")
        payload = self._readonly(
            "locate",
            {
                "camera": camera,
                "row": row,
                "col": col,
                "text": text,
                "min_score": float(min_score),
                "near": None if near is None else [int(near[0]), int(near[1])],
            },
        )
        centre = payload.get("centre_xyz")
        size = payload.get("size_xyz")
        uv = payload.get("pixel_uv")
        return ObjectPose(
            found=bool(payload.get("found")),
            centre_xyz=tuple(round(float(v), 4) for v in centre) if centre else None,
            top_z=float(payload["top_z"]) if payload.get("top_z") is not None else None,
            height_m=float(payload["height_m"]) if payload.get("height_m") is not None else None,
            size_xyz=tuple(round(float(v), 4) for v in size) if size else None,
            pixel_uv=(int(uv[0]), int(uv[1])) if uv else None,
            area_px=int(payload.get("area_px") or 0),
            colour=payload.get("colour"),
            camera=str(payload.get("camera", camera)),
            note=payload.get("note"),
            tilt_deg=payload.get("tilt_deg"),
            long_axis_deg=payload.get("long_axis_deg"),
            support_z=payload.get("support_z"),
            method=payload.get("method"),
            score=payload.get("score"),
            candidates=tuple(payload.get("candidates") or ()),
            grasp_fingers=tuple(payload["grasp_fingers"]) if payload.get("grasp_fingers") else None,
            grasp_yaw=payload.get("grasp_yaw"),
            width_m=payload.get("width_m"),
            length_m=payload.get("length_m"),
        )


ADAPTER = RobotAdapter(
    name="robocoliseum",
    robot_cls=RobocoliseumRobot,
    host="pyrualean.hosts.geniesim_robocoliseum",
    knowledge="robocoliseum",
    blurb="a dual-arm G2 humanoid with Omnipicker grippers in the GenieSim benchmark",
    guides_subdir="",
    guide_names=(),
    show_examples="`robo.image('head')`, `robo.image('hand_left')` or `robo.image('hand_right')`",
    services="",
    services_novla="",
)


__all__ = [
    "ADAPTER",
    "ArmState",
    "BaseMove",
    "BothMove",
    "RobocoliseumRobot",
    "Gripper",
    "HOME_MAX_ACTIONS",
    "Home",
    "MOVE_MAX_ACTIONS",
    "MOVE_TOL_DEG",
    "MOVE_TOL_M",
    "Move",
    "ObjectPose",
    "Segment",
    "State",
    "Waist",
    "WorldPoint",
]
