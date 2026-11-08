# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LIBERO-PRO's Franka with the primitives of ``robodojo_xpl_rgb``: colour images only.

``pyrualean.libero`` drives LIBERO through RPent's own tools (OSC servos by position, separate
wrist turns, perception through the depth image and per-pixel world maps).  The robot here is
the same arm with the primitive set of ``robodojo_xpl_rgb``, for one arm: ``move_to`` a grasp
point with an orientation in degrees, ``set_gripper`` a full stroke that reports what it holds,
and perception from the two colour cameras alone (``locate``
measures an object from its masks in the cameras that see it, :mod:`pyrualean.hosts.libero_sight`).
No depth image, world map or object pose reaches the policy; the host reads one depth frame at
boot to find the table's height.

The tools are this library's own (:mod:`pyrualean.hosts.libero_tools`, registered on RPent's
LIBERO toolkit by :mod:`pyrualean.hosts.rpent_libero_rgb`).  The frozen Pi0.5 skills of the
``libero`` robot stay available under ``--primitives full``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ._robot import RobotBase
from .libero import Contact, Pick
from .robots import RobotAdapter

#: The arm's motion reaches the target when within these (metres, degrees).
MOVE_TOL_M = 0.01
MOVE_TOL_DEG = 5.0
#: A fully open Panda gripper: the two finger joints sum to this (metres).
OPEN_SPAN_M = 0.08


def _quat_matrix(q: Sequence[float]) -> list[list[float]]:
    x, y, z, w = (float(v) for v in q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _angles(q: Sequence[float]) -> tuple[float, float, float]:
    import numpy as np

    from .hosts.libero_tools import angles

    return angles(np.asarray(_quat_matrix(q)))


# ---------------------------------------------------------------------------
# Result types (returned to the policy; all fields are plain Python values)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """The robot at one recorded step.

    step: index of the recorded step (0 = initial scene).  A new step is
        recorded after every motion call.
    task: the task instruction given by the benchmark (authoritative).
    tcp_pos: the grasp point [x, y, z] in the world frame, metres: between the
        finger pads.  ``move_to`` drives this point.
    pitch / yaw / roll: the gripper's orientation, degrees (see the class
        docstring): ``pitch`` 0 points level and 90 straight down; pointing
        straight down, ``yaw`` alone says how the fingers lie (the heading of
        the line they close along) and ``roll`` reads 0.
    eef_quat: the hand's orientation as a unit quaternion [x, y, z, w].
    gripper: measured gripper opening, normalised: 1.0 = open (8 cm between
        the fingers), 0.0 = closed on nothing.
        (Gripper COMMANDS use the opposite sign convention: see ``set_gripper``.)
    object_names: names of the movable objects in the scene (no poses).
    terminated: the benchmark's success predicate has fired.
    truncated: the simulator step budget is exhausted.
    """

    step: int
    task: str
    tcp_pos: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    eef_quat: tuple[float, float, float, float]
    gripper: float
    object_names: tuple[str, ...]
    terminated: bool
    truncated: bool


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
    note: set when the pose turned half a turn about the pointing direction
        was taken (the empty fingers are interchangeable).
    blocked: the arm stopped short of the target: something is in the way, a
        held object is pushing against something, or the pose is out of the
        arm's reach.
    steps_used: simulator steps consumed.
    terminated: the task predicate fired during the motion.
    truncated: the step budget ran out.
    joint_limit: set when the motion ended with a joint of the arm at its
        limit (which joint, and what to do): the arm is wound up that way and
        will not unwind by itself; turn back the way it came or turn less.
    """

    target_xyz: tuple[float, float, float]
    start_xyz: tuple[float, float, float]
    final_xyz: tuple[float, float, float]
    dist_to_target_m: float
    orientation_error_deg: float
    pitch: float
    yaw: float
    roll: float
    reached: bool
    note: str | None
    blocked: bool
    steps_used: int
    terminated: bool
    truncated: bool
    joint_limit: str | None = None


@dataclass(frozen=True)
class Gripper:
    """Outcome of ``set_gripper``.

    commanded: ``"closed"`` or ``"open"``.
    opening: the opening once the fingers stopped (1.0 = open, 0.0 = closed on
        nothing).
    holding: after a close, the fingers stopped apart: something is between
        them.  It does not say what: check that the intended object moves with
        the arm.
    width_m: after a close, the distance between the fingers (about the width
        of what is held; 0.0 when nothing is), else None.
    steps_used: simulator steps consumed.
    terminated / truncated: as for ``Move``.
    note: set after a close whose fingers stopped almost fully open: something
        about as wide as the open gripper is between them (an object may be
        wedged there although ``holding`` is False).
    """

    commanded: str
    opening: float
    holding: bool
    width_m: float | None
    steps_used: int
    terminated: bool
    truncated: bool
    note: str | None = None


@dataclass(frozen=True)
class Segment:
    """Outcome of ``segment``.

    found: SAM3 produced a mask scoring at least ``min_score``.
    centroid: (row, col) of the mask in the 1024 x 1024 image, or None.
    area_px: mask pixels (at that resolution).
    score: SAM3 confidence.
    camera: the camera segmented.
    """

    found: bool
    centroid: tuple[int, int] | None
    area_px: int
    score: float | None
    camera: str


@dataclass(frozen=True)
class SeenPoint:
    """Outcome of ``back_project``: the point seen at one pixel.

    pixel: the (row, col) asked about.
    xyz: world [x, y, z], metres: where the pixel's ray meets the object under
        the pixel, or the table.
    depth_m: distance of that point along the camera's optical axis.
    camera: the camera the pixel was read from.
    surface: ``"object"`` when the pixel lies on something that could be
        measured, ``"table"`` when the point is on the table top: the pixel
        shows the table, or something too flat or too far to measure.
    """

    pixel: tuple[int, int]
    xyz: tuple[float, float, float] | None
    depth_m: float
    camera: str
    surface: str


@dataclass(frozen=True)
class ObjectSight:
    """Outcome of ``locate``: an object measured from its masks in the colour cameras.

    found: SAM3 produced a mask scoring at least ``min_score``.
    center_xyz: [x, y, z] of the middle of the smallest rectangle around the
        object's footprint, at half its height above the table, metres.  A
        handle or a spout is part of the footprint and moves the middle away
        from the body's.  None when no 3D measurement was possible (see
        ``note``).
    top_z: height of the object's top surface.
    height_m: ``top_z`` above the table surface.
    length_m / width_m: the long and the short side of the smallest rectangle
        around the footprint.
    long_axis_deg: direction of the long side on the table, degrees from +x
        (-90 .. 90); of a square, one of its sides.
    grasp_yaw: the ``yaw`` for ``move_to(..., pitch=90, yaw=grasp_yaw)``:
        pointing straight down, the fingers then close across the short side.
    views: how many cameras agreed on the measurement.  1: nothing but the
        camera asked saw the object, and its height is a guess (see
        ``locate``); ``note`` says which guess.
    score: SAM3 confidence.
    centroid: (row, col) of the mask in the 1024 x 1024 image.
    area_px: mask pixels (at that resolution).
    camera: the camera measured from.
    note: what limits the measurement (one camera only; two that agree only
        roughly; the mask is the gripper's own fingers; too flat or too far to
        measure), else None.

    The object is taken to stand on the table.  What other things or the
    gripper hide of it is not measured, so a partly hidden object comes out
    smaller and off-centre.  A held object is not measured correctly.
    """

    found: bool
    center_xyz: tuple[float, float, float] | None
    top_z: float | None
    height_m: float | None
    length_m: float | None
    width_m: float | None
    long_axis_deg: float | None
    grasp_yaw: float | None
    views: int
    score: float | None
    centroid: tuple[int, int] | None
    area_px: int
    camera: str
    note: str | None


STATEFUL_TOOLS = frozenset({"rgb_move_pose", "rgb_grip", "pi0_pick", "pi0_doubled"})


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
    return vals if len(vals) == 3 else None  # type: ignore[return-value]


def _angle(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number of degrees, got {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _command(gripper: Any, *, allow_keep: bool) -> int:
    try:
        g = int(gripper)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"gripper must be robo.CLOSE (1) or robo.OPEN (-1), got {gripper!r}"
        ) from exc
    allowed = (1, -1, 0) if allow_keep else (1, -1)
    if g != gripper or g not in allowed:
        names = (
            "robo.CLOSE (1), robo.OPEN (-1) or robo.KEEP (0)"
            if allow_keep
            else "robo.CLOSE (1) or robo.OPEN (-1)"
        )
        raise ValueError(f"gripper must be {names}, got {gripper!r}")
    return g


def _camera(camera: str, cameras: tuple[str, ...]) -> str:
    if camera not in cameras:
        raise ValueError(f"camera must be one of {cameras}, got {camera!r}")
    return camera


class LiberoColourRobot(RobotBase):
    """A LIBERO-PRO tabletop Franka arm with a two-finger gripper, seen through colour cameras.

    World frame: x and y span the table, z is up; metres.  The robot stands
    at negative x and reaches towards +x.  The table's height differs from
    scene to scene: read the starting state (``state(0)``) and what
    ``locate`` measures rather than assume one.

    ``move_to`` drives the GRASP POINT, between the finger pads
    (``state().tcp_pos``), to a world position, with an orientation given in
    degrees: ``pitch`` is how far the gripper points below the horizontal
    (0 level, 90 straight down, as at the start), ``yaw`` turns it about the
    vertical and ``roll`` turns the fingers about the pointing direction.
    Pointing straight down, the fingers close along the line at ``yaw``
    degrees from +x (counter-clockwise seen from above): ``locate``'s
    ``grasp_yaw`` is that line across an object's narrow side.  Tilted, ``yaw``
    is the heading the gripper points towards on the table (0: towards +x),
    and ``roll`` 0 keeps the fingers closing along that heading, one above the
    other; ``roll`` 90 turns them to close across it, side by side, as at the
    start (which reads roll -90).

    Two gripper conventions, do not mix them up.  COMMANDS are signed:
    ``robo.CLOSE`` (1) closes, ``robo.OPEN`` (-1) opens, ``robo.KEEP`` (0)
    keeps commanding what was commanded last.  MEASUREMENTS
    (``state().gripper``, ``Gripper.opening``) are openings: 1.0 = open,
    0.0 = closed on nothing.

    Every simulator step (one step of ``move_to`` or ``set_gripper``, one
    VLA action) counts against the episode's step budget; when
    it is spent the state reports ``truncated`` and motion calls raise
    ``EpisodeFinished``.  The benchmark's success check runs after every
    step: ``robo.done`` turns True the moment the task is solved, and every
    further motion is refused.  Motion calls block until the primitive
    finishes and return a typed result; nothing is queued.

    Perception is from colour images alone; there is no depth image.  A fixed
    camera over the table (``agentview``, the whole scene) and one on the
    gripper (``wrist``, close range), 1024 x 1024 each, and where each one is,
    is known.  ``locate`` finds an object by a text prompt and measures it in
    3D (centre, top, footprint, heading) from the cameras that see it: it
    takes both to measure a height.  ``segment`` / ``back_project`` give a
    mask centroid and the world point at one pixel.  Object poses are never
    given.

    The frozen Pi0.5 VLA (``pi0_pick``, ``pi0_doubled``) is a learned policy
    fine-tuned on this benchmark's demonstrations.
    """

    CLOSE: int = 1
    OPEN: int = -1
    KEEP: int = 0

    #: Rendered as the result-type section of the API reference (not a constant).
    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        Move,
        Gripper,
        ObjectSight,
        Segment,
        SeenPoint,
        Pick,
        Contact,
    )
    NAME: ClassVar[str] = "libero_rgb"
    #: No world maps for the policy: there is no depth image.
    ABSENT_MEMBERS: ClassVar[frozenset[str]] = frozenset({"world_map"})
    CAMERAS: ClassVar[tuple[str, ...]] = ("agentview", "wrist")
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("agentview", "high"): "agentview_high.png",
        ("wrist", "high"): "wrist_high.png",
    }
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {}
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = ("agentview_high.png", "wrist_high.png")
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = STATEFUL_TOOLS
    #: The frozen Pi0.5 skills (hidden and refused under the no-VLA primitive set).
    VLA_PRIMITIVES: ClassVar[tuple[str, ...]] = ("pi0_pick", "pi0_doubled")
    #: VLA sentences of the remaining docstrings, removed from the no-VLA reference.
    VLA_REFERENCE_EDITS: ClassVar[tuple[tuple[str, str], ...]] = (
        (
            "    Every simulator step (one step of ``move_to`` or ``set_gripper``, one\n"
            "    VLA action) counts against the episode's step budget; when\n",
            "    Every simulator step (one step of ``move_to`` or ``set_gripper``)\n"
            "    counts against the episode's step budget; when\n",
        ),
        (
            "\n\n    The frozen Pi0.5 VLA (``pi0_pick``, ``pi0_doubled``) is a learned policy\n"
            "    fine-tuned on this benchmark's demonstrations.",
            "",
        ),
    )
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        "success",
        "reached",
        "dist_to_target_m",
        "orientation_error_deg",
        "final_xyz",
        "blocked",
        "steps_used",
        "dist_m",
        "joint_limit",
        "note",
        "holding",
        "opening",
        "width_m",
        "chunks_used",
        "found",
        "score",
        "centroid_rc",
        "center_xyz",
        "top_z",
        "grasp_yaw",
        "views",
        "world_xyz",
        "surface",
        "terminated",
        "truncated",
        "step",
    )
    #: Recorded artifacts the policy may read: the colour images only.
    _READABLE: ClassVar[frozenset[str]] = frozenset(
        {"agentview_high.png", "wrist_high.png", "agentview.png", "wrist.png"}
    )

    # -- observation --------------------------------------------------------

    @property
    def task(self) -> str:
        """The task instruction, e.g. ``"put the bowl on the plate"`` (authoritative)."""
        return super().task

    @property
    def done(self) -> bool:
        """True once the benchmark's success predicate has fired (sticky)."""
        return super().done

    def state(self, step: int = -1) -> State:
        """Read the recorded state of the robot.

        Args:
            step: recorded step to read; ``-1`` is the latest, ``0`` the
                initial scene.  A new step is recorded after every motion call.
        Returns:
            State with the grasp point, the gripper's orientation and
            measured opening, the object names and the episode flags.
        """
        return super().state(step)

    def image(self, camera: str = "agentview", *, resolution: str = "high", step: int = -1) -> Any:
        """Load a recorded camera image as a ``numpy`` uint8 array (1024, 1024, 3), RGB.

        Args:
            camera: ``"agentview"`` (fixed view of the whole scene: identify
                objects, destinations and the layout) or ``"wrist"`` (on the
                gripper: what it is about to grasp or is holding).
            resolution: there is one resolution per camera, ``"high"``.  Pixel
                coordinates are (row, col) with row 0 at the top and col 0 at
                the left.
            step: recorded step, ``-1`` = latest.  ``segment``, ``locate`` and
                ``back_project`` always read the CURRENT scene, which is the
                latest recorded step.
        """
        return super().image(camera, resolution=resolution, step=step)

    def world_map(
        self, camera: str = "agentview", *, resolution: str = "high", step: int = -1
    ) -> Any:
        """Absent (not in the API reference): there is no depth image."""
        raise ValueError("there is no depth image: use back_project(row, col, camera) per pixel")

    def camera_meta(self, camera: str = "agentview", *, step: int = -1) -> dict[str, Any]:
        """Return the calibration of one camera at a recorded step as a dict.

        Keys: ``shape`` (image shape), ``intrinsic_matrix`` (3x3) and
        ``extrinsic_matrix`` (4x4 camera-to-world; OpenCV's convention: the
        camera looks down its own +z axis with +y down the image).  The
        agentview camera is fixed; the wrist camera moves with the gripper.
        """
        camera = _camera(camera, self.CAMERAS)
        name = "agentview" if camera == "agentview" else "robot0_eye_in_hand"
        payload = self._readonly("view_camera_meta", {"camera": name, "step": int(step)})
        meta = payload.get("camera_meta") or {}
        return {
            "shape": [1024, 1024, 3],
            "intrinsic_matrix": _scaled_k(meta),
            "extrinsic_matrix": meta.get("extrinsic_cam2world"),
        }

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

        Names are ``agentview_high.png`` and ``wrist_high.png``.
        """
        if name not in self._READABLE:
            raise ValueError(f"artifact must be one of {sorted(self._READABLE)}, got {name!r}")
        return super().artifact(name, step=step)

    # -- perception ---------------------------------------------------------

    def segment(
        self, text_prompt: str, camera: str = "agentview", min_score: float = 0.2
    ) -> Segment:
        """Segment an object in the current image of ``camera`` by a text prompt (SAM3).

        Args:
            text_prompt: what to find, e.g. ``"the red mug"``.  Name one object
                by its visible properties; brand names score badly.
            camera: ``"agentview"`` (default) or ``"wrist"``.
            min_score: masks scoring below this are reported as not found.
        Returns:
            Segment; when ``found``, pass ``centroid`` (row, col) to
            ``back_project`` with the same camera to get a world point.  The
            mask is the best match for the text, not a guarantee of identity:
            check where it is against the image.
        Raises:
            ToolError: the SAM3 service is unavailable.
        """
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        kwargs: dict[str, Any] = {
            "text_prompt": prompt,
            "camera": camera,
            "min_score": float(min_score),
        }
        p = self._readonly("rgb_segment", kwargs)
        centroid = p.get("centroid_rc") or ()
        return Segment(
            found=bool(p.get("found")),
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            score=float(p["score"]) if p.get("score") is not None else None,
            camera=str(p.get("camera", camera)),
        )

    def back_project(self, row: int, col: int, camera: str = "agentview") -> SeenPoint:
        """Return the world point seen at one pixel of the current image of ``camera``.

        There is no depth image, so a pixel alone is only a ray.  The object
        under the pixel is segmented from that pixel (SAM3) and measured as
        ``locate`` measures one, and the answer is where the pixel's ray
        meets it.  A pixel on the table top itself gives the point of the
        table it shows, exactly.

        Args:
            row / col: pixel, row 0 at the top, col 0 at the left (images are
                1024 x 1024).
            camera: the camera the pixel was read from (default ``"agentview"``).
        Returns:
            SeenPoint; check ``surface``.  ``"object"``: ``xyz`` is on the
            object under the pixel, as well as ``locate`` would measure that
            object.  ``"table"``: ``xyz`` is on the table top; a pixel that
            shows something too flat or too far to measure also answers so.
            A thin part in mid-air (a handle, a rim) is not measured
            reliably: reason from the image and from ``locate`` of the whole
            object instead.
        Raises:
            ToolError: pixel out of range, or it looks above the table's plane.
        """
        camera = _camera(camera, self.CAMERAS)
        kwargs: dict[str, Any] = {"row": int(row), "col": int(col), "camera": camera}
        p = self._readonly("rgb_back_project", kwargs)
        pixel = p.get("pixel") or (int(row), int(col))
        return SeenPoint(
            pixel=(int(pixel[0]), int(pixel[1])),
            xyz=_tuple3(p.get("world_xyz")),
            depth_m=float(p.get("depth_m", math.nan)),
            camera=str(p.get("camera", camera)),
            surface=str(p.get("surface") or "table"),
        )

    def locate(
        self, text_prompt: str, camera: str = "agentview", min_score: float = 0.2
    ) -> ObjectSight:
        """Find an object by a text prompt and measure it in 3D from the colour cameras.

        SAM3 segments the current image of ``camera`` (as ``segment``); the
        same object is then looked for in the other camera, and its height
        and footprint are those that fit both cameras that show it.  One
        camera cannot measure a height: a low deep thing and a tall shallow
        one look alike from it.  ``views`` says how many cameras agreed.
        Nothing moves.

        Args:
            text_prompt: one object by its visible properties, e.g. ``"the
                white bowl next to the plate"``.
            camera: ``"agentview"`` (default; the whole scene) or ``"wrist"``
                (close up, from the gripper).
            min_score: masks scoring below this are reported as not found.
        Returns:
            ObjectSight.  For a top grasp: ``move_to([x, y, top_z + 0.08],
            pitch=90)`` above it, ``locate`` it again from there (the wrist
            camera then looks down on it; when the text matches nothing in
            the wrist image the measurement is made from agentview's mask
            with the wrist image as second view, ``camera`` reading
            ``"agentview"``), turn to ``yaw=grasp_yaw`` and go
            down to the height to hold it at.  With ``views == 2`` and no
            ``note`` a box, a can or a bowl is measured to a few millimetres.
            With ``views == 1`` the height is a guess, the tallest upright
            shape that fits the one image: close for a box or a can, wrong
            for a bowl, a cup or anything wider above than below;
            ``center_xyz`` is then off, along the line away from the camera,
            by about half of what the height is wrong by.  An object once
            measured with ``views == 2`` keeps that height in later calls with
            the same prompt.  The mask is the best match for the text, not
            proof of identity: check it against the image.
        Raises:
            ToolError: the SAM3 service is unavailable.
        """
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        kwargs: dict[str, Any] = {
            "text_prompt": prompt,
            "camera": camera,
            "min_score": float(min_score),
        }
        p = self._readonly("rgb_locate", kwargs)
        centroid = p.get("centroid_rc") or ()

        def num(key: str) -> float | None:
            return float(p[key]) if p.get(key) is not None else None

        center = p.get("center_xyz")
        note = p.get("geometry_error") or p.get("height_guess")
        if p.get("measured_from"):
            via = f"measured from agentview: {p['measured_from']}"
            note = f"{note}; {via}" if note else via
        return ObjectSight(
            found=bool(p.get("found")),
            center_xyz=_tuple3(center) if center is not None else None,
            top_z=num("top_z"),
            height_m=num("height_m"),
            length_m=num("length_m"),
            width_m=num("width_m"),
            long_axis_deg=num("long_axis_deg"),
            grasp_yaw=num("grasp_yaw"),
            views=int(p.get("views") or 1),
            score=num("score"),
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            camera=str(p.get("camera", camera)),
            note=str(note) if note else None,
        )

    # -- the VLA ------------------------------------------------------------

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
        ``descent_thresh``, closed on something (finger separation in
        ``[gripper_open_thresh, gripper_closed_thresh)`` metres) and risen by
        ``lift_thresh``.  Pre-position the gripper above the target, keep the
        prompt short (``"pick up the red mug"``) and verify the grasp from the
        state and the wrist image; then carry and release with ``move_to`` and
        ``set_gripper``.

        Returns:
            Pick; ``success`` is a heuristic.
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

    # -- motion -------------------------------------------------------------

    def move_to(
        self,
        xyz: Sequence[float],
        gripper: int = 0,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
    ) -> Move:
        """Move the grasp point to a world position, with an orientation.

        The grasp point is between the finger pads (``tcp_pos``).  LIBERO's
        controller servos position and orientation together, at most 2 cm
        and about 5 degrees per simulator step, until the pose is reached or
        the arm stops making progress.  The path is close to a straight line
        but not planned around anything: rise above what is in the way first.

        Args:
            xyz: target grasp-point position [x, y, z], metres.
            gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1) to command the
                gripper on every step of the motion, ``robo.KEEP`` (0) to keep
                commanding what it was last told (a held object stays
                squeezed).  Keep it closed (1) while carrying an object.
            pitch / yaw / roll: the orientation to arrive with, degrees (the
                class docstring defines them).  An angle left None keeps its
                current value, so a plain position move does not turn the
                gripper.  With an EMPTY open gripper the fingers are
                interchangeable, so the pose turned half a turn about the
                pointing direction may be taken when it needs less turning
                (``note`` says so; yaw / roll then read 180 off); holding
                something, the pose is taken exactly as asked.
        Returns:
            Move; test ``reached`` (within 0.01 m and 5 degrees).  ``blocked``
            means the arm stopped short: something is in the way, or the pose
            is out of reach.
        Raises:
            ValueError: a malformed argument (nothing is sent).
        """
        target = _floats(xyz, 3, "xyz")
        command = _command(gripper, allow_keep=True)
        kwargs: dict[str, Any] = {"xyz": list(target), "gripper": command}
        for name, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
            angle = _angle(value, name)
            if angle is not None:
                kwargs[name] = angle
        return self._stateful("rgb_move_pose", kwargs, lambda p: self._move(target, p))

    def set_gripper(self, gripper: int, opening: float | None = None) -> Gripper:
        """Close or open the gripper, without moving the arm.

        The command is repeated until the fingers stop (about 12 simulator
        steps for a full stroke).  After a close, ``holding`` tells whether
        something stopped the fingers apart and ``width_m`` how wide they
        stand.  Later motions with ``robo.KEEP`` keep commanding what was
        commanded last, so a held object stays squeezed.

        Args:
            gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1).
            opening: with ``robo.OPEN`` only: stop opening at this reading
                (0 < opening <= 1) instead of all the way, e.g. to reach
                between close neighbours.
        Returns:
            Gripper.  ``holding`` is not proof that the INTENDED object is
            held: confirm it moves with the arm (wrist camera, or ``locate``
            it again after a lift).
        """
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
            width = p.get("width_m")
            return Gripper(
                commanded=str(p.get("gripper") or ("closed" if command > 0 else "open")),
                opening=float(p.get("opening", self._latest_state().gripper)),
                holding=bool(p.get("holding")),
                width_m=float(width) if width is not None else None,
                steps_used=int(p.get("steps_used", 0)),
                terminated=bool(p.get("terminated")),
                truncated=bool(p.get("truncated")),
                note=str(p["note"]) if p.get("note") else None,
            )

        return self._stateful("rgb_grip", kwargs, build)

    # -- internals ----------------------------------------------------------

    def _latest_state(self) -> State:
        if self._last_state is None:
            return self.state()
        return self._last_state

    def _move(self, target: tuple[float, ...], p: dict[str, Any]) -> Move:
        commanded = p.get("target_xyz")
        angles = p.get("final_angles") or {}
        return Move(
            target_xyz=_tuple3(commanded if commanded is not None else target),  # type: ignore[arg-type]
            start_xyz=_tuple3(p.get("start_xyz")),  # type: ignore[arg-type]
            final_xyz=_tuple3(p.get("final_xyz")),  # type: ignore[arg-type]
            dist_to_target_m=float(p.get("dist_to_target_m", math.inf)),
            orientation_error_deg=float(p.get("orientation_error_deg", math.nan)),
            pitch=float(angles.get("pitch", math.nan)),
            yaw=float(angles.get("yaw", math.nan)),
            roll=float(angles.get("roll", math.nan)),
            reached=bool(p.get("reached")),
            note=str(p["note"]) if p.get("note") else None,
            blocked=bool(p.get("blocked")),
            steps_used=int(p.get("steps_used", 0)),
            terminated=bool(p.get("terminated")),
            truncated=bool(p.get("truncated")),
            joint_limit=str(p["joint_limit"]) if p.get("joint_limit") else None,
        )

    def _state_from_envelope(self, envelope: dict[str, Any]) -> State:
        raw = envelope.get("state") or {}
        pos = _tuple3(raw.get("robot0_eef_pos")) or (0.0, 0.0, 0.0)
        quat = tuple(float(v) for v in (raw.get("robot0_eef_quat") or (0.0, 0.0, 0.0, 1.0)))
        if len(quat) != 4:
            quat = (0.0, 0.0, 0.0, 1.0)
        qpos = tuple(float(v) for v in (raw.get("robot0_gripper_qpos") or (0.0, 0.0)))[:2]
        opening = (abs(qpos[0]) + abs(qpos[1])) / OPEN_SPAN_M if len(qpos) == 2 else 0.0
        pitch, yaw, roll = _angles(quat)
        return State(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or ""),
            tcp_pos=pos,
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            eef_quat=quat,  # type: ignore[arg-type]
            gripper=round(min(max(opening, 0.0), 1.0), 4),
            object_names=tuple(str(n) for n in raw.get("object_names") or ()),
            terminated=bool(envelope.get("terminated")),
            truncated=bool(envelope.get("truncated")),
        )

    def _summary_dict(self) -> dict[str, Any]:
        """Compact state for the per-turn tool feedback (host-facing)."""
        state = self.state()
        return {
            "step": state.step,
            "tcp": [round(v, 4) for v in state.tcp_pos],
            "pitch_yaw_roll": [round(state.pitch, 1), round(state.yaw, 1), round(state.roll, 1)],
            "gripper": round(state.gripper, 3),
            "terminated": state.terminated,
            "truncated": state.truncated,
        }


def _scaled_k(meta: dict[str, Any]) -> Any:
    """The intrinsics of a recorded camera meta, scaled to the 1024 x 1024 images."""

    k = meta.get("intrinsic_K")
    h, w = meta.get("height"), meta.get("width")
    if k is None or not h or not w:
        return k
    sy, sx = 1024 / float(h), 1024 / float(w)
    return [[k[0][0] * sx, 0.0, k[0][2] * sx], [0.0, k[1][1] * sy, k[1][2] * sy], [0.0, 0.0, 1.0]]


ADAPTER = RobotAdapter(
    name="libero_rgb",
    robot_cls=LiberoColourRobot,
    host="pyrualean.hosts.rpent_libero_rgb",
    knowledge="libero_rgb",
    blurb="a simulated Franka arm in the LIBERO benchmark",
    guides_subdir="",
    guide_names=(),
    show_examples="`robo.show('agentview')` or `robo.show('wrist')`",
    services="a frozen Pi0.5 VLA and a SAM3 segmentation service",
    services_novla="a SAM3 segmentation service",
)


__all__ = [
    "ADAPTER",
    "Gripper",
    "LiberoColourRobot",
    "Move",
    "ObjectSight",
    "STATEFUL_TOOLS",
    "SeenPoint",
    "Segment",
    "State",
]
