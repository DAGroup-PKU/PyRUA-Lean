# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LIBERO tools of this library's own, registered on RPent's LIBERO toolkit at boot.

The primitives of ``robodojo_xpl_rgb`` for LIBERO-PRO's single Franka arm
(:mod:`pyrualean.libero_rgb`), computed in the host from the same observations RPent reads and
commanded through RPent's own action path (``LiberoPrimitives._step_env``):

``rgb_move_pose``     the grasp point to a position AND an orientation (degrees), servoed by
                      LIBERO's OSC_POSE controller, position and rotation together;
``rgb_grip``          a full gripper stroke, repeated until the fingers stop, with what they hold;
``rgb_locate``        an object's 3D extent, heading and top from its masks in the two colour
                      cameras (:mod:`.libero_sight`), no depth image;
``rgb_back_project``  the world point one pixel shows, on the object under it or on the table;
``rgb_segment``       a SAM3 mask's centroid in the current image, no world point.

The names carry a prefix so that none of them replaces one of RPent's LIBERO tools.  Every
motion tool is registered as a stateful tool, so RPent's toolkit records a step after it like
after its own motions; the perception tools are read-only.  Each native action is one
``env.step`` and counts against the episode's budget.

The images measured are rendered from the simulator at SIGHT_SHAPE when a perception tool is
called (the same views RPent records, vertically flipped as it flips them); pixel coordinates
in and out are those of the 1024 x 1024 ``*_high.png`` images the policy reads.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np

from . import libero_sight as sight

#: RPent's camera names for the two views (``agentview`` / ``wrist`` in the robot's API).
CAMERAS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
#: The policy's images (``*_high.png``) and the resolution measured at.
IMAGE_SHAPE = (1024, 1024)
SIGHT_SHAPE = (512, 512)
#: One OSC_POSE action moves the hand up to POS_SCALE_M per unit of action[:3] and turns it
#: ROT_SCALE_RAD per unit of action[3:6] (world-frame axis-angle), measured in the simulator.
POS_SCALE_M = 0.05
ROT_SCALE_RAD = 0.1
#: Largest step commanded per action, and how long a motion may take.
STEP_CLIP_M = 0.02
STEP_CLIP_RAD = 0.08
MOVE_MAX_ACTIONS = 200
#: The motion has arrived when the grasp point is this close (metres) and turned this little.
ARRIVE_M = 0.01
ARRIVE_DEG = 5.0
#: A motion that makes less progress than this over STALL_ACTIONS actions has stopped.
STALL_M = 0.001
STALL_DEG = 0.3
STALL_ACTIONS = 8
#: A gripper stroke stops when the opening changes less than this in one action (metres of
#: finger travel), and takes at most GRIP_MAX_ACTIONS (a full stroke takes about 12).
GRIP_STILL_M = 0.0008
GRIP_MAX_ACTIONS = 20
#: After a close, an opening above HOLD_MIN_OPENING (share of the stroke) means the fingers
#: stopped on something, provided they closed at all: an opening above HOLD_MAX_OPENING after a
#: close is fingers that never met anything (a close across a box's long side stopped at 0.99
#: and lifted nothing).
HOLD_MIN_OPENING = 0.04
HOLD_MAX_OPENING = 0.9
#: The Panda's joint limits (robosuite's panda/robot.xml), radians.  A motion that ends with a
#: joint within JOINT_MARGIN_RAD of one says so: the OSC servo cannot unwind a joint pushed
#: against its limit by itself.  It only reports: stopping motions there (replayed over 50
#: episodes) stopped 29 that would have arrived, and left the arm harder to bring back.
JOINT_LIMITS = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
JOINT_MARGIN_RAD = 0.02
#: A target pitch this close to 90 is straight down: ``yaw`` alone sets the fingers.
VERTICAL_DEG = 1.0
#: The grasp point is RPent's ``robot0_eef_pos`` (robosuite's ``grip_site``, between the finger
#: pads).  The hand body's frame (``robot0_eef_quat``) points along its +z.
POINT_AXIS = 2
#: Without a depth image an object is looked for in the other camera at pixels at least this
#: far apart, and a pixel asked about lies on a measured object when one of that object's
#: measured pixels is this near.  A mask over more than PROBE_TABLE_SHARE of the image is the
#: table.
SEED_APART_PX = 4
PROBE_REACH_PX = 2.5
PROBE_TABLE_SHARE = 0.25
#: A mask on the gripper's own fingers in the wrist camera, the segmenter this unsure of it.
ROBOT_MASK_SCORE = 0.5
ROBOT_MASK_SHARE = 0.5
#: Two cameras whose masks fit the measured solid less well than this agree only roughly.
ROUGH_MATCH = 0.9
#: How a footprint is measured (as in RoboDojo's ``locate``).
FOOTPRINT_CELL_M = 0.002
LONG_FOOTPRINT = 1.4
FOOTPRINT_TRIM = 0.5
FOOT_M = 0.003
FOOT_MIN_HEIGHT_M = 0.012


# ---------------------------------------------------------------------------
# observation and action plumbing
# ---------------------------------------------------------------------------


def _raw(primitives: Any) -> dict[str, Any]:
    return primitives.env.raw_obs()


def _pose(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Grasp point and hand rotation (3 x 3) of an observation."""

    return (
        np.asarray(raw["robot0_eef_pos"], dtype=np.float64),
        sight.quat_matrix(raw["robot0_eef_quat"]),
    )


def opening_m(raw: dict[str, Any]) -> float:
    q = raw["robot0_gripper_qpos"]
    return float(abs(q[0]) + abs(q[1]))


def opening_share(raw: dict[str, Any]) -> float:
    """The gripper's opening as a share of its full stroke: 1.0 open, 0.0 closed."""

    return min(max(opening_m(raw) / sight.OPEN_SPAN_M, 0.0), 1.0)


def _ended(primitives: Any) -> dict[str, bool]:
    return {
        "terminated": bool(primitives.env.terminated),
        "truncated": bool(primitives.env.truncated),
    }


def _over(primitives: Any) -> bool:
    return bool(primitives.env.terminated or primitives.env.truncated)


def _gripper_of(primitives: Any, gripper: Any) -> float:
    """The gripper channel of an action: +1 closes, -1 opens; 0 / None keeps the last command."""

    if gripper is not None and float(gripper) != 0.0:
        g = 1.0 if float(gripper) > 0 else -1.0
        primitives._rgb_grip_command = g
        return g
    return float(getattr(primitives, "_rgb_grip_command", -1.0))


def _step(primitives: Any, dxyz: np.ndarray, drot: np.ndarray, grip: float) -> None:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip(dxyz / POS_SCALE_M, -1.0, 1.0)
    action[3:6] = np.clip(drot / ROT_SCALE_RAD, -1.0, 1.0)
    action[6] = grip
    primitives._step_env(action)


# ---------------------------------------------------------------------------
# orientation (degrees, as in robodojo_xpl_rgb)
# ---------------------------------------------------------------------------


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


#: The hand straight down with the fingers closing along world +x / -x (yaw 0), roll 0: the
#: hand's +z (pointing) is world -z, its closing axis (y, sight.FINGER_AXIS) world +x.
_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_VERTICAL = math.sin(math.radians(1.0))


def orientation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """The hand rotation for (pitch, yaw, roll) in degrees.

    ``pitch`` 90 points straight down, 0 level; ``yaw`` turns about the vertical: pointing
    straight down it is the heading of the line the fingers close along, degrees from +x;
    tilted, the pointing direction leans away from the robot at yaw 0 (towards +x).  ``roll``
    turns the fingers about the pointing direction.
    """

    tilt = math.radians(90.0 - pitch)
    return _rot_z(math.radians(yaw)) @ _rot_y(-tilt) @ _DOWN @ _rot_z(math.radians(roll))


def angles(rot: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees of a hand rotation; roll 0 when pointing straight down."""

    point = rot[:, POINT_AXIS]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    horizontal = math.hypot(point[0], point[1])
    if horizontal < _VERTICAL:
        jaw = rot[:, sight.FINGER_AXIS]
        return pitch, _wrap(math.degrees(math.atan2(jaw[1], jaw[0]))), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0]))
    base = orientation(pitch, yaw, 0.0)
    rest = base.T @ rot  # a rotation about the pointing axis by roll
    roll = math.degrees(math.atan2(rest[1, 0], rest[0, 0]))
    return pitch, _wrap(yaw), _wrap(roll)


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two rotations."""

    c = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _rotvec(rot: np.ndarray) -> np.ndarray:
    c = (np.trace(rot) - 1.0) / 2.0
    angle = math.acos(max(-1.0, min(1.0, c)))
    if angle < 1e-9:
        return np.zeros(3)
    w = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    s = math.sin(angle)
    if s < 1e-6:  # angle near pi: the axis from the symmetric part
        axis = np.sqrt(np.maximum(np.diag(rot) + 1.0, 0.0) / 2.0)
        axis = axis * np.sign(w + (w == 0))
        return axis / np.linalg.norm(axis) * angle
    return w / (2 * s) * angle


def _summary(pos: np.ndarray, rot: np.ndarray) -> dict[str, Any]:
    pitch, yaw, roll = angles(rot)
    return {
        "xyz": [round(float(v), 4) for v in pos],
        "pitch": round(pitch, 1),
        "yaw": round(yaw, 1),
        "roll": round(roll, 1),
    }


# ---------------------------------------------------------------------------
# motion
# ---------------------------------------------------------------------------


def _joint_at_limit(q: np.ndarray) -> str | None:
    """The first joint within JOINT_MARGIN_RAD of a limit, and what to do about it."""

    for i, ((lo, hi), now) in enumerate(zip(JOINT_LIMITS, q, strict=True)):
        if now - lo < JOINT_MARGIN_RAD:
            where = f"lower limit ({now:.2f} of {lo:.2f} rad)"
            return f"joint {i + 1} at its {where}{_way_out(i, False)}"
        if hi - now < JOINT_MARGIN_RAD:
            where = f"upper limit ({now:.2f} of {hi:.2f} rad)"
            return f"joint {i + 1} at its {where}{_way_out(i, True)}"
    return None


def _way_out(joint: int, upper: bool) -> str:
    """What to do about a joint at a limit: joint 4 is the elbow, joints 5-7 the wrist."""

    if joint == 3:
        return (
            ": the arm is stretched out as far as it goes; bring the grasp point nearer the "
            "robot or higher"
            if upper
            else ": the arm is folded as far as it goes; move the grasp point further from the "
            "robot or lower"
        )
    if joint >= 4:
        return ": the wrist is wound up; turn the gripper back the way it came, or turn less"
    return ": the pose is at the edge of the arm's reach; approach it another way"


def _servo(
    primitives: Any,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    grip: float,
    max_actions: int,
) -> tuple[int, bool]:
    """Servo the grasp point to a pose; (actions used, stopped short)."""

    steps = 0
    best = (math.inf, math.inf)
    since = 0
    while steps < max_actions and not _over(primitives):
        pos, rot = _pose(_raw(primitives))
        dist = float(np.linalg.norm(target_pos - pos))
        turn = rotation_error(rot, target_rot)
        if dist <= ARRIVE_M * 0.5 and turn <= ARRIVE_DEG * 0.5:
            return steps, False
        if dist < best[0] - STALL_M or turn < best[1] - STALL_DEG:
            best, since = (min(dist, best[0]), min(turn, best[1])), 0
        else:
            since += 1
            if since >= STALL_ACTIONS:
                return steps, True
        dxyz = target_pos - pos
        n = float(np.linalg.norm(dxyz))
        if n > STEP_CLIP_M:
            dxyz = dxyz * (STEP_CLIP_M / n)
        drot = _rotvec(target_rot @ rot.T)
        a = float(np.linalg.norm(drot))
        if a > STEP_CLIP_RAD:
            drot = drot * (STEP_CLIP_RAD / a)
        primitives._check_cancelled()
        _step(primitives, dxyz, drot, grip)
        steps += 1
    return steps, False


def move_pose(
    primitives: Any,
    xyz: Sequence[float],
    gripper: float = 0,
    pitch: float | None = None,
    yaw: float | None = None,
    roll: float | None = None,
) -> dict[str, Any]:
    """Move the grasp point to ``xyz`` and an orientation: the tool behind ``move_to``.

    ``pitch`` / ``yaw`` / ``roll`` in degrees (:func:`orientation`); an angle left None keeps
    its current value, except that pointing straight down ``yaw`` alone sets the fingers: there
    a roll left None is 0, and a yaw left None is the line the fingers close along now (the
    start pose, 3 degrees off vertical, reads roll 90, which kept would turn them).  The two
    fingers are interchangeable while they hold nothing, so the target turned half a turn about
    the pointing axis is taken when that needs less turning.
    """

    try:
        xyz = [float(v) for v in xyz]
    except (TypeError, ValueError):
        return {"error": "xyz must be three numbers"}
    if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
        return {"error": "xyz must be three finite numbers"}
    for name, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
        if value is not None and not math.isfinite(float(value)):
            return {"error": f"{name} must be a finite number"}
    raw = _raw(primitives)
    grip = _gripper_of(primitives, gripper)
    pos, rot = _pose(raw)
    cur = angles(rot)
    want = tuple(
        float(v) if v is not None else c for v, c in zip((pitch, yaw, roll), cur, strict=True)
    )
    if roll is None and abs(want[0] - 90.0) < VERTICAL_DEG:
        jaw = rot[:, sight.FINGER_AXIS]
        line = math.degrees(math.atan2(jaw[1], jaw[0])) if yaw is None else want[1]
        want = (want[0], line, 0.0)
    target = orientation(*want)
    note = None
    if grip < 0 and opening_share(raw) >= 0.9:
        flipped = orientation(want[0], want[1], want[2] + 180.0)
        if rotation_error(rot, flipped) + 1.0 < rotation_error(rot, target):
            target = flipped
            note = (
                "took the pose turned half a turn about the pointing direction (the empty "
                "fingers are interchangeable): yaw / roll read 180 off"
            )
    goal = np.asarray(xyz, dtype=np.float64)
    steps, stalled = _servo(primitives, goal, target, grip, MOVE_MAX_ACTIONS)
    raw_now = _raw(primitives)
    pos_now, rot_now = _pose(raw_now)
    limit = _joint_at_limit(np.asarray(raw_now["robot0_joint_pos"], dtype=np.float64))
    dist = float(np.linalg.norm(pos_now - goal))
    turn = rotation_error(rot_now, target)
    now = _summary(pos_now, rot_now)
    out: dict[str, Any] = {
        "target_xyz": [round(v, 4) for v in xyz],
        "target_angles": {
            "pitch": round(want[0], 1),
            "yaw": round(want[1], 1),
            "roll": round(want[2], 1),
        },
        "start_xyz": [round(float(v), 4) for v in pos],
        "final_xyz": now["xyz"],
        "final_angles": {k: now[k] for k in ("pitch", "yaw", "roll")},
        "dist_to_target_m": round(dist, 4),
        "orientation_error_deg": round(turn, 1),
        "reached": bool(dist <= ARRIVE_M and turn <= ARRIVE_DEG),
        "steps_used": steps,
        **_ended(primitives),
    }
    if note:
        out["note"] = note
    if stalled and not out["reached"]:
        out["blocked"] = (
            "the arm stopped short of the target: something is in the way, a held object is "
            "pushing against something, or the pose is out of the arm's reach"
        )
    if limit is not None:
        out["joint_limit"] = limit
    return out


def grip(primitives: Any, gripper: float = 1, opening: float | None = None) -> dict[str, Any]:
    """Close (1) or open (-1) the gripper: repeat the command until the fingers stop.

    ``opening`` (0..1, with an open command only) stops opening at that reading.
    """

    try:
        g = float(gripper)
    except (TypeError, ValueError):
        return {"error": "gripper must be 1 (close) or -1 (open)"}
    if g == 0.0:
        return {"error": "gripper must be 1 (close) or -1 (open)"}
    closing = g > 0
    if opening is not None:
        try:
            opening = float(opening)
        except (TypeError, ValueError):
            return {"error": "opening must be a number between 0 and 1"}
        if closing or not 0.0 < opening <= 1.0:
            return {"error": "opening is for an open command, between 0 (excluded) and 1"}
    command = _gripper_of(primitives, 1.0 if closing else -1.0)
    raw = _raw(primitives)
    pos, rot = _pose(raw)
    before = opening_share(raw)
    last = opening_m(raw)
    steps = 0
    while steps < GRIP_MAX_ACTIONS and not _over(primitives):
        if opening is not None and opening_share(_raw(primitives)) >= opening:
            break
        primitives._check_cancelled()
        # hold the pose while the fingers move
        now_pos, now_rot = _pose(_raw(primitives))
        _step(primitives, pos - now_pos, _rotvec(rot @ now_rot.T), command)
        steps += 1
        now = opening_m(_raw(primitives))
        if steps >= 2 and abs(now - last) < GRIP_STILL_M:
            break
        last = now
    if opening is not None:
        # stop the fingers where they are: the next motion holds them there
        primitives._rgb_grip_command = 1.0 if opening_share(_raw(primitives)) < 0.5 else -1.0
    raw = _raw(primitives)
    measured = opening_share(raw)
    holding = bool(closing and HOLD_MIN_OPENING < measured < HOLD_MAX_OPENING)
    out = {
        "gripper": "closed" if closing else "open",
        "opening": round(measured, 3),
        "opening_before": round(before, 3),
        "holding": holding,
        "width_m": round(opening_m(raw), 4) if closing and holding else (0.0 if closing else None),
        "steps_used": steps,
        **_ended(primitives),
    }
    if closing and measured >= HOLD_MAX_OPENING:
        out["note"] = (
            "the fingers stopped almost fully open: something about as wide as the open "
            "gripper (8 cm) is between them, or they press on top of something.  Nothing is "
            "held, but an object may be wedged between the fingers: look before moving"
        )
    return out


# ---------------------------------------------------------------------------
# perception from colour alone
# ---------------------------------------------------------------------------


def _render(primitives: Any, camera: str) -> np.ndarray:
    """The current image of ``camera`` at SIGHT_SHAPE, flipped as RPent flips its records."""

    rgb = primitives.env.render_camera(
        camera_name=CAMERAS[camera], height=SIGHT_SHAPE[0], width=SIGHT_SHAPE[1], depth=False
    )
    if isinstance(rgb, (tuple, list)):
        rgb = rgb[0]
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[::-1])


def _view(primitives: Any, camera: str, raw: dict[str, Any]) -> sight.View:
    """A camera's pose: agentview is fixed, the wrist camera rides on the hand."""

    metas = primitives._rgb_camera_meta
    if camera == "agentview":
        return sight.view_from_meta(metas["agentview"], SIGHT_SHAPE)
    k = sight.view_from_meta(metas["wrist"], SIGHT_SHAPE)
    return sight.View(
        [[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]],
        sight.hand_pose(raw["robot0_eef_pos"], raw["robot0_eef_quat"])
        @ np.asarray(sight.WRIST_MOUNT),
        SIGHT_SHAPE,
    )


def _unknown(view: sight.View, camera: str, raw: dict[str, Any]) -> np.ndarray:
    if camera != "wrist":
        return np.zeros(view.shape, dtype=bool)
    return sight.robot_pixels(
        view, raw["robot0_eef_pos"], raw["robot0_eef_quat"], opening_share(raw)
    )


def _to_sight(row: float, col: float) -> tuple[int, int]:
    return (
        int(round(row * SIGHT_SHAPE[0] / IMAGE_SHAPE[0])),
        int(round(col * SIGHT_SHAPE[1] / IMAGE_SHAPE[1])),
    )


def _to_image(row: float, col: float) -> tuple[int, int]:
    return (
        int(round(row * IMAGE_SHAPE[0] / SIGHT_SHAPE[0])),
        int(round(col * IMAGE_SHAPE[1] / SIGHT_SHAPE[1])),
    )


def _cells(xy: np.ndarray, cell: float = FOOTPRINT_CELL_M) -> np.ndarray:
    _, cell_of, count = np.unique(
        np.floor(xy / cell).astype(np.int64), axis=0, return_inverse=True, return_counts=True
    )
    cell_of = cell_of.reshape(-1)
    sums = np.stack([np.bincount(cell_of, weights=xy[:, i]) for i in (0, 1)], axis=1)
    return sums / count[:, None]


def _footprint(xy: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """A footprint's rectangle: the heading of its long side (degrees from +x, -90 .. 90), its
    length and width, and its centre (as RoboDojo's ``locate`` measures one)."""

    cells = _cells(xy)
    trim = [FOOTPRINT_TRIM, 100.0 - FOOTPRINT_TRIM]

    def extents(deg: float) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
        a = math.radians(deg)
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        (ul, uh), (vl, vh) = np.percentile(cells @ u, trim), np.percentile(cells @ v, trim)
        return float((uh - ul) * (vh - vl)), ul, uh, vl, vh, u, v

    if len(cells) < 3:
        c = cells.mean(axis=0)
        return 0.0, 0.0, 0.0, c
    spread, axes = np.linalg.eigh(np.cov((cells - cells.mean(axis=0)).T))
    if spread[1] >= LONG_FOOTPRINT * max(float(spread[0]), 1e-12):
        heading = math.degrees(math.atan2(axes[1, 1], axes[0, 1]))
    else:
        coarse = min(np.arange(0.0, 90.0, 2.0), key=lambda d: extents(d)[0])
        heading = min(np.arange(coarse - 2.0, coarse + 2.01, 0.25), key=lambda d: extents(d)[0])
    _, ul, uh, vl, vh, u, v = extents(heading)
    centre = u * (ul + uh) / 2 + v * (vl + vh) / 2
    length, width = uh - ul, vh - vl
    if width > length:
        length, width, heading = width, length, heading + 90.0
    return (heading + 90.0) % 180.0 - 90.0, float(length), float(width), centre


def object_geometry(points: np.ndarray, table_z: float) -> dict[str, Any] | None:
    """Extent, heading and heights of an object from its surface points (N x 3)."""

    pts = points[points[:, 2] > table_z + 0.004]
    if len(pts) < 12:
        return None
    med = np.median(pts[:, :2], axis=0)
    spread = np.median(np.linalg.norm(pts[:, :2] - med, axis=1))
    pts = pts[np.linalg.norm(pts[:, :2] - med, axis=1) <= max(4.0 * spread, 0.03)]
    top = float(np.percentile(pts[:, 2], 97))
    low = float(pts[:, 2].min())
    standing = pts[pts[:, 2] > low + FOOT_M] if top - low > FOOT_MIN_HEIGHT_M else pts
    heading, length, width, mid = _footprint(standing[:, :2])
    return {
        "center_xyz": [
            round(float(mid[0]), 4),
            round(float(mid[1]), 4),
            round((top + table_z) / 2, 4),
        ],
        "top_z": round(top, 4),
        "height_m": round(top - table_z, 4),
        "length_m": round(length, 4),
        "width_m": round(width, 4),
        "long_axis_deg": round(heading, 1),
        # pointing straight down, this yaw closes the fingers across the narrow side
        "grasp_yaw": round((heading + 180.0) % 180.0 - 90.0, 1),
        "points": int(len(pts)),
    }


def _scene(primitives: Any, images: dict[str, np.ndarray]) -> dict[str, Any]:
    """What has been measured in these images (kept until they change)."""

    key = tuple(
        zlib.crc32(np.ascontiguousarray(images[c][::8, ::8]).tobytes()) for c in sorted(images)
    )
    scene = getattr(primitives, "_rgb_scene", None)
    if not isinstance(scene, dict) or scene.get("key") != key:
        scene = {"key": key, "regions": [], "flat": []}
        primitives._rgb_scene = scene
    return scene


def _measure(
    primitives: Any,
    raw: dict[str, Any],
    images: dict[str, np.ndarray],
    camera: str,
    mask: np.ndarray,
    text_prompt: str | None,
    min_score: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """An object's geometry from its mask in ``camera`` and what the other camera shows of it."""

    table_z = float(primitives._rgb_table_z)
    seen: dict[str, Any] = {"views": 1}
    view = _view(primitives, camera, raw)
    first = sight.Sight(view, mask)
    sam3 = primitives._sam3_client
    others = []
    spots = sight.seeds(first, table_z)
    for name, color in images.items():
        if name == camera:
            continue
        other = _view(primitives, name, raw)
        unknown = _unknown(other, name, raw)
        masks = []
        if text_prompt:
            res = sam3.segment(color, text_prompt=text_prompt, min_score=min_score)
            if res.found:
                masks.append(np.asarray(res.mask, dtype=bool))
        tried: list[tuple[int, int]] = []
        if len(spots):
            r, c, depth = other.project(spots)
            for ri, ci, di in zip(r, c, depth, strict=True):
                at = (int(round(float(ri))), int(round(float(ci))))
                inside = 0 <= at[0] < other.shape[0] and 0 <= at[1] < other.shape[1]
                if di <= 0.02 or not inside or unknown[at]:
                    continue
                if any(abs(at[0] - a) + abs(at[1] - b) < SEED_APART_PX for a, b in tried):
                    continue
                if any(m[at] for m in masks):
                    continue
                tried.append(at)
                res = sam3.segment(color, point=[at[0], at[1]], min_score=min_score)
                if res.found:
                    masks.append(np.asarray(res.mask, dtype=bool))
        if masks:
            others.append([sight.Sight(other, m, unknown) for m in masks])
    key = " ".join(str(text_prompt or "").lower().split())
    heights = primitives.__dict__.setdefault("_rgb_heights", {})
    known = heights.get(key) if key else None
    res = sight.measure(first, others, table_z, known=known)
    if res is None:
        return None, seen
    seen["views"] = int(res["views"])
    if res["views"] >= 2:
        rough = res["match"] is not None and res["match"] < ROUGH_MATCH
        if key and not rough:
            heights[key] = (float(res["height_m"]), str(res["kind"]), res["match"])
        if rough:
            seen["guess"] = (
                "the cameras agree only roughly: its height may be centimetres off "
                "(locate it again from above before a grasp)"
            )
    elif res.get("remembered"):
        seen["guess"] = "its height as measured before"
    else:
        seen["guess"] = (
            "one camera sees it: its height is a guess (locate it again from above before a grasp)"
        )
    points = np.asarray(res["points"], dtype=np.float64)
    _scene(primitives, images)["regions"].append(
        {"camera": camera, "rows": res["rows"], "cols": res["cols"], "points": points}
    )
    return object_geometry(points, table_z), seen


def _images(primitives: Any) -> dict[str, np.ndarray]:
    return {camera: _render(primitives, camera) for camera in CAMERAS}


def _camera_ok(camera: str) -> str | None:
    return None if camera in CAMERAS else f"camera must be one of {sorted(CAMERAS)}"


def segment(
    primitives: Any, text_prompt: str, camera: str = "agentview", min_score: float = 0.2
) -> dict[str, Any]:
    """SAM3 on the current image of ``camera``: the mask's centroid, area and score."""

    if (why := _camera_ok(camera)) is not None:
        return {"error": why}
    if not isinstance(text_prompt, str) or not text_prompt.strip():
        return {"error": "text_prompt must be a non-empty string"}
    res = primitives._sam3_client.segment(
        _render(primitives, camera), text_prompt=text_prompt, min_score=float(min_score)
    )
    out: dict[str, Any] = {"camera": camera, "found": bool(res.found), "text_prompt": text_prompt}
    if not res.found:
        return out
    mask = np.asarray(res.mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    scale = (IMAGE_SHAPE[0] / SIGHT_SHAPE[0]) * (IMAGE_SHAPE[1] / SIGHT_SHAPE[1])
    out.update(
        score=round(float(res.score), 3),
        area_px=int(mask.sum() * scale),
        centroid_rc=list(_to_image(rows.mean(), cols.mean())) if len(rows) else None,
    )
    return out


def locate(
    primitives: Any, text_prompt: str, camera: str = "agentview", min_score: float = 0.2
) -> dict[str, Any]:
    """Segment an object by text and measure it in 3D from the two colour cameras."""

    if (why := _camera_ok(camera)) is not None:
        return {"error": why}
    if not isinstance(text_prompt, str) or not text_prompt.strip():
        return {"error": "text_prompt must be a non-empty string"}
    raw = _raw(primitives)
    images = _images(primitives)
    sam3 = primitives._sam3_client
    res = sam3.segment(images[camera], text_prompt=text_prompt, min_score=float(min_score))
    via = None
    if not res.found and camera == "wrist":
        # Straight from above an object shows its top only (a can's lid, a bottle's cap), which
        # a text prompt often misses: measure from agentview's mask instead, with the wrist
        # image as its second view (where the wrist camera sees the object it settles the
        # height just the same).
        camera, via = "agentview", "the wrist image matched nothing for the text"
        res = sam3.segment(images[camera], text_prompt=text_prompt, min_score=float(min_score))
    if not res.found:
        return {"camera": camera, "found": False, "text_prompt": text_prompt}
    mask = np.asarray(res.mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    scale = (IMAGE_SHAPE[0] / SIGHT_SHAPE[0]) * (IMAGE_SHAPE[1] / SIGHT_SHAPE[1])
    out: dict[str, Any] = {
        "camera": camera,
        "found": True,
        "text_prompt": text_prompt,
        "score": round(float(res.score), 3),
        "area_px": int(mask.sum() * scale),
        "centroid_rc": list(_to_image(rows.mean(), cols.mean())) if len(rows) else None,
    }
    if via:
        out["measured_from"] = via
    view = _view(primitives, camera, raw)
    unknown = _unknown(view, camera, raw)
    share = float(unknown[mask].mean()) if mask.any() else 0.0
    if share >= ROBOT_MASK_SHARE and float(res.score) < ROBOT_MASK_SCORE:
        out["views"] = 1
        out["geometry_error"] = "the mask lies on the gripper's own fingers"
        return out
    geo, seen = _measure(primitives, raw, images, camera, mask, text_prompt, float(min_score))
    out["views"] = seen["views"]
    if seen.get("guess"):
        out["height_guess"] = seen["guess"]
    if geo is None:
        out["geometry_error"] = "too little of it above the table (flat, far or hidden)"
        return out
    out.update(geo)
    return out


def back_project(primitives: Any, row: int, col: int, camera: str = "agentview") -> dict[str, Any]:
    """The world point a pixel of the ``*_high.png`` image shows, without a depth image."""

    if (why := _camera_ok(camera)) is not None:
        return {"error": why}
    row, col = int(row), int(col)
    if not (0 <= row < IMAGE_SHAPE[0] and 0 <= col < IMAGE_SHAPE[1]):
        return {
            "error": f"pixel [{row}, {col}] is outside the "
            f"{IMAGE_SHAPE[0]} x {IMAGE_SHAPE[1]} image"
        }
    raw = _raw(primitives)
    images = _images(primitives)
    view = _view(primitives, camera, raw)
    table_z = float(primitives._rgb_table_z)
    sr, sc = _to_sight(row, col)
    sr, sc = min(sr, SIGHT_SHAPE[0] - 1), min(sc, SIGHT_SHAPE[1] - 1)

    def answer(xyz: np.ndarray, surface: str) -> dict[str, Any]:
        return {
            "camera": camera,
            "pixel": [row, col],
            "world_xyz": [round(float(v), 4) for v in xyz],
            "depth_m": round(float(view.depth_of(xyz)), 4),
            "surface": surface,
        }

    def on_table() -> dict[str, Any]:
        xy, ok = view.on_plane(np.array([sr]), np.array([sc]), table_z)
        if not ok[0]:
            return {
                "error": "the pixel looks above the table's plane: "
                "there is no depth image to say where"
            }
        return answer(np.array([xy[0, 0], xy[0, 1], table_z]), "table")

    def in_region(region: dict[str, Any]) -> dict[str, Any] | None:
        if region["camera"] != camera or len(region["rows"]) == 0:
            return None
        gap = (region["rows"] - sr) ** 2 + (region["cols"] - sc) ** 2
        i = int(np.argmin(gap))
        if gap[i] > PROBE_REACH_PX**2:
            return None
        if gap[i] == 0:
            return answer(region["points"][i], "object")
        d = view.rays(np.array([sr]), np.array([sc]))[0]
        return answer(view.pos + d * float(view.depth_of(region["points"][i])), "object")

    scene = _scene(primitives, images)
    for region in scene["regions"]:
        if (hit := in_region(region)) is not None:
            return hit
    for flat in scene["flat"]:
        if flat["camera"] == camera and flat["mask"][sr, sc]:
            return on_table()
    res = primitives._sam3_client.segment(images[camera], point=[sr, sc], min_score=0.0)
    mask = np.asarray(res.mask, dtype=bool) if res.found else None
    if mask is None or not mask[sr, sc] or mask.sum() > PROBE_TABLE_SHARE * mask.size:
        if mask is not None and mask[sr, sc]:
            scene["flat"].append({"camera": camera, "mask": mask})
        return on_table()
    geo, _seen = _measure(primitives, raw, images, camera, mask, None, 0.0)
    if geo is not None and (hit := in_region(scene["regions"][-1])) is not None:
        return hit
    scene["flat"].append({"camera": camera, "mask": mask})
    return on_table()


for _f in (segment, locate, back_project):
    _f._readonly = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_CAMERA = {"type": "string", "enum": sorted(CAMERAS)}
SPECS: dict[str, dict[str, Any]] = {
    "rgb_move_pose": {
        "name": "rgb_move_pose",
        "description": "Move the grasp point to a position and orientation (degrees); angles "
        "left out keep their current value.",
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {"type": "array", "items": {"type": "number"}},
                "gripper": {"type": "number"},
                "pitch": {"type": "number"},
                "yaw": {"type": "number"},
                "roll": {"type": "number"},
            },
            "required": ["xyz"],
        },
    },
    "rgb_grip": {
        "name": "rgb_grip",
        "description": "Close (1) or open (-1) the gripper fully and report what it holds.",
        "input_schema": {
            "type": "object",
            "properties": {"gripper": {"type": "number"}, "opening": {"type": "number"}},
            "required": ["gripper"],
        },
    },
    "rgb_locate": {
        "name": "rgb_locate",
        "description": "Segment an object by text and measure its extent, heading and height "
        "from the colour cameras.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text_prompt": {"type": "string"},
                "camera": _CAMERA,
                "min_score": {"type": "number"},
            },
            "required": ["text_prompt"],
        },
    },
    "rgb_segment": {
        "name": "rgb_segment",
        "description": "Segment an object by text in the current image: centroid, area, score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text_prompt": {"type": "string"},
                "camera": _CAMERA,
                "min_score": {"type": "number"},
            },
            "required": ["text_prompt"],
        },
    },
    "rgb_back_project": {
        "name": "rgb_back_project",
        "description": "The world point a pixel shows: on the object under it, else on the table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {"type": "integer"},
                "col": {"type": "integer"},
                "camera": _CAMERA,
            },
            "required": ["row", "col"],
        },
    },
}


def register(toolkit: Any) -> dict[str, Any]:
    """Add this module's tools to a freshly booted LIBERO toolkit.

    Reads, once: the start position (for the record), the two cameras' calibration, and the table's
    height from one agentview depth frame (the only depth read; perception after that is
    colour alone).  Returns those for the run's record.
    """

    primitives = toolkit._primitives
    env = primitives.env
    raw = env.raw_obs()
    pos, _rot = _pose(raw)
    metas = {
        name: env.get_camera_meta(cam, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
        for name, cam in CAMERAS.items()
    }
    if any(m is None for m in metas.values()):
        raise RuntimeError("LIBERO camera calibration missing")
    _rgb, depth = env.render_camera(camera_name="agentview", height=256, width=256, depth=True)
    meta_low = env.get_camera_meta("agentview", 256, 256)
    world = _world_of_depth(np.asarray(depth), meta_low)
    table_z = sight.table_height(world, float(pos[2]))
    if table_z is None:
        raise RuntimeError("could not find the table in the agentview depth frame")
    primitives._rgb_camera_meta = metas
    primitives._rgb_table_z = table_z
    primitives._rgb_grip_command = -1.0 if opening_share(raw) > 0.5 else 1.0
    handlers = {
        "rgb_move_pose": partial(move_pose, primitives),
        "rgb_grip": partial(grip, primitives),
        "rgb_locate": partial(locate, primitives),
        "rgb_segment": partial(segment, primitives),
        "rgb_back_project": partial(back_project, primitives),
    }
    for name, handler in handlers.items():
        toolkit.add_tool(name, SPECS[name], handler)
    return {"table_z": round(table_z, 4), "start_xyz": [round(float(v), 4) for v in pos]}


def _world_of_depth(depth: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """World points of one raw depth frame (robosuite's normalised depth, the buffer upside
    down), as RPent turns its depth records into world maps."""

    near, far = float(meta["depth_near"]), float(meta["depth_far"])
    d = np.asarray(depth, dtype=np.float64)
    d = d[..., 0] if d.ndim == 3 else d
    metric = near / (1.0 - d * (1.0 - near / far))
    metric = metric[::-1]
    view = sight.view_from_meta(meta)
    rows, cols = np.mgrid[0 : metric.shape[0], 0 : metric.shape[1]]
    pts = view.pos + view.rays(rows.ravel(), cols.ravel()) * metric.ravel()[:, None]
    return pts.reshape(*metric.shape, 3)


__all__ = [
    "CAMERAS",
    "IMAGE_SHAPE",
    "SIGHT_SHAPE",
    "SPECS",
    "angles",
    "back_project",
    "grip",
    "locate",
    "move_pose",
    "object_geometry",
    "orientation",
    "register",
    "segment",
]
