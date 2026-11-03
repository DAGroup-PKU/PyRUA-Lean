# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The G2's primitives over a GenieSim bridge, with no simulator in this process.

Frames.  Everything here is in ``base_link``: the robot's own frame, z up, x forward,
left arm on +y.  The benchmark reports each arm's END-EFFECTOR LINK pose there.  A motion
is solved on this side, on the benchmark's own IK (:mod:`.robocoliseum_motion`,
:mod:`.robocoliseum_reach`), and sent as ``JOINT_ABS`` joint targets.

Tool geometry, measured from the robot's own URDF (``G2_omnipicker.urdf``):

  * the pointing direction is the end-effector link's +z,
  * the TOOL CENTRE sits 0.15 m along it (``gripper_*_center_link``; the same offset the
    benchmark's own frame contract records),
  * the fingers close along the link's +y (the outer finger sits at +0.0195 m, the inner
    at -0.0195 m), so +x is the normal of the jaw plane.

A tool pose is therefore: where the tool centre is, the direction it points, and the
horizontal line the fingers close along, turned about the pointing direction by
``yaw_deg``.

One action moves little.  The benchmark takes each joint target and then waits for the
joints to settle, so a target far from where the arm is makes one violent jump; a path is
therefore sent as many small joint steps, and the arm is re-read after every action.

Gripper values are the benchmark's own: NEGATIVE is open (about -0.785), 0 is closed.
The robot class reports the opposite sign; see :class:`pyrualean.robocoliseum.RobocoliseumRobot`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import geniesim_bridge as bridge_mod
from . import robocoliseum_motion as motion
from . import robocoliseum_reach as reach

#: Distance from the end-effector link to the tool centre, along the pointing direction.
#: The gripper's centre: ``gripper_{l,r}_center_link`` sits 0.14308 m out along the end
#: link's z (URDF ``idx92_gripper_r_center_joint``).  The benchmark's reach and pick-up checks
#: measure from that link, so the grasp point is that link.
TOOL_OFFSET_M = 0.143
#: A motion has arrived when the tool centre is this close and this well turned.  The
#: simulator settles a joint target to ~1 cm and the IK is solved per action, so a
#: millimetre tolerance would never be met and every move would spend its whole budget.
ARRIVE_M = 0.015
ARRIVE_DEG = 5.0
#: A motion that stops making progress for this many actions has met something (a limit,
#: an object, the table) and reports where it stopped instead of burning the budget.
STALL_ACTIONS = 4
#: After every action the benchmark waits for the arm joints to come within 0.01 rad of
#: their target, giving up after 1.0 s (``PiEnv._wait_arm_settled``).  An arm that is
#: pressed against something never gets there, so each of its actions takes the whole
#: second; an arm at the edge of its reach is given joint targets it can reach and settles.
#: Measured: 0.15-0.25 s per action in free motion and at the edge of reach, 1.10-1.13 s
#: from the moment the fingers met the table.  An action at least this long was resisted.
RESISTED_S = 0.7
#: Actions a motion may spend by default.  A turn and a translation are separate motions
#: (see ``move_to``), and turning alone can drift the tool centre by ~0.24 m before the
#: translation starts, so the default has to cover a turn plus a long trip.
MOVE_MAX_ACTIONS = 60
#: Gripper command values, in the benchmark's convention: open is negative.
GRIP_OPEN = -0.785
GRIP_CLOSE = 0.0
#: A close that leaves the fingers apart by more than this is holding something.
HOLD_MIN_GAP = 0.05
#: Actions a grip may spend before it reports the opening it reached.  Measured on an
#: empty G2 gripper: a close goes -0.785, -0.58, -0.45, -0.32, -0.25, -0.20 (after 6
#: actions — where a close on a 5 cm block also stops, so a cap of 6 read every empty
#: close as holding), and settles below -0.01 after about 16; an open takes about 14.
GRIP_MAX_ACTIONS = 20
#: An opening change below this means the fingers have stopped.
GRIP_STILL = 0.005

#: The frame every position and direction is expressed in, and the frame actions are
#: declared in — the same frame, measured.  It is the frame the benchmark reports each
#: arm's end-effector pose in: sending that pose straight back holds the joints still
#: (0.0001 rad), and a target moved along one of its axes moves the arm that way (probes:
#: +3 cm asked, +1.2/+2.5/+2.3 cm measured in the same direction, closing the rest over
#: the following actions).  Actions must NOT be built from the other reported frame
#: (``arm_base_link``): that one moves the joints by 2.35 rad.
FRAME = "base_link"
#: Named directions as seen from the upper body facing base_link +x; ``named_direction``
#: turns the horizontal ones with the torso, so "forward" is always the torso's front.
POINTS: dict[str, tuple[float, float, float]] = {
    "down": (0.0, 0.0, -1.0),
    "up": (0.0, 0.0, 1.0),
    "forward": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
}
ARMS = ("left", "right")


# ---------------------------------------------------------------------------
# rotation helpers (numpy only: this package has no science dependencies)
# ---------------------------------------------------------------------------


def unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n <= 1e-12:
        raise ValueError("direction must be a non-zero vector")
    return a / n


def quat_xyzw_matrix(quat: Sequence[float]) -> np.ndarray:
    """A unit quaternion [x, y, z, w] as a 3x3 rotation matrix."""

    x, y, z, w = (float(v) for v in np.asarray(quat, dtype=np.float64).reshape(4))
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-12:
        raise ValueError("a zero quaternion is not a rotation")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quat_xyzw(rot: np.ndarray) -> list[float]:
    """A 3x3 rotation matrix as a unit quaternion [x, y, z, w], canonicalised to w >= 0."""

    m = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return [float(v) for v in (x, y, z, w)]


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    """The angle between two directions, degrees."""

    va, vb = unit(a), unit(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(va @ vb)))))


# ---------------------------------------------------------------------------
# tool geometry
# ---------------------------------------------------------------------------


def torso_yaw(waist: Sequence[float]) -> float:
    """The upper body's heading about base_link z, radians (the waist turn joint, idx05).

    0 faces base_link +x; +pi/2 faces +y (to the base's left).  Some tasks START turned:
    the warehouse tasks begin at +1.57, so their table is along +y, not +x.
    """

    return float(waist[4]) if len(waist) >= 5 else 0.0


def named_direction(name: str, yaw: float = 0.0) -> np.ndarray:
    """A named direction in base_link.  The horizontal ones are relative to where the
    upper body faces (``yaw``): ``"forward"`` is the torso's front, not base_link +x."""

    if name not in POINTS:
        raise ValueError(f"point must be one of {sorted(POINTS)} or a direction")
    x, y, z = POINTS[name]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float64)


def tool_centre(flange_pos: Sequence[float], rot: np.ndarray) -> np.ndarray:
    """The tool centre implied by an end-effector link position and rotation."""

    return np.asarray(flange_pos, dtype=np.float64) + TOOL_OFFSET_M * rot[:, 2]


def flange_target(centre: Sequence[float], rot: np.ndarray) -> np.ndarray:
    """The end-effector link position that puts the tool centre at ``centre``."""

    return np.asarray(centre, dtype=np.float64) - TOOL_OFFSET_M * rot[:, 2]


# ---------------------------------------------------------------------------
# what the tools are bound to
# ---------------------------------------------------------------------------


class Primitives:
    """The bridge, the episode's first pose, and the last gripper command per arm."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.start_joints: dict[str, list[float]] = {}
        self.start_waist: list[float] = []
        self.grip_commands: dict[str, float] = {}
        #: The joint angles each arm was last commanded (an arm that holds is held at its
        #: command, not at its measurement: a loaded arm sags, and re-commanding the sag
        #: lets it sink further every action).
        self.joint_commands: dict[str, list[float]] = {}
        #: Surfaces measured under located objects, (x, y, z) in FRAME: the fingertips must
        #: not be sent below the one under where they go.
        self.surfaces: list[tuple[float, float, float]] = []
        self.last_act_s = 0.0

    def obs(self) -> bridge_mod.Observation:
        return self.env.get_obs()

    def frames(self) -> dict[str, np.ndarray]:
        return self.env.frames()

    def status(self) -> dict[str, Any]:
        try:
            status = self.env.get_status()
        except Exception:  # noqa: BLE001 - optional metadata
            return {}
        return status if isinstance(status, dict) else {}

    def pose(self, arm: str, frame: str = "arm_base_link") -> tuple[np.ndarray, np.ndarray]:
        """One arm's end-effector link ``(position, rotation)`` in ``frame``."""

        position, quat = self.obs().flange(arm, frame)
        return position, quat_xyzw_matrix(quat)

    def arm_state(self, arm: str) -> dict[str, Any]:
        """One arm's tool-centre pose, pointing direction and gripper opening."""

        position, rot = self.pose(arm, FRAME)
        centre = tool_centre(position, rot)
        return {
            "arm": arm,
            "tool_centre": [round(float(v), 4) for v in centre],
            "pointing": [round(float(v), 3) for v in rot[:, 2]],
            "finger_axis": [round(float(v), 3) for v in rot[:, 0]],
            "flange": [round(float(v), 4) for v in position],
            "flange_quat_xyzw": matrix_quat_xyzw(rot),
            "angles": dict(zip(("pitch", "yaw", "roll"), motion.angles(rot), strict=False)),
            "opening": round(float(opening(self.obs(), arm)), 3),
        }

    def act(self, action: dict[str, Any]) -> bridge_mod.Observation:
        start = time.monotonic()
        obs, _ = self.env.step(action)
        #: How long the benchmark took to answer: see RESISTED_S.
        self.last_act_s = time.monotonic() - start
        return obs


def opening(obs: bridge_mod.Observation, arm: str) -> float:
    """A measured gripper opening in the benchmark's convention (open is negative)."""

    if len(obs.gripper) < 2:
        return 0.0
    return float(obs.gripper[0] if arm == "left" else obs.gripper[1])


def commanded_grip(primitives: Primitives, arm: str, obs: bridge_mod.Observation) -> float:
    """What this arm's gripper was last told, or its measured opening before any command.

    Every action repeats the last command for the gripper it is not changing: re-sending
    the opening a held object leaves (a close stopped at -0.25 by what it holds) would
    command the fingers to that opening and take the squeeze off.
    """

    return float(primitives.grip_commands.get(arm, opening(obs, arm)))


def action(
    primitives: Primitives,
    arm: str,
    flange_pos: Sequence[float],
    rot: np.ndarray,
    grip: float | None,
) -> dict[str, Any]:
    """One ``EEF_ABS`` action: this arm to a pose, the other held where it is.

    ``flange_pos`` and ``rot`` are in :data:`FRAME`, which is also the frame the action is
    declared in — no conversion exists between them.
    ``grip`` None repeats this arm's last command (not its measured opening: commanding
    the opening a held object leaves takes the squeeze off).
    """

    obs = primitives.obs()
    other = "right" if arm == "left" else "left"
    theirs_pos, theirs_rot = primitives.pose(other, FRAME)
    if grip is None:
        grip = commanded_grip(primitives, arm, obs)
    primitives.grip_commands[arm] = float(grip)
    mine = [float(v) for v in flange_pos] + matrix_quat_xyzw(rot)
    theirs = [float(v) for v in theirs_pos] + matrix_quat_xyzw(theirs_rot)
    left, right = (mine, theirs) if arm == "left" else (theirs, mine)
    theirs_grip = commanded_grip(primitives, other, obs)
    left_grip, right_grip = (
        (float(grip), theirs_grip) if arm == "left" else (theirs_grip, float(grip))
    )
    return {
        "kind": "EEF_ABS",
        "eef_frame": FRAME,
        "left_arm": left,
        "right_arm": right,
        "left_gripper": left_grip,
        "right_gripper": right_grip,
    }


#: ``move_base`` sends the base at most this far / turned this much per action.
BASE_STEP_M = 0.03
BASE_STEP_DEG = 5.0
#: Still actions after a base motion, for the chassis servo to finish the last piece.
BASE_SETTLE_ACTIONS = 3
BASE_MAX_M = 2.0
BASE_MAX_DEG = 180.0


def move_base(
    primitives: Primitives,
    forward: float = 0.0,
    left: float = 0.0,
    turn_deg: float = 0.0,
) -> dict[str, Any]:
    """Drive the wheeled base: ``forward`` / ``left`` in metres along the base's own axes,
    ``turn_deg`` counter-clockwise.  Only tasks with a free base move; elsewhere the base
    is fixed and nothing happens.

    OPEN LOOP: the benchmark reports no base pose, so this says what was commanded, not
    where the base went.  base_link moves with the base — every pose the arms report is
    still right after a move — but object positions measured before it are stale; look again.
    """

    values = [float(forward), float(left), float(turn_deg)]
    if not all(math.isfinite(v) for v in values):
        return {"error": "forward, left and turn_deg must be finite"}
    if math.hypot(values[0], values[1]) > BASE_MAX_M or abs(values[2]) > BASE_MAX_DEG:
        return {"error": f"one move_base is at most {BASE_MAX_M} m and {BASE_MAX_DEG} degrees"}
    pieces = max(
        1,
        int(math.ceil(math.hypot(values[0], values[1]) / BASE_STEP_M - 1e-9)),
        int(math.ceil(abs(values[2]) / BASE_STEP_DEG - 1e-9)),
    )
    before = {arm: primitives.pose(arm, "arm_base_link")[0] for arm in ARMS}
    spent = 0
    for index in range(pieces + BASE_SETTLE_ACTIONS):
        obs = primitives.obs()
        measured = list(obs.gripper) + [0.0, 0.0]
        step = (
            [values[0] / pieces, values[1] / pieces, math.radians(values[2]) / pieces]
            if index < pieces
            else [0.0, 0.0, 0.0]
        )
        primitives.act(
            {
                "kind": "JOINT_ABS",
                "left_arm": held_joints(primitives, "left", obs),
                "right_arm": held_joints(primitives, "right", obs),
                "left_gripper": float(primitives.grip_commands.get("left", measured[0])),
                "right_gripper": float(primitives.grip_commands.get("right", measured[1])),
                "wheels": step,
            }
        )
        spent += 1
    after = {arm: primitives.pose(arm, "arm_base_link")[0] for arm in ARMS}
    drift = max(float(np.linalg.norm(after[a] - before[a])) for a in ARMS)
    return {
        "commanded": {"forward_m": values[0], "left_m": values[1], "turn_deg": values[2]},
        "steps_used": spent,
        "arms_drift_m": round(drift, 4),
        "note": "open loop: confirm where the base went with the head camera",
    }


ALT_CHEST = ((0.05, 0.0), (0.10, 0.0), (0.10, -0.05), (0.15, 0.0))


# ---------------------------------------------------------------------------
# motion solved in the library, commanded as joint angles (the RoboDojo design)
# ---------------------------------------------------------------------------

#: A motion that has followed its path closes the last gap with at most this many actions,
#: re-solving from the joints it measures.
SETTLE_ACTIONS = 6
#: Arrival: the tool centre within this and the whole orientation within ARRIVE_DEG.
POSE_ARRIVE_M = 0.01
#: Alternatives tried for a pose out of reach, cheapest change first.
POSE_ALT_PITCH = (15.0, -15.0, 30.0, -30.0, 45.0)
POSE_ALT_Z = (0.03, 0.06, -0.03)
POSE_ALT_YAW = (20.0, -20.0, 40.0, -40.0)


def held_joints(primitives: Primitives, arm: str, obs: bridge_mod.Observation) -> list[float]:
    measured = [float(v) for v in obs.arm_joints]
    return list(
        primitives.joint_commands.get(arm) or (measured[:7] if arm == "left" else measured[7:14])
    )


def joint_action(
    primitives: Primitives, arm: str, q: Sequence[float], grip: float | None
) -> dict[str, Any]:
    """One ``JOINT_ABS`` action: this arm to ``q``, the other held at its last command."""

    obs = primitives.obs()
    other = "right" if arm == "left" else "left"
    mine = [float(v) for v in q]
    theirs = held_joints(primitives, other, obs)
    primitives.joint_commands[arm] = mine
    primitives.joint_commands[other] = theirs
    if grip is None:
        grip = commanded_grip(primitives, arm, obs)
    primitives.grip_commands[arm] = float(grip)
    theirs_grip = commanded_grip(primitives, other, obs)
    left, right = (mine, theirs) if arm == "left" else (theirs, mine)
    return {
        "kind": "JOINT_ABS",
        "left_arm": left,
        "right_arm": right,
        "left_gripper": float(grip) if arm == "left" else theirs_grip,
        "right_gripper": float(grip) if arm == "right" else theirs_grip,
    }


def _pose_now(primitives: Primitives, arm: str) -> tuple[np.ndarray, np.ndarray]:
    position, rot = primitives.pose(arm, FRAME)
    return tool_centre(position, rot), rot


def _pose_offers(
    primitives: Primitives,
    kin: motion.Arm,
    arm: str,
    xyz: np.ndarray,
    pitch: float,
    yaw: float,
    roll: float,
    seeds: list[np.ndarray],
) -> list[dict[str, Any]]:
    """Nearby requests the IK says ARE reachable, each as ``move_to`` arguments."""

    def ok(
        c: np.ndarray, p: float, y: float, r: float, chest: tuple[float, float] = (0.0, 0.0)
    ) -> bool:
        k = kin
        if chest != (0.0, 0.0):
            k = motion.Arm(primitives.obs(), arm)
            k.base = k.base.copy()
            k.base[0, 3] += chest[0]
            k.base[2, 3] += chest[1]
            k.inv_base = np.linalg.inv(k.base)
        _, pe, re = motion.solve_goal(k, c, motion.orientation(p, y, r), seeds)
        return pe <= motion.REACH_M and re <= motion.REACH_DEG

    offers: list[dict[str, Any]] = []
    target = [round(float(v), 4) for v in xyz]
    for dp in POSE_ALT_PITCH:
        p = max(-89.0, min(90.0, pitch + dp))
        if p != pitch and ok(xyz, p, yaw, roll):
            offers.append(
                {
                    "change": f"pitch={p:.0f} at this position",
                    "move_to": {"xyz": target, "pitch": p, "yaw": yaw, "roll": roll},
                }
            )
            break
    for dz in POSE_ALT_Z:
        c = xyz + np.array([0.0, 0.0, dz])
        if ok(c, pitch, yaw, roll):
            offers.append(
                {
                    "change": f"z={c[2]:.3f} with this orientation",
                    "move_to": {
                        "xyz": [round(float(v), 4) for v in c],
                        "pitch": pitch,
                        "yaw": yaw,
                        "roll": roll,
                    },
                }
            )
            break
    for dy in POSE_ALT_YAW:
        if ok(xyz, pitch, yaw + dy, roll):
            offers.append(
                {
                    "change": f"yaw={motion._wrap(yaw + dy):.0f} at this position",
                    "move_to": {
                        "xyz": target,
                        "pitch": pitch,
                        "yaw": motion._wrap(yaw + dy),
                        "roll": roll,
                    },
                }
            )
            break
    waist = [float(v) for v in primitives.obs().waist]
    if len(waist) >= 5:
        for forward, up in ALT_CHEST:
            joints, _why = waist_ik(waist, forward, up, 0.0, 0.0)
            if joints is not None and ok(xyz, pitch, yaw, roll, (forward, up)):
                offers.append(
                    {
                        "change": f"move_waist(forward={forward}"
                        + (f", up={up}" if up else "")
                        + ") first, then this pose",
                        "move_waist": {"forward": forward, "up": up},
                        "move_to": {"xyz": target, "pitch": pitch, "yaw": yaw, "roll": roll},
                    }
                )
                break
    return offers


#: The fingertips reach about this far beyond the grasp point along the pointing direction.
FINGERTIP_M = 0.02
#: A surface measured within this horizontal distance is the one under a point.
SURFACE_RADIUS_M = 0.15
#: The fingertips may go this far below a measured surface (a press, a firm grasp).
SURFACE_SLACK_M = 0.005


def _surface_under(primitives: Primitives, xy: np.ndarray) -> float | None:
    best = None
    for sx, sy, sz in primitives.surfaces:
        d = math.hypot(sx - xy[0], sy - xy[1])
        if d <= SURFACE_RADIUS_M and (best is None or d < best[0]):
            best = (d, sz)
    return None if best is None else best[1]


def _below_surface(primitives: Primitives, centre: np.ndarray, rot: np.ndarray) -> str | None:
    """Why ``(centre, rot)`` puts the fingertips into a measured surface, or None."""

    tip = centre + FINGERTIP_M * rot[:, 2]
    surface = _surface_under(primitives, tip[:2])
    if surface is not None and tip[2] < surface - SURFACE_SLACK_M:
        return (
            f"the fingertips would go {100 * (surface - tip[2]):.1f} cm into the surface "
            f"measured at z={surface:.3f} under that point (the grasp point must stay about "
            f"{100 * FINGERTIP_M:.0f} cm above a surface it points down at)"
        )
    return None


def _half_turn(rot: np.ndarray) -> np.ndarray:
    """The same pointing with the fingers swapped: a half turn about the pointing axis."""

    return rot @ np.diag([-1.0, -1.0, 1.0])


def move_pose(
    primitives: Primitives,
    arm: str = "right",
    xyz: Sequence[float] | None = None,
    *,
    pitch: float | None = None,
    yaw: float | None = None,
    roll: float | None = None,
    grip: float | None = None,
    path: str = "auto",
    max_actions: int = MOVE_MAX_ACTIONS,
) -> dict[str, Any]:
    """Move one arm's grasp point to ``xyz`` with an orientation, solved here, sent as joints.

    The pose is solved first (the vendor IK, from the current joints and from the start
    joints); one missed by more than 2 cm / 8 degrees is refused where the arm stands, with
    nearby poses that are reachable.  ``path``: ``"line"`` follows a straight line of the
    grasp point (about 2.5 cm or 8 degrees per action) or refuses, ``"joint"`` swings the
    joints straight to the solution (0.15 rad per joint per action, the grasp point does not
    follow a line), ``"auto"`` the line where it can be followed, else the swing.
    """

    if arm not in ARMS:
        return {"error": "arm must be 'left' or 'right'"}
    if path not in ("auto", "line", "joint"):
        return {"error": "path must be 'auto', 'line' or 'joint'"}
    if not isinstance(max_actions, int) or max_actions < 1:
        return {"error": "max_actions must be a positive integer"}
    if not reach.available():
        return {"error": "the arm's IK (ik_solver) is not installed"}
    start_c, start_r = _pose_now(primitives, arm)
    goal_c = start_c if xyz is None else np.asarray(xyz, dtype=np.float64).reshape(-1)
    if goal_c.shape != (3,) or not np.all(np.isfinite(goal_c)):
        return {"error": "xyz must be three finite numbers"}
    now = motion.angles(start_r)
    asked = (pitch, yaw, roll)
    p, y, r = (float(v) if v is not None else now[k] for k, v in enumerate(asked))
    goal_r = start_r if all(v is None for v in asked) else motion.orientation(p, y, r)
    obs = primitives.obs()
    kin = motion.Arm(obs, arm)
    seeds = [kin.q] + (
        [np.asarray(primitives.start_joints[arm])] if primitives.start_joints.get(arm) else []
    )
    surface_why = _below_surface(primitives, goal_c, goal_r)
    # An empty open gripper's fingers are interchangeable: the pose turned half a turn about
    # the pointing direction is the same grasp, and the one nearer the arm's joints is taken.
    empty = (grip is None or float(grip) < GRIP_CLOSE - 0.1) and opening(obs, arm) < GRIP_OPEN * 0.8
    turned = None
    if empty and not all(v is None for v in asked):
        q_a, pe_a, re_a = kin.solve(goal_c, goal_r, kin.q)
        alt_r = _half_turn(goal_r)
        q_b, pe_b, re_b = kin.solve(goal_c, alt_r, kin.q)
        ok_a = pe_a <= motion.NEAR_M and re_a <= motion.NEAR_DEG
        ok_b = pe_b <= motion.NEAR_M and re_b <= motion.NEAR_DEG
        if ok_b and (
            not ok_a
            or float(np.max(np.abs(q_b - kin.q))) < float(np.max(np.abs(q_a - kin.q))) - 0.05
        ):
            goal_r, turned = (
                alt_r,
                "the fingers were turned half a turn (the same grasp, empty gripper)",
            )
    goal_q, pos_err, rot_err = motion.solve_goal(kin, goal_c, goal_r, seeds)
    # The path goes to the solution nearest the joints the arm has: another seed can find
    # the pose on another branch of the redundant arm, and a swing there leaves the line
    # (measured: from the same pose above an object, the descent swung within 1.2 cm of the
    # line to the current-joints solution and was refused for the start-joints one).
    near_q, near_pe, near_re = kin.solve(goal_c, goal_r, kin.q)
    if (
        near_pe <= max(pos_err, motion.REACH_M) + 0.002
        and near_re <= max(rot_err, motion.REACH_DEG) + 1.0
    ):
        goal_q, pos_err, rot_err = near_q, near_pe, near_re

    def result(**fields: Any) -> dict[str, Any]:
        c, rot = _pose_now(primitives, arm)
        dist = float(np.linalg.norm(goal_c - c))
        rerr = motion.rotation_error(rot, goal_r)
        out = {
            "arm": arm,
            "target_tool_centre": [round(float(v), 4) for v in goal_c],
            "final_tool_centre": [round(float(v), 4) for v in c],
            "dist_to_target_m": round(dist, 4),
            "rotation_error_deg": round(rerr, 1),
            "pointing_error_deg": round(angle_between(rot[:, 2], goal_r[:, 2]), 1),
            "angles": dict(zip(("pitch", "yaw", "roll"), motion.angles(rot), strict=False)),
            "reached": bool(dist <= POSE_ARRIVE_M and rerr <= ARRIVE_DEG),
            "opening": round(opening(primitives.obs(), arm), 3),
            "steps_planned": 0,
            "truncated": False,
            "alternatives": [],
            "reach_hint": None,
            "blocked": False,
            "stop_reason": None,
            "note": None,
            "path": None,
        }
        out.update(fields)
        if out["reached"]:
            out["stop_reason"] = None
        return out

    if surface_why is not None:
        return result(
            steps_used=0, stop_reason="out_of_reach", reach_hint=f"not attempted: {surface_why}."
        )
    if pos_err > motion.NEAR_M or rot_err > motion.NEAR_DEG:
        offers = _pose_offers(primitives, kin, arm, goal_c, p, y, r, seeds)
        hint = (
            f"not attempted: out of reach — the closest pose this arm can take is "
            f"{pos_err * 100:.1f} cm and {rot_err:.0f} deg away. "
            + (
                "Reachable instead: " + "; ".join(o["change"] for o in offers) + "."
                if offers
                else "No nearby pitch, height, heading or waist lean reaches it: "
                "use the other arm, or bring the object closer."
            )
        )
        return result(
            steps_used=0,
            stop_reason="out_of_reach",
            reach_hint=hint,
            alternatives=offers,
            ik_error={"pos_m": round(pos_err, 4), "rot_deg": round(rot_err, 1)},
        )
    note = turned
    if pos_err > motion.REACH_M or rot_err > motion.REACH_DEG:
        note = (note + "; " if note else "") + (
            f"the exact pose is out of reach; went to the closest one, "
            f"{pos_err * 100:.1f} cm and {rot_err:.0f} deg from it"
        )
        goal_c, goal_r = kin.tool(goal_q)

    waypoints, used = None, path
    if path in ("line", "auto"):
        waypoints, why = motion.line_path(kin, goal_c, goal_r)
        if waypoints is None:
            # the arm's current joint arrangement cannot follow the line: take the goal's
            # arrangement where the arm stands, then the line (planned from the goal back)
            waypoints = motion.rearranged_line(kin, goal_c, goal_r, goal_q)
            if waypoints is not None:
                note = (note + "; " if note else "") + (
                    "rearranged the arm's joints in place first (its seventh joint), then "
                    "followed the line"
                )
        if waypoints is None:
            # the line's IK chain broke, but a swing may stay near the line anyway
            waypoints = motion.swing_near_line(kin, goal_c, goal_r, goal_q)
            if waypoints is not None:
                note = (note + "; " if note else "") + (
                    "the straight line needed the arm to rearrange its joints: swung there "
                    "within 1.5 cm of the line"
                )
        if waypoints is None and path == "line":
            return result(
                steps_used=0,
                stop_reason="out_of_reach",
                path="line",
                reach_hint=f"not attempted: {why} (the pose itself is reachable: "
                "path='auto' swings the joints there)",
            )
        used = "line" if waypoints is not None else "joint"
    if waypoints is None:
        waypoints = motion.joint_path(kin.q, goal_q)
    planned = len(waypoints)
    if planned > max_actions:
        waypoints = waypoints[:max_actions]

    spent, resisted_run, resisted_total = 0, 0, 0
    stop = None
    for q in waypoints:
        primitives.act(joint_action(primitives, arm, q, grip))
        spent += 1
        slow = primitives.last_act_s >= RESISTED_S
        resisted_total += int(slow)
        resisted_run = resisted_run + 1 if slow else 0
        if resisted_run >= STALL_ACTIONS:
            stop = "blocked"
            break
    # Close the last gap from what the joints measure: re-solve and re-send the goal.
    if stop is None:
        for _ in range(min(SETTLE_ACTIONS, max(0, max_actions - spent))):
            c, rot = _pose_now(primitives, arm)
            if (
                float(np.linalg.norm(goal_c - c)) <= POSE_ARRIVE_M * 0.5
                and motion.rotation_error(rot, goal_r) <= ARRIVE_DEG * 0.5
            ):
                break
            kin = motion.Arm(primitives.obs(), arm)
            q, _, _ = kin.solve(goal_c, goal_r, kin.q)
            primitives.act(joint_action(primitives, arm, q, grip))
            spent += 1
            slow = primitives.last_act_s >= RESISTED_S
            resisted_total += int(slow)
            resisted_run = resisted_run + 1 if slow else 0
            if resisted_run >= STALL_ACTIONS:
                stop = "blocked"
                break
    out = result(
        steps_used=spent,
        steps_planned=planned,
        path=used,
        note=note,
        resisted_actions=resisted_total,
        truncated=bool(planned > max_actions),
    )
    if not out["reached"]:
        out["stop_reason"] = stop or (
            "budget" if planned > max_actions else "blocked" if resisted_total else "tracking"
        )
        out["blocked"] = out["stop_reason"] == "blocked"
    return out


def move_both_pose(
    primitives: Primitives,
    left: Sequence[float] | None = None,
    right: Sequence[float] | None = None,
    *,
    grip: float | None = None,
    max_actions: int = MOVE_MAX_ACTIONS,
) -> dict[str, Any]:
    """Move both grasp points together along straight lines, each keeping its orientation.

    Both lines are cut into the same number of waypoints, so the hands start and arrive
    together and keep their offset — for an object carried in two hands.  An arm given
    None holds.  Refused, with nothing moved, when either line leaves its arm's reach.
    """

    goals = {"left": left, "right": right}
    plans: dict[str, list[np.ndarray]] = {}
    for arm, xyz in goals.items():
        if xyz is None:
            continue
        c = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if c.shape != (3,) or not np.all(np.isfinite(c)):
            return {"error": f"{arm} must be three finite numbers or None"}
        kin = motion.Arm(primitives.obs(), arm)
        _, rot = _pose_now(primitives, arm)
        waypoints, why = motion.line_path(kin, c, rot)
        if waypoints is None:
            return {
                "reached": False,
                "steps_used": 0,
                "stop_reason": "out_of_reach",
                "reach_hint": f"not attempted: the {arm} arm's {why}",
            }
        plans[arm] = waypoints
    if not plans:
        return {"error": "give left, right or both"}
    n = max(len(w) for w in plans.values())
    for arm, w in plans.items():  # resample both to n waypoints, in step
        idx = np.linspace(0, len(w) - 1, n)
        plans[arm] = [w[int(round(i))] for i in idx]
    spent, resisted_run = 0, 0
    stop = None
    for k in range(min(n, max_actions)):
        obs = primitives.obs()
        q = {
            arm: (plans[arm][k] if arm in plans else held_joints(primitives, arm, obs))
            for arm in ARMS
        }
        for arm in ARMS:
            primitives.joint_commands[arm] = [float(v) for v in q[arm]]
        grips = {
            arm: (
                float(grip)
                if grip is not None and arm in plans
                else commanded_grip(primitives, arm, obs)
            )
            for arm in ARMS
        }
        primitives.grip_commands.update(grips)
        primitives.act(
            {
                "kind": "JOINT_ABS",
                "left_arm": [float(v) for v in q["left"]],
                "right_arm": [float(v) for v in q["right"]],
                "left_gripper": grips["left"],
                "right_gripper": grips["right"],
            }
        )
        spent += 1
        resisted_run = resisted_run + 1 if primitives.last_act_s >= RESISTED_S else 0
        if resisted_run >= STALL_ACTIONS:
            stop = "blocked"
            break
    out: dict[str, Any] = {"steps_used": spent}
    reached = True
    for arm in plans:
        c, _ = _pose_now(primitives, arm)
        dist = float(np.linalg.norm(np.asarray(goals[arm], dtype=np.float64) - c))
        ok = dist <= POSE_ARRIVE_M
        reached = reached and ok
        out[arm] = {
            "final_tool_centre": [round(float(v), 4) for v in c],
            "dist_to_target_m": round(dist, 4),
            "reached": ok,
            "stop_reason": None if ok else (stop or "tracking"),
        }
    out["reached"] = reached
    return out


#: The benchmark runs its task checker once every this many native actions
#: (``PiEnv.step``: ``current_step % 30 == 0``).
CHECK_PERIOD = 30


def wait_for_check(primitives: Primitives, checks: int = 1) -> dict[str, Any]:
    """Hold everything still until the benchmark's stage check has run ``checks`` times."""

    if not isinstance(checks, int) or not 1 <= checks <= 5:
        return {"error": "checks must be an integer from 1 to 5"}
    spent = 0
    for _ in range(checks):
        spent += hold_until_check(primitives)
    status = primitives.status()
    return {
        "steps_used": spent,
        "checks": checks,
        "status": {k: status[k] for k in ("step", "step_limit") if k in status},
    }


def hold_until_check(primitives: Primitives) -> int:
    """Hold both arms and both grippers still until the benchmark's checker has run once.

    Returns the actions spent (at most :data:`CHECK_PERIOD`).  The checker only looks every
    :data:`CHECK_PERIOD` actions, and a task stage starts — and records its starting state —
    at the check where the stage before it passed.  A pick-up stage that starts after the
    object is already in the air measures the lift from the air, sees none, and fails; so
    after a grasp closes, the scene is held until a check has seen it, and only then lifted.

    The step counter here counts the benchmark's requests, one more than the actions it has
    taken (the first request comes before any action), so the check at action ``k * 30``
    has happened once the counter passes ``k * 30 + 1``.
    """

    status = primitives.status()
    try:
        step = int(status.get("step"))
    except (TypeError, ValueError):
        return 0
    until = ((step - 1) // CHECK_PERIOD + 1) * CHECK_PERIOD + 1
    spent = 0
    while spent <= CHECK_PERIOD:
        try:
            if int(primitives.status().get("step") or 0) >= until:
                break
        except (TypeError, ValueError):
            break
        obs = primitives.obs()
        if len(obs.arm_joints) < 14:
            break
        measured = list(obs.gripper) + [0.0, 0.0]
        primitives.act(
            {
                "kind": "JOINT_ABS",
                "left_arm": held_joints(primitives, "left", obs),
                "right_arm": held_joints(primitives, "right", obs),
                "left_gripper": float(primitives.grip_commands.get("left", measured[0])),
                "right_gripper": float(primitives.grip_commands.get("right", measured[1])),
            }
        )
        spent += 1
    return spent


def grip(
    primitives: Primitives, arm: str = "right", close: bool = True, wait_for_check: bool = True
) -> dict[str, Any]:
    """Close or open one gripper fully, and report the opening the fingers reached.

    After a close that holds something, ``wait_for_check`` keeps the scene still until the
    benchmark's checker has run (:func:`hold_until_check`), so a lift that follows is
    measured from the object resting on the table.
    """

    if arm not in ARMS:
        return {"error": "arm must be 'left' or 'right'"}
    command = GRIP_CLOSE if close else GRIP_OPEN
    before = opening(primitives.obs(), arm)
    measured, last, spent = before, before, 0
    for _ in range(GRIP_MAX_ACTIONS):
        primitives.act(
            joint_action(primitives, arm, held_joints(primitives, arm, primitives.obs()), command)
        )
        spent += 1
        measured = opening(primitives.obs(), arm)
        if spent >= 2 and abs(measured - last) < GRIP_STILL:
            break
        last = measured
    # Open is negative: a close that stops short of 0 has something between the fingers.
    holding = bool(close and measured < -HOLD_MIN_GAP)
    waited = hold_until_check(primitives) if (holding and wait_for_check) else 0
    status = primitives.status()
    return {
        "arm": arm,
        "commanded": "closed" if close else "open",
        "opening": round(measured, 3),
        "opening_before": round(before, 3),
        "holding": holding,
        "steps_used": spent + waited,
        "steps_waiting_for_check": waited,
        "status": {k: status[k] for k in ("step", "step_limit") if k in status},
    }


# ---------------------------------------------------------------------------
# the waist
# ---------------------------------------------------------------------------

#: The G2's torso chain, from its URDF (``G2_omnipicker.urdf``): three joints pitching about
#: base_link y stacked like knee, hip and waist, then a roll and a yaw.  Pitch is positive
#: forward (+x).  Measured in the simulator, +0.1 rad on each pitch joint moves both hands
#: forward/down by (+9.3/-6.2), (+5.0/-7.1), (+1.7/-5.7) cm, which these numbers reproduce.
WAIST_PIVOT_XZ = (0.102, 0.144)  # idx01 in base_link
WAIST_LINKS = (0.36, 0.38)  # idx01 -> idx02 -> idx03
WAIST_LIMITS = ((-1.06, 0.0), (0.0, 2.65), (-1.91, 1.57), (-0.43, 0.43), (-3.05, 3.05))
#: Largest joint change commanded per action while the waist moves (radians): the joints
#: track a 0.1 rad step in about 8 actions, so this keeps the hands from swinging.
WAIST_STEP_RAD = 0.04
WAIST_TOL_RAD = 0.01
WAIST_MAX_ACTIONS = 60


def waist_chest(joints: Sequence[float]) -> tuple[float, float, float]:
    """The chest pivot (idx03) in base_link (x, z) and the torso's total pitch (rad)."""

    t1, t2, t3 = (float(v) for v in joints[:3])
    l1, l2 = WAIST_LINKS
    x = WAIST_PIVOT_XZ[0] + l1 * math.sin(t1) + l2 * math.sin(t1 + t2)
    z = WAIST_PIVOT_XZ[1] + l1 * math.cos(t1) + l2 * math.cos(t1 + t2)
    return x, z, t1 + t2 + t3


def waist_ik(
    current: Sequence[float],
    forward: float = 0.0,
    up: float = 0.0,
    turn_deg: float = 0.0,
    lean_deg: float = 0.0,
) -> tuple[list[float] | None, str | None]:
    """Waist joints that move the chest by (forward, up) metres, keeping the torso's pitch
    (plus ``lean_deg``), and turn the upper body by ``turn_deg``.  ``(None, why)`` when
    the pose is outside the joints' range."""

    x, z, pitch = waist_chest(current)
    tx, tz = x + float(forward), z + float(up)
    pitch += math.radians(float(lean_deg))
    l1, l2 = WAIST_LINKS
    vx, vz = tx - WAIST_PIVOT_XZ[0], tz - WAIST_PIVOT_XZ[1]
    d2 = vx * vx + vz * vz
    c2 = (d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if not -1.0 <= c2 <= 1.0:
        return None, "the chest cannot reach that far (the legs would have to stretch)"
    t2 = math.acos(c2)  # the knee bends forward only (idx02 >= 0)
    t1 = math.atan2(vx, vz) - math.atan2(l2 * math.sin(t2), l1 + l2 * math.cos(t2))
    t3 = pitch - t1 - t2
    joints = [t1, t2, t3, float(current[3]), float(current[4]) + math.radians(float(turn_deg))]
    names = ("knee", "hip", "waist pitch", "waist roll", "waist turn")
    for value, (lo, hi), name in zip(joints, WAIST_LIMITS, names, strict=False):
        if not lo - 1e-6 <= value <= hi + 1e-6:
            return (
                None,
                f"{name} joint would leave its range "
                f"({value:+.2f} rad, range {lo:+.2f}..{hi:+.2f})",
            )
    return joints, None


def move_waist(
    primitives: Primitives,
    forward: float = 0.0,
    up: float = 0.0,
    turn_deg: float = 0.0,
    lean_deg: float = 0.0,
    reset: bool = False,
    max_actions: int = WAIST_MAX_ACTIONS,
) -> dict[str, Any]:
    """Move the chest (and with it both arms, as they are) by a few centimetres, or turn it.

    The arms keep their joint angles and the grippers their last command, so the hands and
    anything held ride along with the torso.  ``reset`` returns the waist to the angles the
    episode started with.
    """

    obs = primitives.obs()
    current = [float(v) for v in obs.waist]
    if len(current) < 5:
        return {"error": "this robot reports no waist joints"}
    if reset:
        if not primitives.start_waist:
            return {"error": "the start waist was not recorded"}
        target, why = list(primitives.start_waist), None
    else:
        target, why = waist_ik(current, forward, up, turn_deg, lean_deg)
    before_x, before_z, _ = waist_chest(current)
    if target is None:
        return {
            "moved": False,
            "error_reason": why,
            "steps_used": 0,
            "chest_xz": [round(before_x, 4), round(before_z, 4)],
        }
    arms = held_joints(primitives, "left", obs) + held_joints(primitives, "right", obs)
    measured = list(obs.gripper) + [0.0, 0.0]
    grips = (
        float(primitives.grip_commands.get("left", measured[0])),
        float(primitives.grip_commands.get("right", measured[1])),
    )
    spent = 0
    command = list(current)
    while spent < max_actions:
        now = [float(v) for v in primitives.obs().waist]
        error = max(abs(a - b) for a, b in zip(now, target, strict=False))
        if error <= WAIST_TOL_RAD:
            break
        command = [
            c + max(-WAIST_STEP_RAD, min(WAIST_STEP_RAD, t - c))
            for c, t in zip(command, target, strict=False)
        ]
        primitives.act(
            {
                "kind": "JOINT_ABS",
                "left_arm": arms[:7],
                "right_arm": arms[7:14],
                "left_gripper": grips[0],
                "right_gripper": grips[1],
                "waist": command,
            }
        )
        spent += 1
    final = [float(v) for v in primitives.obs().waist]
    x, z, pitch = waist_chest(final)
    error = max(abs(a - b) for a, b in zip(final, target, strict=False))
    return {
        "moved": True,
        "reached": bool(error <= WAIST_TOL_RAD * 3),
        "chest_moved_m": [round(x - before_x, 4), round(z - before_z, 4)],
        "chest_xz": [round(x, 4), round(z, 4)],
        "torso_pitch_deg": round(math.degrees(pitch), 1),
        "turn_deg": round(math.degrees(final[4]), 1),
        "max_joint_error_rad": round(error, 4),
        "steps_used": spent,
    }


#: ``home`` first lifts a grasp point lower than this, by at most HOME_LIFT_M.
HOME_CLEAR_Z = 1.0
HOME_LIFT_M = 0.06


def home(
    primitives: Primitives,
    arm: str = "both",
    max_actions: int = 25,
    tol: float = 0.02,
) -> dict[str, Any]:
    """Return one arm or both to the joint angles the episode started with."""

    if arm not in (*ARMS, "both"):
        return {"error": "arm must be 'left', 'right' or 'both'"}
    if not primitives.start_joints:
        return {"error": "the start pose was not recorded"}
    arms = ARMS if arm == "both" else (arm,)
    obs = primitives.obs()
    if len(obs.arm_joints) < 14:
        return {"error": "the observation has no arm joints"}
    spent = 0
    # A grasp point low over the table first rises straight up (up to HOME_LIFT_M), so the
    # swing back starts clear of what it was working on — the swing plans no path.
    for name in arms:
        centre, _ = _pose_now(primitives, name)
        if centre[2] < HOME_CLEAR_Z:
            lift = min(HOME_LIFT_M, HOME_CLEAR_Z - float(centre[2]))
            up = move_pose(
                primitives,
                name,
                (centre + np.array([0.0, 0.0, lift])).tolist(),
                path="line",
                max_actions=6,
            )
            spent += int(up.get("steps_used", 0) or 0)
    error = float("inf")
    for _ in range(max_actions):
        obs = primitives.obs()
        current = [float(v) for v in obs.arm_joints]
        left = held_joints(primitives, "left", obs) if "left" not in arms else list(current[:7])
        right = (
            held_joints(primitives, "right", obs) if "right" not in arms else list(current[7:14])
        )
        errors = []
        for name in arms:
            target = primitives.start_joints[name]
            got = left if name == "left" else right
            errors.extend(abs(a - b) for a, b in zip(got, target, strict=True))
            if name == "left":
                left = list(target)
            else:
                right = list(target)
        error = max(errors)
        if error <= tol:
            break
        primitives.joint_commands["left"], primitives.joint_commands["right"] = (
            list(left),
            list(right),
        )
        primitives.act(
            {
                "kind": "JOINT_ABS",
                "left_arm": left,
                "right_arm": right,
                "left_gripper": commanded_grip(primitives, "left", obs),
                "right_gripper": commanded_grip(primitives, "right", obs),
            }
        )
        spent += 1
    obs = primitives.obs()
    current = [float(v) for v in obs.arm_joints]
    errors = []
    for name in arms:
        got = current[:7] if name == "left" else current[7:14]
        errors.extend(abs(a - b) for a, b in zip(got, primitives.start_joints[name], strict=True))
    error = max(errors) if errors else float("nan")
    return {
        "arm": arm,
        "reached": bool(error <= tol),
        "max_joint_error_rad": round(float(error), 4),
        "steps_used": spent,
        "truncated": bool(spent >= max_actions and error > tol),
    }


__all__ = [
    "ARMS",
    "ARRIVE_DEG",
    "ARRIVE_M",
    "GRIP_CLOSE",
    "GRIP_OPEN",
    "MOVE_MAX_ACTIONS",
    "POINTS",
    "Primitives",
    "TOOL_OFFSET_M",
    "action",
    "angle_between",
    "flange_target",
    "grip",
    "home",
    "matrix_quat_xyzw",
    "move_base",
    "move_both_pose",
    "wait_for_check",
    "move_pose",
    "move_waist",
    "named_direction",
    "torso_yaw",
    "waist_chest",
    "waist_ik",
    "opening",
    "quat_xyzw_matrix",
    "tool_centre",
    "unit",
]
