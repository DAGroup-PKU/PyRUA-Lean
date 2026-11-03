# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""G2 arm motion solved in the library and commanded as joint angles.

The RoboDojo primitives solve each arm's inverse kinematics for the whole pose before moving,
refuse a pose out of reach, and send joint targets along a planned path.  This is the same for
the G2.  Sending ``EEF_ABS`` instead hands the benchmark's relaxed IK one small target at a
time: it advances about 1 cm per action and can settle in a local minimum short of a pose the
arm can take (measured: 5.8 cm short of a target 15 cm forward and 5 cm down).  Here the same
solver (``robocoliseum_reach``: the vendor ``ik_solver`` on ``G2_NO_GRIPPER.urdf``) is asked
for the goal, the path is planned in the library, and ``JOINT_ABS`` actions follow it; the
benchmark then waits for each action's joints to settle (``PiEnv._wait_arm_settled``, 0.01 rad,
1 s).

Orientation is three angles in base_link, as on RoboDojo, in degrees:

- ``pitch``: 0 points level, 90 straight down;
- ``yaw``: the heading of the pointing direction, 0 along base_link +x (away from the
  robot), +90 towards the robot's left (+y);
- ``roll``: the fingers' turn about the pointing direction; at 0 they close horizontally.

Pointing straight down, the fingers close along the line at ``yaw + 90`` degrees from +x:
``yaw`` equal to an object's long-axis heading closes them across it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import robocoliseum_reach as reach

#: A straight-line path is cut into waypoints at most this far / this much turned apart.
LINE_STEP_M = 0.025
LINE_STEP_DEG = 8.0
#: A joint swing moves each joint at most this much per action.
JOINT_STEP_RAD = 0.15
#: A waypoint the solver cannot take within this is off the line; a jump of the joints
#: larger than this between two waypoints means the line passes a singularity or a flip.
LINE_TOL_M = 0.004
LINE_TOL_DEG = 2.0
LINE_JUMP_RAD = 0.35
#: Solves at most per waypoint with the seed-near solver (each moves a joint ~0.04 rad).
NEAR_SOLVES_MAX = 60
#: A pose the solver reaches within this is reachable.
REACH_M = 0.005
REACH_DEG = 2.0
#: A pose missed by at most this much is still driven to (the closest pose, with a note).
NEAR_M = 0.02
NEAR_DEG = 8.0

#: The end link's axes in the canonical frame of the three angles (pointing = canonical +x,
#: the line the fingers close along = canonical +y).  On the G2 the end link points along +z
#: and its fingers close along its +x: the URDF mounts the gripper base turned 90 degrees
#: about z (``idx91_ee_r_joint`` / ``idx51_ee_l_joint``) and puts the two fingers at
#: y = -/+0.0195 m of that base, i.e. along the end link's x.  Columns: end-link x, y, z.
_TOOL_IN_CANON = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def orientation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """The G2 end-link rotation (columns: x = the finger line, y, z = pointing) for the
    angles, degrees."""

    canon = _rz(math.radians(yaw)) @ _ry(math.radians(pitch)) @ _rx(math.radians(roll))
    return canon @ _TOOL_IN_CANON


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def angles(rot: np.ndarray) -> tuple[float, float, float]:
    """``(pitch, yaw, roll)`` of a G2 tool rotation; roll 0 within a degree of vertical."""

    canon = rot @ _TOOL_IN_CANON.T
    point = canon[:, 0]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    if math.hypot(point[0], point[1]) < math.sin(math.radians(1.0)):
        # heading and roll turn about the same axis: report it all as the heading
        jaw = canon[:, 1]
        yaw = math.degrees(math.atan2(-jaw[0], jaw[1]))
        return round(pitch, 1), round(_wrap(yaw), 1), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0]))
    rest = orientation(pitch, yaw, 0.0).T @ rot
    rest = _TOOL_IN_CANON @ rest @ _TOOL_IN_CANON.T
    roll = math.degrees(math.atan2(rest[2, 1], rest[1, 1]))
    return round(pitch, 1), round(_wrap(yaw), 1), round(_wrap(roll), 1)


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    c = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([a, b])))([t]).as_matrix()[0]


class Arm:
    """One arm's kinematics now: the arm base where the waist puts it, FK, and IK.

    ``base`` is ``T_base_link<-arm_base_link`` from the observation (the benchmark reports
    each flange in both frames); the arm chain itself is fixed.
    """

    def __init__(self, obs: Any, arm: str) -> None:
        self.arm = arm
        self.base = reach.base_from_arm_base(obs)
        self.inv_base = np.linalg.inv(self.base)
        joints = [float(v) for v in obs.arm_joints]
        self.q = np.asarray(joints[:7] if arm == "left" else joints[7:14], dtype=np.float64)
        self.solver = reach._solver(arm, reach.GOAL_CONFIG)
        #: The benchmark's own configuration, with its velocity cost: a solve stays near its
        #: seed, so consecutive waypoints of a line get consecutive joint angles.  The goal
        #: solver has no such cost, and on a redundant 7-joint arm it can answer each
        #: waypoint with another branch (measured: every waypoint of a 12 cm descent pointing
        #: down was reachable, yet consecutive goal solutions jumped by more than 0.35 rad
        #: and the line was refused as a flip).
        self.tracker = reach._solver(arm, reach.STEP_CONFIG)

    def flange(self, q: Sequence[float]) -> np.ndarray:
        return self.base @ np.asarray(
            self.solver.compute_fk(np.asarray(q, dtype=np.float64)), dtype=np.float64
        ).reshape(4, 4)

    def tool(self, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        from .robocoliseum_tools import tool_centre

        t = self.flange(q)
        return tool_centre(t[:3, 3], t[:3, :3]), t[:3, :3]

    def solve(
        self,
        centre: Sequence[float],
        rot: np.ndarray,
        seed: Sequence[float],
        solves: int = 4,
        near: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Joint angles taking the tool centre to ``(centre, rot)`` from ``seed``, and the
        residual (metres, degrees) of the best the solver found.  ``near`` uses the solver
        that stays close to the seed (for the waypoints of a path)."""

        from scipy.spatial.transform import Rotation

        from .robocoliseum_tools import flange_target

        target = np.eye(4)
        target[:3, :3] = rot
        target[:3, 3] = flange_target(np.asarray(centre, dtype=np.float64), rot)
        local = self.inv_base @ target
        solver = self.tracker if near else self.solver
        quat = Rotation.from_matrix(local[:3, :3]).as_quat()
        with reach._LOCK:
            q = np.asarray(seed, dtype=np.float64)
            # The near solver moves each joint at most about 0.04 rad per solve (its velocity
            # cost), so it is iterated, re-seeded from its own answer as the benchmark does
            # per action, until it stops gaining; the goal solver converges in a few.
            for k in range(NEAR_SOLVES_MAX if near else solves):
                solver.sync_target_with_joints(q)
                solver.update_target_quat(local[:3, 3], quat)
                q_next = np.asarray(solver.solve(), dtype=np.float64)
                moved = float(np.max(np.abs(q_next - q)))
                q = q_next
                if near and k >= solves and moved < 1e-4:
                    break
        got_c, got_r = self.tool(q)
        return q, float(np.linalg.norm(got_c - np.asarray(centre))), rotation_error(got_r, rot)


def solve_goal(
    arm: Arm, centre: Sequence[float], rot: np.ndarray, seeds: Sequence[Sequence[float]]
) -> tuple[np.ndarray, float, float]:
    """The best of several seeds (the current joints first)."""

    best = None
    for seed in seeds:
        q, pe, re = arm.solve(centre, rot, seed)
        score = pe + re * 0.002  # a degree weighs like 2 mm
        if best is None or score < best[0]:
            best = (score, q, pe, re)
        if pe <= REACH_M and re <= REACH_DEG:
            break
    assert best is not None
    return best[1], best[2], best[3]


def line_path(
    arm: Arm, goal_centre: np.ndarray, goal_rot: np.ndarray
) -> tuple[list[np.ndarray] | None, str | None]:
    """Joint waypoints that move the tool centre along a straight line (orientation slerped)
    to the goal, or ``(None, why)`` where the line cannot be followed."""

    start_c, start_r = arm.tool(arm.q)
    dist = float(np.linalg.norm(goal_centre - start_c))
    turn = rotation_error(start_r, goal_rot)
    n = max(
        1, int(math.ceil(dist / LINE_STEP_M - 1e-9)), int(math.ceil(turn / LINE_STEP_DEG - 1e-9))
    )
    path, q = [], arm.q
    for i in range(1, n + 1):
        t = i / n
        c = (1 - t) * start_c + t * goal_centre
        r = _slerp(start_r, goal_rot, t) if turn > 1e-6 else goal_rot
        # Near the last waypoint's joints first (the seed-near solver), then refined by the
        # exact solver from there: started that close, it converges on the same branch.
        q_near, _, _ = arm.solve(c, r, q, solves=6, near=True)
        q_next, pe, re = arm.solve(c, r, q_near, solves=2)
        if pe > LINE_TOL_M or re > LINE_TOL_DEG:
            return None, f"the line leaves the arm's reach {t * dist * 100:.0f} cm along"
        if float(np.max(np.abs(q_next - q))) > LINE_JUMP_RAD:
            return None, "the line passes a pose where the arm would have to flip"
        path.append(q_next)
        q = q_next
    return path, None


#: A joint swing whose grasp point stays this close to the straight line (and orientation
#: this close to the goal's) is as good as the line.  Measured: a 12 cm descent pointing
#: down whose IK chain broke (the arm had to redistribute joints 1, 3 and 5 by 0.4-0.5 rad)
#: swung there in 4 actions within 1.2 cm and 2 degrees of the line.
SWING_LINE_DEV_M = 0.015
SWING_LINE_DEV_DEG = 10.0


def swing_near_line(
    arm: Arm, goal_centre: np.ndarray, goal_rot: np.ndarray, goal_q: np.ndarray
) -> list[np.ndarray] | None:
    """The joint swing to ``goal_q`` when its grasp point stays near the straight line."""

    start_c, start_r = arm.tool(arm.q)
    seg = goal_centre - start_c
    length2 = float(seg @ seg)
    for qi in joint_path(arm.q, goal_q, step=0.02):
        ci, ri = arm.tool(qi)
        t = 0.0 if length2 < 1e-12 else min(1.0, max(0.0, float((ci - start_c) @ seg) / length2))
        if float(np.linalg.norm(ci - (start_c + t * seg))) > SWING_LINE_DEV_M:
            return None
        if min(rotation_error(ri, goal_rot), rotation_error(ri, start_r)) > SWING_LINE_DEV_DEG:
            return None
    return joint_path(arm.q, goal_q)


#: Rearranging the joints in place before a line (the arm has seven joints, one more than a
#: pose needs) may move the grasp point at most this much on the way.
REARRANGE_DEV_M = 0.02
REARRANGE_DEV_DEG = 8.0


def rearranged_line(
    arm: Arm, goal_centre: np.ndarray, goal_rot: np.ndarray, goal_q: np.ndarray
) -> list[np.ndarray] | None:
    """A line the current joint arrangement cannot follow, planned from the goal backwards.

    From the goal's joints the line is followed back to the start pose, which gives the start
    pose in the goal's arrangement; the arm first swings into that arrangement where it is
    (the grasp point moving at most REARRANGE_DEV_M), then follows the line.  Measured: from
    above an object, pointing down, the descent needed the wrist pitch past its limit in the
    arm's current arrangement; the goal's arrangement differs by 0.4-0.6 rad on joints 1, 4
    and 5 and follows the same line without a jump.
    """

    start_c, start_r = arm.tool(arm.q)
    back = Arm.__new__(Arm)
    back.__dict__.update(arm.__dict__)
    back.q = np.asarray(goal_q, dtype=np.float64)
    reverse, why = line_path(back, start_c, start_r)
    if reverse is None:
        return None
    start_alt = reverse[-1]
    for qi in joint_path(arm.q, start_alt, step=0.02):
        ci, ri = arm.tool(qi)
        if (
            float(np.linalg.norm(ci - start_c)) > REARRANGE_DEV_M
            or rotation_error(ri, start_r) > REARRANGE_DEV_DEG
        ):
            return None
    forward = list(reversed(reverse[:-1])) + [np.asarray(goal_q, dtype=np.float64)]
    return joint_path(arm.q, start_alt) + forward


def joint_path(q0: np.ndarray, q1: np.ndarray, step: float = JOINT_STEP_RAD) -> list[np.ndarray]:
    n = max(1, int(math.ceil(float(np.max(np.abs(q1 - q0))) / step - 1e-9)))
    return [q0 + (q1 - q0) * (i / n) for i in range(1, n + 1)]


__all__ = [
    "Arm",
    "JOINT_STEP_RAD",
    "LINE_STEP_DEG",
    "LINE_STEP_M",
    "NEAR_DEG",
    "NEAR_M",
    "REACH_DEG",
    "REACH_M",
    "angles",
    "joint_path",
    "line_path",
    "orientation",
    "rotation_error",
    "solve_goal",
    "rearranged_line",
    "swing_near_line",
]
