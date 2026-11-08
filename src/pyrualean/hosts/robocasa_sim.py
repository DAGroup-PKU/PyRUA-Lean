# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""A RoboCasa365 kitchen in a process of its own, with the robot's motion primitives.

The host of ``--robot robocasa_rgb`` (:mod:`.robocasa_rgb`) runs in the library's interpreter
(SAM3, the agent runtimes); RoboCasa, robosuite and MuJoCo run in RoboCasa's own (Python 3.10).
This file is that second process: run it as a script with RoboCasa's interpreter.  It imports
nothing of this package, and nothing but the standard library and numpy at module level, so the
host may import its codec (:func:`pack` / :func:`unpack`) too.

The kitchen is built as RPent builds it for RoboCasa365's Target50 evaluation (the
``PandaOmron`` robot, its default composite controller, the ``target`` split's ten layouts, the
cell's seed) and is never changed: the primitives send the benchmark's own 12-number action,
one ``env.step`` per native action.  The benchmark's rules are kept here as well:

*   the episode's budget is RoboCasa's official horizon of the task
    (``robocasa.utils.dataset_registry_utils.get_task_horizon``) in native actions;
*   the task's own ``_check_success`` is evaluated after every native action and the episode
    ends the moment it holds.  Nothing that reaches the policy says so: the state only reports
    that the episode is over (``over``), as RoboDojo's client does.  The verdict goes to the
    host alone (op ``verdict``).

Frames.  Everything the policy sees is in the ROBOT frame: x to the robot's right, y ahead of
it, z up from the floor (world z); metres.  Its origin is on the floor under the base's turning
axis, so it moves with the base; ``base`` in the state is where the base stands in the frame
it had when the episode started (its odometry).  Orientations of the gripper are RoboDojo's:
pitch below the horizontal, yaw of the pointing direction (0 straight ahead, +90 to the
robot's left), roll about the pointing direction, in degrees.  Cameras are reported in the
OpenCV convention (x right, y down, z ahead of the lens) with intrinsics for the rendered size.

Motion runs the benchmark's operational-space controller (``OSC_POSE``, delta input): each
native action asks for the step from where the grasp point IS to the next waypoint, so the arm
follows straight lines closely and holds still in between (a zero action lets it sag:
1.7 cm over 40 actions).  Plans are checked on a copy of the robot's own MuJoCo model, with the
controller's own redundancy resolution (:meth:`Kitchen._osc_joints`): inverse kinematics says
whether a pose is in reach, the emulated controller which way the joints will go and whether
one would meet its stop.  Neither knows anything of the kitchen around the robot, so contact
shows only as the arm no longer getting nearer its waypoint (``blocked``).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: The three robot cameras of RoboCasa's datasets and evaluation helper, by the policy's names.
CAMERAS = {
    "agentview_left": "robot0_agentview_left",
    "agentview_right": "robot0_agentview_right",
    "eye_in_hand": "robot0_eye_in_hand",
}
WIDTH, HEIGHT = 640, 480
#: RPent's Target50 scenes: the ``target`` split's ten (layout, style) pairs.
TARGET_LAYOUTS = tuple(zip(range(1, 11), range(1, 11)))
#: Resets before the episode starts, as RPent's env server and toolkit make them.
RPENT_RESETS = 2

#: Action layout of PandaOmron's HYBRID_MOBILE_BASE controller.
A_ARM, A_GRIP, A_BASE, A_TORSO, A_MODE = slice(0, 6), 6, slice(7, 10), 10, 11
ARM_POS_MAX = 0.05  # metres per unit of the arm's position input
ARM_ROT_MAX = 0.5  # radians per unit of its rotation input
BASE_SPEED = 0.025  # metres per native action at a base command of 1 (measured: 0.0118 at 0.5)
BASE_TURN = math.radians(3.3)  # radians per native action at a turn command of 1
TORSO_RANGE = (0.0, 0.34)

#: Waypoints of a line, at most this far apart.  The controller turns the gripper
#: about 3 degrees per action, so it is fed the next waypoint only once it has nearly caught up
#: (FOLLOW_M / FOLLOW_DEG).  Measured on seven moves (2026-10-28): every pose reached within
#: 4 mm and 0.7 degrees, a 19 cm line followed within 2.1 mm, 25 % fewer actions than with
#: waypoints 2.5 degrees apart and a gain of 1.5.
WAY_M = 0.01
WAY_DEG = 4.0
#: The servo asks for this multiple of the remaining step (the controller covers part of it).
ARM_GAIN = 2.0
#: Arrival: the grasp point this close and the gripper turned this little.
ARRIVE_M = 0.01
ARRIVE_DEG = 5.0
#: Settling on the last waypoint: until this close, or no closer for SETTLE_STALL actions
#: (straight down, the gripper circles in on its pose: six actions stopped it 1.2 cm short).
SETTLE_ACTIONS = 30
SETTLE_STALL = 10
SETTLE_M = 0.002
#: A target pitch this close to 90 is straight down, where yaw alone places the fingers.
VERTICAL_DEG = 1.0
SETTLE_DEG = 1.0
#: A plan that brings a joint this close to its stop is not followable: the controller pins the
#: joint there while the IK had turned the others instead (the wrist turned the short way into
#: its stop and ended 4 degrees off).
JOINT_MARGIN_RAD = 0.05
#: The next waypoint is taken once the gripper is this near the current one.
FOLLOW_M = 0.02
FOLLOW_DEG = 10.0
#: The arm has been stopped by something when it gets no nearer its waypoint for this many
#: native actions.
BLOCK_ACTIONS = 8
#: A pose the IK misses by no more than this is taken (the closest one), with a note.
NEAR_M = 0.02
NEAR_DEG = 8.0
IK_TOL_M = 0.001
IK_TOL_DEG = 0.5
#: Fraction of the way back to the starting joints the controller's nullspace pull covers per
#: planning step (its gain is 10 against the task's 150).
POSTURE_PULL = 0.05

#: Finger gap (metres) of a fully open Panda gripper; the opening reads gap / this.
FULL_GAP_M = 0.08
#: After a close, fingers stopped further apart than this hold something.
HOLD_GAP_M = 0.005
GRIP_STILL_M = 0.0005
GRIP_MAX_ACTIONS = 18
#: An empty gripper opened at least this far may arrive turned half a turn (fingers alike).
OPEN_FOR_HALF_TURN = 0.85

#: ``home`` first lifts a grasp point that is away from home straight up by up to this much.
HOME_LIFT_M = 0.06

BASE_TOL_M = 0.015
BASE_TOL_DEG = 1.0
BASE_STALL_ACTIONS = 15
BASE_STALL_M = 0.003
TORSO_TOL_M = 0.002


# ---------------------------------------------------------------------------
# codec: plain values and arrays across two interpreters (and two numpy versions)
# ---------------------------------------------------------------------------


def pack(value: Any) -> Any:
    """``value`` with every numpy array / scalar replaced by plain data (arrays as bytes)."""

    if isinstance(value, np.ndarray):
        return {
            "__nd__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": np.ascontiguousarray(value).tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    return value


def unpack(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__nd__"):
            return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(
                value["shape"]
            )
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# rotations (the gripper angles are RoboDojo's: hosts/robodojo_kinematics.py)
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


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def orientation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """The gripper's rotation (columns: pointing direction, finger line, third) in the robot frame."""

    return (
        _rot_z(math.radians(90.0 + yaw)) @ _rot_y(math.radians(pitch)) @ _rot_x(math.radians(roll))
    )


def angles(rot: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees of a gripper rotation; roll 0 when pointing straight down."""

    point = rot[:, 0]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    if math.hypot(point[0], point[1]) < math.sin(math.radians(1.0)):
        jaw = rot[:, 1]
        return pitch, wrap(math.degrees(math.atan2(-jaw[1], -jaw[0]))), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    rest = orientation(pitch, yaw, 0.0).T @ rot
    return pitch, wrap(yaw), wrap(math.degrees(math.atan2(rest[2, 1], rest[1, 1])))


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two rotations."""

    c = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def rotvec(rot: np.ndarray) -> np.ndarray:
    c = (np.trace(rot) - 1.0) / 2.0
    angle = math.acos(max(-1.0, min(1.0, c)))
    if angle < 1e-9:
        return np.zeros(3)
    w = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    s = math.sin(angle)
    if s < 1e-6:
        axis = np.sqrt(np.maximum(np.diag(rot) + 1.0, 0.0) / 2.0)
        axis = axis * np.sign(w + (w == 0))
        return axis / np.linalg.norm(axis) * angle
    return w / (2 * s) * angle


def from_rotvec(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a @ from_rotvec(rotvec(a.T @ b) * t)


def quat_wxyz(m: np.ndarray) -> list[float]:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    n = math.sqrt(sum(v * v for v in q))
    q = tuple(v / n for v in q)
    return [float(v) for v in (q if q[0] >= 0 else tuple(-v for v in q))]


def _r(v: Any, nd: int = 4) -> Any:
    if isinstance(v, (list, tuple, np.ndarray)):
        return [round(float(x), nd) for x in v]
    return round(float(v), nd)


# ---------------------------------------------------------------------------
# the kitchen
# ---------------------------------------------------------------------------


class Over(Exception):
    """The episode is over (budget spent or the task done): stop the primitive."""


class Kitchen:
    """One RoboCasa365 episode and the robot's primitives on it."""

    def __init__(
        self, task: str, split: str = "target", seed: int = 1, horizon: int | None = None
    ) -> None:
        import mujoco
        import robocasa  # noqa: F401 - registers the kitchen tasks
        import robosuite
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from robosuite.controllers import load_composite_controller_config

        if split != "target":
            raise ValueError("only the target split (RoboCasa365's evaluation scenes) is built")
        self.mj = mujoco
        self.task = task
        self.seed = int(seed)
        self.env = robosuite.make(
            env_name=task,
            robots="PandaOmron",
            controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
            camera_names=list(CAMERAS.values()),
            camera_widths=WIDTH,
            camera_heights=HEIGHT,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=False,
            camera_depths=False,
            seed=self.seed,
            obj_instance_split="target",
            layout_ids=None,
            style_ids=None,
            layout_and_style_ids=list(TARGET_LAYOUTS),
        )
        # The scene of a (task, seed) is the one RPent's Target50 runs (and PyRUA-Lean's RoboCasa
        # results) were scored on: RPent's env server resets the kitchen once when it starts and
        # once more when its toolkit connects, and every reset draws a new layout and new objects
        # (one reset gave "Open the right drawer." in layout 5 where theirs is "the left drawer"
        # in layout 6).  The host also fixes PYTHONHASHSEED=0, as PyRUA-Lean's host does.
        for _ in range(RPENT_RESETS):
            self.env.reset()
        self.horizon = int(horizon) if horizon else int(get_task_horizon(task))
        self.instruction = str(self.env.get_ep_meta().get("lang") or "")
        sim = self.env.sim
        self.sim = sim
        self.m = sim.model._model
        self.d = sim.data._data
        m = self.m
        name = sim.model
        self.base_body = name.body_name2id("mobilebase0_base")
        self.support_body = name.body_name2id("mobilebase0_support")
        self.hand_body = name.body_name2id("robot0_right_hand")
        self.grip_site = name.site_name2id("gripper0_right_grip_site")
        arm_joints = [name.joint_name2id(f"robot0_joint{i}") for i in range(1, 8)]
        self.arm_q = np.array([m.jnt_qposadr[j] for j in arm_joints])
        self.arm_v = np.array([m.jnt_dofadr[j] for j in arm_joints])
        self.limits = np.array([m.jnt_range[j] for j in arm_joints])
        fingers = [name.joint_name2id(f"gripper0_right_finger_joint{i}") for i in (1, 2)]
        self.finger_q = [m.jnt_qposadr[j] for j in fingers]
        self.torso_q = m.jnt_qposadr[name.joint_name2id("mobilebase0_joint_torso_height")]
        self.turn_axis = np.array(m.jnt_pos[name.joint_name2id("mobilebase0_joint_mobile_yaw")], float)
        self.cam_ids = {k: name.camera_name2id(v) for k, v in CAMERAS.items()}
        self.arm_ctrl = self.env.robots[0].composite_controller.part_controllers["right"]
        self.scratch = mujoco.MjData(m)
        self.steps = 0
        self.solved = False
        self.over: str | None = None
        self.grip_sign = 0.0
        p, yaw = self._base_world()
        self.start_base = (p.copy(), yaw)
        self.start_q = self.d.qpos[self.arm_q].copy()
        # where the gripper starts, relative to the torso it is mounted on (what ``home`` restores)
        self.start_in_support = self._in_support(*self._gripper_world())
        self._robot_geoms: np.ndarray | None = None
        self.solved = bool(self.env._check_success())

    # -- frames -------------------------------------------------------------

    def _base_world(self) -> tuple[np.ndarray, float]:
        """Where the robot frame's origin is (on the floor, under the base's turning axis) and
        the base's heading.  On the axis, a turn in place leaves the origin where it is (the
        base body's own origin swung 0.21 m round it, and every turn ended in a slow creep
        back to it)."""

        p = self.d.xpos[self.base_body]
        r = self.d.xmat[self.base_body].reshape(3, 3)
        axis = p + r @ self.turn_axis
        return np.array([axis[0], axis[1], 0.0]), math.atan2(r[1, 0], r[0, 0])

    def _robot_rot(self) -> np.ndarray:
        """World-from-robot rotation: robot x = the base's right, y = its forward, z = up."""

        _, yaw = self._base_world()
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return np.column_stack([-left, fwd, [0.0, 0.0, 1.0]])

    def to_robot(self, p_w: np.ndarray) -> np.ndarray:
        o, _ = self._base_world()
        return self._robot_rot().T @ (np.asarray(p_w, float) - o)

    def to_world(self, p_r: np.ndarray) -> np.ndarray:
        o, _ = self._base_world()
        return o + self._robot_rot() @ np.asarray(p_r, float)

    def _gripper_world(self, d: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """(grasp point, gripper rotation) in the world: columns pointing, finger line, third."""

        d = self.d if d is None else d
        h = d.xmat[self.hand_body].reshape(3, 3)
        return d.site_xpos[self.grip_site].copy(), np.column_stack([h[:, 2], h[:, 1], -h[:, 0]])

    def gripper_robot(self) -> tuple[np.ndarray, np.ndarray]:
        p, r = self._gripper_world()
        rr = self._robot_rot()
        o, _ = self._base_world()
        return rr.T @ (p - o), rr.T @ r

    def _support(self) -> tuple[np.ndarray, np.ndarray]:
        return self.d.xpos[self.support_body].copy(), self.d.xmat[self.support_body].reshape(3, 3)

    def _in_support(self, p_w: np.ndarray, r_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sp, sr = self._support()
        return sr.T @ (p_w - sp), sr.T @ r_w

    def _from_support(self, p_s: np.ndarray, r_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sp, sr = self._support()
        return sp + sr @ p_s, sr @ r_s

    def opening(self) -> float:
        return min(1.0, max(0.0, self.gap() / FULL_GAP_M))

    def gap(self) -> float:
        return float(self.d.qpos[self.finger_q[0]] - self.d.qpos[self.finger_q[1]])

    def torso(self) -> float:
        return float(self.d.qpos[self.torso_q])

    # -- the benchmark's clock ----------------------------------------------

    def _step(self, action: np.ndarray) -> None:
        if self.over:
            raise Over(self.over)
        self.env.step(action)
        self.steps += 1
        if self.env._check_success():
            self.solved = True
            self.over = "ended"
        elif self.steps >= self.horizon:
            self.over = "budget"

    def status(self) -> dict[str, Any]:
        return {"step": self.steps, "step_limit": self.horizon, "over": self.over is not None}

    def _action(self, *, grip: float | None = None, mode: float = -1.0) -> np.ndarray:
        a = np.zeros(12)
        a[A_GRIP] = self.grip_sign if grip is None else grip
        a[A_MODE] = mode
        return a

    def _servo(self, a: np.ndarray, p_w: np.ndarray, r_w: np.ndarray) -> None:
        """Fill ``a``'s arm part: the step from where the gripper is to ``p_w`` / ``r_w``."""

        cur_p, cur_r = self._gripper_world()
        o = self.arm_ctrl.origin_ori
        dp = o.T @ (np.asarray(p_w) - cur_p) * ARM_GAIN / ARM_POS_MAX
        dr = o.T @ rotvec(np.asarray(r_w) @ cur_r.T) * ARM_GAIN / ARM_ROT_MAX
        for part in (dp, dr):
            big = float(np.max(np.abs(part)))
            if big > 1.0:
                part /= big
        a[0:3], a[3:6] = dp, dr

    def _hold_action(self, hold: tuple[np.ndarray, np.ndarray], **kw: Any) -> np.ndarray:
        """An action that keeps the gripper where ``hold`` (support frame) puts it."""

        a = self._action(**kw)
        self._servo(a, *self._from_support(*hold))
        return a

    # -- kinematics ---------------------------------------------------------

    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.scratch
        s.qpos[:] = self.d.qpos
        s.qpos[self.arm_q] = q
        self.mj.mj_kinematics(self.m, s)
        return self._gripper_world(s)

    def _ik_from(self, p_w: np.ndarray, r_w: np.ndarray, seed: np.ndarray, iters: int = 120):
        s, m = self.scratch, self.m
        q = np.clip(np.asarray(seed, float).copy(), self.limits[:, 0] + 1e-3, self.limits[:, 1] - 1e-3)
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        err_m = err_deg = math.inf
        for _ in range(iters):
            s.qpos[:] = self.d.qpos
            s.qpos[self.arm_q] = q
            self.mj.mj_kinematics(m, s)
            self.mj.mj_comPos(m, s)
            cur_p, cur_r = self._gripper_world(s)
            ep = p_w - cur_p
            er = rotvec(r_w @ cur_r.T)
            err_m, err_deg = float(np.linalg.norm(ep)), math.degrees(float(np.linalg.norm(er)))
            if err_m < IK_TOL_M and err_deg < IK_TOL_DEG:
                break
            self.mj.mj_jacSite(m, s, jp, jr, self.grip_site)
            j = np.vstack([jp[:, self.arm_v], 0.3 * jr[:, self.arm_v]])
            e = np.r_[ep, 0.3 * er]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), e)
            big = float(np.max(np.abs(dq)))
            if big > 0.3:
                dq *= 0.3 / big
            q = np.clip(q + dq, self.limits[:, 0] + 1e-3, self.limits[:, 1] - 1e-3)
        return q, err_m, err_deg

    def ik(self, p_w: np.ndarray, r_w: np.ndarray, seeds: list[np.ndarray] | None = None):
        """(joints, position error m, rotation error deg) of the best of several starts."""

        q0 = self.d.qpos[self.arm_q].copy()
        starts = [q0, self.start_q.copy()] + list(seeds or [])
        for turn in (-math.pi / 2, math.pi / 2):
            q = q0.copy()
            q[6] += turn
            starts.append(q)
        for turn in (-math.pi, math.pi):  # the wrist the other way round
            q = self.start_q.copy()
            q[6] = max(self.limits[6, 0] + 0.2, min(self.limits[6, 1] - 0.2, q[6] + turn))
            starts.append(q)
        best = None
        for start in starts:
            q, em, ed = self._ik_from(p_w, r_w, start)
            missed = em > IK_TOL_M * 3 or ed > IK_TOL_DEG * 3
            score = (missed, self._at_stop(q), em + ed / 100.0, float(np.sum(np.abs(q - q0))))
            if best is None or score < best[0]:
                best = (score, q, em, ed)
            if not score[0] and not score[1] and float(np.max(np.abs(q - q0))) < 1.0:
                break
        return best[1], best[2], best[3]

    def _at_stop(self, q: np.ndarray) -> bool:
        return bool(
            np.any(q < self.limits[:, 0] + JOINT_MARGIN_RAD)
            or np.any(q > self.limits[:, 1] - JOINT_MARGIN_RAD)
        )

    def _osc_joints(self, q: np.ndarray, poses: list) -> tuple[np.ndarray, int | None, str | None]:
        """Where the controller's own redundancy resolution takes the joints along ``poses``.

        The benchmark's OSC moves the joints by the mass-weighted pseudo-inverse of the
        gripper's Jacobian and pulls them, in its nullspace, towards the joints the robot
        started with (``nullspace_torques``, gain 10 against 150).  The same law, on the
        robot's own model, predicts which way the joints go and whether one meets its stop:
        an IK that is free to choose its own solution turned the others instead and promised
        poses the controller then pinned against the wrist's stop.  Returns the joints at the
        end, the index of the first pose that fails and why (None, None when all are fine).
        """

        s, m = self.scratch, self.m
        q = np.asarray(q, float).copy()
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        full = np.zeros((m.nv, m.nv))
        sub = np.ix_(self.arm_v, self.arm_v)
        for index, (p, r) in enumerate(poses):
            last = index == len(poses) - 1
            for _ in range(12 if last else 3):
                s.qpos[:] = self.d.qpos
                s.qpos[self.arm_q] = q
                self.mj.mj_kinematics(m, s)
                self.mj.mj_comPos(m, s)
                self.mj.mj_crb(m, s)
                self.mj.mj_fullM(m, full, s.qM)
                cur_p, cur_r = self._gripper_world(s)
                e = np.r_[p - cur_p, rotvec(r @ cur_r.T)]
                if float(np.linalg.norm(e[:3])) < 2e-4 and float(np.linalg.norm(e[3:])) < 2e-3:
                    break
                self.mj.mj_jacSite(m, s, jp, jr, self.grip_site)
                j = np.vstack([jp[:, self.arm_v], jr[:, self.arm_v]])
                mi = np.linalg.inv(full[sub])
                jbar = mi @ j.T @ np.linalg.pinv(j @ mi @ j.T, rcond=1e-4)
                null = np.eye(7) - jbar @ j
                dq = jbar @ e + null @ (POSTURE_PULL * (self.start_q - q))
                big = float(np.max(np.abs(dq)))
                if big > 0.2:
                    dq *= 0.2 / big
                q = q + dq
            low = q < self.limits[:, 0] + JOINT_MARGIN_RAD
            high = q > self.limits[:, 1] - JOINT_MARGIN_RAD
            if np.any(low | high):
                joint = int(np.argmax(low | high)) + 1
                what = "the wrist" if joint == 7 else f"joint {joint}"
                return q, index, f"{what} would reach its stop"
        s.qpos[:] = self.d.qpos
        s.qpos[self.arm_q] = q
        self.mj.mj_kinematics(m, s)
        cur_p, cur_r = self._gripper_world(s)
        p, r = poses[-1]
        if float(np.linalg.norm(p - cur_p)) > 0.004 or rotation_error(r, cur_r) > 2.0:
            return q, len(poses) - 1, "the arm cannot get there along it"
        return q, None, None

    def _poses(self, start: tuple, goal: tuple, long_way: bool = False) -> list:
        """World poses along a straight line from ``start`` to ``goal``, turning the short way
        round (or the long way: the same axis, the other sense)."""

        (p0, r0), (p1, r1) = start, goal
        w = rotvec(r1 @ r0.T)
        angle = float(np.linalg.norm(w))
        if long_way:
            if angle < math.radians(20.0):
                return []
            w = -w / angle * (2 * math.pi - angle)
            angle = 2 * math.pi - angle
        dist = float(np.linalg.norm(p1 - p0))
        n = max(1, math.ceil(dist / WAY_M), math.ceil(math.degrees(angle) / WAY_DEG))
        return [(p0 + (p1 - p0) * (i / n), from_rotvec(w * (i / n)) @ r0) for i in range(1, n + 1)]

    def _plan(self, start: tuple, goal: tuple, path: str) -> tuple[list, str | None, str | None]:
        """(world poses, the path taken, why nothing can be done) from ``start`` to ``goal``.

        ``line``: a straight line, the gripper turning the short way round or, where that
        takes the wrist into its stop, the long way.  ``joint`` / ``auto`` (when no line is
        possible): by way of the gripper's starting pose, which turns the arm round.
        """

        q = self.d.qpos[self.arm_q].copy()
        why = None
        if path in ("line", "auto"):
            for long_way in (False, True):
                poses = self._poses(start, goal, long_way)
                if not poses:
                    continue
                _, failed, reason = self._osc_joints(q, poses)
                if failed is None:
                    return poses, "line", None
                if why is None:
                    along = float(np.linalg.norm(poses[failed][0] - start[0]))
                    why = f"along the straight line {reason} {along:.3f} m from the start"
            if path == "line":
                return [], None, why
        home = self._from_support(*self.start_in_support)
        for via in ((home[0] + np.array([0.0, 0.0, 0.0]), home[1]),):
            for first_long in (False, True):
                legs = self._poses(start, via, first_long)
                if not legs and first_long:
                    continue
                qv, failed, _ = self._osc_joints(q, legs) if legs else (q, None, None)
                if failed is not None:
                    continue
                for second_long in (False, True):
                    rest = self._poses(via, goal, second_long)
                    if not rest:
                        continue
                    _, failed, reason = self._osc_joints(qv, rest)
                    if failed is None:
                        return legs + rest, "joint", None
                    why = why or f"by way of the starting pose {reason}"
        return [], None, why or "no way there was found"

    # -- running waypoints ----------------------------------------------------

    def _follow(self, poses: list, grip: float) -> dict[str, Any]:
        """Servo through world poses, then settle on the last.

        The next waypoint is taken once the gripper is within FOLLOW_M / FOLLOW_DEG of the
        current one, so the arm goes at the controller's own pace and never falls so far behind
        that a turn could be taken the other way round.  The arm is blocked when it gets no
        nearer to its waypoint for BLOCK_ACTIONS actions.
        """

        used = 0
        blocked = False
        index = 0
        best = math.inf
        since = 0
        cap = 4 * len(poses) + 40
        while index < len(poses) and used < cap:
            p, r = poses[index]
            a = self._action(grip=grip)
            self._servo(a, p, r)
            self._step(a)
            used += 1
            if self.over:
                break
            cur_p, cur_r = self._gripper_world()
            lag_m = float(np.linalg.norm(p - cur_p))
            lag_deg = rotation_error(r, cur_r)
            if lag_m <= FOLLOW_M and lag_deg <= FOLLOW_DEG:
                index += 1
                best, since = math.inf, 0
                continue
            score = lag_m + lag_deg / 500.0
            if score < best - 0.0005:
                best, since = score, 0
            else:
                since += 1
                if since >= BLOCK_ACTIONS:
                    blocked = True
                    break
        index = min(index, len(poses) - 1)
        if not blocked and not self.over and poses:
            p, r = poses[-1]
            best = math.inf
            since = 0
            for _ in range(SETTLE_ACTIONS):
                cur_p, cur_r = self._gripper_world()
                dist, turn = float(np.linalg.norm(p - cur_p)), rotation_error(r, cur_r)
                if dist < SETTLE_M and turn < SETTLE_DEG:
                    break
                score = dist + turn / 500.0  # 1 cm ~ 5 degrees
                if score < best - 0.0005:
                    best, since = score, 0
                else:
                    since += 1
                    if since >= SETTLE_STALL:
                        break
                a = self._action(grip=grip)
                self._servo(a, p, r)
                self._step(a)
                used += 1
                if self.over:
                    break
            cur_p, cur_r = self._gripper_world()
            stalled = float(np.linalg.norm(p - cur_p)) > ARRIVE_M or rotation_error(r, cur_r) > ARRIVE_DEG
            blocked = bool(stalled and not self.over)
        return {"steps_used": used, "blocked": blocked, "waypoints_done": index + 1}

    def _grip_sign(self, gripper: Any) -> float:
        g = float(gripper or 0)
        if g > 0:
            self.grip_sign = 1.0
        elif g < 0:
            self.grip_sign = -1.0
        return self.grip_sign if g != 0 else 0.0

    def _pose_summary(self) -> dict[str, Any]:
        p, r = self.gripper_robot()
        pitch, yaw, roll = angles(r)
        return {
            "xyz": _r(p),
            "angles": {"pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1)},
        }

    # -- motion primitives ---------------------------------------------------

    def _target(self, xyz, pitch, yaw, roll) -> tuple[np.ndarray, np.ndarray, tuple]:
        _, cur = self.gripper_robot()
        now = angles(cur)
        want = tuple(float(v) if v is not None else c for v, c in zip((pitch, yaw, roll), now))
        # Straight down, a roll left None is 0 and a yaw left None is the line the fingers close
        # along now: a gripper stopped a degree or more off vertical reads a roll that, kept, would
        # turn the fingers up to 90 degrees away from the yaw asked for.
        if roll is None and abs(want[0] - 90.0) < VERTICAL_DEG:
            jaw = cur[:, 1]
            line = wrap(math.degrees(math.atan2(-jaw[1], -jaw[0]))) if yaw is None else want[1]
            want = (want[0], line, 0.0)
        return np.asarray(xyz, float), orientation(*want), want

    def _alternatives(self, xyz: np.ndarray, want: tuple) -> list[str]:
        def ok(p_r: np.ndarray, a: tuple) -> bool:
            _, em, ed = self.ik(self.to_world(p_r), self._robot_rot() @ orientation(*a))
            return em <= IK_TOL_M * 3 and ed <= IK_TOL_DEG * 3

        out = []
        pitch, yaw, roll = want
        for p in (60.0, 45.0, 30.0, 0.0):
            if p < pitch - 5.0 and ok(xyz, (p, yaw, roll)):
                out.append(f"pitch={p:.0f} at this position")
                break
        for dz in (-0.03, -0.06, -0.10, 0.03, 0.06, 0.10):
            if ok(xyz + np.array([0.0, 0.0, dz]), want):
                out.append(f"z={xyz[2] + dz:.3f} with this orientation")
                break
        for dy in (-0.05, -0.10, -0.15):
            if ok(xyz + np.array([0.0, dy, 0.0]), want):
                out.append(
                    f"this pose {-dy:.2f} m nearer the robot (drive the base {-dy:.2f} m closer)"
                )
                break
        return out

    def move_pose(self, xyz, gripper=0, pitch=None, yaw=None, roll=None, path="auto") -> dict:
        if path not in ("auto", "line", "joint"):
            return {"error": "path must be 'auto', 'line' or 'joint'"}
        xyz_r, rot_r, want = self._target(xyz, pitch, yaw, roll)
        grip = self._grip_sign(gripper)
        start = self.gripper_robot()
        rr = self._robot_rot()
        empty = self.opening() >= OPEN_FOR_HALF_TURN and grip <= 0
        targets = [rot_r]
        if empty:
            targets.append(rot_r @ _rot_x(math.pi))
        options = []
        first_why = None
        for rot in targets:
            p_w, r_w = self.to_world(xyz_r), rr @ rot
            qs, em, ed = self.ik(p_w, r_w)
            near = em <= NEAR_M and ed <= NEAR_DEG
            if em > IK_TOL_M * 3 or ed > IK_TOL_DEG * 3:
                if not near:
                    why = (
                        f"out of reach: the closest pose the arm can take is {em:.3f} m and "
                        f"{ed:.0f} deg away"
                    )
                    first_why = first_why or why
                    continue
            note = None
            goal = (p_w, r_w)
            if em > IK_TOL_M * 3 or ed > IK_TOL_DEG * 3:
                goal = self._fk(qs)
                note = (
                    f"the exact pose is out of reach; went to the closest one, {em:.3f} m and "
                    f"{ed:.0f} deg from it"
                )
            poses, taken, why = self._plan(self._gripper_world(), goal, path)
            if taken is None:
                if path == "line" and why:
                    why += " (the target itself is reachable: path='auto' may go round)"
                first_why = first_why or why
                continue
            # an exact pose before a near miss, a line before going round, then the shorter
            options.append((note is not None, taken != "line", len(poses), poses, taken, rot, note, goal))
        result: dict[str, Any] = {
            "arm": "right",
            "target_xyz": _r(xyz_r),
            "target_angles": {"pitch": round(want[0], 1), "yaw": round(want[1], 1), "roll": round(want[2], 1)},
            "start_xyz": _r(start[0]),
            "steps_used": 0,
            "path": None,
        }
        if not options:
            why = first_why or "no plan"
            if why.startswith("out of reach"):
                alts = self._alternatives(xyz_r, want)
                why += (
                    "; reachable instead: " + ", or ".join(alts)
                    if alts
                    else "; nothing close is reachable either: move the base or raise the torso"
                )
            now = self._pose_summary()
            result.update(
                final_xyz=now["xyz"],
                final_angles=now["angles"],
                dist_to_target_m=round(float(np.linalg.norm(np.asarray(now["xyz"]) - xyz_r)), 4),
                orientation_error_deg=round(rotation_error(start[1], rot_r), 1),
                reached=False,
                ik_error=why,
            )
            return result
        *_, poses, taken, rot, note, goal = min(options, key=lambda o: o[:3])
        if taken == "joint":
            went = "went by way of the gripper's starting pose to turn the arm round"
            note = f"{note}; {went}" if note else went
        if rot is not targets[0]:
            half = (
                "took the pose turned half a turn about the pointing direction (the empty "
                "fingers are interchangeable): yaw / roll read 180 off"
            )
            note = f"{note}; {half}" if note else half
        result["path"] = taken
        if note:
            result["note"] = note
        run = self._follow(poses, grip)
        cur_p, cur_r = self._gripper_world()
        dist = float(np.linalg.norm(cur_p - self.to_world(xyz_r)))
        turn = rotation_error(cur_r, goal[1])
        now = self._pose_summary()
        result.update(
            final_xyz=now["xyz"],
            final_angles=now["angles"],
            dist_to_target_m=round(dist, 4),
            orientation_error_deg=round(turn, 1),
            reached=bool(dist <= ARRIVE_M and turn <= ARRIVE_DEG),
            steps_used=run["steps_used"],
            ik_error=None,
        )
        if run["blocked"] and not result["reached"]:
            result["blocked"] = "the arm stopped short of the planned pose: something is in the way"
        return result

    def grip(self, gripper=1, opening=None) -> dict:
        g = float(gripper)
        if g not in (1.0, -1.0):
            return {"error": "gripper must be 1 (close) or -1 (open)"}
        if opening is not None:
            opening = float(opening)
            if g > 0 or not 0.0 < opening <= 1.0:
                return {"error": "opening is for an open command, between 0 (excluded) and 1"}
        hold = self._in_support(*self._gripper_world())
        before = self.opening()
        used = 0
        last = self.gap()
        sign = g
        for _ in range(GRIP_MAX_ACTIONS):
            if opening is not None and self.opening() >= opening - 0.01:
                break
            self._step(self._hold_action(hold, grip=sign))
            used += 1
            now = self.gap()
            if used >= 3 and abs(now - last) < GRIP_STILL_M:
                break
            last = now
            if self.over:
                break
        # from here on KEEP (0) leaves the controller's finger target where this stroke left it
        self.grip_sign = 1.0 if g > 0 else -1.0
        if opening is not None:
            self.grip_sign = 0.0
        closing = g > 0
        gap = self.gap()
        holding = bool(closing and gap > HOLD_GAP_M)
        return {
            "arm": "right",
            "gripper": "closed" if closing else "open",
            "opening": round(self.opening(), 3),
            "opening_before": round(before, 3),
            "holding": holding,
            "width_m": round(gap, 4) if holding else (0.0 if closing else None),
            "steps_used": used,
            "status": {"step": self.steps, "step_limit": self.horizon},
        }

    def home(self) -> dict:
        target = self._from_support(*self.start_in_support)
        used = 0
        lifted = None
        cur_p, cur_r = self._gripper_world()
        away = float(np.linalg.norm(cur_p - target[0])) > ARRIVE_M
        why = None
        if away:
            for dz in (HOME_LIFT_M, HOME_LIFT_M * 2 / 3, HOME_LIFT_M / 3):
                up = (cur_p + np.array([0.0, 0.0, dz]), cur_r)
                poses, taken, _ = self._plan((cur_p, cur_r), up, "line")
                if taken:
                    used += self._follow(poses, 0.0)["steps_used"]
                    lifted = round(dz, 3)
                    break
            if not self.over:
                target = self._from_support(*self.start_in_support)
                poses, taken, why = self._plan(self._gripper_world(), target, "line")
                if taken:
                    used += self._follow(poses, 0.0)["steps_used"]
        target = self._from_support(*self.start_in_support)
        cur_p, cur_r = self._gripper_world()
        dist = float(np.linalg.norm(cur_p - target[0]))
        turn = rotation_error(cur_r, target[1])
        q_err = float(np.max(np.abs(self.d.qpos[self.arm_q] - self.start_q)))
        out = {
            "arm": "right",
            "reached": bool(dist <= ARRIVE_M and turn <= ARRIVE_DEG),
            "dist_m": round(dist, 4),
            "orientation_error_deg": round(turn, 1),
            "max_joint_error_rad": round(q_err, 4),
            "steps_used": used,
        }
        if lifted:
            out["lifted_m"] = lifted
        if why and not out["reached"]:
            out["note"] = why
        return out

    def move_base(self, dx=0.0, dy=0.0, dyaw=0.0) -> dict:
        """Drive the base by (dx right, dy ahead) metres and turn it dyaw degrees (CCW)."""

        o0, yaw0 = self._base_world()
        rr0 = self._robot_rot()
        goal_xy = o0 + rr0 @ np.array([float(dx), float(dy), 0.0])
        goal_yaw = yaw0 + math.radians(float(dyaw))
        dist0 = float(np.hypot(dx, dy))
        cap = int(60 + dist0 / (BASE_SPEED * 0.6) + abs(float(dyaw)) / math.degrees(BASE_TURN * 0.6))
        used = 0
        best = math.inf
        since = 0
        blocked = False
        for _ in range(cap):
            o, yaw = self._base_world()
            e_w = goal_xy - o
            e_yaw = math.atan2(math.sin(goal_yaw - yaw), math.cos(goal_yaw - yaw))
            dist = float(np.hypot(e_w[0], e_w[1]))
            if dist < BASE_TOL_M and abs(math.degrees(e_yaw)) < BASE_TOL_DEG:
                break
            fwd = e_w[0] * math.cos(yaw) + e_w[1] * math.sin(yaw)
            left = -e_w[0] * math.sin(yaw) + e_w[1] * math.cos(yaw)
            cmd = np.array([fwd / (BASE_SPEED * 4), left / (BASE_SPEED * 4), e_yaw / (BASE_TURN * 1.5)])
            big = float(np.max(np.abs(cmd[:2])))
            if big > 1.0:
                cmd[:2] /= big
            cmd[2] = max(-1.0, min(1.0, cmd[2]))
            a = self._action(mode=1.0)
            a[A_BASE] = cmd
            self._step(a)
            used += 1
            progress = dist + abs(e_yaw) * 0.3
            if progress < best - BASE_STALL_M:
                best = progress
                since = 0
            else:
                since += 1
            if since >= BASE_STALL_ACTIONS:
                blocked = True
                break
            if self.over:
                break
        # let it come to rest, still in base mode (the arm keeps its place on the base)
        for _ in range(3):
            if self.over:
                break
            self._step(self._action(mode=1.0))
            used += 1
        o, yaw = self._base_world()
        moved = rr0.T @ (o - o0)
        e = goal_xy - o
        out = {
            "moved": [round(float(moved[0]), 4), round(float(moved[1]), 4)],
            "turned_deg": round(math.degrees(math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))), 2),
            "dist_to_target_m": round(float(np.hypot(e[0], e[1])), 4),
            "yaw_error_deg": round(math.degrees(math.atan2(math.sin(goal_yaw - yaw), math.cos(goal_yaw - yaw))), 2),
            "steps_used": used,
        }
        out["reached"] = bool(out["dist_to_target_m"] <= 0.02 and abs(out["yaw_error_deg"]) <= 2.0)
        if blocked and not out["reached"]:
            out["blocked"] = "the base stopped short: something is in the way (there is no path planning)"
        return out

    def set_torso(self, height) -> dict:
        h = float(height)
        if not TORSO_RANGE[0] <= h <= TORSO_RANGE[1]:
            return {"error": f"height must be within {TORSO_RANGE[0]} .. {TORSO_RANGE[1]} m"}
        hold = self._in_support(*self._gripper_world())
        before = self.torso()
        used = 0
        for _ in range(40):
            err = h - self.torso()
            if abs(err) < TORSO_TOL_M:
                break
            a = self._hold_action(hold)
            a[A_TORSO] = max(-1.0, min(1.0, err / 0.025))
            self._step(a)
            used += 1
            if self.over:
                break
        for _ in range(3):  # the arm catches up with the torso
            if self.over:
                break
            self._step(self._hold_action(hold))
            used += 1
        return {
            "height_m": round(self.torso(), 4),
            "height_before_m": round(before, 4),
            "reached": abs(self.torso() - h) < 0.005,
            "steps_used": used,
        }

    # -- observation --------------------------------------------------------

    def state(self) -> dict[str, Any]:
        p, r = self.gripper_robot()
        pitch, yaw, roll = angles(r)
        hand_p = self.to_robot(self.d.xpos[self.hand_body])
        o, byaw = self._base_world()
        o0, yaw0 = self.start_base
        c, s = math.cos(yaw0), math.sin(yaw0)
        rel = o - o0
        odo = np.array([rel[0] * s - rel[1] * c, rel[0] * c + rel[1] * s])  # start frame: x right
        return {
            "arm": {
                "tcp_pos": _r(p, 5),
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "eef_pos": _r(hand_p, 5),
                "eef_quat": _r(quat_wxyz(r), 5),
                "gripper": round(self.opening(), 4),
                "finger_gap_m": round(self.gap(), 4),
                "joints": _r(self.d.qpos[self.arm_q], 5),
            },
            "torso_m": round(self.torso(), 4),
            "base": [round(float(odo[0]), 4), round(float(odo[1]), 4),
                     round(math.degrees(math.atan2(math.sin(byaw - yaw0), math.cos(byaw - yaw0))), 2)],
        }

    def cameras(self) -> dict[str, Any]:
        """Each camera's colour image (row 0 at the top) and calibration in the robot frame."""

        out = {}
        rr = self._robot_rot()
        o, _ = self._base_world()
        flip = np.diag([1.0, -1.0, -1.0])  # MuJoCo's camera looks down -z with +y up
        for name, cid in self.cam_ids.items():
            rgb = self.sim.render(width=WIDTH, height=HEIGHT, camera_name=CAMERAS[name])
            rgb = np.ascontiguousarray(np.flipud(rgb)[..., :3]).astype(np.uint8)
            fovy = math.radians(float(self.m.cam_fovy[cid]))
            f = (HEIGHT / 2) / math.tan(fovy / 2)
            pose = np.eye(4)
            pose[:3, :3] = rr.T @ self.d.cam_xmat[cid].reshape(3, 3) @ flip
            pose[:3, 3] = rr.T @ (self.d.cam_xpos[cid] - o)
            out[name] = {
                "color": rgb,
                "intrinsic": [[f, 0.0, WIDTH / 2], [0.0, f, HEIGHT / 2], [0.0, 0.0, 1.0]],
                "cam_to_robot": pose.tolist(),
                "shape": [HEIGHT, WIDTH, 3],
            }
            out[name]["robot_mask"], out[name]["robot_depth"] = self._robot_pixels(name)
        return out

    def _robot_pixels(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """Where the robot's own body is in the image, and how far: its model, drawn on its own.

        What a real robot works out by projecting its URDF at its joint angles (RoboDojo's
        ``robot_pixels`` does that analytically).  The depth is given on the robot's own pixels
        only (NaN everywhere else): nothing of the kitchen is in either.
        """

        import robosuite.utils.camera_utils as cu

        if self._robot_geoms is None:
            body_of = self.m.geom_bodyid
            names = [self.sim.model.body_id2name(int(b)) or "" for b in body_of]
            mine = [n.startswith(("robot0_", "gripper0_", "mobilebase0_")) for n in names]
            self._robot_geoms = np.array(mine, bool)
        seg = self.sim.render(
            width=WIDTH, height=HEIGHT, camera_name=CAMERAS[camera], segmentation=True
        )
        seg = np.flipud(seg)
        kind, ids = seg[..., 0], seg[..., 1]
        geom = kind == int(self.mj.mjtObj.mjOBJ_GEOM)
        ids = np.where(geom & (ids >= 0) & (ids < len(self._robot_geoms)), ids, 0)
        mine = np.ascontiguousarray(geom & self._robot_geoms[ids])
        _, d = self.sim.render(width=WIDTH, height=HEIGHT, camera_name=CAMERAS[camera], depth=True)
        d = np.clip(np.nan_to_num(d, nan=1.0), 0.0, 1.0)
        depth = np.flipud(cu.get_real_depth_map(self.sim, d[..., None])[..., 0]).astype(np.float32)
        return mine, np.ascontiguousarray(np.where(mine, depth, np.nan).astype(np.float32))

    def observe(self, images: bool = True) -> dict[str, Any]:
        obs = {"state": self.state(), "status": self.status(), "instruction": self.instruction}
        if images:
            obs["cameras"] = self.cameras()
        return obs

    # -- development only (never served to a policy) -------------------------

    def depth_truth(self, camera: str) -> np.ndarray:
        """Metric depth along the optical axis, for checking perception offline."""

        import robosuite.utils.camera_utils as cu

        _, d = self.sim.render(width=WIDTH, height=HEIGHT, camera_name=CAMERAS[camera], depth=True)
        d = np.clip(np.nan_to_num(d, nan=1.0), 0.0, 1.0)
        return np.flipud(cu.get_real_depth_map(self.sim, d[..., None])[..., 0])


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------

#: Ops that run the simulator; their reply carries the observation after them.
MOTIONS = ("move_pose", "grip", "home", "move_base", "set_torso")


def handle(kitchen: Kitchen, op: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    if op == "hello":
        return {
            "task": kitchen.task,
            "seed": kitchen.seed,
            "instruction": kitchen.instruction,
            "horizon": kitchen.horizon,
            "cameras": list(CAMERAS),
            "size": [HEIGHT, WIDTH],
            "solved_at_start": kitchen.solved,
        }
    if op == "observe":
        return kitchen.observe(images=bool(kwargs.get("images", True)))
    if op == "verdict":  # host only: the task's own predicate, and whether the episode is over
        return {"solved": kitchen.solved, "over": kitchen.over, **kitchen.status()}
    if op == "depth_truth":  # development checks only (the host never serves it to a policy)
        return {"depth": kitchen.depth_truth(str(kwargs["camera"]))}
    if op in MOTIONS:
        if kitchen.over:
            payload: dict[str, Any] = {"steps_used": 0, "over": True}
        else:
            try:
                payload = getattr(kitchen, op)(**kwargs)
            except Over:
                payload = {"over": True}
            except TypeError as exc:
                payload = {"error": f"{op}: {exc}"}
        return {"payload": payload, "obs": kitchen.observe(images=True)}
    return {"error": f"no op {op!r}"}


def serve(kitchen: Kitchen, port: int, authkey: bytes) -> None:
    with Listener(("127.0.0.1", port), authkey=authkey) as listener:
        print(f"READY {port}", flush=True)
        with listener.accept() as conn:
            while True:
                try:
                    request = conn.recv()
                except EOFError:
                    return
                op = request.get("op")
                if op == "close":
                    conn.send({"ok": True, "result": {}})
                    return
                started = time.perf_counter()
                try:
                    result = handle(kitchen, str(op), dict(request.get("kwargs") or {}))
                    reply = {"ok": True, "result": pack(result)}
                except Exception:  # noqa: BLE001 - the host shows the traceback
                    reply = {"ok": False, "error": traceback.format_exc(limit=8)}
                reply["elapsed_s"] = time.perf_counter() - started
                conn.send(reply)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="target")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=None, help="override (development only)")
    args = parser.parse_args(argv)
    authkey = os.environ.get("ROBOCASA_SIM_AUTHKEY", "pyrualean").encode()
    kitchen = Kitchen(args.task, args.split, args.seed, horizon=args.horizon)
    serve(kitchen, args.port, authkey)
    return 0


if __name__ == "__main__":
    sys.exit(main())
