# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboDojo tools of this library's own, registered on RPent's toolkit at boot.

RPent's RoboDojo toolset moves an arm only by position (``move_to``) and grips one native
action at a time (``set_gripper``); its perception returns one mask centroid and one surface
point.  The tools here are what a program needs on top of that, computed in the host from the
same observations and commanded through the same action path:

``move_pose``    the grasp point to a position AND an orientation, along a straight line,
                 with the host's own IK (:mod:`.robodojo_kinematics`);
``grip``         a full gripper stroke, repeated until the fingers stop, with what they hold;
``locate``       an object's 3D extent, heading and top from its mask and the depth image;
``home``         the joints the episode started with (:func:`home`, kept from before).

Every tool is registered as a stateful (not read-only) tool, so RPent's toolkit records a step
after it like after its own motions.  Each native action is one ``env.step`` and counts against
the episode's budget; a tool stops at the step limit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np

from ..robodojo import HOME_MAX_ACTIONS, HOME_TOL_RAD
from . import robodojo_kinematics as kin

ARMS = kin.ARMS
#: Extra native actions at the end of a motion while the joints settle on the target.
SETTLE_ACTIONS = 8
#: The motion has arrived when the grasp point is this close (metres) and turned this little.
ARRIVE_M = 0.01
ARRIVE_DEG = 5.0
#: A gripper stroke stops when the opening changes less than this in one action.
GRIP_STILL = 0.004
GRIP_MAX_ACTIONS = 10
#: Largest joint change commanded in one action of a joint-space swing (radians).
JOINT_STEP_RAD = 0.15
#: A pose the IK misses by no more than this is taken (the closest one), with a note: in the
#: first sweep 11 of 17 refused targets were within 2 cm and 7 degrees of a reachable pose.
NEAR_M = 0.02
NEAR_DEG = 8.0
#: ``home`` first lifts a grasp point lower than this straight up (by up to HOME_LIFT_M) so the
#: swing back does not sweep across the table: one swing knocked a placed board off.
HOME_LIFT_BELOW = 1.0
HOME_LIFT_M = 0.06
#: ``move_pose`` may take the half-turned pose (fingers swapped) only with a gripper this open.
OPEN_FOR_HALF_TURN = 0.9
#: A gripper closed on nothing reads an opening of 0.0 (measured); after a close, an opening
#: above this means the fingers stopped on something.
HOLD_MIN_OPENING = 0.05
#: Gap between the fingers at opening 1.0, metres (each finger travels 0.044 m; measured: a
#: 3.9 cm block read 0.432).
FULL_GAP_M = 0.088


# ---------------------------------------------------------------------------
# observation and action plumbing
# ---------------------------------------------------------------------------


def _obs(primitives: Any) -> dict[str, Any]:
    obs = primitives.env.get_obs()
    primitives._last_obs = obs
    return obs


def _joints(state: dict[str, Any], arm: str) -> list[float]:
    return [float(v) for v in state[f"{arm}_arm_joint_state"]]


def _opening(state: dict[str, Any], arm: str) -> float:
    value = state.get(f"{arm}_ee_joint_state")
    for _ in range(4):
        if value is None or isinstance(value, (int, float)):
            break
        try:
            value = next(iter(value))
        except (TypeError, StopIteration):
            break
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _command(gripper: Any) -> float | None:
    """1 (close) -> 0.0, -1 (open) -> 1.0, 0 / None (keep) -> None."""

    if gripper is None:
        return None
    g = float(gripper)
    if g > 0:
        return 0.0
    if g < 0:
        return 1.0
    return None


def _act(
    primitives: Any,
    state: dict[str, Any],
    joints: dict[str, Sequence[float]],
    grips: dict[str, float | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One native action: the arms named in ``joints`` to those angles, the others held."""

    action: dict[str, list[float]] = {}
    for arm in ARMS:
        q = joints.get(arm)
        action[f"{arm}_arm_joint_state"] = [
            float(v) for v in (q if q is not None else _joints(state, arm))
        ]
        g = grips.get(arm)
        if g is None:
            g = _keep(primitives, state, arm)
        _remember(primitives, arm, g)
        action[f"{arm}_ee_joint_state"] = [float(g)]
    _obs_step, _reward, _done, info = primitives.env.step(action)
    obs = _obs(primitives)
    return obs["state"], (info.get("status") or {}) if isinstance(info, dict) else {}


def _remember(primitives: Any, arm: str, command: float) -> None:
    commands = getattr(primitives, "_grip_commands", None)
    if commands is None:
        commands = {}
        try:
            primitives._grip_commands = commands
        except AttributeError:  # an object that takes no attributes: KEEP measures, as before
            return
    commands[arm] = float(command)


def _keep(primitives: Any, state: dict[str, Any], arm: str) -> float:
    """The opening to command for a gripper that keeps what it has.

    The last command again, not the measured opening: commanding the opening a held object
    leaves takes the squeeze off (a held object was lost after the OTHER arm's motions 9.6 % of
    the time, after its own 6.1 %).  Unless the gripper is no longer where that command put it
    (the VLA moved it in between): closed further than commanded -> keep it closed; opened
    further -> follow it.
    """

    measured = _opening(state, arm)
    last = (getattr(primitives, "_grip_commands", None) or {}).get(arm)
    if last is None:
        return measured
    if last <= 0.05:  # last told to close
        return 0.0 if measured < 0.95 else measured
    if abs(measured - last) <= 0.08:
        return last
    return 0.0 if measured < last else measured


def _spent(status: dict[str, Any]) -> bool:
    try:
        return int(status.get("step", 0)) >= int(status.get("step_limit", 1 << 30))
    except (TypeError, ValueError):
        return False


def _tcp(arm: str, q: Sequence[float]) -> list[float]:
    return [round(float(v), 4) for v in kin.tcp(kin.fk(arm, q))]


def _pose_summary(arm: str, q: Sequence[float]) -> dict[str, Any]:
    pose = kin.fk(arm, q)
    pitch, yaw, roll = kin.angles(pose[:3, :3])
    return {
        "xyz": [round(float(v), 4) for v in kin.tcp(pose)],
        "pitch": round(pitch, 1),
        "yaw": round(yaw, 1),
        "roll": round(roll, 1),
    }


# ---------------------------------------------------------------------------
# motion
# ---------------------------------------------------------------------------


def _target(arm: str, q: Sequence[float], xyz, pitch, yaw, roll) -> tuple[np.ndarray, tuple]:
    cur = kin.angles(kin.fk(arm, q)[:3, :3])
    want = tuple(
        float(v) if v is not None else c for v, c in zip((pitch, yaw, roll), cur, strict=True)
    )
    t = np.eye(4)
    t[:3, :3] = kin.orientation(*want)
    t[:3, 3] = [float(v) for v in xyz]
    return t, want


def _joint_path_clear(arm: str, q0: Sequence[float], q1: Sequence[float]) -> bool:
    a, b = np.asarray(q0, float), np.asarray(q1, float)
    return all(
        not kin.clearance_violations(arm, a + (b - a) * t) for t in np.linspace(0, 1, 11)[1:]
    )


def _pose_of(arm: str, q: Sequence[float]) -> np.ndarray:
    """The grasp-point pose (rotation of link6, position of the grasp point) of joints ``q``."""

    t = kin.fk(arm, q)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3]
    out[:3, 3] = kin.tcp(t)
    return out


def _plan(
    arm: str, q: Sequence[float], target: np.ndarray, path: str, start_joints
) -> tuple[list, str | None, str | None, str | None]:
    """(joint waypoints, the path taken, why nothing can be done, a note) for one target pose.

    The pose itself is solved first, so a refusal says whether the TARGET is out of reach or
    only the straight line to it.  A target missed by at most NEAR_M / NEAR_DEG is replaced by
    the closest reachable pose, and the note says so.
    """

    res = kin.best_solution(arm, target, q, extra_seeds=[start_joints[arm]])
    near = res.pos_err_m <= NEAR_M and res.rot_err_deg <= NEAR_DEG
    if not res.ok and not near:
        return [], None, _unreachable(res), None
    if res.collides:
        return [], None, "every solution puts part of the arm into the table", None
    note = None
    goal = target
    if not res.ok:
        goal = _pose_of(arm, res.q)
        note = (
            f"the exact pose is out of reach; went to the closest one, {res.pos_err_m:.3f} m "
            f"and {res.rot_err_deg:.0f} deg from it"
        )
    if path in ("line", "auto"):
        ways, why = kin.line_path(arm, q, goal)
        if why is None:
            return _followable(q, ways), "line", None, note
        if path == "line":
            return [], "line", why + " (the target itself is reachable: path='auto' swings)", None
    if not _joint_path_clear(arm, q, res.q):
        return [], "joint", "the joint-space swing to this pose passes through the table", None
    return _followable(q, [res.q]), "joint", None, note


def _unreachable(res: kin.IKResult) -> str:
    return (
        f"out of reach: the closest pose the arm can take is {res.pos_err_m:.3f} m and "
        f"{res.rot_err_deg:.0f} deg away"
    )


def _alternatives(
    arm: str, q: Sequence[float], xyz: Sequence[float], want: tuple, start_joints
) -> list[str]:
    """Nearby poses that ARE reachable: a less steep pitch here, or this pose higher / lower."""

    def ok(p: tuple[float, float, float], angles: tuple[float, float, float]) -> bool:
        t = np.eye(4)
        t[:3, :3] = kin.orientation(*angles)
        t[:3, 3] = p
        res = kin.best_solution(arm, t, q, extra_seeds=[start_joints[arm]])
        return res.ok and not res.collides

    out = []
    pitch, yaw, roll = want
    for p in (60.0, 45.0, 30.0, 0.0):
        if p < pitch - 5.0 and ok(tuple(xyz), (p, yaw, roll)):
            out.append(f"pitch={p:.0f} at this position")
            break
    for dz in (-0.03, -0.06, -0.10, 0.03, 0.06):
        z = xyz[2] + dz
        if z > kin.TABLE_Z + 0.015 and ok((xyz[0], xyz[1], z), want):
            out.append(f"z={z:.3f} with this orientation")
            break
    return out


def _followable(q: Sequence[float], ways: Sequence[Sequence[float]]) -> list[list[float]]:
    """Waypoints split into steps the simulator can follow in one action each.

    It turns a joint about 0.2 rad per action; an arm that then falls behind its waypoints is
    one that something stopped.
    """

    out: list[list[float]] = []
    a = np.asarray(q, float)
    for way in ways:
        b = np.asarray(way, float)
        n = max(1, math.ceil(float(np.max(np.abs(b - a))) / JOINT_STEP_RAD))
        out.extend(list(a + (b - a) * (i / n)) for i in range(1, n + 1))
        a = b
    return out


def move_pose(
    primitives: Any,
    start_joints: dict[str, list[float]],
    xyz: Sequence[float],
    arm: str = "right",
    gripper: float = 0,
    pitch: float | None = None,
    yaw: float | None = None,
    roll: float | None = None,
    path: str = "auto",
) -> dict[str, Any]:
    """Move one arm's grasp point to ``xyz`` and an orientation: the tool behind ``move_to``.

    ``pitch`` / ``yaw`` / ``roll`` in degrees (:mod:`.robodojo_kinematics`); an angle left None
    keeps its current value, so a plain position move no longer turns the wrist.  The two
    fingers are interchangeable, so the target turned half a turn about the pointing axis is
    tried too and the one needing less joint travel is taken.  ``path``: ``"line"`` moves the
    grasp point along a straight line (waypoints every 2.5 cm or 8 degrees, each solved from the
    last), ``"joint"`` swings the joints straight to the solution, ``"auto"`` takes the line and
    falls back to the swing when the line cannot be followed.  Nothing moves when the target is
    out of reach or the motion would take the arm into the table: the result says why.
    """

    if arm not in ARMS:
        return {"error": "arm must be 'left' or 'right'"}
    if path not in ("auto", "line", "joint"):
        return {"error": "path must be 'auto', 'line' or 'joint'"}
    try:
        xyz = [float(v) for v in xyz]
    except (TypeError, ValueError):
        return {"error": "xyz must be three numbers"}
    if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
        return {"error": "xyz must be three finite numbers"}
    grip = _command(gripper)
    state = _obs(primitives)["state"]
    q = _joints(state, arm)
    start = _tcp(arm, q)
    target, want = _target(arm, q, xyz, pitch, yaw, roll)
    flipped, _ = _target(arm, q, xyz, want[0], want[1], want[2] + 180.0)
    # The fingers are interchangeable only while they hold nothing: with an object in the hand
    # the half-turned pose turns the object (172 of 3373 moves made holding something arrived
    # turned 180 degrees: pours tipped the wrong way, a laptop stood up upside down).
    empty = _opening(state, arm) >= OPEN_FOR_HALF_TURN and grip != 0.0
    options = []
    first_why: str | None = None
    for t in (target, flipped) if empty else (target,):
        ways, taken, why, note = _plan(arm, q, t, path, start_joints)
        if why is None:
            travel = float(np.sum(np.abs(np.asarray(ways[-1]) - np.asarray(q))))
            # an exact solution beats a near miss
            options.append((note is not None, travel, ways, taken, t, note))
        elif first_why is None or first_why.startswith("out of reach"):
            first_why = why
    result: dict[str, Any] = {
        "arm": arm,
        "target_xyz": [round(v, 4) for v in xyz],
        "target_angles": {
            "pitch": round(want[0], 1),
            "yaw": round(want[1], 1),
            "roll": round(want[2], 1),
        },
        "start_xyz": start,
        "steps_used": 0,
        "path": None,
    }
    if not options:
        if first_why and first_why.startswith("out of reach"):
            alts = _alternatives(arm, q, xyz, want, start_joints)
            first_why += (
                "; reachable instead: " + ", or ".join(alts)
                if alts
                else "; nothing close is reachable either: use the other arm or move the object"
            )
        now = _pose_summary(arm, q)
        result.update(
            final_xyz=now["xyz"],
            final_angles={k: now[k] for k in ("pitch", "yaw", "roll")},
            dist_to_target_m=round(float(np.linalg.norm(np.asarray(now["xyz"]) - xyz)), 4),
            orientation_error_deg=round(
                kin.rotation_error(kin.fk(arm, q)[:3, :3], target[:3, :3]), 1
            ),
            reached=False,
            ik_error=first_why,
        )
        return result
    _near, _travel, ways, taken, goal, note = min(options, key=lambda o: (o[0], o[1]))
    result["path"] = taken
    if goal is flipped:
        half = (
            "took the pose turned half a turn about the pointing direction (the empty "
            "fingers are interchangeable): yaw / roll read 180 off"
        )
        note = f"{note}; {half}" if note else half
    if note:
        result["note"] = note
    steps = 0
    status: dict[str, Any] = {}
    for way in ways:
        primitives._check_cancelled()
        state, status = _act(primitives, state, {arm: way}, {arm: grip})
        steps += 1
        if _spent(status):
            break
    final_q = ways[-1]
    for _ in range(SETTLE_ACTIONS):
        if _spent(status):
            break
        err = max(abs(a - b) for a, b in zip(_joints(state, arm), final_q, strict=True))
        if err < 0.01:
            break
        primitives._check_cancelled()
        state, status = _act(primitives, state, {arm: final_q}, {arm: grip})
        steps += 1
    q_now = _joints(state, arm)
    now = _pose_summary(arm, q_now)
    dist = float(np.linalg.norm(np.asarray(kin.tcp(kin.fk(arm, q_now))) - np.asarray(xyz)))
    turn = kin.rotation_error(kin.fk(arm, q_now)[:3, :3], goal[:3, :3])
    result.update(
        final_xyz=now["xyz"],
        final_angles={k: now[k] for k in ("pitch", "yaw", "roll")},
        dist_to_target_m=round(dist, 4),
        orientation_error_deg=round(turn, 1),
        reached=bool(dist <= ARRIVE_M and turn <= ARRIVE_DEG),
        steps_used=steps,
        ik_error=None,
    )
    # behind the planned pose by more than a few mm: a joint lag of 0.03 rad is already ~1 cm
    # at the grasp point (a press stalled 1.5 cm short read as no lag at all)
    planned = kin.tcp(kin.fk(arm, final_q))
    lag_m = float(np.linalg.norm(np.asarray(kin.tcp(kin.fk(arm, q_now))) - planned))
    if lag_m > 0.005 and not _spent(status):
        result["blocked"] = "the arm stopped short of the planned pose: something is in the way"
    return result


def grip(
    primitives: Any, arm: str = "right", gripper: float = 1, opening: float | None = None
) -> dict[str, Any]:
    """Close (1) or open (-1) one gripper: repeat the command until the fingers stop.

    ``opening`` (0..1, with an open command only) opens to that fraction of the full gap
    instead of all the way, e.g. to reach between close neighbours.
    """

    if arm not in ARMS:
        return {"error": "arm must be 'left' or 'right'"}
    command = _command(gripper)
    if command is None:
        return {"error": "gripper must be 1 (close) or -1 (open)"}
    if opening is not None:
        try:
            opening = float(opening)
        except (TypeError, ValueError):
            return {"error": "opening must be a number between 0 and 1"}
        if command < 0.5 or not 0.0 < opening <= 1.0:
            return {"error": "opening is for an open command, between 0 (excluded) and 1"}
        command = opening
    state = _obs(primitives)["state"]
    hold = {a: _joints(state, a) for a in ARMS}  # both arms keep the joints they had
    before = _opening(state, arm)
    steps = 0
    status: dict[str, Any] = {}
    last = before
    for _ in range(GRIP_MAX_ACTIONS):
        primitives._check_cancelled()
        state, status = _act(primitives, state, hold, {arm: command})
        steps += 1
        now = _opening(state, arm)
        if steps >= 2 and abs(now - last) < GRIP_STILL:
            break
        last = now
        if _spent(status):
            break
    measured = _opening(state, arm)
    closing = command == 0.0
    holding = bool(closing and measured > HOLD_MIN_OPENING)
    width = measured * FULL_GAP_M
    return {
        "arm": arm,
        "gripper": "closed" if closing else "open",
        "opening": round(measured, 3),
        "opening_before": round(before, 3),
        "holding": holding,
        "width_m": round(max(width, 0.0), 4) if closing else None,
        "steps_used": steps,
        "status": {k: status[k] for k in ("step", "step_limit") if k in status},
    }


def home(
    primitives: Any,
    start_joints: dict[str, list[float]],
    arm: str = "both",
    tol: float = HOME_TOL_RAD,
    max_steps: int = HOME_MAX_ACTIONS,
    lift: bool = True,
) -> dict[str, Any]:
    """Drive one arm or both back to ``start_joints``: the tool behind ``robo.home``.

    With ``lift``, an arm away from its start whose grasp point is below HOME_LIFT_BELOW first
    goes straight up by HOME_LIFT_M (or less, where the line ends sooner), orientation kept.

    The same joint action as ``move_pose``, the other arm held where it is and each gripper
    commanded its measured opening, with the joint target given instead of solved for.  The
    simulator moves a joint about 0.2 rad in one action: a return from a short reach took 5,
    and of 135 returns in one sweep half were still under way after 10 (the limit then).
    """

    if arm not in (*ARMS, "both"):
        return {"error": "arm must be 'left', 'right' or 'both'"}
    arms = ARMS if arm == "both" else (arm,)

    def error_of(state: dict[str, Any]) -> float:
        gaps = [0.0]
        for name in arms:
            gaps += [
                abs(a - b) for a, b in zip(_joints(state, name), start_joints[name], strict=True)
            ]
        return max(gaps)

    state = _obs(primitives)["state"]
    error = error_of(state)
    steps_used = 0
    status: dict[str, Any] = {}
    lifted: dict[str, float] = {}
    for name in arms if lift and error > tol else ():
        q = _joints(state, name)
        if max(abs(a - b) for a, b in zip(q, start_joints[name], strict=True)) <= tol:
            continue
        pose = _pose_of(name, q)
        if pose[2, 3] >= HOME_LIFT_BELOW:
            continue
        for dz in (HOME_LIFT_M, HOME_LIFT_M * 2 / 3, HOME_LIFT_M / 3):
            up = pose.copy()
            up[2, 3] += dz
            ways, why = kin.line_path(name, q, up)
            if why is None:
                for way in _followable(q, ways):
                    primitives._check_cancelled()
                    state, status = _act(primitives, state, {name: way}, {})
                    steps_used += 1
                    if _spent(status):
                        break
                lifted[name] = round(dz, 3)
                break
        if _spent(status):
            break
    error = error_of(state)
    swing = 0
    while error > tol and swing < max_steps and not _spent(status):
        primitives._check_cancelled()
        state, status = _act(primitives, state, {name: start_joints[name] for name in arms}, {})
        steps_used += 1
        swing += 1
        error = error_of(state)
    out = {
        "arm": arm,
        "reached": error <= tol,
        "max_joint_error_rad": round(error, 4),
        "steps_used": steps_used,
    }
    if lifted:
        out["lifted_m"] = lifted
    return out


# ---------------------------------------------------------------------------
# perception
# ---------------------------------------------------------------------------


def _obs_state(primitives: Any) -> dict[str, Any]:
    obs = getattr(primitives, "_last_obs", None) or _obs(primitives)
    return obs.get("state") or {}


def _on_robot(state: dict[str, Any], points: np.ndarray, radius: float = 0.045) -> np.ndarray:
    """Which world points lie on either arm: within ``radius`` of its links (base to the tips)."""

    hit = np.zeros(len(points), dtype=bool)
    for arm in ARMS:
        try:
            q = _joints(state, arm)
        except (KeyError, TypeError, ValueError):
            continue
        if len(q) != 6:
            continue
        fr = kin.frames(arm, q)
        chain = [f[:3, 3] for f in fr]
        chain.append(fr[-1][:3, 3] + fr[-1][:3, 0] * kin.TIP_OFFSET)
        for a, b in zip(chain, chain[1:], strict=False):
            ab = b - a
            n = float(ab @ ab)
            t = np.clip(((points - a) @ ab) / n, 0.0, 1.0) if n > 1e-12 else np.zeros(len(points))
            d = np.linalg.norm(points - (a + t[:, None] * ab), axis=1)
            hit |= d < radius
    return hit


def _camera(obs: dict[str, Any], camera: str) -> dict[str, Any] | None:
    return (obs.get("vision") or {}).get(camera)


def world_points(
    cam: dict[str, Any], rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World points of pixels through the depth image, and which pixels had a valid depth.

    The same projection as RPent's ``back_project``: Isaac cameras look along -Z and the
    depth (``distance_to_image_plane``) is along the optical axis.
    """

    depth = cam.get("distance_to_image_plane")
    if depth is None:
        depth = cam.get("depth")
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    k = np.asarray(cam["intrinsic_matrix"], dtype=np.float64)
    t = np.asarray(cam["extrinsic_matrix"], dtype=np.float64)
    d = depth[rows, cols]
    valid = np.isfinite(d) & (d > 0)
    r, c, d = rows[valid], cols[valid], d[valid]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    p = np.stack([(c - cx) / fx * d, -(r - cy) / fy * d, -d, np.ones_like(d)])
    return (t @ p)[:3].T, valid


def object_geometry(points: np.ndarray) -> dict[str, Any] | None:
    """Extent, heading and heights of an object from its visible surface points (N x 3)."""

    pts = points[points[:, 2] > kin.TABLE_Z + 0.004]  # the table seen through gaps in the mask
    if len(pts) < 12:
        return None
    med = np.median(pts[:, :2], axis=0)
    spread = np.median(np.linalg.norm(pts[:, :2] - med, axis=1))
    keep = np.linalg.norm(pts[:, :2] - med, axis=1) <= max(4.0 * spread, 0.03)
    pts = pts[keep]  # pixels of other things at the mask's edge
    xy = pts[:, :2]
    centre = xy.mean(axis=0)
    cov = np.cov((xy - centre).T)
    w, v = np.linalg.eigh(cov)
    major = v[:, int(np.argmax(w))]
    minor = np.array([-major[1], major[0]])
    along = (xy - centre) @ major
    across = (xy - centre) @ minor
    lo_a, hi_a = np.percentile(along, [2, 98])
    lo_c, hi_c = np.percentile(across, [2, 98])
    mid = centre + major * (lo_a + hi_a) / 2 + minor * (lo_c + hi_c) / 2
    top = float(np.percentile(pts[:, 2], 97))
    heading = math.degrees(math.atan2(major[1], major[0]))
    heading = (heading + 90.0) % 180.0 - 90.0  # an axis: -90 .. 90 from +x
    across_heading = (heading + 180.0) % 180.0 - 90.0
    return {
        "center_xyz": [
            round(float(mid[0]), 4),
            round(float(mid[1]), 4),
            round((top + kin.TABLE_Z) / 2, 4),
        ],
        "top_z": round(top, 4),
        "height_m": round(top - kin.TABLE_Z, 4),
        "length_m": round(float(hi_a - lo_a), 4),
        "width_m": round(float(hi_c - lo_c), 4),
        "long_axis_deg": round(heading, 1),
        # pointing straight down, this yaw closes the fingers across the narrow side
        "grasp_yaw": round(across_heading, 1),
        "points": int(len(pts)),
    }


def locate(
    primitives: Any,
    text_prompt: str,
    camera: str = "cam_head",
    min_score: float = 0.2,
) -> dict[str, Any]:
    """Segment an object by text and measure it in 3D from the depth image."""

    if not isinstance(text_prompt, str) or not text_prompt.strip():
        return {"error": "text_prompt must be a non-empty string"}
    cam = _camera(_obs(primitives), camera)
    if cam is None:
        return {"error": f"camera {camera!r} not in observation"}
    color = cam.get("color")
    if color is None:
        return {"error": "camera color image missing"}
    sam3 = getattr(primitives, "sam3_client", None)
    if sam3 is None:
        return {"error": "sam3_client not configured"}
    res = sam3.segment(color, text_prompt=text_prompt, min_score=float(min_score))
    if not res.found:
        return {"camera": camera, "found": False, "text_prompt": text_prompt}
    mask = np.asarray(res.mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    out: dict[str, Any] = {
        "camera": camera,
        "found": True,
        "text_prompt": text_prompt,
        "score": round(float(res.score), 3),
        "area_px": int(mask.sum()),
        "centroid_rc": [int(rows.mean()), int(cols.mean())] if len(rows) else None,
    }
    try:
        points, _valid = world_points(cam, rows, cols)
    except Exception as exc:  # noqa: BLE001 - a missing depth image is a result, not a crash
        out["geometry_error"] = f"no depth: {exc}"
        return out
    on_robot = _on_robot(_obs_state(primitives), points)
    if on_robot.any():
        out["robot_px_fraction"] = round(float(on_robot.mean()), 3)
        points = points[~on_robot]
    geo = object_geometry(points)
    if geo is None:
        out["geometry_error"] = (
            "the mask lies on the robot's own arm or gripper"
            if out.get("robot_px_fraction", 0) > 0.5
            else "too few points above the table (flat, far or hidden)"
        )
        return out
    out.update(geo)
    return out


#: Changes nothing in the scene: RPent's toolkit records no step after it (its ``readonly``).
locate._readonly = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_ARM = {"type": "string", "enum": list(ARMS)}
SPECS: dict[str, dict[str, Any]] = {
    "move_pose": {
        "name": "move_pose",
        "description": "Move an arm's grasp point to a position and orientation (degrees) with "
        "the host's IK; angles left out keep their current value.",
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {"type": "array", "items": {"type": "number"}},
                "arm": _ARM,
                "gripper": {"type": "number"},
                "pitch": {"type": "number"},
                "yaw": {"type": "number"},
                "roll": {"type": "number"},
                "path": {"type": "string", "enum": ["auto", "line", "joint"]},
            },
            "required": ["xyz"],
        },
    },
    "grip": {
        "name": "grip",
        "description": "Close (1) or open (-1) a gripper fully and report what it holds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": _ARM,
                "gripper": {"type": "number"},
                "opening": {"type": "number"},
            },
            "required": ["gripper"],
        },
    },
    "locate": {
        "name": "locate",
        "description": "Segment an object by text and measure its extent, heading and height.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text_prompt": {"type": "string"},
                "camera": {"type": "string"},
                "min_score": {"type": "number"},
            },
            "required": ["text_prompt"],
        },
    },
    "home": {
        "name": "home",
        "description": "Return one arm or both to the joint angles the episode started with; "
        "gripper openings are kept.",
        "input_schema": {
            "type": "object",
            "properties": {"arm": {"type": "string", "enum": [*ARMS, "both"]}},
            "required": [],
        },
    },
}


def register(toolkit: Any) -> None:
    """Add this module's tools to a freshly booted RoboDojo toolkit.

    The joint angles of ``home`` are read here: boot leaves both arms at their origin.  RPent's
    toolkit records a step after every handler except ``locate`` (marked read-only; RPent
    looks through the ``partial``).
    """

    primitives = toolkit._primitives
    state = primitives.env.get_obs().get("state", {})
    start = {arm: _joints(state, arm) for arm in ARMS}
    handlers = {
        "move_pose": partial(move_pose, primitives, start),
        "grip": partial(grip, primitives),
        "locate": partial(locate, primitives),
        "home": partial(home, primitives, start),
    }
    for name, handler in handlers.items():
        toolkit.add_tool(name, SPECS[name], handler)


__all__ = [
    "FULL_GAP_M",
    "HOLD_MIN_OPENING",
    "HOME_MAX_ACTIONS",
    "HOME_TOL_RAD",
    "SPECS",
    "grip",
    "home",
    "locate",
    "move_pose",
    "object_geometry",
    "register",
    "world_points",
]
