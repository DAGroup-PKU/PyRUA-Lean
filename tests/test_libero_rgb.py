# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LIBERO with the colour-only primitives of robodojo_xpl_rgb: the tools against a fake
LIBERO env (an OSC_POSE integrator, a gripper, two pinhole cameras that see one box), and the
robot class against a fake backend."""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pyrualean.hosts import libero_sight as sight
from pyrualean.hosts import libero_tools as tools
from pyrualean.libero_rgb import LiberoColourRobot, ObjectSight
from pyrualean.prompt import api_reference, knowledge_text
from pyrualean.robots import get_adapter

TABLE_Z = 0.0
#: The box: centre on the table, half sizes (x, y, z), turned this much about z.
BOX_CENTRE = (0.05, -0.08, TABLE_Z + 0.03)
BOX_HALF = (0.04, 0.02, 0.03)
BOX_YAW_DEG = 30.0
AGENTVIEW_META = {
    "camera_name": "agentview",
    "height": 1024,
    "width": 1024,
    "intrinsic_K": [[1236.0773, 0.0, 512.0], [0.0, 1236.0773, 512.0], [0.0, 0.0, 1.0]],
    "extrinsic_cam2world": [
        [0.0, 0.5287697, -0.8487653, 0.8965774],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -0.8487653, -0.5287697, 0.65],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "depth_near": 0.0106,
    "depth_far": 530.49,
}
WRIST_META = {
    "camera_name": "robot0_eye_in_hand",
    "height": 1024,
    "width": 1024,
    "intrinsic_K": [[667.2514, 0.0, 512.0], [0.0, 667.2514, 512.0], [0.0, 0.0, 1.0]],
    "extrinsic_cam2world": np.eye(4).tolist(),
}
START_POS = (-0.148, 0.0, 0.261)
#: robosuite's start: the hand pointing down, tilted 3 degrees ([x, y, z, w]).
START_QUAT = (0.9995966, 0.0002462, -0.0284001, -0.000007)
#: The mounted Panda's starting joints (liberopro MountedPanda.init_qpos).
START_JOINTS = (0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, math.pi / 4)


def _quat_of(rot: np.ndarray) -> list[float]:
    m = rot
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    y = math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    z = math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
    x = math.copysign(x, m[2, 1] - m[1, 2])
    y = math.copysign(y, m[0, 2] - m[2, 0])
    z = math.copysign(z, m[1, 0] - m[0, 1])
    return [x, y, z, w]


def _axis_angle(v: np.ndarray) -> np.ndarray:
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx


def _box_mask(view: sight.View) -> np.ndarray:
    """The box as a camera sees it: pixels whose ray meets it (a ray-box test per pixel)."""

    h, w = view.shape
    rows, cols = np.mgrid[0:h, 0:w]
    d = view.rays(rows.ravel(), cols.ravel())
    c, s = math.cos(math.radians(BOX_YAW_DEG)), math.sin(math.radians(BOX_YAW_DEG))
    to_box = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # world -> box frame
    o = (view.pos - np.asarray(BOX_CENTRE)) @ to_box.T
    dl = d @ to_box.T
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = (-np.asarray(BOX_HALF) - o) / dl
        hi = (np.asarray(BOX_HALF) - o) / dl
    near = np.nanmax(np.minimum(lo, hi), axis=1)
    far = np.nanmin(np.maximum(lo, hi), axis=1)
    return ((far >= near) & (far > 0)).reshape(h, w)


class FakeEnv:
    """LIBERO as RPent's env client shows it: raw_obs, step, render_camera, camera meta."""

    def __init__(self) -> None:
        self.pos = np.asarray(START_POS, dtype=float)
        self.rot = sight.quat_matrix(START_QUAT)
        self.opening = 0.078
        self.terminated = False
        self.truncated = False
        self.steps = 0
        self.wall_z: float | None = None  # the hand cannot go below this (an obstacle)
        self.joints = np.asarray(START_JOINTS, dtype=float)
        self.joint_rate: np.ndarray | None = None  # joint change per step (no arm model here)

    def raw_obs(self) -> dict[str, Any]:
        return {
            "robot0_eef_pos": self.pos.tolist(),
            "robot0_eef_quat": _quat_of(self.rot),
            "robot0_gripper_qpos": [self.opening / 2, -self.opening / 2],
            "robot0_joint_pos": self.joints.tolist(),
        }

    def step(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=float)
        self.pos = self.pos + np.clip(a[:3], -1, 1) * tools.POS_SCALE_M * 0.9
        if self.wall_z is not None:
            self.pos[2] = max(self.pos[2], self.wall_z)
        self.rot = _axis_angle(np.clip(a[3:6], -1, 1) * tools.ROT_SCALE_RAD * 0.9) @ self.rot
        target = 0.0 if a[6] > 0 else 0.08
        floor = 0.03 if getattr(self, "holds", False) and a[6] > 0 else 0.0
        self.opening = max(
            floor, self.opening + float(np.clip(target - self.opening, -0.007, 0.007))
        )
        if self.joint_rate is not None:
            self.joints = self.joints + self.joint_rate
        self.steps += 1

    def get_camera_meta(self, camera_name: str, height: int, width: int) -> dict[str, Any]:
        meta = dict(AGENTVIEW_META if camera_name == "agentview" else WRIST_META)
        k = np.asarray(meta["intrinsic_K"], dtype=float) * (height / meta["height"])
        k[2, 2] = 1.0
        meta.update(intrinsic_K=k.tolist(), height=height, width=width)
        return meta

    def render_camera(self, camera_name: str, height: int, width: int, depth: bool = False):
        name = "agentview" if camera_name == "agentview" else "wrist"
        raw = self.raw_obs()
        view = (
            tools._view(self._primitives, name, raw)
            if name == "wrist"
            else sight.view_from_meta(self.get_camera_meta("agentview", height, width))
        )
        mask = _box_mask(view)
        rgb = np.full((height, width, 3), 200, dtype=np.uint8)
        rgb[mask] = (200, 30, 30)
        rgb = rgb[::-1]  # the raw buffer is upside down
        if not depth:
            return rgb
        # a depth frame of the bare table, robosuite-normalised
        near, far = 0.0106, 530.49
        v = sight.view_from_meta(self.get_camera_meta("agentview", height, width))
        rows, cols = np.mgrid[0:height, 0:width]
        d = v.rays(rows.ravel(), cols.ravel())
        t = (TABLE_Z - v.pos[2]) / d[:, 2]
        metric = np.where(t > 0, t, far).reshape(height, width)
        norm = (1.0 - near / metric) / (1.0 - near / far)
        return rgb, norm[::-1]


class FakeSam3:
    """SAM3 that finds the red box by any prompt, or by a point on it."""

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        mask = np.all(np.asarray(image) == (200, 30, 30), axis=-1)
        if point is not None and not mask[point[0], point[1]]:
            return SimpleNamespace(found=False, mask=None, score=0.0)
        return SimpleNamespace(found=bool(mask.any()), mask=mask, score=0.9)


class FakeToolkit:
    def __init__(self) -> None:
        self.env = FakeEnv()
        self._primitives = SimpleNamespace(
            env=self.env,
            _sam3_client=FakeSam3(),
            _check_cancelled=lambda: None,
            _step_env=self.env.step,
        )
        self.env._primitives = self._primitives
        self.tools: dict[str, Any] = {}

    def add_tool(self, name, spec, handler):
        self.tools[name] = handler


@pytest.fixture
def kit() -> FakeToolkit:
    k = FakeToolkit()
    record = tools.register(k)
    assert abs(record["table_z"] - TABLE_Z) < 0.003
    return k


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------


def test_orientation_and_angles_agree_away_from_vertical():
    for p, y, r in itertools.product([85, 60, 30, 0], [-120, -45, 0, 30, 90, 170], [-60, 0, 45]):
        rot = tools.orientation(p, y, r)
        assert tools.rotation_error(tools.orientation(*tools.angles(rot)), rot) < 1e-4


def test_straight_down_yaw_is_the_line_the_fingers_close_along():
    rot = tools.orientation(90, 30, 0)
    assert np.allclose(rot[:, tools.POINT_AXIS], [0, 0, -1], atol=1e-9)
    jaw = rot[:, sight.FINGER_AXIS]
    assert math.isclose(math.degrees(math.atan2(jaw[1], jaw[0])), 30.0, abs_tol=1e-6)
    assert tools.angles(rot) == pytest.approx((90.0, 30.0, 0.0), abs=1e-6)


def test_tilted_roll_0_closes_along_the_heading_and_roll_90_across_it():
    along = tools.orientation(60, 0, 0)[:, sight.FINGER_AXIS]
    assert abs(along[1]) < 1e-9 and abs(along[2]) > 0.4  # in the vertical plane through +x
    across = tools.orientation(60, 0, 90)[:, sight.FINGER_AXIS]
    assert abs(across[2]) < 1e-9 and abs(abs(across[1]) - 1.0) < 1e-9  # level, along y
    _pitch, _yaw, roll = tools.angles(sight.quat_matrix(START_QUAT))
    assert abs(abs(roll) - 90.0) < 2.0


def test_start_pose_reads_pointing_down():
    pitch, _yaw, _roll = tools.angles(sight.quat_matrix(START_QUAT))
    assert 85.0 < pitch <= 90.0


def test_finger_axis_is_the_one_the_panda_fingers_slide_along():
    """Against robosuite's own Panda model, not this module's conventions: at the starting
    joints the hand body (whose orientation is ``robot0_eef_quat``) has START_QUAT, and the two
    fingers stand apart along its FINGER_AXIS."""

    mujoco = pytest.importorskip("mujoco")
    grippers = pytest.importorskip("robosuite.models.grippers")
    robots = pytest.importorskip("robosuite.models.robots")
    arm = robots.Panda()
    arm.add_gripper(grippers.PandaGripper(idn=0))
    model = mujoco.MjModel.from_xml_string(arm.get_xml())
    data = mujoco.MjData(model)
    data.qpos[:7] = START_JOINTS
    data.qpos[7:9] = (0.04, -0.04)
    mujoco.mj_kinematics(model, data)
    hand = data.body("robot0_right_hand").xmat.reshape(3, 3)
    assert tools.rotation_error(hand, sight.quat_matrix(START_QUAT)) < 0.5
    apart = data.body("gripper0_leftfinger").xpos - data.body("gripper0_rightfinger").xpos
    apart = apart / np.linalg.norm(apart)
    assert abs(float(hand[:, sight.FINGER_AXIS] @ apart)) > 0.999


# ---------------------------------------------------------------------------
# motion and gripper
# ---------------------------------------------------------------------------


def test_move_pose_reaches_position_and_orientation(kit):
    out = kit.tools["rgb_move_pose"](xyz=[0.0, 0.1, 0.15], pitch=90, yaw=40)
    assert out["reached"], out
    assert out["dist_to_target_m"] <= 0.01 and out["orientation_error_deg"] <= 5
    # from the start pose (3 degrees off vertical) the fingers end on the line asked for
    assert abs(((out["final_angles"]["yaw"] - 40.0) + 90.0) % 180.0 - 90.0) < 5, out
    assert not out.get("blocked")
    assert out["steps_used"] > 0


def test_move_pose_reports_blocked_when_the_arm_cannot_get_there(kit):
    kit.env.wall_z = 0.12
    out = kit.tools["rgb_move_pose"](xyz=[-0.1, 0.0, 0.05])
    assert not out["reached"] and out.get("blocked")
    assert out["final_xyz"][2] == pytest.approx(0.12, abs=1e-6)


def test_move_pose_refuses_bad_xyz(kit):
    assert "error" in kit.tools["rgb_move_pose"](xyz=[0, 0])
    assert "error" in kit.tools["rgb_move_pose"](xyz=[0, float("nan"), 0])


def test_grip_closes_on_nothing_and_on_something(kit):
    out = kit.tools["rgb_grip"](gripper=1)
    assert out["gripper"] == "closed" and not out["holding"] and out["opening"] < 0.04
    kit.tools["rgb_grip"](gripper=-1)
    kit.env.holds = True
    out = kit.tools["rgb_grip"](gripper=1)
    assert out["holding"] and out["width_m"] == pytest.approx(0.03, abs=0.002)
    # a later motion with KEEP keeps the fingers closed
    kit.tools["rgb_move_pose"](xyz=[-0.1, 0.0, 0.3], gripper=0)
    assert kit.env.opening == pytest.approx(0.03, abs=1e-6)


def test_grip_partial_opening(kit):
    kit.tools["rgb_grip"](gripper=1)
    out = kit.tools["rgb_grip"](gripper=-1, opening=0.5)
    assert 0.45 <= out["opening"] <= 0.65
    assert "error" in kit.tools["rgb_grip"](gripper=1, opening=0.5)


def test_move_reports_a_joint_left_at_its_limit_without_stopping(kit):
    lo = tools.JOINT_LIMITS[4][0]
    kit.env.joints[4] = lo + 0.2
    kit.env.joint_rate = np.array([0.0, 0.0, 0.0, 0.0, -0.03, 0.0, 0.0])
    out = kit.tools["rgb_move_pose"](xyz=[0.0, 0.1, 0.15], pitch=90, yaw=40)
    assert out["reached"], out  # the motion is not cut short
    assert "joint 5 at its lower limit" in out.get("joint_limit", ""), out
    assert "wrist" in out["joint_limit"]
    # an arm clear of every limit reports nothing
    kit.env.joint_rate = None
    kit.env.joints = np.asarray(START_JOINTS, dtype=float)
    out = kit.tools["rgb_move_pose"](xyz=[0.0, 0.1, 0.2], pitch=90, yaw=40)
    assert out["reached"] and "joint_limit" not in out, out


def test_there_is_no_home(kit):
    assert "rgb_home" not in kit.tools
    assert not hasattr(LiberoColourRobot, "home")


def test_close_stopped_almost_fully_open_says_so(kit):
    kit._primitives._step_env = _wedge(kit._primitives._step_env, kit.env)
    out = kit.tools["rgb_grip"](gripper=1)
    assert not out["holding"] and out["opening"] >= tools.HOLD_MAX_OPENING
    assert "wedged" in out.get("note", ""), out


def _wedge(step, env):
    """Fingers that meet something 7.8 cm wide: they cannot close below it."""

    def wedged(action):
        step(action)
        env.opening = max(env.opening, 0.078)

    return wedged


# ---------------------------------------------------------------------------
# perception from colour alone
# ---------------------------------------------------------------------------


def test_locate_measures_a_box_from_agentview_alone(kit):
    out = kit.tools["rgb_locate"](text_prompt="the red box")
    assert out["found"] and out["views"] in (1, 2)
    assert out["center_xyz"][0] == pytest.approx(BOX_CENTRE[0], abs=0.01)
    assert out["center_xyz"][1] == pytest.approx(BOX_CENTRE[1], abs=0.01)


def test_locate_from_above_takes_both_cameras_and_measures_the_box(kit):
    # the wrist camera over the box, pointing down
    kit.tools["rgb_move_pose"](xyz=[BOX_CENTRE[0] - 0.05, BOX_CENTRE[1], 0.16], pitch=90, yaw=0)
    out = kit.tools["rgb_locate"](text_prompt="the red box")
    assert out["views"] == 2, out
    assert out["center_xyz"][0] == pytest.approx(BOX_CENTRE[0], abs=0.006)
    assert out["center_xyz"][1] == pytest.approx(BOX_CENTRE[1], abs=0.006)
    assert out["top_z"] == pytest.approx(TABLE_Z + 2 * BOX_HALF[2], abs=0.006)
    assert out["length_m"] == pytest.approx(2 * BOX_HALF[0], abs=0.008)
    assert out["width_m"] == pytest.approx(2 * BOX_HALF[1], abs=0.008)
    assert abs(((out["long_axis_deg"] - BOX_YAW_DEG) + 90) % 180 - 90) < 6


def test_back_project_on_the_table_and_on_the_box(kit):
    view = sight.view_from_meta(AGENTVIEW_META)
    r, c, _ = view.project(np.array([[0.0, 0.15, TABLE_Z]]))
    out = kit.tools["rgb_back_project"](row=int(round(r[0])), col=int(round(c[0])))
    assert out["surface"] == "table"
    assert out["world_xyz"] == pytest.approx([0.0, 0.15, TABLE_Z], abs=0.004)
    r, c, _ = view.project(np.array([BOX_CENTRE[:2] + (TABLE_Z + 2 * BOX_HALF[2],)]))
    out = kit.tools["rgb_back_project"](row=int(round(r[0])), col=int(round(c[0])))
    assert out["surface"] == "object"
    assert out["world_xyz"][2] > TABLE_Z + 0.01


def test_segment_answers_in_the_policy_images_pixels(kit):
    out = kit.tools["rgb_segment"](text_prompt="red box")
    view = sight.view_from_meta(AGENTVIEW_META)
    r, c, _ = view.project(np.array([BOX_CENTRE]))
    assert out["found"]
    assert abs(out["centroid_rc"][0] - r[0]) < 40 and abs(out["centroid_rc"][1] - c[0]) < 40


def test_wrist_view_follows_the_hand():
    view = sight.wrist_view(np.eye(3), (1024, 1024), START_POS, START_QUAT)
    # the recorded wrist camera at the start pose (from the simulator)
    want = np.array(
        [
            [0.00049, -0.99839, -0.05678, -0.09304],
            [-1.0, -0.00049, 0.0, 0.0],
            [-0.00003, 0.05678, -0.99839, 0.35528],
        ]
    )
    assert np.allclose(view.pose[:3], want, atol=2e-3)


# ---------------------------------------------------------------------------
# the robot class
# ---------------------------------------------------------------------------


class Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def envelope(self) -> dict[str, Any]:
        return {
            "step": 3,
            "task_language": "put the box in the basket",
            "terminated": False,
            "truncated": False,
            "state": {
                "robot0_eef_pos": list(START_POS),
                "robot0_eef_quat": list(START_QUAT),
                "robot0_gripper_qpos": [0.039, -0.039],
                "object_names": ["box_1", "basket_1"],
            },
        }

    def execute_tool(self, name, kwargs):
        self.calls.append((name, dict(kwargs)))
        if name == "view_env_state":
            return SimpleNamespace(result=self.envelope())
        if name == "rgb_locate":
            return SimpleNamespace(
                result={
                    "found": True,
                    "camera": kwargs["camera"],
                    "center_xyz": [0.05, -0.08, 0.03],
                    "top_z": 0.06,
                    "views": 2,
                    "grasp_yaw": -60.0,
                    "score": 0.9,
                    "centroid_rc": [600, 500],
                    "area_px": 4000,
                }
            )
        if name == "rgb_move_pose":
            return SimpleNamespace(
                result={
                    "reached": True,
                    "target_xyz": kwargs["xyz"],
                    "start_xyz": list(START_POS),
                    "final_xyz": kwargs["xyz"],
                    "final_angles": {"pitch": 90.0, "yaw": 0.0, "roll": 0.0},
                    "dist_to_target_m": 0.002,
                    "orientation_error_deg": 1.0,
                    "steps_used": 7,
                    "terminated": False,
                    "truncated": False,
                },
                envelope=self.envelope(),
            )
        raise AssertionError(name)

    def solved(self) -> bool:
        return False

    def env_steps(self):
        return None

    def artifact(self, name, step=-1):
        return b""

    def cancel(self):
        pass


def test_robot_class_maps_onto_the_tools():
    backend = Backend()
    robo = LiberoColourRobot(backend, primitives="no-vla")
    state = robo.state()
    assert state.tcp_pos == pytest.approx(START_POS)
    assert state.gripper == pytest.approx(0.975, abs=1e-3)
    assert 85 < state.pitch <= 90
    seen = robo.locate("the red box")
    assert isinstance(seen, ObjectSight) and seen.views == 2 and seen.grasp_yaw == -60.0
    move = robo.move_to([0.0, 0.1, 0.2], gripper=robo.OPEN, pitch=90)
    assert move.reached and backend.calls[-1] == (
        "rgb_move_pose",
        {"xyz": [0.0, 0.1, 0.2], "gripper": -1, "pitch": 90.0},
    )
    with pytest.raises(ValueError):
        robo.move_to([0, 0, 0], gripper=0.5)
    with pytest.raises(RuntimeError):
        robo.pi0_pick("pick up the box")
    with pytest.raises(ValueError):
        robo.artifact("agentview_depth.npz")
    with pytest.raises(ValueError):
        robo.world_map()


def test_adapter_reference_and_knowledge():
    adapter = get_adapter("libero_rgb")
    assert adapter.robot_cls is LiberoColourRobot and adapter.live_success
    full = api_reference(LiberoColourRobot)
    novla = api_reference(LiberoColourRobot, primitives="no-vla")
    for ref in (full, novla):
        assert "world_map" not in ref and "region_center" not in ref and "rotate_wrist" not in ref
    assert "pi0_pick" in full and "pi0_pick" not in novla and "VLA" not in novla
    for primitives in ("full", "no-vla"):
        text = knowledge_text("libero_rgb", primitives)
        assert "depth image" in text and "world_map" not in text and "region_center" not in text
    assert "pi0_pick" not in knowledge_text("libero_rgb", "no-vla")


def test_no_vla_server_boot_option():
    import argparse
    import inspect
    from pathlib import Path

    from pyrualean.hosts import rpent_libero, rpent_libero_rgb

    parser = argparse.ArgumentParser()
    rpent_libero_rgb.add_boot_args(parser)
    assert rpent_libero_rgb.boot_options(parser.parse_args([]))["vla"] is True
    options = rpent_libero_rgb.boot_options(parser.parse_args(["--no-vla-server"]))
    assert options == {"vla_endpoint": None, "sam3_endpoint": None, "vla": False}
    assert set(options) <= set(inspect.signature(rpent_libero.boot).parameters)
    cell = rpent_libero.Cell("libero_10_swap", 2, 3)
    with pytest.raises(ValueError, match="vla=False"):
        rpent_libero.boot(cell, Path("unused"), vla=False, vla_endpoint="127.0.0.1:1")
