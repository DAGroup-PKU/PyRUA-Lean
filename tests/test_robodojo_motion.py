"""The host's own RoboDojo kinematics and tools, against recorded poses and a joint-level fake."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from pyrualean.hosts import robodojo_kinematics as kin  # noqa: E402
from pyrualean.hosts import robodojo_tools as tools  # noqa: E402
from pyrualean.robodojo import gripper_angles  # noqa: E402

#: (arm, the six joints, the reported end-effector pose [x, y, z, w, qx, qy, qz]) from RoboDojo
#: runs: the start pose and three far from it per arm.
RECORDED = [
    ("left", [0, 0, 0, 0, 0, 0], [-0.299529, -0.3523, 0.9215, 0.707, 0, 0, 0.707214]),
    (
        "left",
        [-0.463398, 1.478828, 1.433324, -1.684324, 0.396853, 0.65755],
        [-0.187647, -0.251156, 1.079924, 0.743492, -0.295802, 0.598789, 0.034236],
    ),
    (
        "left",
        [-0.992635, 1.840495, 1.219056, -0.865604, 0.10427, 0.155253],
        [-0.021588, -0.272402, 0.926642, 0.730915, -0.170437, 0.649313, 0.12291],
    ),
    (
        "left",
        [0.050997, 0.215586, 0.163312, -0.364276, 0.171936, -0.743844],
        [-0.299566, -0.352105, 0.926144, 0.623485, -0.416651, -0.115607, 0.651386],
    ),
    ("right", [0, 0, 0, 0, 0, 0], [0.300471, -0.3523, 0.9215, 0.707, 0, 0, 0.707214]),
    (
        "right",
        [4.589652, 0.134254, 1.759569, 1.221606, -1.130783, -0.163558],
        [-0.081483, -0.377174, 1.165678, 0.080724, -0.58861, -0.80436, 0.005223],
    ),
    (
        "right",
        [-0.163935, 2.309968, 1.016312, 1.604508, -0.169491, -0.958521],
        [0.362536, -0.044351, 0.869725, 0.659467, -0.241512, -0.434965, 0.563543],
    ),
    (
        "right",
        [-0.262874, 1.693729, 1.048803, -0.924472, -0.00046, 0.932941],
        [0.373659, -0.177829, 0.928281, 0.694764, -0.13139, 0.694625, 0.132421],
    ),
]


def _pose(xyz, pitch, yaw, roll=0.0):
    t = np.eye(4)
    t[:3, :3] = kin.orientation(pitch, yaw, roll)
    t[:3, 3] = xyz
    return t


# -- kinematics ------------------------------------------------------------------


def test_fk_reproduces_the_poses_the_simulator_reported():
    for arm, q, pose in RECORDED:
        t = kin.fk(arm, q)
        assert np.linalg.norm(t[:3, 3] - pose[:3]) < 0.001, (arm, q)
        assert kin.rotation_error(t[:3, :3], kin.rot_from_quat(pose[3:])) < 0.5, (arm, q)
    start = kin.fk("left", [0] * 6)
    assert kin.angles(start[:3, :3]) == pytest.approx((0, 0, 0), abs=0.1)  # level, away
    assert kin.tcp(start) == pytest.approx((-0.2995, -0.2073, 0.9215), abs=1e-3)


def test_angles_are_the_convention_the_robot_reports():
    rng = np.random.default_rng(3)
    for _ in range(200):
        pitch, yaw, roll = rng.uniform(-85, 85), rng.uniform(-179, 179), rng.uniform(-179, 179)
        rot = kin.orientation(pitch, yaw, roll)
        assert kin.angles(rot) == pytest.approx((pitch, yaw, roll), abs=1e-6)
        assert gripper_angles(kin.quat_wxyz(rot)) == pytest.approx((pitch, yaw, roll), abs=1e-4)
    down = kin.orientation(90, 30, 0)
    assert down[:, 0] == pytest.approx((0, 0, -1), abs=1e-9)  # pointing straight down
    # the fingers close along the line at ``yaw`` from +x: grasp_yaw is that line
    assert abs(down[:, 1] @ (math.cos(math.radians(30)), math.sin(math.radians(30)), 0)) == (
        pytest.approx(1.0)
    )
    level = kin.orientation(0, 90, 0)
    assert level[:, 0] == pytest.approx((-1, 0, 0), abs=1e-9)  # yaw +90: towards the robot's left


def test_ik_solves_top_grasps_in_front_of_each_arm():
    for arm, x in (("left", -0.2), ("right", 0.2)):
        for yaw in (0.0, 45.0, 90.0):
            target = _pose((x, -0.15, 0.82), 90, yaw)
            res = kin.best_solution(arm, target, [0.0] * 6)
            assert res.ok and not res.collides, (arm, yaw, res)
            end = kin.fk(arm, res.q)
            assert np.linalg.norm(kin.tcp(end) - target[:3, 3]) < 0.002
            assert kin.rotation_error(end[:3, :3], target[:3, :3]) < 1.0


def test_line_path_keeps_the_grasp_point_on_the_line():
    above = kin.best_solution("right", _pose((0.25, -0.15, 0.95), 90, 0), [0.0] * 6)
    q0 = above.q
    p0 = kin.tcp(kin.fk("right", q0))
    target = _pose(p0 + np.array([0.0, 0.05, -0.12]), 90, 0)  # a descent, pointing down
    ways, why = kin.line_path("right", q0, target)
    assert why is None and len(ways) == math.ceil(0.13 / 0.02)
    points = [kin.tcp(kin.fk("right", q)) for q in ways]
    direction = (target[:3, 3] - p0) / np.linalg.norm(target[:3, 3] - p0)
    for prev, point in zip([p0, *points], points, strict=False):
        assert np.linalg.norm(point - prev) <= 0.0205
        off = (point - p0) - ((point - p0) @ direction) * direction
        assert np.linalg.norm(off) < 0.003
    assert np.linalg.norm(points[-1] - target[:3, 3]) < 0.002


def test_poses_in_the_table_are_flagged():
    below = _pose((0.25, -0.15, kin.TABLE_Z - 0.02), 90, 0)
    res = kin.best_solution("right", below, [0.0] * 6)
    assert not res.ok or res.collides
    assert kin.clearance_violations("right", [0.0] * 6) == []


# -- a joint-level simulator fake ------------------------------------------------


class _SimArms:
    """Six-joint arms that turn each joint at most 0.2 rad per action, as RoboDojo's do.

    A gripper moves 0.3 of its range per action towards the command and, closing, stops at
    ``block[arm]`` (an object between the fingers).
    """

    def __init__(self, step_limit: int = 500) -> None:
        self.joints = {arm: [0.0] * 6 for arm in kin.ARMS}
        self.grip = {arm: 1.0 for arm in kin.ARMS}
        self.block: dict[str, float] = {}
        self.step_count, self.step_limit = 0, step_limit
        self.actions: list[dict[str, list[float]]] = []
        self.vision: dict[str, Any] = {}

    def get_obs(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for arm in kin.ARMS:
            state[f"{arm}_arm_joint_state"] = tuple(self.joints[arm])
            state[f"{arm}_ee_joint_state"] = [self.grip[arm]]
        return {"state": state, "vision": self.vision}

    def step(self, action: dict[str, list[float]]):
        self.actions.append(action)
        self.step_count += 1
        for arm in kin.ARMS:
            goal = action[f"{arm}_arm_joint_state"]
            self.joints[arm] = [
                now + max(-0.2, min(0.2, g - now))
                for now, g in zip(self.joints[arm], goal, strict=True)
            ]
            want = action[f"{arm}_ee_joint_state"][0]
            now = self.grip[arm]
            nxt = now + max(-0.3, min(0.3, want - now))
            if want < now and arm in self.block:
                nxt = max(nxt, self.block[arm])
            self.grip[arm] = nxt
        status = {"step": self.step_count, "step_limit": self.step_limit}
        return self.get_obs(), 0.0, False, {"status": status}


class _Primitives:
    def __init__(self, env: _SimArms, sam3: Any = None) -> None:
        self.env = env
        self.sam3_client = sam3
        self._last_obs = None
        self.checked = 0

    def _check_cancelled(self) -> None:
        self.checked += 1


@pytest.fixture
def sim():
    env = _SimArms()
    return env, _Primitives(env), {arm: [0.0] * 6 for arm in kin.ARMS}


def _now(env: _SimArms, arm: str) -> np.ndarray:
    return kin.fk(arm, env.joints[arm])


def test_move_pose_points_down_over_a_spot_and_holds_the_other_arm(sim):
    env, prims, start = sim
    r = tools.move_pose(prims, start, [0.25, -0.15, 0.90], arm="right", pitch=90, yaw=0, gripper=-1)
    assert r["reached"] and r["ik_error"] is None and "blocked" not in r, r
    assert r["path"] in ("line", "joint") and r["steps_used"] == len(env.actions) == prims.checked
    assert r["dist_to_target_m"] <= 0.01 and r["orientation_error_deg"] <= 5.0
    assert r["final_angles"]["pitch"] == pytest.approx(90, abs=5)
    end = _now(env, "right")
    assert kin.tcp(end) == pytest.approx((0.25, -0.15, 0.90), abs=0.01)
    assert end[:3, 0] == pytest.approx((0, 0, -1), abs=0.09)
    for action in env.actions:
        assert action["left_arm_joint_state"] == [0.0] * 6  # the other arm holds
        assert action["right_ee_joint_state"] == [1.0]  # open, as commanded
        assert action["left_ee_joint_state"] == [1.0]  # the measured opening
    for a, b in zip(env.actions, env.actions[1:], strict=False):
        jump = max(
            abs(x - y)
            for x, y in zip(a["right_arm_joint_state"], b["right_arm_joint_state"], strict=True)
        )
        assert jump <= tools.JOINT_STEP_RAD + 1e-9
    # angles left out keep their value: a plain descent stays pointing down
    down = tools.move_pose(prims, start, [0.25, -0.15, 0.84], arm="right", path="line")
    assert (
        down["reached"]
        and down["path"] == "line"
        and down["steps_used"] <= 6 + tools.SETTLE_ACTIONS
    )
    assert down["target_angles"]["pitch"] == pytest.approx(90, abs=5)
    assert _now(env, "right")[:3, 0] == pytest.approx((0, 0, -1), abs=0.09)


def test_move_pose_takes_the_half_turned_grasp_when_it_is_nearer(sim):
    env, prims, start = sim
    first = tools.move_pose(prims, start, [0.25, -0.15, 0.90], arm="right", pitch=90, yaw=0)
    assert first["reached"]
    wrist = env.joints["right"][5]
    again = tools.move_pose(prims, start, [0.25, -0.15, 0.90], arm="right", pitch=90, yaw=180)
    # the same fingers, swapped: no half turn of the wrist
    assert again["reached"] and again["steps_used"] <= tools.SETTLE_ACTIONS
    assert env.joints["right"][5] == pytest.approx(wrist, abs=0.3)
    assert "half a turn" in again["note"]


def test_move_pose_never_swaps_the_fingers_of_a_gripper_that_holds_something(sim):
    env, prims, start = sim
    assert tools.move_pose(prims, start, [0.25, -0.15, 0.90], arm="right", pitch=90, yaw=0)[
        "reached"
    ]
    env.block["right"] = 0.40  # an object between the fingers
    assert tools.grip(prims, "right", 1)["holding"]
    wrist = env.joints["right"][5]
    turn = tools.move_pose(prims, start, [0.25, -0.15, 0.90], arm="right", pitch=90, yaw=90)
    assert turn["reached"] and "note" not in turn
    assert abs(env.joints["right"][5] - wrist) == pytest.approx(math.pi / 2, abs=0.1)  # turned


def test_move_pose_refuses_what_it_cannot_do_and_moves_nothing(sim):
    env, prims, start = sim
    far = tools.move_pose(prims, start, [1.5, 0.0, 0.9], arm="right")
    assert far["reached"] is False and far["steps_used"] == 0 and far["path"] is None
    assert far["ik_error"].startswith("out of reach") and env.actions == []
    assert far["final_xyz"] == pytest.approx(kin.tcp(_now(env, "right")), abs=1e-3)
    sunk = tools.move_pose(prims, start, [0.25, -0.15, kin.TABLE_Z - 0.03], arm="right", pitch=90)
    assert sunk["reached"] is False and sunk["ik_error"] and env.actions == []
    line = tools.move_pose(prims, start, [1.5, 0.0, 0.9], arm="right", path="line")
    assert line["ik_error"].startswith("out of reach")  # the target itself, not the line
    assert "error" in tools.move_pose(prims, start, [0, 0, 1], arm="middle")
    assert "error" in tools.move_pose(prims, start, [0, 0, 1], path="curve")
    assert "error" in tools.move_pose(prims, start, [0, 1], arm="left")
    assert env.actions == []


def test_move_pose_names_what_is_reachable_instead_and_takes_a_near_miss(sim):
    env, prims, start = sim
    high = tools.move_pose(prims, start, [0.30, -0.10, 1.00], arm="right", pitch=90, yaw=0)
    assert high["steps_used"] == 0 and env.actions == []  # straight down is out of reach at z 1.0
    assert high["ik_error"].startswith("out of reach")
    assert "reachable instead: pitch=" in high["ik_error"]
    # a pose the arm misses by a centimetre is taken, and the result says so
    near = None
    for y in np.arange(0.0, 0.2, 0.005):
        t = _pose((0.30, y, 0.80), 90, 0)
        res = kin.best_solution("right", t, [0.0] * 6, extra_seeds=[[0.0] * 6])
        if not res.ok and res.pos_err_m <= 0.012 and res.rot_err_deg <= 5 and not res.collides:
            near = float(y)
            break
    assert near is not None
    r = tools.move_pose(prims, start, [0.30, near, 0.80], arm="right", pitch=90, yaw=0)
    assert r["ik_error"] is None and "closest one" in r["note"] and r["steps_used"] > 0
    assert r["dist_to_target_m"] <= 0.013


def test_move_pose_reports_an_arm_that_was_stopped(sim):
    env, prims, start = sim
    stuck = env.step

    def held_back(action):  # something stops joint 2 of the right arm
        joint = env.joints["right"][1]
        out = stuck(action)
        env.joints["right"][1] = joint
        return out

    env.step = held_back
    r = tools.move_pose(prims, start, [0.25, -0.15, 0.90], arm="right", pitch=90, path="joint")
    assert r["reached"] is False and "something is in the way" in r["blocked"]


def test_keep_squeezes_what_the_last_command_closed_on(sim):
    env, prims, start = sim
    env.block["right"] = 0.40
    assert tools.grip(prims, "right", 1)["holding"]  # fingers stopped at 0.40
    before = len(env.actions)
    assert tools.move_pose(prims, start, [0.25, -0.15, 0.95], arm="right", pitch=90)["reached"]
    keep = [a["right_ee_joint_state"] for a in env.actions[before:]]
    assert keep and all(k == [0.0] for k in keep)  # still commanded closed, not 0.40
    assert all(a["left_ee_joint_state"] == [1.0] for a in env.actions[before:])  # never commanded
    # someone else (the VLA) opened it fully in between: follow, do not close it again
    env.grip["right"] = 1.0
    del env.block["right"]
    tools.move_pose(prims, start, [0.25, -0.15, 0.92], arm="right", path="line")
    assert env.actions[-1]["right_ee_joint_state"] == [1.0]


def test_grip_opens_part_way_and_keep_holds_that_opening(sim):
    env, prims, start = sim
    tools.grip(prims, "right", 1)
    half = tools.grip(prims, "right", -1, opening=0.5)
    assert half["gripper"] == "open" and half["opening"] == pytest.approx(0.5)
    assert half["width_m"] is None and not half["holding"]
    tools.move_pose(prims, start, [0.25, -0.15, 0.90], arm="right", pitch=90)
    assert env.actions[-1]["right_ee_joint_state"] == [0.5]
    assert "error" in tools.grip(prims, "right", 1, opening=0.5)  # a close takes no opening
    assert "error" in tools.grip(prims, "right", -1, opening=0.0)
    assert "error" in tools.grip(prims, "right", -1, opening="wide")


def test_grip_runs_a_full_stroke_and_reports_what_it_holds(sim):
    env, prims, _start = sim
    env.joints["left"] = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
    empty = tools.grip(prims, "right", 1)
    assert empty["gripper"] == "closed" and empty["opening"] == 0.0 and not empty["holding"]
    assert empty["width_m"] == 0.0 and empty["opening_before"] == 1.0
    assert 2 <= empty["steps_used"] <= tools.GRIP_MAX_ACTIONS
    assert empty["status"] == {"step": env.step_count, "step_limit": 500}
    for action in env.actions:  # neither arm moves
        assert action["left_arm_joint_state"] == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        assert action["right_arm_joint_state"] == [0.0] * 6
    opened = tools.grip(prims, "right", -1)
    assert opened["opening"] == 1.0 and not opened["holding"] and opened["width_m"] is None
    env.block["right"] = 0.30  # RoboDojo's 3.5 cm block held across two faces
    held = tools.grip(prims, "right", 1)
    assert held["holding"] and held["opening"] == pytest.approx(0.30)
    assert held["width_m"] == pytest.approx(0.035, abs=0.0005)
    tools.grip(prims, "right", -1)
    env.block["right"] = 0.433  # the same block across its diagonal
    assert tools.grip(prims, "right", 1)["width_m"] == pytest.approx(
        0.035 * math.sqrt(2), abs=0.0005
    )
    tools.grip(prims, "right", -1)
    env.block["right"] = 0.03  # something thin: a coin, a key, a mug's handle
    thin = tools.grip(prims, "right", 1)
    assert thin["holding"] and thin["width_m"] == pytest.approx(0.0056, abs=0.0005)
    assert "error" in tools.grip(prims, "right", 0)


def test_home_lifts_first_then_brings_both_arms_back(sim):
    env, prims, start = sim
    assert tools.move_pose(prims, start, [0.25, -0.15, 0.84], arm="right", pitch=90)["reached"]
    before = len(env.actions)
    back = tools.home(prims, start)
    assert back["reached"] and back["max_joint_error_rad"] <= tools.HOME_TOL_RAD
    assert back["lifted_m"] == {"right": tools.HOME_LIFT_M}
    first = kin.tcp(kin.fk("right", env.actions[before + 1]["right_arm_joint_state"]))
    assert first[2] > 0.84 and first[:2] == pytest.approx((0.25, -0.15), abs=0.005)  # up first
    assert env.joints["right"] == pytest.approx([0.0] * 6, abs=tools.HOME_TOL_RAD)


# -- perception ------------------------------------------------------------------

TABLE_DEPTH = 1.0  # the camera is 1 m above the table, looking straight down


def _overhead(box_center, length, width, height, heading_deg):
    """An overhead camera (Isaac convention: looks along -Z) over one box on the table."""

    rows, cols, f = 240, 320, 300.0
    k = [[f, 0, cols / 2], [0, f, rows / 2], [0, 0, 1]]
    ext = np.eye(4)
    ext[:3, 3] = (box_center[0], box_center[1], kin.TABLE_Z + TABLE_DEPTH)
    r, c = np.mgrid[0:rows, 0:cols]
    top_depth = TABLE_DEPTH - height
    x = (c - cols / 2) / f * top_depth
    y = -(r - rows / 2) / f * top_depth
    h = math.radians(heading_deg)
    along = x * math.cos(h) + y * math.sin(h)
    across = -x * math.sin(h) + y * math.cos(h)
    mask = (np.abs(along) <= length / 2) & (np.abs(across) <= width / 2)
    depth = np.where(mask, top_depth, TABLE_DEPTH)
    cam = {
        "color": np.zeros((rows, cols, 3), np.uint8),
        "distance_to_image_plane": depth,
        "intrinsic_matrix": k,
        "extrinsic_matrix": ext.tolist(),
    }
    return cam, mask


class _Sam3:
    def __init__(self, mask) -> None:
        self.mask = mask
        self.calls: list[tuple[str, float]] = []

    def segment(self, color, *, text_prompt: str, min_score: float):
        self.calls.append((text_prompt, min_score))
        if "block" not in text_prompt:
            return SimpleNamespace(found=False, mask=None, score=0.0)
        return SimpleNamespace(found=True, mask=self.mask, score=0.91)


def test_world_points_follow_rpents_back_projection():
    cam, _mask = _overhead((0.1, -0.1), 0.1, 0.04, 0.05, 0.0)
    pts, valid = tools.world_points(cam, np.array([120, 0]), np.array([160, 0]))
    assert valid.all()
    assert pts[0] == pytest.approx((0.1, -0.1, kin.TABLE_Z + 0.05))  # the centre of the box top
    corner = (0 - 160) / 300.0 * TABLE_DEPTH, -(0 - 120) / 300.0 * TABLE_DEPTH
    assert pts[1] == pytest.approx((0.1 + corner[0], -0.1 + corner[1], kin.TABLE_Z))


def test_locate_measures_a_turned_block_from_its_mask_and_depth():
    cam, mask = _overhead((0.1, -0.1), 0.10, 0.04, 0.05, 30.0)
    env = _SimArms()
    env.vision = {"cam_head": cam}
    sam3 = _Sam3(mask)
    prims = _Primitives(env, sam3)
    obj = tools.locate(prims, "the red block")
    assert obj["found"] and obj["score"] == 0.91 and obj["area_px"] == int(mask.sum())
    assert obj["center_xyz"][:2] == pytest.approx((0.1, -0.1), abs=0.005)
    assert obj["top_z"] == pytest.approx(kin.TABLE_Z + 0.05, abs=0.002)
    assert obj["height_m"] == pytest.approx(0.05, abs=0.002)
    assert obj["center_xyz"][2] == pytest.approx(kin.TABLE_Z + 0.025, abs=0.002)
    assert obj["length_m"] == pytest.approx(0.10, abs=0.01)
    assert obj["width_m"] == pytest.approx(0.04, abs=0.01)
    assert obj["long_axis_deg"] == pytest.approx(30, abs=2)
    assert obj["grasp_yaw"] == pytest.approx(-60, abs=2)  # the fingers close across it
    assert env.actions == [] and sam3.calls == [("the red block", 0.2)]
    assert tools.locate._readonly is True
    missing = tools.locate(prims, "the green bottle")
    assert missing == {"camera": "cam_head", "found": False, "text_prompt": "the green bottle"}
    assert "error" in tools.locate(prims, "the red block", camera="cam_left_wrist")
    assert "error" in tools.locate(_Primitives(env), "the red block")  # no SAM3 service
    flat, flat_mask = _overhead((0.1, -0.1), 0.10, 0.04, 0.001, 0.0)
    env.vision = {"cam_head": flat}
    thin = tools.locate(_Primitives(env, _Sam3(flat_mask)), "a paper block")
    assert thin["found"] and "too few points above the table" in thin["geometry_error"]


def _render(shapes, rows=480, cols=640, f=400.0):
    """The head camera's view of boxes, upright cylinders and cones on the table, ray cast.

    The camera sits where RoboDojo's does (above the robot's edge of the table, looking down
    the table), so it sees an object's top and the sides that face it.  Returns the camera
    dict ``locate`` reads (depth along the optical axis) and, per pixel, the index of the
    shape it shows (-1: the table).
    """

    pos, target = np.array([0.0, -0.41, 1.308]), np.array([0.0, 0.05, kin.TABLE_Z])
    back = (pos - target) / np.linalg.norm(pos - target)  # Isaac cameras look along -Z
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    ext = np.eye(4)
    ext[:3, 0], ext[:3, 1], ext[:3, 2], ext[:3, 3] = right, np.cross(back, right), back, pos
    r, c = np.mgrid[0:rows, 0:cols]
    ray = np.stack([(c - cols / 2) / f, -(r - rows / 2) / f, -np.ones((rows, cols))], axis=-1)
    d = ray @ ext[:3, :3].T  # world direction per unit of depth
    hit = np.full((rows, cols), np.inf)
    owner = np.full((rows, cols), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for index, shape in enumerate(shapes):
            cx, cy = shape["center"]
            z0 = kin.TABLE_Z + shape.get("base", 0.0)
            if "size" in shape:
                lx, ly, h = shape["size"]
                a = math.radians(shape.get("yaw", 0.0))
                to_box = np.array(
                    [[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1]]
                )
                o = to_box @ (pos - np.array([cx, cy, z0 + h / 2]))
                db = d @ to_box.T
                half = np.array([lx / 2, ly / 2, h / 2])
                t1, t2 = (-half - o) / db, (half - o) / db
                near, far = np.minimum(t1, t2).max(axis=-1), np.maximum(t1, t2).min(axis=-1)
                t = np.where((near <= far) & (near > 0), near, np.inf)
            else:  # a cylinder, or a cone frustum when "top_radius" differs: radius = r0 + rk * t
                rad, h = shape["radius"], shape["height"]
                top_rad = shape.get("top_radius", rad)
                slope = (top_rad - rad) / h
                r0, rk = rad + slope * (pos[2] - z0), slope * d[..., 2]
                ox, oy = pos[0] - cx, pos[1] - cy
                qa = d[..., 0] ** 2 + d[..., 1] ** 2 - rk**2
                qb = 2 * (ox * d[..., 0] + oy * d[..., 1] - r0 * rk)
                disc = qb**2 - 4 * qa * (ox**2 + oy**2 - r0**2)
                side = (-qb - np.sqrt(np.where(disc >= 0, disc, np.nan))) / (2 * qa)
                z = pos[2] + side * d[..., 2]
                side = np.where((disc >= 0) & (side > 0) & (z >= z0) & (z <= z0 + h), side, np.inf)
                cap = (z0 + h - pos[2]) / d[..., 2]
                px, py = pos[0] + cap * d[..., 0] - cx, pos[1] + cap * d[..., 1] - cy
                on_cap = (cap > 0) & (px**2 + py**2 <= top_rad**2)
                t = np.minimum(side, np.where(on_cap, cap, np.inf))
            owner = np.where(t < hit, index, owner)
            hit = np.minimum(hit, t)
        table = (kin.TABLE_Z - pos[2]) / d[..., 2]
    owner = np.where(hit < table, owner, -1)
    cam = {
        "color": np.zeros((rows, cols, 3), np.uint8),
        "distance_to_image_plane": np.minimum(hit, table),
        "intrinsic_matrix": [[f, 0, cols / 2], [0, f, rows / 2], [0, 0, 1]],
        "extrinsic_matrix": ext.tolist(),
    }
    return cam, owner


def _head_view(shapes):
    cam, owner = _render(shapes)
    return cam, owner >= 0


def _located(shapes, mask=None):
    cam, seen = _head_view(shapes)
    mask = seen if mask is None else mask
    assert mask.sum() > 200
    env = _SimArms()
    env.vision = {"cam_head": cam}
    return tools.locate(_Primitives(env, _Sam3(mask)), "the block")


@pytest.mark.parametrize(
    ("x", "y", "yaw"),
    [
        (0.378, -0.169, 0.0),
        (0.244, -0.228, 0.0),
        (-0.3, -0.1, 0.0),
        (0.0, -0.1, 0.0),
        (0.3, -0.1, 20.0),
        (-0.2, 0.05, -35.0),
    ],
)
def test_locate_finds_the_faces_of_a_cube_the_head_camera_sees_from_the_side(x, y, yaw):
    # Off the camera's axis two side faces are in view.  Their pixels all fall on two edges of
    # the footprint, and the principal axis of the pixels is then the cube's diagonal: in
    # RoboDojo the blocks of stack_blocks, all square to the table, came back with a
    # grasp_yaw of 44 to 47 degrees.  The fingers have to close across two faces.
    obj = _located([{"center": (x, y), "size": (0.035, 0.035, 0.035), "yaw": yaw}])
    off_a_face = (obj["grasp_yaw"] - yaw) % 90.0
    assert min(off_a_face, 90.0 - off_a_face) <= 3.0
    assert obj["center_xyz"][:2] == pytest.approx((x, y), abs=0.003)
    assert obj["length_m"] == pytest.approx(0.035, abs=0.003)
    assert obj["width_m"] == pytest.approx(0.035, abs=0.003)
    assert obj["top_z"] == pytest.approx(kin.TABLE_Z + 0.035, abs=0.002)


def test_locate_measures_a_long_box_the_head_camera_sees_from_the_side():
    obj = _located([{"center": (0.25, -0.1), "size": (0.10, 0.04, 0.05), "yaw": 30.0}])
    assert obj["long_axis_deg"] == pytest.approx(30, abs=2)
    assert obj["grasp_yaw"] == pytest.approx(-60, abs=2)
    assert obj["length_m"] == pytest.approx(0.10, abs=0.004)
    assert obj["width_m"] == pytest.approx(0.04, abs=0.004)
    assert obj["center_xyz"][:2] == pytest.approx((0.25, -0.1), abs=0.003)


def test_locate_completes_the_far_side_of_a_round_object():
    # A bottle: the camera sees the near half of the body and shoulder and all of the narrow
    # top, so the visible points end at the neck.  Each slice of the body is an arc, which is
    # the circle.
    centre = (0.3, 0.0)
    bottle = [
        {"center": centre, "radius": 0.035, "height": 0.10},
        {"center": centre, "radius": 0.035, "top_radius": 0.010, "height": 0.03, "base": 0.10},
        {"center": centre, "radius": 0.010, "height": 0.07, "base": 0.13},
    ]
    obj = _located(bottle)
    assert obj["center_xyz"][:2] == pytest.approx(centre, abs=0.003)
    assert obj["length_m"] == pytest.approx(0.07, abs=0.004)
    assert obj["width_m"] == pytest.approx(0.07, abs=0.004)
    assert obj["top_z"] == pytest.approx(kin.TABLE_Z + 0.20, abs=0.003)
    # the visible points by themselves stop at the neck: 1 cm towards the camera, 2 cm short
    cam, mask = _head_view(bottle)
    seen, _ = tools.world_points(cam, *np.nonzero(mask))
    _heading, _length, width, mid = tools._footprint(seen[:, :2])
    assert np.linalg.norm(mid - centre) > 0.008 and width < 0.052


def _arcs(centres, radius=0.04, per_slice=40):
    """Surface points of the near half (towards -y) of one circle per 1 cm of height."""

    turn = np.linspace(math.radians(190), math.radians(350), per_slice)
    return np.vstack(
        [
            np.c_[
                cx + radius * np.cos(turn),
                cy + radius * np.sin(turn),
                np.full(per_slice, kin.TABLE_Z + 0.009 + 0.01 * i),
            ]
            for i, (cx, cy) in enumerate(centres)
        ]
    )


def test_circles_that_do_not_share_an_axis_complete_nothing():
    upright = _arcs([(0.3, 0.0)] * 6)
    assert len(tools._far_side(upright)) == 6 * 90
    # one slice off the axis is a misfit: it is dropped, the others still count
    one_off = _arcs([(0.3, 0.0)] * 5 + [(0.33, 0.0)])
    assert len(tools._far_side(one_off)) == 5 * 90
    # A body that is round, but not about one upright axis: a piggy bank standing on its feet
    # gave four circles of which two were alike, and they put 2 cm on its width.
    lying = _arcs([(0.30, 0.0), (0.30, 0.0), (0.315, 0.0), (0.33, 0.0), (0.345, 0.0), (0.36, 0.0)])
    assert len(tools._far_side(lying)) == 0
    geo = tools.object_geometry(lying)
    assert geo["width_m"] == pytest.approx(0.033, abs=0.004)  # the near arcs, nothing behind them


def test_locate_adds_nothing_behind_a_flat_sided_object():
    # a plate standing across the view, and a box with a corner towards the camera: neither
    # slice is an arc, and the top covers the footprint
    plate = _located([{"center": (0.0, 0.05), "size": (0.20, 0.02, 0.15)}])
    assert plate["width_m"] == pytest.approx(0.02, abs=0.004)
    assert plate["center_xyz"][:2] == pytest.approx((0.0, 0.05), abs=0.003)
    box = _located([{"center": (0.2, 0.0), "size": (0.06, 0.06, 0.12), "yaw": 45.0}])
    assert box["length_m"] == pytest.approx(0.06, abs=0.004)
    assert box["width_m"] == pytest.approx(0.06, abs=0.004)
    assert box["center_xyz"][:2] == pytest.approx((0.2, 0.0), abs=0.003)


def test_locate_lays_a_t_along_its_bar_not_along_its_smallest_rectangle():
    # A T is clearly longer one way: its principal axes are its bar and its stem.  The
    # smallest rectangle around a trimmed T sits askew (12 degrees on RoboDojo's push-T block).
    yaw, at = 30.0, np.array([0.1, -0.15])
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    turn = np.array([[c, -s], [s, c]])
    bar = {"center": tuple(at + turn @ (0.0, 0.018)), "size": (0.08, 0.02, 0.015), "yaw": yaw}
    stem = {"center": tuple(at + turn @ (0.0, -0.010)), "size": (0.02, 0.036, 0.015), "yaw": yaw}
    obj = _located([bar, stem])
    assert obj["long_axis_deg"] == pytest.approx(yaw, abs=3)
    assert obj["center_xyz"][:2] == pytest.approx(tuple(at), abs=0.003)
    assert obj["length_m"] == pytest.approx(0.08, abs=0.004)
    assert obj["width_m"] == pytest.approx(0.056, abs=0.004)


def test_locate_leaves_out_what_the_object_stands_on():
    # the mask of a cube on a low pad takes a strip of the pad behind it along
    centre = (-0.115, -0.2)
    pad = {"center": centre, "size": (0.08, 0.08, 0.0055)}
    cube = {"center": centre, "size": (0.035, 0.035, 0.035), "base": 0.0055}
    cam, owner = _render([pad, cube])
    mask = owner == 1
    rows, cols = np.nonzero(owner == 0)
    on_pad, _ = tools.world_points(cam, rows, cols)
    strip = (np.abs(on_pad[:, 0] - centre[0]) < 0.02) & (on_pad[:, 1] > centre[1] + 0.0175)
    strip &= on_pad[:, 1] < centre[1] + 0.032
    assert strip.sum() > 20
    mask[rows[strip], cols[strip]] = True
    obj = _located([pad, cube], mask)
    assert obj["center_xyz"][:2] == pytest.approx(centre, abs=0.003)
    assert obj["length_m"] == pytest.approx(0.035, abs=0.004)
    assert obj["top_z"] == pytest.approx(kin.TABLE_Z + 0.0405, abs=0.002)
    # a flat thing has no foot to leave out: all of it is measured
    flat = _located([{"center": (0.1, -0.1), "size": (0.10, 0.04, 0.01), "yaw": 20.0}])
    assert flat["length_m"] == pytest.approx(0.10, abs=0.004)
    assert flat["width_m"] == pytest.approx(0.04, abs=0.004)


def test_a_footprint_of_a_few_points_in_a_row_is_still_a_rectangle():
    row = np.array([[0.1 + 0.004 * i, -0.1, kin.TABLE_Z + 0.02] for i in range(12)])
    geo = tools.object_geometry(row)
    assert geo["long_axis_deg"] == pytest.approx(0, abs=1)
    assert geo["length_m"] == pytest.approx(0.046, abs=0.004) and geo["width_m"] <= 0.004


def test_points_on_the_arms_are_told_apart_from_the_scene():
    q = [0.0] * 6
    state = {f"{a}_arm_joint_state": q for a in kin.ARMS}
    fr = kin.frames("right", q)
    on_arm = np.array([fr[2][:3, 3], fr[-1][:3, 3], kin.tcp(kin.fk("right", q))])
    table = np.array([[0.1, -0.1, kin.TABLE_Z + 0.03], [0.0, 0.1, kin.TABLE_Z + 0.05]])
    hit = tools._on_robot(state, np.vstack([on_arm, table]))
    assert hit.tolist() == [True, True, True, False, False]


def test_points_on_the_open_fingers_and_their_rail_are_the_robots_too():
    # The fingers stand 4.5 to 7 cm off the gripper's axis when open, further than the arm's
    # links are wide: a mug under the homed gripper came back 4 cm off with them in its mask.
    q = [0.0] * 6
    hand = kin.frames("right", q)[-1]

    def at(forward, side, up=0.0):
        return hand[:3, 3] + hand[:3, 0] * forward + hand[:3, 1] * side + hand[:3, 2] * up

    points = np.array(
        [
            at(0.12, 0.058),  # the middle of one open finger
            at(0.15, -0.050, 0.02),  # the other, near its tip
            at(0.075, 0.085),  # the end of the rail
            at(kin.TIP_OFFSET + 0.08, 0.0),  # what the gripper points at, 8 cm ahead
            at(0.12, 0.0, -0.09),  # the table under the open gripper
        ]
    )
    state = {f"{a}_arm_joint_state": q for a in kin.ARMS}
    opened = tools._on_robot(state | {f"{a}_ee_joint_state": [1.0] for a in kin.ARMS}, points)
    assert opened.tolist() == [True, True, True, False, False]
    closed = tools._on_robot(state | {f"{a}_ee_joint_state": [0.0] for a in kin.ARMS}, points)
    assert closed.tolist() == [False, False, True, False, False]  # the fingers are at the axis


def test_register_adds_the_four_tools_and_only_locate_is_read_only(sim):
    env, prims, _start = sim
    env.joints["left"] = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0]  # where boot left it
    added: dict[str, Any] = {}

    class _Toolkit:
        _primitives = prims

        def add_tool(self, name, spec, handler):
            added[name] = (spec, handler)

    tools.register(_Toolkit())
    assert set(added) == {"move_pose", "grip", "locate", "home"}
    for name, (spec, handler) in added.items():
        assert spec is tools.SPECS[name] and spec["name"] == name
        # RPent's own test of read-only handlers: unwrap partials, read ``_readonly``
        assert getattr(handler.func, "_readonly", False) is (name == "locate")
    env.joints["left"] = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    assert added["home"][1](arm="left")["reached"]
    assert env.joints["left"] == pytest.approx([0.0, 0.3, 0.0, 0.0, 0.0, 0.0], abs=0.01)
