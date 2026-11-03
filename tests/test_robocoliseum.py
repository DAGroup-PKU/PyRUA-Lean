"""The RoboColiseum robot (GenieSim's G2) over a scripted benchmark: bridge, host, primitives.

The scripted benchmark answers the bridge as GenieSim's client does, one request per action:
the arms take each ``JOINT_ABS`` target at once, a gripper moves towards its command and
stops apart where something is between its fingers, and the head camera sees a table with
one block on it from straight above, with depth and calibration.  The end poses are the
arm's forward kinematics on the benchmark's own solver when ``ik_solver`` is installed
(``pip install -e ".[robocoliseum]"``); the motion tests need it and skip otherwise.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import pytest

msgpack = pytest.importorskip("msgpack")
pytest.importorskip("PIL")

from pyrualean import ToolError, api_reference, get_adapter, knowledge_text  # noqa: E402
from pyrualean.hosts import geniesim_bridge as bridge_mod  # noqa: E402
from pyrualean.hosts import geniesim_robocoliseum as host  # noqa: E402
from pyrualean.hosts import robocoliseum_reach as reach  # noqa: E402
from pyrualean.hosts import robocoliseum_tools as tools  # noqa: E402
from pyrualean.prompt import _public_members  # noqa: E402
from pyrualean.robocoliseum import (  # noqa: E402
    Gripper,
    Move,
    ObjectPose,
    RobocoliseumRobot,
    WorldPoint,
)

IK = pytest.mark.skipif(
    not reach.available(), reason='needs ik_solver: pip install -e ".[robocoliseum]"'
)

#: The arms at the start of a recorded episode, and their poses in arm_base_link then.
START_JOINTS = [0.739, -0.717, -1.5244, -1.5376, 0.2781, -0.9259, -0.8393,
                -0.739, -0.717, 1.5244, -1.5376, -0.2781, -0.9259, 0.8393]  # fmt: skip
START_POSES = {
    "left": ([0.43715, -0.09347, 0.40065], [0.17704, 0.93234, 0.08029, 0.30487]),
    "right": ([0.43715, -0.09347, -0.40065], [0.30487, 0.08029, 0.93234, 0.17704]),
}
#: base_link <- arm_base_link with the waist where it was.
T_BASE_ARM = np.array(
    [[0.8855, -0.46464, 0.0, 0.1899], [0.0, 0.0, 1.0, 0.0],
     [-0.46464, -0.8855, 0.0, 1.14342], [0.0, 0.0, 0.0, 1.0]]
)  # fmt: skip
T_BASE_ARM[:3, :3] = np.linalg.svd(T_BASE_ARM[:3, :3])[0] @ np.linalg.svd(T_BASE_ARM[:3, :3])[2]

#: The head camera: straight down from 1.6 m, image +x along base -y, image +y along base -x.
H, W, FX = 120, 160, 150.0
CAM = np.eye(4)
CAM[:3, :3] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
CAM[:3, 3] = [0.75, -0.10, 1.60]
TABLE_Z, BLOCK_TOP = 0.78, 0.83
BLOCK = (slice(50, 70), slice(70, 90))  # rows, cols


def pixel_point(row: int, col: int, depth: float) -> np.ndarray:
    optical = np.array([(col - W / 2) / FX * depth, (row - H / 2) / FX * depth, depth, 1.0])
    return (CAM @ optical)[:3]


def _jpeg(image: np.ndarray) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(image).save(out, format="JPEG", quality=95)
    return out.getvalue()


def _quat_xyzw(rot: np.ndarray) -> list[float]:
    return [float(v) for v in tools.matrix_quat_xyzw(rot)]


class FakeBenchmark:
    """GenieSim's client, scripted: it asks for one action at a time from a thread."""

    #: Where a closing gripper stops when something is between its fingers.
    HELD = -0.30

    def __init__(self, bridge: bridge_mod.Bridge, held: tuple[str, ...] = ()) -> None:
        self.bridge = bridge
        self.joints = list(START_JOINTS)
        self.grip = [-0.785, -0.785]
        self.held = set(held)
        self.kinds: list[str] = []
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        colour = np.full((H, W, 3), 150, dtype=np.uint8)
        colour[BLOCK] = (200, 30, 30)
        self.head_jpeg, self.hand_jpeg = _jpeg(colour), _jpeg(np.full((H, W, 3), 90, np.uint8))
        depth = np.full((H, W), CAM[2, 3] - TABLE_Z)
        depth[BLOCK] = CAM[2, 3] - BLOCK_TOP
        self.depth = np.round(depth * 1000).astype(np.uint16).tobytes()

    def arm_pose(self, arm: str) -> np.ndarray:
        """arm_base_link <- the arm's end link, at the current joints."""

        q = self.joints[:7] if arm == "left" else self.joints[7:14]
        if reach.available():
            solver = reach._solver(arm, reach.STEP_CONFIG)
            return np.asarray(solver.compute_fk(np.asarray(q, dtype=float))).reshape(4, 4)
        assert q == (START_JOINTS[:7] if arm == "left" else START_JOINTS[7:14])
        position, quat = START_POSES[arm]
        out = np.eye(4)
        out[:3, :3] = tools.quat_xyzw_matrix(quat)
        out[:3, 3] = position
        return out

    def params(self) -> dict[str, Any]:
        poses: dict[str, Any] = {"base_link": {}, "arm_base_link": {}}
        for arm in ("left", "right"):
            in_arm = self.arm_pose(arm)
            for frame, pose in (("arm_base_link", in_arm), ("base_link", T_BASE_ARM @ in_arm)):
                poses[frame][f"{arm}_arm"] = {
                    "position": [float(v) for v in pose[:3, 3]],
                    "orientation": _quat_xyzw(pose[:3, :3]),
                }
        calibrated = {
            "available": True, "fx": FX, "fy": FX, "cx": W / 2, "cy": H / 2,
            "optical_to_frame": CAM.tolist(), "depth_scale_m": 0.001,
        }  # fmt: skip

        def image(data: bytes, calibration: dict[str, Any]) -> dict[str, Any]:
            return {"encoding": "JPEG", "image_data": data, "height": H, "width": W,
                    "calibration": calibration}  # fmt: skip

        return {
            "images": {
                "head": image(self.head_jpeg, calibrated),
                "head_depth": {
                    "encoding": "RAW_UINT16",
                    "image_data": self.depth,
                    "height": H,
                    "width": W,
                },  # fmt: skip
                "hand_left": image(self.hand_jpeg, {"available": False}),
                "hand_right": image(self.hand_jpeg, {"available": False}),
            },
            "states": {
                "arm_joint_states": list(self.joints),
                "gripper_states": list(self.grip),
                "waist_joint_states": [0.0] * 5,
                "end_pose": poses,
            },
            "prompt": "pick up the red block",
            "robot_type": "G2_omnipicker",
            "task_name": "pick_block_color",
        }

    def _run(self) -> None:
        while not self._stop:
            self.bridge.update_obs(self.params())
            action = self.bridge.get_action(timeout_s=30)
            if not self._stop and not action.get("hold"):
                self.apply(action)

    def apply(self, action: dict[str, Any]) -> None:
        self.kinds.append(str(action.get("kind")))
        if action.get("kind") == "JOINT_ABS":
            self.joints = [float(v) for v in [*action["left_arm"], *action["right_arm"]]]
        for k, arm in enumerate(("left", "right")):
            command = float(action[f"{arm}_gripper"])
            now = self.grip[k] + float(np.clip(command - self.grip[k], -0.1, 0.1))
            if arm in self.held and command > self.grip[k]:
                now = min(now, self.HELD)
            self.grip[k] = now

    def stop(self) -> None:
        self._stop = True
        self.bridge.finish()
        self.thread.join(5)


@pytest.fixture
def g2(tmp_path, monkeypatch):
    """``make(held=...)`` -> (robot, toolkit, scripted benchmark), with no SAM3 service."""

    monkeypatch.setenv("PYRUA_SAM3_ENDPOINT", "http://127.0.0.1:9")
    made: list[FakeBenchmark] = []

    def make(held: tuple[str, ...] = ()) -> tuple[RobocoliseumRobot, host.Toolkit, FakeBenchmark]:
        bridge = bridge_mod.Bridge()
        sim = FakeBenchmark(bridge, held=held)
        made.append(sim)
        sim.thread.start()
        toolkit = host.Toolkit(bridge, tmp_path / f"episode_{len(made)}")
        return RobocoliseumRobot(host.make_backend(toolkit)), toolkit, sim

    yield make
    for sim in made:
        sim.stop()


# -- the wire ----------------------------------------------------------------------------


def test_a_request_parses_into_cameras_with_depth_and_calibration():
    params = FakeBenchmark(bridge_mod.Bridge()).params()
    obs = bridge_mod.parse_payload(params, step=3)
    assert (obs.step, obs.instruction, obs.task_name) == (3, "pick up the red block",
                                                          "pick_block_color")  # fmt: skip
    head = obs.camera("head")
    assert (head.width, head.height, head.fx, head.cx) == (W, H, FX, W / 2)
    assert head.depth[0, 0] == pytest.approx(CAM[2, 3] - TABLE_Z, abs=1e-3)
    assert head.depth[60, 80] == pytest.approx(CAM[2, 3] - BLOCK_TOP, abs=1e-3)
    np.testing.assert_allclose(head.optical_to_frame, CAM)
    left = obs.camera("hand_left")
    assert left.fx is None and left.depth is None and left.optical_to_frame is None
    assert len(obs.arm_joints) == 14 and obs.gripper == [-0.785, -0.785]
    assert set(bridge_mod.colour_frames(params)) == {"head", "hand_left", "hand_right"}


def test_an_answer_is_plain_msgpack_of_one_joint_action():
    obs = bridge_mod.parse_payload(FakeBenchmark(bridge_mod.Bridge()).params(), step=0)
    hold = bridge_mod.hold_action(obs)
    assert hold["left_arm"] == START_JOINTS[:7] and hold["right_gripper"] == -0.785
    # The benchmark unpacks with plain msgpack: everything must be a native list or number.
    body = msgpack.unpackb(msgpack.packb(bridge_mod.envelope(hold)), raw=False)["result"]
    assert body["left_arm"] == {"kind": "JOINT_ABS", "values": [START_JOINTS[:7]]}
    assert body["right_effector"] == [[-0.785]] and body["need_depth"] is True
    assert "episode_done" not in body
    done = bridge_mod.envelope(dict(hold, episode_done=True))["result"]
    assert done["episode_done"] is True


def test_the_bridge_follows_the_benchmarks_episodes():
    bridge = bridge_mod.Bridge()
    params = FakeBenchmark(bridge).params()
    assert bridge.update_obs(params) is not None and bridge.episode == 1
    bridge.finish()  # the program is done: holds, never episode_done
    held = bridge.get_action(timeout_s=1)
    assert held.get("hold") and not held.get("episode_done")
    assert bridge.update_obs(dict(params, episode_done=True)) is None  # the closing request
    assert bridge.closed == 1 and bridge.over == {"episode_done": True}
    assert bridge.update_obs(params) is not None and bridge.episode == 2
    assert bridge.requests == 1  # the step counter restarts with the episode
    bridge.drain()  # nobody plays on: the episode in play is ended by us
    assert bridge.get_action(timeout_s=1).get("episode_done") is True


def test_a_request_that_is_not_infer_gets_no_answer():
    bridge = bridge_mod.Bridge()
    ping = msgpack.packb({"method": "ping", "params": {}})
    assert asyncio.run(bridge_mod.reply(bridge, ping)) is None
    assert bridge.requests == 0


def test_an_ended_run_is_gone_for_good():
    bridge = bridge_mod.Bridge()
    assert bridge.connected()  # before its first connection: still to come
    bridge.update_obs(FakeBenchmark(bridge).params())
    bridge.end_run("session closed")
    assert not bridge.connected(grace_s=1e9)
    assert bridge.over == {"run_ended": "session closed"}
    with pytest.raises(bridge_mod.EpisodeOver):
        bridge.step_env(bridge_mod.hold_action(bridge.get_obs()), timeout_s=1)


# -- the robot class ---------------------------------------------------------------------


def test_the_api_reference_lists_every_primitive_and_result_type():
    text = api_reference(RobocoliseumRobot)
    for name in _public_members(RobocoliseumRobot):
        assert f"robo.{name}" in text, name
    for name in ("move_to", "move_both", "move_base", "move_waist", "set_gripper", "home",
                 "wait_for_check", "segment", "back_project", "locate"):  # fmt: skip
        assert f"robo.{name}(" in text, name
    for cls in RobocoliseumRobot.RESULT_TYPES:
        assert f"class {cls.__name__}(" in text
    assert "robo.done" not in text and "robo.world_map" not in text
    assert RobocoliseumRobot.ABSENT_MEMBERS == frozenset({"done", "world_map"})
    assert RobocoliseumRobot.VLA_PRIMITIVES == ()


def test_the_adapter_names_the_host_and_the_knowledge():
    adapter = get_adapter("robocoliseum")
    assert adapter.robot_cls is RobocoliseumRobot and adapter.knowledge == "robocoliseum"
    assert adapter.host_module() is host
    text = knowledge_text("robocoliseum")
    assert "no task recipes" in text.splitlines()[0]
    assert "wait_for_check" in text


# -- the host ----------------------------------------------------------------------------


def test_cell_and_boot_options_have_no_launcher():
    parser = ArgumentParser()
    host.add_cell_args(parser)
    host.add_boot_args(parser)
    args = parser.parse_args(["--task", "g2op_if_pick_billiards_color", "--port", "9001"])
    cell = host.cell_from_args(args)
    assert (cell.task, cell.suite, cell.seed) == ("g2op_if_pick_billiards_color", "g2_op", 1)
    assert set(host.boot_options(args)) == {"host", "port", "output_root", "startup_timeout_s"}
    assert host.boot_options(args)["port"] == 9001


def test_scores_are_read_from_the_runs_own_result_file(tmp_path):
    bridge = bridge_mod.Bridge()
    out = tmp_path / "benchmark" / "pick_block_color" / "sub"
    out.mkdir(parents=True)
    old = out / "evaluate_ret_00.json"
    old.write_text(json.dumps({"details": [{"result": {"scores": {"E2E": 0}}}]}))
    os.utime(old, (time.time() - 3600, time.time() - 3600))  # an earlier run's file
    benchmark = host.Benchmark(bridge, output_root=str(tmp_path))
    assert benchmark.details() == [] and benchmark.result(0) is None
    steps = [{"score": 1.0}, {"score": 1.0}]
    entry = {"task_instruction": "pick up the red block", "policy_dispatch_count": 90,
             "result": {"scores": {"E2E": 1, "STEPS": steps}}}  # fmt: skip
    (out / "evaluate_ret_01.json").write_text(json.dumps({"details": [entry]}))
    scores = benchmark.result(0)
    assert scores["E2E"] == 1 and scores["instruction"] == "pick up the red block"
    assert len(scores["STEPS"]) == 2 and scores["policy_dispatch_count"] == 90
    assert host.Benchmark(bridge).result(0) is None  # the online platform keeps its scores


def test_the_card_states_the_pose_and_the_stage_budgets(g2):
    robo, toolkit, _ = g2()
    cell = host.Cell(task="g2op_if_pick_block_color", stage_budgets=(200, 200))
    card = host.dump_card(toolkit, cell, toolkit.out_dir)
    written = json.loads((toolkit.out_dir / "card.json").read_text())
    assert written["task_language"] == "pick up the red block" and card.max_env_steps == 400
    assert card.facts["stage budgets (native actions)"].startswith("200, 200")
    assert set(card.images) == {"head", "hand_left", "hand_right"}
    assert "right tool centre [x, y, z]" in card.facts


# -- perception --------------------------------------------------------------------------


def test_back_project_measures_through_the_published_calibration(g2):
    robo, toolkit, _ = g2()
    point = robo.back_project(10, 20)
    assert isinstance(point, WorldPoint)
    np.testing.assert_allclose(point.xyz, pixel_point(10, 20, CAM[2, 3] - TABLE_Z), atol=2e-3)
    on_block = robo.back_project(60, 80)
    assert on_block.xyz[2] == pytest.approx(BLOCK_TOP, abs=2e-3)
    assert toolkit.env_steps() == 1  # the first frame carried depth: nothing was spent
    with pytest.raises(ToolError, match="calibration|depth"):
        robo.back_project(10, 20, camera="hand_left")


def test_locate_at_a_pixel_measures_the_block_on_the_table(g2):
    robo, _, _ = g2()
    block = robo.locate(row=60, col=80)
    assert isinstance(block, ObjectPose) and block.found and block.note is None
    centre = pixel_point(59.5, 79.5, CAM[2, 3] - BLOCK_TOP)
    assert block.centre_xyz[0] == pytest.approx(centre[0], abs=0.01)
    assert block.centre_xyz[1] == pytest.approx(centre[1], abs=0.01)
    assert block.top_z == pytest.approx(BLOCK_TOP, abs=3e-3)
    assert block.support_z == pytest.approx(TABLE_Z, abs=3e-3)
    assert block.height_m == pytest.approx(BLOCK_TOP - TABLE_Z, abs=5e-3)


def test_locate_by_name_says_when_there_is_no_sam3(g2):
    robo, _, _ = g2()
    found = robo.locate("red block")
    assert not found.found and "SAM3" in (found.note or "")


# -- gripper and checks ------------------------------------------------------------------


def test_a_close_that_holds_waits_for_the_next_check(g2):
    robo, toolkit, sim = g2(held=("right",))
    grip = robo.set_gripper("right", robo.CLOSE)
    assert isinstance(grip, Gripper) and grip.holding and grip.commanded == "closed"
    assert sim.grip[1] == pytest.approx(FakeBenchmark.HELD)
    # held still until the check at a multiple of 30 actions had run
    assert (toolkit.env_steps() - 1) % tools.CHECK_PERIOD == 0 and toolkit.env_steps() > 1


def test_a_close_on_nothing_shuts_and_does_not_wait(g2):
    robo, toolkit, sim = g2()
    grip = robo.set_gripper("left", robo.CLOSE)
    assert not grip.holding and grip.opening == pytest.approx(0.0, abs=0.02)
    assert grip.steps_used < tools.CHECK_PERIOD and sim.grip[0] == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        robo.set_gripper("left", robo.KEEP)


def test_wait_for_check_ends_just_after_a_check(g2):
    robo, toolkit, sim = g2()
    spent = robo.wait_for_check()
    assert 0 < spent <= tools.CHECK_PERIOD
    assert (toolkit.env_steps() - 1) % tools.CHECK_PERIOD == 0
    assert sim.joints == START_JOINTS  # nothing moved
    two = robo.wait_for_check(checks=2)
    assert two == 2 * tools.CHECK_PERIOD
    with pytest.raises(ValueError):
        robo.wait_for_check(checks=0)


# -- motion ------------------------------------------------------------------------------


@IK
def test_move_to_reaches_a_nearby_pose_with_joint_targets(g2):
    robo, _, sim = g2()
    start = np.asarray(robo.state().arm("right").tool_centre)
    move = robo.move_to(tuple(start + [0.0, 0.0, 0.05]), arm="right")
    assert isinstance(move, Move) and move.reached and move.steps_used > 0
    assert move.dist_to_target_m <= 0.015
    assert set(sim.kinds) == {"JOINT_ABS"}
    assert sim.joints[:7] == START_JOINTS[:7]  # the other arm held where it was


@IK
def test_move_to_out_of_reach_is_refused_before_any_action(g2):
    robo, toolkit, sim = g2()
    start = np.asarray(robo.state().arm("right").tool_centre)
    steps = toolkit.env_steps()
    move = robo.move_to(tuple(start + [0.8, 0.0, 0.4]), arm="right")
    assert not move.reached and move.steps_used == 0
    assert move.stop_reason == "out_of_reach" or move.alternatives
    assert toolkit.env_steps() == steps and sim.joints == START_JOINTS


def test_motion_arguments_are_checked_before_anything_moves(g2):
    robo, toolkit, sim = g2()
    with pytest.raises(ValueError):
        robo.move_to((0.5, 0.0, 1.0), arm="middle")
    with pytest.raises((ValueError, ToolError)):
        robo.move_to((0.5, 0.0, 1.0), path="sideways")
    assert sim.joints == START_JOINTS and toolkit.env_steps() == 1


def test_the_recorded_steps_follow_the_actions(g2, tmp_path):
    robo, toolkit, _ = g2(held=("right",))
    robo.set_gripper("right", robo.CLOSE)
    steps = sorted(p.name for p in Path(toolkit.steps_dir).glob("*.json"))
    assert steps[0] == "0000.json" and len(steps) >= 2
