# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboDojo's Pi_05 beside the colour-only tools: ``pi0_pick`` over the toolkit that needs no
RPent, its client's wire format, and the robot ``robodojo_xpl_rgb_pi05``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyrualean._backend import ToolError
from pyrualean.hosts import pi05_xpolicylab as pi05
from pyrualean.hosts import robodojo_kinematics as kin
from pyrualean.hosts import robodojo_native as native
from pyrualean.prompt import TaskCard, api_reference, knowledge_text, render_prompt
from pyrualean.robodojo import Pick
from pyrualean.robots import get_adapter


def _sibling(name: str):
    """Another test module of this directory (the tests are imported one by one)."""

    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_colour = _sibling("test_robodojo_colour")
Scene, Segmenter, CUBE, BOWL = _colour.Scene, _colour.Segmenter, _colour.CUBE, _colour.BOWL


def _z(arm: str, q: list[float]) -> float:
    return float(kin.fk(arm, q)[2, 3])


def _dip(arm: str = "right", depth: float = 0.08) -> list[float]:
    """Joint angles of ``arm`` that put its end effector ``depth`` below the start pose."""

    start = _z(arm, [0.0] * 6)
    for j1 in np.linspace(0.0, 1.5, 151):
        for j2 in np.linspace(0.0, 1.5, 151):
            q = [0.0, float(j1), float(j2), 0.0, 0.0, 0.0]
            if start - _z(arm, q) >= depth:
                return q
    raise AssertionError("no dip found")


class FakePolicy:
    """``predict(obs)`` -> one chunk, as :class:`pyrualean.hosts.pi05_xpolicylab.Pi05Client`."""

    def __init__(self, chunks: list[list[dict[str, list[float]]]]) -> None:
        self.chunks = chunks
        self.seen: list[dict[str, Any]] = []

    def predict(self, obs: dict[str, Any]) -> list[dict[str, list[float]]]:
        self.seen.append(obs)
        return self.chunks[min(len(self.seen) - 1, len(self.chunks) - 1)]


def _action(right: list[float], right_grip: float) -> dict[str, list[float]]:
    return {
        "left_arm_joint_state": [0.0] * 6,
        "left_ee_joint_state": [1.0],
        "right_arm_joint_state": list(right),
        "right_ee_joint_state": [right_grip],
    }


def _pick_chunk() -> list[dict[str, list[float]]]:
    """Down with the gripper open, close, back up, then reopen (what must never be executed)."""

    dip = _dip()
    down = [_action(dip, 1.0)] * 12
    close = [_action(dip, 0.0)] * 6
    up = [_action([0.0] * 6, 0.0)] * 12
    reopen = [_action([0.0] * 6, 1.0)] * 20
    return down + close + up + reopen


@pytest.fixture
def scene():
    return Scene([CUBE, BOWL])


def test_pi0_pick_stops_the_chunk_once_an_arm_has_lifted_what_it_closed_on(scene, tmp_path):
    policy = FakePolicy([_pick_chunk()])
    toolkit = native.NativeToolkit(scene, Segmenter(scene), tmp_path, vla_client=policy)
    out = toolkit.execute_tool("pi0_pick", {"max_chunks": 3})
    result = out["log"]["result"]
    assert result["success"] is True and result["chunks_used"] == 1
    assert result["instruction"] == "put the cube in the bowl"
    right = result["arms"]["right"]
    assert right["start_eef_z"] - right["min_eef_z"] >= pi05.DESCEND_M
    assert right["peak_lift_m"] >= pi05.LIFT_M and right["final_gripper"] < pi05.CLOSED_BELOW
    assert scene.grip["right"] < pi05.CLOSED_BELOW  # the reopening actions were cut
    assert scene.step_count < len(_pick_chunk())
    assert out["step"] == 1 and len(toolkit.state.records) == 2  # recorded like a motion
    # the policy saw the observation the client sends, with the instruction as its prompt
    assert policy.seen[0]["instruction"] == "put the cube in the bowl"


def test_pi0_pick_runs_its_chunks_and_stops_at_the_budget(scene, tmp_path):
    still = [[_action([0.0] * 6, 1.0)] * 50]
    policy = FakePolicy(still)
    toolkit = native.NativeToolkit(scene, Segmenter(scene), tmp_path, vla_client=policy)
    out = toolkit.execute_tool("pi0_pick", {"max_chunks": 2, "prompt": "pick the red cube"})
    result = out["log"]["result"]
    assert result["success"] is False and result["chunks_used"] == 2 and not result["terminated"]
    assert scene.step_count == 100 and policy.seen[1]["instruction"] == "pick the red cube"
    scene.step_limit = scene.step_count + 30
    result = toolkit.execute_tool("pi0_pick", {})["log"]["result"]
    assert result["terminated"] is True and scene.step_count == scene.step_limit


def test_pi0_pick_refuses_template_markers_and_needs_a_policy(scene, tmp_path):
    toolkit = native.NativeToolkit(scene, Segmenter(scene), tmp_path, vla_client=FakePolicy([]))
    refused = toolkit.execute_tool("pi0_pick", {"prompt": "pick <target>"})
    assert "template" in refused["log"]["result"]["error"] and scene.step_count == 0
    robo = get_adapter("robodojo_xpl_rgb_pi05").robot_cls(native.NativeBackend(toolkit))
    with pytest.raises(ToolError, match="template"):
        robo._stateful("pi0_pick", {"prompt": "pick <target>"}, lambda p: p)
    plain = native.NativeToolkit(scene, Segmenter(scene), tmp_path / "plain")
    assert "no tool" in plain.execute_tool("pi0_pick", {})["error"]


def test_the_robot_reads_pi0_pick_and_hides_it_under_no_vla(scene, tmp_path):
    adapter = get_adapter("robodojo_xpl_rgb_pi05")
    policy = FakePolicy([_pick_chunk()])
    backend = native.NativeBackend(
        native.NativeToolkit(scene, Segmenter(scene), tmp_path, vla_client=policy)
    )
    robo = adapter.robot_cls(backend, primitives="full")
    pick = robo.pi0_pick(max_chunks=2)
    assert isinstance(pick, Pick) and pick.success and pick.chunks_used == 1
    assert pick.right["peak_lift_m"] >= pi05.LIFT_M and not pick.truncated
    assert robo.state().step == 1
    # a KEEP after the policy holds what the policy closed on
    robo.move_to(robo.state().right.tcp_pos, arm="right")
    assert scene.grip["right"] < pi05.CLOSED_BELOW
    hidden = adapter.robot_cls(backend, primitives="no-vla")
    with pytest.raises(RuntimeError, match="no-VLA"):
        hidden.pi0_pick()


def test_the_prompt_of_the_robot_names_the_policy_only_with_the_full_set():
    adapter = get_adapter("robodojo_xpl_rgb_pi05")
    assert adapter.host.endswith("xpolicylab_robodojo_rgb_pi05")
    assert adapter.knowledge == "robodojo_xpl_rgb"
    full = api_reference(adapter.robot_cls, primitives="full")
    assert "robo.pi0_pick(" in full and "there is no depth image" in full
    assert "pi0_pick" not in api_reference(adapter.robot_cls, primitives="no-vla")
    assert "pi0_pick" in knowledge_text(adapter.knowledge, primitives="full")
    card = TaskCard(
        suite="arx_x5",
        task="stack_blocks",
        seed=0,
        task_language="stack the blocks",
        object_names=[],
        robot="robodojo_xpl_rgb_pi05",
    )
    full_prompt = render_prompt(card, primitives="full").system
    assert "a frozen Pi0.5 VLA and a SAM3 segmentation service" in full_prompt
    assert "class RobodojoOfficialColourPi05Robot" in full_prompt
    plain = render_prompt(card, primitives="no-vla").system
    assert "Pi0.5" not in plain and "pi0_pick" not in plain


def test_the_client_sends_what_the_policy_reads_in_openpis_wire_format():
    obs = {
        "vision": {
            camera: {
                "color": np.full((480, 640, 4), i, dtype=np.uint8),
                "depth": np.ones((480, 640)),
                "intrinsic_matrix": np.eye(3),
            }
            for i, camera in enumerate(pi05.CAMERAS)
        },
        "state": {
            "left_arm_joint_state": [0.1] * 6,
            "right_arm_joint_state": [0.2] * 6,
            "left_ee_joint_state": [1.0],
            "right_ee_joint_state": [0.0],
            "left_ee_pose": [0.0] * 7,
        },
    }
    wire = pi05.wire_obs(obs, "stack the blocks")
    assert wire["instruction"] == "stack the blocks"
    assert sorted(wire["state"]) == [
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    ]
    for i, camera in enumerate(pi05.CAMERAS):
        assert list(wire["vision"][camera]) == ["color"]
        assert wire["vision"][camera]["color"].shape == (480, 640, 3)
        assert int(wire["vision"][camera]["color"][0, 0, 0]) == i
    back = pi05.unpackb(pi05.packb(wire))
    assert np.array_equal(back["vision"]["cam_head"]["color"], wire["vision"]["cam_head"]["color"])
    assert back["state"]["right_arm_joint_state"].dtype == np.float64
    with pytest.raises(ValueError, match="cam_left_wrist"):
        pi05.wire_obs({"vision": {"cam_head": {"color": np.zeros((2, 2, 3))}}, "state": {}}, "x")


def test_starting_the_server_refuses_what_is_not_a_checkpoint_or_an_environment(tmp_path):
    xpolicylab = tmp_path / "bench" / "XPolicyLab"
    xpolicylab.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="env_cfg"):
        pi05.start(xpolicylab, str(tmp_path), tmp_path / "out")
    (tmp_path / "bench" / "env_cfg").mkdir()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        pi05.start(xpolicylab, str(tmp_path), tmp_path / "out")
    step = tmp_path / "59999"
    (step / "params").mkdir(parents=True)
    (step / "assets" / "arx_x5_sim").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="uv environment"):
        pi05.start(xpolicylab, str(step), tmp_path / "out")
