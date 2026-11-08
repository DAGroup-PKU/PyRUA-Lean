# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboDojo as its own evaluation shows it: the colour-only tools, the toolkit that needs no
RPent, and the bridge to XPolicyLab's evaluation client."""

from __future__ import annotations

import importlib.util
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyrualean._backend import EpisodeFinished
from pyrualean.hosts import robodojo_kinematics as kin
from pyrualean.hosts import robodojo_native as native
from pyrualean.hosts import robodojo_sight as sight
from pyrualean.hosts import robodojo_tools as tools
from pyrualean.hosts import xpolicylab_bridge as xb
from pyrualean.hosts.sam3_local import Segmented
from pyrualean.prompt import api_reference, knowledge_text
from pyrualean.robodojo_xpl import RobodojoOfficialRobot
from pyrualean.robodojo_xpl_rgb import ObjectSight, RobodojoOfficialColourRobot, SeenPoint
from pyrualean.robots import get_adapter


def _sibling(name: str):
    """Another test module of this directory (the tests are imported one by one)."""

    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_scenes = _sibling("test_robodojo_sight")
HOME, render = _scenes.HOME, _scenes.render

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CUBE = {"center": (0.25, 0.05), "size": (0.035, 0.035, 0.035), "yaw": 30}
BOWL = {"center": (-0.20, 0.02), "radius": 0.03, "top_radius": 0.058, "height": 0.05}
NAMES = {"the cube": 0, "the bowl": 1}


def _ee_pose(arm: str, q: list[float]) -> list[float]:
    pose = kin.fk(arm, q)
    return [*(float(v) for v in pose[:3, 3]), *(float(v) for v in kin.quat_wxyz(pose[:3, :3]))]


class Scene:
    """Two arms and some shapes on the table, as RoboDojo's evaluation client shows them: three
    colour images and the state.  The images carry no picture, only which camera they are (the
    segmenter below is given the shapes instead)."""

    def __init__(self, shapes: list[dict[str, Any]], step_limit: int = 500) -> None:
        self.shapes = shapes
        self.joints = {arm: [0.0] * 6 for arm in kin.ARMS}
        self.grip = {arm: 1.0 for arm in kin.ARMS}
        self.step_count, self.step_limit = 0, step_limit
        self.blind: set[str] = set()

    def state(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for arm in kin.ARMS:
            out[f"{arm}_arm_joint_state"] = list(self.joints[arm])
            out[f"{arm}_ee_joint_state"] = [self.grip[arm]]
            out[f"{arm}_ee_pose"] = _ee_pose(arm, self.joints[arm])
        return out

    def view(self, camera: str) -> sight.View:
        state = self.state()
        if camera == "cam_head":
            return sight.head_view()
        return sight.wrist_view(state["left_ee_pose" if "left" in camera else "right_ee_pose"])

    def get_obs(self) -> dict[str, Any]:
        vision = {}
        for index, camera in enumerate(CAMERAS):
            color = np.zeros((480, 640, 3), dtype=np.uint8)
            color[..., 0] = index  # which camera this is, for the segmenter
            color[..., 1] = self.step_count % 251
            vision[camera] = {"color": color, "shape": [480, 640, 3]}
        return {"state": self.state(), "vision": vision, "instruction": "put the cube in the bowl"}

    def get_status(self) -> dict[str, Any]:
        return {"step": self.step_count, "step_limit": self.step_limit}

    def step(self, action: dict[str, list[float]]):
        self.step_count += 1
        for arm in kin.ARMS:
            goal = action[f"{arm}_arm_joint_state"]
            self.joints[arm] = [
                now + max(-0.2, min(0.2, g - now))
                for now, g in zip(self.joints[arm], goal, strict=True)
            ]
            want = action[f"{arm}_ee_joint_state"][0]
            self.grip[arm] += max(-0.3, min(0.3, want - self.grip[arm]))
        return self.get_obs(), 0.0, False, {"status": self.get_status()}


class Segmenter:
    """SAM3's interface over a ray-cast scene: an object by its name, or by a pixel on it."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.calls: list[tuple[str, Any]] = []

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        camera = CAMERAS[int(np.asarray(image)[0, 0, 0])]
        self.calls.append((camera, text_prompt if point is None else tuple(point)))
        if camera in self.scene.blind:
            return Segmented(False)
        owner = render(self.scene.view(camera), self.scene.shapes)
        if point is not None:
            index = int(owner[int(point[0]), int(point[1])])
        else:
            index = NAMES.get(" ".join(str(text_prompt).lower().split()), -2)
        mask = owner == index
        if index == -2 or mask.sum() < 20:
            return Segmented(False)
        return Segmented(True, score=0.9, box=None, mask=mask)


@pytest.fixture
def world():
    scene = Scene([CUBE, BOWL])
    sam3 = Segmenter(scene)
    prim = native._Primitives(scene, sam3)
    return scene, sam3, prim


def _off(xyz, shape) -> float:
    return math.hypot(xyz[0] - shape["center"][0], xyz[1] - shape["center"][1])


def test_locate_measures_from_colour_images_when_there_is_no_depth(world):
    scene, sam3, prim = world
    out = tools.locate(prim, "the cube")
    assert out["found"] and out["views"] == 2 and "geometry_error" not in out
    assert _off(out["center_xyz"], CUBE) < 0.003
    assert out["height_m"] == pytest.approx(0.035, abs=0.004)
    assert "height_guess" not in out
    # the other cameras were asked for it by name and by where it should appear
    asked = {camera for camera, _ in sam3.calls}
    assert asked == set(CAMERAS)
    bowl = tools.locate(prim, "the bowl")
    assert bowl["views"] == 2
    assert _off(bowl["center_xyz"], BOWL) < 0.003
    assert bowl["length_m"] == pytest.approx(0.116, abs=0.008)


def test_colour_only_is_what_the_host_says_or_what_the_observation_lacks():
    class Plain:
        pass

    assert tools.colour_only(Plain(), {"color": 1})
    assert not tools.colour_only(Plain(), {"color": 1, "distance_to_image_plane": 1})
    told = Plain()
    told.colour_only = True
    assert tools.colour_only(told, {"color": 1, "distance_to_image_plane": 1})


def test_a_height_two_cameras_agreed_on_is_kept_for_when_one_sees_the_object(world):
    scene, _sam3, prim = world
    first = tools.locate(prim, "the bowl")
    assert first["views"] == 2
    scene.blind = {"cam_left_wrist", "cam_right_wrist"}
    scene.step_count += 1  # a new observation
    again = tools.locate(prim, "The  bowl")
    assert again["views"] == 1 and again["height_guess"] == "its height as measured before"
    assert _off(again["center_xyz"], BOWL) < 0.004
    # something never measured with two cameras is a guess, and says so
    scene.step_count += 1
    cube = tools.locate(prim, "the cube")
    assert cube["views"] == 1 and "guess" in cube["height_guess"]
    assert _off(cube["center_xyz"], CUBE) < 0.006


def test_a_pixel_is_answered_by_the_object_under_it_or_by_the_table(world):
    scene, _sam3, prim = world
    head = sight.head_view()
    owner = render(head, scene.shapes)
    rows, cols = np.nonzero(owner == 0)
    row, col = int(np.median(rows)), int(np.median(cols))
    on = tools.probe(prim, row, col)
    assert on["surface"] == "object" and on["pixel"] == [row, col]
    assert on["world_xyz"][2] > kin.TABLE_Z + 0.01
    assert _off(on["world_xyz"], CUBE) < 0.03
    # a second pixel of the same object needs no second measurement
    before = len(_sam3.calls)
    again = tools.probe(prim, row + 1, col + 1)
    assert again["surface"] == "object" and len(_sam3.calls) == before
    # the table answers with the point of its top the pixel shows
    table = tools.probe(prim, 300, 320)
    xy, _ok = head.on_plane(np.array([300]), np.array([320]), kin.TABLE_Z)
    assert table["surface"] == "table"
    assert table["world_xyz"] == pytest.approx([xy[0, 0], xy[0, 1], kin.TABLE_Z], abs=1e-4)
    assert "error" in tools.probe(prim, 5, 320)  # past the table's far edge: the sky
    assert "error" in tools.probe(prim, 900, 0)


def test_the_toolkit_records_steps_and_wraps_what_changes_the_scene(world, tmp_path):
    scene, sam3, _prim = world
    toolkit = native.NativeToolkit(scene, sam3, tmp_path, verdict=lambda: False)
    first = toolkit.execute_tool("view_env_state", {"step": 0})
    assert first["step"] == 0 and first["task_language"] == "put the cube in the bowl"
    assert first["state"]["state"]["left_ee_pose"] == pytest.approx(_ee_pose("left", [0.0] * 6))
    assert toolkit.state.load_bytes("cam_head.png", step=0)[:4] == b"\x89PNG"
    moved = toolkit.execute_tool(
        "move_pose", {"xyz": [0.25, 0.0, 0.95], "arm": "right", "pitch": 45}
    )
    assert moved["step"] == 1 and moved["log"]["result"]["reached"] is True
    assert moved["truncated"] is False and len(toolkit.state.records) == 2
    found = toolkit.execute_tool("segment", {"text_prompt": "the cube"})
    assert found["found"] and found["mask_moments"][0] > 100 and "log" not in found
    assert "error" in toolkit.execute_tool("pi0_pick", {})
    assert "error" in toolkit.execute_tool("view_env_state", {"step": 7})
    backend = native.NativeBackend(toolkit)
    assert backend.env_steps() == scene.step_count > 0
    backend.cancel()
    stopped = toolkit.execute_tool("home", {"arm": "both"})
    assert stopped["log"]["result"]["interrupted"] is True


def test_the_colour_robot_reads_the_same_api_without_depth(world, tmp_path):
    scene, sam3, _prim = world
    backend = native.NativeBackend(native.NativeToolkit(scene, sam3, tmp_path))
    robo = RobodojoOfficialColourRobot(backend, primitives="no-vla")
    assert robo.task == "put the cube in the bowl"
    seen = robo.locate("the cube")
    assert isinstance(seen, ObjectSight) and seen.views == 2 and seen.note is None
    assert _off(seen.center_xyz, CUBE) < 0.003
    point = robo.back_project(300, 320)
    assert isinstance(point, SeenPoint) and point.surface == "table"
    meta = robo.camera_meta("cam_left_wrist")
    assert sorted(meta) == ["extrinsic_matrix", "intrinsic_matrix", "shape"]
    assert np.allclose(
        np.array(meta["extrinsic_matrix"])[:3, 3], sight.wrist_view(HOME["left"]).pos, atol=2e-3
    )
    move = robo.move_to([0.25, 0.0, 0.95], arm="right", pitch=45)
    assert move.reached and robo.state().step == 1
    assert robo.image("cam_head").shape == (480, 640, 3)
    scene.step_limit = scene.step_count  # the budget is spent
    robo.set_gripper("right", robo.CLOSE)
    with pytest.raises(EpisodeFinished):
        robo.move_to([0.25, 0.0, 0.95], arm="right")


def test_the_reference_and_the_knowledge_of_the_colour_robot_promise_no_depth():
    adapter = get_adapter("robodojo_xpl_rgb")
    reference = api_reference(adapter.robot_cls, primitives="no-vla")
    assert "there is no depth image" in reference and "views" in reference
    assert "through the depth image" not in reference
    knowledge = knowledge_text(adapter.knowledge, primitives="no-vla")
    assert "no depth image" in knowledge and "pi0_pick" not in knowledge
    assert "`views >= 2`" in knowledge
    assert adapter.host.endswith("xpolicylab_robodojo_rgb")


# -- the bridge to XPolicyLab's evaluation client ---------------------------------------------


def _deploy():
    path = Path(__file__).resolve().parents[1] / "integrations/xpolicylab/PyRUA_Lean/deploy.py"
    spec = importlib.util.spec_from_file_location("pyrua_lean_deploy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TaskEnv:
    """What deploy.py uses of RoboDojo's evaluation environment."""

    def __init__(self, scene: Scene, step_lim: int, solved_at: int | None = None) -> None:
        self.scene, self.step_lim, self.solved_at = scene, step_lim, solved_at
        self.take_action_cnt, self.success, self.end_flag = [0], [True], [False]
        self.task_name, self.eval_seed, self.env_seeds = "stack_blocks", 0, [3]
        self.final_checks = 0

    def get_obs(self):
        return self.scene.get_obs()

    def take_action(self, action):
        self.take_action_cnt[0] += 1
        self.scene.step(action)
        self.is_episode_end()

    def get_running_env_idx_list(self):
        return [] if self.end_flag[0] else [0]

    def is_episode_end(self):
        if self.end_flag[0]:
            return True
        final = self.take_action_cnt[0] >= self.step_lim or not self.success[0]
        self.final_checks += final
        if self.solved_at is not None and self.take_action_cnt[0] >= self.solved_at:
            self.end_flag[0], self.success[0] = True, True
        elif final:
            self.end_flag[0], self.success[0] = True, False
        return self.end_flag[0]


class Client:
    """XPolicyLab's model client, calling the policy model in the same process."""

    def __init__(self, model: xb.PolicyModel) -> None:
        self.model = model
        self.calls: list[str] = []

    def call(self, func_name=None, obs=None):
        self.calls.append(func_name)
        # the protocol's own messages; any other name is a method of the model
        method = getattr(self.model, {"trial_end": "on_trial_end"}.get(func_name, func_name))
        return method() if obs is None else method(obs)


def _episode(task_env: TaskEnv, poll_s: float = 0.05):
    bridge = xb.Bridge(poll_s=poll_s)
    client = Client(xb.PolicyModel(bridge))
    thread = threading.Thread(
        target=_deploy().eval_one_episode, args=(task_env, client), daemon=True
    )
    thread.start()
    bridge.first_obs(5.0, alive=thread.is_alive)
    return bridge, client, thread


def test_the_tools_step_roboDojos_client_through_the_bridge(tmp_path):
    scene = Scene([CUBE, BOWL])
    task_env = TaskEnv(scene, step_lim=200)
    bridge, client, thread = _episode(task_env)
    env = xb.BridgeEnv(bridge)
    assert env.get_status() == {
        "pictures_behind": 1,
        "step": 0,
        "step_limit": 200,
        "task_name": "stack_blocks",
        "eval_seed": 0,
        "layout": 3,
    }
    toolkit = native.NativeToolkit(
        env, Segmenter(scene), tmp_path, verdict=lambda: bool((bridge.over or {}).get("success"))
    )
    backend = native.NativeBackend(toolkit)
    moved = backend.execute_tool(
        "move_pose", {"xyz": [0.25, 0.0, 0.95], "arm": "right", "pitch": 45}
    )
    assert moved["log"]["result"]["reached"] is True
    assert (
        task_env.take_action_cnt[0] == backend.env_steps() == moved["log"]["result"]["steps_used"]
    )
    assert backend.solved() is False and thread.is_alive()
    backend.conclude()  # the program has finished: RoboDojo judges the scene as it stands
    thread.join(5.0)
    assert not thread.is_alive() and task_env.final_checks == 1
    assert bridge.over["success"] is False and backend.solved() is False
    assert client.calls[0] == "reset" and client.calls[-1] == "trial_end"
    assert "wait" not in client.calls
    # nothing more can be done once it is over
    late = backend.execute_tool("grip", {"arm": "right", "gripper": 1})
    assert late["truncated"] is True and late["log"]["result"]["steps_used"] <= 1


def test_the_client_ends_the_episode_itself_when_the_task_is_solved(tmp_path):
    scene = Scene([CUBE])
    task_env = TaskEnv(scene, step_lim=200, solved_at=3)
    bridge, _client, thread = _episode(task_env)
    env = xb.BridgeEnv(bridge)
    for expected_done in (False, False, True):
        _obs, _reward, done, info = env.step(
            {
                f"{arm}_{part}": ([0.0] * 6 if part == "arm_joint_state" else [1.0])
                for arm in kin.ARMS
                for part in ("arm_joint_state", "ee_joint_state")
            }
        )
        assert done is expected_done
    thread.join(5.0)
    assert bridge.over["success"] is True and bridge.over["step"] == 3
    assert info["status"]["over"] is True and info["status"]["step_limit"] == info["status"]["step"]
    with pytest.raises(xb.EpisodeOver):
        bridge.step({})


def test_get_action_does_not_keep_the_client_waiting_while_the_program_thinks():
    bridge = xb.Bridge(poll_s=0.02)
    model = xb.PolicyModel(bridge)
    model.update_obs({"state": {}, "vision": {}})
    assert model.get_action() == xb.WAIT
    bridge.finish()
    assert model.get_action() == []
    deploy = _deploy()
    assert deploy._waiting(xb.WAIT) and not deploy._waiting([]) and not deploy._waiting([{"a": 1}])


# -- the adapter's model.py: XPolicyLab starts the policy server, the episode runs inside it --


def _adapter_model(monkeypatch):
    """integrations/xpolicylab/PyRUA_Lean/model.py, over XPolicyLab's template as it is."""

    import sys
    import types

    package = types.ModuleType("XPolicyLab")
    template = types.ModuleType("XPolicyLab.model_template")

    class ModelTemplate:
        def __init__(self) -> None:
            self.model = None

        def update_obs(self, obs):
            raise NotImplementedError

        def get_action(self):
            raise NotImplementedError

        def reset(self):
            pass

    template.ModelTemplate = ModelTemplate
    package.model_template = template
    utils = types.ModuleType("XPolicyLab.utils")
    process_data = types.ModuleType("XPolicyLab.utils.process_data")
    robots = {
        "arx_x5": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
        "franka": {"arm_dim": [7], "ee_dim": [1]},
    }
    process_data.get_robot_action_dim_info = robots.__getitem__
    utils.process_data = process_data
    package.utils = utils
    monkeypatch.setitem(sys.modules, "XPolicyLab", package)
    monkeypatch.setitem(sys.modules, "XPolicyLab.model_template", template)
    monkeypatch.setitem(sys.modules, "XPolicyLab.utils", utils)
    monkeypatch.setitem(sys.modules, "XPolicyLab.utils.process_data", process_data)
    path = Path(__file__).resolve().parents[1] / "integrations/xpolicylab/PyRUA_Lean/model.py"
    spec = importlib.util.spec_from_file_location("pyrua_lean_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_adapters_model_runs_an_episode_for_each_reset(monkeypatch, tmp_path):
    from pyrualean import play

    played: list[dict[str, Any]] = []

    def play_episode(cell, out_dir, **kwargs):
        # what pyrualean.play does with the host, without a model: boot, move, conclude
        host = get_adapter(kwargs["robot"]).host_module()
        toolkit, daemons = host.boot(cell, out_dir, env_only=True, **kwargs["boot_options"])
        backend = host.make_backend(toolkit)
        moved = backend.execute_tool(
            "move_pose", {"xyz": [0.25, 0.0, 0.95], "arm": "right", "pitch": 45}
        )
        backend.conclude()
        played.append(
            {
                "cell": cell,
                "robot": kwargs["robot"],
                "primitives": kwargs["primitives"],
                "steps": moved["log"]["result"]["steps_used"],
                "daemons": daemons,
                "success": backend.solved(),
            }
        )
        return {"status": "completed", "environment_success": backend.solved(), "task": cell.task}

    monkeypatch.setattr(play, "play_episode", play_episode)
    module = _adapter_model(monkeypatch)
    model = module.Model(
        {
            "poll_s": 0.05,
            "model": "some-model",
            "env_cfg_type": "arx_x5",
            "result_dir": tmp_path,
            "robot": "robodojo_xpl_rgb",  # the scene below sends no depth
        }
    )
    client = Client(model)
    deploy = _deploy()
    for solved_at in (None, 3):  # two episodes in a row, as RoboDojo's client runs them
        task_env = TaskEnv(Scene([CUBE]), step_lim=200, solved_at=solved_at)
        deploy.eval_one_episode(task_env, client)  # returns: nothing is left waiting
        model._episode.join(5.0)
        assert not model._episode.is_alive()
        last = played[-1]
        assert last["cell"].task == "stack_blocks" and last["cell"].seed == 3
        assert last["robot"] == "robodojo_xpl_rgb" and last["primitives"] == "no-vla"
        assert last["daemons"] == []  # XPolicyLab started the pair, the host started nothing
        assert last["success"] is (solved_at is not None)
        assert task_env.take_action_cnt[0] == (last["steps"] if solved_at is None else solved_at)
    assert len(played) == 2 and [r["environment_success"] for r in model.results] == [False, True]


def test_an_episode_under_the_adapter_runs_the_real_loop(monkeypatch, tmp_path):
    """model.py with pyrualean.play as it is: only the provider and SAM3 are stood in for."""

    from pyrualean import openai_runtime as orun
    from pyrualean.hosts import sam3_local

    scene = DepthScene([CUBE])

    def reply(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
        call = {"name": name, "arguments": json.dumps(arguments)}
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": call}],
        }
        return {
            "choices": [{"message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    cell = (
        "seen = robo.locate('the cube')\n"
        "moved = robo.move_to([0.25, 0.0, 0.95], arm='right', pitch=45)\n"
        "print(seen.found, moved.reached)\n"
    )
    replies = [
        reply("python", {"code": cell}, "c1"),
        reply("finish", {"status": "success", "summary": "moved"}, "c2"),
    ]
    sent: list[dict[str, Any]] = []

    def transport(api_key):
        def post(url, body, timeout_s):
            sent.append({"url": url, "model": body["model"], "max_tokens": body.get("max_tokens")})
            return 200, replies.pop(0)

        return post

    monkeypatch.setattr(orun, "httpx_transport", transport)
    loaded: list[Any] = []

    def load(checkpoint=None, cuda_device=None):
        loaded.append(Segmenter(scene))
        return loaded[-1]

    monkeypatch.setattr(sam3_local, "LocalSam3", load)
    monkeypatch.setattr(sam3_local, "_SHARED", {})
    monkeypatch.setenv("OPENAI_BASE_URL", "http://provider.test/v1")
    monkeypatch.chdir(tmp_path)  # an episode moves into its workspace; pytest moves back
    module = _adapter_model(monkeypatch)
    cfg = {
        "poll_s": 0.05,
        "model": "some-model",
        "env_cfg_type": "arx_x5",
        "result_dir": "results",
        "max_turns": 5,
        "openai_options": {"max_tokens": 64, "nudges": 2},
    }
    model = module.Model(cfg)
    task_env = TaskEnv(scene, step_lim=200)
    _deploy().eval_one_episode(task_env, Client(model))
    # the client is answered when the episode's record is written: nothing left to wait for
    assert not model._episode.is_alive() and replies == []
    assert [r["status"] for r in model.results] == ["completed"]
    assert sent[0] == {
        "url": "http://provider.test/v1/chat/completions",
        "model": "some-model",
        "max_tokens": 64,
    }
    (out,) = (tmp_path / "results").iterdir()  # under the directory the server started in
    result = json.loads((out / "result.json").read_text())
    assert result["robot"] == "robodojo_xpl_rgb" and result["finish"]["status"] == "success"
    assert result["host"]["boot_options"] == {"bridge": "<Bridge>"}
    assert result["environment_success"] is False and result["status"] == "completed"
    assert result["env_steps"] == task_env.take_action_cnt[0] > 0
    (turn,) = [
        t for t in json.loads((out / "tool_turns.json").read_text()) if t["tool"] == "python"
    ]
    assert turn["ok"] is True and turn["moved"] is True
    # the next episode of the same server: SAM3 is the one already loaded
    replies.append(reply("finish", {"status": "stuck", "summary": "nothing to do"}, "c3"))
    _deploy().eval_one_episode(TaskEnv(scene, step_lim=200), Client(model))
    model._episode.join(30.0)
    assert len(model.results) == 2 and len(loaded) == 1
    assert len(list((tmp_path / "results").iterdir())) == 2


def test_a_trial_the_client_ends_is_wrapped_up_without_another_request(monkeypatch, tmp_path):
    """RoboDojo ends an episode the moment its check passes, mid-program.  The verdict's
    acknowledgement then waits for the episode's record: a request to the model at that
    point (and a rate limit on it) once held it past the client's timeout, and the trial
    the client had just judged solved was not recorded."""

    from pyrualean import openai_runtime as orun
    from pyrualean.hosts import sam3_local

    scene = DepthScene([CUBE])
    cell = "moved = robo.move_to([0.25, 0.0, 0.95], arm='right', pitch=45)\nprint(moved.reached)\n"
    call = {"name": "python", "arguments": json.dumps({"code": cell})}
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": call}],
    }
    replies = [{"choices": [{"message": message, "finish_reason": "tool_calls"}], "usage": {}}]
    asked: list[int] = []

    def transport(api_key):
        def post(url, body, timeout_s):
            asked.append(len(body["messages"]))
            return (200, replies.pop(0)) if replies else (429, "")  # a second request would stall

        return post

    monkeypatch.setattr(orun, "httpx_transport", transport)
    monkeypatch.setattr(sam3_local, "LocalSam3", lambda checkpoint=None, cuda_device=None: None)
    monkeypatch.setattr(sam3_local, "_SHARED", {})
    monkeypatch.setenv("OPENAI_BASE_URL", "http://provider.test/v1")
    monkeypatch.chdir(tmp_path)
    module = _adapter_model(monkeypatch)
    cfg = {"poll_s": 0.05, "model": "some-model", "result_dir": "results", "wrap_up_s": 20.0}
    model = module.Model({**cfg, "env_cfg_type": "arx_x5", "max_turns": 5})
    task_env = TaskEnv(scene, step_lim=200, solved_at=3)  # solved three actions into the move
    began = time.perf_counter()
    _deploy().eval_one_episode(task_env, Client(model))
    assert time.perf_counter() - began < 15.0  # the acknowledgement did not wait for a model
    assert not model._episode.is_alive() and asked == [1]
    assert task_env.success == [True] and task_env.take_action_cnt[0] == 3
    (out,) = (tmp_path / "results").iterdir()
    result = json.loads((out / "result.json").read_text())
    assert result["environment_success"] is True and result["status"] == "completed"
    assert result["generation"]["requests"] == 1 and result["finish"] is None


def test_the_clients_trial_stands_when_the_verdict_cannot_be_delivered():
    class Deaf(Client):
        def call(self, func_name=None, obs=None):
            if func_name == "trial_end":
                raise TimeoutError("timeout waiting for MessageType.TRIAL_END_ACK")
            return super().call(func_name, obs)

    bridge = xb.Bridge(poll_s=0.05)
    task_env = TaskEnv(Scene([CUBE]), step_lim=200)
    thread = threading.Thread(
        target=_deploy().eval_one_episode,
        args=(task_env, Deaf(xb.PolicyModel(bridge))),
        daemon=True,
    )
    thread.start()
    bridge.first_obs(5.0, alive=thread.is_alive)
    bridge.finish()  # the program has nothing to do
    thread.join(5.0)
    assert not thread.is_alive() and task_env.end_flag == [True]  # judged, and no exception


def test_the_adapter_runs_the_colour_robot_whatever_the_client_sends(monkeypatch, tmp_path):
    from pyrualean import play

    robots: list[str] = []

    def play_episode(cell, out_dir, **kwargs):
        robots.append(kwargs["robot"])
        kwargs["boot_options"]["bridge"].finish()
        return {"status": "completed", "environment_success": False, "task": cell.task}

    monkeypatch.setattr(play, "play_episode", play_episode)
    module = _adapter_model(monkeypatch)
    cfg = {"poll_s": 0.05, "model": "some-model", "result_dir": tmp_path, "action_type": "joint"}
    # a depth image in the observation changes nothing: the adapter's robot does not read it
    for scene in (DepthScene([CUBE]), Scene([CUBE])):
        model = module.Model({**cfg, "env_cfg_type": "arx_x5"})
        model.reset()
        model.update_obs(scene.get_obs())
        model._episode.join(5.0)
        assert robots[-1] == "robodojo_xpl_rgb"
    # the robot deploy.yml names is the robot that runs
    import yaml

    deploy = Path(module.__file__).with_name("deploy.yml")
    assert yaml.safe_load(deploy.read_text())["robot"] == "robodojo_xpl_rgb"
    model = module.Model({**cfg, "robot": "some_other_robot"})
    model.reset()
    model.update_obs(Scene([CUBE]).get_obs())
    model._episode.join(5.0)
    assert robots[-1] == "some_other_robot"
    # what the kinematics do not drive is refused when the server starts, not mid-episode
    with pytest.raises(ValueError, match="six-joint"):
        module.Model({**cfg, "env_cfg_type": "franka"})
    with pytest.raises(ValueError, match="joint"):
        module.Model({**cfg, "action_type": "ee"})


def test_a_reset_does_not_rewrite_a_verdict_the_client_gave():
    bridge = xb.Bridge()
    bridge.episode_end({"success": True, "step": 7})
    bridge.abandon()
    assert bridge.over == {"success": True, "step": 7}
    left = xb.Bridge()
    left.abandon()  # an episode the client left without a verdict
    assert left.over["abandoned"] is True and left.get_action() == []
    with pytest.raises(xb.EpisodeOver):
        left.step({})


# -- what a deployed arm does to the pictures -------------------------------------------------


class LateScene(Scene):
    """RoboDojo's evaluation client: each observation's pictures show the robot one observation
    earlier, and the host says so (``image_state``)."""

    def __init__(self, shapes, tell: bool = True) -> None:
        super().__init__(shapes)
        self.tell = tell
        self.pictured = self.state()
        self._now = self.state()

    def step(self, action):
        out = super().step(action)
        self.pictured, self._now = self._now, self.state()
        return out

    def view(self, camera: str) -> sight.View:
        if camera == "cam_head":
            return sight.head_view()
        return sight.wrist_view(
            self.pictured["left_ee_pose" if "left" in camera else "right_ee_pose"]
        )

    def get_obs(self):
        obs = super().get_obs()
        if self.tell:
            obs["image_state"] = self.pictured
        return obs


def _swing(scene: Scene) -> None:
    """One action that turns the right arm, so that its camera has just moved."""

    goal = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    hold = {f"{arm}_ee_joint_state": [1.0] for arm in kin.ARMS}
    scene.step({"left_arm_joint_state": [0.0] * 6, "right_arm_joint_state": goal, **hold})


def test_pictures_one_observation_late_are_read_with_the_state_they_show():
    told = LateScene([CUBE])
    _swing(told)
    assert told.pictured != told.state()  # the arm moved in the last action
    out = tools.locate(native._Primitives(told, Segmenter(told)), "the cube")
    assert out["views"] == 2 and _off(out["center_xyz"], CUBE) < 0.003
    assert out["height_m"] == pytest.approx(0.035, abs=0.004)
    # the same pictures read against the state of the moment: the wrist camera is taken to be
    # where it no longer was, and the two cameras cannot agree on the cube
    untold = LateScene([CUBE], tell=False)
    _swing(untold)
    wrong = tools.locate(native._Primitives(untold, Segmenter(untold)), "the cube")
    worse = wrong["views"] == 1 or abs(wrong["height_m"] - 0.035) > 0.006
    assert worse or _off(wrong["center_xyz"], CUBE) > 0.004


def test_the_bridge_says_which_state_the_pictures_show():
    def obs(state, behind):
        info = {xb.STATUS_KEY: {"step": 0, "pictures_behind": behind}}
        return {"state": state, "vision": {}, "additional_info": info}

    first, second = {"right_ee_pose": [1.0]}, {"right_ee_pose": [2.0]}
    # an older RoboDojo client: the pictures are of the observation before
    bridge = xb.Bridge()
    bridge.update_obs(obs(first, 1))
    assert bridge.get_obs()["image_state"] is first  # nothing has moved yet
    bridge.update_obs(obs(second, 1))
    assert bridge.get_obs()["image_state"] is first and bridge.get_obs()["state"] is second
    assert tools._shown(bridge.get_obs()) is first
    # a client that renders the current state before it captures (RoboDojo 6e3190f and later)
    bridge = xb.Bridge()
    bridge.update_obs(obs(first, 0))
    bridge.update_obs(obs(second, 0))
    assert bridge.get_obs()["image_state"] is second
    assert tools._shown({"state": second}) is second  # a host that says nothing


def test_deploy_tells_an_older_client_from_one_that_renders_before_it_captures(monkeypatch):
    import sys
    import types

    deploy = _deploy()
    for name in [n for n in sys.modules if n == "env" or n.startswith("env.")]:
        monkeypatch.delitem(sys.modules, name)
    assert deploy._pictures_behind() == 1  # no RoboDojo render_sync module: the older client
    packages = {}
    for name in ("env", "env.camera_manager", "env.camera_manager.capture"):
        packages[name] = types.ModuleType(name)
        packages[name].__path__ = []  # a package
        monkeypatch.setitem(sys.modules, name, packages[name])
    sync = types.ModuleType("env.camera_manager.capture.render_sync")
    monkeypatch.setitem(sys.modules, sync.__name__, sync)
    packages["env.camera_manager.capture"].render_sync = sync
    assert deploy._pictures_behind() == 0
    assert deploy._status(TaskEnv(Scene([CUBE]), step_lim=10))["pictures_behind"] == 0


class Told(Segmenter):
    """A segmenter that answers one prompt with a given mask and score."""

    def __init__(self, scene, mask, score) -> None:
        super().__init__(scene)
        self.mask, self.score = mask, score

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        if (
            text_prompt == "the white block"
            and CAMERAS[int(np.asarray(image)[0, 0, 0])] == "cam_head"
        ):
            return Segmented(True, score=self.score, box=None, mask=self.mask)
        return super().segment(image, text_prompt=text_prompt, point=point, min_score=min_score)


def test_a_mask_on_the_robots_own_gripper_is_not_measured_as_an_object():
    scene = Scene([CUBE])
    head = sight.head_view()
    body = kin.fk("right", scene.joints["right"])
    at = body[:3, 3] + body[:3, 2] * sight.BODY_Z_M + body[:3, 0] * 0.05  # the camera housing
    r, c, _ = head.project(at[None])
    rows, cols = np.mgrid[0:480, 0:640]
    on_gripper = (rows - r[0]) ** 2 + (cols - c[0]) ** 2 <= 8**2
    unsure = tools.locate(
        native._Primitives(scene, Told(scene, on_gripper, 0.26)), "the white block"
    )
    assert unsure["found"] and "center_xyz" not in unsure
    assert unsure["geometry_error"] == "the mask lies on the robot's own arm or gripper"
    assert unsure["robot_px_fraction"] > 0.9
    # a block seen under the open gripper lies on that outline too, and is found for certain
    sure = tools.locate(native._Primitives(scene, Told(scene, on_gripper, 0.85)), "the white block")
    assert "center_xyz" in sure and sure["robot_px_fraction"] > 0.9


def test_what_is_not_on_the_table_is_not_located():
    scene = Scene([CUBE])
    far = np.zeros((480, 640), dtype=bool)
    far[24:36, 483:540] = True  # something white on the wall behind the table
    out = tools.locate(native._Primitives(scene, Told(scene, far, 0.32)), "the white block")
    assert out["found"] and "center_xyz" not in out
    assert out["geometry_error"] == "what the mask shows is not on the table"


def test_a_height_the_cameras_agreed_on_better_stands_against_a_poorer_look():
    cams = _scenes.views()
    head_owner = render(cams["cam_head"], [CUBE])
    head = sight.Sight(cams["cam_head"], head_owner == 0)
    wrist = render(cams["cam_right_wrist"], [CUBE]) == 0
    full = sight.measure(head, [sight.Sight(cams["cam_right_wrist"], wrist)])
    assert full["views"] == 2 and not full["remembered"] and full["match"] > 0.95
    # the same camera now shows the lower half of the cube only (the picture's edge, a finger)
    rows = np.nonzero(wrist.any(axis=1))[0]
    half = wrist.copy()
    half[: (rows[0] + rows[-1]) // 2] = False
    poor = sight.measure(head, [sight.Sight(cams["cam_right_wrist"], half)])
    kept = sight.measure(
        head,
        [sight.Sight(cams["cam_right_wrist"], half)],
        known=(full["height_m"], full["kind"], full["match"]),
    )
    assert kept["remembered"] and kept["views"] == 1
    assert kept["height_m"] == pytest.approx(full["height_m"], abs=0.0021)
    assert poor["views"] == 1 or poor["match"] < full["match"] - sight.KNOWN_BETTER
    # ... but not once a prism that tall no longer fits what the head camera shows
    low = dict(CUBE, size=(0.035, 0.035, 0.012))
    flat = sight.Sight(cams["cam_head"], render(cams["cam_head"], [low]) == 0)
    fresh = sight.measure(flat, [], known=(0.12, "prism", 0.99))
    assert not fresh["remembered"] and fresh["height_m"] < 0.03


class HalfSeen(Segmenter):
    """A segmenter that gives the wrist cameras one side of an object only."""

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        res = super().segment(image, text_prompt=text_prompt, point=point, min_score=min_score)
        if res.found and CAMERAS[int(np.asarray(image)[0, 0, 0])] != "cam_head":
            cols = np.nonzero(res.mask.any(axis=0))[0]
            side = res.mask.copy()
            side[:, (cols[0] + cols[-1]) // 2 :] = False
            return Segmented(True, score=res.score, box=None, mask=side)
        return res


def test_cameras_that_agree_only_roughly_say_so():
    scene = Scene([CUBE])
    whole = tools.locate(native._Primitives(scene, Segmenter(scene)), "the cube")
    assert whole["views"] == 2 and "height_guess" not in whole
    out = tools.locate(native._Primitives(scene, HalfSeen(scene)), "the cube")
    # half a mask fits no solid well: the height is not to be relied on, and the result says so
    assert out["views"] == 1 or "roughly" in out["height_guess"]
    assert _off(out["center_xyz"], CUBE) < 0.02


def test_an_object_the_arm_now_hides_part_of_gives_no_wrong_place():
    scene = Scene([dict(CUBE, size=(0.05, 0.05, 0.19))])  # a tall box, both cameras on it
    prim = native._Primitives(scene, Segmenter(scene))
    whole = tools.locate(prim, "the cube")
    assert whole["views"] == 2 and whole["height_m"] == pytest.approx(0.19, abs=0.02)
    # later the arm is over it: the head camera shows its lower part, no other camera any
    head = sight.head_view()
    low = render(head, scene.shapes) == 0
    rows = np.nonzero(low.any(axis=1))[0]
    low[: (rows[0] + rows[-1]) // 2] = False
    cut = int(np.nonzero(low.any(axis=1))[0][0])
    cols = np.nonzero(low[cut])[0]
    arm = np.zeros_like(low)
    arm[max(0, cut - 30) : cut, cols[0] - 5 : cols[-1] + 6] = True  # the robot, right above the cut

    class Hidden(Segmenter):
        def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
            if CAMERAS[int(np.asarray(image)[0, 0, 0])] == "cam_head" and point is None:
                return Segmented(True, score=0.9, box=None, mask=low)
            return Segmented(False)

    assert tools._behind_robot(low, arm) and not tools._behind_robot(low, np.zeros_like(low))
    prim.sam3_client = Hidden(scene)
    scene.step_count += 1
    real = sight.robot_pixels
    try:
        sight.robot_pixels = lambda *a, **k: arm  # the outline of an arm that is over the box
        hidden = tools.locate(prim, "the cube")
    finally:
        sight.robot_pixels = real
    assert "center_xyz" not in hidden and "hides part of it" in hidden["geometry_error"]
    # the same half mask with no arm near it: the object is lower now, and is measured afresh
    scene.step_count += 1
    fresh = tools.locate(prim, "the cube")
    assert "center_xyz" in fresh and fresh["views"] == 1


def test_a_height_is_not_kept_from_a_rough_reading_nor_used_where_a_taller_thing_fits():
    cams = _scenes.views()
    cube = sight.Sight(cams["cam_head"], render(cams["cam_head"], [CUBE]) == 0)
    # the cube's own height is the tallest that fits its mask: it is taken back
    same = sight.measure(cube, [], known=(0.035, "prism", 0.97))
    assert same["remembered"] and not same["outgrown"]
    # a mask that allows something far taller (the cube with the arm above it, as one mask) is
    # not the cube at rest: its remembered height is left alone
    tall = dict(CUBE, size=(0.035, 0.035, 0.19))
    merged = sight.Sight(cams["cam_head"], render(cams["cam_head"], [tall]) == 0)
    other = sight.measure(merged, [], known=(0.035, "prism", 0.97))
    assert not other["remembered"] and other["outgrown"]

    # and the tools keep a height only when the cameras agreed on it well
    scene = Scene([CUBE])
    prim = native._Primitives(scene, HalfSeen(scene))
    rough = tools.locate(prim, "the cube")
    if rough["views"] >= 2:
        assert "roughly" in rough["height_guess"] and not tools._heights(prim)
    prim.sam3_client = Segmenter(scene)
    scene.step_count += 1
    clean = tools.locate(prim, "the cube")
    assert clean["views"] == 2 and "the cube" in tools._heights(prim)


# -- the same robot with the depth image the client can be configured to send ------------------


def _depth(view: sight.View, shapes: list[dict[str, Any]]) -> np.ndarray:
    """What a depth camera at ``view`` reads of boxes on the table: distance along its axis."""

    rows, cols = np.mgrid[0 : view.shape[0], 0 : view.shape[1]]
    d = view.rays(rows, cols)
    pos = view.pos
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = (kin.TABLE_Z - pos[2]) / d[..., 2]
        hit = np.where(hit > 0, hit, np.inf)
        for shape in shapes:
            lx, ly, h = shape["size"]
            a = math.radians(shape.get("yaw", 0.0))
            to_box = np.array(
                [[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1]]
            )
            centre = np.array([*shape["center"], kin.TABLE_Z + h / 2])
            o, db = to_box @ (pos - centre), d @ to_box.T
            half = np.array([lx / 2, ly / 2, h / 2])
            t1, t2 = (-half - o) / db, (half - o) / db
            near, far = np.minimum(t1, t2).max(axis=-1), np.maximum(t1, t2).min(axis=-1)
            hit = np.minimum(hit, np.where((near <= far) & (near > 0), near, np.inf))
    points = pos + d * np.where(np.isfinite(hit), hit, 0.0)[..., None]
    axial = -((points - pos) @ view.rot)[..., 2]
    return np.where(np.isfinite(hit), axial, 0.0).astype(np.float32)


class DepthScene(Scene):
    """RoboDojo's client with depth and the camera matrices switched on in its env_cfg."""

    def __init__(self, shapes, key: str = "depth") -> None:
        super().__init__(shapes)
        self.key = key

    def get_obs(self):
        obs = super().get_obs()
        for camera, cam in obs["vision"].items():
            view = self.view(camera)
            depth = _depth(view, self.shapes)
            if self.key == "approximate_depth":  # whole millimetres, as in RoboDojo's data
                cam[self.key] = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
            else:
                cam[self.key] = depth
            cam["intrinsic_matrix"] = [[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]]
            cam["extrinsic_matrix"] = view.pose.tolist()
        return obs


@pytest.mark.parametrize("key", ["depth", "approximate_depth"])
def test_with_depth_in_the_observation_the_tools_read_it(key, tmp_path):
    scene = DepthScene([CUBE], key)
    sam3 = Segmenter(scene)
    toolkit = native.NativeToolkit(scene, sam3, tmp_path)  # the observation decides
    backend = native.NativeBackend(toolkit)
    assert backend.colour_only is False
    out = backend.execute_tool("locate", {"text_prompt": "the cube"})
    assert out["found"] and "views" not in out and "geometry_error" not in out
    assert _off(out["center_xyz"], CUBE) < 0.003
    assert out["height_m"] == pytest.approx(0.035, abs=0.003)
    assert {camera for camera, _ in sam3.calls} == {"cam_head"}  # no other camera is asked
    # a pixel is the point the depth image has there
    rows, cols = np.nonzero(render(sight.head_view(), scene.shapes) == 0)
    top = backend.execute_tool(
        "back_project", {"row": int(rows.min()) + 3, "col": int(np.median(cols))}
    )
    assert top["world_xyz"][2] == pytest.approx(kin.TABLE_Z + 0.035, abs=0.003)
    assert "surface" not in top and top["depth_m"] > 0.3
    assert "error" in backend.execute_tool("back_project", {"row": 900, "col": 0})
    # ... and the robot of pyrualean.robodojo runs on it as it is
    robo = RobodojoOfficialRobot(backend, primitives="no-vla")
    pose = robo.locate("the cube")
    assert type(pose).__name__ == "ObjectPose" and _off(pose.center_xyz, CUBE) < 0.003
    meta = robo.camera_meta("cam_right_wrist")
    assert meta["has_depth"] is True and len(meta["extrinsic_matrix"]) == 4
    assert robo.back_project(int(rows.min()) + 3, int(np.median(cols))).xyz[2] > kin.TABLE_Z + 0.03
    # told to read no depth, the same toolkit measures from the colour images
    plain = native.NativeToolkit(scene, Segmenter(scene), tmp_path / "rgb", colour_only=True)
    assert plain.execute_tool("locate", {"text_prompt": "the cube"})["views"] == 2


def test_the_xpolicylab_robot_is_the_robodojo_robot_and_asks_for_depth(tmp_path):
    from pyrualean.hosts import xpolicylab_robodojo as xhost
    from pyrualean.hosts.rpent_robodojo import Cell
    from pyrualean.robodojo import ADAPTER as base

    adapter = get_adapter("robodojo_xpl")
    assert adapter.knowledge == base.knowledge and adapter.host.endswith("xpolicylab_robodojo")
    assert api_reference(adapter.robot_cls, primitives="no-vla") == api_reference(
        base.robot_cls, primitives="no-vla"
    ).replace(base.robot_cls.__name__, adapter.robot_cls.__name__)
    cell = Cell(suite="arx_x5", task="stack_blocks", seed=0)
    # a client that sends depth: the tools read it
    bridge = xb.Bridge()
    bridge.update_obs(DepthScene([CUBE]).get_obs())
    toolkit, daemons = xhost.boot(cell, tmp_path / "a", env_only=True, bridge=bridge)
    assert daemons == [] and toolkit._primitives.colour_only is False
    # a client left at its default: refused, with the way out
    bare = xb.Bridge()
    bare.update_obs(Scene([CUBE]).get_obs())
    with pytest.raises(RuntimeError, match="robodojo_xpl_rgb"):
        xhost.boot(cell, tmp_path / "b", env_only=True, bridge=bare)
    rgb = get_adapter("robodojo_xpl_rgb").host_module()
    toolkit, _ = rgb.boot(cell, tmp_path / "c", env_only=True, bridge=bare)
    assert toolkit._primitives.colour_only is True


def _checkout(tmp_path):
    """A RoboDojo checkout as far as the layout reads it: the tree, XPolicyLab inside, Assets."""

    import yaml

    from pyrualean.hosts import xpolicylab_robodojo as xhost

    source = tmp_path / "RoboDojo"
    for part in xhost.client_root().SOURCE_PARTS:
        (source / part).mkdir(parents=True)
    (source / "env" / "global_configs.py").write_text("ROOT = __file__\n")
    (source / "env_cfg" / "camera").mkdir()
    top = {
        "config_name": "arx_x5",
        "config": {"camera": "camera_config"},
        "observation": {"vision": {"depth": False, "intrinsic_matrix": False, "shape": True}},
    }
    (source / "env_cfg" / "arx_x5.yml").write_text(yaml.safe_dump(top))
    camera = {"annotator": {"cam_head": {"enabled": True, "rgb_capture": {"type": "rgb"}}}}
    (source / "env_cfg" / "camera" / "camera_config.yml").write_text(yaml.safe_dump(camera))
    xpl = source / "XPolicyLab"
    for name in ("client_server/ws", "utils", "policy/ACT", ".git"):
        (xpl / name).mkdir(parents=True)
    (xpl / "client_server" / "ws" / "model_server.py").write_text("")
    (xpl / "utils" / "process_data.py").write_text("")
    (xpl / "policy" / "__init__.py").write_text("")
    (xpl / "policy" / "ACT" / "model.py").write_text("")
    layouts = source / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layouts.mkdir(parents=True)
    for layout in range(4):
        (layouts / f"stack_blocks_{layout}.json").write_text("{}")
    (layouts / "stack_blocks_random_0.json").write_text("{}")
    return source, top


def test_the_host_switches_depth_on_in_its_own_copy_of_the_config(tmp_path):
    import yaml

    from pyrualean.hosts import xpolicylab_robodojo as xhost

    source, top = _checkout(tmp_path)
    xpl = xhost._xpolicylab_root(None, source)  # the checkout's own
    assert xpl == source / "XPolicyLab"
    bench = xhost.lay_out(tmp_path / "bench", source, xpl, depth=True)
    # copies: RoboDojo finds env_cfg and Assets by ".." from its own files
    for part in xhost.client_root().SOURCE_PARTS:
        assert (bench / part).is_dir() and not (bench / part).is_symlink()
    assert (bench / "env" / "global_configs.py").is_file()
    assert (bench / "Assets").is_symlink() and (bench / "Assets" / "Eval_Layout").is_dir()
    vision = yaml.safe_load((bench / "env_cfg" / "arx_x5.yml").read_text())["observation"]["vision"]
    assert vision == {
        "depth": True,
        "intrinsic_matrix": True,
        "extrinsic_matrix": True,
        "shape": True,
    }
    captures = yaml.safe_load((bench / "env_cfg" / "camera" / "camera_config.yml").read_text())
    for name in ("common", "cam_head", "cam_left_wrist", "cam_right_wrist"):
        got = captures["annotator"][name]["distance_to_image_plane_capture"]
        assert got["type"] == "distance_to_image_plane"
    assert captures["annotator"]["cam_head"]["rgb_capture"] == {"type": "rgb"}
    # XPolicyLab's own code and this policy, no other policy, nothing of its repository
    assert (bench / "XPolicyLab" / "utils" / "process_data.py").is_file()
    assert sorted(p.name for p in (bench / "XPolicyLab" / "policy").iterdir()) == [
        "PyRUA_Lean",
        "__init__.py",
    ]
    assert (bench / "XPolicyLab" / "policy" / "PyRUA_Lean" / "deploy.py").is_file()
    assert not (bench / "XPolicyLab" / ".git").exists()
    # RoboDojo's own tree is as it was; without depth the copy keeps its switches
    assert yaml.safe_load((source / "env_cfg" / "arx_x5.yml").read_text()) == top
    plain = xhost.lay_out(tmp_path / "plain", source, xpl)
    assert yaml.safe_load((plain / "env_cfg" / "arx_x5.yml").read_text()) == top


def test_the_host_picks_one_layout_the_way_xpolicylab_does(tmp_path):
    from pyrualean.hosts import xpolicylab_robodojo as xhost
    from pyrualean.hosts.rpent_robodojo import Cell

    source, _ = _checkout(tmp_path)
    bench = xhost.lay_out(tmp_path / "bench", source, source / "XPolicyLab", depth=True)
    cell = Cell(suite="arx_x5", task="stack_blocks", seed=2)
    info = "ckpt_name=pyrualean,action_type=joint"
    manifest = xhost.only_layout(bench, cell, eval_seed=0, additional_info=info, run_id="run-7")
    # where RoboDojo's client looks for it, relative to the directory it runs from
    assert manifest == (
        bench / "eval_result/RoboDojo/stack_blocks/PyRUA_Lean/arx_x5" / f"0_{info}"
        / "_resume_run-7.json"
    )  # fmt: skip
    data = json.loads(manifest.read_text())
    # stack_blocks_random_0.json is another task's layout: four layouts, three left out
    assert data["abandoned_layout_ids"] == [0, 1, 3] and data["completed_layout_ids"] == []
    assert data["success_nums"] == data["fail_nums"] == 0 and data["details"] == {}
    assert data["save_dir"] == str(manifest.parent / "run-7")
    with pytest.raises(ValueError, match="holds 4"):
        xhost.only_layout(
            bench,
            Cell(suite="arx_x5", task="stack_blocks", seed=4),
            eval_seed=0,
            additional_info=info,
            run_id="run-8",
        )


def test_a_root_with_depth_can_write_its_results_into_the_checkout(tmp_path):
    import yaml

    from pyrualean.hosts import xpolicylab_robodojo as xhost

    source, top = _checkout(tmp_path)
    module = xhost.client_root()
    root = module.lay_out(tmp_path / "root", source, results=source / "eval_result")
    assert (root / "eval_result").is_symlink()
    assert (root / "eval_result").resolve() == (source / "eval_result").resolve()
    vision = yaml.safe_load((root / "env_cfg" / "arx_x5.yml").read_text())["observation"]["vision"]
    assert vision["depth"] is True and vision["extrinsic_matrix"] is True
    # no policy named: none is copied; RoboDojo's own tree is as it was
    assert sorted(p.name for p in (root / "XPolicyLab" / "policy").iterdir()) == ["__init__.py"]
    assert yaml.safe_load((source / "env_cfg" / "arx_x5.yml").read_text()) == top


def test_the_adapter_starts_the_client_as_robodojo_configures_it():
    """The submitted directory holds nothing that changes what the client sends."""

    adapter = Path(__file__).resolve().parents[1] / "integrations/xpolicylab/PyRUA_Lean"
    assert not (adapter / "client_root.py").exists()
    client = (adapter / "setup_eval_env_client.sh").read_text()
    assert "env_cfg" not in client.replace("env_cfg_type", "") and "client_root" not in client
    import yaml

    assert "observation" not in yaml.safe_load((adapter / "deploy.yml").read_text())


def test_a_late_picture_gets_the_camera_pose_it_was_taken_from():
    def obs(pose, behind):
        info = {xb.STATUS_KEY: {"pictures_behind": behind}}
        cam = {"color": 0, "extrinsic_matrix": pose}
        return {"state": {"q": pose}, "vision": {"cam_right_wrist": cam}, "additional_info": info}

    old = xb.Bridge()
    old.update_obs(obs("first", 1))
    old.update_obs(obs("second", 1))
    got = old.get_obs()
    assert got["vision"]["cam_right_wrist"]["extrinsic_matrix"] == "first"
    assert got["image_state"] == {"q": "first"} and got["state"] == {"q": "second"}
    new = xb.Bridge()
    new.update_obs(obs("first", 0))
    new.update_obs(obs("second", 0))
    assert new.get_obs()["vision"]["cam_right_wrist"]["extrinsic_matrix"] == "second"


def test_the_adapter_holds_still_in_xpolicylabs_debug_loop(monkeypatch):
    """EVAL_ENV_TYPE=debug: no simulator, no model; the wiring is all that runs."""

    monkeypatch.setenv("EVAL_ENV_TYPE", "debug")
    module = _adapter_model(monkeypatch)
    model = module.Model({"poll_s": 0.05})
    deploy = _deploy()

    class DebugEnv:  # XPolicyLab's TestEnv: its own counter names, nothing to judge
        episode_step, episode_step_limit = 0, 4
        seen: list[dict[str, Any]] = []

        def get_obs(self):
            state = {f"{arm}_arm_joint_state": np.full(6, 0.1, np.float32) for arm in kin.ARMS}
            state.update({f"{arm}_ee_joint_state": np.ones(1, np.float32) for arm in kin.ARMS})
            state["left_ee_pose"] = np.ones(7, np.float32)
            return {"state": state, "vision": {}, "additional_info": {"frequency": 30}}

        def take_action(self, action):
            self.seen.append(action)
            self.episode_step += 1

        def is_episode_end(self):
            return self.episode_step >= self.episode_step_limit

    env = DebugEnv()
    deploy.eval_one_episode(env, Client(model))
    assert len(env.seen) == 4 and model._episode is None  # no episode was started
    assert sorted(env.seen[0]) == sorted(
        f"{arm}_{part}" for arm in kin.ARMS for part in ("arm_joint_state", "ee_joint_state")
    )
    assert env.seen[0]["left_arm_joint_state"] == pytest.approx([0.1] * 6)
    assert deploy._status(env)["step"] == 4 and deploy._status(env)["step_limit"] == 4
