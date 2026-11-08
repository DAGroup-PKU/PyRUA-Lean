"""The skill library an episode is given, and the stage count its verdict carries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.hosts import robodojo_native as native
from pyrualean.hosts import xpolicylab_bridge as xb
from pyrualean.play import LIBRARY_NOTE, build_prompt, copy_library, library_lessons


def _colour_tests():
    """The fake scene, segmenter and RoboDojo environment of test_robodojo_colour.py."""

    path = Path(__file__).with_name("test_robodojo_colour.py")
    spec = importlib.util.spec_from_file_location("pyrualean_colour_fakes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fakes = _colour_tests()
BOWL, CUBE, Scene, Segmenter, TaskEnv, _episode = (
    _fakes.BOWL,
    _fakes.CUBE,
    _fakes.Scene,
    _fakes.Segmenter,
    _fakes.TaskEnv,
    _fakes._episode,
)


def _library(root, skill="def hello():\n    return 'v1'\n", lesson="## Grasp\nlower slowly\n"):
    (root / "skills").mkdir(parents=True, exist_ok=True)
    (root / "lessons").mkdir(parents=True, exist_ok=True)
    (root / "skills" / "grasping.py").write_text(skill)
    (root / "lessons" / "grasp.md").write_text(lesson)
    return root


def test_copy_library_records_hashes_and_a_cell_imports_it(robo, tmp_path):
    lib = _library(tmp_path / "lib")
    workspace = tmp_path / "ws"
    record = copy_library(lib, workspace / "library")
    assert set(record["files"]) == {"skills/grasping.py", "lessons/grasp.md"}
    assert len(record["sha256"]) == 64 and record["source"] == str(lib.resolve())
    cells = ArmToolkit(robo, arm="cells", workspace=workspace)
    out = cells.execute_tool(
        "python", {"code": "from library.grasping import hello; print(hello())"}
    )
    assert out.result["stdout"] == "v1\n"


def test_a_later_episode_imports_its_own_library_not_a_cached_one(robo, tmp_path):
    # the adapter runs every episode in one process: an earlier library must not be reused
    first = tmp_path / "ep1" / "workspace"
    copy_library(_library(tmp_path / "lib1"), first / "library")
    ArmToolkit(robo, arm="cells", workspace=first).execute_tool(
        "python", {"code": "import library.grasping"}
    )
    second = tmp_path / "ep2" / "workspace"
    lib2 = _library(tmp_path / "lib2", skill="def hello():\n    return 'v2'\n")
    copy_library(lib2, second / "library")
    out = ArmToolkit(robo, arm="cells", workspace=second).execute_tool(
        "python", {"code": "from library.grasping import hello; print(hello())"}
    )
    assert out.result["stdout"] == "v2\n"
    assert str(first.resolve()) not in sys.path and str(second.resolve()) in sys.path


def test_the_record_hash_changes_with_the_library(tmp_path):
    one = copy_library(_library(tmp_path / "a"), tmp_path / "w1" / "library")
    two = copy_library(
        _library(tmp_path / "b", lesson="## Grasp\nlower faster\n"), tmp_path / "w2" / "library"
    )
    assert one["sha256"] != two["sha256"]
    assert one["files"]["skills/grasping.py"] == two["files"]["skills/grasping.py"]


def test_an_empty_or_missing_library_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        copy_library(tmp_path / "nowhere", tmp_path / "ws" / "library")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        copy_library(tmp_path / "empty", tmp_path / "ws" / "library")


def test_the_prompt_carries_the_note_and_the_lessons_only_with_a_library(tmp_path):
    lib = _library(tmp_path / "lib")
    card = SimpleNamespace(
        robot="robodojo_xpl_rgb", max_env_steps=400, render=lambda: "# Task\nplug"
    )
    common = dict(budget_s=600, max_programs=None, primitives="no-vla")
    plain = build_prompt("cells", card, **common)
    with_lib = build_prompt("cells", card, library=lib, **common)
    assert LIBRARY_NOTE not in plain and "Lessons from the skill library" not in plain
    assert LIBRARY_NOTE in with_lib and "lower slowly" in with_lib
    assert with_lib.index("lower slowly") < with_lib.index("# Task\nplug")
    assert library_lessons(tmp_path / "nowhere") == ""


class StagedTaskEnv(TaskEnv):
    """A task whose ordered checks pop as RoboDojo's reward manager pops them."""

    def __init__(self, *args, stages: int, pop_after: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reward_manager = SimpleNamespace(check_list=[[["check"]] * stages])
        self.pop_after = pop_after

    def take_action(self, action):
        super().take_action(action)
        if self.take_action_cnt[0] == self.pop_after and self.reward_manager.check_list[0]:
            self.reward_manager.check_list[0].pop(0)


def test_the_verdict_records_how_many_stages_a_failed_episode_passed(tmp_path):
    scene = Scene([CUBE, BOWL])
    task_env = StagedTaskEnv(scene, step_lim=200, stages=4, pop_after=1)
    bridge, _client, thread = _episode(task_env)
    toolkit = native.NativeToolkit(
        xb.BridgeEnv(bridge),
        Segmenter(scene),
        tmp_path,
        verdict=lambda: bool((bridge.over or {}).get("success")),
    )
    backend = native.NativeBackend(toolkit)
    backend.execute_tool("move_pose", {"xyz": [0.25, 0.0, 0.95], "arm": "right", "pitch": 45})
    backend.conclude()
    thread.join(5.0)
    record = backend.verdict_record()
    assert record["success"] is False
    assert record["stages_total"] == 4 and record["stages_passed"] == 1


def test_an_environment_without_a_reward_manager_reports_no_stages(tmp_path):
    scene = Scene([CUBE])
    task_env = TaskEnv(scene, step_lim=200)
    bridge, _client, thread = _episode(task_env)
    backend = native.NativeBackend(
        native.NativeToolkit(xb.BridgeEnv(bridge), Segmenter(scene), tmp_path)
    )
    backend.conclude()
    thread.join(5.0)
    assert "stages_total" not in backend.verdict_record()


class RewardManager:
    """RoboDojo's reward manager as deploy.py reads it: ordered stages and check_once."""

    def __init__(self, stages, holding):
        self.check_list = [stages]
        self.holding = holding
        self.evaluated = []

    def check_once(self, check, env_idx):
        self.evaluated.append(check[0])
        return self.holding[check[0]]


class BlockedTaskEnv(TaskEnv):
    def __init__(self, *args, manager, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reward_manager = manager


def _plug_stage():
    return [
        ("is_A_depth_in_B", {"label_A": "charger", "label_B": "socket", "z_threshold": 0.015}),
        ("is_A_in_B", {"label_A": "charger", "label_B": "socket"}),
        ("is_axis_up", {"label": "charger", "axis": [0, 1, 0]}),
        ("all_robot_back_to_origin", {}),
    ]


def _blocked(tmp_path, stage, holding):
    scene = Scene([CUBE])
    manager = RewardManager([stage], holding)
    task_env = BlockedTaskEnv(scene, step_lim=200, manager=manager)
    bridge, _client, thread = _episode(task_env)
    backend = native.NativeBackend(
        native.NativeToolkit(xb.BridgeEnv(bridge), Segmenter(scene), tmp_path)
    )
    backend.conclude()
    thread.join(5.0)
    return backend.verdict_record(), manager


def test_a_failed_verdict_says_which_condition_of_the_blocking_stage_fails(tmp_path):
    holding = {
        "is_A_depth_in_B": False,
        "is_A_in_B": True,
        "is_axis_up": True,
        "all_robot_back_to_origin": True,
    }
    record, manager = _blocked(tmp_path, _plug_stage(), holding)
    conditions = {c["check"]: c for c in record["blocking_stage"]}
    assert conditions["is_A_depth_in_B"]["holds"] is False
    assert conditions["is_A_in_B"]["holds"] is True
    assert conditions["is_A_in_B"]["labels"] == {"label_A": "charger", "label_B": "socket"}
    assert manager.evaluated == [name for name, _ in _plug_stage()]


def test_a_stage_with_a_check_that_writes_state_is_named_but_not_evaluated(tmp_path):
    stage = [("is_lift", {"label": "target0"}), ("update_object_state", {"label": "target0"})]
    record, manager = _blocked(tmp_path, stage, {"is_lift": True, "update_object_state": True})
    assert [c["holds"] for c in record["blocking_stage"]] == [None, None]
    assert [c["check"] for c in record["blocking_stage"]] == ["is_lift", "update_object_state"]
    assert manager.evaluated == []
