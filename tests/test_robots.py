"""Multi-robot plumbing: registry, base class, generic task card and contract wording."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from pyrualean import LiberoRobot, RobotBase, TaskCard, api_reference, get_adapter
from pyrualean.arms import ArmToolkit, arm_contract, robot_words
from pyrualean.prompt import _public_members
from pyrualean.robots import ROBOTS, RobotAdapter


@dataclass(frozen=True)
class ToyState:
    step: int
    task: str
    eef: list[float]
    terminated: bool


class ToyRobot(RobotBase):
    """A one-tool robot to exercise the base class over the fake backend."""

    NAME: ClassVar[str] = "toy"
    CAMERAS: ClassVar[tuple[str, ...]] = ("head",)
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {("head", "high"): "agentview_high.png"}
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("head", "high"): "agentview_world_high.npz"
    }
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = ("agentview_high.png",)
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = frozenset({"move_to"})

    def _state_from_envelope(self, envelope: dict[str, Any]) -> ToyState:
        raw = envelope.get("state") or {}
        return ToyState(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or ""),
            eef=list(raw.get("robot0_eef_pos") or ()),
            terminated=bool(envelope.get("terminated")),
        )

    def _summary_dict(self) -> dict[str, Any]:
        state = self.state()
        return {"step": state.step, "eef": state.eef, "terminated": state.terminated}

    def go(self, xyz: list[float]) -> dict[str, Any]:
        """Move the arm (a stateful primitive)."""
        return self._stateful("move_to", {"xyz": [float(v) for v in xyz]}, lambda p: dict(p))


def test_registry_knows_libero_and_lists_the_other_robots():
    adapter = get_adapter("libero")
    assert isinstance(adapter, RobotAdapter)
    assert adapter.robot_cls is LiberoRobot and adapter.knowledge == "libero"
    assert adapter.cameras == ("agentview", "wrist")
    assert ROBOTS == ("libero", "robocasa", "robotwin")
    with pytest.raises(ValueError):
        get_adapter("nao")


def test_base_class_drives_the_fake_backend(backend):
    robo = ToyRobot(backend)
    assert robo.task == backend.TASK and not robo.done
    out = robo.go([0.05, -0.10, 0.6])
    assert out["final_dist_m"] == 0.0
    records = robo._ledger.records
    assert [r.tool for r in records] == ["view_env_state", "move_to"]
    assert records[1].stateful and records[1].env_steps
    assert robo._summary_dict()["eef"] == [0.05, -0.1, 0.6]  # reads the state again
    robo.show("head")
    assert robo._show_artifacts(robo._take_show_requests()) == ["agentview_high.png"]
    with pytest.raises(ValueError):
        robo.show("wrist")
    assert robo.image("head").shape == (1024, 1024, 3)
    assert robo.world_map("head").shape == (1024, 1024, 3)
    with pytest.raises(ValueError):
        robo.world_map("head", resolution="low")


def test_api_reference_walks_the_mro_and_hides_base_configuration():
    text = api_reference(ToyRobot)
    assert "robo.go(xyz: list[float])" in text and "robo.state(" in text and "robo.show(" in text
    assert "CAMERAS" not in text and "IMAGE_ARTIFACTS" not in text and "_summary_dict" not in text
    names = list(_public_members(LiberoRobot))
    assert names[:3] == ["task", "done", "state"] and "move_to" in names


def test_arm_toolkit_uses_the_robot_configuration(backend, tmp_path):
    kit = ArmToolkit(ToyRobot(backend), arm="cells", workspace=tmp_path / "ws", images="on-motion")
    moved = kit.execute_tool("python", {"code": "robo.go([0.05, -0.10, 0.6])"})
    assert moved.result["images"] == ["agentview_high.png"] and len(moved.images) == 1
    assert moved.result["state"]["eef"] == [0.05, -0.1, 0.6]


def test_generic_task_card_and_contract_wording(tmp_path):
    img = tmp_path / "head.png"
    img.write_bytes(b"png")
    card = TaskCard(
        robot="robocasa",
        suite="target",
        task="OpenDrawer",
        seed=1,
        task_language="open the drawer",
        object_names=[],
        eef_pos=[0, 0, 0],
        eef_quat=[0, 0, 0, 1],
        gripper_opening=0.0,
        images={"agentview": str(img)},
        facts={"base position [x, y]": "[1.20, -0.35] m"},
    )
    text = card.render()
    assert "- robot: robocasa" in text and "- task: OpenDrawer" in text
    assert "- variant: target" in text and "base position [x, y]: [1.20, -0.35] m" in text
    assert "`robo.image('agentview')`" in text
    adapter = RobotAdapter(
        name="toy",
        robot_cls=ToyRobot,
        host="pyrualean.hosts.rpent_libero",
        knowledge="libero",
        blurb="a toy robot",
        show_examples="`robo.show('head')`",
        services="a frozen VLA",
    )
    words = robot_words(adapter)
    assert words["svc"] == ", a frozen VLA" and words["svc_nl"] == ", a\nfrozen VLA"
    cells = arm_contract(
        "cells", budget_s=1, max_env_steps=1, max_programs=None, images="on-demand", robot=words
    )
    assert (
        "You control a toy robot" in cells
        and "`ToyRobot` bound to the live simulator, a\nfrozen VLA" in cells
    )
    assert "`robo.show('head')`" in cells and "LIBERO" not in cells.split("# How the episode")[0]


class _NpyArtifacts:
    """Wrap the fake backend so a ``.npy`` world map can be served."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def artifact(self, name: str, step: int = -1) -> bytes:
        if name.endswith(".npy"):
            import io

            import numpy as np

            buf = io.BytesIO()
            np.save(buf, np.ones((4, 5, 3), dtype=np.float32))
            return buf.getvalue()
        return self._inner.artifact(name, step)


class NpyRobot(ToyRobot):
    """A robot whose world maps are bare ``.npy`` arrays (RoboTwin style)."""

    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {("head", "high"): "head_world.npy"}


@dataclass(frozen=True)
class BudgetState(ToyState):
    truncated: bool = False


class BudgetRobot(ToyRobot):
    """A robot whose step budget only the parsed state reports (RPent never sets truncated)."""

    RESULT_TYPES = (BudgetState,)

    def _state_from_envelope(self, envelope: dict[str, Any]) -> BudgetState:
        base = super()._state_from_envelope(envelope)
        return BudgetState(**base.__dict__, truncated=base.step >= 1)


def test_world_map_reads_bare_npy_arrays(backend):
    robo = NpyRobot(_NpyArtifacts(backend))
    world = robo.world_map("head")
    assert world.shape == (4, 5, 3) and float(world[0, 0, 2]) == 1.0


def test_absorb_reads_the_episode_flags_from_the_state(backend):
    robo = BudgetRobot(backend)
    robo.go([0.05, -0.10, 0.6])
    assert robo.state().truncated is True
    with pytest.raises(BaseException) as info:  # EpisodeFinished derives from BaseException
        robo.go([0.06, -0.10, 0.6])
    assert type(info.value).__name__ == "EpisodeFinished" and info.value.reason == "truncated"


def test_api_reference_hides_result_types_and_classvar_configuration():
    text = api_reference(BudgetRobot)
    assert "RESULT_TYPES" not in text and "robo.NAME" not in text
    assert "BudgetState" in text and "truncated" in text


def test_task_card_without_a_single_arm_pose_renders_and_round_trips(tmp_path):
    card = TaskCard(
        robot="robotwin",
        suite="demo_randomized",
        task="beat_block_hammer",
        seed=3,
        task_language="hit the block",
        object_names=[],
        facts={"left arm": "[0, 0, 0] m"},
    )
    assert card.eef_pos is None and card.gripper_opening is None
    text = card.render()
    assert "- robot: robotwin" in text and "left arm: [0, 0, 0] m" in text
    path = tmp_path / "card.json"
    path.write_text(card.to_json())
    assert TaskCard.from_json(path) == card
    libero_without_pose = TaskCard(
        suite="libero_goal", task=7, seed=0, task_language="x", object_names=["a"]
    )
    assert "- robot: libero" in libero_without_pose.render()  # generic layout, no crash


def test_rpent_backend_falls_back_to_rpent_internals():
    from pyrualean.hosts.common import RpentBackend

    class _Primitives:
        def recorded_frame_count(self) -> int:
            return 5

    class _Bare:  # RPent's base Toolkit.solved raises; RoboTwin keeps the flag in _latest_status
        _latest_status = {"eval_success": True}
        _primitives = _Primitives()

        def solved(self) -> bool:
            raise NotImplementedError

    class _Full:
        primitives = _Primitives()

        def solved(self) -> bool:
            return False

    assert RpentBackend(_Bare()).solved() is True and RpentBackend(_Bare()).env_steps() == 5
    assert RpentBackend(_Full()).solved() is False and RpentBackend(_Full()).env_steps() == 5
    assert RpentBackend(object()).env_steps() is None
