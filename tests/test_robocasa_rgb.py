"""RoboCasa365 with RoboDojo's primitives: what can be checked without the simulator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pyrualean._backend import EpisodeFinished
from pyrualean.hosts import robocasa_sight as sight
from pyrualean.hosts.robocasa_sim import Kitchen, angles, orientation, pack, unpack
from pyrualean.prompt import TaskCard, api_reference, render_prompt
from pyrualean.robocasa_rgb import ADAPTER, RobocasaColourRobot
from pyrualean.robots import get_adapter

ARM = {
    "tcp_pos": [0.01, 0.24, 1.30],
    "pitch": 72.0,
    "yaw": 16.0,
    "roll": -166.0,
    "eef_pos": [0.02, 0.21, 1.39],
    "eef_quat": [0.41, 0.54, 0.60, -0.43],
    "gripper": 0.51,
    "finger_gap_m": 0.041,
    "joints": [0.0] * 7,
}


class FakeBackend:
    """Answers like :class:`pyrualean.hosts.robocasa_rgb.RobocasaToolkit`, without a kitchen."""

    def __init__(self, over_after: int | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.step = 0
        self.over_after = over_after

    def _envelope(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        over = self.over_after is not None and self.step >= self.over_after
        return {
            "step": self.step,
            "task_language": "Open the right drawer.",
            "terminated": False,
            "truncated": over,
            "state": {
                "state": {"arm": ARM, "torso_m": 0.0, "base": [0.0, 0.0, 0.0]},
                "instruction": "Open the right drawer.",
                "cameras": {"eye_in_hand": {"shape": [480, 640, 3]}},
            },
            "log": {"tool": None, "result": payload or {}},
        }

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        self.calls.append((name, dict(input_dict)))
        if name == "view_env_state":
            return self._envelope()
        if name == "locate":
            return {"found": True, "center_xyz": [0.4, 0.5, 0.8], "views": 2, "score": 0.7}
        self.step += 1
        return self._envelope({"reached": True, "steps_used": 5, "final_angles": {"pitch": 90}})

    def solved(self) -> bool:
        return False

    def artifact(self, name: str, step: int = -1) -> bytes:
        raise FileNotFoundError(name)

    def env_steps(self) -> int | None:
        return self.step

    def cancel(self) -> None:
        pass


def test_adapter_is_registered():
    assert get_adapter("robocasa_rgb") is ADAPTER
    assert ADAPTER.cameras == ("agentview_left", "agentview_right", "eye_in_hand")
    assert not ADAPTER.live_success  # the actor sees no success, as on RoboDojo


def test_reference_has_robodojos_primitives_and_no_success_flag():
    ref = api_reference(RobocasaColourRobot, primitives="no-vla")
    for member in ("move_to", "set_gripper", "home", "locate", "back_project",
                   "move_base", "set_torso", "camera_meta"):
        assert f"robo.{member}" in ref
    assert "robo.done" not in ref and "success_criteria" not in ref and "task_progress" not in ref
    card = TaskCard(suite="target", task="OpenDrawer", seed=1, task_language="Open the right drawer.",
                    object_names=[], max_env_steps=750, robot="robocasa_rgb")
    bundle = render_prompt(card, primitives="no-vla")
    assert "No result carries a task-success flag" in bundle.system
    assert "simulator step budget: 750" in bundle.user


def test_there_is_no_move_arc():
    from pyrualean.hosts.robocasa_sim import MOTIONS, Kitchen

    assert not hasattr(RobocasaColourRobot, "move_arc")
    assert "move_arc" not in RobocasaColourRobot.STATEFUL_TOOLS
    assert "move_arc" not in MOTIONS and not hasattr(Kitchen, "move_arc")
    assert "move_arc" not in api_reference(RobocasaColourRobot, primitives="no-vla")


def test_one_arm_and_absent_done():
    robo = RobocasaColourRobot(FakeBackend(), primitives="no-vla")
    with pytest.raises(AttributeError):
        _ = robo.done
    with pytest.raises(ValueError):
        robo.move_to([0.4, 0.5, 0.9], arm="left")
    move = robo.move_to([0.4, 0.5, 0.9], arm="right", gripper=robo.CLOSE, pitch=90, path="line")
    assert move.reached and move.steps_used == 5
    name, kwargs = robo._backend.calls[-1]
    assert name == "move_pose" and kwargs["pitch"] == 90.0 and "yaw" not in kwargs
    home = robo.home("both")  # learned RoboDojo code homes "both": there is one arm
    assert home.arm == "right"


def test_state_and_locate_parse():
    robo = RobocasaColourRobot(FakeBackend(), primitives="no-vla")
    state = robo.state()
    assert state.right.tcp_pos == (0.01, 0.24, 1.30) and state.arm("right") is state.right
    assert state.base == (0.0, 0.0, 0.0) and not state.truncated
    sight_ = robo.locate("the drawer handle")
    assert sight_.found and sight_.views == 2 and sight_.center_xyz == (0.4, 0.5, 0.8)


def test_episode_end_latches():
    robo = RobocasaColourRobot(FakeBackend(over_after=1), primitives="no-vla")
    robo.move_base(0.0, -0.2, 0.0)
    with pytest.raises(EpisodeFinished):
        robo.set_torso(0.1)


def test_codec_round_trip():
    value = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": [np.bool_(True), 2.5], "c": "x"}
    back = unpack(pack(value))
    assert np.array_equal(back["a"], value["a"]) and back["b"] == [True, 2.5] and back["c"] == "x"


@pytest.mark.parametrize("pitch,yaw,roll", [(0, 0, 0), (45, -30, 20), (89.0, 120, -60), (90, 30, 0)])
def test_gripper_angles_round_trip(pitch, yaw, roll):
    p, y, r = angles(orientation(pitch, yaw, roll))
    if pitch >= 89.5:  # straight down: yaw carries yaw - roll
        assert math.isclose(p, pitch, abs_tol=1e-6)
        return
    assert math.isclose(p, pitch, abs_tol=1e-6)
    assert math.isclose(y, yaw, abs_tol=1e-6) and math.isclose(r, roll, abs_tol=1e-6)


def _finger_line(rot: np.ndarray) -> float:
    """Heading of the line the fingers close along, degrees mod 180 (the fingers are a pair)."""
    jaw = rot[:, 1]
    return math.degrees(math.atan2(jaw[1], jaw[0])) % 180.0


@pytest.mark.parametrize("now", [(88.0, 0.0, -30.0), (87.5, 37.0, 7.0), (87.0, 120.0, 90.0)])
def test_a_vertical_target_ignores_the_roll_read_off_vertical(now):
    """A gripper stopped 2-3 degrees off vertical reads a noisy roll; ``pitch=90`` must not keep it."""
    cur = orientation(*now)
    kitchen = type("K", (), {"gripper_robot": lambda self: (np.zeros(3), cur)})()
    _, rot, want = Kitchen._target(kitchen, [0.0, 0.0, 0.0], 90.0, 75.0, None)
    assert want == (90.0, 75.0, 0.0)
    assert math.isclose(_finger_line(rot), _finger_line(orientation(90.0, 75.0, 0.0)), abs_tol=1e-6)
    _, rot, want = Kitchen._target(kitchen, [0.0, 0.0, 0.0], 90.0, None, None)
    assert want[2] == 0.0 and math.isclose(_finger_line(rot), _finger_line(cur), abs_tol=1e-6)
    _, _, want = Kitchen._target(kitchen, [0.0, 0.0, 0.0], 90.0, 75.0, 20.0)
    assert want == (90.0, 75.0, 20.0)  # an explicit roll is kept
    _, _, want = Kitchen._target(kitchen, [0.0, 0.0, 0.0], None, None, None)
    assert want == angles(cur)  # a plain position move does not turn the gripper


def test_carve_finds_a_box_from_two_views():
    """Two synthetic pinhole cameras see a 6 cm cube; its hull's middle is where the cube is."""

    k = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])

    def camera(pos, look):
        z = np.asarray(look, float) - pos
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        t = np.eye(4)
        t[:3, :3] = np.column_stack([x, y, z])
        t[:3, 3] = pos
        return t

    cube = np.array([0.1, 0.6, 0.93])
    views, masks = {}, {}
    for name, pos in (("a", [-0.35, 0.0, 1.4]), ("b", [0.35, 0.0, 1.4])):
        t = camera(np.array(pos), cube)
        view = sight.View(name, np.zeros((480, 640, 3), np.uint8), k, t,
                          np.zeros((480, 640), bool), np.full((480, 640), np.nan, np.float32))
        g = np.stack(np.meshgrid(*[np.linspace(-0.03, 0.03, 25)] * 3), -1).reshape(-1, 3) + cube
        u, v, _ = view.project(g)
        mask = np.zeros((480, 640), bool)
        mask[np.round(v).astype(int), np.round(u).astype(int)] = True
        views[name], masks[name] = view, [sight._grow(mask, 2)]
    hull, used, agree = sight.carve(views, "a", masks)
    assert set(used) == {"a", "b"} and agree["b"] > 0.9
    centre = np.median(hull[:, :3], axis=0)
    assert np.linalg.norm(centre[:2] - cube[:2]) < 0.02
