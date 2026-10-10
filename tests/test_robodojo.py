"""Behaviour of the typed RoboDojo API over an RPent-shaped fake backend, plus its host."""

from __future__ import annotations

import io
import math
import os
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from pyrualean import (
    EpisodeFinished,
    TaskCard,
    ToolError,
    api_reference,
    get_adapter,
    knowledge_text,
    render_prompt,
)
from pyrualean.arms import ArmToolkit, arm_contract, robot_words
from pyrualean.hosts import robodojo_env_server as support_arm
from pyrualean.hosts import rpent_robodojo as host
from pyrualean.play import build_prompt
from pyrualean.prompt import _public_members
from pyrualean.robodojo import (
    HOME_TOL_RAD,
    ArmState,
    Gripper,
    Home,
    Move,
    ObjectPose,
    Pick,
    RobodojoRobot,
    Segment,
    State,
    WorldPoint,
)

ROWS, COLS = 480, 640


def _axis_quat(degrees: float, axis: int) -> list[float]:
    half = math.radians(degrees) / 2
    quat = [math.cos(half), 0.0, 0.0, 0.0]
    quat[1 + axis] = math.sin(half)
    return quat


def _mul(a: list[float], b: list[float]) -> list[float]:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _quat(pitch: float, yaw: float, roll: float) -> list[float]:
    """The library's convention: Rz(90 + yaw) Ry(pitch) Rx(roll), as [w, x, y, z]."""
    return _mul(_mul(_axis_quat(90.0 + yaw, 2), _axis_quat(pitch, 1)), _axis_quat(roll, 0))


def _pointing(quat: list[float]) -> list[float]:
    w, x, y, z = quat
    return [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)]


class _ToolResult:
    """Shape of ``rpent.tools.toolkit.ToolResult`` (only ``.result`` matters)."""

    def __init__(self, name: str, result: dict[str, Any]) -> None:
        self.name = name
        self.result = result


class FakeRobodojoBackend:
    """Two arms, a bottle and a bin.

    Mirrors RPent's RoboDojo envelope (``robots/robodojo/tools.py``): every
    state-changing tool returns a ``view_env_state`` dict whose ``log.result``
    holds the primitive's own return value; ``segment`` / ``back_project`` /
    ``locate`` return their payload directly.  The motion tools are
    PyRUA-Lean's own (``hosts/robodojo_tools.py``): an arm is its grasp point
    and its pitch / yaw / roll; the recorded ``*_ee_pose`` is the flange, 0.145 m
    behind the grasp point.  The recorded state never says whether the task is
    solved (``solved()`` does: bottle released over the bin).
    """

    TASK = "Put the green bottle into the dustbin."
    STEP_LIMIT = 120
    TABLE_Z = 0.765
    REACH_Z = 0.78  # grasp points below this have no IK solution
    BOTTLE = [0.10, -0.10, 0.85]  # centre; 0.17 m tall, 0.04 m across
    BIN = [-0.60, -0.10, 0.80]
    READONLY = ("view_env_state", "segment", "back_project", "locate")

    def __init__(self) -> None:
        self.arms: dict[str, dict[str, Any]] = {
            "left": {"tcp": [-0.30, -0.20, 1.00], "angles": [0.0, 0.0, 0.0], "grip": 1.0},
            "right": {"tcp": [0.30, -0.20, 1.00], "angles": [0.0, 0.0, 0.0], "grip": 1.0},
        }
        for arm in self.arms.values():
            arm["wrist"] = 0.0
        self.start = {
            name: (list(arm["tcp"]), list(arm["angles"])) for name, arm in self.arms.items()
        }
        self.bottle = list(self.BOTTLE)
        self.step_count = 0
        self.holding: str | None = None
        self.in_bin = False
        self.vla = True
        self.steps: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.solved_calls = 0
        self.cancelled = False
        self._record(None, None)

    # -- Backend contract ---------------------------------------------------

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        self.calls.append((name, dict(input_dict)))
        handler = getattr(self, f"_do_{name}", None)
        if handler is None:
            return _ToolResult(name, {"error": f"unknown tool: {name}"})
        failed = False
        try:
            if self.cancelled and name not in self.READONLY:
                result = {
                    "error": "tool operation interrupted",
                    "code": "tool_cancelled",
                    "interrupted": True,
                }
                failed = True
            else:
                result = handler(**input_dict)
        except TypeError as exc:
            result = {"error": f"bad arguments for {name}: {exc}", "got": input_dict}
            failed = True
        except (ValueError, RuntimeError) as exc:
            result = {"error": str(exc), "traceback": "Traceback ..."}
            failed = True
        if name in self.READONLY:
            return _ToolResult(name, result)
        self._record({"action": name, **input_dict}, result)
        envelope = self._view(-1)
        if failed:
            for key, value in result.items():
                envelope.setdefault(key, value)
        return _ToolResult(name, envelope)

    def solved(self) -> bool:
        self.solved_calls += 1
        return self.in_bin

    def artifact(self, name: str, step: int = -1) -> bytes:
        if name in ("cam_head.png", "cam_left_wrist.png", "cam_right_wrist.png"):
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (COLS, ROWS), (10, 20, 30)).save(buf, format="PNG")
            return buf.getvalue()
        raise FileNotFoundError(name)

    def env_steps(self) -> int:
        return self.step_count

    def cancel(self) -> None:
        self.cancelled = True

    # -- native bookkeeping -------------------------------------------------

    def _spent(self) -> bool:
        return self.step_count >= self.STEP_LIMIT

    def _tick(self, n: int = 1) -> int:
        executed = max(0, min(int(n), self.STEP_LIMIT - self.step_count))
        self.step_count += executed
        return executed

    def _summary(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for name, arm in self.arms.items():
            quat = _quat(*arm["angles"])
            point = _pointing(quat)
            flange = [t - 0.145 * p for t, p in zip(arm["tcp"], point, strict=True)]
            state[f"{name}_ee_pose"] = [*flange, *quat]
            state[f"{name}_ee_joint_state"] = [arm["grip"]]
            state[f"{name}_arm_joint_state"] = [0.0, 0.5, 0.5, 0.0, 0.0, math.radians(arm["wrist"])]
        calibration = {
            "shape": [ROWS, COLS, 3],
            "has_depth": True,
            "intrinsic_matrix": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            "extrinsic_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.3], [0, 0, 0, 1]],
            "color_dtype": "uint8",
        }
        return {
            "instruction": self.TASK,
            "cameras": {camera: dict(calibration) for camera in host.CAMERAS},
            "state": state,
            "eef": {
                "left": state["left_ee_pose"],
                "right": state["right_ee_pose"],
            },
        }

    def _record(self, command: dict[str, Any] | None, result: dict[str, Any] | None) -> None:
        self.steps.append(
            {
                "state": self._summary(),
                "truncated": self._spent(),
                "command": command,
                "result": result,
            }
        )

    def _view(self, step: int) -> dict[str, Any]:
        try:
            record = self.steps[step]
        except IndexError:
            return {"error": f"state step not available: {step}"}
        index = step if step >= 0 else len(self.steps) + step
        return {
            "step": index,
            "truncated": record["truncated"],
            "state": record["state"],
            "artifacts": ["cam_head.png", "cam_left_wrist.png", "cam_right_wrist.png"],
            "task_language": self.TASK,
            "log": {"command": record["command"], "result": record["result"], "elapsed_s": 0.5},
            "_image_bytes": b"png",
            "_image_cam_bytes": b"png",
            "_image_wrist_bytes": b"png",
        }

    def _angles(self, arm: str) -> dict[str, float]:
        pitch, yaw, roll = self.arms[arm]["angles"]
        return {"pitch": pitch, "yaw": yaw, "roll": roll}

    def _gripper(self, arm: str, gripper: float) -> None:
        """A gripper command given with a motion: close holds what is held, open lets go."""
        if gripper > 0:
            self.arms[arm]["grip"] = 0.4 if self.holding == arm else 0.0
        elif gripper < 0:
            self.arms[arm]["grip"] = 1.0
            if self.holding == arm:
                self._release(arm)

    def _release(self, arm: str) -> None:
        self.holding = None
        pos = self.arms[arm]["tcp"]
        self.in_bin = math.dist(pos[:2], self.BIN[:2]) < 0.05 and pos[2] <= 0.90

    # -- RPent's tools (payload keys as in robots/robodojo/tools.py) ----------

    def _do_view_env_state(self, step: int = -1) -> dict[str, Any]:
        return self._view(step)

    def _do_segment(
        self, text_prompt: str, camera: str = "cam_head", min_score: float = 0.2
    ) -> dict[str, Any]:
        if camera == "cam_right_wrist":
            return {"error": "sam3_client not configured"}
        if "bottle" not in text_prompt or min_score > 0.9:
            return {"camera": camera, "found": False, "text_prompt": text_prompt}
        rows, cols, count = 200 * 50, 300 * 50, 50
        return {
            "camera": camera,
            "found": True,
            "score": 0.83,
            "box_px": [290.0, 180.0, 310.0, 220.0],
            "anchor_method": "sam3_mask_centroid_floor_v1",
            "mask_shape": [ROWS, COLS],
            "mask_moments": [count, rows, cols],
            "centroid_rc": [rows // count, cols // count],
        }

    def _do_back_project(self, row: int, col: int, camera: str = "cam_head") -> dict[str, Any]:
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return {"error": f"pixel out of range: row 0..{ROWS - 1}, col 0..{COLS - 1}"}
        if row < 10:
            return {"error": "invalid depth at pixel: inf"}
        xyz = self.BOTTLE if (row, col) == (200, 300) else [0.0, 0.0, self.TABLE_Z]
        return {
            "camera": camera,
            "pixel": [int(row), int(col)],
            "depth_m": 0.6123,
            "world_xyz": [round(float(v), 4) for v in xyz],
        }

    def _do_pi0_pick(
        self, prompt: str | None = None, arm: str = "right", max_chunks: int = 8
    ) -> dict[str, Any]:
        instruction = self.TASK if prompt is None else prompt
        if "<" in instruction or ">" in instruction:
            raise ValueError("RoboDojo instruction contains unresolved template markers")
        if not self.vla:
            return {"error": "vla_client not configured"}
        executed = self._tick(50)
        success = executed == 50
        left = self.arms["left"]
        if success:
            left["grip"] = 0.2
            self.holding = "left"
        return {
            "monitored_arm": arm,
            "instruction": instruction,
            "success": success,
            "chunks_used": 1,
            "arms": {
                "left": {
                    "start_eef_z": 1.0,
                    "min_eef_z": 0.86,
                    "peak_eef_z": 0.93 if success else 0.86,
                    "peak_lift_m": 0.07 if success else 0.0,
                    "start_gripper": 1.0,
                    "final_gripper": left["grip"],
                },
                "right": {
                    "start_eef_z": 1.0,
                    "min_eef_z": 1.0,
                    "peak_eef_z": 1.0,
                    "peak_lift_m": 0.0,
                    "start_gripper": 1.0,
                    "final_gripper": 1.0,
                },
            },
            "terminated": self._spent(),
        }

    # -- PyRUA-Lean's own tools (payload keys as in hosts/robodojo_tools.py) --

    def _do_locate(
        self, text_prompt: str, camera: str = "cam_head", min_score: float = 0.2
    ) -> dict[str, Any]:
        if camera == "cam_right_wrist":
            return {"error": "sam3_client not configured"}
        if "bottle" not in text_prompt or min_score > 0.9:
            return {"camera": camera, "found": False, "text_prompt": text_prompt}
        return {
            "camera": camera,
            "found": True,
            "score": 0.83,
            "centroid_rc": [200, 300],
            "area_px": 50,
            "points": 48,
            "center_xyz": [round(v, 4) for v in self.bottle],
            "top_z": round(self.bottle[2] + 0.085, 4),
            "height_m": 0.17,
            "length_m": 0.041,
            "width_m": 0.039,
            "long_axis_deg": 0.0,
            "grasp_yaw": 90.0,
        }

    def _do_move_pose(
        self,
        xyz: list[float],
        arm: str = "right",
        gripper: float = 0,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
        path: str = "auto",
    ) -> dict[str, Any]:
        target = [float(v) for v in xyz]
        state = self.arms[arm]
        start, start_angles = list(state["tcp"]), list(state["angles"])
        goal = [
            a if v is None else float(v)
            for a, v in zip(start_angles, (pitch, yaw, roll), strict=True)
        ]
        out: dict[str, Any] = {
            "arm": arm,
            "target_xyz": target,
            "target_angles": dict(zip(("pitch", "yaw", "roll"), goal, strict=True)),
            "start_xyz": start,
            "ik_error": None,
            "blocked": False,
        }
        if target[2] < self.REACH_Z:
            out["ik_error"] = "no IK solution within 2 mm / 1 deg (best 3.4 cm off)"
            n = 0
        else:
            turn = max(abs(g - a) for g, a in zip(goal, start_angles, strict=True))
            n = math.ceil(max(math.dist(start, target) / 0.025, turn / 8.0))
        done = 0
        while done < n and self._tick():
            done += 1
            f = done / n
            state["tcp"] = [a + (b - a) * f for a, b in zip(start, target, strict=True)]
            state["angles"] = [a + (b - a) * f for a, b in zip(start_angles, goal, strict=True)]
            self._gripper(arm, gripper)
            if self.holding == arm:
                self.bottle = list(state["tcp"])
        left = math.dist(state["tcp"], target)
        error = max(abs(g - a) for g, a in zip(goal, state["angles"], strict=True))
        out.update(
            final_xyz=[round(v, 4) for v in state["tcp"]],
            final_angles=self._angles(arm),
            dist_to_target_m=round(left, 4),
            orientation_error_deg=round(error, 2),
            reached=out["ik_error"] is None and left <= 0.01 and error <= 5.0,
            steps_used=done,
            path=None if out["ik_error"] else ("joint" if path == "joint" else "line"),
        )
        return out

    def _do_grip(self, arm: str, gripper: float, opening: float | None = None) -> dict[str, Any]:
        state = self.arms[arm]
        before = state["grip"]
        steps = self._tick(3)
        if gripper < 0 and opening is not None:
            state["grip"] = float(opening)
            if self.holding == arm:
                self._release(arm)
        elif gripper > 0:
            if self.holding is None and math.dist(state["tcp"], self.bottle) < 0.03:
                self.holding = arm
            state["grip"] = 0.4 if self.holding == arm else 0.0
        else:
            state["grip"] = 1.0
            if self.holding == arm:
                self._release(arm)
        holding = gripper > 0 and self.holding == arm
        return {
            "arm": arm,
            "gripper": "closed" if gripper > 0 else "open",
            "opening": state["grip"],
            "opening_before": before,
            "holding": holding,
            "width_m": round(state["grip"] * 0.088, 4) if holding else None,
            "steps_used": steps,
            "status": {"step": self.step_count, "step_limit": self.STEP_LIMIT},
        }

    def _do_home(self, arm: str = "both") -> dict[str, Any]:
        """Half of the way back per native action (joint-space in the real tool)."""
        if arm not in ("left", "right", "both"):
            return {"error": "arm must be 'left', 'right' or 'both'"}
        names = ("left", "right") if arm == "both" else (arm,)

        def error() -> float:
            return max(math.dist(self.arms[n]["tcp"], self.start[n][0]) for n in names)

        steps_used = 0
        while error() > 0.01 and steps_used < 25 and self._tick():
            steps_used += 1
            for n in names:
                (pos, angles), (start, start_angles) = (
                    (self.arms[n]["tcp"], self.arms[n]["angles"]),
                    self.start[n],
                )
                if math.dist(pos, start) <= 0.05:
                    self.arms[n]["tcp"], self.arms[n]["angles"] = list(start), list(start_angles)
                else:
                    self.arms[n]["tcp"] = [(a + b) / 2 for a, b in zip(pos, start, strict=True)]
                    self.arms[n]["angles"] = [
                        (a + b) / 2 for a, b in zip(angles, start_angles, strict=True)
                    ]
        return {
            "arm": arm,
            "reached": error() <= 0.01,
            "max_joint_error_rad": round(error(), 4),
            "steps_used": steps_used,
        }


@pytest.fixture
def rd_backend() -> FakeRobodojoBackend:
    return FakeRobodojoBackend()


@pytest.fixture
def rd(rd_backend) -> RobodojoRobot:
    return RobodojoRobot(rd_backend)


# -- state ---------------------------------------------------------------------


def test_state_reads_both_arms_and_never_reports_success(rd, rd_backend):
    state = rd.state()
    assert isinstance(state, State) and isinstance(state.left, ArmState)
    assert state.step == 0 and state.task == rd_backend.TASK and rd.task == rd_backend.TASK
    assert state.left.tcp_pos == pytest.approx((-0.30, -0.20, 1.00))
    assert state.left.eef_pos == pytest.approx((-0.30, -0.345, 1.00))  # the flange, behind it
    assert state.left.eef_quat == pytest.approx(
        (0.7071, 0.0, 0.0, 0.7071), abs=1e-4
    )  # [w, x, y, z]
    assert (state.left.pitch, state.left.yaw, state.left.roll) == (0.0, 0.0, 0.0)
    assert state.right.gripper == 1.0 and len(state.right.joints) == 6
    assert state.arm("right") is state.right
    assert not state.truncated and not hasattr(state, "terminated")
    with pytest.raises(ValueError):
        state.arm("middle")


def test_gripper_angles_follow_the_documented_convention():
    from pyrualean.robodojo import gripper_angles

    for pitch, yaw, roll in ((0, 0, 0), (30, -40, 25), (-20, 170, -90), (60, 90, 10)):
        assert gripper_angles(_quat(pitch, yaw, roll)) == pytest.approx(
            (pitch, yaw, roll), abs=1e-6
        )
    # straight down, yaw and roll turn about the same axis (roll the other way): roll reads 0
    assert gripper_angles(_quat(90, 30, 0)) == pytest.approx((90, 30, 0), abs=1e-6)
    assert gripper_angles(_quat(90, 10, 20)) == pytest.approx((90, -10, 0), abs=1e-6)
    # within a degree of vertical too: no meaningless yaw / roll pair of -170 / -170
    assert gripper_angles(_quat(89.6, 30, 10)) == pytest.approx((89.6, 20, 0), abs=0.01)


def test_there_is_no_done_and_the_backend_is_never_asked(rd, rd_backend):
    rd_backend.in_bin = True  # the environment's own predicate is satisfied
    with pytest.raises(AttributeError, match="there is no robo.done"):
        _ = rd.done
    assert not hasattr(rd, "done")
    assert rd._reported_success() is False  # what the hosts ask instead
    rd.move_to([0.30, -0.20, 0.95])  # motion is still possible: nothing latched a success
    assert rd._reported_success() is False and rd_backend.solved_calls == 0
    assert rd_backend.solved() is True  # the host's end-of-episode read still works


def test_absent_members_are_declared_by_robodojo_only():
    assert RobodojoRobot.ABSENT_MEMBERS == {"done", "world_map"}
    members = _public_members(RobodojoRobot)
    assert "done" not in members and "world_map" not in members
    assert {"task", "state", "image", "show", "artifact"} <= set(members)
    for name in ("libero", "robocasa", "robotwin"):
        other = get_adapter(name).robot_cls
        assert other.ABSENT_MEMBERS == frozenset()
        assert {"done", "world_map"} <= set(_public_members(other))


def test_live_envelopes_carry_numpy_arrays(rd_backend):
    """RPent's in-memory record may hold poses as numpy arrays before it is serialised."""
    np = pytest.importorskip("numpy")
    original_view = rd_backend._view

    def numpy_view(step):
        envelope = original_view(step)
        raw = dict(envelope["state"]["state"])
        for key, value in list(raw.items()):
            raw[key] = np.asarray(value, dtype=np.float64)
        envelope["state"] = dict(envelope["state"], state=raw)
        return envelope

    rd_backend._view = numpy_view
    state = RobodojoRobot(rd_backend).state()
    assert state.left.tcp_pos == pytest.approx((-0.30, -0.20, 1.00)) and state.left.gripper == 1.0


def test_images_camera_meta_and_no_world_map(rd):
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    image = rd.image("cam_left_wrist")
    assert image.shape == (ROWS, COLS, 3) and image.dtype == np.uint8
    assert rd.image().shape == (ROWS, COLS, 3)  # cam_head by default
    meta = rd.camera_meta("cam_head")
    assert meta["shape"] == [ROWS, COLS, 3] and meta["has_depth"] is True
    assert len(meta["intrinsic_matrix"]) == 3 and len(meta["extrinsic_matrix"]) == 4
    with pytest.raises(ValueError, match="camera"):
        rd.image("head")
    with pytest.raises(ValueError, match="camera"):
        rd.camera_meta("wrist")
    with pytest.raises(ValueError, match="back_project"):
        rd.world_map("cam_head")
    with pytest.raises(ToolError, match="not available"):
        rd.state(step=99)


# -- perception ----------------------------------------------------------------


def test_segment_then_back_project_localises_the_object(rd, rd_backend):
    seg = rd.segment("the green bottle")
    assert isinstance(seg, Segment) and seg.found and seg.camera == "cam_head"
    assert seg.centroid == (200, 300) and seg.area_px == 50 and seg.score == 0.83
    assert seg.box == [290.0, 180.0, 310.0, 220.0]
    point = rd.back_project(*seg.centroid, seg.camera)
    assert isinstance(point, WorldPoint) and point.pixel == (200, 300)
    assert point.xyz == tuple(rd_backend.BOTTLE) and point.depth_m == 0.6123
    assert rd_backend.calls[-2] == (
        "segment",
        {"text_prompt": "the green bottle", "camera": "cam_head", "min_score": 0.2},
    )
    assert rd_backend.calls[-1] == ("back_project", {"row": 200, "col": 300, "camera": "cam_head"})
    missing = rd.segment("the red mug")
    assert missing == Segment(False, None, 0, None, None, "cam_head")
    assert not rd.segment("the green bottle", min_score=0.95).found


def test_locate_measures_the_object_without_moving(rd, rd_backend):
    obj = rd.locate("the green bottle")
    assert isinstance(obj, ObjectPose) and obj.found and obj.camera == "cam_head"
    assert obj.center_xyz == pytest.approx(tuple(rd_backend.BOTTLE))
    assert obj.top_z == pytest.approx(0.935) and obj.height_m == 0.17
    assert obj.grasp_yaw == 90.0 and obj.long_axis_deg == 0.0 and obj.width_m == 0.039
    assert obj.centroid == (200, 300) and obj.area_px == 50 and obj.note is None
    assert rd_backend.calls[-1] == (
        "locate",
        {"text_prompt": "the green bottle", "camera": "cam_head", "min_score": 0.2},
    )
    assert rd.state().step == 0 and rd_backend.step_count == 0
    missing = rd.locate("the red mug", camera="cam_left_wrist")
    assert not missing.found and missing.center_xyz is None and missing.top_z is None
    assert missing.camera == "cam_left_wrist"
    with pytest.raises(ToolError, match="sam3_client not configured"):
        rd.locate("the green bottle", camera="cam_right_wrist")


def test_perception_errors_and_argument_checks(rd, rd_backend):
    with pytest.raises(ToolError, match="sam3_client not configured"):
        rd.segment("the green bottle", camera="cam_right_wrist")
    with pytest.raises(ToolError, match="pixel out of range"):
        rd.back_project(999, 5)
    with pytest.raises(ToolError, match="invalid depth"):
        rd.back_project(3, 5)
    before = len(rd_backend.calls)
    with pytest.raises(ValueError, match="camera"):
        rd.segment("the bottle", camera="agentview")
    with pytest.raises(ValueError, match="text_prompt"):
        rd.segment("  ")
    with pytest.raises(ValueError, match="text_prompt"):
        rd.locate("")
    with pytest.raises(ValueError, match="camera"):
        rd.locate("the bottle", camera="head")
    with pytest.raises(ValueError, match="camera"):
        rd.back_project(1, 1, camera="head")
    assert len(rd_backend.calls) == before  # nothing invalid reached the backend
    assert rd.state().step == 0  # read-only tools record no step


# -- motion --------------------------------------------------------------------


def test_move_to_drives_the_grasp_point_and_sends_only_the_angles_given(rd, rd_backend):
    move = rd.move_to([0.30, -0.20, 0.90])
    assert isinstance(move, Move) and move.arm == "right" and move.path == "line"
    assert move.target_xyz == (0.30, -0.20, 0.90)
    assert move.start_xyz == pytest.approx((0.30, -0.20, 1.00))
    assert move.final_xyz == pytest.approx((0.30, -0.20, 0.90))
    assert move.reached and move.dist_to_target_m <= 0.01 and move.steps_used == 4
    assert move.orientation_error_deg == 0.0 and (move.pitch, move.yaw, move.roll) == (0, 0, 0)
    assert move.ik_error is None and not move.blocked and not move.truncated
    assert rd_backend.calls[-1] == (
        "move_pose",
        {"xyz": [0.30, -0.20, 0.90], "arm": "right", "gripper": 0, "path": "auto"},
    )
    state = rd.state()
    assert state.step == 1 and state.right.tcp_pos == pytest.approx((0.30, -0.20, 0.90))
    down = rd.move_to([0.10, -0.10, 1.00], pitch=90, yaw=30)
    assert rd_backend.calls[-1][1] == {
        "xyz": [0.10, -0.10, 1.00],
        "arm": "right",
        "gripper": 0,
        "path": "auto",
        "pitch": 90.0,
        "yaw": 30.0,
    }
    assert down.reached and (down.pitch, down.yaw) == (90.0, 30.0)
    pointed = rd.state().right
    assert (pointed.pitch, pointed.yaw, pointed.roll) == pytest.approx((90, 30, 0), abs=0.01)
    assert pointed.tcp_pos == pytest.approx((0.10, -0.10, 1.00))
    assert pointed.eef_pos == pytest.approx((0.10, -0.10, 1.145))  # the flange is above it
    left = rd.move_to((-0.30, -0.20, 0.95), arm=rd.LEFT, gripper=rd.CLOSE, path="joint")
    sent = rd_backend.calls[-1][1]
    assert sent["gripper"] == 1 and sent["path"] == "joint" and "pitch" not in sent
    assert left.arm == "left" and left.path == "joint" and rd.state().left.gripper == 0.0


def test_move_to_out_of_reach_is_a_result_and_spends_nothing(rd, rd_backend):
    move = rd.move_to([0.30, -0.20, 0.70], gripper=rd.KEEP)  # below what the IK can reach
    assert not move.reached and move.steps_used == 0 and move.path is None
    assert move.final_xyz == pytest.approx((0.30, -0.20, 1.00))
    assert move.dist_to_target_m == pytest.approx(0.30)
    assert move.ik_error.startswith("no IK solution") and rd_backend.step_count == 0


def test_motion_argument_checks_reach_nothing(rd, rd_backend):
    before = len(rd_backend.calls)
    with pytest.raises(ValueError, match="xyz"):
        rd.move_to([0.1, 0.2])
    with pytest.raises(ValueError, match="xyz"):
        rd.move_to([0.1, 0.2, math.nan])
    with pytest.raises(ValueError, match="arm"):
        rd.move_to([0.1, 0.2, 0.9], arm="middle")
    with pytest.raises(ValueError, match="gripper"):
        rd.move_to([0.1, 0.2, 0.9], gripper=0.5)
    with pytest.raises(ValueError, match="pitch"):
        rd.move_to([0.1, 0.2, 0.9], pitch="down")
    with pytest.raises(ValueError, match="yaw"):
        rd.move_to([0.1, 0.2, 0.9], yaw=math.inf)
    with pytest.raises(ValueError, match="path"):
        rd.move_to([0.1, 0.2, 0.9], path="curve")
    with pytest.raises(ValueError, match="gripper"):
        rd.set_gripper("left", rd.KEEP)  # set_gripper has no "keep"
    with pytest.raises(ValueError, match="max_chunks"):
        rd.pi0_pick(max_chunks=0)
    with pytest.raises(ValueError, match="template"):
        rd.pi0_pick("pick the <target>")
    with pytest.raises(ValueError, match="prompt"):
        rd.pi0_pick("  ")
    assert len(rd_backend.calls) == before


def test_home_returns_the_named_arm_and_keeps_its_gripper(rd, rd_backend):
    start = rd.state()
    rd.move_to([-0.10, 0.05, 0.85], arm="left", gripper=rd.CLOSE, pitch=60)
    rd.move_to([0.20, 0.00, 0.90], arm="right")
    away = rd.state()
    before = rd_backend.step_count
    back = rd.home("left")
    assert isinstance(back, Home) and back.arm == "left" and back.reached is True
    assert back.max_joint_error_rad <= 0.01 and back.truncated is False
    assert back.steps_used == rd_backend.step_count - before and 1 <= back.steps_used <= 25
    state = rd.state()
    assert state.left.tcp_pos == pytest.approx(start.left.tcp_pos) and state.left.pitch == 0.0
    assert state.left.gripper == 0.0
    assert state.right.tcp_pos == away.right.tcp_pos  # the arm that was not named stays
    assert rd_backend.calls[-2] == ("home", {"arm": "left"})
    both = rd.home()
    assert both.arm == "both" and both.reached is True
    assert rd.state().right.tcp_pos == pytest.approx(start.right.tcp_pos)
    assert rd.home().steps_used == 0  # already there: nothing is spent
    record = [r for r in rd._ledger.records if r.tool == "home"][0]
    assert record.stateful is True and record.env_steps == back.steps_used
    assert record.summary == {
        "reached": True,
        "steps_used": back.steps_used,
        "max_joint_error_rad": back.max_joint_error_rad,
    }


def test_home_checks_its_argument_and_reports_a_return_the_budget_cut_short(rd, rd_backend):
    calls = len(rd_backend.calls)
    for bad in ("up", "", None, 1):
        with pytest.raises(ValueError):
            rd.home(bad)
    assert len(rd_backend.calls) == calls
    rd.move_to([-0.10, 0.05, 0.85], arm="left")
    rd_backend.step_count = rd_backend.STEP_LIMIT - 1
    cut = rd.home("left")
    assert cut.reached is False and cut.steps_used == 1 and cut.truncated is True
    assert cut.max_joint_error_rad > 0.01


def test_set_gripper_runs_a_full_stroke_and_says_whether_it_holds(rd, rd_backend):
    closed = rd.set_gripper("left", rd.CLOSE)
    assert isinstance(closed, Gripper) and closed.commanded == "closed"
    assert closed.opening == 0.0 and not closed.holding and closed.width_m is None
    assert closed.steps_used == 3 and closed.step == 3 and closed.step_limit == 120
    assert rd_backend.calls[-1] == ("grip", {"arm": "left", "gripper": 1})
    opened = rd.set_gripper("left", rd.OPEN)
    assert opened.commanded == "open" and opened.opening == 1.0 and opened.step == 6
    assert not opened.holding and not opened.truncated
    narrow = rd.set_gripper("right", rd.OPEN, opening=0.4)
    assert narrow.opening == 0.4 and rd_backend.calls[-1] == (
        "grip",
        {"arm": "right", "gripper": -1, "opening": 0.4},
    )
    calls = len(rd_backend.calls)
    for bad in ((rd.CLOSE, 0.5), (rd.OPEN, 0.0), (rd.OPEN, 1.5), (rd.OPEN, "wide")):
        with pytest.raises(ValueError, match="opening"):
            rd.set_gripper("right", bad[0], opening=bad[1])
    assert len(rd_backend.calls) == calls


# -- the VLA -------------------------------------------------------------------


def test_pi0_pick_reports_both_arms_and_a_provisional_success(rd, rd_backend):
    pick = rd.pi0_pick()
    assert isinstance(pick, Pick) and pick.success and pick.chunks_used == 1
    assert pick.instruction == rd_backend.TASK and pick.monitored_arm == "right"
    assert pick.left["peak_lift_m"] == 0.07 and pick.left["final_gripper"] == 0.2
    assert pick.right["peak_lift_m"] == 0.0 and set(pick.left) == set(pick.right)
    assert not pick.budget_exhausted and not pick.truncated
    assert rd_backend.calls[-1] == ("pi0_pick", {"arm": "right", "max_chunks": 8})
    again = rd.pi0_pick("pick the green bottle next to the bin", arm="left", max_chunks=2)
    assert again.instruction == "pick the green bottle next to the bin"
    assert rd_backend.calls[-1][1] == {
        "arm": "left",
        "max_chunks": 2,
        "prompt": "pick the green bottle next to the bin",
    }
    assert rd._reported_success() is False  # a grasp is not the task


def test_pi0_pick_without_a_policy_service_is_a_tool_error(rd, rd_backend):
    rd_backend.vla = False
    with pytest.raises(ToolError, match="vla_client not configured"):
        rd.pi0_pick()
    assert rd.state().step == 1  # RPent records a step for the failed call


def test_no_vla_set_refuses_the_pick(rd_backend):
    robo = RobodojoRobot(rd_backend, primitives="no-vla")
    with pytest.raises(RuntimeError, match="pi0_pick is not available"):
        robo.pi0_pick()
    assert rd_backend.calls == []
    assert robo.move_to([0.30, -0.20, 0.95]).reached


# -- a scripted top grasp, budget, cancellation ----------------------------------


def test_locate_grasp_carry_and_release_over_the_bin(rd, rd_backend):
    obj = rd.locate("the green bottle")
    x, y, z = obj.center_xyz
    assert rd.set_gripper("left", rd.OPEN).opening == 1.0
    above = rd.move_to([x, y, obj.top_z + 0.08], arm="left", pitch=90, yaw=obj.grasp_yaw)
    assert above.reached and above.pitch == 90.0
    assert rd.move_to([x, y, z], arm="left", gripper=rd.OPEN, path="line").reached
    held = rd.set_gripper("left", rd.CLOSE)
    assert held.holding and held.opening == 0.4 and held.width_m == pytest.approx(0.0352)
    rd.move_to([x, y, 1.00], arm="left", gripper=rd.CLOSE)
    assert rd.locate("the green bottle").center_xyz == pytest.approx((x, y, 1.00))
    bx, by, _ = rd_backend.BIN
    assert rd.move_to([bx, by, 0.88], arm="left", gripper=rd.CLOSE).reached
    assert rd_backend.holding == "left" and not rd_backend.in_bin
    let_go = rd.set_gripper("left", rd.OPEN)
    assert not let_go.holding and let_go.opening == 1.0
    # The environment is solved, and the robot still does not say so.
    assert rd_backend.in_bin and rd._reported_success() is False
    assert rd.home().reached and not rd.state().truncated


def test_budget_truncates_the_episode_without_a_success(rd, rd_backend):
    first = rd.pi0_pick(max_chunks=1)
    second = rd.pi0_pick(max_chunks=1)
    assert first.success and second.success and rd_backend.step_count == 100
    last = rd.pi0_pick(max_chunks=1)  # only 20 native actions left
    assert not last.success and last.budget_exhausted and last.truncated
    assert rd.state().truncated and rd._reported_success() is False
    with pytest.raises(EpisodeFinished) as stop:
        rd.move_to([0.30, -0.20, 0.95])
    assert stop.value.reason == "truncated"
    assert rd.segment("the green bottle").found  # looking is still allowed
    assert rd.locate("the green bottle").found


def test_cancelled_motion_ends_the_episode(rd, rd_backend):
    rd._stop_episode("timeout")
    assert rd_backend.cancelled
    with pytest.raises(EpisodeFinished) as stop:
        rd.move_to([0.30, -0.20, 0.95])
    assert stop.value.reason == "timeout"
    fresh_backend = FakeRobodojoBackend()
    fresh = RobodojoRobot(fresh_backend)
    fresh_backend.cancelled = True  # RPent interrupts the tool in flight
    with pytest.raises(EpisodeFinished) as stop:
        fresh.set_gripper("left", fresh.CLOSE)
    assert stop.value.reason == "cancelled"


def test_ledger_marks_stateful_calls_and_counts_native_actions(rd, rd_backend):
    rd.segment("the green bottle")
    rd.locate("the green bottle")
    rd.move_to([0.30, -0.20, 0.90])
    rd.pi0_pick(max_chunks=1)
    records = [r for r in rd._ledger.records if r.tool != "view_env_state"]
    assert [r.tool for r in records] == ["segment", "locate", "move_pose", "pi0_pick"]
    assert [r.stateful for r in records] == [False, False, True, True]
    assert records[0].summary == {"found": True, "score": 0.83, "centroid_rc": [200, 300]}
    assert records[1].summary["center_xyz"] == rd_backend.BOTTLE
    assert records[1].summary["grasp_yaw"] == 90.0
    assert records[2].env_steps == 4 and records[3].env_steps == 50
    assert records[2].summary["reached"] is True and records[3].summary["success"] is True
    assert rd._ledger.env_steps() == rd_backend.step_count == 54
    assert set(RobodojoRobot.STATEFUL_TOOLS) == {
        "move_pose",
        "grip",
        "home",
        "pi0_pick",
    }


# -- prompt, adapter, arm toolkit ------------------------------------------------

VLA_TOKENS = re.compile(r"pi0|lingbot|rldx|vla", re.IGNORECASE)


def test_api_reference_lists_every_public_method_and_result_type():
    text = api_reference(RobodojoRobot)
    for name in _public_members(RobodojoRobot):
        assert f"robo.{name}" in text, name
    # absent members and the success vocabulary are not part of the reference
    assert "robo.done" not in text and "robo.world_map" not in text
    assert "terminated: bool" not in text and "ABSENT_MEMBERS" not in text
    for name in (
        "segment",
        "back_project",
        "locate",
        "pi0_pick",
        "move_to",
        "home",
        "set_gripper",
        "camera_meta",
    ):
        assert f"robo.{name}(" in text, name
    for gone in ("stabilize", "place_in_bin", "BinPlace", "rotate_wrist", "robo.press("):
        assert gone not in text, gone
    for cls in RobodojoRobot.RESULT_TYPES:
        assert f"class {cls.__name__}(" in text
    assert "robo.LEFT = 'left'" in text and "robo.RIGHT = 'right'" in text
    assert "robo.CLOSE = 1" in text and "robo.OPEN = -1" in text and "robo.KEEP = 0" in text
    assert "RESULT_TYPES" not in text and "SUMMARY_KEYS" not in text and "CAMERAS" not in text
    assert "    pitch: float | None = None,\n    yaw: float | None = None,\n" in text
    assert "    path: str = 'auto',\n) -> Move" in text
    assert "robo.home(arm: str = 'both') -> Home" in text
    assert (
        "robo.pi0_pick(\n    prompt: str | None = None,\n    arm: str = 'right',\n"
        "    max_chunks: int = 8,\n) -> Pick"
    ) in text
    assert "min_score: float = 0.2" in text
    novla = api_reference(RobodojoRobot, primitives="no-vla")
    assert not VLA_TOKENS.search(novla), sorted(set(VLA_TOKENS.findall(novla)))
    assert "class Pick(" not in novla and "robo.move_to(" in novla and "class Move(" in novla
    assert "robo.home(" in novla and "class Home(" in novla  # the return is no VLA primitive
    assert "robo.locate(" in novla and "robo.set_gripper(" in novla


def test_adapter_and_prompt_say_that_nothing_reports_success():
    adapter = get_adapter("robodojo")
    assert adapter.robot_cls is RobodojoRobot and adapter.knowledge == "robodojo"
    assert adapter.cameras == ("cam_head", "cam_left_wrist", "cam_right_wrist")
    assert adapter.host == "pyrualean.hosts.rpent_robodojo"
    assert adapter.guides_subdir == "" and adapter.guide_names == ()
    assert adapter.live_success is False
    assert all(get_adapter(name).live_success for name in ("libero", "robocasa", "robotwin"))
    assert robot_words(adapter)["scoring"] == "final"
    assert robot_words(get_adapter("robotwin"))["scoring"] == "live"
    card = TaskCard(
        robot="robodojo",
        suite="arx_x5",
        task="put_bottles_into_dustbin",
        seed=0,
        task_language="Put the bottles into the dustbin.",
        object_names=[],
        facts={"left gripper opening": "1.00 (1 = open, 0 = closed)"},
        max_env_steps=700,
    )
    bundle = render_prompt(card, budget_s=600)
    assert "a dual-arm ARX-X5 robot in the RoboDojo benchmark" in bundle.system
    assert "class RobodojoRobot" in bundle.system
    assert "# Operating knowledge for RoboDojo" in bundle.system
    assert "judged by the benchmark from the scene `run` leaves behind" in bundle.system
    assert "task predicate fires" not in bundle.system and "robo.done" not in bundle.system
    assert "- robot: robodojo" in bundle.user and "left gripper opening: 1.00" in bundle.user
    know = knowledge_text("robodojo")
    for banned in ("MEMORY.md", "task_only", "recipe.jsonl", "audit", "server", "launch"):
        assert banned not in know, banned
    for arm in ("cells", "program"):
        text = build_prompt(
            arm, card, budget_s=600, max_programs=None, images="on-demand", feedback="pure"
        )
        assert "Success is judged by the benchmark from the final scene" in text
        assert "no result carries a" in text and "task predicate fires" not in text
        assert "robo.done" not in text and "`terminated`" not in text
        assert "Once the task terminates" not in text
        assert "or 700 simulator steps; the scene is judged as it is" in text
        assert "counts as a failure" not in text  # a limit ends the episode, the scene decides
    calls = build_prompt(
        "cells",
        card,
        budget_s=600,
        max_programs=None,
        images="on-demand",
        feedback="pure",
        budget_mode="calls",
    )
    assert "At either limit the scene is judged\n  as it is" in calls
    assert "counts as a" not in calls and "counts as a failure" not in bundle.system
    assert "of 600 s or by the simulator step budget (700 steps)" in bundle.system
    assert "reason: truncated | timeout | cancelled" in bundle.system
    cells = build_prompt(
        "cells", card, budget_s=600, max_programs=None, images="on-demand", feedback="pure"
    )
    assert "`RobodojoRobot` bound to the live simulator, a\nfrozen Pi0.5 VLA and a SAM3" in cells
    assert "`robo.show('cam_head')`, `robo.show('cam_left_wrist')`" in cells
    novla = build_prompt(
        "cells",
        card,
        budget_s=600,
        max_programs=None,
        images="on-demand",
        feedback="pure",
        primitives="no-vla",
    )
    assert not VLA_TOKENS.search(novla), sorted(set(VLA_TOKENS.findall(novla)))
    assert "live simulator, a\nSAM3 segmentation service" in novla


def test_live_scoring_contract_is_the_default_wording():
    """The other robots' contract keeps its scoring bullets word for word."""
    kwargs = dict(budget_s=600, max_env_steps=10000, max_programs=None)
    cells = arm_contract("cells", **kwargs)
    assert (
        "- Success = the benchmark's own task predicate fires (`terminated`) at any\n"
        "  moment.  The simulator checks it, not you; motion results carry a\n"
        "  `terminated` field and `robo.done` reports it.\n"
        "- The episode ends after 600 s"
    ) in cells
    assert "- Once the task terminates, motion calls raise `EpisodeFinished`: call\n" in cells
    program = arm_contract("program", **kwargs)
    assert "moment while a program runs.  The simulator checks it, not you.\n- The epi" in program
    assert "simulator steps; that counts as a failure." in cells
    final = arm_contract("cells", robot=robot_words(get_adapter("robodojo")), **kwargs)
    assert final != cells and "Success is judged by the benchmark" in final
    libero = render_prompt(budget_s=600).system
    assert "- The episode counts as a failure when `run` raises an uncaught exception,\n" in libero
    assert "reason: terminated | truncated | timeout | cancelled" in libero


def test_arm_toolkit_attaches_the_three_views_and_never_terminates(rd_backend, tmp_path):
    pytest.importorskip("PIL")
    robo = RobodojoRobot(rd_backend)
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", images="on-motion")
    moved = kit.execute_tool("python", {"code": "robo.move_to([0.30, -0.20, 0.90])"})
    assert moved.result["moved"] is True and moved.result["env_steps"] == 4
    assert moved.result["images"] == ["cam_head.png", "cam_left_wrist.png", "cam_right_wrist.png"]
    assert moved.result["state"]["right_tcp"] == pytest.approx([0.30, -0.20, 0.90])
    assert moved.result["state"]["right_pitch_yaw_roll"] == [0.0, 0.0, 0.0]
    assert moved.result["state"]["truncated"] is False
    assert "episode_done" not in moved.result  # no success field at all, not a constant False
    rd_backend.in_bin = True
    after = kit.execute_tool("python", {"code": "print(robo.done)"})
    assert "AttributeError" in after.result["exception"]
    assert "there is no robo.done" in after.result["exception"]
    assert "episode_done" not in after.result and after.result["moved"] is False
    finished = kit.execute_tool("finish", {"status": "success", "summary": "bottle is in the bin"})
    assert finished.result == {"ok": True, "status": "success", "summary": "bottle is in the bin"}
    assert "environment_terminated" not in kit.finish and kit.turns[-1]["terminated"] is False
    demand = ArmToolkit(
        RobodojoRobot(rd_backend),
        arm="cells",
        workspace=tmp_path / "ws2",
        images="on-demand",
        feedback="pure",
    )
    looked = demand.execute_tool(
        "python", {"code": "robo.show('cam_left_wrist'); print(robo.task)"}
    )
    assert looked.result["images"] == ["cam_left_wrist.png"] and len(looked.images) == 1
    assert looked.result["stdout"].strip() == rd_backend.TASK and "state" not in looked.result


# -- host --------------------------------------------------------------------------


class _FakeState:
    """Enough of RPent's ``EnvState`` for ``dump_card``: artifact paths."""

    def __init__(self, backend: FakeRobodojoBackend, root: Path) -> None:
        self.backend = backend
        self.root = root

    def artifact_path(self, name: str, *, step: int = -1) -> Path:
        path = self.root / name / f"{step:02d}{Path(name).suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.backend.artifact(name, step))
        return path


class _FakeEnv:
    def __init__(self, backend: FakeRobodojoBackend) -> None:
        self.backend = backend

    def get_status(self) -> dict[str, Any]:
        # RPent's development status also carries the success flag; the host ignores it.
        return {
            "step": self.backend.step_count,
            "step_limit": self.backend.STEP_LIMIT,
            "success": self.backend.in_bin,
        }


class _FakeToolkit:
    def __init__(self, backend: FakeRobodojoBackend, root: Path) -> None:
        self.backend = backend
        self.state = _FakeState(backend, root)
        self._primitives = type("P", (), {"env": _FakeEnv(backend)})()

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        return self.backend.execute_tool(name, input_dict)

    def solved(self) -> bool:
        return self.backend.in_bin


def test_cell_and_argument_helpers():
    cell = host.Cell("arx_x5", "put_bottles_into_dustbin", 3, cuda_device=1)
    assert cell.tag == "put_bottles_into_dustbin_l3" and cell.max_episode_steps == 700
    args = Namespace(task="stack_bowls_random", suite=None, seed=2, max_episode_steps=500)
    assert host.cell_from_args(args) == host.Cell("arx_x5", "stack_bowls_random", 2, 500, None)
    options = host.boot_options(
        Namespace(vla_endpoint="127.0.0.1:9000", sim_python="/sim/python", policy_backend=None)
    )
    assert options == {
        "vla_endpoint": "127.0.0.1:9000",
        "sam3_endpoint": None,
        "env_endpoint": None,
        "source_root": None,
        "xpolicylab_root": None,
        "sim_python": "/sim/python",
        "pi05_python": None,
        "policy_backend": "rlinf",
        "action_type": "joint",
        "connect_reset": None,
    }
    import argparse

    parser = argparse.ArgumentParser()
    host.add_cell_args(parser)
    host.add_boot_args(parser)
    ns = parser.parse_args(["--task", "fill_pen_holder", "--seed", "4"])
    assert ns.suite == "arx_x5" and host.cell_from_args(ns).seed == 4
    assert ns.action_type == "joint" and ns.policy_backend in host.POLICY_BACKENDS
    assert ns.connect_reset is False  # RPent's second reset is the exception, not the rule
    assert parser.parse_args(["--task", "make_toast", "--connect-reset"]).connect_reset is True
    assert parser.parse_args(["--task", "make_toast", "--no-connect-reset"]).connect_reset is False
    import inspect

    # every boot option is a keyword of boot()
    assert set(host.boot_options(ns)) <= set(inspect.signature(host.boot).parameters)
    with pytest.raises(ValueError, match="policy_backend"):
        host.boot(cell, Path("unused"), policy_backend="openvla")
    with pytest.raises(ValueError, match="action_type"):
        host.boot(cell, Path("unused"), action_type="delta")


def test_dump_card_writes_json_images_and_facts(rd_backend, tmp_path):
    pytest.importorskip("PIL")
    toolkit = _FakeToolkit(rd_backend, tmp_path / "artifacts")
    cell = host.Cell("arx_x5", "put_bottles_into_dustbin", 0, cuda_device=1)
    card = host.dump_card(toolkit, cell, tmp_path)
    assert (tmp_path / "card.json").is_file()
    assert set(card.images) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
    assert all(Path(p).is_file() for p in card.images.values())
    assert card.robot == "robodojo" and card.task == "put_bottles_into_dustbin"
    assert card.task_language == rd_backend.TASK and card.object_names == []
    assert card.eef_pos is None and card.eef_quat is None and card.gripper_opening is None
    assert card.max_env_steps == 120  # the simulator's step_lim, not the nominal 700
    facts = card.facts
    assert facts["left arm initial grasp point [x, y, z]"] == "[-0.300, -0.200, 1.000] m"
    assert facts["right gripper initial pitch / yaw / roll"] == "0 / 0 / 0 deg"
    assert facts["left gripper opening"] == "1.00 (1 = open, 0 = closed)"
    assert facts["native action budget"].startswith("120 ")
    text = card.render()
    assert "- robot: robodojo" in text and "- variant: arx_x5" in text
    assert "`robo.image('cam_head')`" in text and "step 0 of your run" in text
    assert "- simulator step budget: 120" in text
    assert not VLA_TOKENS.search(text) and "success" not in text
    assert TaskCard.from_json(tmp_path / "card.json") == card


def test_backend_reads_the_native_counter_and_the_final_verdict(rd_backend, tmp_path):
    toolkit = _FakeToolkit(rd_backend, tmp_path)
    backend = host.make_backend(toolkit)
    assert isinstance(backend, host.RobodojoBackend)
    assert backend.env_steps() == 0 and backend.solved() is False
    rd_backend.step_count, rd_backend.in_bin = 17, True
    assert backend.env_steps() == 17 and backend.solved() is True

    class _Bare:
        def solved(self) -> bool:
            return False

    assert host.RobodojoBackend(_Bare()).env_steps() is None


class _JointEnv:
    """Two arms that cover 60 % of the remaining joint error per action."""

    START = {"left": [0.0, 0.5, 0.5, 0.0, 0.0, 0.0], "right": [0.0, 0.4, 0.6, 0.0, 0.0, 0.0]}

    def __init__(self, step_limit: int = 100) -> None:
        self.joints = {arm: list(values) for arm, values in self.START.items()}
        self.grip = {"left": 0.31, "right": 1.0}
        self.step_count, self.step_limit = 0, step_limit
        self.actions: list[dict[str, list[float]]] = []

    def get_obs(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for arm in ("left", "right"):
            state[f"{arm}_arm_joint_state"] = tuple(self.joints[arm])  # not a list, as live
            state[f"{arm}_ee_joint_state"] = [self.grip[arm]]
        return {"state": state}

    def step(self, action: dict[str, list[float]]):
        self.actions.append(action)
        self.step_count += 1
        for arm in ("left", "right"):
            target = action[f"{arm}_arm_joint_state"]
            self.joints[arm] = [
                now + 0.6 * (goal - now) for now, goal in zip(self.joints[arm], target, strict=True)
            ]
            self.grip[arm] = action[f"{arm}_ee_joint_state"][0]
        status = {"step": self.step_count, "step_limit": self.step_limit}
        return self.get_obs(), 0.0, False, {"status": status}


class _JointPrimitives:
    def __init__(self, env: _JointEnv) -> None:
        self.env = env
        self._last_obs = None
        self.checked = 0

    def _check_cancelled(self) -> None:
        self.checked += 1


def test_home_tool_commands_the_starting_joints_and_holds_everything_else():
    env = _JointEnv()
    primitives = _JointPrimitives(env)
    start = {arm: list(values) for arm, values in env.START.items()}
    assert host.home(primitives, start, arm="left", lift=False) == {
        "arm": "left",
        "reached": True,
        "max_joint_error_rad": 0.0,
        "steps_used": 0,
    }
    env.joints = {"left": [0.4, 0.9, 0.1, 0.0, 0.0, 0.0], "right": [0.3, 0.2, 0.8, 0.0, 0.0, 0.0]}
    result = host.home(primitives, start, arm="left", lift=False)
    assert result["arm"] == "left" and result["reached"] is True
    assert result["steps_used"] == len(env.actions) == primitives.checked == 5
    assert result["max_joint_error_rad"] <= HOME_TOL_RAD
    for action in env.actions:
        assert action["left_arm_joint_state"] == start["left"]
        assert action["left_ee_joint_state"] == [0.31]  # the opening it had: a held object stays
        assert action["right_ee_joint_state"] == [1.0]
    assert env.actions[0]["right_arm_joint_state"] == [0.3, 0.2, 0.8, 0.0, 0.0, 0.0]  # it holds
    assert env.joints["right"] == pytest.approx([0.3, 0.2, 0.8, 0.0, 0.0, 0.0])
    assert primitives._last_obs == env.get_obs()
    both = host.home(primitives, start, lift=False)
    assert both["arm"] == "both" and both["reached"] is True
    assert env.joints["right"] == pytest.approx(start["right"], abs=HOME_TOL_RAD)
    assert "error" in host.home(primitives, start, arm="up")


def test_home_tool_stops_at_its_own_limit_and_at_the_step_limit():
    start = {arm: list(values) for arm, values in _JointEnv.START.items()}
    far = _JointEnv()
    far.joints = {"left": [3.0] * 6, "right": [3.0] * 6}
    slow = host.home(_JointPrimitives(far), start, max_steps=2, lift=False)
    assert slow["reached"] is False and slow["steps_used"] == 2
    assert slow["max_joint_error_rad"] > 0.4
    spent = _JointEnv(step_limit=1)
    spent.joints = {"left": [3.0] * 6, "right": [3.0] * 6}
    cut = host.home(_JointPrimitives(spent), start, lift=False)
    assert cut["reached"] is False and cut["steps_used"] == 1


def test_register_home_reads_the_starting_joints_and_adds_a_recorded_tool():
    env = _JointEnv()
    added: dict[str, Any] = {}

    class _Toolkit:
        _primitives = _JointPrimitives(env)

        def add_tool(self, name: str, spec: dict[str, Any], handler: Any) -> None:
            added[name] = (spec, handler)

    host.register_home(_Toolkit())
    spec, handler = added["home"]
    assert spec is host.HOME_SPEC and spec["name"] == "home"
    assert spec["input_schema"]["properties"]["arm"]["enum"] == ["left", "right", "both"]
    # RPent's toolkit records a step after every handler it does not find marked read-only
    assert not getattr(handler, "_readonly", False) and not getattr(host.home, "_readonly", False)
    env.joints["left"] = [0.4, 0.9, 0.1, 0.0, 0.0, 0.0]  # after registration: it read the start
    assert handler(arm="left")["reached"] is True
    assert env.joints["left"] == pytest.approx(env.START["left"], abs=HOME_TOL_RAD)
    assert handler()["arm"] == "both"


def test_connect_without_reset_patches_rpents_client_and_restores_it(monkeypatch):
    """The second reset is skipped only inside the context, and RPent's class is put back."""

    import sys
    import types

    calls: list[str] = []

    class BaseEnvClient:
        def __init__(self, client, *, expected_meta, reset_on_connect=True):
            self._client = client
            if reset_on_connect:
                calls.append("reset")

    class RoboDojoEnvClient(BaseEnvClient):
        def __init__(self, client, *, expected_meta):
            self.eval_fair = expected_meta.get("mode") == "eval-fair"
            reset = not self.eval_fair
            super().__init__(client, expected_meta=expected_meta, reset_on_connect=reset)

        def get_obs(self):
            calls.append("get_obs")
            return {"vision": {}}

    for name in ("robots", "robots.robodojo", "rpent", "rpent.robots", "rpent.robots.components"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    client_module = types.ModuleType("robots.robodojo.env_client")
    client_module.RoboDojoEnvClient = RoboDojoEnvClient
    base_module = types.ModuleType("rpent.robots.components.env_client_base")
    base_module.BaseEnvClient = BaseEnvClient
    monkeypatch.setitem(sys.modules, "robots.robodojo.env_client", client_module)
    monkeypatch.setitem(sys.modules, "rpent.robots.components.env_client_base", base_module)
    monkeypatch.setattr(sys.modules["robots.robodojo"], "env_client", client_module, raising=False)

    original = RoboDojoEnvClient.__init__
    RoboDojoEnvClient(object(), expected_meta={"mode": "dev"})
    assert calls == ["reset"]
    calls.clear()
    with host._connect_without_reset():
        client = RoboDojoEnvClient(object(), expected_meta={"mode": "dev"})
    assert calls == ["get_obs"] and client.last_obs == {"vision": {}} and client.eval_fair is False
    assert RoboDojoEnvClient.__init__ is original

    monkeypatch.setenv("ROBODOJO_CONNECT_RESET", "1")
    assert host._connect_reset_default() is True
    monkeypatch.setenv("ROBODOJO_CONNECT_RESET", "0")
    assert host._connect_reset_default() is False
    monkeypatch.delenv("ROBODOJO_CONNECT_RESET")
    assert host._connect_reset_default() is False


# -- the benchmark's third arm (robodojo_env_server) ----------------------------------


def _fake_bridge(monkeypatch, interact: bool = True):
    """A ``robodojo_runtime.bridge`` and a collect-client module shaped like the real ones."""

    import sys
    import types

    collect = types.ModuleType("fake_collect_env")

    def _expand_control_info(env, control_info, env_idx):
        return [dict(control_info) for _ in range(env.substeps)]

    collect._expand_control_info = _expand_control_info

    class CollectEnv:
        num_envs = 1
        substeps = 3

        def __init__(self):
            self.interact = interact
            self.support_arm_action = [[]]
            self.pushed: list[list[dict]] = []
            self.turns: list[str] = []

        def query_support_arm_traj(self, env_idx):
            self.turns.append(f"query{env_idx}")
            if self.turns.count(f"query{env_idx}") == 1:  # one trajectory, queued once
                self.support_arm_action[env_idx].extend({"support_arm0": i} for i in range(4))

        def check_support_arm_stable(self, env_idx):
            self.turns.append(f"check{env_idx}")

        def apply_target(self, control_info, env_idx=0):
            # the collect client calls the module function, looked up at call time
            self.pushed.append(collect._expand_control_info(self, control_info, env_idx))

    CollectEnv.__module__ = "fake_collect_env"
    monkeypatch.setitem(sys.modules, "fake_collect_env", collect)
    bridge = types.ModuleType("fake_bridge")

    class NativeEnv:
        def __init__(self, task_config):
            self.env = CollectEnv()

        def reset(self, seed=None):
            bridge._standard_reset(self.env, seed)

        def apply_action(self, action, action_type="joint", *, eval_fair=False):
            self.env.apply_target({"left_arm": action}, 0)

    bridge.NativeEnv = NativeEnv
    bridge._standard_reset = lambda env, layout: env.turns.append("reset")
    bridge._random_reset = lambda env: env.turns.append("random reset")
    return bridge, collect


def test_third_arm_is_queued_replayed_and_given_its_turn(monkeypatch):
    bridge, collect = _fake_bridge(monkeypatch)
    support_arm.install(bridge)
    support_arm.install(bridge)  # a second install changes nothing
    native = bridge.NativeEnv({})
    env = native.env
    native.reset(0)
    assert env.turns == ["reset", "query0"]  # queued once the checks are registered
    assert len(env.support_arm_action[0]) == 4
    native.apply_action([0.1], "joint", eval_fair=False)
    first = env.pushed[-1]
    assert [step.get("support_arm0") for step in first] == [0, 1, 2]  # one queued step a substep
    assert all(step["left_arm"] == [0.1] for step in first)  # the robot's own target is kept
    assert env.turns[-2:] == ["query0", "check0"]  # its turn after the action
    native.apply_action([0.2])
    assert [step.get("support_arm0") for step in env.pushed[-1]] == [3, None, None]
    assert env.support_arm_action == [[]]


def test_tasks_without_a_third_arm_run_as_before(monkeypatch):
    bridge, collect = _fake_bridge(monkeypatch, interact=False)
    support_arm.install(bridge)
    native = bridge.NativeEnv({})
    native.reset(0)
    native.apply_action([0.1])
    assert native.env.turns == ["reset"]
    assert native.env.pushed == [[{"left_arm": [0.1]}] * 3]


def test_host_starts_the_env_server_through_this_library(monkeypatch):
    import sys
    import types

    started: list[dict] = []

    def ProcessDaemon(**kwargs):  # noqa: N802 - RPent's class name
        started.append(kwargs)
        return kwargs

    robot_spec = types.ModuleType("robots.robodojo.robot_spec")
    robot_spec.ProcessDaemon = ProcessDaemon
    for name in ("robots", "robots.robodojo"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "robots.robodojo.robot_spec", robot_spec)
    monkeypatch.setattr(sys.modules["robots.robodojo"], "robot_spec", robot_spec, raising=False)
    monkeypatch.setenv("PYTHONPATH", "/rpent")
    script = os.path.join("/r", "robots", "robodojo", "env_server.py")
    with host._env_server_through_pyrualean():
        robot_spec.ProcessDaemon(
            name="robodojo_env_server",
            cmd=["/sim/python", "-u", script, "--task", "make_kong"],
            env_overrides={"A": "1"},
        )
        robot_spec.ProcessDaemon(name="robodojo_vla_server", cmd=["/py", "serve.py"])
    assert robot_spec.ProcessDaemon is ProcessDaemon  # put back
    env_server, vla = started
    assert env_server["cmd"] == [
        "/sim/python",
        "-u",
        "-m",
        "pyrualean.hosts.robodojo_env_server",
        script,
        "--task",
        "make_kong",
    ]
    paths = env_server["env_overrides"]["PYTHONPATH"].split(os.pathsep)
    assert paths[-1] == "/rpent" and Path(paths[0], "pyrualean", "robodojo.py").is_file()
    assert env_server["env_overrides"]["A"] == "1"
    assert vla == {"name": "robodojo_vla_server", "cmd": ["/py", "serve.py"]}  # untouched
    monkeypatch.setenv("ROBODOJO_SUPPORT_ARM", "upstream")
    assert host._support_arm_default() is False
    monkeypatch.delenv("ROBODOJO_SUPPORT_ARM")
    assert host._support_arm_default() is True


def test_env_server_entry_runs_rpents_script_with_its_arguments(monkeypatch, tmp_path):
    import sys

    script = tmp_path / "env_server.py"
    script.write_text("import sys, json\nprint(json.dumps(sys.argv))\n")
    installed: list[bool] = []
    monkeypatch.setattr(support_arm, "install", lambda bridge=None: installed.append(True))
    monkeypatch.setattr(sys, "argv", ["x"])
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    support_arm.main([str(script), "--task", "make_kong"])
    assert installed == [True]
    assert out.getvalue().strip() == f'["{script}", "--task", "make_kong"]'
    with pytest.raises(SystemExit):
        support_arm.main([])
