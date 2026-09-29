"""Behaviour of the typed RoboTwin API over an RPent-shaped fake backend, plus its host."""

from __future__ import annotations

import io
import json
import math
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
from pyrualean.arms import ArmToolkit
from pyrualean.hosts import rpent_robotwin as host
from pyrualean.play import build_prompt
from pyrualean.prompt import _public_members
from pyrualean.robotwin import (
    ArmState,
    Gripper,
    Move,
    RobotwinRobot,
    State,
    VlaRun,
    WorldRegion,
    WorldSamples,
)

ROWS, COLS = 240, 320


class _ToolResult:
    """Shape of ``rpent.tools.toolkit.ToolResult`` (only ``.result`` matters)."""

    def __init__(self, name: str, result: dict[str, Any]) -> None:
        self.name = name
        self.result = result


class FakeRobotwinBackend:
    """Two arms, a hammer and a block.

    Mirrors RPent's RoboTwin envelope (``robots/robotwin/tools.py``): every
    state-changing tool returns a ``view_env_state`` dict whose ``log.result``
    holds the primitive's own return value; ``sample_world_xyz`` /
    ``query_world_map`` return their payload directly.  Success = the VLA
    grasps the hammer (first chunk) and hits the block (next call).
    """

    TASK = "Grab the nail-driving hammer with the left arm and hit the block"
    STEP_LIM = 300
    TABLE_Z = 0.715
    HAMMER_Z = 0.78
    HAMMER_BOX = (100, 140, 140, 180)  # rows/cols of the hammer in the head view

    def __init__(self) -> None:
        self.arms: dict[str, dict[str, Any]] = {
            "left": {"eef": [-0.298, -0.314, 0.942], "quat": [0.7, 0.0, 0.0, 0.714], "grip": 1.0},
            "right": {"eef": [0.306, -0.313, 0.941], "quat": [0.704, 0.0, 0.0, 0.711], "grip": 1.0},
        }
        self.actions = 0
        self.policy_actions = 0
        self.eval_success = False
        self.holding = False
        self.steps: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.cancelled = False
        self.error_next: dict[str, Any] | None = None
        self._record({"action": "reset"}, {"success": True})

    # -- Backend contract ---------------------------------------------------

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        self.calls.append((name, dict(input_dict)))
        if name in ("view_env_state", "sample_world_xyz", "query_world_map"):
            return _ToolResult(name, self._readonly(name, input_dict))
        if self.error_next is not None:
            result, self.error_next = self.error_next, None
        elif self.cancelled:
            result = {
                "error": "tool operation interrupted",
                "code": "tool_cancelled",
                "interrupted": True,
            }
        else:
            handler = getattr(self, f"_do_{name}", None)
            if handler is None:
                return _ToolResult(name, {"error": f"unknown tool: {name}"})
            try:
                result = handler(**input_dict)
            except TypeError as exc:
                result = {"error": f"bad arguments for {name}: {exc}", "got": input_dict}
            except (ValueError, RuntimeError) as exc:
                result = {"error": str(exc), "traceback": "Traceback ..."}
        self._record({"action": name, **input_dict}, result)
        envelope = self._view(-1)
        if result.get("interrupted") or "error" in result:
            for key, value in result.items():
                envelope.setdefault(key, value)
        return _ToolResult(name, envelope)

    def solved(self) -> bool:
        return self.eval_success

    def artifact(self, name: str, step: int = -1) -> bytes:
        import numpy as np

        if name.endswith("_rgb.png"):
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (COLS, ROWS), (10, 20, 30)).save(buf, format="PNG")
            return buf.getvalue()
        if name.endswith("_depth.npy"):
            buf = io.BytesIO()
            np.save(buf, np.full((ROWS, COLS), 0.9, dtype=np.float32))
            return buf.getvalue()
        if name.endswith("_world_xyz.npy"):
            world = np.zeros((ROWS, COLS, 3), dtype=np.float32)
            for r in range(ROWS):
                for c in range(COLS):
                    world[r, c] = self._world_at(r, c)
            buf = io.BytesIO()
            np.save(buf, world)
            return buf.getvalue()
        if name.endswith("_camera_meta.json"):
            meta = {
                "intrinsic_K": [[300, 0, 160], [0, 300, 120], [0, 0, 1]],
                "extrinsic_cv": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                "cam2world_gl": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "width": COLS,
                "height": ROWS,
            }
            return json.dumps(meta).encode()
        raise FileNotFoundError(name)

    def env_steps(self) -> int:
        return self.actions

    def cancel(self) -> None:
        self.cancelled = True

    # -- native bookkeeping -------------------------------------------------

    def _status(self) -> dict[str, Any]:
        return {
            "eval_success": self.eval_success,
            "take_action_cnt": self.actions,
            "step_lim": self.STEP_LIM,
            "actual_seed": 100001,
            "policy_actions": self.policy_actions,
            "native_actions": self.actions,
        }

    def _budget_exhausted(self) -> bool:
        return self.actions >= self.STEP_LIM

    def _run(self, n: int) -> int:
        if self.eval_success or self._budget_exhausted():
            return 0
        executed = min(int(n), self.STEP_LIM - self.actions)
        self.actions += executed
        return executed

    def _completion(self, requested: int, executed: int) -> dict[str, Any]:
        if self.eval_success:
            reason = "native_success"
        elif self._budget_exhausted():
            reason = "budget_exhausted"
        elif executed == requested:
            reason = "completed"
        else:
            reason = "runtime_failure"
        return {
            "completed": executed == requested,
            "requested_steps": requested,
            "executed_steps": executed,
            "stop_reason": reason,
        }

    def _tcp(self, arm: str) -> list[float]:
        eef = self.arms[arm]["eef"]
        return [eef[0], eef[1] + 0.12, eef[2]]

    def _require_active(self) -> None:
        if self.eval_success or self._budget_exhausted():
            raise RuntimeError("RoboTwin common episode is terminal; reset is required")

    # -- primitives (payload keys as in robots/robotwin/primitives.py) --------

    def _do_render(self) -> dict[str, Any]:
        return {"success": True}

    def _do_move_to(self, arm, xyz, quat=None, gripper=None, substeps=25):
        self._require_active()
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        target = [float(v) for v in xyz]
        if target[2] > 1.5 or math.dist(target, self.arms[arm]["eef"]) > 1.0:
            return {
                "completed": False,
                "requested_steps": 0,
                "executed_steps": 0,
                "stop_reason": "plan_failed",
                "success": False,
                "plan_status": "Fail",
                "hint": "target may be unreachable or in collision",
            }
        waypoints = 3 if int(substeps) == 0 else int(substeps)
        executed = self._run(waypoints)
        if executed == waypoints:
            self.arms[arm]["eef"] = target
            if quat is not None:
                self.arms[arm]["quat"] = [float(v) for v in quat]
        if gripper is not None:
            self.arms[arm]["grip"] = float(gripper)
        final = self.arms[arm]["eef"]
        return {
            "action_type": "qpos",
            "requested_actions": waypoints,
            "executed_actions": executed,
            "episode_status": self._status(),
            **self._completion(waypoints, executed),
            "success": True,
            "plan_status": "Success",
            "waypoints": waypoints,
            "final_eef_xyz": list(final),
            "final_dist_m": round(math.dist(final, target), 6),
        }

    def _do_rotate_wrist(self, arm, delta_yaw_deg, gripper=None, substeps=25):
        yaw = math.radians(float(delta_yaw_deg))
        w, x, y, z = self.arms[arm]["quat"]
        cw, cz = math.cos(yaw / 2), math.sin(yaw / 2)
        quat = [cw * w - cz * z, cw * x - cz * y, cw * y + cz * x, cw * z + cz * w]
        result = self._do_move_to(
            arm, list(self.arms[arm]["eef"]), quat=quat, gripper=gripper, substeps=substeps
        )
        result["requested_delta_yaw_deg"] = float(delta_yaw_deg)
        return result

    def _do_set_gripper(self, arm, val, steps=10):
        self._require_active()
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        executed = self._run(int(steps))
        if executed:
            self.arms[arm]["grip"] = float(val)
        return {
            "action_type": "qpos",
            "requested_actions": int(steps),
            "executed_actions": executed,
            "episode_status": self._status(),
            **self._completion(int(steps), executed),
            "success": True,
            "gripper_val": self.arms[arm]["grip"],
        }

    def _do_release(self, arm, val=1.0, steps=10):
        return self._do_set_gripper(arm, val, steps)

    def _do_lingbot_act(self, chunks=4, use_length=50, prompt=None):
        if int(chunks) < 1:
            raise ValueError("chunks must be at least 1")
        if int(use_length) != 50:
            raise ValueError("RoboTwin LingBot requires use_length=50")
        requested = int(chunks) * 50
        executed = 0
        for _ in range(int(chunks)):
            if self.eval_success or self._budget_exhausted():
                break
            if not self.holding:
                n = self._run(50)
                if n == 50:
                    self.holding = True
                    self.arms["left"]["eef"] = [-0.05, -0.08, 0.86]
                    self.arms["left"]["grip"] = 0.0
            else:
                n = self._run(8)
                if n == 8:
                    self.eval_success = True
            executed += n
            self.policy_actions += n
        return {
            **self._completion(requested, executed),
            "success": True,
            "prompt": self.TASK,
            "agent_prompt_ignored": prompt is not None,
            "ignored_agent_prompt": prompt,
            "episode_status": self._status(),
        }

    # -- read-only ------------------------------------------------------------

    def _world_at(self, r: int, c: int) -> list[float]:
        r0, c0, r1, c1 = self.HAMMER_BOX
        z = self.HAMMER_Z if r0 <= r < r1 and c0 <= c < c1 else self.TABLE_Z
        return [(c - 160) * 0.0025, (120 - r) * 0.0025, z]

    def _world_error(self, code: str, message: str, **details: Any) -> dict[str, Any]:
        return {"success": False, "error": {"code": code, "message": message, **details}}

    def _readonly(self, name: str, input_dict: dict[str, Any]) -> dict[str, Any]:
        if name == "view_env_state":
            step = int(input_dict.get("step", -1))
            if step >= len(self.steps):
                return {"error": f"state step not available: step {step}"}
            return self._view(step)
        view = input_dict["view"]
        step = input_dict.get("step")
        if view not in ("head", "left_wrist", "right_wrist"):
            return self._world_error("view_not_found", "unknown view", view=view)
        if step is not None and step >= len(self.steps):
            return self._world_error("state_not_found", "no such step", step=step)
        step_idx = self.steps[-1]["step_idx"] if step in (None, -1) else int(step)
        base = {
            "success": True,
            "step_idx": step_idx,
            "view": view,
            "coordinate_space": view,
            "image_shape": [ROWS, COLS],
            "pixel_order": "row_col",
            "coordinate_order": "xyz",
            "frame": "world",
            "unit": "metre",
        }
        if name == "sample_world_xyz":
            samples = []
            for pixel in input_dict["pixels"]:
                r, c = int(pixel[0]), int(pixel[1])
                if not (0 <= r < ROWS and 0 <= c < COLS):
                    return self._world_error(
                        "pixel_out_of_bounds",
                        "The pixel is outside this view's world map.",
                        pixel=[r, c],
                        shape=[ROWS, COLS],
                        valid_row_range=[0, ROWS - 1],
                        valid_col_range=[0, COLS - 1],
                    )
                samples.append(
                    {
                        "pixel": [r, c],
                        "valid": True,
                        "xyz": self._world_at(r, c),
                        "valid_points": 9,
                        "valid_coordinates": [9, 9, 9],
                    }
                )
            return {**base, "neighborhood": input_dict.get("neighborhood", 1), "samples": samples}
        r0, c0, r1, c1 = (int(v) for v in input_dict["bbox"])
        if not (0 <= r0 < r1 <= ROWS and 0 <= c0 < c1 <= COLS):
            return self._world_error("bbox_out_of_bounds", "bbox must be inside the view")
        pts = [(r, c, self._world_at(r, c)) for r in range(r0, r1) for c in range(c0, c1)]
        limit = int(input_dict.get("max_points", 256))
        chosen = pts[:: max(1, len(pts) // limit)][:limit]
        zs = sorted(p[2][2] for p in pts)
        return {
            **base,
            "bbox": [r0, c0, r1, c1],
            "bbox_interval": "half_open",
            "valid_points": len(pts),
            "returned_points": len(chosen),
            "xyz_min": [min(p[2][i] for p in pts) for i in range(3)],
            "xyz_max": [max(p[2][i] for p in pts) for i in range(3)],
            "xyz_median": [pts[len(pts) // 2][2][0], pts[len(pts) // 2][2][1], zs[len(zs) // 2]],
            "points": [{"pixel": [r, c], "xyz": xyz} for r, c, xyz in chosen],
        }

    def _robot_state(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for arm in ("left", "right"):
            out[f"{arm}_eef_pose"] = list(self.arms[arm]["eef"]) + list(self.arms[arm]["quat"])
            out[f"{arm}_tcp_pose"] = self._tcp(arm) + list(self.arms[arm]["quat"])
            out[f"{arm}_gripper"] = self.arms[arm]["grip"]
        out["qpos_target14"] = (
            [0.0] * 6 + [self.arms["left"]["grip"]] + [0.0] * 6 + [self.arms["right"]["grip"]]
        )
        out["arm_qpos_real12"] = [0.0] * 12
        return out

    def _record(self, command: dict[str, Any], result: dict[str, Any]) -> None:
        idx = len(self.steps)
        views = ("head", "left_wrist", "right_wrist")
        self.steps.append(
            {
                "step_idx": idx,
                "state": {
                    "step_idx": idx,
                    "task_name": "beat_block_hammer",
                    "task_language": self.TASK,
                    "robot_state": self._robot_state(),
                    "episode_status": self._status(),
                    "artifacts": {v: {"rgb": f"{v}_rgb.png/{idx:02d}.png"} for v in views},
                    "view_specs": {
                        v: {"coordinate_space": v, "image_shape": [ROWS, COLS]} for v in views
                    },
                    "log": {"command": command, "result": result, "elapsed_s": 0.5},
                },
                "terminated": self.eval_success,
                "truncated": False,
                "command": command,
                "result": result,
            }
        )

    def _view(self, step: int) -> dict[str, Any]:
        record = self.steps[step]
        return {
            "step": record["step_idx"],
            "terminated": record["terminated"],
            "truncated": record["truncated"],
            "state": json.loads(json.dumps(record["state"])),
            "artifacts": sorted(
                f"{v}_{f}" for v in ("head", "left_wrist", "right_wrist") for f in ("rgb.png",)
            ),
            "task_language": self.TASK,
            "log": {"command": record["command"], "result": record["result"], "elapsed_s": 0.5},
            "_image_bytes": b"png",
            "_image_cam_bytes": b"png",
            "_image_wrist_bytes": b"png",
        }


@pytest.fixture
def rt_backend() -> FakeRobotwinBackend:
    return FakeRobotwinBackend()


@pytest.fixture
def rt(rt_backend) -> RobotwinRobot:
    return RobotwinRobot(rt_backend)


# -- state ---------------------------------------------------------------------


def test_state_reads_both_arms_and_the_episode_status(rt, rt_backend):
    state = rt.state()
    assert isinstance(state, State) and isinstance(state.left, ArmState)
    assert state.task == rt_backend.TASK and rt.task == rt_backend.TASK
    assert state.left.eef_pos == (-0.298, -0.314, 0.942)
    assert state.left.eef_quat == (0.7, 0.0, 0.0, 0.714)  # [w, x, y, z]
    assert state.left.tcp_pos == pytest.approx((-0.298, -0.194, 0.942))
    assert state.left.approach == pytest.approx((0.0, 1.0, 0.0))
    assert state.left.eef_yaw == pytest.approx(math.pi / 2, abs=0.03)
    assert state.right.gripper == 1.0 and state.arm("right") is state.right
    assert state.eval_success is False and state.take_action_cnt == 0
    assert state.step_lim == 300 and state.actions_left == 300
    assert not state.terminated and not state.truncated and not rt.done
    with pytest.raises(ValueError):
        state.arm("middle")


def test_live_envelopes_carry_numpy_arrays(rt_backend):
    """RPent's in-memory ``view_env_state`` holds ``robot_state`` poses as numpy arrays."""
    np = pytest.importorskip("numpy")
    original_view = rt_backend._view

    def numpy_view(step):
        envelope = original_view(step)
        robot = envelope["state"]["robot_state"]
        for key, value in list(robot.items()):
            if isinstance(value, list):
                robot[key] = np.asarray(value, dtype=np.float64)
        return envelope

    rt_backend._view = numpy_view
    robo = RobotwinRobot(rt_backend)
    state = robo.state()
    assert robo.task == rt_backend.TASK
    assert state.left.eef_pos == (-0.298, -0.314, 0.942)
    assert state.right.tcp_quat == (0.704, 0.0, 0.0, 0.711)
    assert robo.move_to("left", [-0.2, -0.1, 0.9], substeps=2).reached
    json.dumps(robo._summary_dict())


def test_summary_dict_is_compact_and_json_friendly(rt):
    summary = rt._summary_dict()
    assert summary["step"] == 0 and summary["actions"] == "0/300"
    assert summary["left_gripper"] == 1.0 and summary["eval_success"] is False
    assert set(summary) >= {"left_eef", "right_eef", "terminated", "truncated"}
    json.dumps(summary)


# -- scripted motion --------------------------------------------------------------


def test_move_to_passes_rpent_argument_names_and_returns_a_typed_result(rt, rt_backend):
    result = rt.move_to("left", [-0.2, -0.1, 0.9], substeps=4)
    assert isinstance(result, Move)
    assert result.planned and result.reached and result.final_dist_m == 0.0
    assert result.actions_used == 4 and result.waypoints == 4
    assert result.stop_reason == "completed" and result.arm == "left"
    name, kwargs = rt_backend.calls[-1]
    assert name == "move_to"
    assert kwargs == {"arm": "left", "xyz": [-0.2, -0.1, 0.9], "substeps": 4}
    rt.move_to("right", [0.3, -0.2, 0.9], quat=[1, 0, 0, 0], gripper=0.5, substeps=2)
    kwargs = rt_backend.calls[-1][1]
    assert kwargs["quat"] == [1.0, 0.0, 0.0, 0.0] and kwargs["gripper"] == 0.5
    assert rt.state().right.gripper == 0.5 and rt.state().right.eef_quat == (1.0, 0.0, 0.0, 0.0)


def test_plan_failure_is_a_measured_outcome_not_an_exception(rt):
    result = rt.move_to("left", [0.0, 0.0, 2.5])
    assert result.planned is False and result.reached is False
    assert result.final_eef_pos is None and math.isinf(result.final_dist_m)
    assert result.stop_reason == "plan_failed" and result.actions_used == 0
    assert rt.state().take_action_cnt == 0


def test_library_side_argument_validation(rt, rt_backend):
    before = len(rt_backend.calls)
    with pytest.raises(ValueError, match="arm"):
        rt.move_to("middle", [0.0, 0.0, 0.9])
    with pytest.raises(ValueError):
        rt.move_to("left", [0.0, 0.0])
    with pytest.raises(ValueError):
        rt.move_to("left", [0.0, 0.0, 0.9], quat=[1, 0, 0])
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        rt.move_to("left", [0.0, 0.0, 0.9], gripper=-1.0)
    with pytest.raises(ValueError):
        rt.move_to("left", [0.0, 0.0, 0.9], substeps=-1)
    with pytest.raises(ValueError):
        rt.set_gripper("left", 1.5)
    with pytest.raises(ValueError):
        rt.set_gripper("left", 0.5, steps=0)
    with pytest.raises(ValueError, match="chunks"):
        rt.lingbot_act(0)
    with pytest.raises(ValueError, match="use_length"):
        rt.lingbot_act(1, use_length=25)
    with pytest.raises(ValueError):
        rt.sample_world_xyz("rear", [[1, 2]])
    with pytest.raises(ValueError):
        rt.sample_world_xyz("head", [[1, 2, 3]])
    with pytest.raises(ValueError):
        rt.query_world_map("head", [1, 2, 3])
    with pytest.raises(ValueError):
        rt.query_world_map("head", [0, 0, 10, 10], max_points=0)
    assert len(rt_backend.calls) == before  # nothing was dispatched


def test_rotate_wrist_keeps_the_position_and_reports_the_yaw(rt, rt_backend):
    before = rt.state().right.eef_pos
    result = rt.rotate_wrist("right", 10.0, substeps=3)
    assert isinstance(result, Move) and result.planned and result.actions_used == 3
    assert result.target_xyz == before and result.final_eef_pos == before
    assert rt_backend.calls[-1][1] == {"arm": "right", "delta_yaw_deg": 10.0, "substeps": 3}
    assert rt.state().right.eef_yaw == pytest.approx(math.pi / 2 + math.radians(10), abs=0.03)


def test_gripper_commands(rt, rt_backend):
    closed = rt.set_gripper("left", RobotwinRobot.CLOSE, steps=5)
    assert isinstance(closed, Gripper) and closed.final == 0.0 and closed.actions_used == 5
    assert rt_backend.calls[-1][1] == {"arm": "left", "val": 0.0, "steps": 5}
    opened = rt.release("left")
    assert opened.target == 1.0 and opened.final == 1.0 and opened.actions_used == 10
    assert rt_backend.calls[-1] == ("release", {"arm": "left", "val": 1.0, "steps": 10})
    assert rt.state().take_action_cnt == 15


# -- the VLA and the episode end --------------------------------------------------


def test_vla_runs_terminate_the_episode_and_freeze_motion(rt, rt_backend):
    first = rt.lingbot_act(chunks=2)
    assert isinstance(first, VlaRun)
    assert first.chunks_requested == 2 and first.actions_requested == 100
    assert first.actions_used == 58 and first.stop_reason == "native_success"
    assert first.instruction == rt_backend.TASK and first.terminated
    assert rt_backend.calls[-1][1] == {"chunks": 2, "use_length": 50}
    assert rt.done and rt_backend.solved()
    state = rt.state()
    assert state.eval_success and state.terminated and state.left.gripper == 0.0
    with pytest.raises(EpisodeFinished) as info:
        rt.move_to("left", [-0.2, -0.1, 0.9])
    assert info.value.reason == "terminated"
    assert rt.state().terminated  # read-only calls still work


def test_prompt_argument_is_recorded_but_the_task_language_is_used(rt, rt_backend):
    run = rt.lingbot_act(1, prompt="grab it")
    assert rt_backend.calls[-1][1]["prompt"] == "grab it"
    assert run.instruction == rt_backend.TASK and run.actions_used == 50
    assert not run.terminated and not rt.done


def test_step_budget_exhaustion_is_truncation(rt, rt_backend):
    rt_backend.holding = True  # skip the grasp; every VLA call now needs the block
    rt_backend.STEP_LIM = 20
    result = rt.move_to("left", [-0.2, -0.1, 0.9], substeps=25)
    assert result.actions_used == 20 and result.stop_reason == "budget_exhausted"
    assert result.truncated and rt.state().truncated and rt.state().actions_left == 0
    with pytest.raises(EpisodeFinished) as info:
        rt.set_gripper("left", 0.0)
    assert info.value.reason == "truncated"


def test_render_records_a_fresh_step_without_moving(rt, rt_backend):
    state = rt.render()
    assert isinstance(state, State) and state.step == 1 and state.take_action_cnt == 0
    assert rt_backend.calls[-1] == ("render", {})
    record = rt._ledger.records[-1]
    assert record.tool == "render" and record.stateful and record.env_steps == 0


def test_backend_error_payload_becomes_tool_error(rt, rt_backend):
    rt_backend.error_next = {"error": "simulated failure", "traceback": "..."}
    with pytest.raises(ToolError, match="simulated failure") as info:
        rt.set_gripper("left", 0.0)
    assert info.value.tool == "set_gripper"
    assert rt._ledger.records[-1].ok is False


def test_cancellation_surfaces_as_episode_finished(rt, rt_backend):
    rt._stop_episode("timeout")
    with pytest.raises(EpisodeFinished) as info:
        rt.lingbot_act(1)
    assert info.value.reason == "timeout" and rt_backend.cancelled


# -- perception ------------------------------------------------------------------


def test_world_queries_return_typed_results(rt, rt_backend):
    samples = rt.sample_world_xyz("head", [[120, 160], [10, 10]])
    assert isinstance(samples, WorldSamples) and len(samples.samples) == 2
    assert samples.samples[0].pixel == (120, 160)
    assert samples.samples[0].xyz == pytest.approx((0.0, 0.0, rt_backend.HAMMER_Z))
    assert samples.xyz[1][2] == pytest.approx(rt_backend.TABLE_Z)
    assert samples.image_shape == (240, 320) and samples.step == 0
    assert rt_backend.calls[-1][1] == {
        "view": "head",
        "pixels": [[120, 160], [10, 10]],
        "step": -1,
        "neighborhood": 1,
    }
    single = rt.sample_world_xyz("left_wrist", (5, 6), neighborhood=2)
    assert single.samples[0].pixel == (5, 6) and rt_backend.calls[-1][1]["neighborhood"] == 2
    region = rt.query_world_map("head", rt_backend.HAMMER_BOX, max_points=8)
    assert isinstance(region, WorldRegion) and region.valid_points == 1600
    assert region.xyz_median[2] == pytest.approx(rt_backend.HAMMER_Z)
    assert len(region.points) <= 8 and region.points[0][0] == (100, 140)
    assert rt_backend.calls[-1][1]["bbox"] == [100, 140, 140, 180]


def test_world_query_errors_carry_the_rpent_error_code(rt):
    with pytest.raises(ToolError, match="pixel_out_of_bounds") as info:
        rt.sample_world_xyz("head", [[500, 500]])
    assert info.value.payload["error"]["valid_row_range"] == [0, 239]
    with pytest.raises(ToolError, match="bbox_out_of_bounds"):
        rt.query_world_map("head", [0, 0, 500, 500])
    with pytest.raises(ToolError, match="state_not_found"):
        rt.sample_world_xyz("head", [[1, 1]], step=7)


def test_images_world_maps_depth_and_calibration_decode(rt):
    pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")
    image = rt.image("head")
    assert image.shape == (240, 320, 3) and image.dtype == np.uint8
    assert rt.image("right_wrist").shape == (240, 320, 3)
    world = rt.world_map("head")
    assert world.shape == (240, 320, 3) and world.dtype == np.float32
    assert float(np.nanmedian(world[..., 2])) == pytest.approx(0.715)
    assert rt.depth("left_wrist").shape == (240, 320)
    assert rt.camera_meta("head")["width"] == 320
    with pytest.raises(ValueError):
        rt.image("rear")
    with pytest.raises(ValueError):
        rt.world_map("head", resolution="low")
    with pytest.raises(ValueError):
        rt.depth("rear")


def test_ledger_records_every_call_with_env_steps(rt, rt_backend):
    rt.state()
    rt.move_to("left", [-0.2, -0.1, 0.9], substeps=4)
    rt.lingbot_act(1)
    records = rt._ledger.records
    assert [r.tool for r in records] == ["view_env_state", "move_to", "lingbot_act"]
    assert records[0].stateful is False and records[1].stateful is True
    assert records[1].env_steps == 4 and records[2].env_steps == 50
    assert records[1].summary["stop_reason"] == "completed"
    assert rt._ledger.env_steps() == rt_backend.actions == 54


# -- prompt, adapter, arm toolkit ------------------------------------------------


def _public_methods() -> list[str]:
    return list(_public_members(RobotwinRobot))


def test_api_reference_lists_every_public_method_and_result_type():
    text = api_reference(RobotwinRobot)
    for name in _public_methods():
        assert f"robo.{name}" in text, name
    for name in (
        "render",
        "sample_world_xyz",
        "query_world_map",
        "lingbot_act",
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
        "depth",
        "camera_meta",
    ):
        assert f"robo.{name}(" in text, name
    for cls in RobotwinRobot.RESULT_TYPES:
        assert f"class {cls.__name__}(" in text
    assert "robo.LEFT = 'left'" in text and "robo.RIGHT = 'right'" in text
    assert "robo.OPEN = 1.0" in text and "robo.CLOSE = 0.0" in text
    assert "robo.EEF_TO_TCP_M = 0.12" in text
    assert "RESULT_TYPES" not in text and "SUMMARY_KEYS" not in text and "CAMERAS" not in text
    assert "[w, x, y, z]" in text and "0.12" in text
    assert "robo.move_to(\n    arm: str,\n    xyz: Sequence[float],\n    *,\n    quat" in text
    assert "chunks: int = 4" in text and "substeps: int = 25" in text and "steps: int = 10" in text


def test_adapter_and_prompt_wording():
    adapter = get_adapter("robotwin")
    assert adapter.robot_cls is RobotwinRobot and adapter.knowledge == "robotwin"
    assert adapter.cameras == ("head", "left_wrist", "right_wrist")
    assert adapter.host == "pyrualean.hosts.rpent_robotwin"
    assert adapter.guides_subdir == "robots/robotwin/guides"
    assert adapter.guide_names == ("GUIDE_RPENT.md",)
    card = TaskCard(
        robot="robotwin",
        suite="demo_randomized",
        task="beat_block_hammer",
        seed=100000,
        task_language="Catch the hammer and use it on the block.",
        object_names=[],
        facts={"left gripper": "1.00 (1 = open, 0 = closed)"},
    )
    bundle = render_prompt(card, budget_s=600)
    assert "a dual-arm robot in the RoboTwin benchmark" in bundle.system
    assert "class RobotwinRobot" in bundle.system
    assert "# Operating knowledge for RoboTwin" in bundle.system
    know = knowledge_text("robotwin")
    for banned in ("MEMORY.md", "task_only", "recipe.jsonl", "audit", "server", "launch"):
        assert banned not in know, banned
    assert "- robot: robotwin" in bundle.user and "left gripper: 1.00" in bundle.user
    cells = build_prompt(
        "cells", card, budget_s=600, max_programs=None, images="on-demand", feedback="pure"
    )
    assert "`RobotwinRobot` bound to the live simulator, a\nfrozen LingBot-VLA" in cells
    assert "`robo.show('head')`, `robo.show('left_wrist')` or `robo.show('right_wrist')`" in cells


def test_arm_toolkit_attaches_the_three_views(rt_backend, tmp_path):
    pytest.importorskip("PIL")
    robo = RobotwinRobot(rt_backend)
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", images="on-motion")
    code = "robo.move_to('left', [-0.2, -0.1, 0.9], substeps=2)"
    moved = kit.execute_tool("python", {"code": code})
    assert moved.result["moved"] is True and moved.result["env_steps"] == 2
    assert moved.result["images"] == ["head_rgb.png", "left_wrist_rgb.png", "right_wrist_rgb.png"]
    assert moved.result["state"]["actions"] == "2/300"
    demand = ArmToolkit(
        RobotwinRobot(rt_backend),
        arm="cells",
        workspace=tmp_path / "ws2",
        images="on-demand",
        feedback="pure",
    )
    looked = demand.execute_tool("python", {"code": "robo.show('left_wrist'); print(robo.task)"})
    assert looked.result["images"] == ["left_wrist_rgb.png"] and len(looked.images) == 1
    assert looked.result["stdout"].strip() == rt_backend.TASK and "state" not in looked.result


# -- host --------------------------------------------------------------------------


class _FakeState:
    """Enough of RPent's ``EnvState`` for ``dump_card``: artifact paths and loads."""

    def __init__(self, backend: FakeRobotwinBackend, root: Path) -> None:
        self.backend = backend
        self.root = root

    def artifact_path(self, name: str, *, step: int = -1) -> Path:
        suffix = Path(name).suffix
        path = self.root / name / f"{step:02d}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.backend.artifact(name, step))
        return path

    def load(self, name: str, *, step: int = -1) -> Any:
        import numpy as np

        return np.load(io.BytesIO(self.backend.artifact(name, step)))


class _FakeToolkit:
    def __init__(self, backend: FakeRobotwinBackend, root: Path) -> None:
        self.backend = backend
        self.state = _FakeState(backend, root)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        return self.backend.execute_tool(name, input_dict)


def test_cell_and_argument_helpers():
    cell = host.Cell("demo_randomized", "beat_block_hammer", 100000, cuda_device=7)
    assert cell.tag == "beat_block_hammer_randomized_s100000"
    assert host.Cell("demo_clean", "stack_bowls_two", 3).tag == "stack_bowls_two_clean_s3"
    args = Namespace(task="place_shoe", suite=None, seed=100001, max_episode_steps=500)
    parsed = host.cell_from_args(args)
    assert parsed == host.Cell("demo_randomized", "place_shoe", 100001, 500, None)
    options = host.boot_options(
        Namespace(vla_endpoint="127.0.0.1:9000", vla_model_path="/m", robotwin_assets_path="/a")
    )
    assert options == {
        "vla_endpoint": "127.0.0.1:9000",
        "env_endpoint": None,
        "vla_model_path": "/m",
        "robotwin_assets_path": "/a",
        "lingbot_robot_config": None,
    }
    import argparse

    parser = argparse.ArgumentParser()
    host.add_cell_args(parser)
    host.add_boot_args(parser)
    ns = parser.parse_args(["--task", "beat_block_hammer", "--seed", "100003"])
    assert ns.suite == "demo_randomized" and host.cell_from_args(ns).seed == 100003


def test_dump_card_writes_json_images_and_facts(rt_backend, tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    toolkit = _FakeToolkit(rt_backend, tmp_path / "artifacts")
    cell = host.Cell("demo_randomized", "beat_block_hammer", 100001, cuda_device=7)
    card = host.dump_card(toolkit, cell, tmp_path)
    assert (tmp_path / "card.json").is_file()
    assert set(card.images) == {"head", "left_wrist", "right_wrist"}
    assert all(Path(p).is_file() for p in card.images.values())
    assert card.robot == "robotwin" and card.task == "beat_block_hammer"
    assert card.task_language == rt_backend.TASK and card.object_names == []
    assert card.eef_pos is None and card.eef_quat is None and card.gripper_opening is None
    facts = card.facts
    assert facts["left arm initial EEF position [x, y, z]"] == "[-0.298, -0.314, 0.942] m"
    assert facts["left arm initial EEF orientation [w, x, y, z]"] == "[0.700, 0.000, 0.000, 0.714]"
    assert facts["right arm initial TCP (gripper centre) [x, y, z]"] == "[0.306, -0.193, 0.941] m"
    assert facts["native action budget (step_lim)"] == 300
    assert facts["table surface height (median z of the head world map at step 0)"] == "0.715 m"
    text = card.render()
    assert "- robot: robotwin" in text and "- variant: demo_randomized" in text
    assert "`robo.image('head')`" in text and "step 0 of your run" in text
    assert TaskCard.from_json(tmp_path / "card.json") == card


def test_backend_subclass_reads_rpent_toolkit_internals():
    class _Primitives:
        native_actions = 17

        def status(self):
            return {"eval_success": True, "take_action_cnt": 17}

    class _Toolkit:
        _primitives = _Primitives()
        _latest_status = {"eval_success": False}

    backend = host.RobotwinBackend(_Toolkit())
    assert backend.solved() is True and backend.env_steps() == 17

    class _Bare:
        _latest_status = {"eval_success": True}

    assert host.RobotwinBackend(_Bare()).solved() is True
    assert host.RobotwinBackend(_Bare()).env_steps() is None
    assert isinstance(host.make_backend(_Bare()), host.RobotwinBackend)
