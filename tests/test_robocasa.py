"""Behaviour of the typed RoboCasa365 API over an RPent-shaped fake backend, plus its host."""

from __future__ import annotations

import argparse
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
from pyrualean.hosts import rpent_robocasa as host
from pyrualean.play import build_prompt
from pyrualean.prompt import _public_members
from pyrualean.robocasa import (
    BackProjection,
    BaseMove,
    Grasp,
    Gripper,
    HeightQuery,
    Move,
    Navigation,
    Pitch,
    RobocasaRobot,
    Skill,
    State,
)

np = pytest.importorskip("numpy")

SIZE = 256
SPEED = 0.0023  # base travel per env step, metres


class _ToolResult:
    """Shape of ``rpent.tools.toolkit.ToolResult`` (only ``.result`` matters)."""

    def __init__(self, name: str, result: dict[str, Any]) -> None:
        self.name = name
        self.result = result


class FakeRobocasaBackend:
    """A kitchen with a drawer handle and a can on the counter.

    Mirrors RPent's RoboCasa envelope (``robots/robocasa/tools.py``): every
    state-changing tool returns a ``view_env_state`` dict whose ``log.result``
    holds the primitive's own return value and which carries ``success`` /
    ``robocasa_terminated`` / ``task_progress`` / ``vla_desync`` but no
    top-level ``terminated`` / ``truncated``; ``back_project_batch`` and
    ``query_world_map`` return their payload directly.  All poses are plain
    lists, as RPent's ``current_state_dict`` produces them.  Success = the
    drawer joint opens past 0.95, which only the VLA achieves (26 chunks)
    when the base stands within 0.8 m of the handle.
    """

    TASK = "Open the left drawer."
    HANDLE = (0.944, -0.598, 0.783)
    HANDLE_BOX = (195, 215, 50, 85)  # rows / cols of the handle in the agentview
    CAN = (1.40, -0.60, 0.95)
    CAN_BOX = (120, 140, 150, 170)
    COUNTER_Z = 0.90
    KEEP_HEAVY = 25
    CRITERIA = (
        "# SUCCESS CONDITION\n    def _check_success(self):\n        return joint_p >= 0.95\n"
    )

    def __init__(self) -> None:
        self.eef = [1.453, -0.677, 1.306]
        self.tilt = 0.0
        self.qpos = [0.0206, -0.0205]
        self.base = [1.432, -0.916, 0.700]
        self.base_quat = [0.0, 0.0, 0.7071068, 0.7071068]
        self.joint_p = 0.001
        self.frames = 0
        self.vla_desync = True
        self.jac_calibrated = False
        self.heading_calibrated = False
        self.holding = False
        self.steps: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.cancelled = False
        self.error_next: dict[str, Any] | None = None
        self.capture_error_next = False
        self.truncate_next = False
        self._maps: dict[str, Any] = {}
        self._record(None, None)

    # -- Backend contract ---------------------------------------------------

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        self.calls.append((name, dict(input_dict)))
        if name in ("view_env_state", "back_project_batch", "query_world_map"):
            return _ToolResult(name, self._readonly(name, input_dict))
        raised = False
        if self.error_next is not None:
            result, self.error_next = self.error_next, None
            raised = True
        elif self.cancelled:
            result = {
                "error": "tool operation interrupted",
                "code": "tool_cancelled",
                "interrupted": True,
            }
            raised = True
        else:
            handler = getattr(self, f"_do_{name}", None)
            if handler is None:
                return _ToolResult(name, {"error": f"unknown tool: {name}"})
            try:
                result = handler(**input_dict)
            except TypeError as exc:
                result = {"error": f"bad arguments for {name}: {exc}", "got": input_dict}
                raised = True
        if self.capture_error_next:
            # RPent: get_env_state raised, so the raw result comes back with
            # state_capture_error and no state key (rpent/tools/toolkit.py:306-312).
            self.capture_error_next = False
            captured = dict(result)
            captured["state_capture_error"] = "disk full"
            captured.setdefault("error", f"failed to capture state after {name}: disk full")
            captured.setdefault("traceback", "Traceback ...")
            return _ToolResult(name, captured)
        self._record({"action": name, **input_dict}, result)
        envelope = self._view(-1)
        envelope["agent_elapsed_s"] = 0.5
        if result.get("interrupted"):
            envelope.update(result)
        elif raised:
            for key, value in result.items():
                envelope.setdefault(key, value)
        return _ToolResult(name, envelope)

    def solved(self) -> bool:
        return bool(self.steps[-1]["success"])

    def artifact(self, name: str, step: int | None = -1) -> bytes:
        if step is None:
            if name == "success_criteria.md":
                return self.CRITERIA.encode()
            raise FileNotFoundError(name)
        latest = len(self.steps) - 1
        idx = latest if step == -1 else int(step)
        if not 0 <= idx <= latest:
            raise FileNotFoundError(name)
        camera = name.split("_")[0].split(".")[0]
        if name.endswith(".png"):
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (SIZE, SIZE), (10, 20, 30)).save(buf, format="PNG")
            return buf.getvalue()
        if name.endswith("_world.npz"):
            if idx <= latest - self.KEEP_HEAVY:
                raise FileNotFoundError(name)  # pruned heavy artifact
            buf = io.BytesIO()
            np.savez(buf, array=self._map(camera))
            return buf.getvalue()
        if name.endswith("_depth.npz") and camera in ("agentview", "wrist"):
            buf = io.BytesIO()
            np.savez(buf, array=np.full((SIZE, SIZE), 1.0, dtype=np.float32))
            return buf.getvalue()
        if name.endswith("_metadata.json") and camera in ("agentview", "wrist"):
            meta = {
                "camera_name": camera,
                "height": SIZE,
                "width": SIZE,
                "intrinsic": [[221.7, 0, 128], [0, 221.7, 128], [0, 0, 1]],
                "extrinsic_cam2world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "depth_near": 0.01,
                "depth_far": 50.0,
            }
            return json.dumps(meta).encode()
        raise FileNotFoundError(name)

    def env_steps(self) -> int:
        return self.frames

    def cancel(self) -> None:
        self.cancelled = True

    # -- world model ----------------------------------------------------------

    def _map(self, camera: str) -> Any:
        if camera not in self._maps:
            rows, cols = np.indices((SIZE, SIZE))
            if camera == "navview":
                x = self.base[0] - 0.5 + cols * 0.004
                y = self.base[1] + 0.3 + rows * 0.004
                z = np.where(rows >= 100, 0.0, 0.5)
                world = np.stack([x, y, z], axis=-1)
            elif camera == "wrist":
                x = self.eef[0] - 0.1 + cols * 0.0008
                y = self.eef[1] - 0.1 + rows * 0.0008
                world = np.stack([x, y, np.full((SIZE, SIZE), self.COUNTER_Z)], axis=-1)
            else:
                x = 0.85 + cols * 0.002
                y = -0.2 - rows * 0.002
                world = np.stack([x, y, np.full((SIZE, SIZE), self.COUNTER_Z)], axis=-1)
                world[rows >= 230, 2] = 0.0  # floor at the bottom of the image
                world[rows < 10] = 0.0  # no depth (sky)
                r0, r1, c0, c1 = self.HANDLE_BOX
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        world[r, c] = [
                            self.HANDLE[0] + (c - (c0 + c1) / 2) * 0.001,
                            self.HANDLE[1] + (r - (r0 + r1) / 2) * 0.001,
                            self.HANDLE[2],
                        ]
                r0, r1, c0, c1 = self.CAN_BOX
                world[r0 : r1 + 1, c0 : c1 + 1] = self.CAN
            self._maps[camera] = world.astype(np.float32)
        return self._maps[camera]

    def _yaw(self) -> float:
        x, y, z, w = self.base_quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _set_yaw(self, yaw: float) -> None:
        self.base_quat = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]

    def _eef_quat(self) -> list[float]:
        half = (self.tilt + math.pi) / 2  # pointing down, tilted about world x
        return [math.sin(half), 0.0, 0.0, math.cos(half)]

    def _success(self) -> bool:
        return self.joint_p >= 0.95

    def _step(self, n: int) -> None:
        self.frames += int(n)

    def _apply_gripper(self, gripper: Any) -> None:
        if gripper is None or gripper == "hold":
            return
        g = float(gripper)
        if g > 0:
            self.qpos = [0.012, -0.012] if self.holding else [0.0, 0.0]
        elif g < 0:
            self.qpos = [0.04, -0.04]
            self.holding = False

    # -- primitives (payload keys as in robots/robocasa/primitives.py) ---------

    def _do_move_to(self, xyz, gripper="hold", step_clip=0.02, max_steps=200, tol=0.012):
        self.vla_desync = True
        if not self.jac_calibrated:
            self._step(9)  # position jacobian probe (3 axes x 3 steps)
            self.jac_calibrated = True
        target = [float(v) for v in xyz]
        if target[2] < 0.75:  # below the counter top: the servo stalls on the surface
            self._step(int(max_steps))
            self.eef = [target[0], target[1], 0.80]
            self._apply_gripper(gripper)
            return {
                "ok": False,
                "steps": int(max_steps),
                "final_dist": round(math.dist(self.eef, target), 4),
                "eef": list(self.eef),
                "gripper_qpos": round(self.qpos[0], 4),
            }
        dist = math.dist(self.eef, target)
        steps = 0 if dist < tol else min(int(max_steps), int(dist / step_clip) + 1)
        self._step(steps)
        self.eef = target
        self._apply_gripper(gripper)
        return {
            "ok": True,
            "steps": steps,
            "final_dist": 0.0,
            "eef": list(self.eef),
            "gripper_qpos": round(self.qpos[0], 4),
        }

    def _do_move_delta(self, dxyz, gripper="hold", step_clip=0.02, max_steps=80):
        target = [c + float(d) for c, d in zip(self.eef, dxyz, strict=True)]
        return self._do_move_to(target, gripper, step_clip, max_steps)

    def _do_rotate_pitch(self, target_pitch=0.6, gripper=1, n=12):
        self.vla_desync = True
        self.tilt += max(-1.5, min(1.5, float(target_pitch)))
        self._step(int(n))
        self._apply_gripper(gripper)
        return {"ok": True, "eef": list(self.eef)}

    def _do_set_gripper(self, gripper=1, steps=10):
        self.vla_desync = True
        self._step(int(steps))
        self._apply_gripper(float(gripper))
        return {"ok": True, "gripper_qpos": list(self.qpos)}

    def _do_release(self, steps=10):
        return self._do_set_gripper(-1.0, steps=steps)

    def _do_scripted_grasp(self, xyz, approach_z=0.10, grasp_z_offset=0.0, step_clip=0.02):
        t = [float(v) for v in xyz]
        self._do_set_gripper(-1.0, steps=4)
        r = self._do_move_to([t[0], t[1], t[2] + approach_z], -1.0, step_clip)
        if not r["ok"]:
            return {**r, "stage": "approach"}
        r = self._do_move_to([t[0], t[1], t[2] + grasp_z_offset], -1.0, 0.012, 200, 0.01)
        if not r["ok"]:
            return {**r, "stage": "descent"}
        self.holding = math.dist(t, self.CAN) < 0.03
        self._do_set_gripper(1.0, steps=14)
        r = self._do_move_to([t[0], t[1], t[2] + approach_z + 0.05], "hold", 0.015)
        if not r["ok"]:
            return {**r, "stage": "lift"}
        return {"ok": True, "gripper_qpos": list(self.qpos), "eef": list(self.eef)}

    def _do_navigate_to(self, xy, tol=0.20, max_steps=300, gripper="hold"):
        self.vla_desync = True
        if not self.heading_calibrated:
            self._step(6)
            self.heading_calibrated = True
        target = [float(xy[0]), float(xy[1])]
        start = [self.base[0], self.base[1]]
        dist = math.dist(start, target)
        yaw = math.atan2(target[1] - start[1], target[0] - start[0]) if dist > 1e-9 else self._yaw()
        need = max(0.0, dist - tol + 0.001)
        blocked = target[0] < 0.6  # a wall on that side: the base rams it after 5 cm
        travel = min(need, 0.05 if blocked else int(max_steps) * SPEED)
        exhausted = blocked or travel < need - 1e-9
        steps = int(max_steps) if exhausted else (math.ceil(travel / SPEED) if travel > 0 else 0)
        dx, dy = travel * math.cos(yaw), travel * math.sin(yaw)
        self.base[0] += dx
        self.base[1] += dy
        self.eef[0] += dx
        self.eef[1] += dy
        self._set_yaw(yaw)
        self._step(steps)
        self.jac_calibrated = False
        final = math.dist(self.base[:2], target)
        out = {
            "ok": final < tol,
            "steps": steps,
            "final_dist": final,
            "moved": travel,
            "start_pos": start,
            "base_pos": list(self.base),
        }
        if not out["ok"]:
            out["stuck"] = travel < 0.12
        return out

    def _do_move_base(self, forward=0, lateral=0, turn=0, steps=10, gripper="hold"):
        self.vla_desync = True
        clip = lambda v: max(-1.0, min(1.0, float(v)))  # noqa: E731
        f, lat = clip(forward) * SPEED * int(steps), clip(lateral) * SPEED * int(steps)
        yaw = self._yaw()
        dx = f * math.cos(yaw) + lat * math.sin(yaw)
        dy = f * math.sin(yaw) - lat * math.cos(yaw)
        self.base[0] += dx
        self.base[1] += dy
        self.eef[0] += dx
        self.eef[1] += dy
        self._set_yaw(yaw + clip(turn) * 0.01 * int(steps))
        self._step(int(steps))
        self.jac_calibrated = False
        return {"ok": True, "base_moved": [dx, dy, 0.0], "base_pos": list(self.base)}

    def _do_rldx_skill(
        self,
        base_clip=None,
        max_chunks=70,
        use_prompt=None,
        prompt="",
        force_reset=False,
        n_action_steps=8,
        settle_patience=999,
        settle_eps=0.012,
    ):
        for name, value in (
            ("max_chunks", max_chunks),
            ("n_action_steps", n_action_steps),
            ("settle_patience", settle_patience),
        ):
            if value < 1:
                return {"error": f"{name} must be positive; VLA was not executed"}
        effective_max = min(int(max_chunks), 40)  # the protocol's RLDX_MAX_CHUNKS
        overridden = prompt != self.TASK
        self.vla_desync = False
        near = math.dist(self.base[:2], self.HANDLE[:2]) < 0.8
        chunks = applied = 0
        status = "cap"
        for c in range(effective_max):
            chunks, applied = c + 1, applied + int(n_action_steps)
            self._step(int(n_action_steps))
            if near and chunks >= 26:
                self.joint_p = 0.9801
                self.eef = [0.969, -0.950, 0.800]
                self.tilt = 1.2
                self.qpos = [-0.00004, -0.0266]
                status = "success"
                break
        result = {
            "ok": True,
            "prompt": prompt,
            "status": status,
            "chunks": chunks,
            "steps_applied": applied,
            "grasped": False,
            "grasp_detected": near,
            "grasp_contact": False,
            "held_apart": near and status == "success",
            "grasp_obj": None,
            "gripper_qpos": round(self.qpos[0], 3),
            "peak_lift": 0.05,
            "base_clip": base_clip,
            "base_drift": 0.003,
            "effective_prompt": self.TASK,
            "effective_max_chunks": effective_max,
            "effective_n_action_steps": int(n_action_steps),
            "effective_settle_patience": int(settle_patience),
            "prompt_overridden": overridden,
        }
        if overridden:
            result["requested_prompt"] = prompt
        return result

    def _do_rldx_arm(self, base_clip=0.1, **kwargs):
        return self._do_rldx_skill(base_clip=base_clip, **kwargs)

    # -- read-only ------------------------------------------------------------

    def _readonly(self, name: str, input_dict: dict[str, Any]) -> dict[str, Any]:
        latest = len(self.steps) - 1
        if name == "view_env_state":
            step = input_dict.get("step")
            idx = latest if step in (None, -1) else int(step)
            if idx > latest:
                return {"error": f"state step not available: step {idx} not present"}
            return self._view(idx)
        camera = input_dict.get("camera") or "agentview"
        resolution = input_dict.get("resolution") or "low"
        if camera not in ("agentview", "navview", "wrist"):
            return {"error": f"bad camera '{camera}' (use agentview, navview, or wrist)"}
        if resolution == "high":
            return {"error": f"{camera} has no {resolution}-resolution world map"}
        world = self._map(camera)
        if name == "back_project_batch":
            step = input_dict.get("step")
            idx = latest if step in (None, -1) else int(step)
            if idx > latest:
                return {"error": f"state step not available: step {idx} not present"}
            if idx <= latest - self.KEEP_HEAVY:
                return {"error": f"{camera}_world.npz not found for step {idx}: pruned"}
            results, valid = [], []
            for pixel in input_dict["pixels"]:
                r, c = int(pixel[0]), int(pixel[1])
                if not (0 <= r < SIZE and 0 <= c < SIZE):
                    results.append(
                        {
                            "pixel": pixel,
                            "world_xyz": None,
                            "valid": False,
                            "error": f"pixel ({r},{c}) out of bounds ({SIZE}x{SIZE})",
                        }
                    )
                    continue
                xyz = world[r, c, :3]
                if abs(float(xyz.sum())) <= 1e-6:
                    results.append(
                        {
                            "pixel": pixel,
                            "world_xyz": None,
                            "valid": False,
                            "error": "invalid world xyz at pixel",
                        }
                    )
                    continue
                results.append(
                    {
                        "pixel": [r, c],
                        "world_xyz": [round(float(v), 4) for v in xyz],
                        "valid": True,
                        "error": None,
                    }
                )
                valid.append([float(v) for v in xyz])
            summary: dict[str, Any] = {"valid_count": len(valid), "total_count": len(results)}
            if valid:
                median = np.median(np.asarray(valid), axis=0)
                summary["median_xyz"] = [round(float(v), 4) for v in median]
            return {
                "results": results,
                "summary": summary,
                "step": idx,
                "camera": camera,
                "resolution": resolution,
            }
        # query_world_map: always the latest step, coarse grid clustering
        z = world[:, :, 2]
        mask = (z >= float(input_dict.get("z_min", 0.85))) & (
            z <= float(input_dict.get("z_max", 0.95))
        )
        if input_dict.get("x_range"):
            lo, hi = input_dict["x_range"]
            mask &= (world[:, :, 0] >= lo) & (world[:, :, 0] <= hi)
        if input_dict.get("y_range"):
            lo, hi = input_dict["y_range"]
            mask &= (world[:, :, 1] >= lo) & (world[:, :, 1] <= hi)
        ys, xs = np.where(mask)
        min_size = int(input_dict.get("min_cluster_size", 10))
        if len(ys) < min_size:
            return {"clusters": [], "summary": {"total_clusters": 0, "total_pixels_matched": 0}}
        cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for i in range(0, len(ys), 5):
            y, x = int(ys[i]), int(xs[i])
            cells.setdefault((y // 32, x // 32), []).append((y, x))
        clusters = []
        for pixels in cells.values():
            if len(pixels) < min_size:
                continue
            pts = np.asarray([world[y, x, :3] for y, x in pixels])
            center, lo, hi = np.median(pts, axis=0), pts.min(axis=0), pts.max(axis=0)
            clusters.append(
                {
                    "center_xyz": [round(float(v), 4) for v in center],
                    "pixel_count": len(pixels),
                    "bbox_xyz": {
                        "min": [round(float(v), 4) for v in lo],
                        "max": [round(float(v), 4) for v in hi],
                    },
                    "sample_pixels": [list(pixels[len(pixels) // 2])],
                }
            )
        clusters.sort(key=lambda c: -c["pixel_count"])
        return {
            "clusters": clusters[:20],
            "summary": {"total_clusters": len(clusters[:20]), "total_pixels_matched": int(len(ys))},
        }

    def _record(self, command: dict[str, Any] | None, result: dict[str, Any] | None) -> None:
        self.steps.append(
            {
                "step_idx": len(self.steps),
                "state": {
                    "robot0_eef_pos": list(self.eef),
                    "robot0_eef_quat": self._eef_quat(),
                    "robot0_gripper_qpos": list(self.qpos),
                    "robot0_base_pos": list(self.base),
                    "robot0_base_quat": list(self.base_quat),
                },
                "success": self._success(),
                "task_progress": {"success": self._success(), "joint_p": round(self.joint_p, 4)},
                "vla_desync": self.vla_desync,
                "truncated": self.truncate_next,
                "command": command,
                "result": result,
            }
        )

    def _view(self, step: int) -> dict[str, Any]:
        record = self.steps[step]
        out = {
            "step": record["step_idx"],
            "task_progress": dict(record["task_progress"]),
            "task_language": self.TASK,
            "state": json.loads(json.dumps(record["state"])),
            "robocasa_terminated": record["success"],
            "vla_desync": record["vla_desync"],
            "success": record["success"],
            "log": {
                "command": record["command"],
                "result": record["result"],
                "elapsed_s": 0.5 if record["command"] else None,
            },
            "images": [
                {"role": "calibration_frame", "camera": "agentview", "artifact": "agentview.png"},
                {"role": "nav_view", "camera": "navview", "artifact": "navview.png"},
                {"role": "calibration_frame", "camera": "wrist", "artifact": "wrist.png"},
            ],
            "_image_cam_bytes": b"png",
            "_image_nav_bytes": b"png",
            "_image_wrist_bytes": b"png",
        }
        if record["truncated"]:  # RPent never sets it for this robot; exercises the base path
            out["truncated"] = True
        return out


@pytest.fixture
def rc_backend() -> FakeRobocasaBackend:
    return FakeRobocasaBackend()


@pytest.fixture
def rc(rc_backend) -> RobocasaRobot:
    return RobocasaRobot(rc_backend)


# -- state ---------------------------------------------------------------------


def test_state_reads_the_envelope(rc, rc_backend):
    state = rc.state()
    assert isinstance(state, State)
    assert state.task == rc_backend.TASK and rc.task == rc_backend.TASK
    assert state.step == 0 and state.eef_pos == (1.453, -0.677, 1.306)
    assert state.eef_quat == (1.0, 0.0, 0.0, pytest.approx(0.0, abs=1e-12))
    assert state.eef_approach == pytest.approx((0.0, 0.0, -1.0))
    assert state.eef_tilt == pytest.approx(0.0, abs=1e-9)
    assert state.gripper_qpos == (0.0206, -0.0205)
    assert state.gripper_opening == pytest.approx(0.0411)
    assert state.base_pos == (1.432, -0.916, 0.7) and state.base_yaw == pytest.approx(math.pi / 2)
    assert state.task_progress == {"success": False, "joint_p": 0.001}
    assert state.vla_desync is True and state.success is False
    assert not state.terminated and not state.truncated and not rc.done


def test_state_properties_on_the_recorded_smoke_pose():
    """The step-0 pose of runs/smoke-robocasa-OpenDrawer_s1: down, tilted 0.33 rad towards +y."""
    state = State(
        step=0,
        task="Open the left drawer.",
        eef_pos=(1.453, -0.677, 1.306),
        eef_quat=(0.6906, 0.7039, 0.0948, -0.1363),
        gripper_qpos=(0.0206, -0.0205),
        gripper_opening=0.0411,
        base_pos=(1.432, -0.916, 0.7),
        base_quat=(0.0, 0.0, 0.7071068, 0.7071067),
        success=False,
        task_progress={},
        vla_desync=True,
        terminated=False,
        truncated=False,
    )
    ax, ay, az = state.eef_approach
    assert az < -0.9 and ay > 0.3 and abs(ax) < 0.1
    assert state.eef_tilt == pytest.approx(0.334, abs=0.01)
    assert state.base_yaw == pytest.approx(math.pi / 2, abs=1e-6)


def test_live_envelopes_may_carry_numpy_arrays(rc_backend):
    original_view = rc_backend._view

    def numpy_view(step):
        envelope = original_view(step)
        raw = envelope["state"]
        for key, value in list(raw.items()):
            raw[key] = np.asarray(value, dtype=np.float64)
        return envelope

    rc_backend._view = numpy_view
    robo = RobocasaRobot(rc_backend)
    state = robo.state()
    assert state.eef_pos == (1.453, -0.677, 1.306)
    assert state.base_quat[2] == pytest.approx(0.7071, abs=1e-4)
    assert robo.move_to([1.30, -0.60, 1.10]).reached
    json.dumps(robo._summary_dict())


def test_summary_dict_is_compact_and_json_friendly(rc):
    summary = rc._summary_dict()
    assert summary["step"] == 0 and summary["success"] is False
    assert summary["base_xy"] == [1.432, -0.916] and summary["base_yaw"] == pytest.approx(1.571)
    assert summary["task_progress"] == {"success": False, "joint_p": 0.001}
    assert set(summary) >= {"eef_pos", "eef_tilt", "gripper_opening", "vla_desync"}
    json.dumps(summary)


# -- scripted arm motion ----------------------------------------------------------


def test_move_to_passes_rpent_argument_names_and_returns_a_typed_result(rc, rc_backend):
    result = rc.move_to([1.30, -0.60, 1.10])
    assert isinstance(result, Move)
    assert result.reached and result.final_dist_m == 0.0 and result.steps_used == 14
    assert result.final_eef_pos == (1.3, -0.6, 1.1) and result.max_steps == 200
    assert result.gripper_q0 == 0.0206 and result.terminated is False
    name, kwargs = rc_backend.calls[-1]
    assert name == "move_to"
    assert kwargs == {
        "xyz": [1.3, -0.6, 1.1],
        "gripper": "hold",
        "step_clip": 0.02,
        "max_steps": 200,
        "tol": 0.012,
    }
    record = rc._ledger.records[-1]
    assert record.stateful and record.env_steps == 9 + 14  # jacobian calibration + servo steps
    assert record.summary["ok"] is True and record.summary["success"] is False
    rc.move_to([1.30, -0.60, 1.00], gripper=RobocasaRobot.OPEN, step_clip=0.012, tol=0.01)
    kwargs = rc_backend.calls[-1][1]
    assert kwargs["gripper"] == -1.0 and kwargs["step_clip"] == 0.012 and kwargs["tol"] == 0.01
    assert rc.state().gripper_opening == pytest.approx(0.08)
    rc.move_to([1.30, -0.60, 1.05], gripper=RobocasaRobot.CLOSE)
    assert rc_backend.calls[-1][1]["gripper"] == 1.0 and rc.state().gripper_opening == 0.0


def test_a_stalled_servo_is_a_measured_outcome_not_an_exception(rc):
    rc.move_to([1.30, -0.60, 1.10])
    rc.move_to([1.30, -0.60, 1.00])
    result = rc.move_to([1.30, -0.60, 0.74])  # below the counter top
    assert result.reached is False and result.steps_used == 200
    assert result.final_eef_pos[2] == pytest.approx(0.80)
    assert result.final_dist_m == pytest.approx(0.06)


def test_moves_over_0_30_m_and_bad_arguments_are_refused_before_dispatch(rc, rc_backend):
    before = len(rc_backend.calls)
    with pytest.raises(ValueError, match="0.3"):
        rc.move_to([0.9, -0.6, 0.9])
    with pytest.raises(ValueError, match="0.3"):
        rc.move_delta([0.0, 0.0, -0.35])
    with pytest.raises(ValueError, match="0.3"):
        rc.scripted_grasp([0.9, -0.6, 0.8])
    with pytest.raises(ValueError):
        rc.move_to([1.3, -0.6])
    with pytest.raises(ValueError, match="hold"):
        rc.move_to([1.3, -0.6, 1.1], gripper="open")
    with pytest.raises(ValueError):
        rc.move_to([1.3, -0.6, 1.1], gripper=2.0)
    with pytest.raises(ValueError):
        rc.move_to([1.3, -0.6, 1.1], step_clip=0.0)
    with pytest.raises(ValueError):
        rc.move_to([1.3, -0.6, 1.1], max_steps=0)
    with pytest.raises(ValueError):
        rc.rotate_pitch(0.3, gripper="hold")
    with pytest.raises(ValueError):
        rc.set_gripper("hold")
    with pytest.raises(ValueError):
        rc.set_gripper(1.0, steps=0)
    with pytest.raises(ValueError):
        rc.release(steps=0)
    with pytest.raises(ValueError):
        rc.navigate_to([1.0])
    with pytest.raises(ValueError):
        rc.navigate_to([1.0, 0.0], tol=0.0)
    with pytest.raises(ValueError):
        rc.move_base(forward=math.nan)
    with pytest.raises(ValueError):
        rc.move_base(steps=0)
    with pytest.raises(ValueError, match="max_chunks"):
        rc.rldx_skill(max_chunks=0)
    with pytest.raises(ValueError, match="base_clip"):
        rc.rldx_arm(base_clip=-0.1)
    with pytest.raises(ValueError, match="settle_eps"):
        rc.rldx_skill(settle_eps=0.0)
    with pytest.raises(ValueError):
        rc.back_project_batch([[1, 2, 3]])
    with pytest.raises(ValueError):
        rc.back_project_batch([[1, 2]] * 51)
    with pytest.raises(ValueError):
        rc.back_project_batch([[1, 2]], camera="rear")
    with pytest.raises(ValueError):
        rc.back_project_batch([[1, 2]], resolution="high")
    with pytest.raises(ValueError):
        rc.query_world_map(0.9, 0.8)
    with pytest.raises(ValueError):
        rc.query_world_map(camera="rear")
    with pytest.raises(ValueError):
        rc.query_world_map(x_range=[1.0])
    # only the first reach check read the state; nothing was dispatched
    assert [c[0] for c in rc_backend.calls[before:]] == ["view_env_state"]


def test_move_delta_is_relative_to_the_current_position(rc, rc_backend):
    result = rc.move_delta([0.0, 0.0, -0.10])
    assert isinstance(result, Move) and result.reached
    assert result.target_xyz == pytest.approx((1.453, -0.677, 1.206))
    assert result.final_eef_pos == pytest.approx((1.453, -0.677, 1.206))
    assert result.max_steps == 80 and result.steps_used == 6
    assert rc_backend.calls[-1][1] == {
        "dxyz": [0.0, 0.0, -0.1],
        "gripper": "hold",
        "step_clip": 0.02,
        "max_steps": 80,
    }


def test_rotate_pitch_is_relative_and_reports_no_angle(rc, rc_backend):
    result = rc.rotate_pitch()
    assert isinstance(result, Pitch) and result.target_pitch == 0.6 and result.steps_used == 12
    assert result.final_eef_pos == (1.453, -0.677, 1.306)
    assert rc_backend.calls[-1][1] == {"target_pitch": 0.6, "gripper": 1.0, "n": 12}
    assert rc.state().eef_tilt == pytest.approx(0.6)
    rc.rotate_pitch(-0.2, gripper=-1.0, n=4)
    assert rc.state().eef_tilt == pytest.approx(0.4) and rc.state().gripper_opening == 0.08
    assert [r for r in rc._ledger.records if r.tool == "rotate_pitch"][-1].env_steps == 4


def test_gripper_commands(rc, rc_backend):
    closed = rc.set_gripper(RobocasaRobot.CLOSE, steps=8)
    assert isinstance(closed, Gripper) and closed.gripper == 1.0 and closed.steps == 8
    assert closed.gripper_qpos == (0.0, 0.0) and closed.gripper_opening == 0.0
    assert rc_backend.calls[-1] == ("set_gripper", {"gripper": 1.0, "steps": 8})
    opened = rc.release()
    assert opened.gripper == -1.0 and opened.gripper_opening == pytest.approx(0.08)
    assert rc_backend.calls[-1] == ("release", {"steps": 10})
    assert rc._ledger.env_steps() == rc_backend.frames == 18


def test_scripted_grasp_success_and_the_stage_that_stalled(rc, rc_backend):
    rc.move_to([1.40, -0.60, 1.10])
    frames = rc_backend.frames
    grasp = rc.scripted_grasp(rc_backend.CAN)
    assert isinstance(grasp, Grasp) and grasp.ok and grasp.stage is None
    assert grasp.gripper_opening == pytest.approx(0.024)  # fingers stopped on the can
    assert grasp.final_eef_pos == pytest.approx((1.40, -0.60, 1.10)) and grasp.final_dist_m == 0.0
    assert rc_backend.calls[-1][1] == {
        "xyz": [1.4, -0.6, 0.95],
        "approach_z": 0.1,
        "grasp_z_offset": 0.0,
        "step_clip": 0.02,
    }
    assert rc_backend.frames - frames == 4 + 3 + 9 + 14 + 11
    rc.move_to([1.40, -0.60, 1.05])
    failed = rc.scripted_grasp([1.40, -0.60, 0.70])  # descent target below the counter
    assert failed.ok is False and failed.stage == "descent"
    assert failed.final_dist_m == pytest.approx(0.10) and failed.final_eef_pos[2] == pytest.approx(
        0.8
    )


# -- base motion ------------------------------------------------------------------


def test_navigate_to_drives_the_base_and_moves_the_arm_with_it(rc, rc_backend):
    result = rc.navigate_to([1.432, 0.0], tol=0.3)
    assert isinstance(result, Navigation) and result.reached and not result.stuck
    assert result.steps_used == 269 and result.max_steps == 300
    assert result.moved_m == pytest.approx(0.617, abs=1e-3)
    assert result.final_dist_m == pytest.approx(0.299, abs=1e-3)
    assert result.start_xy == (1.432, -0.916) and result.base_pos[1] == pytest.approx(
        -0.299, abs=1e-3
    )
    assert rc_backend.calls[-1][1] == {
        "xy": [1.432, 0.0],
        "tol": 0.3,
        "max_steps": 300,
        "gripper": "hold",
    }
    state = rc.state()
    assert state.eef_pos[1] == pytest.approx(-0.677 + 0.617, abs=1e-3)  # the arm rode along
    assert state.base_yaw == pytest.approx(math.pi / 2) and state.vla_desync
    drive = [r for r in rc._ledger.records if r.tool == "navigate_to"][-1]
    assert drive.env_steps == 6 + 269  # heading calibration + drive
    rc.move_to([1.432, -0.06 + 0.2, 1.10])
    assert rc._ledger.records[-1].env_steps >= 9  # the arm jacobian is recalibrated after driving
    stuck = rc.navigate_to([0.3, -0.9], max_steps=100)
    assert stuck.reached is False and stuck.stuck and stuck.steps_used == 100
    assert stuck.moved_m == pytest.approx(0.05)
    assert rc.navigate_to([1.432, 0.1, 5.0], tol=0.5).target_xy == (1.432, 0.1)  # z ignored


def test_move_base_reports_the_measured_displacement(rc, rc_backend):
    result = rc.move_base(forward=0.4, steps=10)
    assert isinstance(result, BaseMove) and result.steps == 10
    assert result.base_moved == pytest.approx((0.0, 0.0092, 0.0), abs=1e-6)
    assert result.base_pos[1] == pytest.approx(-0.916 + 0.0092)
    assert rc_backend.calls[-1][1] == {
        "forward": 0.4,
        "lateral": 0.0,
        "turn": 0.0,
        "steps": 10,
        "gripper": "hold",
    }
    rc.move_base(turn=0.3, steps=5, gripper=RobocasaRobot.CLOSE)
    assert rc_backend.calls[-1][1]["gripper"] == 1.0
    assert rc.state().base_yaw == pytest.approx(math.pi / 2 + 0.015)


# -- the VLA and the episode end --------------------------------------------------


def test_the_vla_solves_the_task_and_success_ends_the_episode(rc, rc_backend):
    skill = rc.rldx_skill()
    assert isinstance(skill, Skill)
    assert skill.status == "success" and skill.chunks_used == 26 and skill.steps_applied == 208
    assert skill.max_chunks == 40  # the protocol's cap, not the argument
    assert skill.instruction == rc_backend.TASK and skill.terminated
    assert skill.held_apart and skill.grasp_detected and not skill.grasped
    assert skill.base_clip is None and skill.base_drift_m == 0.003 and skill.peak_lift_m == 0.05
    name, kwargs = rc_backend.calls[-1]
    assert name == "rldx_skill"
    assert kwargs == {
        "prompt": rc_backend.TASK,  # the library sends the task language, like RPent's agent
        "base_clip": None,
        "max_chunks": 70,
        "force_reset": False,
        "n_action_steps": 8,
        "settle_patience": 999,
        "settle_eps": 0.012,
    }
    assert rc.done and rc_backend.solved()
    state = rc.state()
    assert state.success and state.terminated and not state.vla_desync
    assert state.task_progress == {"success": True, "joint_p": 0.9801}
    with pytest.raises(EpisodeFinished) as info:
        rc.move_to([1.0, -0.9, 0.9])
    assert info.value.reason == "terminated"
    assert rc.state().success  # read-only calls still work
    assert rc.back_project_batch([[203, 60]]).valid_count == 1
    record = rc._ledger.records[1]
    assert record.tool == "rldx_skill" and record.env_steps == 208
    assert record.summary["status"] == "success" and record.summary["chunks"] == 26
    assert record.summary["success"] is True


def test_the_vla_caps_when_the_base_is_far_and_arm_mode_clamps_the_base(rc, rc_backend):
    rc.navigate_to([1.432, 0.9], tol=0.3, max_steps=1000)  # more than 0.8 m from the handle
    skill = rc.rldx_arm()
    assert skill.status == "cap" and skill.chunks_used == 40 and skill.steps_applied == 320
    assert skill.base_clip == 0.1 and not skill.terminated and not rc.done
    assert rc_backend.calls[-1][0] == "rldx_arm" and rc_backend.calls[-1][1]["base_clip"] == 0.1
    again = rc.rldx_arm(max_chunks=3, force_reset=True)
    assert again.chunks_used == 3 and again.max_chunks == 3
    assert rc_backend.calls[-1][1]["force_reset"] is True
    assert not rc.state().vla_desync


def test_vla_refusals_and_backend_errors_become_tool_errors(rc, rc_backend):
    rc_backend.error_next = {"error": "max_chunks must be positive; VLA was not executed"}
    with pytest.raises(ToolError, match="VLA was not executed") as info:
        rc.rldx_skill()
    assert info.value.tool == "rldx_skill" and rc._ledger.records[-1].ok is False
    rc_backend.error_next = {"error": "simulated failure", "traceback": "..."}
    with pytest.raises(ToolError, match="simulated failure"):
        rc.set_gripper(1.0)


def test_state_capture_failure_is_an_infrastructure_error_not_a_step(rc, rc_backend):
    rc.state()
    steps_before = len(rc_backend.steps)
    rc_backend.capture_error_next = True
    with pytest.raises(ToolError, match="failed to capture state") as info:
        rc.move_to([1.30, -0.60, 1.10])
    assert info.value.payload["state_capture_error"] == "disk full"
    assert len(rc_backend.steps) == steps_before and rc.state().step == 0
    assert [r for r in rc._ledger.records if r.tool == "move_to"][-1].ok is False
    assert not rc.done


def test_a_reported_truncation_latches_like_success(rc, rc_backend):
    rc_backend.truncate_next = True
    result = rc.set_gripper(1.0)
    assert isinstance(result, Gripper) and rc.state().truncated and not rc.state().terminated
    with pytest.raises(EpisodeFinished) as info:
        rc.set_gripper(-1.0)
    assert info.value.reason == "truncated"


def test_cancellation_surfaces_as_episode_finished(rc, rc_backend):
    rc._stop_episode("timeout")
    with pytest.raises(EpisodeFinished) as info:
        rc.rldx_skill()
    assert info.value.reason == "timeout" and rc_backend.cancelled


# -- perception ------------------------------------------------------------------


def test_back_project_batch_returns_typed_points_and_a_median(rc, rc_backend):
    pixels = [[203, 60], [204, 65], [203, 70], [202, 76], [205, 68]]
    result = rc.back_project_batch(pixels)
    assert isinstance(result, BackProjection) and len(result.points) == 5
    assert result.valid_count == 5 and result.total_count == 5 and result.step == 0
    assert result.median_xyz == pytest.approx(rc_backend.HANDLE, abs=0.02)
    assert all(p.valid for p in result.points) and len(result.xyz) == 5
    assert result.points[0].pixel == (203, 60) and result.camera == "agentview"
    assert rc_backend.calls[-1][1] == {
        "pixels": pixels,
        "step": -1,
        "camera": "agentview",
        "resolution": "low",
    }
    mixed = rc.back_project_batch([[300, 5], [2, 2], [130, 160]], camera="agentview", step=0)
    assert mixed.valid_count == 1 and mixed.median_xyz == pytest.approx(rc_backend.CAN, abs=1e-4)
    assert mixed.points[0].valid is False and "out of bounds" in mixed.points[0].error
    assert mixed.points[1].world_xyz is None and "invalid" in mixed.points[1].error
    single = rc.back_project_batch((130, 160), camera="wrist")
    assert single.points[0].pixel == (130, 160) and single.camera == "wrist"
    assert rc._ledger.records[-1].summary["median_xyz"] == pytest.approx(single.median_xyz)
    with pytest.raises(ToolError, match="not available"):
        rc.back_project_batch([[1, 1]], step=7)


def test_query_world_map_finds_height_bands_on_the_latest_map(rc, rc_backend):
    handles = rc.query_world_map(0.75, 0.80)
    assert isinstance(handles, HeightQuery) and handles.clusters and handles.total_pixels > 0
    biggest = handles.clusters[0]
    assert biggest.center_xyz[2] == pytest.approx(rc_backend.HANDLE[2])
    assert biggest.center_xyz[:2] == pytest.approx(rc_backend.HANDLE[:2], abs=0.05)
    assert biggest.bbox_min[2] <= biggest.center_xyz[2] <= biggest.bbox_max[2]
    assert 195 <= biggest.sample_pixel[0] <= 215 and biggest.pixel_count >= 10
    assert rc_backend.calls[-1][1] == {
        "z_min": 0.75,
        "z_max": 0.8,
        "x_range": None,
        "y_range": None,
        "camera": "agentview",
        "resolution": "low",
        "min_cluster_size": 10,
    }
    floor = rc.query_world_map(0.0, 0.12, camera="navview", x_range=[0.5, 2.5], y_range=[-1, 1])
    assert floor.clusters and all(c.center_xyz[2] == 0.0 for c in floor.clusters)
    assert rc_backend.calls[-1][1]["x_range"] == [0.5, 2.5]
    assert rc._ledger.records[-1].summary["center_xyz"] == list(floor.clusters[0].center_xyz)
    nothing = rc.query_world_map(5.0, 6.0)
    assert nothing.clusters == () and nothing.total_pixels == 0


def test_images_world_maps_depth_calibration_and_criteria_decode(rc):
    pytest.importorskip("PIL")
    image = rc.image()
    assert image.shape == (256, 256, 3) and image.dtype == np.uint8
    assert rc.image("navview").shape == (256, 256, 3) and rc.image("wrist").shape == (256, 256, 3)
    assert rc.floor_overlay().shape == (256, 256, 3)
    world = rc.world_map("agentview")
    assert world.shape == (256, 256, 3) and world.dtype == np.float32
    assert float(np.median(world[100:200, 100:140, 2])) == pytest.approx(0.90)
    assert rc.world_map("navview").shape == (256, 256, 3)
    assert rc.depth("wrist").shape == (256, 256) and rc.depth().dtype == np.float32
    assert rc.camera_meta("wrist")["camera_name"] == "wrist" and rc.camera_meta()["width"] == 256
    assert "_check_success" in rc.success_criteria()
    record = rc._ledger.records[-1]
    assert record.tool == "artifact" and record.kwargs == {
        "name": "success_criteria.md",
        "step": None,
    }
    with pytest.raises(ValueError):
        rc.image("rear")
    with pytest.raises(ValueError):
        rc.image("agentview", resolution="high")
    with pytest.raises(ValueError):
        rc.world_map("agentview", resolution="high")
    with pytest.raises(ValueError):
        rc.depth("navview")
    with pytest.raises(ValueError):
        rc.camera_meta("navview")
    with pytest.raises(ValueError):
        rc.show("rear")


def test_world_maps_expire_after_25_steps(rc):
    for _ in range(26):
        rc.set_gripper(1.0, steps=1)
    assert rc.state().step == 26
    assert rc.world_map("agentview", step=2).shape == (256, 256, 3)
    with pytest.raises(ToolError, match="older than 25"):
        rc.world_map("agentview", step=1)
    with pytest.raises(ToolError, match="pruned"):
        rc.back_project_batch([[130, 160]], step=1)
    assert rc.back_project_batch([[130, 160]]).valid_count == 1


def test_ledger_records_every_call_with_env_steps(rc, rc_backend):
    rc.state()
    rc.move_to([1.30, -0.60, 1.10])
    rc.back_project_batch([[203, 60]])
    rc.rldx_skill()
    records = rc._ledger.records
    assert [r.tool for r in records] == [
        "view_env_state",
        "move_to",
        "back_project_batch",
        "rldx_skill",
    ]
    assert records[0].stateful is False and records[0].summary["task_progress"]["joint_p"] == 0.001
    assert records[0].summary["success"] is False and records[0].summary["vla_desync"] is True
    assert "_image_cam_bytes" not in json.dumps(records[0].to_dict())
    assert records[1].stateful and records[1].env_steps == 23
    assert records[2].summary["valid_count"] == 1
    assert records[3].env_steps == 208 and records[3].summary["steps_applied"] == 208
    assert rc._ledger.env_steps() == rc_backend.frames == 23 + 208


# -- prompt, adapter, arm toolkit ------------------------------------------------


def test_api_reference_lists_every_public_method_and_result_type():
    text = api_reference(RobocasaRobot)
    for name in _public_members(RobocasaRobot):
        assert f"robo.{name}" in text, name
    for name in (
        "state",
        "image",
        "world_map",
        "depth",
        "floor_overlay",
        "camera_meta",
        "success_criteria",
        "back_project_batch",
        "query_world_map",
        "rldx_skill",
        "rldx_arm",
        "move_to",
        "move_delta",
        "rotate_pitch",
        "set_gripper",
        "release",
        "scripted_grasp",
        "navigate_to",
        "move_base",
    ):
        assert f"robo.{name}(" in text, name
    for cls in RobocasaRobot.RESULT_TYPES:
        assert f"class {cls.__name__}(" in text
    assert "robo.OPEN = -1.0" in text and "robo.CLOSE = 1.0" in text
    assert "robo.HOLD = 'hold'" in text and "robo.MAX_MOVE_M = 0.3" in text
    for hidden in ("RESULT_TYPES", "SUMMARY_KEYS", "CAMERAS", "DEPTH_ARTIFACTS", "robo._"):
        assert hidden not in text, hidden
    assert "_state_from_envelope" not in text and "_summary_dict" not in text
    assert "[x, y, z, w]" in text and "0.30 m" in text
    assert (
        "robo.move_to(\n    xyz: Sequence[float],\n    gripper: float | str = 'hold',\n    *,\n"
        "    step_clip: float = 0.02,\n    max_steps: int = 200,\n    tol: float = 0.012,\n"
        ") -> Move"
    ) in text
    assert "max_chunks: int = 70" in text and "base_clip: float | None = 0.1" in text
    assert "tol: float = 0.2" in text and "max_steps: int = 300" in text
    assert (
        "robo.rotate_pitch(\n    target_pitch: float = 0.6,\n    *,\n    gripper: float = 1.0,\n"
        "    n: int = 12,\n) -> Pitch"
    ) in text
    assert (
        ".base_yaw -> float" in text and ".eef_tilt -> float" in text and ".valid -> bool" in text
    )


def test_adapter_and_prompt_wording():
    adapter = get_adapter("robocasa")
    assert adapter.robot_cls is RobocasaRobot and adapter.knowledge == "robocasa"
    assert adapter.cameras == ("agentview", "navview", "wrist")
    assert adapter.host == "pyrualean.hosts.rpent_robocasa"
    assert adapter.guides_subdir == "" and adapter.guide_names == ()
    card = TaskCard(
        robot="robocasa",
        suite="target",
        task="OpenDrawer",
        seed=1,
        task_language="Open the left drawer.",
        object_names=[],
        facts={"initial base position [x, y, z]": "[1.432, -0.916, 0.700] m"},
    )
    bundle = render_prompt(card, budget_s=600)
    assert "a mobile-base PandaOmron robot in the RoboCasa365 kitchen benchmark" in bundle.system
    assert "class RobocasaRobot" in bundle.system
    assert "# Operating knowledge for RoboCasa365" in bundle.system
    know = knowledge_text("robocasa")
    for banned in ("memory", "recipe", "audit", "launch", "server", "task_only", "explore"):
        assert banned not in know.lower(), banned
    for kept in (
        "## Gripper",
        "## Navigation",
        "## The VLA",
        "## Perception",
        "## Observable gates",
    ):
        assert kept in know
    assert "loop inside one cell" in know and "vla_desync" in know
    assert "- robot: robocasa" in bundle.user and "initial base position" in bundle.user
    cells = build_prompt(
        "cells", card, budget_s=600, max_programs=None, images="on-demand", feedback="pure"
    )
    assert "`RobocasaRobot` bound to the live simulator, a\nfrozen RLDX-1 VLA" in cells
    assert "`robo.show('agentview')`, `robo.show('navview')` or `robo.show('wrist')`" in cells


def test_arm_toolkit_attaches_the_three_views(rc_backend, tmp_path):
    pytest.importorskip("PIL")
    robo = RobocasaRobot(rc_backend)
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", images="on-motion")
    moved = kit.execute_tool("python", {"code": "robo.move_to([1.30, -0.60, 1.10])"})
    assert moved.result["moved"] is True and moved.result["env_steps"] == 23
    assert moved.result["images"] == ["agentview.png", "navview.png", "wrist.png"]
    assert len(moved.images) == 3 and moved.result["state"]["success"] is False
    assert moved.result["state"]["task_progress"] == {"success": False, "joint_p": 0.001}
    demand = ArmToolkit(
        RobocasaRobot(rc_backend),
        arm="cells",
        workspace=tmp_path / "ws2",
        images="on-demand",
        feedback="pure",
    )
    looked = demand.execute_tool("python", {"code": "robo.show('navview'); print(robo.task)"})
    assert looked.result["images"] == ["navview.png"] and len(looked.images) == 1
    assert looked.result["stdout"].strip() == rc_backend.TASK and "state" not in looked.result


# -- host --------------------------------------------------------------------------


class _FakeState:
    """Enough of RPent's ``EnvState`` for ``dump_card``: artifact paths."""

    def __init__(self, backend: FakeRobocasaBackend, root: Path) -> None:
        self.backend = backend
        self.root = root

    def artifact_path(self, name: str, *, step: int = -1) -> Path:
        suffix = Path(name).suffix
        path = self.root / name / f"{step:02d}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.backend.artifact(name, step))
        return path


class _FakeToolkit:
    def __init__(self, backend: FakeRobocasaBackend, root: Path) -> None:
        self.backend = backend
        self.state = _FakeState(backend, root)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        return self.backend.execute_tool(name, input_dict)


def test_cell_and_argument_helpers():
    cell = host.Cell("target", "OpenDrawer", 1, cuda_device=6)
    assert cell.tag == "OpenDrawer_target_s1" and cell.max_episode_steps == 10000
    args = Namespace(task="TurnOnMicrowave", suite=None, seed=3, max_episode_steps=500)
    assert host.cell_from_args(args) == host.Cell("target", "TurnOnMicrowave", 3, 500, None)
    options = host.boot_options(
        Namespace(vla_endpoint="127.0.0.1:9000", vla_model_path="/m", robocasa_assets_path="/a")
    )
    assert options == {
        "vla_endpoint": "127.0.0.1:9000",
        "env_endpoint": None,
        "vla_model_path": "/m",
        "robocasa_assets_path": "/a",
        "memory_dir": None,
    }
    parser = argparse.ArgumentParser()
    host.add_cell_args(parser)
    host.add_boot_args(parser)
    ns = parser.parse_args(["--task", "OpenDrawer"])
    assert (
        ns.suite == "target"
        and ns.seed == 1
        and host.cell_from_args(ns).tag == "OpenDrawer_target_s1"
    )
    ns = parser.parse_args(["--task", "OpenCabinet", "--suite", "pretrain", "--seed", "7"])
    assert host.cell_from_args(ns) == host.Cell("pretrain", "OpenCabinet", 7)
    with pytest.raises(SystemExit):
        parser.parse_args(["--task", "OpenDrawer", "--suite", "random"])
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_memory_root_prefers_the_tracked_empty_corpus(tmp_path):
    tracked = host.EMPTY_MEMORY
    if (tracked / "MEMORY.md").is_file():
        assert host.memory_root(tmp_path) == tracked
    else:
        private = host.memory_root(tmp_path)
        assert private == tmp_path / "_memory" and (private / "MEMORY.md").is_file()
    with pytest.raises(FileNotFoundError):
        host.memory_root(tmp_path, str(tmp_path / "corpus"))
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "MEMORY.md").touch()
    assert host.memory_root(tmp_path, str(tmp_path / "corpus")) == (tmp_path / "corpus").resolve()


def test_dump_card_writes_json_images_and_facts(rc_backend, tmp_path):
    pytest.importorskip("PIL")
    toolkit = _FakeToolkit(rc_backend, tmp_path / "artifacts")
    cell = host.Cell("target", "OpenDrawer", 1, cuda_device=6)
    card = host.dump_card(toolkit, cell, tmp_path)
    assert (tmp_path / "card.json").is_file()
    assert set(card.images) == {"agentview", "navview", "wrist"}
    assert all(Path(p).is_file() for p in card.images.values())
    assert card.robot == "robocasa" and card.task == "OpenDrawer" and card.suite == "target"
    assert card.task_language == rc_backend.TASK and card.object_names == []
    assert card.eef_pos is None and card.eef_quat is None and card.gripper_opening is None
    facts = card.facts
    assert facts["initial base position [x, y, z]"] == "[1.432, -0.916, 0.700] m"
    assert facts["initial base heading (yaw about world z)"] == "1.571 rad (90.0 deg)"
    assert facts["initial gripper (EEF) position [x, y, z]"] == "[1.453, -0.677, 1.306] m"
    assert facts["initial gripper orientation [x, y, z, w]"] == "[1.000, 0.000, 0.000, 0.000]"
    assert facts["initial gripper opening (|q0| + |q1|)"].startswith("0.041 ")
    assert json.loads(facts["task_progress at step 0 (the success predicate's live values)"]) == {
        "joint_p": 0.001,
        "success": False,
    }
    text = card.render()
    assert "- robot: robocasa" in text and "- variant: target" in text
    assert "`robo.image('agentview')`" in text and "step 0 of your run" in text
    assert TaskCard.from_json(tmp_path / "card.json") == card
    assert rc_backend.calls[-1] == ("view_env_state", {"step": 0})
