"""A scripted LIBERO-shaped backend that mimics RPent's Toolkit contract."""

from __future__ import annotations

import io
import math
from typing import Any

import pytest


class _ToolResult:
    """Shape of ``rpent.tools.toolkit.ToolResult`` (only ``.result`` matters)."""

    def __init__(self, name: str, result: dict[str, Any]) -> None:
        self.name = name
        self.result = result


class FakeLiberoBackend:
    """One bowl, one plate.  Success = release while holding the bowl over the plate.

    Mirrors RPent's envelope: state-changing tools return a ``view_env_state``
    dict whose ``log.result`` holds the primitive's own return value; read-only
    tools return their payload directly.
    """

    TASK = "put the black bowl on the plate"
    BOWL = (0.05, -0.10, 0.44)
    PLATE = (0.05, 0.20, 0.42)

    def __init__(self) -> None:
        self.eef = [0.0, 0.0, 0.68]
        self.gripper = 0.08
        self.holding = False
        self.terminated = False
        self.truncated = False
        self._solved = False
        self.steps: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.frames = 0
        self.cancelled = False
        self.error_next: dict[str, Any] | None = None
        self._record(command=None, result=None)

    # -- Backend contract ---------------------------------------------------

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        self.calls.append((name, dict(input_dict)))
        if name in ("view_env_state", "view_camera_meta", "segment", "back_project"):
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
                result = {"error": f"bad arguments for {name}: {exc}"}
        self._record(command={"action": name, **input_dict}, result=result)
        envelope = self._view(-1)
        if result.get("interrupted") or "error" in result:
            for key, value in result.items():
                envelope.setdefault(key, value)
        return _ToolResult(name, envelope)

    def solved(self) -> bool:
        return self._solved

    def artifact(self, name: str, step: int = -1) -> bytes:
        if name.endswith(".png"):
            from PIL import Image

            size = 1024 if "high" in name else 256
            buf = io.BytesIO()
            Image.new("RGB", (size, size), (10, 20, 30)).save(buf, format="PNG")
            return buf.getvalue()
        if name.endswith(".npz"):
            import numpy as np

            size = 1024 if "high" in name else 256
            buf = io.BytesIO()
            np.savez(buf, array=np.zeros((size, size, 3), dtype=np.float32))
            return buf.getvalue()
        raise FileNotFoundError(name)

    def env_steps(self) -> int:
        return self.frames

    def cancel(self) -> None:
        self.cancelled = True

    # -- primitives -----------------------------------------------------------

    def _step(self, n: int) -> None:
        self.frames += n

    def _do_move_to(self, xyz, gripper=-1.0, tol=0.012, step_clip=0.025, max_steps=80, **_):
        target = [float(v) for v in xyz]
        dist = math.dist(self.eef, target)
        steps = min(max_steps, int(dist / step_clip) + 1)
        self._step(steps)
        if float(gripper) < 0:
            self.holding = False
            self.gripper = 0.08
        if dist > 0.6:  # unreachable: stall
            final_dist = dist - 0.3
        else:
            self.eef = target
            final_dist = 0.0
        return {
            "name": "move_to",
            "target_xyz": target,
            "final_eef_pos": list(self.eef),
            "final_dist_m": round(final_dist, 4),
            "steps_used": steps,
            "max_steps": max_steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def _do_pi0_pick(self, prompt, max_chunks=24, lift_thresh=0.05, **_):
        chunks = min(max_chunks, 4)
        self._step(chunks * 5)
        near = math.dist(self.eef[:2], self.BOWL[:2]) < 0.1
        if near:
            self.holding = True
            self.gripper = 0.03
            self.eef = [self.BOWL[0], self.BOWL[1], self.BOWL[2] + 0.1]
        return {
            "name": "pick",
            "instruction": prompt,
            "success": near,
            "chunks_used": chunks,
            "max_chunks": max_chunks,
            "peak_lift_m": 0.1 if near else 0.0,
            "min_gripper_opening": 0.03 if near else 0.0,
            "final_gripper_opening": self.gripper,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "diagnostics": {"descent_done": True},
        }

    def _do_set_gripper(self, gripper=-1.0, steps=5):
        self._step(steps)
        return {
            "name": "set_gripper",
            "gripper": float(gripper),
            "steps": steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def _do_release(self, max_steps=20):
        self._step(3)
        over_plate = math.dist(self.eef[:2], self.PLATE[:2]) < 0.05
        if self.holding and over_plate:
            self.terminated = True
            self._solved = True
        self.holding = False
        start = self.gripper
        self.gripper = 0.08
        return {
            "name": "release",
            "steps_used": 3,
            "start_gripper_opening": start,
            "peak_gripper_opening": 0.08,
            "final_gripper_opening": 0.08,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def _do_rotate_wrist(self, target_yaw=None, delta_yaw=None, gripper=1.0, **_):
        if target_yaw is None and delta_yaw is None:
            return {"name": "rotate_wrist", "error": "need target_yaw or delta_yaw"}
        self._step(4)
        target = target_yaw if target_yaw is not None else delta_yaw
        return {
            "name": "rotate_wrist",
            "start_yaw": 0.0,
            "target_yaw": target,
            "final_yaw": target,
            "final_err": 0.0,
            "steps_used": 4,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    # -- read-only ------------------------------------------------------------

    def _readonly(self, name: str, input_dict: dict[str, Any]) -> dict[str, Any]:
        if name == "view_env_state":
            return self._view(int(input_dict.get("step", -1)))
        if name == "view_camera_meta":
            return {"camera": input_dict.get("camera", "agentview"), "camera_meta": {"fx": 1.0}}
        if name == "segment":
            prompt = str(input_dict.get("prompt", "")).lower()
            if "bowl" in prompt:
                xyz, score = self.BOWL, 0.8
            elif "plate" in prompt:
                xyz, score = self.PLATE, 0.7
            else:
                return {
                    "found": False,
                    "step": 0,
                    "camera": "agentview",
                    "score": None,
                    "box": None,
                    "world_xyz": None,
                    "world_error": "no mask",
                    "error": "SAM3 found no mask",
                    "segment_artifact": "segment_00.json",
                }
            return {
                "found": True,
                "step": 0,
                "camera": input_dict.get("camera", "agentview"),
                "score": score,
                "box": [1, 2, 3, 4],
                "world_xyz": list(xyz),
                "world_error": None,
                "segment_artifact": "segment_00.json",
                "overlay_artifact": "segment_overlay_00.png",
            }
        if name == "back_project":
            if "row_range" in input_dict:
                return {
                    "mode": "region",
                    "center_xyz": list(self.PLATE),
                    "median_xyz": list(self.PLATE),
                    "n_valid": 100,
                    "camera": "agentview",
                    "step": 0,
                }
            row, col = int(input_dict["row"]), int(input_dict["col"])
            if not (0 <= row < 1024 and 0 <= col < 1024):
                return {"error": f"pixel ({row},{col}) out of bounds; agentview image is 1024x1024"}
            return {
                "camera": "agentview",
                "resolution": "high",
                "pixel": [row, col],
                "world_xyz": [row / 1000, col / 1000, 0.42],
                "step": 0,
                "image_size": [1024, 1024],
            }
        raise KeyError(name)

    def _record(self, *, command, result) -> None:
        self.steps.append(
            {
                "step_idx": len(self.steps),
                "state": {
                    "robot0_eef_pos": list(self.eef),
                    "robot0_eef_quat": [1.0, 0.0, 0.0, 0.0],
                    "robot0_gripper_qpos": [self.gripper / 2, -self.gripper / 2],
                    "object_names": ["akita_black_bowl_1", "plate_1"],
                },
                "terminated": self.terminated,
                "truncated": self.truncated,
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
            "state": dict(record["state"]),
            "artifacts": ["agentview_high.png", "wrist_high.png"],
            "task_language": self.TASK,
            "log": {"command": record["command"], "result": record["result"], "elapsed_s": 0.1},
            "_image_cam_bytes": b"png",
        }


@pytest.fixture
def backend() -> FakeLiberoBackend:
    return FakeLiberoBackend()


@pytest.fixture
def robo(backend):
    from pyrualean import LiberoRobot

    return LiberoRobot(backend)
