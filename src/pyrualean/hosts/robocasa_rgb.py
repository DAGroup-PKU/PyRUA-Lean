# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The host of ``--robot robocasa_rgb``: RoboCasa365 with RoboDojo's primitives, colour only.

The kitchen runs in RoboCasa's own interpreter (:mod:`.robocasa_sim`, started here as a
subprocess and spoken to over a local socket); this process holds SAM3 and measures from the
colour images (:mod:`.robocasa_sight`).  :class:`RobocasaToolkit` is what the robot class
calls (``execute_tool``), records a step after every motion (the three images and each
camera's calibration), and alone asks the kitchen for the verdict, once the policy is done.

Needs: ``--sim-python`` (or ``ROBOCASA_SIM_PYTHON``), an interpreter with robocasa and
robosuite (RPent's RoboCasa venv); ``ROBOCASA_ASSETS_PATH``; ``SAM3_CHECKPOINT_PATH``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np

from ..prompt import TaskCard
from . import robocasa_sight as sight
from .robocasa_sim import CAMERAS, MOTIONS, unpack

ROBOT = "robocasa_rgb"
SIM_SCRIPT = Path(__file__).with_name("robocasa_sim.py")
BOOT_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class Cell:
    """One RoboCasa365 evaluation cell: a task, the ``target`` split and a scene seed.

    ``max_episode_steps`` is nominal: the budget is the task's official horizon, which
    ``dump_card`` reads from the kitchen.
    """

    suite: str
    task: str
    seed: int
    max_episode_steps: int = 10000
    cuda_device: int | None = None

    @property
    def tag(self) -> str:
        return f"{self.task}_{self.suite}_s{self.seed}"


def add_cell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, help="RoboCasa365 task name, e.g. OpenDrawer")
    parser.add_argument(
        "--suite", default="target", choices=("target",), help="RoboCasa data split"
    )
    parser.add_argument("--seed", type=int, default=1, help="scene seed (Target50: 1-10 atomic)")


def add_boot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sim-python",
        default=os.environ.get("ROBOCASA_SIM_PYTHON"),
        help="interpreter with robocasa + robosuite that runs the kitchen"
        " (default: $ROBOCASA_SIM_PYTHON)",
    )
    parser.add_argument(
        "--robocasa-assets-path",
        default=os.environ.get("ROBOCASA_ASSETS_PATH"),
        help="RoboCasa asset root (default: $ROBOCASA_ASSETS_PATH)",
    )


def boot_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sim_python": getattr(args, "sim_python", None),
        "robocasa_assets_path": getattr(args, "robocasa_assets_path", None),
    }


def cell_from_args(args: argparse.Namespace) -> Cell:
    return Cell(
        suite=str(getattr(args, "suite", None) or "target"),
        task=str(args.task),
        seed=int(args.seed),
        max_episode_steps=int(getattr(args, "max_episode_steps", 10000)),
        cuda_device=getattr(args, "cuda_device", None),
    )


# ---------------------------------------------------------------------------
# the kitchen process
# ---------------------------------------------------------------------------


class Kitchen:
    """The kitchen subprocess and the connection to it."""

    def __init__(self, proc: subprocess.Popen, conn: Any, log: Any) -> None:
        self.proc = proc
        self.conn = conn
        self.log = log
        self._lock = threading.Lock()

    def call(self, op: str, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            self.conn.send({"op": op, "kwargs": kwargs})
            reply = self.conn.recv()
        if not reply.get("ok"):
            raise RuntimeError(f"kitchen op {op!r} failed:\n{reply.get('error')}")
        return unpack(reply["result"])

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            with self._lock:
                self.conn.send({"op": "close"})
        with contextlib.suppress(Exception):
            self.conn.close()
        with contextlib.suppress(Exception):
            self.proc.terminate()
            self.proc.wait(timeout=20)
        with contextlib.suppress(Exception):
            self.proc.kill()
        with contextlib.suppress(Exception):
            self.log.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_kitchen(
    cell: Cell,
    out_dir: Path,
    sim_python: str | None = None,
    assets: str | None = None,
    timeout_s: float = BOOT_TIMEOUT_S,
) -> Kitchen:
    python = sim_python or os.environ.get("ROBOCASA_SIM_PYTHON")
    if not python:
        raise FileNotFoundError(
            "the kitchen's interpreter: pass --sim-python or set ROBOCASA_SIM_PYTHON"
        )
    port = _free_port()
    key = secrets.token_hex(16)
    env = dict(os.environ)
    # PYTHONHASHSEED=0 as PyRUA-Lean's RoboCasa host sets it: some kitchen choices follow string
    # hashing, so a (task, seed) scene is reproducible only with a fixed hash seed
    env.update(
        MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", ROBOCASA_SIM_AUTHKEY=key
    )
    if assets:
        env["ROBOCASA_ASSETS_PATH"] = str(Path(assets).expanduser())
    if cell.cuda_device is not None:
        env["MUJOCO_EGL_DEVICE_ID"] = str(cell.cuda_device)
    env.pop("PYTHONPATH", None)  # the kitchen's interpreter brings its own packages
    log = open(out_dir / "kitchen.log", "ab")  # noqa: SIM115 - lives with the process
    cmd = [python, str(SIM_SCRIPT), "--task", cell.task, "--split", cell.suite,
           "--seed", str(cell.seed), "--port", str(port)]  # fmt: skip
    proc = subprocess.Popen(  # noqa: S603 - our own script, fixed arguments
        cmd, stdout=subprocess.PIPE, stderr=log, env=env, start_new_session=True
    )
    deadline = time.time() + timeout_s
    assert proc.stdout is not None
    while True:
        if proc.poll() is not None:
            log.close()
            tail = (out_dir / "kitchen.log").read_text(errors="replace")[-3000:]
            raise RuntimeError(f"the kitchen process exited while booting:\n{tail}")
        if time.time() > deadline:
            proc.kill()
            raise TimeoutError("the kitchen did not boot in time")
        line = proc.stdout.readline().decode(errors="replace")
        if line.startswith("READY"):
            break
    conn = Client(("127.0.0.1", port), authkey=key.encode())
    return Kitchen(proc, conn, log)


# ---------------------------------------------------------------------------
# recorded steps
# ---------------------------------------------------------------------------


def _png(image: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image)[..., :3].astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _policy_camera(cam: dict[str, Any]) -> dict[str, Any]:
    """A camera's calibration as the policy reads it (RoboDojo's ``camera_meta``)."""

    pose = np.asarray(cam["cam_to_robot"], float) @ np.diag([1.0, -1.0, -1.0, 1.0])
    return {
        "shape": list(cam["shape"]),
        "intrinsic_matrix": np.asarray(cam["intrinsic"], float).tolist(),
        "extrinsic_matrix": pose.tolist(),
    }


class Steps:
    """The recorded steps of an episode: the state in memory, the images on disk."""

    def __init__(self, out_dir: Path) -> None:
        self.root = Path(out_dir) / "steps"
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    def record(self, obs: dict[str, Any]) -> int:
        step = len(self.records)
        folder = self.root / f"{step:04d}"
        folder.mkdir(exist_ok=True)
        cameras = obs.get("cameras") or {}
        for name, cam in cameras.items():
            (folder / f"{name}.png").write_bytes(_png(cam["color"]))
        status = obs.get("status") or {}
        self.records.append(
            {
                "state": obs.get("state") or {},
                "instruction": str(obs.get("instruction") or ""),
                "truncated": bool(status.get("over")),
                "status": dict(status),
                "cameras": {name: _policy_camera(cam) for name, cam in cameras.items()},
            }
        )
        return step

    def index(self, step: int) -> int:
        if not self.records:
            raise IndexError("no step recorded")
        index = len(self.records) + step if step < 0 else step
        if not 0 <= index < len(self.records):
            raise IndexError(f"step {step} is not recorded (0 .. {len(self.records) - 1})")
        return index

    def artifact_path(self, name: str, step: int = -1) -> Path:
        return self.root / f"{self.index(step):04d}" / name

    def load_bytes(self, name: str, step: int = -1) -> bytes:
        return self.artifact_path(name, step).read_bytes()


# ---------------------------------------------------------------------------
# the toolkit
# ---------------------------------------------------------------------------


class RobocasaToolkit:
    """The robot's tools: motions in the kitchen, perception here, a recorded step per motion."""

    def __init__(self, kitchen: Kitchen, sam3: Any, out_dir: Path) -> None:
        self.kitchen = kitchen
        self.sam3 = sam3
        self.state = Steps(out_dir)
        self.hello = kitchen.call("hello")
        self._obs = kitchen.call("observe", images=True)
        self._lock = threading.Lock()
        #: heights measured with two cameras and a measured support, by prompt
        self._heights: dict[str, float] = {}
        self.state.record(self._obs)

    # -- host interface -------------------------------------------------------

    def _status(self) -> dict[str, Any]:
        return dict((self.state.records[-1] if self.state.records else {}).get("status") or {})

    def solved(self) -> bool:
        return bool(self.kitchen.call("verdict").get("solved"))

    def episode_over(self) -> bool:
        return bool(self._status().get("over"))

    def verdict_record(self) -> dict[str, Any] | None:
        return self.kitchen.call("verdict")

    def conclude(self) -> None:
        """RoboCasa judges after every native action; nothing is left to judge at the end."""

    def cancel_active_and_wait(self) -> None:
        pass

    def close(self) -> None:
        self.kitchen.stop()

    # -- tools -----------------------------------------------------------------

    def _envelope(self, step: int, tool: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        index = self.state.index(step)
        record = self.state.records[index]
        return {
            "step": index,
            "task_language": record["instruction"],
            "terminated": False,
            "truncated": record["truncated"],
            "state": {
                "state": record["state"],
                "instruction": record["instruction"],
                "cameras": record["cameras"],
            },
            "log": {"tool": tool, "result": payload},
        }

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(input_dict or {})
        if name == "view_env_state":
            try:
                return self._envelope(int(kwargs.get("step", -1)), name, {})
            except IndexError as exc:
                return {"error": str(exc)}
        with self._lock:
            if name in MOTIONS:
                reply = self.kitchen.call(name, **kwargs)
                self._obs = reply["obs"]
                step = self.state.record(self._obs)
                return self._envelope(step, name, reply["payload"])
            handler = {
                "segment": self.segment,
                "locate": self.locate,
                "back_project": self.back_project,
            }.get(name)
            if handler is None:
                return {"error": f"this host has no tool {name!r}"}
            try:
                return handler(**kwargs)
            except TypeError as exc:
                return {"error": f"{name}: {exc}"}

    def _views(self) -> dict[str, sight.View]:
        return sight.views_from_cameras(self._obs["cameras"])

    def _need_sam(self) -> Any:
        if self.sam3 is None:
            raise RuntimeError("SAM3 is not loaded in this host (env_only boot)")
        return self.sam3

    def segment(self, text_prompt: str, camera: str = "agentview_left", min_score: float = 0.2):
        if camera not in CAMERAS:
            return {"error": f"camera must be one of {tuple(CAMERAS)}"}
        view = self._views()[camera]
        found = sight.candidates(self._need_sam(), view.rgb, str(text_prompt), float(min_score), 1)
        if not found:
            return {"camera": camera, "found": False, "text_prompt": text_prompt}
        mask, score = found[0]
        rows, cols = np.nonzero(mask)
        return {
            "camera": camera,
            "found": True,
            "score": round(score, 4),
            "box_px": [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())],
            "area_px": int(len(rows)),
            "centroid_rc": [int(rows.mean()), int(cols.mean())],
        }

    def locate(self, text_prompt: str, camera: str = "agentview_left", min_score: float = 0.2):
        if camera not in CAMERAS:
            return {"error": f"camera must be one of {tuple(CAMERAS)}"}
        prompt = str(text_prompt).strip()
        views = self._views()
        sam = self._need_sam()
        cands = {n: sight.candidates(sam, v.rgb, prompt, float(min_score)) for n, v in views.items()}
        if not cands[camera]:
            return {"camera": camera, "found": False, "text_prompt": prompt}
        mask, score = cands[camera][0]
        rows, cols = np.nonzero(mask)
        out: dict[str, Any] = {
            "camera": camera,
            "found": True,
            "score": round(score, 4),
            "centroid_rc": [int(rows.mean()), int(cols.mean())],
            "area_px": int(len(rows)),
        }
        on_robot = float((mask & views[camera].robot).sum()) / max(1, int(mask.sum()))
        if on_robot > 0.5:
            out["geometry_error"] = "the mask is the robot's own body: no position"
            return out
        masks = {n: [m for m, _ in c] for n, c in cands.items()}
        masks[camera] = [mask]
        m = sight.measure(views, camera, masks, known_height=self._heights.get(prompt))
        out.update({k: v for k, v in m.items() if k not in ("agreement", "hull_points", "hull_z")})
        if m.get("bottom_from") == "outlines" and "top_range" not in m and m.get("height_m"):
            self._heights[prompt] = float(m["height_m"])
        return out

    def back_project(self, row: int, col: int, camera: str = "agentview_left"):
        if camera not in CAMERAS:
            return {"error": f"camera must be one of {tuple(CAMERAS)}"}
        row, col = int(row), int(col)
        if not (0 <= row < 480 and 0 <= col < 640):
            return {"error": f"pixel ({row}, {col}) is outside the 480 x 640 image"}
        res = sight.probe(self._views(), self._need_sam(), camera, row, col)
        out = {"camera": camera, "pixel": [row, col], "views": res.get("views", 1)}
        if "xyz" in res:
            out.update(
                world_xyz=res["xyz"],
                depth_m=res["depth_m"],
                surface="object",
                normal=res.get("normal"),
            )
        else:
            out.update(world_xyz=[float("nan")] * 3, surface="unmeasured", why=res.get("why"))
        return out


class RobocasaBackend:
    """A :class:`RobocasaToolkit` as a :class:`pyrualean.Backend`."""

    def __init__(self, toolkit: RobocasaToolkit) -> None:
        self._toolkit = toolkit

    @property
    def toolkit(self) -> RobocasaToolkit:
        return self._toolkit

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        return self._toolkit.execute_tool(name, input_dict)

    def conclude(self) -> None:
        self._toolkit.conclude()

    def solved(self) -> bool:
        return self._toolkit.solved()

    def episode_over(self) -> bool:
        return self._toolkit.episode_over()

    def verdict_record(self) -> dict[str, Any] | None:
        return self._toolkit.verdict_record()

    def artifact(self, name: str, step: int = -1) -> bytes:
        return self._toolkit.state.load_bytes(name, step=step)

    def env_steps(self) -> int | None:
        step = self._toolkit._status().get("step")
        try:
            return int(step)
        except (TypeError, ValueError):
            return None

    def cancel(self) -> None:
        self._toolkit.cancel_active_and_wait()


class _Daemon:
    def __init__(self, kitchen: Kitchen) -> None:
        self.kitchen = kitchen

    def stop(self) -> None:
        self.kitchen.stop()


def boot(
    cell: Cell,
    out_dir: Path,
    *,
    rpent_root: str | None = None,  # noqa: ARG001 - the host interface; nothing of RPent's runs
    env_only: bool = False,
    sim_python: str | None = None,
    robocasa_assets_path: str | None = None,
    **_: Any,
) -> tuple[RobocasaToolkit, list[Any]]:
    """Start the kitchen and (unless ``env_only``) SAM3; return ``(toolkit, daemons)``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    kitchen = start_kitchen(cell, out_dir, sim_python, robocasa_assets_path)
    try:
        sam = None
        if not env_only:
            from . import sam3_local

            sam = sam3_local.shared(cuda_device=cell.cuda_device)
        toolkit = RobocasaToolkit(kitchen, sam, out_dir)
        (out_dir / "host.json").write_text(
            json.dumps(
                {"robot": ROBOT, "sim_python": sim_python, "hello": toolkit.hello}, indent=1
            )
            + "\n"
        )
    except BaseException:
        kitchen.stop()
        raise
    return toolkit, [_Daemon(kitchen)]


def make_backend(toolkit: RobocasaToolkit) -> RobocasaBackend:
    return RobocasaBackend(toolkit)


def _fmt(values: Any) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"


def dump_card(toolkit: RobocasaToolkit, cell: Cell, out_dir: Path, robot: str = ROBOT) -> TaskCard:
    """Write ``card.json`` plus the step-0 images and return the card (nothing privileged)."""

    record = toolkit.state.records[0]
    arm = (record["state"] or {}).get("arm") or {}
    card_dir = out_dir / "card"
    card_dir.mkdir(exist_ok=True)
    images: dict[str, str] = {}
    for camera in CAMERAS:
        source = toolkit.state.artifact_path(f"{camera}.png", step=0)
        if source.is_file():
            target = card_dir / f"{camera}.png"
            shutil.copyfile(source, target)
            images[camera] = str(target)
    budget = int(toolkit.hello.get("horizon") or cell.max_episode_steps)
    facts = {
        "arm initial grasp point [x, y, z]": f"{_fmt(arm.get('tcp_pos') or [])} m",
        "gripper initial pitch / yaw / roll": (
            f"{round(arm.get('pitch', 0))} / {round(arm.get('yaw', 0))} / "
            f"{round(arm.get('roll', 0))} deg"
        ),
        "gripper opening": f"{float(arm.get('gripper', 0)):.2f} (about 0.97 = open, 0 = closed)",
        "torso lift": f"{float((record['state'] or {}).get('torso_m', 0)):.2f} m (0 .. 0.34)",
        "native action budget": f"{budget} (RoboCasa's horizon for this task; a motion call "
        "spends one or more)",
    }
    card = TaskCard(
        suite=cell.suite,
        task=cell.task,
        seed=cell.seed,
        task_language=record["instruction"],
        object_names=[],
        images=images,
        max_env_steps=budget,
        robot=robot,
        facts=facts,
    )
    (out_dir / "card.json").write_text(card.to_json() + "\n", encoding="utf-8")
    return card


def main(argv: list[str] | None = None) -> int:
    """Boot one cell, print its card and a few primitive results (a smoke test)."""

    parser = argparse.ArgumentParser(description="RoboCasa365 colour-only host smoke test")
    add_cell_args(parser)
    add_boot_args(parser)
    parser.add_argument("--out", default="runs/robocasa_rgb_smoke")
    args = parser.parse_args(argv)
    cell = cell_from_args(args)
    toolkit, daemons = boot(cell, Path(args.out), **boot_options(args))
    try:
        print(dump_card(toolkit, cell, Path(args.out)).render())
    finally:
        for d in daemons:
            d.stop()
    return 0


__all__ = [
    "Cell",
    "RobocasaBackend",
    "RobocasaToolkit",
    "add_boot_args",
    "add_cell_args",
    "boot",
    "boot_options",
    "cell_from_args",
    "dump_card",
    "make_backend",
]


if __name__ == "__main__":
    sys.exit(main())
