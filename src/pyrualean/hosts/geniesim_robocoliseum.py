# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The GenieSim benchmark as the environment this library's tools step.

The benchmark runs in its own process (Isaac Sim, its own Python, its own container) and
calls a policy over a WebSocket: it sends an observation and takes the action that comes
back.  This host is that policy: a :class:`~.geniesim_bridge.Server` (the benchmark dials
us) or a :class:`~.geniesim_bridge.TunnelClient` (we dial the online platform's gateway)
carries the requests, :class:`Toolkit` turns the robot's tools into actions, and the
benchmark drives the simulator in between.

This host never starts a simulator: GenieSim's own launcher, a cluster pod or the online
platform starts the run, and it connects.  ``boot`` serves the socket and waits for the
episode's first observation; :func:`dump_card` writes ``card.json`` and the first
observation's images, and the arm toolkit then runs a program against it.  A run plays
several episodes over one connection; a :class:`Session` serves all of them.

Step accounting: an ``EEF_ABS``/``JOINT_ABS`` reply of one action is one simulator step,
which is what the benchmark's own step budget (``StepOut``) counts.  ``Toolkit.env_steps``
reports the number of requests the benchmark has made, which is that count.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..prompt import TaskCard
from . import geniesim_bridge as bridge_mod
from . import robocoliseum_sight as sight
from . import robocoliseum_tools as tools

#: The three cameras the benchmark publishes.
CAMERAS = bridge_mod.CAMERAS
#: Step-0 artifacts copied next to the card, keyed by the name the card uses.
CARD_IMAGES = {camera: f"{camera}.png" for camera in CAMERAS}


@dataclass(frozen=True)
class Cell:
    """One benchmark task, as the card and ``play`` record it.

    suite: the benchmark platform (``g2_op`` for the G2), recorded in the card.
    task: the task's name, e.g. its config file stem ``g2op_if_pick_block_color``.
    seed: the run's ``--benchmark.seed``, recorded (the run is started elsewhere).
    max_episode_steps: the benchmark's own budget is in its task file (``StepOut``); this
        is what the card shows and what ``play`` records.
    """

    suite: str = "g2_op"
    task: str = "g2op_if_pick_block_color"
    seed: int = 1
    max_episode_steps: int = 200
    #: Each task stage's own action budget (its ``StepOut``), in order, when known.
    stage_budgets: tuple[int, ...] = ()
    cuda_device: int | None = None

    @property
    def tag(self) -> str:
        return f"{self.task}_s{self.seed}"


def add_cell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task", default="g2op_if_pick_block_color", help="benchmark config file stem"
    )
    parser.add_argument("--suite", default="g2_op", help="benchmark platform name")
    parser.add_argument("--seed", type=int, default=1, help="the run's --benchmark.seed (recorded)")


def add_boot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1", help="address the bridge listens on")
    parser.add_argument("--port", type=int, default=8999, help="port the benchmark connects to")
    parser.add_argument(
        "--output-root",
        default=os.environ.get("GENIESIM_OUTPUT_ROOT"),
        help="the benchmark run's output directory, to read its scores (optional)",
    )
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=900.0,
        help="how long the benchmark may take to reach its first request",
    )


def cell_from_args(args: argparse.Namespace) -> Cell:
    return Cell(
        suite=str(getattr(args, "suite", None) or "g2_op"),
        task=str(args.task),
        seed=int(getattr(args, "seed", 1)),
        max_episode_steps=int(getattr(args, "max_episode_steps", 200) or 200),
    )


def boot_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "host": getattr(args, "host", None) or "127.0.0.1",
        "port": int(getattr(args, "port", None) or 8999),
        "output_root": getattr(args, "output_root", None),
        "startup_timeout_s": getattr(args, "startup_timeout_s", None),
    }


# ---------------------------------------------------------------------------
# the benchmark process
# ---------------------------------------------------------------------------


class Benchmark:
    """The benchmark run a session talks to: started elsewhere, never by this host.

    GenieSim's own launcher, a cluster pod or the online platform starts the run, and it
    connects.  It is alive while its connection is (with a grace period for the client's
    reconnects).  Its scores are the result file it writes under ``output_root`` on this
    machine; without ``output_root`` (the online platform keeps them) there are none to read.
    """

    def __init__(
        self,
        bridge: bridge_mod.Bridge,
        output_root: str | None = None,
        grace_s: float = 180.0,
    ) -> None:
        self.bridge = bridge
        #: Where the benchmark writes ``benchmark/<task>/<sub_task>/evaluate_ret_NN.json``.
        self.output_root = output_root
        self.grace_s = grace_s
        # The run's result file is written after it connected: anything older is a
        # different run's.
        self._started_at = time.time() - 5.0

    def alive(self) -> bool:
        return self.bridge.connected(self.grace_s)

    def wait(self, timeout_s: float) -> bool:
        """Wait for the run to go (its connection closed for good); True when it did."""

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.alive():
                return True
            time.sleep(2.0)
        return False

    def details(self) -> list[dict[str, Any]]:
        """Every episode entry the run has written so far (its ``evaluate_ret`` file).

        Only a result file written after the run connected counts: the output tree keeps
        every earlier run's files.  The benchmark rewrites the file after each episode.
        """

        if not self.output_root:
            return []
        pattern = f"{self.output_root}/benchmark/*/*/evaluate_ret_*.json"
        script = (
            f"for f in $(ls -t {pattern} 2>/dev/null); do "
            f'if [ $(stat -c %Y "$f") -ge {int(self._started_at)} ]; then cat "$f"; fi; break; done'
        )
        try:
            out = subprocess.run(  # noqa: S603
                self._shell(script), capture_output=True, text=True, timeout=30, check=False
            )
            data = json.loads(out.stdout) if out.stdout.strip() else None
        except (subprocess.SubprocessError, ValueError):
            return []
        if not isinstance(data, dict):
            return []
        return list(data.get("details") or [])

    def _shell(self, script: str) -> list[str]:
        """The command that runs ``script`` where the run's files are."""

        return ["sh", "-c", script]

    @staticmethod
    def scores_of(detail: dict[str, Any]) -> dict[str, Any]:
        """One episode entry as its scores (``STEPS`` per stage, ``E2E``) and instruction."""

        scores = dict((detail.get("result") or {}).get("scores") or {})
        scores["instruction"] = detail.get("task_instruction")
        scores["policy_dispatch_count"] = detail.get("policy_dispatch_count")
        return scores

    def result(self, index: int = 0, wait_s: float = 0.0) -> dict[str, Any] | None:
        """The scores of the run's episode ``index`` (0 = the first), once written."""

        if not self.output_root:
            return None
        deadline = time.monotonic() + wait_s
        while True:
            details = self.details()
            if len(details) > index:
                return self.scores_of(details[index])
            if time.monotonic() >= deadline:
                return None
            time.sleep(2.0)

    def stop(self) -> None:
        """The run is not this host's to stop."""

    def tail(self, lines: int = 40) -> str:
        return ""


# ---------------------------------------------------------------------------
# the toolkit
# ---------------------------------------------------------------------------


class Toolkit:
    """The G2's tools over a live bridge, with the steps they record."""

    #: Tools that answer with their own payload instead of a state envelope: they do not
    #: move the robot, and ``RobotBase._readonly`` reads their result directly.
    READ_ONLY = frozenset({"segment", "locate", "back_project"})

    def __init__(self, bridge: bridge_mod.Bridge, out_dir: Path, session: Any = None) -> None:
        self.bridge = bridge
        #: The :class:`Session` this episode belongs to (server and benchmark live there).
        self.session = session
        #: Which of the run's episodes this toolkit plays (1 = the first).
        self.episode = bridge.episode
        self.verdict: dict[str, Any] | None = None
        self.out_dir = Path(out_dir)
        self.steps_dir = self.out_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        obs = bridge.first_obs()
        self._primitives = tools.Primitives(_Env(bridge))
        self._primitives.start_joints = {
            "left": [float(v) for v in obs.arm_joints[:7]],
            "right": [float(v) for v in obs.arm_joints[7:14]],
        }
        self._primitives.start_waist = [float(v) for v in obs.waist]
        # Keys are the names the robot class sends (its method names), not the internal
        # function names.
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "move_to": tools.move_pose,
            "move_both": tools.move_both_pose,
            "move_base": tools.move_base,
            "move_waist": tools.move_waist,
            "set_gripper": tools.grip,
            "home": tools.home,
            "wait_for_check": tools.wait_for_check,
            "segment": self._segment,
            "locate": self._locate,
            "back_project": self._back_project,
        }
        self._lock = __import__("threading").Lock()
        self._record(obs)

    # -- recordings ---------------------------------------------------------------------

    def _record(self, obs: bridge_mod.Observation) -> int:
        """Store one step's images and state; returns its index."""

        record = {
            "instruction": obs.instruction,
            "arm_joints": list(obs.arm_joints),
            "gripper": list(obs.gripper),
            "end_pose": obs.end_pose,
        }
        (self.steps_dir / f"{obs.step:04d}.json").write_text(json.dumps(record, indent=1))
        for name, frame in self.bridge.frames().items():
            self._write_png(name, frame, obs.step)
        return obs.step

    def _write_png(self, name: str, frame: np.ndarray, step: int) -> None:
        from PIL import Image

        folder = self.steps_dir / f"{step:04d}"
        folder.mkdir(exist_ok=True)
        Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8)).save(folder / f"{name}.png")

    def artifact_path(self, name: str, step: int = -1) -> Path:
        index = self.bridge.get_obs().step if step < 0 else int(step)
        return self.steps_dir / f"{index:04d}" / name

    def load_bytes(self, name: str, step: int = -1) -> bytes:
        path = self.artifact_path(name, step)
        if path.is_file():
            return path.read_bytes()
        # The first observation may not have been stored under its number yet.
        obs = self.bridge.get_obs()
        frames = self.bridge.frames()
        camera = name.rsplit(".", 1)[0]
        if camera in frames and step in (-1, obs.step):
            import io

            from PIL import Image

            buffer = io.BytesIO()
            Image.fromarray(np.asarray(frames[camera])[..., :3].astype(np.uint8)).save(
                buffer, format="PNG"
            )
            return buffer.getvalue()
        raise FileNotFoundError(f"no artifact {name!r} at step {step}")

    # -- perception ---------------------------------------------------------------------

    def _with_depth(self, camera: str) -> Any:
        """The current camera, after making sure depth and calibration have arrived.

        Depth is switched on by the benchmark AFTER a step: our reply asks for it, the
        client records the request, and the next observation carries it.  A read-only tool
        never steps, so on the episode's first request there is no depth and there never
        would be — the perception tools take one hold action to bring it in, which changes
        nothing in the scene.  Costs one step of the budget, once per episode.
        """

        obs = self.bridge.get_obs()
        cam = obs.cameras.get(camera)
        if cam is not None and cam.depth is not None and cam.fx is not None:
            return cam
        for _ in range(2):
            self._primitives.act(
                {
                    "kind": "JOINT_ABS",
                    "left_arm": list(obs.arm_joints[:7]),
                    "right_arm": list(obs.arm_joints[7:14]),
                    "left_gripper": (float(obs.gripper[0]) if obs.gripper else 0.0),
                    "right_gripper": (float(obs.gripper[1]) if len(obs.gripper) > 1 else 0.0),
                }
            )
            obs = self.bridge.get_obs()
            cam = obs.cameras.get(camera)
            if cam is not None and cam.depth is not None and cam.fx is not None:
                return cam
        return cam if cam is not None else self._camera(camera)

    def _camera(self, camera: str) -> Any:
        name = sight.camera_name(camera, CAMERAS)
        obs = self.bridge.get_obs()
        cam = obs.cameras.get(name)
        if cam is None:
            raise ValueError(f"camera {name!r} is not in this observation")
        return cam

    def _frame(self, camera: str) -> np.ndarray:
        frames = self.bridge.frames()
        frame = frames.get(camera)
        if frame is None:
            raise ValueError(f"camera {camera!r} sent no image")
        return frame

    # -- SAM3 ---------------------------------------------------------------------------

    _sam3_client: Any = None
    _sam3_alive: bool | None = None

    def _sam3(self) -> Any:
        """The machine's SAM3 service (``$PYRUA_SAM3_ENDPOINT``, default 127.0.0.1:8790),
        or None when it is not running — asked once per episode."""

        if self._sam3_alive is None:
            from . import sam3_http

            client = sam3_http.Sam3Http(os.environ.get("PYRUA_SAM3_ENDPOINT"))
            self._sam3_alive = client.alive()
            self._sam3_client = client if self._sam3_alive else None
        return self._sam3_client

    #: Instances of one name a text locate reports, best first, and how many ``near`` may
    #: choose among.  Measured: on the supermarket shelf, "orange juice bottle" scores the
    #: bottles on the shelf 0.86-0.94 and the one in the basket 0.35 — eighth.
    SAM3_CANDIDATES = 5
    SAM3_SEARCH = 25

    def _sam3_region(
        self,
        frame: np.ndarray,
        cam: Any,
        method: str,
        min_score: float,
        near: Any = None,
        **prompt: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """``(region or None, what SAM3 said)`` for a text or a ``point`` prompt.

        For a text prompt ``said["candidates"]`` lists every instance of the name SAM3
        scored above ``min_score`` (pixel, score, area), and ``near`` = (row, col) picks
        the one whose mask holds that pixel, else the one centred nearest it, instead of
        the best-scoring.
        """

        client = self._sam3()
        if client is None:
            return None, {"note": "the SAM3 service is not running on this machine"}
        text = prompt.get("text_prompt")
        found = client.segment(
            frame,
            min_score=0.05 if (text and near is not None) else min_score,
            top_k=self.SAM3_SEARCH if text else 1,
            **prompt,
        )
        said: dict[str, Any] = {
            "score": None if found.score is None else round(float(found.score), 3)
        }
        candidates = list(getattr(client, "last_candidates", []) or [])
        if text and candidates:
            listed = []
            for cand in candidates:
                rows, cols = np.nonzero(cand["mask"])
                listed.append(
                    {
                        "pixel_uv": [int(rows.mean()), int(cols.mean())],
                        "score": round(float(cand["score"]), 3),
                        "area_px": int(rows.size),
                    }
                )
            said["candidates"] = [u for u in listed if u["score"] >= min_score][
                : self.SAM3_CANDIDATES
            ]
            if near is not None:
                r, c = int(near[0]), int(near[1])
                inside = [
                    k
                    for k, cand in enumerate(candidates)
                    if 0 <= r < cand["mask"].shape[0]
                    and 0 <= c < cand["mask"].shape[1]
                    and cand["mask"][r, c]
                ]
                pick = (
                    inside[0]
                    if inside
                    else int(
                        np.argmin(
                            [
                                (u["pixel_uv"][0] - r) ** 2 + (u["pixel_uv"][1] - c) ** 2
                                for u in listed
                            ]
                        )
                    )
                )
                said["score"] = listed[pick]["score"]
                said["chosen"] = listed[pick]
                region = sight.region_from_mask(frame, cam, candidates[pick]["mask"], method)
                return (region if region.get("found") else None), said
        if not found.found or found.mask is None:
            said["note"] = found.reason or "SAM3 found nothing"
            return None, said
        if found.mask.mean() > sight.MAX_OBJECT_FRACTION * 3:
            said["note"] = (
                f"SAM3's mask covers {found.mask.mean():.0%} of the image: not one object"
            )
            return None, said
        region = sight.region_from_mask(frame, cam, found.mask, method)
        return (region if region.get("found") else None), said

    def _segment(
        self,
        camera: str = "head",
        row: int | None = None,
        col: int | None = None,
        text: str | None = None,
        min_score: float = 0.2,
    ) -> dict[str, Any]:
        name = sight.camera_name(camera, CAMERAS)
        frame = self._frame(name)
        if text:
            region, said = self._sam3_region(
                frame, self._with_depth(name), "sam3_text", min_score, text_prompt=str(text)
            )
            if region is None:
                return {"found": False, "camera": name, "text": text, **said}
            found = {**region, **said}
        else:
            found = sight.segment(frame, row, col)
            found["method"] = "colour"
        if not found.get("found"):
            return {"found": False, "camera": name}
        return {
            "found": True,
            "camera": name,
            "centroid_rc": found["centroid_rc"],
            "box_rc": found["box_rc"],
            "area_px": found["area_px"],
            "colour": found["colour"],
            "method": found.get("method"),
            "score": found.get("score"),
        }

    def _locate(
        self,
        camera: str = "head",
        row: int | None = None,
        col: int | None = None,
        text: str | None = None,
        min_score: float = 0.2,
        near: Any = None,
    ) -> dict[str, Any]:
        name = sight.camera_name(camera, CAMERAS)
        cam = self._with_depth(name)
        frame = self._frame(name)
        said: dict[str, Any] = {}
        if text:
            region, said = self._sam3_region(
                frame, cam, "sam3_text", min_score, near=near, text_prompt=str(text)
            )
            if region is None:
                return {"found": False, "camera": name, "text": text, **said}
        else:
            region = None
            if row is not None and col is not None:
                # By shape first: what stands above its surface, whatever its colour (it
                # separates white cartons on a white table).  Then SAM3's mask at the pixel —
                # thin and small things geometry misses — then the colour region.
                region = sight.segment_geometric(frame, cam, int(row), int(col))
                if region is None:
                    region, said = self._sam3_region(
                        frame, cam, "sam3_point", 0.0, point=[int(row), int(col)]
                    )
            if region is None:
                region = sight.segment(frame, row, col, depth=cam.depth)
        measured = sight.locate(cam, region)
        measured["camera"] = name
        if (
            said.get("score") is not None
            and region is not None
            and str(region.get("method", "")).startswith("sam3")
        ):
            measured["score"] = said["score"]
        if text:
            measured["text"] = text
            if said.get("candidates"):
                measured["candidates"] = said["candidates"]
        # Remember the surface it stands on, for move_to's check that the fingertips are not
        # sent into it (the latest measurement near a place replaces older ones there).
        centre, support = measured.get("centre_xyz"), measured.get("support_z")
        if measured.get("found") and centre and support is not None:
            known = [
                s
                for s in self._primitives.surfaces
                if (s[0] - centre[0]) ** 2 + (s[1] - centre[1]) ** 2 > 0.05**2
            ]
            self._primitives.surfaces = (
                known + [(float(centre[0]), float(centre[1]), float(support))]
            )[-30:]
        return measured

    def _back_project(self, row: int, col: int, camera: str = "head") -> dict[str, Any]:
        name = sight.camera_name(camera, CAMERAS)
        measured = sight.back_project(self._with_depth(name), int(row), int(col))
        measured["camera"] = name
        return measured

    # -- the backend contract -----------------------------------------------------------

    def _envelope(self, tool: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        obs = self.bridge.get_obs()
        arms = {name: self._primitives.arm_state(name) for name in tools.ARMS}
        return {
            "step": obs.step,
            "task_language": obs.instruction,
            "terminated": False,
            "truncated": False,
            "state": {
                "state": {
                    "arm_state": arms,
                    "waist": list(obs.waist),
                    "torso": _torso(obs.waist),
                    "left_arm_joint_state": list(obs.arm_joints[:7]),
                    "right_arm_joint_state": list(obs.arm_joints[7:14]),
                    "left_ee_joint_state": [float(obs.gripper[0])] if obs.gripper else [0.0],
                    "right_ee_joint_state": [float(obs.gripper[1])]
                    if len(obs.gripper) > 1
                    else [0.0],
                    "instruction": obs.instruction,
                },
                "cameras": {
                    name: {
                        "shape": [camera.height, camera.width, 3],
                        "has_depth": camera.depth is not None,
                    }
                    for name, camera in obs.cameras.items()
                },
            },
            "log": {"tool": tool, "result": payload},
        }

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(input_dict or {})
        if name == "view_env_state":
            obs = self.bridge.get_obs()
            self._record(obs)
            return self._envelope(name, {})
        handler = self._tools.get(name)
        if handler is None:
            return {"error": f"this benchmark host has no tool {name!r}"}
        with self._lock:
            try:
                # Tools taking the primitives get them positionally; the perception helpers
                # are bound methods that already hold what they need.
                if getattr(handler, "__self__", None) is not None:
                    payload = handler(**kwargs)
                else:
                    payload = handler(self._primitives, **kwargs)
            except bridge_mod.EpisodeOver:
                payload = {"interrupted": True, "code": "tool_cancelled"}
            except TypeError as exc:
                return {"error": f"{name}: {exc}"}
            except Exception as exc:  # noqa: BLE001 - reported as a tool error
                return {"error": f"{name}: {type(exc).__name__}: {exc}"}
            if "error" not in payload:
                self._record(self.bridge.get_obs())
            # A read-only tool answers with its own payload: `RobotBase._readonly` reads it
            # directly, and wrapping it in the state envelope would hide the fields (a
            # `locate` result read back as an envelope has no "found").
            if name in self.READ_ONLY:
                return payload
            return self._envelope(name, payload)

    # -- the rest ----------------------------------------------------------------------

    #: How long the benchmark may take, after the program has finished, to end the episode
    #: itself: a failed stage is only ended by its step budget, thousands of steps on the
    #: long tasks.
    VERDICT_TIMEOUT_S = 3600.0

    def solved(self) -> bool:
        """The benchmark's own verdict: its end-to-end score, once it has written it.

        False until :meth:`conclude` has read the result file; nothing in the episode's
        tool results says whether the task was solved.
        """

        return bool(self.verdict and float(self.verdict.get("E2E") or 0.0) >= 1.0)

    def episode_over(self) -> bool:
        return self.bridge.over is not None

    def conclude(self) -> None:
        """Let the benchmark end this episode by itself, and read its score.

        Finishing the program does not end the episode: the bridge keeps the robot still,
        grippers as last commanded, and the benchmark ends the episode at the check where
        the task passes or when a stage's step budget is spent — its own two ways out,
        both of which clear its task checker for the next episode (see
        ``Bridge.get_action``).  This waits for that closing request, then for the
        benchmark to write this episode's entry.
        """

        self.bridge.finish()
        session = self.session
        alive = session.benchmark.alive if session is not None else None
        self.bridge.wait_closed(self.episode, self.VERDICT_TIMEOUT_S, alive=alive)
        if session is not None:
            self.verdict = session.benchmark.result(self.episode - 1, wait_s=120.0)
            (self.out_dir / "benchmark_verdict.json").write_text(
                json.dumps(self.verdict, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    def env_steps(self) -> int | None:
        return self.bridge.step

    def cancel_active_and_wait(self) -> None:
        self.bridge.abandon()

    def close(self) -> None:
        """Nothing to release: the server and the benchmark belong to the session."""


def _torso(waist: Any) -> dict[str, Any]:
    """Where the upper body faces, in robo.FRAME (base_link is fixed to the base)."""

    yaw = tools.torso_yaw(list(waist or []))
    forward = tools.named_direction("forward", yaw)
    return {
        "heading_deg": round(float(np.degrees(yaw)), 1),
        "forward": [round(float(v), 3) for v in forward],
    }


class _Env:
    """The bridge as the tools' environment."""

    def __init__(self, bridge: bridge_mod.Bridge) -> None:
        self.bridge = bridge

    def get_obs(self) -> bridge_mod.Observation:
        return self.bridge.get_obs()

    def frames(self) -> dict[str, np.ndarray]:
        return self.bridge.frames()

    def get_status(self) -> dict[str, Any]:
        return self.bridge.status()

    def step(self, action: dict[str, Any]) -> tuple[bridge_mod.Observation, bool]:
        return self.bridge.step_env(action)


class Backend:
    """A :class:`Toolkit` as a :class:`pyrualean.Backend`."""

    def __init__(self, toolkit: Toolkit) -> None:
        self._toolkit = toolkit

    @property
    def toolkit(self) -> Toolkit:
        return self._toolkit

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        return self._toolkit.execute_tool(name, input_dict)

    def conclude(self) -> None:
        self._toolkit.conclude()

    def solved(self) -> bool:
        return self._toolkit.solved()

    def episode_over(self) -> bool:
        return self._toolkit.episode_over()

    def artifact(self, name: str, step: int = -1) -> bytes:
        return self._toolkit.load_bytes(name, step=step)

    def env_steps(self) -> int | None:
        return self._toolkit.env_steps()

    def cancel(self) -> None:
        self._toolkit.cancel_active_and_wait()


def make_backend(toolkit: Toolkit) -> Backend:
    return Backend(toolkit)


# ---------------------------------------------------------------------------
# boot / card
# ---------------------------------------------------------------------------


class NoMoreEpisodes(RuntimeError):
    """The benchmark run has no further episode to play (it finished, or it exited)."""


class Session:
    """One benchmark run: the WebSocket server and the benchmark process, for all its episodes.

    A benchmark run plays several episodes (instructions, scene instances) one after
    another over one connection.  The session outlives each of them; :meth:`next_toolkit`
    waits for the next episode to begin and returns the tools for it, and a fresh program
    (a fresh agent conversation) plays each one.  :meth:`stop` ends whatever is left,
    lets the benchmark write its results and exit, and returns every episode's scores.
    """

    #: How long the benchmark may take to reach its first request, and between episodes.
    STARTUP_TIMEOUT_S = 900.0
    BETWEEN_EPISODES_S = 300.0
    #: How long the benchmark may take to finish the remaining episodes and exit.
    EXIT_TIMEOUT_S = 600.0

    def __init__(
        self,
        cell: Cell,
        out_dir: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8999,
        output_root: str | None = None,
        startup_timeout_s: float | None = None,
        **_: Any,
    ) -> None:
        """Serve ``host:port`` and wait for a run started elsewhere to connect."""

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.bridge = bridge_mod.Bridge()
        self.server = bridge_mod.Server(self.bridge, host, port)
        if not self.server.alive():
            raise RuntimeError(
                f"bridge server failed to start on {host}:{port}: {self.server.error}"
            )
        self.benchmark = Benchmark(self.bridge, output_root=output_root)
        self._startup_timeout_s = float(startup_timeout_s or self.STARTUP_TIMEOUT_S)
        self.played = 0

    @classmethod
    def attach(cls, bridge: bridge_mod.Bridge, benchmark: Benchmark, out_dir: Path) -> Session:
        """A session for a run that connected by itself (a remote worker), on ``bridge``.

        The server is shared by every run and belongs to whoever started it.
        """

        self = cls.__new__(cls)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = bridge
        self.server = None
        self.benchmark = benchmark
        self._startup_timeout_s = cls.STARTUP_TIMEOUT_S
        self.played = 0
        return self

    def next_toolkit(self, out_dir: Path) -> Toolkit:
        """Wait for the next episode to begin; the tools for it, recording into ``out_dir``.

        Raises :class:`NoMoreEpisodes` when the run has none left.
        """

        number = self.played + 1
        timeout = self._startup_timeout_s if number == 1 else self.BETWEEN_EPISODES_S
        obs = self.bridge.wait_episode(number, timeout, alive=self.benchmark.alive)
        if obs is None:
            if number == 1:
                tail = self.benchmark.tail()
                self.stop()
                raise RuntimeError(
                    f"the benchmark sent no first observation\n--- benchmark tail ---\n{tail}"
                )
            raise NoMoreEpisodes(f"the run has no episode {number}")
        self.played = number
        return Toolkit(self.bridge, out_dir, session=self)

    def has_next(self) -> bool:
        """Whether another episode begins (waits for it; does not take it)."""

        number = self.played + 1
        timeout = self._startup_timeout_s if number == 1 else self.BETWEEN_EPISODES_S
        return self.bridge.wait_episode(number, timeout, alive=self.benchmark.alive) is not None

    def stop(self) -> list[dict[str, Any]]:
        """End the run: finish what is left, let the benchmark exit, return every score."""

        self.bridge.drain()
        exited = self.benchmark.wait(self.EXIT_TIMEOUT_S)
        details = self.benchmark.details()
        if self.server is not None:
            with contextlib.suppress(Exception):
                self.server.stop()
        if not exited:
            self.benchmark.stop()
        scores = [Benchmark.scores_of(d) for d in details]
        (self.out_dir / "run_scores.json").write_text(
            json.dumps(scores, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return scores


class _OwnedSession:
    """What ``play`` stops after an episode whose :func:`boot` started its own session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def stop(self) -> None:
        self.session.stop()


def boot(
    cell: Cell,
    out_dir: Path,
    *,
    session: Session | None = None,
    **options: Any,
) -> tuple[Toolkit, list[Any]]:
    """The tools for the next episode; returns ``(toolkit, daemons)`` for ``play``.

    With ``session`` (a :class:`Session` the caller keeps across episodes), this waits for
    that run's next episode and returns nothing for ``play`` to stop.  Without one, it starts
    a run of its own for this single episode and hands ``play`` that run to stop afterwards.
    """

    options.pop("rpent_root", None)
    if session is not None:
        return session.next_toolkit(Path(out_dir)), []
    owned = Session(cell, Path(out_dir), **options)
    return owned.next_toolkit(Path(out_dir)), [_OwnedSession(owned)]


def dump_card(toolkit: Toolkit, cell: Cell, out_dir: Path) -> TaskCard:
    """Write ``card.json`` plus the first observation's images and return the card."""

    out_dir = Path(out_dir)
    obs = toolkit.bridge.get_obs()
    card_dir = out_dir / "card"
    card_dir.mkdir(exist_ok=True)
    images: dict[str, str] = {}
    frames = toolkit.bridge.frames()
    for camera, name in CARD_IMAGES.items():
        frame = frames.get(camera)
        if frame is None:
            continue
        from PIL import Image

        target = card_dir / name
        Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8)).save(target)
        images[camera] = str(target)
    facts: dict[str, Any] = {}
    for arm in ("left", "right"):
        state = toolkit._primitives.arm_state(arm)
        facts[f"{arm} tool centre [x, y, z]"] = f"{state['tool_centre']} m (base_link)"
        facts[f"{arm} gripper opening"] = f"{state['opening']:.3f} (0 = closed, -0.785 = open)"
    torso = _torso(obs.waist)
    facts["upper body faces"] = (
        f"{torso['forward']} in robo.FRAME (turned {torso['heading_deg']} deg from base_link +x): "
        "the table and the task are in front of the upper body; point='forward' means this"
    )
    if cell.stage_budgets:
        facts["stage budgets (native actions)"] = (
            ", ".join(str(b) for b in cell.stage_budgets)
            + " — one per task stage, in order; a stage that spends its budget fails"
        )
    status = toolkit.bridge.status()
    card = TaskCard(
        suite=cell.suite,
        task=cell.task,
        seed=cell.seed,
        task_language=obs.instruction,
        object_names=[],
        images=images,
        max_env_steps=int(
            sum(cell.stage_budgets)
            if cell.stage_budgets
            else (status.get("step_limit") or cell.max_episode_steps)
        ),
        robot="robocoliseum",
        facts=facts,
    )
    (out_dir / "card.json").write_text(card.to_json() + "\n", encoding="utf-8")
    return card


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the online platform
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

#: The run's per-episode programs: ``(cell, episode_dir, session) -> play's result``.
PlayOne = Callable[[Cell, Path, Session], dict[str, Any]]


def serve_online(tunnel: bridge_mod.TunnelClient, out_dir: Path, play_one: PlayOne) -> int:
    """Play every case the platform sends through ``tunnel``, until it drains.

    A case is one session: one benchmark run of one task, several episodes one after
    another, each played by its own program.  The platform keeps the scores; the task and
    its instruction come from the run's own requests.  Returns the episodes played.
    """

    out_dir = Path(out_dir)
    played = 0
    while True:
        opened = tunnel.next_session()
        if opened is None:
            break
        sid, bridge = opened
        # A redial lands on the same session within the gateway's window: alive meanwhile.
        benchmark = Benchmark(bridge, grace_s=tunnel.RECONNECT_WINDOW_S + 15.0)
        session = Session.attach(
            bridge, benchmark, out_dir / (re.sub(r"[^\w.-]", "_", sid)[:80] or "session")
        )
        try:
            while session.has_next():
                obs = bridge.get_obs()
                cell = Cell(task=obs.task_name or "unknown")
                episode_dir = session.out_dir / f"episode_{session.played + 1:03d}"
                try:
                    result = play_one(cell, episode_dir, session)
                except Exception:  # noqa: BLE001 - one episode's failure must not end the job
                    logger.exception("episode %s of session %s failed", session.played + 1, sid)
                    bridge.abandon()
                    result = {"status": "error"}
                played += 1
                row = {
                    "session": sid,
                    "episode": session.played,
                    "task": cell.task,
                    "instruction": obs.instruction,
                    "status": result.get("status"),
                    "env_steps": result.get("env_steps"),
                }
                with open(out_dir / "online.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            session.stop()
    return played


def main(argv: list[str] | None = None) -> int:
    """``agent``: serve one online RoboColiseum job through its tunnel until it drains.

    The job is submitted on the platform first (its ``challenge-submit-job``); each agent
    process serves one tunnel, and a job runs up to its ``parallelism`` of them.  The other
    options are ``play``'s: the arm, the runtime, the model and its budgets.

      export JOB_UUID=... TUNNEL_ENDPOINT=ws://.../api/challenge/tunnel JOB_TOKEN=...
      python -m pyrualean.hosts.geniesim_robocoliseum agent --arm cells --out runs/job \\
          --runtime openai --model ... --base-url ... --api-key-env ...
    """

    from .. import play

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "agent":
        print(
            "usage: python -m pyrualean.hosts.geniesim_robocoliseum agent [options]"
            "  (see agent --help)",
            file=sys.stderr,
        )
        return 2
    parser, host = play.build_parser(
        "robocoliseum", prog="python -m pyrualean.hosts.geniesim_robocoliseum agent"
    )
    parser.set_defaults(robot="robocoliseum")
    online = parser.add_argument_group("online", "the platform's tunnel (its challenge-run-agent)")
    online.add_argument("--job-uuid", default=os.environ.get("JOB_UUID"), help="$JOB_UUID")
    online.add_argument(
        "--endpoint", default=os.environ.get("TUNNEL_ENDPOINT"), help="$TUNNEL_ENDPOINT (ws://...)"
    )
    online.add_argument(
        "--token-env",
        default=None,
        help="the variable holding the tunnel token (default: JOB_TOKEN, else CHALLENGE_TOKEN)",
    )
    online.add_argument(
        "--agent-id", default=None, help="this process's agent id (default: a new UUID)"
    )
    args = parser.parse_args(argv[1:])
    token = (
        os.environ.get(args.token_env)
        if args.token_env
        else os.environ.get("JOB_TOKEN") or os.environ.get("CHALLENGE_TOKEN")
    )
    missing = [
        name
        for name, value in (
            ("--job-uuid", args.job_uuid),
            ("--endpoint", args.endpoint),
            ("a token", token),
        )
        if not value
    ]
    if missing:
        parser.error("missing " + ", ".join(missing))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    options = play.episode_options(args, "robocoliseum", host)

    def play_one(cell: Cell, episode_dir: Path, session: Session) -> dict[str, Any]:
        return play.play_episode(
            cell, episode_dir, **dict(options, boot_options={"session": session})
        )

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tunnel = bridge_mod.TunnelClient(args.endpoint, args.job_uuid, token, agent_id=args.agent_id)
    print(f"agent {tunnel.agent_id} dialling {args.endpoint}", flush=True)
    try:
        played = serve_online(tunnel, out_dir, play_one)
    except KeyboardInterrupt:
        tunnel.stop()
        return 130
    tunnel.stop()
    print(
        json.dumps(
            {
                "agent": tunnel.agent_id,
                "episodes": played,
                "drained": tunnel.drained,
                "error": tunnel.error,
            }
        ),
        flush=True,
    )
    return 0 if tunnel.error is None else 1


__all__ = [
    "Backend",
    "Benchmark",
    "NoMoreEpisodes",
    "Session",
    "CAMERAS",
    "Cell",
    "Toolkit",
    "add_boot_args",
    "add_cell_args",
    "boot",
    "boot_options",
    "cell_from_args",
    "dump_card",
    "make_backend",
    "main",
    "serve_online",
]


if __name__ == "__main__":
    raise SystemExit(main())
