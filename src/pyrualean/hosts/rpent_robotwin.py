# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RPent-backed RoboTwin host.

Boots RPent's own RoboTwin stack (SAPIEN env server, LingBot-VLA server),
builds the same ``RoboTwinToolkit`` its planners use, wraps it as a
:class:`Backend` and runs one policy file against it.  Because every
primitive still goes through ``Toolkit.execute_tool``, the run produces
RPent's native artifacts (``states.json``, ``episode.mp4``, per-step images
and world maps) and is judged by the same ``TASK_ENV.eval_success`` predicate
as an RPent agent run.

RPent's ``RoboTwinToolkit`` leaves ``solved()`` unimplemented and counts
native VLA actions rather than frames, so this host provides
:class:`RobotwinBackend` through ``make_backend(toolkit)``; ``pyrualean.play``
and :func:`run_episode` wrap the plain toolkit in it.

This module needs RPent and its ``robotwin`` extra importable; run it with
that interpreter and the RoboTwin environment variables set (``docs/setup.md``).
One episode per process: ``python -m pyrualean.hosts.rpent_robotwin run ...``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._backend import Ledger
from ..prompt import TaskCard, render_prompt
from ..robotwin import RobotwinRobot
from ..runner import PolicyLoadError, load_policy, policy_sha256, run_policy
from ..sandbox import DESCRIPTION as SANDBOX_DESCRIPTION
from ..sandbox import audit_source, make_builtins
from .common import RESULT_SCHEMA, RpentBackend, resolve_rpent_root, stop_all

#: RPent's native RoboTwin task YAMLs (``robots/robotwin/robot_spec.py``).
TASK_CONFIGS = ("demo_randomized", "demo_clean")
CAMERAS = ("head", "left_wrist", "right_wrist")
#: Step-0 artifacts copied next to the card, keyed by the name the card uses.
CARD_IMAGES = {camera: f"{camera}_rgb.png" for camera in CAMERAS}


@dataclass(frozen=True)
class Cell:
    """One RoboTwin evaluation cell: a task name, a task config and an exact scene seed."""

    suite: str
    task: str
    seed: int
    max_episode_steps: int = 10000
    cuda_device: int | None = None

    @property
    def tag(self) -> str:
        return f"{self.task}_{self.suite.replace('demo_', '')}_s{self.seed}"


def add_cell_args(parser: argparse.ArgumentParser) -> None:
    """RoboTwin cell identifiers (``pyrualean play --robot robotwin`` and the host CLI)."""

    parser.add_argument("--task", required=True, help="RoboTwin task name, e.g. beat_block_hammer")
    parser.add_argument(
        "--suite",
        default="demo_randomized",
        choices=TASK_CONFIGS,
        help="RoboTwin task config (demo_randomized = the evaluation setting)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100000,
        help="exact scene seed (verified seeds: robots/robotwin/eval/demo_randomized.json)",
    )


def add_boot_args(parser: argparse.ArgumentParser) -> None:
    """Options of :func:`boot` that are RoboTwin-specific (env-var defaults as in RPent)."""

    parser.add_argument(
        "--vla-endpoint", default=None, help="[ws://]host:port of a shared LingBot-VLA server"
    )
    parser.add_argument(
        "--env-endpoint", default=None, help="http://host:port of an already running env server"
    )
    parser.add_argument(
        "--vla-model-path",
        default=os.environ.get("LINGBOT_MODEL_PATH"),
        help="LingBot checkpoint directory (default: $LINGBOT_MODEL_PATH)",
    )
    parser.add_argument(
        "--robotwin-assets-path",
        default=os.environ.get("ROBOTWIN_ASSETS_PATH"),
        help="RoboTwin asset snapshot (default: $ROBOTWIN_ASSETS_PATH)",
    )
    parser.add_argument(
        "--lingbot-robot-config",
        default=os.environ.get("LINGBOT_ROBOT_CONFIG"),
        help="LingBot FeatureTransform robot config (default: the checkpoint's)",
    )


def cell_from_args(args: argparse.Namespace) -> Cell:
    return Cell(
        suite=str(getattr(args, "suite", None) or "demo_randomized"),
        task=str(args.task),
        seed=int(args.seed),
        max_episode_steps=int(getattr(args, "max_episode_steps", 10000)),
        cuda_device=getattr(args, "cuda_device", None),
    )


def boot_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vla_endpoint": getattr(args, "vla_endpoint", None),
        "env_endpoint": getattr(args, "env_endpoint", None),
        "vla_model_path": getattr(args, "vla_model_path", None),
        "robotwin_assets_path": getattr(args, "robotwin_assets_path", None),
        "lingbot_robot_config": getattr(args, "lingbot_robot_config", None),
    }


def _eval_success(toolkit: Any) -> bool:
    """RoboTwin's native success flag, read from the toolkit without an RPC."""

    try:
        return bool(toolkit._primitives.status().get("eval_success"))
    except Exception:  # noqa: BLE001 - fall back to the last recorded status
        return bool((getattr(toolkit, "_latest_status", None) or {}).get("eval_success"))


class RobotwinBackend(RpentBackend):
    """RPent's RoboTwin toolkit as a :class:`pyrualean.Backend`.

    ``RoboTwinToolkit`` implements neither ``solved()`` nor a public
    ``primitives`` attribute, which the generic adapter relies on.
    """

    def solved(self) -> bool:
        return _eval_success(self._toolkit)

    def env_steps(self) -> int | None:
        try:
            return int(self._toolkit._primitives.native_actions)
        except Exception:  # noqa: BLE001 - optional metadata
            return super().env_steps()


def make_backend(toolkit: Any) -> RobotwinBackend:
    """The backend ``pyrualean.play`` wraps a booted RoboTwin toolkit in."""

    return RobotwinBackend(toolkit)


def boot(
    cell: Cell,
    out_dir: Path,
    *,
    rpent_root: str | None = None,
    env_only: bool = False,
    vla_endpoint: str | None = None,
    env_endpoint: str | None = None,
    vla_model_path: str | None = None,
    robotwin_assets_path: str | None = None,
    lingbot_robot_config: str | None = None,
) -> tuple[Any, list[Any]]:
    """Start RPent's servers and return ``(toolkit, daemons)``.

    ``env_only`` boots just the simulator (enough to dump a task card); the
    VLA client is then ``None`` and ``lingbot_act`` unusable.
    ``vla_endpoint`` (``[ws://]host:port``) attaches to a shared, already
    running LingBot server instead of spawning a private one (RPent's own
    ``--vla-endpoint``); ``env_endpoint`` likewise for the simulator, which is
    otherwise private to the episode.  ``vla_model_path`` /
    ``robotwin_assets_path`` default to ``LINGBOT_MODEL_PATH`` /
    ``ROBOTWIN_ASSETS_PATH``.
    """

    resolve_rpent_root(rpent_root)
    from robots.robotwin.robot_spec import get_robot_spec
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.utils.logging import init_output_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    init_output_dir(out_dir, verbose=False)
    # An empty, read-only memory corpus: no HF sync, nothing to read (zero-shot).
    memory_dir = out_dir / "_memory"
    memory_dir.mkdir(exist_ok=True)
    (memory_dir / "MEMORY.md").touch()
    events = NullDashboardEventSink()
    args = Namespace(
        task_name=cell.task,
        task_config=cell.suite,
        seed=cell.seed,
        max_episode_steps=cell.max_episode_steps,
        robotwin_assets_path=robotwin_assets_path or os.environ.get("ROBOTWIN_ASSETS_PATH"),
        vla_model_path=vla_model_path or os.environ.get("LINGBOT_MODEL_PATH"),
        lingbot_robot_config=lingbot_robot_config or os.environ.get("LINGBOT_ROBOT_CONFIG"),
        env_endpoint=env_endpoint,
        vla_endpoint=vla_endpoint,
        cuda_device=cell.cuda_device,
        env_cuda_device=None,
        vla_cuda_device=None,
        memory_dir=str(memory_dir),
        output_dir=str(out_dir),
    )
    spec = get_robot_spec()
    components = {"env"} if env_only else None
    daemons, runtime_kwargs = spec.init_runtime(args, out_dir, events, components)
    if env_only:
        runtime_kwargs.setdefault("model", None)
    from robots.robotwin.toolkit import RoboTwinToolkit

    try:
        # What RPent's ``get_toolkit`` does, minus the RunConfig it only reads
        # ``memory_dir`` from.
        toolkit = RoboTwinToolkit(
            runtime_kwargs=runtime_kwargs,
            dashboard_events=events,
            memory=MemoryManager(root=memory_dir),
        )
    except BaseException:
        for daemon in reversed(daemons):
            with contextlib.suppress(Exception):
                daemon.stop()
        raise
    return toolkit, daemons


def _floats(value: Any, n: int) -> list[float]:
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        return [0.0] * n
    return out if len(out) == n else [0.0] * n


def _fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


def _table_height(toolkit: Any) -> float | None:
    """Median z of the head world map at step 0 (the table fills most of the view)."""

    try:
        import numpy as np

        world = np.asarray(toolkit.state.load("head_world_xyz.npy", step=0))
        value = float(np.nanmedian(world[..., 2]))
    except Exception:  # noqa: BLE001 - optional fact
        return None
    return value if value == value else None  # NaN check


def dump_card(toolkit: Any, cell: Cell, out_dir: Path) -> TaskCard:
    """Write ``card.json`` plus the step-0 images and return the card.

    The facts are what RPent's agent also sees in its first
    ``view_env_state``: both arms' initial EEF / TCP poses and gripper values,
    the native-action budget and the table height measured from the head
    world map.  Nothing privileged (no object poses).
    """

    view = toolkit.execute_tool("view_env_state", {"step": 0}).result
    raw = view.get("state") or {}
    robot = raw.get("robot_state") or {}
    status = raw.get("episode_status") or {}
    card_dir = out_dir / "card"
    card_dir.mkdir(exist_ok=True)
    images: dict[str, str] = {}
    for camera, name in CARD_IMAGES.items():
        try:
            source = toolkit.state.artifact_path(name, step=0)
        except Exception:  # noqa: BLE001 - optional artifact
            continue
        if source.is_file():
            target = card_dir / f"{camera}.png"
            shutil.copyfile(source, target)
            images[camera] = str(target)
    facts: dict[str, Any] = {}
    for arm in ("left", "right"):
        eef = _floats(robot.get(f"{arm}_eef_pose"), 7)
        tcp = _floats(robot.get(f"{arm}_tcp_pose"), 7)
        facts[f"{arm} arm initial EEF position [x, y, z]"] = f"{_fmt(eef[:3])} m"
        facts[f"{arm} arm initial EEF orientation [w, x, y, z]"] = _fmt(eef[3:])
        facts[f"{arm} arm initial TCP (gripper centre) [x, y, z]"] = f"{_fmt(tcp[:3])} m"
        gripper = float(robot.get(f"{arm}_gripper", 0.0) or 0.0)
        facts[f"{arm} gripper"] = f"{gripper:.2f} (1 = open, 0 = closed)"
    table = _table_height(toolkit)
    if table is not None:
        facts["table surface height (median z of the head world map at step 0)"] = f"{table:.3f} m"
    if status.get("step_lim") is not None:
        facts["native action budget (step_lim)"] = int(status["step_lim"])
    card = TaskCard(
        suite=cell.suite,
        task=cell.task,
        seed=cell.seed,
        task_language=str(view.get("task_language") or raw.get("task_language") or ""),
        object_names=[],
        images=images,
        max_env_steps=cell.max_episode_steps,
        robot="robotwin",
        facts=facts,
    )
    (out_dir / "card.json").write_text(card.to_json() + "\n", encoding="utf-8")
    return card


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_episode(
    policy_path: Path,
    cell: Cell,
    out_dir: Path,
    *,
    timeout_s: float = 1200.0,
    rpent_root: str | None = None,
    boot_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one policy file for one cell and write ``result.json``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    options = dict(boot_options or {})
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "system": "pyrualean",
        "backend": "robotwin",
        "robot": "robotwin",
        "suite": cell.suite,
        "task": cell.task,
        "seed": cell.seed,
        "task_language": None,
        "policy": {
            "path": str(policy_path),
            "sha256": policy_sha256(policy_path),
            "bytes": policy_path.stat().st_size,
        },
        "environment_success": False,
        "status": "boot_error",
        "reason": None,
        "error": None,
        "calls": 0,
        "stateful_calls": 0,
        "env_steps": 0,
        "env_steps_total": None,
        "policy_wall_s": None,
        "startup_s": None,
        "total_wall_s": None,
        "timeout_s": timeout_s,
        "max_episode_steps": cell.max_episode_steps,
        "prompt": None,
        "host": {
            "python": sys.executable,
            "cuda_device": cell.cuda_device,
            "task_config": cell.suite,
            "boot_options": options,
        },
        "created_at": created_at,
        "notes": [],
    }
    shutil.copyfile(policy_path, out_dir / "policy.py")

    toolkit = None
    daemons: list[Any] = []
    ledger = Ledger()
    try:
        toolkit, daemons = boot(cell, out_dir, rpent_root=rpent_root, **options)
        result["host"]["rpent_root"] = str(resolve_rpent_root(rpent_root))
        result["startup_s"] = time.perf_counter() - started
        backend = make_backend(toolkit)
        card = dump_card(toolkit, cell, out_dir)
        result["task_language"] = card.task_language
        bundle = render_prompt(card, budget_s=int(timeout_s))
        prompt_text = bundle.text()
        (out_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        result["prompt"] = {
            "chars": len(prompt_text),
            "sha256": _sha256_text(prompt_text),
            "images": list(bundle.images),
        }

        robo = RobotwinRobot(backend, ledger=ledger)
        _install_sigterm()
        scratch = out_dir / "workspace"
        scratch.mkdir(exist_ok=True)
        result["host"]["sandbox"] = SANDBOX_DESCRIPTION
        result["policy"]["sandbox_flags"] = audit_source(
            policy_path.read_text(encoding="utf-8", errors="replace"), workspace=scratch
        )
        try:
            policy = load_policy(policy_path, builtins=make_builtins(scratch))
        except PolicyLoadError as exc:
            result["status"] = "error"
            result["reason"] = "load"
            result["error"] = {"type": "PolicyLoadError", "message": str(exc)}
        else:
            with open(out_dir / "policy_output.txt", "w", encoding="utf-8") as handle:
                with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                    outcome = run_policy(policy, robo, timeout_s=timeout_s)
            result["status"] = outcome.status
            result["reason"] = outcome.reason
            result["policy_wall_s"] = outcome.elapsed_s
            if outcome.error_type:
                result["error"] = {
                    "type": outcome.error_type,
                    "message": outcome.error_message,
                    "traceback": outcome.traceback,
                }
        result["environment_success"] = backend.solved()
        result["env_steps_total"] = backend.env_steps()
    except BaseException as exc:  # noqa: BLE001 - the result records boot failures
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if result["status"] == "boot_error":
            result["notes"].append("host failed before the policy started")
        else:
            result["status"] = "error"
        if isinstance(exc, KeyboardInterrupt):
            result["reason"] = "interrupted"
    finally:
        result["calls"] = len(ledger)
        result["stateful_calls"] = ledger.stateful_calls()
        result["env_steps"] = ledger.env_steps()
        ledger.write_jsonl(out_dir / "calls.jsonl")
        stop_all(toolkit, daemons)
        result["total_wall_s"] = time.perf_counter() - started
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return result


def _install_sigterm() -> None:
    def handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise KeyboardInterrupt(f"signal {signum}")

    with contextlib.suppress(ValueError):  # not in the main thread
        signal.signal(signal.SIGTERM, handler)


def card_only(
    cell: Cell,
    out_dir: Path,
    *,
    rpent_root: str | None = None,
    boot_options: dict[str, Any] | None = None,
) -> TaskCard:
    """Boot only the simulator, dump the task card and shut down."""

    options = {k: v for k, v in (boot_options or {}).items() if k != "vla_endpoint"}
    toolkit, daemons = boot(cell, out_dir, rpent_root=rpent_root, env_only=True, **options)
    try:
        return dump_card(toolkit, cell, out_dir)
    finally:
        stop_all(toolkit, daemons)


def _add_cell_args(parser: argparse.ArgumentParser) -> None:
    add_cell_args(parser)
    add_boot_args(parser)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--rpent-root", default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyrualean.hosts.rpent_robotwin", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one policy file for one cell")
    run.add_argument("--policy", type=Path, required=True)
    run.add_argument("--timeout-s", type=float, default=1200.0)
    _add_cell_args(run)
    card = sub.add_parser("card", help="dump the task card for one cell")
    _add_cell_args(card)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_episode(
            args.policy.resolve(),
            cell_from_args(args),
            args.out.resolve(),
            timeout_s=args.timeout_s,
            rpent_root=args.rpent_root,
            boot_options=boot_options(args),
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "reason",
                        "environment_success",
                        "calls",
                        "env_steps",
                        "policy_wall_s",
                    )
                },
                ensure_ascii=False,
            )
        )
        return 0 if result["status"] != "boot_error" else 2
    card_obj = card_only(
        cell_from_args(args),
        args.out.resolve(),
        rpent_root=args.rpent_root,
        boot_options=boot_options(args),
    )
    print(card_obj.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
