# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RPent-backed LIBERO host.

Boots RPent's own LIBERO stack (env server, Pi0.5 server, SAM3 server), builds
the same ``LiberoToolkit`` its planners use, wraps it as a :class:`Backend`
and runs one policy file against it.  Because every primitive still goes
through ``Toolkit.execute_tool``, the run produces RPent's native artifacts
(``states.json``, ``episode.mp4``, per-step images) and is judged by the same
``toolkit.solved()`` predicate as an RPent agent run.

This module needs RPent and its ``libero-pro`` extra importable; run it with
RPent's interpreter after sourcing its environment (see docs/guide.md).  One
episode per process: ``python -m pyrualean.hosts.rpent_libero run ...``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import signal
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._backend import Ledger
from ..libero import LiberoRobot
from ..prompt import TaskCard, render_prompt
from ..runner import PolicyLoadError, load_policy, policy_sha256, run_policy
from ..sandbox import DESCRIPTION as SANDBOX_DESCRIPTION
from ..sandbox import audit_source, make_builtins
from .common import RESULT_SCHEMA, RpentBackend, resolve_rpent_root

CARD_IMAGES = ("agentview_high.png", "wrist_high.png")


@dataclass(frozen=True)
class Cell:
    """One LIBERO evaluation cell."""

    suite: str
    task: int
    seed: int
    max_episode_steps: int = 10000
    libero_type: str = "pro"
    cuda_device: int | None = None

    @property
    def tag(self) -> str:
        return f"{self.suite.replace('libero_', '')}_t{self.task}_s{self.seed}"


def add_cell_args(parser: argparse.ArgumentParser) -> None:
    """LIBERO cell identifiers (``pyrualean play --robot libero`` and the host CLI)."""

    parser.add_argument("--suite", required=True, help="e.g. libero_object_swap")
    parser.add_argument("--task", type=int, required=True, help="task index within the suite")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--libero-type", default="pro")


def add_boot_args(parser: argparse.ArgumentParser) -> None:
    """Options of :func:`boot` that are LIBERO-specific."""

    parser.add_argument("--vla-endpoint", default=None, help="host:port of a shared Pi0.5 server")
    parser.add_argument("--sam3-endpoint", default=None, help="host:port of a shared SAM3 server")


def cell_from_args(args: argparse.Namespace) -> Cell:
    return Cell(
        suite=str(args.suite),
        task=int(args.task),
        seed=int(args.seed),
        max_episode_steps=int(getattr(args, "max_episode_steps", 10000)),
        libero_type=str(getattr(args, "libero_type", "pro")),
        cuda_device=getattr(args, "cuda_device", None),
    )


def boot_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vla_endpoint": getattr(args, "vla_endpoint", None),
        "sam3_endpoint": getattr(args, "sam3_endpoint", None),
    }


def boot(
    cell: Cell,
    out_dir: Path,
    *,
    rpent_root: str | None = None,
    env_only: bool = False,
    vla_endpoint: str | None = None,
    sam3_endpoint: str | None = None,
) -> tuple[Any, list[Any]]:
    """Start RPent's servers and return ``(toolkit, daemons)``.

    ``env_only`` boots just the simulator (enough to dump a task card); the
    VLA and SAM3 clients are then ``None`` and their primitives unusable.
    ``vla_endpoint`` / ``sam3_endpoint`` (``host:port``) attach to shared,
    already running Pi0.5 / SAM3 servers instead of spawning private ones
    (RPent's own ``--vla-endpoint`` / ``--sam3-endpoint``); the simulator is
    always private to the episode.
    """

    resolve_rpent_root(rpent_root)
    from robots.libero.robot_spec import get_robot_spec, get_toolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.robots.robot_spec import RunConfig
    from rpent.utils.logging import init_output_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    init_output_dir(out_dir, verbose=False)
    events = NullDashboardEventSink()
    args = Namespace(
        suite=cell.suite,
        task=cell.task,
        seed=cell.seed,
        max_episode_steps=cell.max_episode_steps,
        libero_type=cell.libero_type,
        cuda_device=cell.cuda_device,
        env_endpoint=None,
        vla_endpoint=vla_endpoint,
        sam3_endpoint=sam3_endpoint,
        molmo_endpoint=None,
        planner="api",
        collect_flywheel_data=False,
        flywheel_root=None,
    )
    spec = get_robot_spec()
    components = {"env"} if env_only else None
    daemons, runtime_kwargs = spec.init_runtime(args, out_dir, events, components)
    if env_only:
        runtime_kwargs.setdefault("model", None)
        runtime_kwargs.setdefault("sam3_client", None)
    memory_dir = out_dir / "_memory"
    memory_dir.mkdir(exist_ok=True)
    config = RunConfig(
        recipe_tag=cell.tag,
        output_dir=out_dir,
        prompt_vars={
            "suite": cell.suite,
            "task": cell.task,
            "seed": cell.seed,
            "recipe_tag": cell.tag,
            "memory_dir": str(memory_dir),
        },
        task_desc={"suite": cell.suite, "task": cell.task, "seed": cell.seed},
    )
    try:
        toolkit = get_toolkit(
            runtime_kwargs=runtime_kwargs,
            dashboard_events=events,
            config=config,
            mode="evaluation",
            attempts_per_session=0,
            state_output_dir=out_dir,
        )
    except BaseException:
        for daemon in reversed(daemons):
            with contextlib.suppress(Exception):
                daemon.stop()
        raise
    return toolkit, daemons


def dump_card(toolkit: Any, cell: Cell, out_dir: Path) -> TaskCard:
    """Write ``card.json`` plus the step-0 images and return the card."""

    view = toolkit.execute_tool("view_env_state", {"step": 0}).result
    raw = view.get("state") or {}
    qpos = [float(v) for v in raw.get("robot0_gripper_qpos", (0.0, 0.0))]
    card_dir = out_dir / "card"
    card_dir.mkdir(exist_ok=True)
    images: dict[str, str] = {}
    for name in CARD_IMAGES:
        try:
            source = toolkit.state.artifact_path(name, step=0)
        except Exception:  # noqa: BLE001 - optional artifact
            continue
        if source.is_file():
            target = card_dir / name
            shutil.copyfile(source, target)
            images[name.split("_")[0]] = str(target)
    card = TaskCard(
        suite=cell.suite,
        task=cell.task,
        seed=cell.seed,
        task_language=str(view.get("task_language") or ""),
        object_names=[str(n) for n in raw.get("object_names", [])],
        eef_pos=[float(v) for v in raw.get("robot0_eef_pos", (0.0, 0.0, 0.0))],
        eef_quat=[float(v) for v in raw.get("robot0_eef_quat", (0.0, 0.0, 0.0, 1.0))],
        gripper_opening=abs(qpos[0]) + abs(qpos[1]) if len(qpos) >= 2 else 0.0,
        images=images,
        max_env_steps=cell.max_episode_steps,
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
) -> dict[str, Any]:
    """Run one policy file for one cell and write ``result.json``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "system": "pyrualean",
        "backend": "libero",
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
            "libero_type": cell.libero_type,
        },
        "created_at": created_at,
        "notes": [],
    }
    shutil.copyfile(policy_path, out_dir / "policy.py")

    toolkit = None
    daemons: list[Any] = []
    ledger = Ledger()
    try:
        toolkit, daemons = boot(cell, out_dir, rpent_root=rpent_root)
        result["host"]["rpent_root"] = str(resolve_rpent_root(rpent_root))
        result["startup_s"] = time.perf_counter() - started
        backend = RpentBackend(toolkit)
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

        robo = LiberoRobot(backend, ledger=ledger)
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
        result["environment_success"] = bool(toolkit.solved())
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
        if toolkit is not None:
            with contextlib.suppress(Exception):
                toolkit.close()
        for daemon in reversed(daemons):
            with contextlib.suppress(Exception):
                daemon.stop()
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


def card_only(cell: Cell, out_dir: Path, *, rpent_root: str | None = None) -> TaskCard:
    """Boot only the simulator, dump the task card and shut down."""

    toolkit, daemons = boot(cell, out_dir, rpent_root=rpent_root, env_only=True)
    try:
        return dump_card(toolkit, cell, out_dir)
    finally:
        with contextlib.suppress(Exception):
            toolkit.close()
        for daemon in reversed(daemons):
            with contextlib.suppress(Exception):
                daemon.stop()


def _add_cell_args(parser: argparse.ArgumentParser) -> None:
    add_cell_args(parser)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--rpent-root", default=None)


def _cell(args: argparse.Namespace) -> Cell:
    return cell_from_args(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyrualean.hosts.rpent_libero", description=__doc__
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
            _cell(args),
            args.out.resolve(),
            timeout_s=args.timeout_s,
            rpent_root=args.rpent_root,
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
    card_obj = card_only(_cell(args), args.out.resolve(), rpent_root=args.rpent_root)
    print(card_obj.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
