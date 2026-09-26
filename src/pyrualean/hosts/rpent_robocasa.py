# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RPent-backed RoboCasa365 host.

Boots RPent's own RoboCasa stack (robosuite kitchen env server, RLDX-1 VLA
server), builds the same ``RoboCasaToolkit`` its planners use, wraps it as a
:class:`Backend` and runs one policy file against it.  Because every
primitive still goes through ``Toolkit.execute_tool``, the run produces
RPent's native artifacts (``states.json``, ``episode.mp4``,
``success_criteria.md``, per-step images and world maps) and is judged by
the same ``toolkit.solved()`` predicate (the env's own ``_check_success``,
recorded as ``success``) as an RPent agent run.

``RoboCasaToolkit`` implements ``solved()`` and counts env steps as recorded
agentview frames (one per ``env.step``, calibration and VLA steps included),
the same unit as LIBERO, so the generic :class:`RpentBackend` is used
unchanged and this module defines no ``make_backend``.

This module needs RPent and its ``robocasa`` extra importable; run it with
that interpreter and the RoboCasa environment variables set
(``docs/setup.md``).  One episode per process:
``python -m pyrualean.hosts.rpent_robocasa run ...``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
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
from ..robocasa import RobocasaRobot
from ..runner import PolicyLoadError, load_policy, policy_sha256, run_policy
from ..sandbox import DESCRIPTION as SANDBOX_DESCRIPTION
from ..sandbox import audit_source, make_builtins
from .common import RESULT_SCHEMA, RpentBackend, resolve_rpent_root, stop_all

#: RPent's RoboCasa data splits (``robots/robocasa/robot_spec.py``); Target50 uses ``target``.
SPLITS = ("target", "pretrain", "all")
CAMERAS = ("agentview", "navview", "wrist")
#: Step-0 artifacts copied next to the card, keyed by the name the card uses.
CARD_IMAGES = {camera: f"{camera}.png" for camera in CAMERAS}
#: The tracked empty memory corpus (the one the tool-calling arm's zero-shot
#: runs pass to RPent); used when the package runs from this repository.
EMPTY_MEMORY = Path(__file__).resolve().parents[3] / "runs" / "_empty-memory-robocasa"


@dataclass(frozen=True)
class Cell:
    """One RoboCasa365 evaluation cell: a task name, a data split and an exact scene seed.

    ``max_episode_steps`` is nominal: RPent enforces no simulator step budget
    for this robot (``truncated`` never fires); the host's wall-clock and
    decision budgets end an episode.
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
    """RoboCasa cell identifiers (``pyrualean play --robot robocasa`` and the host CLI)."""

    parser.add_argument("--task", required=True, help="RoboCasa365 task name, e.g. OpenDrawer")
    parser.add_argument(
        "--suite",
        default="target",
        choices=SPLITS,
        help="RoboCasa data split (target = the Target50 evaluation setting)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="scene seed (Target50: 1-10 for atomic tasks, 1-5 for composite ones)",
    )


def add_boot_args(parser: argparse.ArgumentParser) -> None:
    """Options of :func:`boot` that are RoboCasa-specific (env-var defaults as in RPent)."""

    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help="[protocol://]host:port of a shared, already running RLDX-1 server",
    )
    parser.add_argument(
        "--env-endpoint", default=None, help="http://host:port of an already running env server"
    )
    parser.add_argument(
        "--vla-model-path",
        default=os.environ.get("RLDX_MODEL_PATH"),
        help="RLDX-1 checkpoint directory (default: $RLDX_MODEL_PATH)",
    )
    parser.add_argument(
        "--robocasa-assets-path",
        default=os.environ.get("ROBOCASA_ASSETS_PATH"),
        help="RoboCasa asset root exported to the env server (default: $ROBOCASA_ASSETS_PATH)",
    )
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="empty memory corpus for RPent's toolkit (default: runs/_empty-memory-robocasa)",
    )


def cell_from_args(args: argparse.Namespace) -> Cell:
    return Cell(
        suite=str(getattr(args, "suite", None) or "target"),
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
        "robocasa_assets_path": getattr(args, "robocasa_assets_path", None),
        "memory_dir": getattr(args, "memory_dir", None),
    }


def memory_root(out_dir: Path, memory_dir: str | None = None) -> Path:
    """An empty, read-only memory corpus for the toolkit (zero-shot: nothing is read).

    ``memory_dir`` when given, else the tracked ``runs/_empty-memory-robocasa``
    when this package runs from its repository, else a private empty corpus
    under the run directory.  RPent's toolkit only binds its file tools to
    it; PyRUA-Lean never exposes those, so nothing is read or written.
    """

    if memory_dir:
        root = Path(memory_dir).expanduser().resolve()
    elif (EMPTY_MEMORY / "MEMORY.md").is_file():
        root = EMPTY_MEMORY
    else:
        root = out_dir / "_memory"
        root.mkdir(parents=True, exist_ok=True)
        (root / "MEMORY.md").touch()
    if not (root / "MEMORY.md").is_file():
        raise FileNotFoundError(f"memory corpus without a MEMORY.md: {root}")
    return root


def boot(
    cell: Cell,
    out_dir: Path,
    *,
    rpent_root: str | None = None,
    env_only: bool = False,
    vla_endpoint: str | None = None,
    env_endpoint: str | None = None,
    vla_model_path: str | None = None,
    robocasa_assets_path: str | None = None,
    memory_dir: str | None = None,
) -> tuple[Any, list[Any]]:
    """Start RPent's servers and return ``(toolkit, daemons)``.

    ``env_only`` boots just the simulator (enough to dump a task card); the
    VLA client is then ``None`` and ``rldx_skill`` / ``rldx_arm`` unusable.
    ``vla_endpoint`` attaches to a shared, already running RLDX-1 server
    instead of spawning a private one (RPent's own ``--vla-endpoint``; the
    server keeps one session per client); ``env_endpoint`` likewise for the
    simulator, which is otherwise private to the episode (it is task/seed
    specific).  ``vla_model_path`` defaults to ``RLDX_MODEL_PATH``,
    ``robocasa_assets_path`` is exported as ``ROBOCASA_ASSETS_PATH`` for the
    env server (the only way robocasa finds its assets).

    A private VLA server resolves its backbone support files from the local
    Hugging Face cache: RLDX asks for revision ``main`` regardless of the
    pinned backbone, evaluation runs with no proxy, and each online retry
    would cost the child its whole boot budget, so ``HF_HUB_OFFLINE`` /
    ``TRANSFORMERS_OFFLINE`` are set (when unset) before spawning it.
    """

    resolve_rpent_root(rpent_root)
    from robots.robocasa.robot_spec import get_robot_spec, get_toolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.robots.robot_spec import RunConfig
    from rpent.utils.logging import init_output_dir

    vla_model_path = vla_model_path or os.environ.get("RLDX_MODEL_PATH")
    if robocasa_assets_path:
        os.environ["ROBOCASA_ASSETS_PATH"] = str(Path(robocasa_assets_path).expanduser())
    if not env_only and vla_endpoint is None:
        if not vla_model_path:
            raise ValueError(
                "vla_model_path (or $RLDX_MODEL_PATH) is required to spawn the RLDX-1 server; "
                "pass vla_endpoint to attach to a shared one"
            )
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # The env server (which copies this environment) builds the kitchen; some of its
    # choices follow Python's string hashing, so a (task, seed) scene is the same from
    # run to run only with a fixed hash seed (and the counter.py fix, docs/setup.md).
    os.environ["PYTHONHASHSEED"] = "0"
    out_dir.mkdir(parents=True, exist_ok=True)
    # ``RoboCasaToolkit`` roots its ``EnvState`` at ``get_output_dir()``, the
    # module global this call sets; without it artifacts land in <repo>/logs.
    init_output_dir(out_dir, verbose=False)
    memory = memory_root(out_dir, memory_dir)
    events = NullDashboardEventSink()
    args = Namespace(
        task_name=cell.task,
        split=cell.suite,
        seed=cell.seed,
        hi_res=0,
        env_endpoint=env_endpoint,
        vla_endpoint=vla_endpoint,
        vla_model_path=vla_model_path,
        cuda_device=cell.cuda_device,
        max_episode_steps=cell.max_episode_steps,
        memory_dir=str(memory),
        output_dir=str(out_dir),
    )
    spec = get_robot_spec()
    components = {"env"} if env_only else None
    daemons, runtime_kwargs = spec.init_runtime(args, out_dir, events, components)
    if env_only:
        runtime_kwargs.setdefault("vla_client", None)
    config = RunConfig(
        recipe_tag=cell.tag,
        output_dir=out_dir,
        prompt_vars={
            "task_name": cell.task,
            "split": cell.suite,
            "seed": cell.seed,
            "recipe_tag": cell.tag,
            "memory_dir": str(memory),
        },
        task_desc={"task_name": cell.task, "split": cell.suite, "seed": cell.seed},
    )
    try:
        toolkit = get_toolkit(runtime_kwargs=runtime_kwargs, dashboard_events=events, config=config)
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


def _yaw(quat_xyzw: list[float]) -> float:
    x, y, z, w = quat_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def dump_card(toolkit: Any, cell: Cell, out_dir: Path) -> TaskCard:
    """Write ``card.json`` plus the step-0 images and return the card.

    The facts are what RPent's agent also sees in its first
    ``view_env_state``: the initial base and gripper poses, the finger
    opening and the step-0 ``task_progress``.  Nothing privileged (no object
    poses; RoboCasa names no objects).
    """

    view = toolkit.execute_tool("view_env_state", {"step": 0}).result
    raw = view.get("state") or {}
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
    base = _floats(raw.get("robot0_base_pos"), 3)
    base_quat = _floats(raw.get("robot0_base_quat"), 4)
    eef = _floats(raw.get("robot0_eef_pos"), 3)
    eef_quat = _floats(raw.get("robot0_eef_quat"), 4)
    qpos = _floats(raw.get("robot0_gripper_qpos"), 2)
    yaw = _yaw(base_quat)
    facts: dict[str, Any] = {
        "initial base position [x, y, z]": f"{_fmt(base)} m",
        "initial base heading (yaw about world z)": f"{yaw:.3f} rad ({math.degrees(yaw):.1f} deg)",
        "initial gripper (EEF) position [x, y, z]": f"{_fmt(eef)} m",
        "initial gripper orientation [x, y, z, w]": _fmt(eef_quat),
        "initial gripper opening (|q0| + |q1|)": (
            f"{abs(qpos[0]) + abs(qpos[1]):.3f} (fingers half open; about 0.08 fully open)"
        ),
    }
    progress = view.get("task_progress")
    if isinstance(progress, dict):
        facts["task_progress at step 0 (the success predicate's live values)"] = json.dumps(
            progress, sort_keys=True
        )
    card = TaskCard(
        suite=cell.suite,
        task=cell.task,
        seed=cell.seed,
        task_language=str(view.get("task_language") or ""),
        object_names=[],
        images=images,
        max_env_steps=cell.max_episode_steps,
        robot="robocasa",
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
        "backend": "robocasa",
        "robot": "robocasa",
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
            "split": cell.suite,
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

        robo = RobocasaRobot(backend, ledger=ledger)
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
        result["environment_success"] = bool(backend.solved())
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
        prog="python -m pyrualean.hosts.rpent_robocasa", description=__doc__
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
