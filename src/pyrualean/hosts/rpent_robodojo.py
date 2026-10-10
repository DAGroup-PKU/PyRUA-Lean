# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RPent-backed RoboDojo host.

Boots RPent's own RoboDojo stack (Isaac Sim env server, Pi0.5 VLA server, SAM3
server), builds the same ``RoboDojoToolkit`` its planners use in development
mode, wraps it as a :class:`Backend` and runs one policy file against it.
Because every primitive still goes through ``Toolkit.execute_tool``, the run
produces RPent's native artifacts (``states.json``, the per-step camera
images, one video per camera) and is judged by the same ``env.is_success``
call RPent's runner makes after its planner has finished.

A RoboDojo cell is a task name and a layout id (the environment seed):
``Cell.seed`` is the layout, ``Cell.suite`` the environment config
(``arx_x5``).  The episode's native-action budget is the task's own
``step_lim``; ``Cell.max_episode_steps`` is only handed to RPent's env server,
whose agent path does not enforce it.

RPent's toolkit counts no frames, so this host provides
:class:`RobodojoBackend` through ``make_backend(toolkit)``: its ``env_steps``
is the simulator's native-action counter.

Four tools are this library's own (:mod:`.robodojo_tools`, registered on RPent's
toolkit at boot by :func:`register_home`): ``move_pose`` (position and
orientation with the host's IK), ``grip`` (a full gripper stroke that reports
what it holds), ``locate`` (an object's 3D extent from its mask and the depth
image) and ``home`` (the joints the episode started with).
RPent's toolset moves by position only, and its IK lets the wrist drift.

The host attaches to the simulator without RPent's second reset
(:func:`_connect_without_reset`): tasks with an articulated object do not
survive it.  ``ROBODOJO_CONNECT_RESET=1`` gives RPent's behaviour back.

This module needs RPent (a revision with ``robots/robodojo``, i.e. ``d08d712``
or later) and its ``robodojo`` extra importable; run it with that interpreter
and the RoboDojo environment variables set (``docs/setup.md``).  One episode
per process: ``python -m pyrualean.hosts.rpent_robodojo run ...``.
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
from ..robodojo import RobodojoRobot, _arm_state
from ..runner import PolicyLoadError, load_policy, policy_sha256, run_policy
from ..sandbox import DESCRIPTION as SANDBOX_DESCRIPTION
from ..sandbox import audit_source, make_builtins
from . import robodojo_support_arm_patch as support_patch
from . import robodojo_tools as tools
from .common import RESULT_SCHEMA, RpentBackend, resolve_rpent_root, stop_all

#: RPent's ``--env-cfg-type`` values (``robots/robodojo/robot_spec.py``).
ENV_CONFIGS = ("arx_x5",)
#: RPent's default ``--max-episode-steps`` for this robot.
DEFAULT_MAX_EPISODE_STEPS = 700
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
#: Step-0 artifacts copied next to the card, keyed by the name the card uses.
CARD_IMAGES = {camera: f"{camera}.png" for camera in CAMERAS}
POLICY_BACKENDS = ("rlinf", "xpolicylab")
ACTION_TYPES = ("joint", "ee")
ARMS = ("left", "right")
#: The schema RPent's toolkit keeps for ``home`` (its planners never see it:
#: PyRUA-Lean's arms build their own tool list).
HOME_SPEC: dict[str, Any] = tools.SPECS["home"]


@dataclass(frozen=True)
class Cell:
    """One RoboDojo evaluation cell: a task name, an env config and a layout id.

    ``seed`` is RoboDojo's layout id (``rpent --layout``, which is also the
    environment seed).  ``max_episode_steps`` is nominal for an agent run: the
    budget the simulator enforces is the task's own ``step_lim``, which
    ``dump_card`` reads from the live environment.
    """

    suite: str
    task: str
    seed: int
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    cuda_device: int | None = None

    @property
    def tag(self) -> str:
        return f"{self.task}_l{self.seed}"


def add_cell_args(parser: argparse.ArgumentParser) -> None:
    """RoboDojo cell identifiers (``pyrualean play --robot robodojo`` and the host CLI)."""

    parser.add_argument(
        "--task", required=True, help="RoboDojo task name, e.g. put_bottles_into_dustbin"
    )
    parser.add_argument(
        "--suite", default="arx_x5", choices=ENV_CONFIGS, help="RoboDojo environment config"
    )
    parser.add_argument("--seed", type=int, default=0, help="layout id (RPent's --layout)")


def add_boot_args(parser: argparse.ArgumentParser) -> None:
    """Options of :func:`boot` that are RoboDojo-specific (env-var defaults)."""

    parser.add_argument(
        "--vla-endpoint", default=None, help="[http://]host:port of a running Pi0.5 server"
    )
    parser.add_argument(
        "--sam3-endpoint", default=None, help="[http://]host:port of a running SAM3 server"
    )
    parser.add_argument(
        "--env-endpoint", default=None, help="[http://]host:port of a running env server"
    )
    parser.add_argument(
        "--source-root",
        default=os.environ.get("ROBODOJO_SOURCE_ROOT"),
        help="RoboDojo source tree (default: the one rlinf-robodojo-runtime ships)",
    )
    parser.add_argument(
        "--xpolicylab-root",
        default=os.environ.get("ROBODOJO_XPOLICYLAB_ROOT"),
        help="XPolicyLab checkout (default: SOURCE_ROOT/XPolicyLab)",
    )
    parser.add_argument(
        "--sim-python",
        default=os.environ.get("ROBODOJO_SIM_PYTHON"),
        help="interpreter with Isaac Sim and RoboDojo (default: this one)",
    )
    parser.add_argument(
        "--pi05-python",
        default=os.environ.get("ROBODOJO_PI05_PYTHON"),
        help="interpreter with the Pi0.5 dependencies (default: this one)",
    )
    parser.add_argument(
        "--policy-backend",
        default=os.environ.get("ROBODOJO_POLICY_BACKEND", "rlinf"),
        choices=POLICY_BACKENDS,
        help="Pi0.5 policy backend (RPent's default: rlinf)",
    )
    parser.add_argument(
        "--action-type", default="joint", choices=ACTION_TYPES, help="VLA action representation"
    )
    parser.add_argument(
        "--connect-reset",
        action=argparse.BooleanOptionalAction,
        default=_connect_reset_default(),
        help="reset the episode a second time when connecting to the simulator, as RPent's "
        "development mode does (env: ROBODOJO_CONNECT_RESET=1).  Off by default: tasks with "
        "an articulated object do not survive it, see boot()",
    )


def cell_from_args(args: argparse.Namespace) -> Cell:
    return Cell(
        suite=str(getattr(args, "suite", None) or "arx_x5"),
        task=str(args.task),
        seed=int(args.seed),
        max_episode_steps=int(getattr(args, "max_episode_steps", DEFAULT_MAX_EPISODE_STEPS)),
        cuda_device=getattr(args, "cuda_device", None),
    )


def _connect_reset_default() -> bool:
    return os.environ.get("ROBODOJO_CONNECT_RESET", "0").strip().lower() in ("1", "true", "yes")


@contextlib.contextmanager
def _connect_without_reset():
    """Make RPent's RoboDojo client attach to the simulator without resetting it again.

    The env server resets to the requested layout while it starts.  RPent's
    development-mode client then resets once more on connect (its eval-fair
    client does not), and that second reset fails for every scene that holds
    an articulated object: the layout manager gives the object a new instance
    name while the stage keeps the one it spawned (``Object toaster_4_33 of
    type articulation not found in env 0``), so the stability check raises.
    Seven of the 54 tasks at layout 0 cannot boot for this reason (RPent
    068cd64).  Skipping the second reset starts the episode from the state the
    eval-fair client starts from.
    """

    from robots.robodojo import env_client
    from rpent.robots.components.env_client_base import BaseEnvClient

    original = env_client.RoboDojoEnvClient.__init__

    def connect(self, client, *, expected_meta):
        self.eval_fair = expected_meta.get("mode") == "eval-fair"
        BaseEnvClient.__init__(self, client, expected_meta=expected_meta, reset_on_connect=False)
        # What the skipped reset would have cached.
        self.last_obs = self.get_obs()

    env_client.RoboDojoEnvClient.__init__ = connect
    try:
        yield
    finally:
        env_client.RoboDojoEnvClient.__init__ = original


def ensure_support_arm() -> None:
    """Refuse a RoboDojo runtime whose collect client never moves the support arm.

    In ``play_tic_tac_toe``, ``make_kong`` and ``imitate_sorting_sequence`` the benchmark drives a
    third arm (an opponent, a demonstrator); the client RPent steps the simulator through never
    queues it, so those tasks cannot be solved (:mod:`pyrualean.hosts.robodojo_support_arm_patch`).
    ``ROBODOJO_SUPPORT_ARM=upstream`` runs the runtime as installed; a runtime that is not
    importable from this interpreter cannot be inspected and is let through.
    """

    if os.environ.get("ROBODOJO_SUPPORT_ARM", "").strip().lower() == "upstream":
        return
    if support_patch.status() == "upstream":
        raise RuntimeError(
            "RoboDojo's collect client never moves the support arm of the interactive tasks: run "
            "`python -m pyrualean.hosts.robodojo_support_arm_patch` with this interpreter "
            "(docs/setup.md), or set ROBODOJO_SUPPORT_ARM=upstream to run it as installed"
        )


def boot_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vla_endpoint": getattr(args, "vla_endpoint", None),
        "sam3_endpoint": getattr(args, "sam3_endpoint", None),
        "env_endpoint": getattr(args, "env_endpoint", None),
        "source_root": getattr(args, "source_root", None),
        "xpolicylab_root": getattr(args, "xpolicylab_root", None),
        "sim_python": getattr(args, "sim_python", None),
        "pi05_python": getattr(args, "pi05_python", None),
        "policy_backend": getattr(args, "policy_backend", None) or "rlinf",
        "action_type": getattr(args, "action_type", None) or "joint",
        "connect_reset": getattr(args, "connect_reset", None),
    }


def _status(toolkit: Any) -> dict[str, Any]:
    """The env server's episode status (``step``, ``step_limit``); ``{}`` if unreadable."""

    try:
        status = toolkit._primitives.env.get_status()
    except Exception:  # noqa: BLE001 - optional metadata
        return {}
    return status if isinstance(status, dict) else {}


class RobodojoBackend(RpentBackend):
    """RPent's RoboDojo toolkit as a :class:`pyrualean.Backend`.

    ``solved()`` is the toolkit's own (``env.is_success``, the call RPent's
    runner makes once its planner is done); the host reads it after the
    episode, the robot class never does.  ``env_steps`` is the simulator's
    native-action counter: RoboDojo's toolkit records no frame count.
    """

    def env_steps(self) -> int | None:
        step = _status(self._toolkit).get("step")
        try:
            return int(step)
        except (TypeError, ValueError):
            return None


def make_backend(toolkit: Any) -> RobodojoBackend:
    """The backend ``pyrualean.play`` wraps a booted RoboDojo toolkit in."""

    return RobodojoBackend(toolkit)


#: ``robo.home`` and the other tools of this library's own (:mod:`.robodojo_tools`).
home = tools.home
register_home = tools.register


def boot(
    cell: Cell,
    out_dir: Path,
    *,
    rpent_root: str | None = None,
    env_only: bool = False,
    vla_endpoint: str | None = None,
    sam3_endpoint: str | None = None,
    env_endpoint: str | None = None,
    source_root: str | None = None,
    xpolicylab_root: str | None = None,
    sim_python: str | None = None,
    pi05_python: str | None = None,
    policy_backend: str = "rlinf",
    action_type: str = "joint",
    connect_reset: bool | None = None,
) -> tuple[Any, list[Any]]:
    """Start RPent's servers and return ``(toolkit, daemons)``.

    ``env_only`` boots just the simulator (enough to dump a task card); the
    VLA and SAM3 clients are then ``None`` and ``pi0_pick`` / ``segment``
    answer with an error.  ``vla_endpoint`` / ``sam3_endpoint`` /
    ``env_endpoint`` attach to already running servers instead of spawning
    private ones (RPent's own flags).  ``sim_python`` / ``pi05_python`` name
    the interpreters of the simulator and the policy server when they are not
    this one; ``source_root`` overrides the RoboDojo tree the runtime wheel
    ships.  The toolkit is RPent's development-mode toolkit (every robot tool,
    no scoring or safety diagnostics), not the eval-fair replay toolkit.
    ``connect_reset`` (default: off; on with ``ROBODOJO_CONNECT_RESET=1``, as
    RPent's development mode) is the client's second reset of an already
    reset episode, which tasks with an articulated object do not survive: see
    :func:`_connect_without_reset`.
    """

    if policy_backend not in POLICY_BACKENDS:
        raise ValueError(f"policy_backend must be one of {POLICY_BACKENDS}")
    if action_type not in ACTION_TYPES:
        raise ValueError(f"action_type must be one of {ACTION_TYPES}")
    resolve_rpent_root(rpent_root)
    from robots.robodojo.robot_spec import get_robot_spec
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
        task=cell.task,
        layout=cell.seed,
        env_cfg_type=cell.suite,
        action_type=action_type,
        max_episode_steps=cell.max_episode_steps,
        cuda_device=cell.cuda_device,
        random=False,
        source_root=source_root,
        xpolicylab_root=xpolicylab_root,
        sim_python=sim_python or sys.executable,
        pi05_python=pi05_python or sys.executable,
        policy_backend=policy_backend,
        env_endpoint=env_endpoint,
        vla_endpoint=vla_endpoint,
        sam3_endpoint=sam3_endpoint,
        # RPent switches the env server and the toolkit to eval-fair replay when
        # the planner is "flash"; a code policy runs in development mode.
        planner=None,
        memory_dir=str(memory_dir),
        output_dir=str(out_dir),
    )
    spec = get_robot_spec()
    components = {"env"} if env_only else None
    own_sim = not sim_python or os.path.abspath(sim_python) == os.path.abspath(sys.executable)
    if env_endpoint is None and own_sim:
        ensure_support_arm()  # the simulator runs this interpreter's RoboDojo runtime
    if connect_reset is None:
        connect_reset = _connect_reset_default()
    with contextlib.nullcontext() if connect_reset else _connect_without_reset():
        daemons, runtime_kwargs = spec.init_runtime(args, out_dir, events, components)
    from robots.robodojo.toolkit import RoboDojoToolkit

    try:
        # What RPent's ``get_toolkit`` does, minus the RunConfig it only reads
        # ``memory_dir`` and the mode from.
        toolkit = RoboDojoToolkit(
            primitives_kwargs=runtime_kwargs,
            dashboard_events=events,
            memory=MemoryManager(root=memory_dir),
            eval_fair=False,
        )
        register_home(toolkit)
    except BaseException:
        for daemon in reversed(daemons):
            with contextlib.suppress(Exception):
                daemon.stop()
        raise
    return toolkit, daemons


def _fmt(values: Any) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


def _opening(value: Any) -> float:
    """A gripper opening recorded as ``[0.97]`` (or a one-value numpy array)."""
    for _ in range(4):
        if value is None or isinstance(value, (int, float)):
            break
        try:
            value = next(iter(value))
        except (TypeError, StopIteration):
            break
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def dump_card(toolkit: Any, cell: Cell, out_dir: Path) -> TaskCard:
    """Write ``card.json`` plus the step-0 images and return the card.

    The facts are what RPent's agent also sees in its first
    ``view_env_state``, in the terms ``move_to`` takes: both arms' initial
    grasp points, gripper angles and openings.  The native-action budget is
    the simulator's ``step_lim`` (RPent's agent meets it as ``step_limit`` in
    its tool results).  Nothing
    privileged: no object poses, no score.
    """

    view = toolkit.execute_tool("view_env_state", {"step": 0}).result
    summary = view.get("state") or {}
    raw = summary.get("state") or {}
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
        start = _arm_state(raw, arm)
        facts[f"{arm} arm initial grasp point [x, y, z]"] = f"{_fmt(start.tcp_pos)} m"
        facts[f"{arm} gripper initial pitch / yaw / roll"] = (
            f"{round(start.pitch)} / {round(start.yaw)} / {round(start.roll)} deg"
        )
        opening = _opening(raw.get(f"{arm}_ee_joint_state"))
        facts[f"{arm} gripper opening"] = f"{opening:.2f} (1 = open, 0 = closed)"
    limit = _status(toolkit).get("step_limit")
    try:
        budget = int(limit)
    except (TypeError, ValueError):
        budget = cell.max_episode_steps
    facts["native action budget"] = f"{budget} (a motion call spends one or more)"
    card = TaskCard(
        suite=cell.suite,
        task=cell.task,
        seed=cell.seed,
        task_language=str(view.get("task_language") or summary.get("instruction") or ""),
        object_names=[],
        images=images,
        max_env_steps=budget,
        robot="robodojo",
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
        "backend": "robodojo",
        "robot": "robodojo",
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
            "layout": cell.seed,
            "env_cfg_type": cell.suite,
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
        result["host"]["step_limit"] = card.max_env_steps
        bundle = render_prompt(card, budget_s=int(timeout_s))
        prompt_text = bundle.text()
        (out_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        result["prompt"] = {
            "chars": len(prompt_text),
            "sha256": _sha256_text(prompt_text),
            "images": list(bundle.images),
        }

        robo = RobodojoRobot(backend, ledger=ledger)
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
        # The official verdict, read once the policy is done (as RPent's runner does).
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

    skip = ("vla_endpoint", "sam3_endpoint")
    options = {k: v for k, v in (boot_options or {}).items() if k not in skip}
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
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--rpent-root", default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyrualean.hosts.rpent_robodojo", description=__doc__
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
