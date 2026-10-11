# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboDojo through its own evaluation client and XPolicyLab.

The host of ``--robot robodojo_xpl``.  RoboDojo's evaluation runs a policy as two processes
that talk XPolicyLab's websocket protocol: the evaluation client owns the simulator
(``src/eval_client/main.py``), builds the observation (three colour images and the robot's
state; a depth image and the camera matrices where its config switches them on, which this
host does for the client it starts) and counts the task's own action budget; a policy server
answers ``update_obs`` / ``get_action``.  This host is the policy server.  :func:`boot`

*   builds the directory the client runs from, with the adapter's own ``client_root.py``
    (``integrations/xpolicylab/PyRUA_Lean``, the directory a submission adds as
    ``policy/PyRUA_Lean``): RoboDojo's code and configuration copied, XPolicyLab's code and
    the adapter beside them, ``Assets`` linked;
*   serves a :class:`.xpolicylab_bridge.Bridge` with XPolicyLab's policy server, in a thread;
*   starts the evaluation client on one task and one layout (``scripts/eval_policy.sh``, the
    script RoboDojo's own CLI ends in), which connects to that server;
*   and returns a :class:`.robodojo_native.NativeToolkit` over the bridge once the first
    observation has arrived.

What the client decides stays the client's: when the task is solved it ends the episode at
once, and when the program has finished it judges the scene as it stands
(``backend.conclude()``).  ``Cell.seed`` is the layout; ``eval_seed`` (0, 1 or 2) picks which
of RoboDojo's three layout sets it is taken from.  The one layout is picked the way
XPolicyLab's ``scripts/run_robodojo_layout_range.sh`` picks layouts: a resume manifest that
leaves every other one out.

Needs: a RoboDojo checkout with its XPolicyLab submodule (``--source-root`` /
``ROBODOJO_SOURCE_ROOT``; ``Assets`` in it, or ``ROBODOJO_ASSETS_ROOT``), ``msgpack-numpy``
for both sides and ``websockets >= 13`` for the server.  ``ROBODOJO_XPL_SITE`` may name a
directory with ``common/`` (what both sides lack) and ``server/`` (what only the server may
see): the simulator's environment keeps the websockets it was built with.  Without a
checkout the tree ``rlinf-robodojo-runtime`` ships is used (an older client), with the
XPolicyLab checkout ``--xpolicylab-root`` / ``ROBODOJO_XPOLICYLAB_ROOT`` names.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..prompt import TaskCard
from ..robodojo import _arm_state
from . import robodojo_tools as tools
from .robodojo_native import CAMERAS, NativeBackend, NativeToolkit
from .rpent_robodojo import Cell, _fmt, _opening, add_cell_args, cell_from_args
from .xpolicylab_bridge import Bridge, BridgeEnv, serve

POLICY_NAME = "PyRUA_Lean"
#: The robot this host serves (the colour-only wrapper names its own).
ROBOT = "robodojo_xpl"
#: The adapter's files, relative to the repository root.
ADAPTER_DIR = Path("integrations") / "xpolicylab" / POLICY_NAME
EVAL_SEEDS = (0, 1, 2)
BOOT_TIMEOUT_S = 1500.0


def add_boot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-root",
        default=os.environ.get("ROBODOJO_SOURCE_ROOT"),
        help="RoboDojo checkout, XPolicyLab inside (default: what rlinf-robodojo-runtime ships)",
    )
    parser.add_argument(
        "--xpolicylab-root",
        default=os.environ.get("ROBODOJO_XPOLICYLAB_ROOT"),
        help="XPolicyLab checkout (default: the one inside the RoboDojo checkout)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=int(os.environ.get("ROBODOJO_EVAL_SEED", "0")),
        choices=EVAL_SEEDS,
        help="which of RoboDojo's three layout sets --seed (the layout) is taken from",
    )


def boot_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_root": getattr(args, "source_root", None),
        "xpolicylab_root": getattr(args, "xpolicylab_root", None),
        "eval_seed": int(getattr(args, "eval_seed", 0) or 0),
    }


def _source_root(explicit: str | None) -> Path:
    explicit = explicit or os.environ.get("ROBODOJO_SOURCE_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        try:
            import robodojo_runtime  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FileNotFoundError(
                "a RoboDojo checkout: pass --source-root or set ROBODOJO_SOURCE_ROOT"
            ) from exc
        root = Path(robodojo_runtime.__file__).resolve().parent / "_source"
    missing = [part for part in client_root().SOURCE_PARTS if not (root / part).exists()]
    if missing or not (root / "src" / "eval_client" / "main.py").is_file():
        raise FileNotFoundError(f"not a RoboDojo source tree with its evaluation client: {root}")
    return root


def _xpolicylab_root(explicit: str | None, source: Path | None = None) -> Path:
    value = explicit or os.environ.get("ROBODOJO_XPOLICYLAB_ROOT")
    if not value and source is not None and (source / "XPolicyLab" / "client_server").is_dir():
        value = str(source / "XPolicyLab")  # the submodule of a RoboDojo checkout
    if not value:
        raise FileNotFoundError(
            "an XPolicyLab checkout: the XPolicyLab directory of the RoboDojo checkout, "
            "--xpolicylab-root or ROBODOJO_XPOLICYLAB_ROOT"
        )
    root = Path(value).expanduser().resolve()
    if not (root / "client_server" / "ws" / "model_server.py").is_file():
        raise FileNotFoundError(f"not an XPolicyLab checkout: {root}")
    return root


def adapter_dir() -> Path:
    """This library's XPolicyLab adapter (``policy/PyRUA_Lean`` of a submission)."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ADAPTER_DIR / "deploy.py").is_file():
            return parent / ADAPTER_DIR
    raise FileNotFoundError(f"{ADAPTER_DIR} not found above {here}")


@functools.lru_cache(maxsize=1)
def client_root() -> Any:
    """The adapter's ``client_root.py``: the one place that builds the client's directory."""

    path = adapter_dir() / "client_root.py"
    spec = importlib.util.spec_from_file_location("pyrualean_xpolicylab_client_root", path)
    if spec is None or spec.loader is None:  # pragma: no cover - a broken checkout
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def with_depth(env_cfg: Path, suite: str) -> None:
    """Switch the depth image and the camera matrices on in a copy of RoboDojo's ``env_cfg``."""

    client_root().with_depth(env_cfg, suite)


def lay_out(
    bench: Path, source: Path, xpolicylab: Path, *, depth: bool = False, suite: str = "arx_x5"
) -> Path:
    """The directory the evaluation client runs from: RoboDojo's code and configuration
    copied from ``source``, XPolicyLab's code and the adapter in ``XPolicyLab/``, ``Assets``
    linked.  The client writes ``eval_result/`` into it.

    With ``depth`` the copy of ``env_cfg`` has the depth image and the camera matrices
    switched on; RoboDojo's tree is not touched.
    """

    return client_root().lay_out(
        bench,
        source,
        depth=depth,
        env_cfg_type=suite,
        xpolicylab=xpolicylab,
        policy_dir=adapter_dir(),
    )


def only_layout(
    bench: Path, cell: Cell, *, eval_seed: int, additional_info: str, run_id: str
) -> Path:
    """Have the client evaluate ``cell.seed`` and no other layout of the task.

    RoboDojo takes layouts from the front; its resume manifest can leave layouts out, which
    is how XPolicyLab's ``scripts/run_robodojo_layout_range.sh`` selects them.  This writes
    that manifest: nothing evaluated yet, every other layout abandoned.
    """

    import yaml

    cfg = yaml.safe_load((bench / "env_cfg" / f"{cell.suite}.yml").read_text(encoding="utf-8"))
    config_name = str(cfg.get("config_name") or cell.suite)
    layouts = bench / "Assets" / "Eval_Layout" / "RoboDojo" / config_name / str(eval_seed)
    pattern = re.compile(rf"{re.escape(cell.task)}_\d+\.json")
    count = (
        sum(1 for p in layouts.iterdir() if pattern.fullmatch(p.name)) if layouts.is_dir() else 0
    )
    if not 0 <= int(cell.seed) < count:
        raise ValueError(
            f"layout {cell.seed} of {cell.task}: {layouts} holds {count} (eval seed {eval_seed})"
        )
    result_dir = (
        bench / "eval_result" / "RoboDojo" / cell.task / POLICY_NAME / config_name
        / f"{eval_seed}_{additional_info}"
    )  # fmt: skip
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = result_dir / f"_resume_{run_id}.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "save_dir": str(result_dir / run_id),
                "task_name": cell.task,
                "policy_name": POLICY_NAME,
                "config_name": config_name,
                "eval_seed": int(eval_seed),
                "additional_info": additional_info,
                "success_nums": 0,
                "fail_nums": 0,
                "unstable_nums": 0,
                "total_score": 0.0,
                "completed_layout_ids": [],
                "abandoned_layout_ids": [i for i in range(count) if i != int(cell.seed)],
                "details": {},
                "restart_count": 0,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _site() -> tuple[str | None, str | None]:
    root = os.environ.get("ROBODOJO_XPL_SITE")
    if not root:
        return None, None
    common, server = Path(root) / "common", Path(root) / "server"
    return (str(common) if common.is_dir() else None), (str(server) if server.is_dir() else None)


def _import_server(bench: Path) -> None:
    """Make XPolicyLab's policy server importable here, with a websockets it can run on."""

    common, server = _site()
    for entry in (str(bench / "XPolicyLab"), str(bench), common, server):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)
    loaded = sys.modules.get("websockets")
    if loaded is not None and int(str(getattr(loaded, "__version__", "0")).split(".")[0]) < 13:
        # the simulator's environment keeps an old websockets; nothing here has used it yet
        for name in [n for n in sys.modules if n == "websockets" or n.startswith("websockets.")]:
            del sys.modules[name]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Client:
    """The evaluation client's process."""

    def __init__(self, proc: subprocess.Popen, bridge: Bridge, log: Any) -> None:
        self.proc, self.bridge, self.log = proc, bridge, log

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        self.bridge.finish()
        try:
            self.proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(self.proc.pid, sig)
                try:
                    self.proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        with contextlib.suppress(Exception):
            self.log.close()


def lacking(obs: dict[str, Any]) -> list[str]:
    """What robot ``robodojo_xpl`` reads and this observation does not hold."""

    head = (obs.get("vision") or {}).get("cam_head") or {}
    return [
        name
        for name, got in (
            ("a depth image", tools.depth_image(head) is not None),
            ("intrinsic_matrix", head.get("intrinsic_matrix") is not None),
            ("extrinsic_matrix", tools.extrinsic_of(head) is not None),
        )
        if not got
    ]


def _toolkit(
    bridge: Bridge, cell: Cell, out_dir: Path, env_only: bool, alive: Any = None, depth: bool = True
) -> NativeToolkit:
    """The robot's tools over ``bridge``; the verdict is the one the client reports."""

    if depth:
        lacks = lacking(bridge.get_obs())
        if lacks:
            raise RuntimeError(
                f"the evaluation client sends no {' / '.join(lacks)}: robot {ROBOT} reads them. "
                "Switch them on in the client's env_cfg (observation.vision depth or "
                "approximate_depth, intrinsic_matrix, extrinsic_matrix, and the "
                "distance_to_image_plane capture of its camera file), or run robot "
                "robodojo_xpl_rgb, which needs none of it."
            )
    sam3 = None
    if not env_only:
        from . import sam3_local

        sam3 = sam3_local.shared(cuda_device=cell.cuda_device)
    return NativeToolkit(
        BridgeEnv(bridge, alive),
        sam3,
        out_dir,
        verdict=lambda: bool((bridge.over or {}).get("success")),
        colour_only=not depth,
    )


def boot(
    cell: Cell,
    out_dir: Path,
    *,
    rpent_root: str | None = None,  # noqa: ARG001 - the host interface; nothing of RPent's runs
    env_only: bool = False,
    source_root: str | None = None,
    xpolicylab_root: str | None = None,
    eval_seed: int = 0,
    boot_timeout_s: float = BOOT_TIMEOUT_S,
    bridge: Bridge | None = None,
    depth: bool = True,
    **_: Any,
) -> tuple[NativeToolkit, list[Any]]:
    """Start the policy server and RoboDojo's evaluation client; return ``(toolkit, daemons)``.

    ``env_only`` leaves SAM3 out (enough to dump a task card).  With ``bridge`` nothing is
    started: the evaluation client is already talking to a policy server that serves that
    bridge (XPolicyLab started both, and the adapter's ``model.py`` runs the episode).
    ``depth`` is what the robot reads: with it the client this host starts is configured to
    send the depth image and the camera matrices, and a client that does not send them is
    refused; without it nothing but the client's default observation is read.
    """

    if eval_seed not in EVAL_SEEDS:
        raise ValueError(f"eval_seed must be one of {EVAL_SEEDS}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if bridge is not None:
        bridge.first_obs(boot_timeout_s)
        return _toolkit(bridge, cell, out_dir, env_only, depth=depth), []
    source = _source_root(source_root)
    xpolicylab = _xpolicylab_root(xpolicylab_root, source)
    bench = lay_out(out_dir / "bench", source, xpolicylab, depth=depth, suite=cell.suite)
    _import_server(bench)
    bridge = Bridge()
    port = _free_port()
    server = serve(bridge, "127.0.0.1", port)
    daemons: list[Any] = [server]
    try:
        common, site_server = _site()
        env = dict(os.environ)
        inherited = [
            p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != site_server
        ]
        env["PYTHONPATH"] = os.pathsep.join([p for p in (common, *inherited) if p])
        additional_info = "ckpt_name=pyrualean,action_type=joint"
        run_id = f"pyrualean-{cell.tag}-s{eval_seed}-{os.getpid()}-{int(time.time())}"
        only_layout(
            bench, cell, eval_seed=eval_seed, additional_info=additional_info, run_id=run_id
        )
        # ROBODOJO_ROOT: where a packaged runtime (rlinf-robodojo-runtime) looks for the code;
        # a checkout finds itself
        env.update(EVAL_NUM="1", ROBODOJO_ROOT=str(bench), ROBODOJO_RUN_ID=run_id)
        gpu = 0 if cell.cuda_device is None else int(cell.cuda_device)
        cmd = [
            "bash",
            "scripts/eval_policy.sh",
            "--root_dir", str(bench),
            "--task_name", cell.task,
            "--env_cfg_type", cell.suite,
            "--device_id", str(gpu),
            "--policy_name", POLICY_NAME,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--protocol", "ws",
            "--eval_batch", "false",
            "--additional_info", additional_info,
            "--seed", str(eval_seed),
        ]  # fmt: skip
        log = open(out_dir / "robodojo_eval_client.log", "ab")  # noqa: SIM115 - lives with the client
        proc = subprocess.Popen(  # noqa: S603 - RoboDojo's own script, fixed arguments
            cmd, cwd=bench, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        client = _Client(proc, bridge, log)
        daemons.append(client)
        (out_dir / "host.json").write_text(
            json.dumps(
                {
                    "bench": str(bench),
                    "source": str(source),
                    "xpolicylab": str(xpolicylab),
                    "run_id": run_id,
                    "port": port,
                    "eval_seed": eval_seed,
                    "layout": cell.seed,
                    "depth": depth,
                    "cmd": cmd,
                },
                indent=1,
            )
            + "\n"
        )
        bridge.first_obs(boot_timeout_s, alive=lambda: client.alive() and server.error is None)
        toolkit = _toolkit(bridge, cell, out_dir, env_only, alive=client.alive, depth=depth)
    except BaseException:
        for daemon in reversed(daemons):
            with contextlib.suppress(Exception):
                daemon.stop()
        raise
    return toolkit, daemons


def make_backend(toolkit: NativeToolkit) -> NativeBackend:
    return NativeBackend(toolkit)


def dump_card(toolkit: NativeToolkit, cell: Cell, out_dir: Path, robot: str = ROBOT) -> TaskCard:
    """Write ``card.json`` plus the step-0 images and return the card.

    The same card as RPent's host writes: both arms' initial grasp points, gripper angles and
    openings, and the task's own native-action budget.  Nothing privileged.
    """

    record = toolkit.state.records[0]
    raw = record["state"]
    card_dir = out_dir / "card"
    card_dir.mkdir(exist_ok=True)
    images: dict[str, str] = {}
    for camera in CAMERAS:
        source = toolkit.state.artifact_path(f"{camera}.png", step=0)
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
    try:
        budget = int(toolkit._status().get("step_limit"))
    except (TypeError, ValueError):
        budget = cell.max_episode_steps
    facts["native action budget"] = f"{budget} (a motion call spends one or more)"
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


__all__ = [
    "Cell",
    "POLICY_NAME",
    "adapter_dir",
    "add_boot_args",
    "add_cell_args",
    "boot",
    "boot_options",
    "cell_from_args",
    "client_root",
    "dump_card",
    "lacking",
    "lay_out",
    "make_backend",
    "only_layout",
    "with_depth",
]
