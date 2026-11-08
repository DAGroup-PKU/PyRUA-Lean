# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboDojo's Pi_05 as XPolicyLab runs it, for the hosts without RPent.

RPent serves RoboDojo's Pi_05 through RLinf, whose channel layout is not the one the published
checkpoints were trained on (``docs/setup.md``).  XPolicyLab's own ``policy/Pi_05`` is the
policy RoboDojo's leaderboard evaluates: openpi's JAX model, the state packed and the actions
unpacked arm by arm by XPolicyLab's ``pack_robot_state`` / ``unpack_robot_state``.  It needs
the uv environment ``policy/Pi_05/install.sh`` builds (JAX, numpy 2), which this library's
environment cannot share a process with, so :func:`start` runs it as a server of its own
(:mod:`.pi05_server`, openpi's websocket protocol) and :class:`Pi05Client` is what the tools
call: ``predict(obs)`` sends one RoboDojo observation and returns the action chunk, a list of
the evaluation client's action dicts (RPent's ``vla_client`` interface).

Settings, each also an environment variable:

``--pi05-checkpoint`` / ``ROBODOJO_PI05_CHECKPOINT``
    the step directory of a published checkpoint (``.../RoboDojo-sim-arx_x5-joint-0/59999``,
    with ``params/`` and ``assets/arx_x5_sim``);
``--pi05-python`` / ``ROBODOJO_PI05_PYTHON``
    the interpreter of that uv environment (default:
    ``<XPolicyLab>/policy/Pi_05/openpi/.venv/bin/python``);
``--pi05-endpoint`` / ``ROBODOJO_PI05_ENDPOINT``
    ``host:port`` of a server already running; nothing is started.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

#: The server is a file run by path in another interpreter.
SERVER = Path(__file__).with_name("pi05_server.py")
#: Loading the checkpoint and compiling the first inference take about a minute.
BOOT_TIMEOUT_S = 900.0
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ARMS = ("left", "right")
#: ``pi0_pick``'s grasp heuristic, RPent's (``robots/robodojo/tools.py``): an arm went down at
#: least DESCEND_M, then rose at least LIFT_M with its gripper below CLOSED_BELOW.
DESCEND_M = 0.06
LIFT_M = 0.04
CLOSED_BELOW = 0.55


def default_python(xpolicylab: Path) -> Path:
    return Path(xpolicylab) / "policy" / "Pi_05" / "openpi" / ".venv" / "bin" / "python"


# openpi's wire format (``openpi_client.msgpack_numpy``), which that package does not need
# to be installed here for: an array is a map of its bytes, dtype and shape.
def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str,
                b"shape": obj.shape}  # fmt: skip
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj: Any) -> bytes:
    import msgpack

    return msgpack.packb(obj, default=_pack_array)


def unpackb(data: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(data, object_hook=_unpack_array)


def wire_obs(obs: dict[str, Any], instruction: str) -> dict[str, Any]:
    """What the policy reads of a RoboDojo observation: the colour images, the state, the text.

    Depth images and camera matrices stay behind; ``Model.update_obs`` reads neither.
    """

    vision = obs.get("vision") or {}
    images: dict[str, Any] = {}
    for camera in CAMERAS:
        color = (vision.get(camera) or {}).get("color")
        if color is None:
            raise ValueError(f"the observation has no vision/{camera}/color")
        images[camera] = {"color": np.ascontiguousarray(np.asarray(color)[..., :3], np.uint8)}
    state = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in (obs.get("state") or {}).items()
        if key.endswith("_joint_state")
    }
    return {"vision": images, "state": state, "instruction": str(instruction)}


class Pi05Client:
    """The policy server's client: ``predict(obs)`` -> one action chunk."""

    def __init__(self, endpoint: str, timeout_s: float = BOOT_TIMEOUT_S) -> None:
        from websockets.sync.client import connect

        uri = endpoint if endpoint.startswith("ws") else f"ws://{endpoint}"
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self._ws = connect(uri, compression=None, max_size=None, open_timeout=30)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(2.0)
        self.metadata = unpackb(self._ws.recv())

    def predict(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        """One chunk for ``obs`` (whose ``instruction`` is the policy's prompt)."""

        self._ws.send(packb(wire_obs(obs, obs.get("instruction") or "")))
        reply = self._ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"Pi_05 server failed:\n{reply}")
        actions = unpackb(reply)["actions"]
        return [
            {key: [float(v) for v in np.asarray(value).reshape(-1)] for key, value in a.items()}
            for a in actions
        ]

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._ws.close()


class Pi05Server:
    """The server's process (a daemon of the host)."""

    def __init__(self, proc: subprocess.Popen, endpoint: str, log: Any) -> None:
        self.proc, self.endpoint, self.log = proc, endpoint, log

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            if self.proc.poll() is not None:
                break
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.proc.pid, sig)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=wait)
        with contextlib.suppress(Exception):
            self.log.close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _listening(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start(
    xpolicylab: Path,
    checkpoint: str,
    out_dir: Path,
    *,
    python: str | None = None,
    cuda_device: int | None = None,
    timeout_s: float = BOOT_TIMEOUT_S,
) -> Pi05Server:
    """Start :mod:`.pi05_server` and return once it accepts connections."""

    xpolicylab = Path(xpolicylab).resolve()
    if not (xpolicylab.parent / "env_cfg").is_dir():
        raise FileNotFoundError(f"XPolicyLab reads env_cfg/ beside it: none at {xpolicylab.parent}")
    ckpt = Path(checkpoint).expanduser().resolve()
    if not (ckpt / "params").is_dir() or not (ckpt / "assets" / "arx_x5_sim").is_dir():
        raise FileNotFoundError(f"not a Pi_05 checkpoint step (params/, assets/arx_x5_sim/): "
                                f"{ckpt}")  # fmt: skip
    interpreter = Path(python or default_python(xpolicylab))
    if not interpreter.is_file():
        raise FileNotFoundError(
            f"Pi_05's uv environment: {interpreter} (bash {xpolicylab}/policy/Pi_05/install.sh)"
        )
    port = _free_port()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # the uv environment is whole; this library's paths break it
    # JAX loads the CUDA 13 libraries the uv environment ships; a system CUDA ahead of them on
    # the library path fails the first inference (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED)
    for name in ("LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH"):
        env.pop(name, None)
    # what the policy needs (~7 GB) and no more: JAX otherwise takes 0.3 of the card up front
    # (policy/Pi_05's setting, 55 GB of a B200), and the simulator and SAM3 beside it run out
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    cmd = [
        str(interpreter), "-P", str(SERVER),
        "--xpolicylab-root", str(xpolicylab),
        "--checkpoint", str(ckpt),
        "--port", str(port),
        "--parent-pid", str(os.getpid()),
    ]  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "pi05_server.log", "ab")  # noqa: SIM115 - lives with the server
    proc = subprocess.Popen(  # noqa: S603 - fixed arguments
        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    server = Pi05Server(proc, f"127.0.0.1:{port}", log)
    deadline = time.monotonic() + timeout_s
    while not _listening(port):
        if not server.alive():
            server.stop()
            raise RuntimeError(f"the Pi_05 server exited: see {out_dir / 'pi05_server.log'}")
        if time.monotonic() > deadline:
            server.stop()
            raise TimeoutError(f"the Pi_05 server did not listen within {timeout_s:.0f} s")
        time.sleep(2.0)
    return server


_SHARED: dict[tuple[str, str], Pi05Server] = {}
_SHARED_LOCK = threading.Lock()


def shared(
    xpolicylab: Path,
    checkpoint: str,
    out_dir: Path,
    *,
    python: str | None = None,
    cuda_device: int | None = None,
) -> Pi05Server:
    """The process's one Pi_05 server for a checkpoint and a card, started on first use.

    An XPolicyLab policy server plays one episode after another: the policy is loaded for
    the first and serves the rest (its log is that first episode's ``pi05_server.log``).  It
    is stopped when this process exits.
    """

    key = (str(Path(checkpoint).expanduser().resolve()), str(cuda_device))
    with _SHARED_LOCK:
        server = _SHARED.get(key)
        if server is None or not server.alive():
            if not _SHARED:
                atexit.register(_stop_shared)
            server = start(xpolicylab, checkpoint, out_dir, python=python, cuda_device=cuda_device)
            _SHARED[key] = server
        return server


def _stop_shared() -> None:
    with _SHARED_LOCK:
        for server in _SHARED.values():
            server.stop()
        _SHARED.clear()


def _z(state: dict[str, Any], arm: str) -> float:
    return float(np.asarray(state[f"{arm}_ee_pose"], dtype=np.float64)[2])


def _grip(state: dict[str, Any], arm: str) -> float:
    return float(np.asarray(state[f"{arm}_ee_joint_state"]).reshape(-1)[0])


def pi0_pick(
    primitives: Any,
    prompt: str | None = None,
    arm: str = "right",
    max_chunks: int = 8,
) -> dict[str, Any]:
    """Closed-loop Pi_05 driven by its own action chunks: RPent's ``pi0_pick``, as is.

    Each chunk is predicted from the current observation and executed action by action; the
    call returns once an arm went down, came back up and has its gripper closed (the chunk is
    cut there: its remaining actions were predicted for an ungrasped state and would reopen
    the gripper), after ``max_chunks`` chunks, or when the budget is spent.  The payload is
    RPent's, which :meth:`pyrualean.robodojo.RobodojoRobot.pi0_pick` reads.
    """

    vla = getattr(primitives, "vla_client", None)
    if vla is None:
        return {"error": "vla_client not configured"}
    obs = primitives.env.get_obs()
    instruction = str(obs.get("instruction") or "") if prompt is None else str(prompt)
    if not instruction.strip():
        return {"error": "RoboDojo instruction must be a nonempty string"}
    if "<" in instruction or ">" in instruction:
        return {"error": "RoboDojo instruction contains unresolved template markers"}
    state = obs["state"]
    track = {a: {"start_z": _z(state, a), "start_grip": _grip(state, a)} for a in ARMS}
    for t in track.values():
        t["min_z"] = t["peak"] = t["start_z"]
        t["last_grip"] = t["start_grip"]
    chunks_used = 0
    success = terminated = False
    for c in range(int(max_chunks)):
        primitives._check_cancelled()
        obs = dict(primitives.env.get_obs())
        obs["instruction"] = instruction
        actions = vla.predict(obs)
        chunks_used = c + 1
        for action in actions:
            primitives._check_cancelled()
            obs_step, _reward, done, info = primitives.env.step(action)
            primitives._last_obs = obs_step
            st = obs_step["state"]
            for a in ARMS:
                t = track[a]
                z, grip = _z(st, a), _grip(st, a)
                t["last_grip"] = grip
                if z < t["min_z"]:
                    t["min_z"] = t["peak"] = z
                else:
                    t["peak"] = max(t["peak"], z)
                if (
                    t["start_z"] - t["min_z"] >= DESCEND_M
                    and t["peak"] - t["min_z"] >= LIFT_M
                    and grip < CLOSED_BELOW
                ):
                    success = True
                    break
            if success:
                break
            status = (info or {}).get("status") or {}
            if done or int(status.get("step", 0)) >= int(status.get("step_limit", 1 << 30)):
                terminated = True
                break
        if success or terminated:
            break
    # the grippers are where the policy left them: a later KEEP holds that, not an older command
    commands = getattr(primitives, "_grip_commands", None)
    if isinstance(commands, dict):
        commands.clear()
    return {
        "monitored_arm": arm,
        "instruction": instruction,
        "success": success,
        "chunks_used": chunks_used,
        "arms": {
            a: {
                "start_eef_z": round(t["start_z"], 4),
                "min_eef_z": round(t["min_z"], 4),
                "peak_eef_z": round(t["peak"], 4),
                "peak_lift_m": round(t["peak"] - t["min_z"], 4),
                "start_gripper": round(t["start_grip"], 3),
                "final_gripper": round(t["last_grip"], 3),
            }
            for a, t in track.items()
        },
        "terminated": terminated,
    }


__all__ = [
    "CAMERAS",
    "Pi05Client",
    "Pi05Server",
    "default_python",
    "pi0_pick",
    "shared",
    "start",
    "wire_obs",
]
