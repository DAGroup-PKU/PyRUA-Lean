# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Load a generated policy file and execute it against a robot.

``load_policy`` imports a file under a unique module name every time, so two
generated programs that both call themselves ``policy.py`` never collide.
``run_policy`` calls ``run(robo)`` with a wall-clock watchdog that stops the
episode through the library (every further motion call raises
``EpisodeFinished``) and interrupts the primitive in flight through the
backend.  Neither function sandboxes the code: evaluations run one episode
per subprocess (see :mod:`pyrualean.hosts.rpent_libero`) and the parent
kills the whole process group when the child ignores the watchdog.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ._backend import EpisodeFinished
from .libero import LiberoRobot


class PolicyLoadError(ValueError):
    """The policy file could not be imported or has no ``run`` callable."""


def policy_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(
    path: str | Path,
    *,
    entrypoint: str = "run",
    builtins: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Import ``path`` as a fresh module and return its ``entrypoint`` callable.

    ``builtins`` (see :func:`pyrualean.sandbox.make_builtins`) becomes the
    module's ``__builtins__``, so the program's imports and ``open`` calls
    go through the sandbox guards.
    """

    location = Path(path).expanduser().resolve()
    if not location.is_file() or location.suffix != ".py":
        raise PolicyLoadError(f"policy must be an existing .py file: {location}")
    module_name = f"pyrualean_policy_{location.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, location)
    if spec is None or spec.loader is None:
        raise PolicyLoadError(f"cannot create an import spec for {location}")
    module = importlib.util.module_from_spec(spec)
    if builtins is not None:
        module.__dict__["__builtins__"] = builtins
    sys.modules[module_name] = module
    parent = str(location.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise PolicyLoadError(f"importing {location} failed: {exc}") from exc
    policy = getattr(module, entrypoint, None)
    if not callable(policy):
        raise PolicyLoadError(f"{location} does not define a callable {entrypoint!r}")
    return policy


@dataclass
class PolicyOutcome:
    """How one ``run(robo)`` call ended (independent of task success)."""

    status: str  # completed | finished | error | timeout
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_policy(
    policy: Callable[..., Any],
    robo: LiberoRobot,
    *,
    timeout_s: float | None = None,
) -> PolicyOutcome:
    """Call ``policy(robo)`` once under a wall-clock watchdog.

    ``timeout_s`` stops the episode via the library and the backend; the
    call then ends with status ``"timeout"``.  Exceptions raised by the
    policy are captured (status ``"error"``) with their traceback; an
    ``EpisodeFinished`` unwinding is a normal end (status ``"finished"``).
    """

    timer: threading.Timer | None = None
    timed_out = threading.Event()

    def on_timeout() -> None:
        timed_out.set()
        robo._stop_episode("timeout")

    if timeout_s is not None and timeout_s > 0:
        timer = threading.Timer(timeout_s, on_timeout)
        timer.daemon = True
        timer.start()

    started = time.perf_counter()
    try:
        result = policy(robo)
        if inspect.isawaitable(result):
            import asyncio

            asyncio.run(result)
        outcome = PolicyOutcome(status="completed")
    except EpisodeFinished as exc:
        outcome = PolicyOutcome(status="finished", reason=exc.reason)
    except KeyboardInterrupt:
        outcome = PolicyOutcome(status="error", reason="interrupted")
    except Exception as exc:  # noqa: BLE001 - the outcome records the failure
        outcome = PolicyOutcome(
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        if timer is not None:
            timer.cancel()
    outcome.elapsed_s = time.perf_counter() - started
    if timed_out.is_set():
        outcome.status = "timeout"
        outcome.reason = "timeout"
    return outcome


def run_host(
    argv: Sequence[str],
    *,
    timeout_s: float,
    log_path: str | Path | None = None,
    env: dict[str, str] | None = None,
    grace_s: float = 30.0,
) -> int:
    """Run a host command in its own process group and enforce a hard timeout.

    On timeout the group receives SIGTERM, then SIGKILL after ``grace_s``.
    SIGTERM/SIGHUP received by the caller while the host runs are forwarded
    to the host's group, so killing the CLI never leaves an orphaned
    simulator behind.  Returns the exit code (``-signal`` when killed).
    """

    log_handle = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "ab")  # noqa: SIM115 - closed below
    previous: dict[int, Any] = {}
    try:
        proc = subprocess.Popen(
            list(argv),
            stdout=log_handle,
            stderr=subprocess.STDOUT if log_handle is not None else None,
            env=env,
            start_new_session=True,
        )

        def forward(signum: int, frame: Any) -> None:  # noqa: ARG001
            _kill_group(proc, signal.SIGTERM)

        for signum in (signal.SIGTERM, signal.SIGHUP):
            try:
                previous[signum] = signal.signal(signum, forward)
            except (ValueError, OSError):  # not the main thread
                pass
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc, signal.SIGTERM)
            try:
                return proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                _kill_group(proc, signal.SIGKILL)
                return proc.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if log_handle is not None:
            log_handle.close()


def _kill_group(proc: subprocess.Popen[Any], sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


__all__ = [
    "PolicyLoadError",
    "PolicyOutcome",
    "load_policy",
    "policy_sha256",
    "run_host",
    "run_policy",
]
