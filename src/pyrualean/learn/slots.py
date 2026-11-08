"""GPU slots shared by every process of a campaign (episodes and the workers' replays).

A slot is a lock file ``slots/gpu<G>-<i>.lock`` held with ``flock`` for as long as the episode
runs; a crashed process releases its slot with its file descriptors.  Starts are spaced by
``stagger_seconds`` across processes (Isaac Sim boots are the expensive moment).
"""

from __future__ import annotations

import contextlib
import fcntl
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def free_gib() -> dict[int, float]:
    """Free memory per GPU (all users), from nvidia-smi; empty when it cannot be read."""

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return {}
    free = {}
    for line in out.splitlines():
        index, mib = (part.strip() for part in line.split(","))
        free[int(index)] = float(mib) / 1024
    return free


@contextlib.contextmanager
def slot(root: Path, settings: dict[str, Any], *, poll_s: float = 5.0) -> Iterator[int]:
    """Hold one free slot; yields its GPU index.

    ``settings["root"]``, when given, is a slot directory shared by several campaigns (they then
    share the GPUs' slot count).  ``settings["min_free_gib"]``: a slot on a GPU with less free
    memory than this (other users included) is not taken; the job waits.
    """

    directory = Path(settings.get("root") or Path(root) / "slots")
    directory.mkdir(parents=True, exist_ok=True)
    names = [
        (int(gpu), directory / f"gpu{gpu}-{index}.lock")
        for index in range(int(settings["per_gpu"]))
        for gpu in settings["gpus"]
    ]
    need = float(settings.get("min_free_gib") or 0)
    while True:
        free = free_gib() if need else {}
        for gpu, path in names:
            if need and free and free.get(gpu, 0.0) < need:
                continue
            handle = path.open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                _stagger(directory, float(settings.get("stagger_seconds", 0)))
                yield gpu
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            return
        time.sleep(poll_s)


def _stagger(directory: Path, gap: float) -> None:
    if gap <= 0:
        return
    with (directory / "stagger.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        text = handle.read().strip()
        last = float(text) if text else 0.0
        wait = last + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        handle.seek(0)
        handle.truncate()
        handle.write(str(time.time()))
