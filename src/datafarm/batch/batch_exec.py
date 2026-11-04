"""Run one batch attempt command and record how it ended.

Usage: python batch_exec.py ATTEMPT_DIR -- CMD...

Writes ``started.json`` (this wrapper's pid) before starting the command and ``exit.json`` (the
return code) atomically once it ends. A supervisor that restarted is no longer the parent of the
attempt, so these two files are how it tells "finished", "crashed" and "still running" apart.
SIGTERM/SIGINT are forwarded to the command; SIGHUP is ignored so a closed terminal does not kill
the attempt. The file is standard-library only and has no package imports, so it can be run by
path from any environment.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _write(path: Path, value: dict) -> None:
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str]) -> int:
    separator = 1
    timeout_s = None
    if len(argv) > 2 and argv[1] == "--timeout-s":
        try:
            timeout_s = float(argv[2])
        except ValueError:
            timeout_s = None
        if timeout_s is None or not math.isfinite(timeout_s) or timeout_s <= 0:
            print("--timeout-s must be a finite positive number", file=sys.stderr)
            return 2
        separator = 3
    if len(argv) <= separator + 1 or argv[separator] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    attempt_dir = Path(argv[0])
    command = argv[separator + 1:]
    attempt_dir.mkdir(parents=True, exist_ok=True)
    _write(attempt_dir / "started.json",
           {"pid": os.getpid(), "started_at": time.time(), "command": command,
            "timeout_s": timeout_s})
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                 start_new_session=timeout_s is not None)
    except OSError as exc:
        _write(attempt_dir / "exit.json",
               {"returncode": 127, "error": f"{type(exc).__name__}: {exc}",
                "finished_at": time.time()})
        return 127

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            if timeout_s is None:
                child.send_signal(signum)
            else:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    timed_out = False
    try:
        code = child.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = child.wait()
    _write(attempt_dir / "exit.json",
           {"returncode": code, "child_pid": child.pid, "finished_at": time.time(),
            "timed_out": timed_out, "timeout_s": timeout_s})
    return 0 if code == 0 and not timed_out else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
