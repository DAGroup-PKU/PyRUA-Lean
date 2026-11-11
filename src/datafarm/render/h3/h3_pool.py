#!/usr/bin/env python3
"""H3 worker pool: resident workers that keep the model loaded, plus elastic workers while the queue has work.

  python h3_pool.py run [--run-dir <run>] [--once]     # single instance per run (flock); loops every `interval_s`
  python h3_pool.py status [--run-dir <run>]

Configuration: <run>/queue/pool.json (h3.worker-pool/1), re-read every cycle, so the operator can change the counts
without a restart:
  {"schema": "h3.worker-pool/1", "profile": "control-turbo",
   "resident": 1,              # workers kept alive at all times (a new episode starts without GPU wait or model load)
   "max_workers": 4,           # resident + elastic ceiling; elastic workers are added while tickets are pending
   "resident_idle_min": 720,   # a resident worker's idle exit (safety net; the pool relaunches it)
   "elastic_idle_min": 20,
   "min_free_mb": 172000,      # a full-residency worker peaks at ~169.5k MiB
   "local_gpus": [0, 1],       # optional: local mode on `local_host`, the scheduler is not used
   "local_host": "<hostname>"}
Scheduler mode submits launch_control_worker.sh (hosts from <run>/queue/ALLOWED_HOSTS, never preempting) and tracks
workers by the scheduler's `leases`; an unknown lease table launches nothing that cycle.  Exit 2 from `submit` (no
GPU free) backs off quietly.  Local mode starts the same worker on a listed GPU of this host that has `min_free_mb`
free (nvidia-smi), tracks it by pid, and refuses to run on any host but `local_host`; it also returns tickets claimed
by dead local workers to the queue (h3_requeue_orphans.py reads scheduler leases and does not see local workers).
State: <run>/queue/pool-state.json.  Stop: touch <run>/queue/STOP (workers and pool exit).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import h3_paths

HERE = Path(__file__).resolve().parent
SCHEMA = "h3.worker-pool/1"
DEFAULTS = {"profile": "control-turbo", "resident": 0, "max_workers": 1, "resident_idle_min": 720,
            "elastic_idle_min": 20, "min_free_mb": 172000, "interval_s": 30, "local_gpus": [], "local_host": None}
HOST = socket.gethostname()
PROCS: dict = {}   # local workers started by this process: id -> Popen
CLAIM_TAG = re.compile(r"(.+)-(\d+)-([A-Za-z][\w.-]*)$")   # running ticket <name>@<host>-<pid>-<profile>.json


def config(run: Path) -> dict:
    path = run / "queue" / "pool.json"
    if not path.is_file():
        raise SystemExit(f"missing {path} (h3.worker-pool/1)")
    cfg = json.loads(path.read_text())
    if cfg.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {cfg.get('schema')!r}, not {SCHEMA}")
    cfg = {**DEFAULTS, **cfg}
    if cfg["local_gpus"] and not cfg["local_host"]:
        raise SystemExit(f"{path}: local_gpus needs local_host (the node those GPUs belong to)")
    if int(cfg["max_workers"]) < int(cfg["resident"]):
        raise SystemExit(f"{path}: max_workers {cfg['max_workers']} < resident {cfg['resident']}")
    return cfg


def live_lease_ids() -> set | None:
    """Lease ids from `$GPU_SCHEDULER leases`, None when the table is unknown."""
    try:
        p = subprocess.run([h3_paths.gpu_scheduler(), "leases"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode:
        return None
    return {line.split()[0] for line in p.stdout.splitlines() if line.strip()}


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    try:   # a zombie is not a worker
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def local_free_mb() -> dict:
    p = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60)
    out = {}
    for line in p.stdout.splitlines():
        try:
            i, free = (int(x) for x in line.split(","))
            out[i] = free
        except ValueError:
            pass
    return out


def worker_env(run: Path, cfg: dict, resident: bool) -> dict:
    idle = int(cfg["resident_idle_min"] if resident else cfg["elastic_idle_min"]) * 60
    return {**os.environ, "H3_RUN": str(run), "H3_PROFILE": cfg["profile"], "H3_IDLE_EXIT": str(idle),
            "H3_MIN_FREE_MB": str(int(cfg["min_free_mb"]))}


def launch_scheduled(run: Path, cfg: dict, resident: bool) -> tuple[dict | None, str, bool]:
    """(worker, message, busy): busy = the scheduler had no matching GPU (exit 2)."""
    p = subprocess.run(["bash", str(HERE / "launch_control_worker.sh"), "full"], env=worker_env(run, cfg, resident),
                       cwd=run, capture_output=True, text=True, timeout=300)
    out = (p.stdout + p.stderr).strip()
    kv = dict(re.findall(r"(\w+)=(\S+)", p.stdout))
    if p.returncode == 0 and kv.get("id"):
        return {"id": kv["id"], "host": kv.get("host"), "log": kv.get("log"), "resident": resident,
                "t": time.time()}, out, False
    return None, out, p.returncode == 2


def launch_local(run: Path, cfg: dict, alive: list, resident: bool) -> tuple[dict | None, str]:
    used = {w.get("gpu") for w in alive}
    free = local_free_mb()
    need = int(cfg["min_free_mb"])
    gpus = [int(g) for g in cfg["local_gpus"] if int(g) not in used and free.get(int(g), 0) >= need]
    if not gpus:
        return None, f"no listed GPU with >= {need} MB free (free {free}, pool uses {sorted(used)})"
    gpu = gpus[0]
    wid = f"local-{HOST}-g{gpu}-{time.strftime('%Y%m%d-%H%M%S')}"
    log = run / "queue/logs" / f"{wid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        proc = subprocess.Popen(["bash", str(HERE / "launch_control_worker.sh"), "full"],
                                env={**worker_env(run, cfg, resident), "H3_LOCAL_GPU": str(gpu)}, cwd=run, stdout=fh,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    PROCS[wid] = proc
    return {"id": wid, "host": HOST, "gpu": gpu, "pid": proc.pid, "log": str(log), "resident": resident,
            "local": True, "t": time.time()}, f"gpu {gpu} pid {proc.pid}"


def local_alive(w: dict) -> bool:
    if w.get("host") != HOST:
        return False
    p = PROCS.get(w["id"])
    return p.poll() is None if p is not None else pid_alive(w.get("pid"))


def requeue_dead_local(run: Path) -> list[str]:
    """Return tickets claimed on this host by a worker pid that is gone (local mode only)."""
    out = []
    for t in sorted((run / "queue/running").glob("*.json")):
        name, _, tag = t.stem.rpartition("@")
        m = CLAIM_TAG.match(tag)
        if not m or m.group(1) != HOST or pid_alive(m.group(2)):
            continue
        try:
            profile = json.loads(t.read_text()).get("profile") or m.group(3)
        except ValueError:
            profile = m.group(3)
        dest = run / "queue/pending" / profile / f"{name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(t.read_text())
        t.unlink()
        out.append(dest.name)
    return out


def cycle(run: Path, st: dict, log) -> float:
    """One pool step; returns seconds to wait before the next one (0: try to add another worker now)."""
    cfg = config(run)
    local = bool(cfg["local_gpus"])
    if local and HOST != cfg["local_host"]:
        raise SystemExit(f"pool.json local_host is {cfg['local_host']}, this host is {HOST}: refusing to start workers")
    pending = sorted((run / "queue/pending" / cfg["profile"]).glob("*.json"))
    running = sorted((run / "queue/running").glob("*.json"))
    if local:
        for w in st["workers"]:
            w["alive"] = bool(w.get("local")) and local_alive(w)
        requeued = requeue_dead_local(run)
        if requeued:
            log(f"requeued from dead local workers: {requeued}")
    else:
        live = live_lease_ids()
        if live is None:
            st["last_error"] = "scheduler leases failed: lease table unknown, launching nothing"
            return float(cfg["interval_s"])
        for w in st["workers"]:
            w["alive"] = w["id"] in live
    st["workers"] = [w for w in st["workers"] if w["alive"] or time.time() - w.get("t", 0) < 3 * 86400][-50:]
    alive = [w for w in st["workers"] if w["alive"]]
    residents = [w for w in alive if w.get("resident")]
    want = max(int(cfg["resident"]), min(int(cfg["max_workers"]), len(pending) + len(running)))
    st.update(pending=len(pending), running=len(running), alive=len(alive), resident_alive=len(residents),
              want=want, checked_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    resident = len(residents) < int(cfg["resident"])
    if not (resident or (pending and len(alive) < want)) or time.time() < st.get("backoff_until", 0):
        return float(cfg["interval_s"])
    kind = "resident" if resident else "elastic"
    if local:
        rec, msg = launch_local(run, cfg, alive, resident)
        busy = rec is None
    else:
        rec, msg, busy = launch_scheduled(run, cfg, resident)
    log(f"launch {kind}: " + (f"{rec['id']} {msg[-200:]}" if rec else ("no GPU free: " if busy else "FAILED: ")
                                                                          + msg[-400:]))
    if rec:
        rec["alive"] = True
        st["workers"].append(rec)
        st.pop("last_error", None)
        return 0.0
    st["last_error"] = msg[-300:]
    st["backoff_until"] = time.time() + (60 if busy else 300)
    return float(cfg["interval_s"])


def run_pool(run: Path, once: bool = False) -> int:
    queue = run / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    lock = open(queue / ".pool.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("h3_pool already running for", run)
        return 0
    state_path = queue / "pool-state.json"
    st = json.loads(state_path.read_text()) if state_path.is_file() else {}
    st.update(pid=os.getpid(), host=HOST, started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    st.setdefault("workers", [])

    def log(msg):
        with open(queue / "pool.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")

    while True:
        if (queue / "STOP").exists():
            log("STOP file seen; pool exits")
            st["stopped_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            state_path.write_text(json.dumps(st, indent=1))
            return 0
        wait = cycle(run, st, log)
        state_path.write_text(json.dumps(st, indent=1))
        if once:
            return 0
        time.sleep(wait)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--once", action="store_true", help="one cycle (tests, cron)")
    s = sub.add_parser("status")
    for p in (r, s):
        h3_paths.add_run_dir_arg(p)
    args = ap.parse_args()
    run = h3_paths.run_dir(args.run_dir)
    if args.cmd == "status":
        path = run / "queue/pool-state.json"
        print(path.read_text() if path.is_file() else "{}")
        return 0
    return run_pool(run, args.once)


if __name__ == "__main__":
    sys.exit(main())
