#!/usr/bin/env python3
"""Run solar_chain.py for several episodes on the local GPUs of a node the run owns.

    python solar_batch_local.py EP_DIR [EP_DIR ...] [--win 447] [--max-gpus 6] [--gpus 0,1,2,3] [--tries 3]

Each EP_DIR holds inputs.json:
    {"carrier": "carrier.mp4", "first": "first-canvas.png", "real": "real-f0-t3.png", "prompt": "prompt.txt",
     "fps": 30, "chain_extra": ["--wins", "141,447", "--handover", "124:h3-f124.png"]}
(relative paths are resolved against EP_DIR) and receives EP_DIR/render/ (solar_chain.py output). A GPU is free when it
uses < 2 GB and none of this batch's chains runs on it; after each start the driver waits for the model load to claim
memory before probing again. A failed chain is retried up to --tries times (finished windows are reused). State is
written to the --state file every minute. Without a node to itself, submit solar_chain.py through the deployment's GPU
scheduler instead (one GPU, at least 172 GB free for --win 532).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_canvas  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def free_gpus(allowed: list[str], busy: set) -> list[str]:
    o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    out = []
    for line in o.strip().splitlines():
        i, used = [x.strip() for x in line.split(",")]
        if i in allowed and int(used) < 2000 and i not in busy:
            out.append(i)
    return out


def chain_cmd(ep: Path, win: int, overlap: int) -> list[str]:
    meta = json.loads((ep / "inputs.json").read_text())

    def p(key):
        v = Path(meta[key])
        return str(v if v.is_absolute() else ep / v)

    return [sys.executable, str(HERE / "solar_chain.py"), "--carrier", p("carrier"), "--first", p("first"),
            "--real", p("real"), "--prompt", p("prompt"), "--root", str(ep / "render"), "--fps", f"{float(meta['fps']):g}",
            "--win", str(win), "--overlap", str(overlap), *[str(x) for x in meta.get("chain_extra", [])]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("episodes", nargs="+", type=Path)
    ap.add_argument("--win", type=int, default=447)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--max-gpus", type=int, default=8)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--load-wait", type=int, default=90, help="seconds to wait after a start before the next probe")
    ap.add_argument("--state", type=Path, default=Path("solar-batch-state.json"))
    a = ap.parse_args(argv)
    if not solar_canvas.legal_length(a.win):
        ap.error(f"--win {a.win} is not 17k+5")
    allowed = a.gpus.split(",")
    eps = {str(e): e.resolve() for e in a.episodes}
    procs: dict[str, tuple[subprocess.Popen, str]] = {}
    tries = {n: 0 for n in eps}
    state = {n: "waiting" for n in eps}
    while True:
        for n, (p, g) in list(procs.items()):
            if p.poll() is not None:
                del procs[n]
                ok = p.returncode == 0 and (eps[n] / "render" / "stitched-t3.mp4").exists()
                state[n] = "done" if ok else ("waiting" if tries[n] < a.tries else "failed")
                print(now(), n, "exit", p.returncode, "->", state[n], flush=True)
        busy = {g for _, g in procs.values()}
        for n, ep in eps.items():
            if state[n] != "waiting" or len(procs) >= a.max_gpus:
                continue
            fg = free_gpus(allowed, busy)
            if not fg:
                break
            g = fg[0]
            root = ep / "render"
            root.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            with open(root / "chain.log", "a") as log:
                procs[n] = (subprocess.Popen(chain_cmd(ep, a.win, a.overlap), env=env, stdout=log,
                                             stderr=subprocess.STDOUT), g)
            tries[n] += 1
            state[n] = "running"
            busy.add(g)
            print(now(), n, "started on GPU", g, "try", tries[n], flush=True)
            time.sleep(a.load_wait)
        a.state.write_text(json.dumps({"at": now(), "state": state, "tries": tries,
                                       "gpus": {n: g for n, (_, g) in procs.items()}}, indent=1))
        if all(s in ("done", "failed") for s in state.values()):
            print(now(), "ALL", json.dumps(state), flush=True)
            return 0 if all(s == "done" for s in state.values()) else 1
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
