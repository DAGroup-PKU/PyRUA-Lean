#!/usr/bin/env python3
"""Requeue running tickets whose worker lease is gone (crashed / timed out workers).

A running ticket is named <ticket>@<host>-<pid>-<profile>.json.  If no live lease on <host> has that pid, the ticket goes
back to pending/<ticket profile>/ (rendered chunks are kept; the worker resumes from the first missing chunk).  Leases
come from `${GPU_SCHEDULER} leases`; the queue is <run>/queue (run directory = --run-dir or env H3_RUN).  One pass per
call: run it when workers may have died, by hand or from whatever already supervises the deployment.

  python3 h3_requeue_orphans.py [--run-dir <run>]
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import h3_paths

# <host>-<pid>-<profile>: the pid is the last all-digit field before a profile that starts with a letter
CLAIM_TAG = re.compile(r"(.+)-(\d+)-([A-Za-z][\w.-]*)$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    h3_paths.add_run_dir_arg(ap)
    args = ap.parse_args()
    run = h3_paths.run_dir(args.run_dir)
    result = subprocess.run([h3_paths.gpu_scheduler(), "leases"], capture_output=True, text=True)
    if result.returncode:  # an unknown lease table says nothing about which workers are alive
        raise SystemExit(f"scheduler leases failed: {result.stderr.strip()[-300:]}")
    live_pids = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].startswith("h3-"):
            live_pids.add((parts[2], parts[4]))  # (host, worker pid)
    for t in sorted((run / "queue/running").glob("*.json")):
        name, _, tag = t.stem.rpartition("@")
        m = CLAIM_TAG.match(tag)
        if not m:
            continue
        host, pid, profile = m.groups()
        if (host, pid) in live_pids:
            continue
        try:  # back to the ticket's own profile queue
            profile = json.loads(t.read_text()).get("profile") or profile
        except ValueError:
            pass
        dest = run / "queue/pending" / profile / f"{name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(t.read_text())
        t.unlink()
        print("requeued", dest.name, "from dead", host, profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
