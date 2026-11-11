#!/usr/bin/env python3
"""Build packages for a recipe over tasks x episodes x modes and enqueue tickets round-robin.

  python h3_enqueue.py [--run-dir <run>] --recipe <run>/recipes/<name>.json --tasks <task-a> <task-b> \
      --modes sim2real [--episodes <id> ...] [--seed-offset 0 1 ... 7] [--priority 100] [--per-chunk] [--build-slots 2]
Bundles are read from <run>/tasks/<task>/, packages go to <run>/pkgs/<recipe>/<task>/<tag>, tickets to <run>/queue/.
--episodes selects bundle episode ids (default: all).  OOD episodes cycle through the bundle's ood_variants (episode k ->
variant k mod n).  --per-chunk writes one ticket per unrendered chunk, so several workers can share one package (chunks
render independently).  Existing packages are reused (not rebuilt); tickets already done/running are not re-queued.
Builds: the packages of the first --seed-offset are built before the others, so they fill the run's HED cache (keyed by
the white chunk) and the other seeds reuse it instead of recomputing identical HED videos.  Every build runs at nice 10
and holds one of --build-slots run-wide slots (<run>/cache/build-slots, flock, shared by concurrent enqueue calls): one
build runs 4 HED processes x 16 threads, and unthrottled builds of many episodes x seeds once drove a 180-core login node
to load ~2500.
--gate <name> (production) refuses unless the h3_gate.py gate of this recipe and task has released every selected
episode: the 2 prefixes first, the canary after both prefixes pass blind review, the rest after the canary passes.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h3_paths

HERE = Path(__file__).resolve().parent


def bundle_file(run: Path, task: str, recipe: dict) -> Path:
    name = f"bundle-{recipe['bundle_variant']}.json" if recipe.get("bundle_variant") else "bundle.json"
    return run / "tasks" / task / name


@contextlib.contextmanager
def build_slot(run: Path, slots: int, poll: float = 5.0):
    """Hold one of `slots` run-wide build slots (flock on <run>/cache/build-slots/slot-<i>) while building."""
    d = run / "cache" / "build-slots"
    d.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(max(1, slots)):
            f = open(d / f"slot-{i}", "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            try:
                yield i
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        time.sleep(poll)


def chunk_tickets(ticket: dict) -> list[tuple[str, dict]]:
    """(name suffix, ticket) per chunk without output; each ticket names a one-job copy of the manifest."""
    package = Path(ticket["package"])
    manifest = json.loads((package / "job-manifest.json").read_text())
    out = []
    for job in manifest["jobs"]:
        if (Path(ticket["output_root"]) / "chunks" / job["chunk_id"] / "segment-manifest.json").exists():
            continue
        path = package / f"job-manifest.{job['chunk_id']}.json"
        path.write_text(json.dumps({**manifest, "jobs": [job]}, indent=2))
        out.append((f"-c{int(job['chunk_index']):02d}", {**ticket, "manifest": str(path)}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--episodes", nargs="*", default=[], help="bundle episode ids (default: all)")
    ap.add_argument("--modes", nargs="+", required=True, choices=["sim2real", "ood"])
    ap.add_argument("--seed-offset", type=int, nargs="+", default=[0],
                    help="one or more seed offsets; the first is built first and fills the HED cache")
    ap.add_argument("--build-slots", type=int, default=int(os.environ.get("H3_BUILD_SLOTS", "2")),
                    help="run-wide concurrent package builds (default env H3_BUILD_SLOTS or 2)")
    ap.add_argument("--priority", type=int, default=100, help="lower runs first (ticket filename prefix)")
    ap.add_argument("--per-chunk", action="store_true", help="one ticket per chunk instead of one per package")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gate", help="h3_gate.py gate name: enqueue only episodes it has released")
    h3_paths.add_run_dir_arg(ap)
    args = ap.parse_args()
    run = h3_paths.run_dir(args.run_dir)
    python = h3_paths.python_exe()
    recipe = json.loads(Path(args.recipe).read_text())
    if recipe.get("backend") != "control-hybrid":
        raise SystemExit("recipe backend must be 'control-hybrid'")
    recipe_name = recipe["name"]
    items = []
    for task in args.tasks:
        bundle = json.loads(bundle_file(run, task, recipe).read_text())
        episodes = [e for e in bundle["episodes"] if not args.episodes or e["id"] in args.episodes]
        variants = [v["name"] for v in bundle["ood_variants"]] if "ood" in args.modes else []
        for k, ep in enumerate(episodes):
            for mode in args.modes:
                variant = variants[k % len(variants)] if mode == "ood" else None
                for offset in args.seed_offset:
                    tag = f"{ep['id']}-{mode}" + (f"-{variant}" if variant else "") + (f"-s{offset}" if offset else "")
                    items.append({"task": task, "episode": ep["id"], "mode": mode, "variant": variant,
                                  "seed_offset": offset, "out": run / "pkgs" / recipe_name / task / tag, "rank": k,
                                  "tag": tag})
    if args.gate:
        if len(args.tasks) != 1:
            raise SystemExit("--gate covers one task")
        import h3_gate
        h3_gate.check_enqueue(run, args.gate, recipe_name, args.tasks[0], sorted({it["episode"] for it in items}))
    # round-robin order: episode rank, then mode, then task, then seed
    items.sort(key=lambda it: (it["rank"], args.modes.index(it["mode"]), args.tasks.index(it["task"]),
                               args.seed_offset.index(it["seed_offset"])))

    def build(it):
        if (it["out"] / "ticket.json").exists():
            return it, "exists"
        cmd = ["nice", "-n", "10", python, str(HERE / "h3_package.py"), "build",
               "--bundle", str(bundle_file(run, it["task"], recipe)), "--recipe", str(Path(args.recipe).resolve()),
               "--episode", it["episode"], "--mode", it["mode"], "--seed-offset", str(it["seed_offset"]),
               "--out", str(it["out"]), "--run-dir", str(run)]
        if it["variant"]:
            cmd += ["--variant", it["variant"]]
        with build_slot(run, args.build_slots):
            r = subprocess.run(cmd, capture_output=True, text=True)
        return it, ("built" if r.returncode == 0 else "FAILED: " + r.stderr[-800:])

    if args.dry_run:
        for it in items:
            print(it["tag"], it["task"])
        return 0
    first = [it for it in items if it["seed_offset"] == args.seed_offset[0]]
    later = [it for it in items if it["seed_offset"] != args.seed_offset[0]]
    with cf.ThreadPoolExecutor(4) as ex:   # first seed first: its builds fill the HED cache the later seeds reuse
        done = dict(zip(map(id, first), ex.map(build, first)))
        done.update(zip(map(id, later), ex.map(build, later)))
    results = [done[id(it)] for it in items]
    q = run / "queue"
    known = set()
    for f in list(q.glob("pending/*/*.json")) + [f for sub in ("running", "done") for f in (q / sub).glob("*.json")]:
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        known.add((d.get("ticket") or d).get("package"))
    for n, (it, status) in enumerate(results):
        if status.startswith("FAILED"):
            print(it["task"], it["tag"], status)
            continue
        ticket = json.loads((it["out"] / "ticket.json").read_text())
        stem = f"{args.priority:03d}-{n:03d}-{recipe_name}-{it['task']}-{it['tag']}"
        if ticket["package"] in known:
            print("skip (already queued/done)", stem)
            continue
        dest = q / "pending" / ticket["profile"]
        dest.mkdir(parents=True, exist_ok=True)
        for suffix, sub in (chunk_tickets(ticket) if args.per_chunk else [("", ticket)]):
            (dest / f"{stem}{suffix}.json").write_text(json.dumps(sub, indent=2))
            print(status, f"{stem}{suffix}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
