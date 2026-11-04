"""``tools/datafarm_runs.py ...``: operator commands for the pipeline run ledger."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from datafarm.ledger.ledger import (
    CellUpdate,
    LedgerError,
    RunLedger,
    git_fingerprint,
    ledger_root,
    list_runs,
    make_run_id,
    plan_stages,
    probe_pid,
)


def _items(args: argparse.Namespace) -> list[str]:
    items: list[str] = []
    if getattr(args, "items", None):
        items.extend(part.strip() for part in args.items.split(",") if part.strip())
    if getattr(args, "items_file", None):
        text = Path(args.items_file).read_text(encoding="utf-8")
        try:
            value = json.loads(text)
        except ValueError:
            value = [line.strip() for line in text.splitlines() if line.strip()]
        if not isinstance(value, list):
            raise LedgerError("--items-file must be a JSON list or one id per line")
        items.extend(str(item) for item in value)
    if getattr(args, "item_pattern", None):
        pattern, start, stop = args.item_pattern.rsplit(":", 2)
        items.extend(pattern % index for index in range(int(start), int(stop)))
    return items


def _open(args: argparse.Namespace, run_id: str) -> RunLedger:
    return RunLedger(ledger_root(args.root) / run_id)


def cmd_create(args: argparse.Namespace) -> int:
    substages: dict[str, list[str]] = {}
    for spec in args.substages or []:
        stage, _, names = spec.partition("=")
        substages[stage] = [name for name in names.split(",") if name]
    sync = []
    for spec in args.sync or []:
        target, _, raw = spec.partition("@")
        sync.append({"callable": target, "args": json.loads(raw) if raw else {}})
    run_id = args.run_id or make_run_id(args.slug)
    ledger = RunLedger.create(
        ledger_root(args.root),
        run_id=run_id,
        title=args.title,
        request=args.request,
        authorization=args.authorization,
        task_family=args.task_family,
        stages=plan_stages(args.entry, args.exit, substages),
        items=_items(args),
        cohort_manifest=args.cohort_manifest,
        code=git_fingerprint(args.code_root) if args.code_root else {},
        configs=args.config or [],
        parent_run=args.parent,
        links=args.link or [],
        sync=sync,
        stale_seconds=args.stale_seconds,
        created_at=args.created_at,
        backfilled_from=args.backfilled_from,
    )
    print(json.dumps({"created": ledger.run_id, "directory": str(ledger.directory)}, indent=2))
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    ledger = _open(args, args.run)
    command = list(getattr(args, "launch_command", None) or [])
    if not command:
        raise LedgerError("launch needs a command after --")
    log = Path(args.log or ledger.directory / "logs" / f"{args.role}-{int(time.time())}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    worker_id = ledger.register_worker(
        role=args.role, host=socket.gethostname(), log=str(log), command=command,
        progress_paths=args.progress or [],
    )
    with open(log, "ab") as stream:
        sentinel = subprocess.Popen(
            [sys.executable, "-m", "datafarm.ledger.ledger_exec", str(ledger.directory),
             worker_id, str(args.heartbeat_seconds), "--", *command],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            cwd=args.cwd or os.getcwd(), start_new_session=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                [str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", "")])},
        )
    ledger.update_worker(worker_id, pid=sentinel.pid, sid=sentinel.pid)
    ledger.record_event("launched", worker=worker_id, pid=sentinel.pid, log=str(log))
    time.sleep(args.settle_seconds)
    alive = probe_pid(None, sentinel.pid) and sentinel.poll() is None
    check = {"worker": worker_id, "sentinel_pid": sentinel.pid, "sentinel_alive": bool(alive),
             "log": str(log)}
    ledger.record_event("persistence_check", check=1, **check)
    print(json.dumps(check, indent=2))
    if not alive:
        print("launch did not persist; see the log", file=sys.stderr)
        return 1
    print(f"second check: tools/datafarm_runs.py check {ledger.run_id}  (in a new shell, after a short wait)")
    return 0


def cmd_attach_worker(args: argparse.Namespace) -> int:
    ledger = _open(args, args.run)
    worker_id = ledger.register_worker(
        role=args.role, host=args.host, pid=args.pid, sid=args.sid, lease_id=args.lease,
        log=args.log, progress_paths=args.progress or [], started_at=args.started_at,
        note=args.note,
    )
    if args.ended_exit_code is not None:
        ledger.end_worker(worker_id, args.ended_exit_code, reason=args.note)
    print(worker_id)
    return 0


def cmd_close_worker(args: argparse.Namespace) -> int:
    ledger = _open(args, args.run)
    if not ledger.close_worker(args.worker, reason=args.reason, exit_code=args.exit_code):
        raise LedgerError(f"worker {args.worker} is unknown or already closed")
    return 0


def cmd_cells(args: argparse.Namespace) -> int:
    ledger = _open(args, args.run)
    items = _items(args) or list(ledger.matrix()["cells"])
    changed = ledger.set_cells(
        [CellUpdate(item, args.column, args.status, args.attempt, args.failure, args.artifact,
                    args.note) for item in items],
        source=args.source,
    )
    print(json.dumps({"changed": changed}))
    return 0


def cmd_event(args: argparse.Namespace) -> int:
    _open(args, args.run).record_event(args.kind, message=args.message, source=args.source)
    return 0


def cmd_declare(args: argparse.Namespace) -> int:
    _open(args, args.run).declare(args.status, reason=args.reason, source=args.source)
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    _open(args, args.run).finalize(args.verification, source=args.source,
                                   allow_gaps=args.allow_gaps)
    return 0


def cmd_supersede(args: argparse.Namespace) -> int:
    _open(args, args.run).supersede(args.by, reason=args.reason)
    return 0


def _age(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 120:
        return f"{seconds}s"
    if seconds < 7200:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"


def _render(status: dict[str, Any], sync: list[dict[str, Any]], *, verbose: bool) -> str:
    lines = [
        f"{status['run_id']}  [{status['state']}]  {status['title']}",
        f"  range {status['range']}  passed-at-exit {status['passed_exit']}/{status['denominator']}"
        f"  last-event {status['last_event_at']}",
    ]
    for note in status["attention"]:
        lines.append(f"  ! {note}")
    for result in sync:
        if not result.get("ok"):
            lines.append(f"  ! sync {result['callable']} failed: {result.get('error')}")
    if status["gaps_by_stop"]:
        stops = ", ".join(
            f"{column}: " + "/".join(f"{k}={v}" for k, v in sorted(counts.items()))
            for column, counts in status["gaps_by_stop"].items()
        )
        lines.append(f"  gaps  {stops}")
    for worker in status["workers"]:
        if not verbose and worker["liveness"] == "exited" and worker.get("exit_code") == 0:
            continue
        where = f"{worker.get('host') or '?'}:{worker.get('pid') or worker.get('lease_id') or '?'}"
        exit_note = f" exit={worker['exit_code']}" if worker["liveness"] == "exited" else ""
        lines.append(
            f"  worker {worker['worker_id']:<28} {worker['liveness']:<14} {where}"
            f"  progress-age {_age(worker.get('last_progress_age_s'))}{exit_note}"
        )
    if status["children"]:
        lines.append(f"  children {', '.join(status['children'])}")
    if verbose:
        for column, counts in status["column_counts"].items():
            lines.append(f"    {column:<32} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def _status_for(ledger: RunLedger, args: argparse.Namespace) -> tuple[dict, list]:
    sync = [] if args.no_sync else ledger.sync()
    return ledger.status(probe=not args.no_probe), sync


def cmd_status(args: argparse.Namespace) -> int:
    root = ledger_root(args.root)
    ledgers = [RunLedger(root / run) for run in args.runs] if args.runs else list_runs(root)
    reports = []
    for ledger in ledgers:
        status, sync = _status_for(ledger, args)
        if not args.runs and not args.all and status["state"] in {"complete", "superseded", "cancelled"}:
            continue
        if args.gaps:
            status["gaps"] = ledger.gaps()
        reports.append((status, sync))
    if args.json:
        print(json.dumps([{**s, "sync": y} for s, y in reports], indent=2, ensure_ascii=False))
    else:
        if not reports:
            print("no open runs (use --all to include finished ones)")
        for status, sync in reports:
            print(_render(status, sync, verbose=args.verbose))
            if args.gaps:
                groups: dict[tuple, list[str]] = {}
                for item, gap in sorted(status["gaps"].items()):
                    key = (gap["column"], gap["status"], gap.get("failure") or "")
                    groups.setdefault(key, []).append(item)
                for (column, state, failure), items in groups.items():
                    shown = items if args.verbose or len(items) <= 12 else items[:12] + ["..."]
                    reason = f" [{failure}]" if failure else ""
                    print(f"    gap {column} {state}{reason} x{len(items)}: {', '.join(shown)}")
            print()
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Independent persistence check: records liveness and progress as evidence."""

    ledger = _open(args, args.run)
    ledger.sync()
    status = ledger.status()
    live = [
        {"worker": w["worker_id"], "liveness": w["liveness"], "pid": w.get("pid"),
         "progress_age_s": w.get("last_progress_age_s")}
        for w in status["workers"] if w["liveness"] != "exited"
    ]
    ledger.record_event("persistence_check", check=2, state=status["state"], workers=live,
                        passed_exit=status["passed_exit"], host=socket.gethostname())
    print(_render(status, [], verbose=True))
    ok = status["state"] == "running"
    print("persistence check:", "PASS" if ok else f"NOT RUNNING ({status['state']})")
    return 0 if ok else 1


def make_parser() -> argparse.ArgumentParser:
    runs = argparse.ArgumentParser(prog="tools/datafarm_runs.py", description="pipeline run ledger")
    runs.add_argument("--root", help="ledger root (default $DATAFARM_LEDGER_ROOT or ./runs/ledger)")
    sub = runs.add_subparsers(dest="runs_command", required=True)

    def with_items(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--items", help="comma-separated item ids")
        parser.add_argument("--items-file", help="JSON list or one id per line")
        parser.add_argument("--item-pattern", help="printf pattern:start:stop, e.g. sample-%%03d:0:100")

    create = sub.add_parser("create", help="register a requested pipeline run")
    create.add_argument("--slug", help="kebab-case slug; run id becomes PRUN-<time>-<slug>")
    create.add_argument("--run-id")
    create.add_argument("--title", required=True)
    create.add_argument("--request", required=True, help="the operator request, in their words")
    create.add_argument("--authorization")
    create.add_argument("--task-family", required=True)
    create.add_argument("--from", dest="entry", required=True)
    create.add_argument("--to", dest="exit", required=True)
    create.add_argument("--substages", action="append", help="stage=a,b,c (repeatable)")
    with_items(create)
    create.add_argument("--cohort-manifest")
    create.add_argument("--code-root")
    create.add_argument("--config", action="append")
    create.add_argument("--parent")
    create.add_argument("--link", action="append", help="external reference for this run, repeatable")
    create.add_argument("--sync", action="append", help="module:function@{json args}")
    create.add_argument("--stale-seconds", type=int, default=900)
    create.add_argument("--created-at", help="original request time when backfilling")
    create.add_argument("--backfilled-from", help="native state path when registering a past run")
    create.set_defaults(handler=cmd_create)

    launch = sub.add_parser("launch", help="start a detached worker tracked by a sentinel")
    launch.add_argument("run")
    launch.add_argument("--role", required=True)
    launch.add_argument("--log")
    launch.add_argument("--cwd")
    launch.add_argument("--progress", action="append", help="file whose mtime signals progress")
    launch.add_argument("--heartbeat-seconds", type=int, default=30)
    launch.add_argument("--settle-seconds", type=float, default=5.0)
    launch.set_defaults(handler=cmd_launch)

    attach = sub.add_parser("attach-worker", help="register an existing process or lease")
    attach.add_argument("run")
    attach.add_argument("--role", required=True)
    attach.add_argument("--host")
    attach.add_argument("--pid", type=int)
    attach.add_argument("--sid", type=int)
    attach.add_argument("--lease")
    attach.add_argument("--log")
    attach.add_argument("--progress", action="append")
    attach.add_argument("--started-at")
    attach.add_argument("--note")
    attach.add_argument("--ended-exit-code", type=int)
    attach.set_defaults(handler=cmd_attach_worker)

    close = sub.add_parser("close-worker", help="record that a worker ended (e.g. replaced by hand)")
    close.add_argument("run")
    close.add_argument("worker")
    close.add_argument("--reason", required=True)
    close.add_argument("--exit-code", type=int)
    close.set_defaults(handler=cmd_close_worker)

    cells = sub.add_parser("cells", help="set item x column status (all items if none given)")
    cells.add_argument("run")
    cells.add_argument("--column", required=True)
    cells.add_argument("--status", required=True)
    with_items(cells)
    cells.add_argument("--attempt", type=int)
    cells.add_argument("--failure")
    cells.add_argument("--artifact")
    cells.add_argument("--note")
    cells.add_argument("--source", default="operator")
    cells.set_defaults(handler=cmd_cells)

    event = sub.add_parser("event", help="append a free-form event (manual intervention, note)")
    event.add_argument("run")
    event.add_argument("--kind", default="note")
    event.add_argument("--message", required=True)
    event.add_argument("--source", default="operator")
    event.set_defaults(handler=cmd_event)

    declare = sub.add_parser("declare", help="declare running/blocked/failed/cancelled")
    declare.add_argument("run")
    declare.add_argument("--status", required=True)
    declare.add_argument("--reason", required=True)
    declare.add_argument("--source", default="operator")
    declare.set_defaults(handler=cmd_declare)

    finalize = sub.add_parser("finalize", help="close a run with its final verification report")
    finalize.add_argument("run")
    finalize.add_argument("--verification", required=True)
    finalize.add_argument("--allow-gaps", action="store_true")
    finalize.add_argument("--source", default="operator")
    finalize.set_defaults(handler=cmd_finalize)

    supersede = sub.add_parser("supersede", help="mark a run replaced by another run")
    supersede.add_argument("run")
    supersede.add_argument("--by", required=True)
    supersede.add_argument("--reason", required=True)
    supersede.set_defaults(handler=cmd_supersede)

    status = sub.add_parser("status", help="derived status of open runs (or the given runs)")
    status.add_argument("runs", nargs="*")
    status.add_argument("--all", action="store_true")
    status.add_argument("--gaps", action="store_true", help="list every item not passed at exit")
    status.add_argument("--json", action="store_true")
    status.add_argument("--verbose", "-v", action="store_true")
    status.add_argument("--no-sync", action="store_true")
    status.add_argument("--no-probe", action="store_true")
    status.set_defaults(handler=cmd_status)

    check = sub.add_parser("check", help="recorded persistence check (second, independent)")
    check.add_argument("run")
    check.set_defaults(handler=cmd_check)
    return runs


def split_launch_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """``launch RUN [opts] -- CMD...``: keep CMD away from argparse."""

    if "launch" in argv and "--" in argv:
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1 :]
    return argv, []


def handle(args: argparse.Namespace) -> int:
    try:
        return args.handler(args)
    except BrokenPipeError:
        return 0
    except LedgerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    argv, launch_command = split_launch_argv(list(sys.argv[1:] if argv is None else argv))
    args = make_parser().parse_args(argv)
    args.launch_command = launch_command
    return handle(args)


__all__ = ["handle", "main", "make_parser", "split_launch_argv"]
