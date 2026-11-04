#!/usr/bin/env python3
"""Generic batch supervisor CLI: drive a JSON task config with ``datafarm.batch.supervisor``.

A per-task ``while True`` supervisor becomes a config file (see docs/reference/batch-supervisor.md):

    tools/supervise_batch.py validate CONFIG            # check templates/policy, print the plan
    tools/supervise_batch.py run CONFIG                 # run to a terminal status (resumable)
    tools/supervise_batch.py status CONFIG              # read-only summary (no lock)
    tools/supervise_batch.py stop CONFIG --reason TEXT  # ask a running supervisor to stop
    tools/supervise_batch.py acknowledge CONFIG --reason TEXT
    tools/supervise_batch.py reopen CONFIG --items A,B --reason TEXT [--extra-attempts N]
    tools/supervise_batch.py resolve CONFIG --item A --outcome not_launched|crashed --reason TEXT
    tools/supervise_batch.py revoke CONFIG --items A,B --reason TEXT [--gate cohort_audit]

Launch production runs through ``tools/datafarm_runs.py launch`` so the ledger sees the worker; with a
``ledger.column`` in the config the supervisor then mirrors per-item cells into that run.

Exit codes of ``run``: 0 complete, 3 complete_with_gaps, 4 stopped_*, 5 paused_*, 2 usage errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datafarm.batch.supervisor import (  # noqa: E402
    COMPLETE,
    COMPLETE_WITH_GAPS,
    PAUSED_STATUSES,
    BatchError,
    BatchPolicy,
    BatchSupervisor,
)
from datafarm.batch.subprocess_task import (  # noqa: E402
    SubprocessTask,
    TaskConfigError,
    load_task_config,
)

STOP_FILE = "STOP"


def _exit_code(status: str) -> int:
    if status == COMPLETE:
        return 0
    if status == COMPLETE_WITH_GAPS:
        return 3
    if status in PAUSED_STATUSES:
        return 5
    return 4


def _open(config_path: Path) -> tuple[BatchSupervisor, dict, Path]:
    config, base_dir, items = load_task_config(config_path)
    adapter = SubprocessTask(config, base_dir=base_dir, items=items)
    policy = BatchPolicy.from_dict(config.get("policy"))
    ledger_column = (config.get("ledger") or {}).get("column")
    stop_file = adapter.work_dir / STOP_FILE

    def stop_check() -> str | None:
        if stop_file.is_file():
            return stop_file.read_text(encoding="utf-8").strip() or "stop file present"
        return None

    supervisor = BatchSupervisor(
        adapter.work_dir / "state", items, adapter.task(), policy,
        batch_id=config.get("batch_id"), ledger="env", ledger_column=ledger_column,
        stop_check=stop_check)
    return supervisor, config, adapter.work_dir


def cmd_validate(args: argparse.Namespace) -> int:
    config, base_dir, items = load_task_config(args.config)
    adapter = SubprocessTask(config, base_dir=base_dir, items=items)
    policy = BatchPolicy.from_dict(config.get("policy"))
    task = adapter.task()
    first = items[0]
    seed = task.seed_for(first, 1) if task.seed_for else None
    values = adapter.values(first, 1, seed if seed is not None else 0, f"{first}.a01.t01")
    argv, cwd, _ = adapter.command(config["launch"], values)
    print(json.dumps({
        "batch_id": config.get("batch_id"), "items": len(items), "first_items": items[:5],
        "work_dir": str(adapter.work_dir), "state_dir": str(adapter.work_dir / "state"),
        "policy": policy.to_dict(), "policy_sha256": policy.sha256(),
        "example_launch": {"argv": argv, "cwd": cwd, "result_path": values["result_path"]},
        "ledger_column": (config.get("ledger") or {}).get("column"),
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    supervisor, _, _ = _open(args.config)
    with supervisor:
        status = supervisor.run(max_ticks=args.max_ticks)
        print(json.dumps({key: value for key, value in supervisor.summary().items()
                          if key not in ("items", "policy")}, indent=2, ensure_ascii=False))
    if args.max_ticks is not None and status not in (COMPLETE, COMPLETE_WITH_GAPS) and \
            not status.startswith(("stopped_", "paused_")):
        return 0
    return _exit_code(status)


def cmd_status(args: argparse.Namespace) -> int:
    config, base_dir, items = load_task_config(args.config)
    adapter = SubprocessTask(config, base_dir=base_dir, items=items)
    summary = BatchSupervisor.read_summary(adapter.work_dir / "state")
    if not args.verbose:
        summary = {key: value for key, value in summary.items() if key not in ("items", "policy")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    config, base_dir, items = load_task_config(args.config)
    adapter = SubprocessTask(config, base_dir=base_dir, items=items)
    adapter.work_dir.mkdir(parents=True, exist_ok=True)
    (adapter.work_dir / STOP_FILE).write_text(args.reason + "\n", encoding="utf-8")
    print(f"wrote {adapter.work_dir / STOP_FILE}; the running supervisor stops launching on its "
          "next tick and drains in-flight attempts")
    return 0


def cmd_acknowledge(args: argparse.Namespace) -> int:
    supervisor, _, work_dir = _open(args.config)
    stop_file = work_dir / STOP_FILE
    if stop_file.is_file():
        stop_file.unlink()  # otherwise the next run would stop again immediately
    with supervisor:
        previous = supervisor.acknowledge(args.reason)
    print(json.dumps({"previous_status": previous, "status": "running"}))
    return 0


def cmd_reopen(args: argparse.Namespace) -> int:
    supervisor, _, _ = _open(args.config)
    with supervisor:
        reopened = supervisor.reopen([item for item in args.items.split(",") if item],
                                     reason=args.reason, extra_attempts=args.extra_attempts)
    print(json.dumps({"reopened": reopened}))
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    """A cohort-level verifier (e.g. a dense audit) withdrew admissions: retry those items."""

    supervisor, _, _ = _open(args.config)
    items = [item for item in args.items.split(",") if item]
    with supervisor:
        for item in items:
            supervisor.revoke_admission(item, reason=args.reason, gate=args.gate)
    print(json.dumps({"revoked": items, "gate": args.gate}))
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    supervisor, _, _ = _open(args.config)
    with supervisor:
        supervisor.resolve_unknown(args.item, outcome=args.outcome, reason=args.reason)
    print(json.dumps({"resolved": args.item, "outcome": args.outcome}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "status", "stop", "acknowledge", "reopen", "resolve",
                 "revoke"):
        command = sub.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "run":
            command.add_argument("--max-ticks", type=int)
        if name == "status":
            command.add_argument("--verbose", "-v", action="store_true")
        if name in ("stop", "acknowledge", "reopen", "resolve", "revoke"):
            command.add_argument("--reason", required=True)
        if name in ("reopen", "revoke"):
            command.add_argument("--items", required=True)
        if name == "reopen":
            command.add_argument("--extra-attempts", type=int, default=1)
        if name == "revoke":
            command.add_argument("--gate", default="cohort_audit")
        if name == "resolve":
            command.add_argument("--item", required=True)
            command.add_argument("--outcome", required=True, choices=("not_launched", "crashed"))
    args = parser.parse_args(argv)
    handlers = {"validate": cmd_validate, "run": cmd_run, "status": cmd_status, "stop": cmd_stop,
                "acknowledge": cmd_acknowledge, "reopen": cmd_reopen, "resolve": cmd_resolve,
                "revoke": cmd_revoke}
    try:
        return handlers[args.command](args)
    except (TaskConfigError, BatchError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
