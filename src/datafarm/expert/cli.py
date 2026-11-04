"""``tools/expert_harness.py`` command line (see docs/reference/expert-harness.md).

Every command prints JSON. Exit codes: 0 ok, 1 failure (lint errors, refused action),
2 usage, 75 round paused (provider outage or operator interrupt; run-round resumes it).
Commands that spend money or GPU time refuse to run without an explicit approval flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from datafarm.expert.adapter import TaskAdapter, load_holdout_seeds, validate_holdout_against_dev, write_holdout_seeds
from datafarm.expert.brief import lint_brief, render_brief
from datafarm.expert.handoff import build_handoff, handoff_markdown, load_bundle
from datafarm.expert.holdout import draw_holdout_seeds, evaluate_holdout
from datafarm.expert.layout import RoundPaths
from datafarm.expert.library import ExpertLibrary, add_from_round
from datafarm.expert.manifest import ManifestGuard
from datafarm.expert.rollout import make_backend
from datafarm.expert.round import (EXIT_PAUSED, RoundRunner, harness_code_sha256, load_round_adapter, manifest_entries,
                    prepare_round, rollout_masks, round_status)
from datafarm.expert.state import EventLog, RoundState
from datafarm.expert.util import Clock, HarnessError, deep_merge


def _print(value: Any) -> None:
    sys.stdout.write(json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _adapter(args: argparse.Namespace) -> TaskAdapter:
    overrides: dict[str, Any] = {}
    if getattr(args, "backend", None):
        overrides = deep_merge(overrides, {"agent": {"backend": args.backend}})
    if getattr(args, "model", None):
        overrides = deep_merge(overrides, {"agent": {"model": args.model}})
    return TaskAdapter.load(args.adapter, overrides=overrides or None)


def cmd_lint_brief(args: argparse.Namespace) -> int:
    adapter = _adapter(args)
    handoff_md = handoff_markdown(load_bundle(args.handoff)) if args.handoff else None
    text = Path(args.brief).read_text(encoding="utf-8") if args.brief else render_brief(adapter, handoff_md=handoff_md)
    seeds = [s.seed for s in load_holdout_seeds(args.holdout_seeds)] if args.holdout_seeds else None
    report = lint_brief(text, adapter, holdout_seeds=seeds)
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8")
    _print({"brief": args.brief or f"rendered from {adapter.brief_template_path.name}", **report.to_dict()})
    return 0 if report.ok else 1


def cmd_prepare(args: argparse.Namespace) -> int:
    adapter = _adapter(args)
    overrides = {k: v for k, v in {"backend": args.backend, "model": args.model}.items() if v}
    result = prepare_round(adapter, args.round_dir, clock=Clock(), round_id=args.round_id,
                           holdout_seeds_path=args.holdout_seeds, handoff_dir=args.handoff,
                           seed_expert=args.seed_expert, library_dir=args.library, references=args.references,
                           overrides={"agent": overrides} if overrides else None)
    _print(result)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    paths = RoundPaths(Path(args.round_dir))
    adapter = load_round_adapter(paths, args.adapter)
    runner = RoundRunner(paths, adapter, clock=Clock(), rollout_backend_kind=args.rollout_backend,
                         isolation_override=args.isolation, approve_agent_spend=args.approve_agent_spend,
                         approve_gpu=args.approve_gpu, tick_interval_s=args.tick,
                         extra_call_ledger=args.call_ledger)
    summary = runner.run()
    _print(summary)
    return EXIT_PAUSED if summary.get("exit_code") == EXIT_PAUSED else 0


def cmd_status(args: argparse.Namespace) -> int:
    _print(round_status(RoundPaths(Path(args.round_dir))))
    return 0


def cmd_eval_holdout(args: argparse.Namespace) -> int:
    paths = RoundPaths(Path(args.round_dir))
    adapter = load_round_adapter(paths, args.adapter)
    kind = args.rollout_backend or adapter.rollout.get("backend")
    if kind == "command" and not args.approve_gpu:
        raise HarnessError("holdout evaluation submits GPU jobs; pass --approve-gpu after the user approved it")
    clock = Clock()
    backend = make_backend(adapter, kind=kind, masked=rollout_masks(adapter, paths, purpose="holdout"),
                           isolation_override=args.isolation, clock_now=clock.now)
    result = evaluate_holdout(paths, adapter, backend, clock=clock, targets=args.targets,
                              max_concurrent=args.max_concurrent, poll_s=args.poll,
                              manifest_entries=manifest_entries(adapter, paths),
                              manifest_extra={"harness_code_sha256": harness_code_sha256()})
    _print({k: result[k] for k in ("round_id", "verifier_version", "seeds", "targets", "final_submission",
                                   "final", "curve", "rollouts")})
    return 0


def cmd_smoke_rollout(args: argparse.Namespace) -> int:
    """One rollout of a given expert outside any round: checks the job template, the rollout
    sandbox and the judge on the real backend before a live round."""

    import time

    from datafarm.expert.audit import audit_expert_source
    from datafarm.expert.rollout import AttemptSpec, prepare_run_dir

    adapter = _adapter(args)
    kind = args.rollout_backend or adapter.rollout.get("backend")
    if kind == "command" and not args.approve_gpu:
        raise HarnessError("a smoke rollout on the command backend submits a GPU job; pass --approve-gpu")
    if args.seed not in adapter.dev_seeds:
        raise HarnessError(f"seed {args.seed} is not a development seed ({adapter.dev_seeds.describe()})")
    source = Path(args.expert).read_bytes()
    audit = audit_expert_source(source, args.expert_class, adapter.expert_policy)
    if not audit.ok:
        raise HarnessError("static expert audit failed: " + audit.summary())
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise HarnessError(f"{out} is not empty")
    spec = AttemptSpec("SMOKE", "dev", args.seed, args.expert_class, out / f"smoke-s{args.seed}",
                       f"xh-smoke-s{args.seed}")
    iso = adapter.isolation("rollout")
    masks = (adapter.shell_paths(adapter.section("workspace").get("protected", []))
             + adapter.shell_paths(iso.get("extra_masked", [])))
    backend = make_backend(adapter, kind=kind, masked=masks, isolation_override=args.isolation)
    prepare_run_dir(spec, source, adapter)
    receipt = backend.launch(spec)
    deadline = time.time() + float(adapter.rollout.get("max_wall_s", 3600)) + 60
    status = "running"
    while status == "running" and time.time() < deadline:
        time.sleep(args.poll)
        status = backend.poll(spec, receipt, time.time())
    outcome = backend.outcome(spec, receipt, status)
    probe = spec.run_dir / "isolation-probe.txt"
    _print({"run_dir": str(spec.run_dir), "status": status, "outcome": outcome.to_dict(),
            "job": json.loads((spec.run_dir / "harness-job.json").read_text())
            if (spec.run_dir / "harness-job.json").exists() else None,
            "isolation_probe": probe.read_text().splitlines() if probe.exists() else None})
    return 0 if outcome.outcome_class in ("judged", "expert_error") else 1


def cmd_amend(args: argparse.Namespace) -> int:
    paths = RoundPaths(Path(args.round_dir))
    adapter = load_round_adapter(paths, args.adapter)
    state = RoundState.load(paths.state)
    clock = Clock()
    guard = ManifestGuard(paths, manifest_entries(adapter, paths), state, EventLog(paths, state.data["round_id"],
                          clock.now), clock.now, extra={"harness_code_sha256": harness_code_sha256()})
    _print(guard.rebaseline(reason=args.reason, operator=args.operator))
    return 0


def cmd_handoff(args: argparse.Namespace) -> int:
    paths = RoundPaths(Path(args.round_dir))
    adapter = load_round_adapter(paths, args.adapter)
    bundle = build_handoff(paths, adapter, args.out, clock=Clock())
    lint = json.loads((Path(args.out) / "next-brief.lint.json").read_text())
    _print({"bundle": str(Path(args.out) / "bundle.json"), "frozen_expert": bundle["frozen_expert"],
            "files": bundle["files"], "next_brief_lint_ok": lint["ok"], "next_brief_errors": lint["errors"]})
    return 0 if lint["ok"] else 1


def cmd_make_holdout_seeds(args: argparse.Namespace) -> int:
    adapter = _adapter(args)
    out = Path(args.out)
    if out.exists():
        raise HarnessError(f"{out} exists; holdout seed files are never overwritten")
    seeds = draw_holdout_seeds(adapter, per_family=args.per_family, spares=args.spares,
                               range_start=args.range_start, range_size=args.range_size, count=args.count)
    validate_holdout_against_dev(seeds, adapter.dev_seeds)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_holdout_seeds(out, seeds, note=f"drawn for {adapter.task_id}; operator-only")
    families: dict[str, int] = {}
    for seed in seeds:
        if not seed.spare:
            families[str(seed.family)] = families.get(str(seed.family), 0) + 1
    _print({"out": str(out), "seeds": len([s for s in seeds if not s.spare]),
            "spares": len([s for s in seeds if s.spare]), "per_family": families,
            "note": "seed values are not printed; the file is mode 0600"})
    return 0


def cmd_library(args: argparse.Namespace) -> int:
    library = ExpertLibrary(args.library)
    if args.library_cmd == "add":
        paths = RoundPaths(Path(args.round_dir))
        adapter = load_round_adapter(paths, args.adapter)
        entry = add_from_round(library, paths, adapter, now=Clock().now(), min_pass_rate=args.min_pass_rate)
        _print({"admitted": entry is not None, "entry": entry})
        return 0
    _print(library.find(task_id=args.task, limit=args.limit))
    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    from datafarm.expert.evalsuite import load_suite, plan_suite, report_suite, run_suite

    clock = Clock()
    if args.suite_cmd == "plan":
        plan = plan_suite(load_suite(args.suite), args.out, now=clock.now())
        _print({"out": args.out, "counts": plan["counts"],
                "cells": [{k: c.get(k) for k in ("cell_id", "status", "missing_interfaces")} for c in plan["cells"]]})
        return 0
    if args.suite_cmd == "run":
        report = run_suite(args.suite, args.out, clock=clock, fake=args.fake,
                           approve_agent_spend=args.approve_agent_spend, approve_gpu=args.approve_gpu,
                           only=args.only or (), isolation=args.isolation)
        _print(report)
        return 0
    _print(report_suite(args.out, now=clock.now()))
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    from datafarm.expert.dryrun import run_dry_run

    result = run_dry_run(args.out, isolation=args.isolation, log=lambda m: print(m, file=sys.stderr))
    _print(result)
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tools/expert_harness.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lint-brief", help="render (or read) a brief and lint it; exit 1 on errors")
    p.add_argument("--adapter", required=True)
    p.add_argument("--brief", help="lint this file instead of rendering the template")
    p.add_argument("--handoff", help="render with this handoff bundle")
    p.add_argument("--holdout-seeds", help="also check the brief does not mention these seeds")
    p.add_argument("--write", help="write the rendered brief here")
    p.add_argument("--backend")
    p.add_argument("--model")
    p.set_defaults(func=cmd_lint_brief)

    p = sub.add_parser("prepare-round", help="create a round directory (brief, tool, private holdout copy)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--round-dir", required=True)
    p.add_argument("--round-id")
    p.add_argument("--holdout-seeds")
    p.add_argument("--handoff")
    p.add_argument("--seed-expert")
    p.add_argument("--library")
    p.add_argument("--references", type=int, default=0)
    p.add_argument("--backend", choices=("claude", "codex", "codex-metered", "fake"))
    p.add_argument("--model")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run-round", help="run (or resume) the session loop of a prepared round")
    p.add_argument("--round-dir", required=True)
    p.add_argument("--adapter", help="defaults to the adapter recorded in round.json")
    p.add_argument("--rollout-backend", choices=("command", "local", "fake"))
    p.add_argument("--isolation", choices=("bwrap", "unshare", "none"))
    p.add_argument("--approve-agent-spend", action="store_true")
    p.add_argument("--approve-gpu", action="store_true")
    p.add_argument("--tick", type=float, default=1.0)
    p.add_argument("--call-ledger", help="also append agent-call records to this shared ledger")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="operator view of a round")
    p.add_argument("--round-dir", required=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("eval-holdout", help="harness-only holdout evaluation of the round's submissions")
    p.add_argument("--round-dir", required=True)
    p.add_argument("--adapter")
    p.add_argument("--targets", choices=("final", "all"))
    p.add_argument("--rollout-backend", choices=("command", "local", "fake"))
    p.add_argument("--isolation", choices=("bwrap", "unshare", "none"))
    p.add_argument("--max-concurrent", type=int)
    p.add_argument("--poll", type=float, default=15.0)
    p.add_argument("--approve-gpu", action="store_true")
    p.set_defaults(func=cmd_eval_holdout)

    p = sub.add_parser("smoke-rollout", help="one rollout of an expert file outside any round (backend check)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--expert", required=True)
    p.add_argument("--class", dest="expert_class", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rollout-backend", choices=("command", "local", "fake"))
    p.add_argument("--isolation", choices=("bwrap", "unshare", "none"))
    p.add_argument("--poll", type=float, default=15.0)
    p.add_argument("--approve-gpu", action="store_true")
    p.set_defaults(func=cmd_smoke_rollout)

    p = sub.add_parser("amend", help="record an operator amendment and re-baseline the tamper manifest")
    p.add_argument("--round-dir", required=True)
    p.add_argument("--adapter")
    p.add_argument("--reason", required=True)
    p.add_argument("--operator", default="operator")
    p.set_defaults(func=cmd_amend)

    p = sub.add_parser("handoff", help="build the handoff bundle and the next brief from a finished round")
    p.add_argument("--round-dir", required=True)
    p.add_argument("--adapter")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_handoff)


    p = sub.add_parser("make-holdout-seeds", help="draw family-balanced holdout seeds (operator-only file)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-family", type=int, default=4)
    p.add_argument("--spares", type=int, default=2)
    p.add_argument("--count", type=int)
    p.add_argument("--range-start", type=int, default=100_000)
    p.add_argument("--range-size", type=int, default=1_000_000)
    p.set_defaults(func=cmd_make_holdout_seeds)

    p = sub.add_parser("library", help="verified expert library")
    lib = p.add_subparsers(dest="library_cmd", required=True)
    q = lib.add_parser("add")
    q.add_argument("--library", required=True)
    q.add_argument("--round-dir", required=True)
    q.add_argument("--adapter")
    q.add_argument("--min-pass-rate", type=float)
    q = lib.add_parser("list")
    q.add_argument("--library", required=True)
    q.add_argument("--task")
    q.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_library)

    p = sub.add_parser("suite", help="task-expert eval suite (plan | run | report)")
    s = p.add_subparsers(dest="suite_cmd", required=True)
    q = s.add_parser("plan")
    q.add_argument("--suite", required=True)
    q.add_argument("--out", required=True)
    q = s.add_parser("run")
    q.add_argument("--suite", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--fake", action="store_true", help="fake agent + fake simulator (no spend)")
    q.add_argument("--only", action="append")
    q.add_argument("--isolation", choices=("bwrap", "unshare", "none"))
    q.add_argument("--approve-agent-spend", action="store_true")
    q.add_argument("--approve-gpu", action="store_true")
    q = s.add_parser("report")
    q.add_argument("--out", required=True)
    p.set_defaults(func=cmd_suite)

    p = sub.add_parser("dry-run", help="fake agent + fake rollouts end to end in a temp directory")
    p.add_argument("--out")
    p.add_argument("--isolation", choices=("auto", "bwrap", "unshare", "none"), default="auto")
    p.set_defaults(func=cmd_dry_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HarnessError as exc:
        _print({"error": type(exc).__name__, "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
