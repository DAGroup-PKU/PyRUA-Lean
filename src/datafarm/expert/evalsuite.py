"""Task-Expert authoring eval suite: tasks x models x repeats, holdout-scored.

Each cell is one complete authoring round (prepare -> sessions -> holdout evaluation).
A round is *verified* when its final expert's holdout pass rate reaches the suite
threshold. Per (task, backend, model) the report gives:

* round-level ``pass@k`` = P(at least one of k independent rounds is verified) and
  ``pass^k`` = P(all k rounds are verified), with the unbiased estimators
  ``1 - C(n-c, k)/C(n, k)`` and ``C(c, k)/C(n, k)`` over n evaluated rounds with c verified;
* mean holdout pass rate, cost per verified expert (known cost / verified rounds), active
  wall time, counted rollouts, infrastructure failures;
* violations: tamper, isolation, concurrency, budget, holdout-seed and non-development
  seed requests, audit and unsafe-path rejections.

Tasks without a working adapter are listed with the exact interfaces they lack instead
of being faked.
"""

from __future__ import annotations

import collections
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from datafarm.expert.adapter import TaskAdapter
from datafarm.expert.layout import RoundPaths
from datafarm.expert.state import RoundState
from datafarm.expert.util import (Clock, ConfigError, HarnessError, atomic_write_json, atomic_write_text, read_json,
                   sha256_json, utc_now)

SUITE_SCHEMA = "datafarm.expert-harness.suite/1"
PLAN_SCHEMA = "datafarm.expert-harness.suite-plan/1"
REPORT_SCHEMA = "datafarm.expert-harness.suite-report/1"


def pass_at_k(n: int, c: int, k: int) -> float | None:
    if n < k or k <= 0:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float | None:
    if n < k or k <= 0:
        return None
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def load_suite(path: str | Path) -> dict[str, Any]:
    suite_path = Path(path)
    suite = read_json(suite_path)
    if suite.get("schema") != SUITE_SCHEMA:
        raise ConfigError(f"{path}: not a {SUITE_SCHEMA} file")
    if not suite.get("tasks") or not suite.get("models"):
        raise ConfigError("a suite needs tasks and models")
    if not isinstance(suite.get("repeats"), int) or suite["repeats"] < 1:
        raise ConfigError("repeats must be a positive integer")
    for task in suite["tasks"]:
        if not task.get("adapter") and not (task.get("unavailable") or {}).get("missing_interfaces"):
            raise ConfigError(f"task {task.get('task_id')} has neither an adapter nor missing_interfaces")
    for model in suite["models"]:
        if model.get("backend") not in ("claude", "codex", "fake") or not model.get("model"):
            raise ConfigError(f"invalid suite model {model!r}")
    suite["_dir"] = str(suite_path.resolve().parent)
    suite["_sha256"] = sha256_json({k: v for k, v in suite.items() if not k.startswith("_")})
    return suite


def plan_suite(suite: Mapping[str, Any], out_dir: str | Path, *, now: float) -> dict[str, Any]:
    out = Path(out_dir)
    cells = []
    for task in suite["tasks"]:
        task_id = task["task_id"]
        if not task.get("adapter"):
            cells.append({"cell_id": f"{task_id}__unavailable", "task_id": task_id, "status": "unavailable",
                          **task.get("unavailable", {})})
            continue
        adapter_path = (Path(suite["_dir"]) / task["adapter"]).resolve()
        adapter = TaskAdapter.load(adapter_path)
        if not adapter.available:
            cells.append({"cell_id": f"{task_id}__unavailable", "task_id": task_id, "status": "unavailable",
                          "missing_interfaces": adapter.raw.get("missing_interfaces")})
            continue
        for model in suite["models"]:
            for repeat in range(1, suite["repeats"] + 1):
                cell_id = f"{task_id}__{model['backend']}-{model['model']}__r{repeat}"
                cells.append({"cell_id": cell_id, "task_id": task_id, "status": "planned",
                              "adapter": str(adapter_path), "backend": model["backend"], "model": model["model"],
                              "repeat": repeat, "holdout_seeds": task.get("holdout_seeds"),
                              "round_dir": str(out / "rounds" / cell_id)})
    plan = {"schema": PLAN_SCHEMA, "suite_id": suite["suite_id"], "suite_sha256": suite["_sha256"],
            "created_at": utc_now(now), "verified_pass_rate": suite.get("verified_pass_rate", 0.5),
            "k": suite.get("k", [1]), "cells": cells,
            "counts": dict(collections.Counter(c["status"] for c in cells))}
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "plan.json", plan)
    return plan


def round_metrics(round_dir: str | Path, verified_pass_rate: float) -> dict[str, Any]:
    paths = RoundPaths(Path(round_dir))
    state = RoundState.load(paths.state)
    data = state.data
    round_json = read_json(paths.round_json)
    budgets = round_json["budgets"]
    holdout = read_json(paths.holdout_results) if paths.holdout_results.exists() else None
    final = (holdout or {}).get("final") or {}
    rate = final.get("pass_rate")
    counters = data["counters"]
    active = float(data.get("active_wall_s_final") or 0.0)
    cost = float(data.get("cost_usd_known") or 0.0)
    kinds = collections.Counter(v["kind"] for v in data.get("violations", []))
    iso = (data.get("isolation") or {}).get("agent") or {}
    rollout_iso = sum(1 for a in data["attempts"].values()
                      if (a.get("outcome") or {}).get("outcome_class") in ("isolation_failed", "isolation_unavailable"))
    violations = {
        "tamper": int(data["status"] == "invalid_tamper" or not data["manifest"].get("ok", True)),
        "isolation": int(not iso.get("ok", False)) + rollout_iso + int(counters.get("unisolated_rollouts", 0) > 0),
        "concurrency": int(int(data.get("max_in_flight_observed") or 0) > budgets["max_concurrent_rollouts"])
        + kinds.get("concurrency", 0) + kinds.get("late_finish", 0),
        "budget": int(counters.get("counted", 0) > budgets["max_counted_rollouts"])
        + int(budgets.get("max_usd") is not None and cost > budgets["max_usd"])
        + int(active > budgets["total_wall_s"] + 900),
        "holdout_seed_requests": counters.get("holdout_seed", 0),
        "non_dev_seed_requests": counters.get("non_dev_seed", 0),
        "audit_rejections": counters.get("audit_rejected", 0),
        "unsafe_paths": counters.get("unsafe_path", 0),
    }
    return {"round_id": data["round_id"], "status": data["status"], "evaluated": holdout is not None,
            "verified": None if rate is None else rate >= verified_pass_rate, "holdout_pass_rate": rate,
            "holdout_passed": final.get("passed"), "holdout_scored": final.get("scored"),
            "cost_usd_known": round(cost, 4), "sessions_unknown_cost": data.get("sessions_unknown_cost", 0),
            "sessions": len(data["sessions"]), "active_wall_s": active,
            "counted_rollouts": counters.get("counted", 0), "infra_failures": counters.get("infra", 0),
            "dev_successes": counters.get("successes", 0), "violations": violations,
            "curve": (holdout or {}).get("curve")}


def report_suite(out_dir: str | Path, *, now: float) -> dict[str, Any]:
    out = Path(out_dir)
    plan = read_json(out / "plan.json")
    threshold = float(plan.get("verified_pass_rate", 0.5))
    ks = plan.get("k", [1])
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for cell in plan["cells"]:
        if cell["status"] != "done":
            continue
        metrics = round_metrics(cell["round_dir"], threshold)
        cell["metrics"] = metrics
        groups[(cell["task_id"], cell["backend"], cell["model"])].append(metrics)
    rows = []
    for (task_id, backend, model), items in sorted(groups.items()):
        evaluated = [m for m in items if m["evaluated"]]
        n = len(evaluated)
        c = sum(1 for m in evaluated if m["verified"])
        total_cost = sum(m["cost_usd_known"] for m in items)
        rates = [m["holdout_pass_rate"] for m in evaluated if m["holdout_pass_rate"] is not None]
        violations: collections.Counter[str] = collections.Counter()
        for m in items:
            violations.update({k: v for k, v in m["violations"].items() if v})
        rows.append({
            "task_id": task_id, "backend": backend, "model": model, "rounds": len(items), "evaluated": n,
            "verified": c, **{f"pass@{k}": pass_at_k(n, c, k) for k in ks},
            **{f"pass^{k}": pass_hat_k(n, c, k) for k in ks},
            "mean_holdout_pass_rate": None if not rates else round(statistics.fmean(rates), 4),
            "cost_usd_known_total": round(total_cost, 4),
            "cost_per_verified_expert": None if c == 0 else round(total_cost / c, 4),
            "sessions_unknown_cost": sum(m["sessions_unknown_cost"] for m in items),
            "median_active_wall_s": statistics.median([m["active_wall_s"] for m in items]) if items else None,
            "mean_counted_rollouts": round(statistics.fmean([m["counted_rollouts"] for m in items]), 2) if items else None,
            "infra_failures": sum(m["infra_failures"] for m in items),
            "violations": dict(violations),
        })
    unavailable = [{"task_id": c["task_id"], "missing_interfaces": c.get("missing_interfaces"),
                    "existing": c.get("existing")} for c in plan["cells"] if c["status"] == "unavailable"]
    report = {"schema": REPORT_SCHEMA, "suite_id": plan["suite_id"], "suite_sha256": plan["suite_sha256"],
              "generated_at": utc_now(now), "verified_pass_rate": threshold, "k": ks, "rows": rows,
              "unavailable_tasks": unavailable,
              "cells": [{k: c.get(k) for k in ("cell_id", "status", "error")} for c in plan["cells"]]}
    atomic_write_json(out / "report.json", report)
    atomic_write_text(out / "report.md", report_markdown(report))
    return report


def report_markdown(report: Mapping[str, Any]) -> str:
    ks = report["k"]
    head = ["task", "backend/model", "rounds", "verified"] + [f"pass@{k}" for k in ks] + \
        [f"pass^{k}" for k in ks] + ["mean holdout", "$/verified", "median wall", "rollouts", "violations"]
    lines = [f"# Task-expert eval suite `{report['suite_id']}`", "",
             f"Generated {report['generated_at']}; a round is verified when its final expert passes at least "
             f"{report['verified_pass_rate']:.0%} of the hidden holdout seeds.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    for row in report["rows"]:
        cells = [row["task_id"], f"{row['backend']}/{row['model']}", row["rounds"], f"{row['verified']}/{row['evaluated']}"]
        cells += [fmt(row[f"pass@{k}"]) for k in ks] + [fmt(row[f"pass^{k}"]) for k in ks]
        cells += [fmt(row["mean_holdout_pass_rate"]), fmt(row["cost_per_verified_expert"]),
                  fmt(row["median_active_wall_s"]), fmt(row["mean_counted_rollouts"]),
                  ", ".join(f"{k}={v}" for k, v in row["violations"].items()) or "none"]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    if report["unavailable_tasks"]:
        lines += ["", "## Declared but unavailable tasks", ""]
        for item in report["unavailable_tasks"]:
            lines.append(f"- **{item['task_id']}**: missing " + "; ".join(item.get("missing_interfaces") or []))
    return "\n".join(lines) + "\n"


def run_suite(suite_path: str | Path, out_dir: str | Path, *, clock: Clock, fake: bool = False,
              approve_agent_spend: bool = False, approve_gpu: bool = False, only: Sequence[str] = (),
              isolation: str | None = None, fast: bool = False) -> dict[str, Any]:
    from datafarm.expert.holdout import evaluate_holdout
    from datafarm.expert.round import RoundRunner, harness_code_sha256, manifest_entries, prepare_round, rollout_masks
    from datafarm.expert.rollout import make_backend

    suite = load_suite(suite_path)
    out = Path(out_dir)
    runtime = Clock()  # rounds run real processes: budgets and holdout deadlines need real time
    plan = read_json(out / "plan.json") if (out / "plan.json").exists() else plan_suite(suite, out, now=clock.now())
    if plan["suite_sha256"] != suite["_sha256"]:
        raise HarnessError("the suite file changed since it was planned; plan into a new directory")
    world = None
    for cell in plan["cells"]:
        if cell["status"] != "planned" or (only and cell["cell_id"] not in only):
            continue
        try:
            overrides = {"agent": {"backend": cell["backend"], "model": cell["model"]}}
            if suite.get("budget_overrides"):
                overrides["budgets"] = suite["budget_overrides"]
            adapter = TaskAdapter.load(cell["adapter"], overrides=overrides)
            holdout_file = cell.get("holdout_seeds")
            if fake:
                import json as _json

                from datafarm.expert.dryrun import build_world, default_plan, make_dry_run_adapter

                world = world or build_world(out / "fake-world")
                plan_path = world["root"] / "plan" / f"{cell['cell_id']}.json"
                plan_path.parent.mkdir(exist_ok=True)
                plan_path.write_text(_json.dumps(default_plan(world, Path(cell["round_dir"]), adversarial=False)))
                adapter = make_dry_run_adapter(adapter, world, isolation=isolation or "none",
                                               model=cell["model"], plan_path=plan_path, name=cell["cell_id"])
                holdout_file = str(world["holdout"])
            paths = RoundPaths(Path(cell["round_dir"]))
            prepare_round(adapter, paths.root, clock=runtime, holdout_seeds_path=holdout_file,
                          round_id=f"XS-{cell['cell_id']}"[:120], overrides=None if fake else overrides)
            runner_kwargs = {"tick_interval_s": 0.1, "poll_interval_s": 0.2, "term_grace_s": 3.0,
                             "drain_timeout_s": 120.0} if fake or fast else {}
            RoundRunner(paths, adapter, clock=runtime, approve_agent_spend=approve_agent_spend,
                        approve_gpu=approve_gpu, isolation_override=isolation if fake else None,
                        **runner_kwargs).run()
            kind = adapter.rollout.get("backend")
            if kind == "command" and not approve_gpu:
                raise HarnessError("holdout evaluation submits GPU jobs; pass --approve-gpu")
            backend = make_backend(adapter, kind=kind, masked=rollout_masks(adapter, paths, purpose="holdout"),
                                   isolation_override=isolation if fake else None)
            evaluate_holdout(paths, adapter, backend, clock=runtime, poll_s=0.2 if fake else 15.0,
                             manifest_entries=manifest_entries(adapter, paths),
                             manifest_extra={"harness_code_sha256": harness_code_sha256()})
            cell["status"] = "done"
        except HarnessError as exc:
            cell["status"] = "failed"
            cell["error"] = str(exc)[:500]
        atomic_write_json(out / "plan.json", plan)
    return report_suite(out, now=clock.now())
