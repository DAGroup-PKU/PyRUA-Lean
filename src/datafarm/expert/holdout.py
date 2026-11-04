"""Harness-only holdout evaluation and the score-vs-cost curve.

Only the harness knows the holdout seeds (``harness/holdout-seeds.json``, masked in every
sandbox) and only the harness runs them, after the round has ended: the final expert and
each submission are executed fresh on every holdout seed with the same job template and
isolation as development rollouts, judged by the same verifier, and summarised per seed
family. Infrastructure failures are retried; an invalid initial state is replaced by a
spare seed of the same family (otherwise invalid starts shrink the holdout and change the
family mix). Results stay in ``harness/``; agents only ever see aggregate
summaries through a later round's handoff bundle.
"""

from __future__ import annotations

import collections
import dataclasses
import re
from typing import Any, Callable, Sequence

from datafarm.expert.adapter import HoldoutSeed, TaskAdapter, load_holdout_seeds
from datafarm.expert.layout import RoundPaths
from datafarm.expert.manifest import ManifestGuard
from datafarm.expert.rollout import AttemptSpec, LaunchError, RolloutBackend, classify_run_dir, prepare_run_dir
from datafarm.expert.state import ACTIVE, EventLog, RoundState
from datafarm.expert.util import Clock, HarnessError, atomic_write_json, sha256_bytes, utc_now

RESULTS_SCHEMA = "datafarm.expert-harness.holdout-results/1"


def holdout_masks(paths: RoundPaths) -> list[str]:
    """Round-private items hidden from holdout rollouts (their run dirs live in harness/)."""

    h = paths.harness
    return [str(p) for p in (paths.holdout_seeds, paths.manifests, paths.sessions, paths.state,
                             paths.private_events, paths.processed, paths.holdout_results,
                             paths.round_result, h / "run.lock")]


def select_targets(submissions: Sequence[dict[str, Any]], mode: str, max_targets: int) -> list[dict[str, Any]]:
    if not submissions:
        return []
    finals = [s for s in submissions if s.get("final")]
    final = finals[-1] if finals else submissions[-1]
    if mode == "final" or max_targets <= 1:
        return [final]
    others = [s for s in submissions if s is not final]
    chosen = others[max(0, len(others) - (max_targets - 1)):] + [final]
    return sorted(chosen, key=lambda s: s["submission_id"])


@dataclasses.dataclass
class _Job:
    key: str
    sha: str
    seed: HoldoutSeed
    tries: int
    spec: AttemptSpec
    status: str = "pending"
    receipt: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None


def evaluate_holdout(paths: RoundPaths, adapter: TaskAdapter, backend: RolloutBackend, *,
                     clock: Clock, targets: str | None = None, max_concurrent: int | None = None,
                     retry_infra: int | None = None, poll_s: float = 15.0,
                     timeout_s: float | None = None, manifest_entries: Sequence[str] = (),
                     manifest_extra: dict[str, str] | None = None) -> dict[str, Any]:
    state = RoundState.load(paths.state)
    events = EventLog(paths, state.data["round_id"], clock.now)
    if state.status in ACTIVE:
        raise HarnessError(f"round is {state.status}; holdout evaluation runs only after the round ends")
    if state.status == "invalid_tamper" or not state.data["manifest"].get("ok", True):
        raise HarnessError("round is invalid (protected files changed); holdout evaluation refused")
    guard = ManifestGuard(paths, manifest_entries, state, events, clock.now, extra=manifest_extra)
    ok, changes = guard.check("pre-holdout")
    if not ok:
        state.status = "invalid_tamper"
        state.save()
        raise HarnessError(f"protected files changed before holdout evaluation ({len(changes)} entries)")
    cfg = adapter.holdout
    seeds = load_holdout_seeds(paths.holdout_seeds)
    primary = [s for s in seeds if not s.spare]
    spares: dict[str | None, list[HoldoutSeed]] = collections.defaultdict(list)
    for s in seeds:
        if s.spare:
            spares[s.family].append(s)
    mode = targets or cfg.get("targets", "all")
    chosen = select_targets(state.data["submissions"], mode, int(cfg.get("max_targets", 6)))
    if not chosen:
        raise HarnessError("no submission to evaluate")
    retry_infra = int(cfg.get("retry_infra", 2) if retry_infra is None else retry_infra)
    max_concurrent = int(max_concurrent or cfg.get("max_concurrent", 8))
    by_sha: dict[str, dict[str, Any]] = {}
    for sub in chosen:
        data = (paths.submissions / sub["submission_id"] / "expert.py").read_bytes()
        if sha256_bytes(data) != sub["expert_sha256"]:
            raise HarnessError(f"submission {sub['submission_id']} snapshot does not match its digest")
        by_sha.setdefault(sub["expert_sha256"], {"class": sub["expert_class"], "data": data,
                                                 "submissions": []})["submissions"].append(sub["submission_id"])
    events.emit("holdout_started", private=True, targets=[s["submission_id"] for s in chosen],
                seeds=len(primary))
    jobs: list[_Job] = []
    counter = 0

    def new_job(sha: str, seed: HoldoutSeed, tries: int) -> _Job:
        nonlocal counter
        counter += 1
        attempt_id = f"H-{sha[:8]}-{seed.seed}-t{tries}"
        run_dir = paths.holdout_dir / "runs" / sha[:12] / f"seed{seed.seed}-t{tries}"
        short = re.sub(r"[^A-Za-z0-9]", "", state.data["round_id"])[-8:]
        spec = AttemptSpec(attempt_id, "holdout", seed.seed, by_sha[sha]["class"], run_dir,
                           f"xh-{short}-h{counter}")
        prepare_run_dir(spec, by_sha[sha]["data"], adapter)
        return _Job(f"{sha}:{seed.seed}", sha, seed, tries, spec)

    for sha in by_sha:
        for seed in primary:
            jobs.append(new_job(sha, seed, 0))
    spare_cursor: dict[tuple[str, str | None], int] = collections.defaultdict(int)
    timeout_s = float(timeout_s if timeout_s is not None else cfg.get("timeout_s", 12 * 3600))
    deadline = clock.now() + timeout_s
    while True:
        running = [j for j in jobs if j.status == "running"]
        for job in [j for j in jobs if j.status == "pending"][: max(0, max_concurrent - len(running))]:
            try:
                job.receipt = dict(backend.launch(job.spec))
                job.status = "running"
            except LaunchError as exc:
                job.status = "pending"
                job.receipt = {"last_error": str(exc)[:200]}
                break
        for job in [j for j in jobs if j.status == "running"]:
            status = backend.poll(job.spec, job.receipt or {}, clock.now())
            if status == "running":
                continue
            outcome = (backend.outcome(job.spec, job.receipt or {}, status) if status == "finished"
                       else classify_run_dir(job.spec.run_dir, {}, status=status))
            job.outcome = outcome.to_dict()
            job.status = "done"
            if outcome.counted:
                continue
            if outcome.outcome_class == "init_invalid":
                key = (job.sha, job.seed.family)
                pool = spares.get(job.seed.family, [])
                if spare_cursor[key] < len(pool):
                    spare = pool[spare_cursor[key]]
                    spare_cursor[key] += 1
                    job.status = "replaced"
                    jobs.append(new_job(job.sha, spare, 0))
                continue
            if job.tries < retry_infra:
                job.status = "retried"
                jobs.append(new_job(job.sha, job.seed, job.tries + 1))
        if not [j for j in jobs if j.status in ("pending", "running")]:
            break
        if clock.now() > deadline:
            for job in jobs:
                if job.status in ("pending", "running"):
                    job.status = "abandoned"
            break
        clock.sleep(poll_s)

    per_sha: dict[str, dict[str, Any]] = {}
    for sha in by_sha:
        final_jobs = [j for j in jobs if j.sha == sha and j.status == "done"]
        scored = [j for j in final_jobs if j.outcome and j.outcome["counted"]]
        families: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        failure_labels: collections.Counter[str] = collections.Counter()
        for job in scored:
            family = job.outcome.get("family") or job.seed.family or "unknown"
            families[family][1] += 1
            if job.outcome.get("success"):
                families[family][0] += 1
            else:
                for label in _failure_labels(job.outcome, adapter):
                    failure_labels[label] += 1
        passed = sum(1 for j in scored if j.outcome.get("success"))
        per_sha[sha] = {
            "submissions": by_sha[sha]["submissions"], "expert_class": by_sha[sha]["class"],
            "scored": len(scored), "passed": passed,
            "pass_rate": None if not scored else round(passed / len(scored), 4),
            "per_family": {k: {"passed": v[0], "scored": v[1]} for k, v in sorted(families.items())},
            "failure_labels": dict(failure_labels.most_common()),
            "unscored": [{"seed": j.seed.seed, "family": j.seed.family,
                          "outcome_class": (j.outcome or {}).get("outcome_class", j.status)}
                         for j in jobs if j.sha == sha and j.status in ("done", "abandoned")
                         and not (j.outcome and j.outcome["counted"])],
            "outcomes": [{"seed": j.seed.seed, "family": j.seed.family, **(j.outcome or {})}
                         for j in final_jobs],
        }
    curve = score_cost_curve(state, per_sha)
    finals = [s for s in state.data["submissions"] if s.get("final")]
    final_sub = finals[-1] if finals else state.data["submissions"][-1]
    result = {
        "schema": RESULTS_SCHEMA, "round_id": state.data["round_id"], "task_id": adapter.task_id,
        "verifier_version": adapter.verifier_version, "evaluated_at": utc_now(clock.now()),
        "seeds": len(primary), "targets": [s["submission_id"] for s in chosen],
        "final_submission": final_sub["submission_id"], "final_expert_sha256": final_sub["expert_sha256"],
        "final": per_sha.get(final_sub["expert_sha256"]), "by_expert": per_sha, "curve": curve,
        "rollouts": len(jobs), "success_metric": adapter.verifier.get("success_metric"),
    }
    atomic_write_json(paths.holdout_results, result, mode=0o600)
    events.emit("holdout_finished", private=True, final_pass_rate=(result["final"] or {}).get("pass_rate"),
                rollouts=len(jobs))
    events.emit("holdout_evaluated", targets=len(chosen))  # public: no scores
    return result


def _failure_labels(outcome: dict[str, Any], adapter: TaskAdapter) -> list[str]:
    fields = outcome.get("fields") or {}
    labels: list[str] = []
    for name in adapter.rollout.get("verdict", {}).get("failure_fields", []):
        value = fields.get(name)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, list):
                    labels.extend(f"{name}:{x}" for x in item)
                elif item:
                    labels.append(f"{name}:{item}")
        elif value:
            labels.append(f"{name}:{value}")
    return labels or ["failed"]


def score_cost_curve(state: RoundState, per_sha: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Submissions in order with the cost/time spent up to them and their holdout score.

    Session cost is known when a session ends, so ``cost_usd_through_session`` is the cost
    of every session up to and including the one that made the submission (an upper
    bound); ``cost_usd_before_session`` is the lower bound."""

    sessions = state.data["sessions"]
    curve = []
    for sub in state.data["submissions"]:
        index = int(sub.get("session_index") or 0)
        before = sum(float(s.get("cost_usd") or 0.0) for s in sessions if int(s["index"]) < index)
        through = sum(float(s.get("cost_usd") or 0.0) for s in sessions if int(s["index"]) <= index)
        score = per_sha.get(sub["expert_sha256"])
        curve.append({
            "submission_id": sub["submission_id"], "final": sub.get("final", False),
            "source": sub.get("source"), "expert_sha256": sub["expert_sha256"],
            "active_wall_s": sub.get("active_wall_s"), "counted_rollouts": sub.get("counted_rollouts"),
            "cost_usd_before_session": round(before, 4), "cost_usd_through_session": round(through, 4),
            "holdout_passed": None if score is None else score["passed"],
            "holdout_scored": None if score is None else score["scored"],
            "holdout_pass_rate": None if score is None else score["pass_rate"],
        })
    return curve


def draw_holdout_seeds(adapter: TaskAdapter, *, per_family: int, spares: int = 1, range_start: int = 100_000,
                       range_size: int = 1_000_000, count: int | None = None, batch: int = 200,
                       run: Callable[..., Any] | None = None, rng: Any = None) -> list[HoldoutSeed]:
    """Draw fresh holdout seeds from a secret offset in a large range, balanced across the
    generator's families (the family of a seed is computed by the adapter's
    ``seeds.family_command``, which prints ``<seed> <family>`` lines). The agent can compute
    families of any seed it knows, so what must stay secret is *which* seeds are used."""

    import secrets
    import subprocess

    run = run or subprocess.run
    rng = rng or secrets.SystemRandom()
    seeds_cfg = adapter.section("seeds")
    dev = adapter.dev_seeds
    start = int(range_start) + rng.randrange(int(range_size))
    command = seeds_cfg.get("family_command")
    families = list(seeds_cfg.get("families", []))
    if not command or not families:
        total = int(count or per_family)
        chosen: list[int] = []
        candidate = start
        while len(chosen) < total:
            if candidate not in dev:
                chosen.append(candidate)
            candidate += 1 + rng.randrange(7)
        return [HoldoutSeed(s) for s in chosen]
    need = per_family + spares
    picked: dict[str, list[int]] = {f: [] for f in families}
    cursor = start
    while any(len(v) < need for v in picked.values()):
        if cursor > start + 200 * batch:
            raise HarnessError("could not find enough seeds per family; check seeds.family_command")
        chunk = [s for s in range(cursor, cursor + batch) if s not in dev]
        cursor += batch
        argv: list[str] = []
        for item in command:
            if item == "{seeds}":
                argv.extend(str(s) for s in chunk)
            else:
                argv.append(adapter.x(item))
        proc = run(argv, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise HarnessError(f"family command failed: {(proc.stderr or '')[-300:]}")
        for line in proc.stdout.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0].isdigit() and fields[1] in picked:
                if len(picked[fields[1]]) < need:
                    picked[fields[1]].append(int(fields[0]))
    return [HoldoutSeed(seed, family, spare=index >= per_family)
            for family in families for index, seed in enumerate(picked[family])]
