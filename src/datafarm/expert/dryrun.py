"""End-to-end dry run with a scripted agent and a fake simulator (no GPU, no API).

It exercises the real code paths of a round -- the template brief and lint,
the agent sandbox, the ``xh`` broker, job scripts with the rollout sandbox, the judge
outside the sandbox, tamper checks, holdout evaluation, handoff, the library and the
agent-call ledger -- in a throw-away directory, then checks the evidence:

* the agent could not read masked paths or write outside its workspace (OS level);
* agent code in the rollout could not read the judge, the credential or the holdout seeds;
* holdout-seed and symlink requests were rejected and recorded as violations;
* every attempt has a launch and an end record; counted/infra classification is right;
* sessions were resumed, and each has one ``datafarm.agent-call/1`` record with its cost;
* tamper checks passed and a modified judge is detected;
* holdout scored both submissions (the curve), nothing about it reached agent-visible files;
* the handoff bundle renders a brief that lints, and the next round starts from it.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from datafarm.agent.call_log import read_calls, summarize

from datafarm.expert.adapter import HoldoutSeed, TaskAdapter, write_holdout_seeds
from datafarm.expert.brief import lint_file
from datafarm.expert.handoff import build_handoff
from datafarm.expert.holdout import evaluate_holdout
from datafarm.expert.isolation import probe_available
from datafarm.expert.layout import RoundPaths
from datafarm.expert.library import ExpertLibrary, add_from_round
from datafarm.expert.manifest import build_manifest, diff_manifests
from datafarm.expert.rollout import make_backend
from datafarm.expert.round import (RoundRunner, harness_code_sha256, manifest_entries, prepare_round, rollout_masks)
from datafarm.expert.state import RoundState
from datafarm.common import read_json_or_none
from datafarm.expert.util import Clock, read_json, read_jsonl, sha256_file

HOLDOUT_SEEDS = tuple(range(5000, 5008))
SPARE_SEEDS = (5100, 5101, 5102, 5103)
LEAKED_SEED = 5003  # the scripted agent tries it on purpose; it must be rejected

EXPERT_TEMPLATE = '''"""Fake expert for the harness dry run (no simulator)."""
FAKE_QUALITY = {quality}
FAKE_STEPS = 3


class FakeExpert:
    def __init__(self, pose_bank, home=None):
        self.done = False
        self.events = []
        self.phase = "start"
        self.fold_index = 0
        self._n = 0

    def next_plan(self, observation):
        self._n += 1
        if self._n > FAKE_STEPS:
            self.done = True
            return None
        return {{"primitive_id": "p%d" % self._n}}

    def complete(self, plan, observation, execution):
        self.events.append(plan["primitive_id"])
'''


def repo_root() -> Path:
    from datafarm.runtime import source_root
    return Path(os.environ["DATAFARM_WORKSPACE"]) if "DATAFARM_WORKSPACE" in os.environ else source_root()


def template_adapter_path() -> Path:
    return repo_root() / "configs" / "expert-harness" / "template.json"


def pick_isolation(requested: str) -> str:
    if requested != "auto":
        return requested
    if probe_available("bwrap")[0]:
        return "bwrap"
    if os.geteuid() == 0 and probe_available("unshare")[0]:
        return "unshare"
    return "none"


def build_world(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o755)
    fakes = Path(__file__).resolve().parent / "fakes"
    world = {"root": root, "judge": root / "judge", "sim": root / "sim", "secrets": root / "secrets",
             "private": root / "private", "docs": root / "docs", "fakes": fakes}
    for key in ("judge", "sim", "secrets", "private", "docs"):
        world[key].mkdir(exist_ok=True)
        os.chmod(world[key], 0o755)
    shutil.copyfile(fakes / "fake_judge.py", world["judge"] / "fake_judge.py")
    shutil.copyfile(fakes / "fake_sim.py", world["sim"] / "fake_sim.py")
    credential = world["secrets"] / "credential.txt"
    credential.write_text("FAKE-CREDENTIAL-DO-NOT-READ\n")
    os.chmod(credential, 0o600)
    (world["docs"] / "interface.md").write_text("Fake expert interface: Class(pose_bank, home=home) with "
                                                "next_plan(observation) and complete(plan, observation, execution).\n")
    world["holdout"] = world["private"] / "holdout.json"
    seeds = [HoldoutSeed(s, ("family_a", "family_b", "family_c", "family_d")[s % 4])
             for s in HOLDOUT_SEEDS]
    seeds += [HoldoutSeed(s, ("family_a", "family_b", "family_c", "family_d")[s % 4], True)
              for s in SPARE_SEEDS]
    write_holdout_seeds(world["holdout"], seeds, note="dry-run holdout seeds")
    return world


def make_dry_run_adapter(base: TaskAdapter, world: dict[str, Path], *, isolation: str, model: str = "fake-model",
                         plan_path: Path | None = None, name: str | None = None) -> TaskAdapter:
    """The template adapter (brief template, verifier text, runtime facts, rules) with a
    fake agent, a fake simulator/judge and a throw-away world."""

    raw = copy.deepcopy(base.raw)
    root = world["root"]
    agent_python = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
    run_as = {"uid": 65534, "gid": 65534} if os.geteuid() == 0 and isolation != "none" else None
    plan_path = plan_path or (root / "plan" / "plan.json")
    raw["title"] = base.title + " [dry run: scripted agent, fake simulator]"
    raw["brief"]["template"] = str(base.brief_template_path)
    raw["agent"] = {"backend": "fake", "model": model, "plan": str(plan_path), "python": agent_python,
                    "auth": {"env_passthrough": []}}
    if base.agent.get("price_table") and (base.dir / base.agent["price_table"]).is_file():
        raw["agent"]["price_table"] = str((base.dir / base.agent["price_table"]).resolve())
    raw["workspace"] = {"readable": [str(world["docs"]), str(world["sim"])],
                        "protected": [str(world["judge"]), str(world["secrets"]), str(world["private"])]}
    raw["verifier"] = {**base.verifier, "files": [str(world["judge"] / "fake_judge.py")]}
    raw["manifest"] = {"files": [str(world["sim"] / "fake_sim.py")]}
    raw["expert"] = {**base.expert_policy, "default_class": "FakeExpert", "default_file": "expert.py",
                     "allowed_imports": ["__future__", "math"]}
    raw["seeds"] = {"dev": {"ranges": [[0, 999]]}}
    raw["holdout"] = {"seeds_file": str(world["holdout"]), "targets": "all", "max_targets": 4,
                      "max_concurrent": 4, "retry_infra": 1, "verified_pass_rate": 0.5, "timeout_s": 900}
    raw["budgets"] = {"wall_per_session_s": 180, "total_wall_s": 1800, "max_counted_rollouts": 10,
                      "max_concurrent_rollouts": 3, "max_sessions": 4, "max_usd": 5.0, "max_infra_failures": 5,
                      "max_idle_sessions": 2, "final_grace_s": 60, "wait_cap_s": 90, "warn_before_wall_s": 60}
    probes = []
    for path in (world["judge"] / "fake_judge.py", world["secrets"] / "credential.txt", world["holdout"]):
        probes += ["--probe", str(path)]
    raw["rollout"] = {
        "backend": "local", "poll_interval_s": 0.2, "heartbeat_interval_s": 1, "stale_after_s": 120,
        "max_wall_s": 300,
        "sim": {"argv": [agent_python, str(world["sim"] / "fake_sim.py"), "--expert",
                         "{run_dir}/expert_snapshot.py:{expert_class}", "--seed", "{seed}", "--output-dir",
                         "{run_dir}/native", *probes],
                "cwd": "{run_dir}/native_workdir", "log": "native.log", "timeout_s": 120,
                "requires_output": "native/trace.json", "writable_run_subdirs": ["native", "native_workdir"]},
        "judge": [{"name": "fake", "argv": [sys.executable, str(world["judge"] / "fake_judge.py"), "{run_dir}"],
                   "stdout": "verdict.stdout", "outputs": ["verdict.json"]}],
        "verdict": {"name": "fake judge", "file": "verdict.json", "success": "m2",
                    "fields": {"m1": "m1", "m2": "m2", "fold_max_xy_mm": "fold_max_xy_mm"}, "failure_fields": []},
        "init_state": {"file": "native/init.json", "valid": "valid", "family": "family"},
        "classify": {"infra_regex": [], "expert_error_regex": ["expert_snapshot\\.py", "trace exceeded"]},
        "record_digests": [str(world["sim"] / "fake_sim.py")],
    }
    raw["isolation"] = {
        "agent": {"mode": isolation, "fallback": None, "base": "allowlist", "run_as": run_as, "share_net": False,
                  "allow_degraded_fallback": True, "extra_ro": [str(world["fakes"]), str(plan_path.parent)]},
        "rollout": {"mode": isolation, "fallback": None, "base": "root_ro", "dev": "minimal", "private_tmp": True,
                    "run_as": None, "writable": [], "extra_masked": []},
    }
    path = root / ("adapter-" + (name or model).replace("/", "_") + ".json")
    path.write_text(json.dumps(raw, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return TaskAdapter.load(path)


def default_plan(world: dict[str, Path], round_dir: Path, *, adversarial: bool = True) -> dict[str, Any]:
    """Two scripted sessions. With ``adversarial`` the agent also probes masked paths,
    writes outside its workspace, asks for a holdout seed and submits a symlinked and an
    escaping expert path -- all of which the harness must refuse and record."""

    harness = round_dir / "harness"
    probe_paths = [str(world["judge"] / "fake_judge.py"), str(world["secrets"] / "credential.txt"),
                   str(world["holdout"]), str(harness / "holdout-seeds.json"), str(harness / "state.json")]
    plan = {"cost_per_session": 0.37, "sessions": [
        [
            {"op": "probe", "paths": probe_paths, "out": "isolation-probe.json"},
            {"op": "write_outside", "paths": [str(round_dir / "rollouts" / "agent-was-here.txt"),
                                              str(round_dir / "ledger" / "agent-was-here.txt"),
                                              str(world["docs"] / "agent-was-here.txt")],
             "out": "write-probe.json"},
            {"op": "write", "path": "expert.py", "text": EXPERT_TEMPLATE.format(quality=0.05)},
            {"op": "xh", "args": ["rollout", "--seed", "3", "--note", "v1 baseline"]},
            {"op": "xh", "args": ["rollout", "--seed", "4", "--note", "v1 baseline"]},
            {"op": "xh", "args": ["rollout", "--seed", str(LEAKED_SEED), "--note", "holdout probe"], "expect": [2]},
            {"op": "symlink", "path": "evil.py", "target": str(world["holdout"])},
            {"op": "xh", "args": ["rollout", "--seed", "3", "--file", "evil.py"], "expect": [2]},
            {"op": "xh", "args": ["rollout", "--seed", "3", "--file", "../harness/holdout-seeds.json"], "expect": [2]},
            {"op": "xh", "args": ["wait", "A-0001", "A-0002", "--timeout", "90"]},
            {"op": "xh", "args": ["submit", "v1 baseline"]},
        ],
        [
            {"op": "write", "path": "expert.py", "text": EXPERT_TEMPLATE.format(quality=0.95)},
            {"op": "xh", "args": ["rollout", "--seed", "7", "--note", "v2"]},
            {"op": "xh", "args": ["rollout", "--seed", "13", "--note", "invalid start"]},
            {"op": "xh", "args": ["wait", "A-0003", "A-0004", "--timeout", "90"]},
            {"op": "write", "path": "REPORT.md", "text": "# Dry-run report\n\nv2 raises FAKE_QUALITY.\n"},
            {"op": "write", "path": "NOTES.md", "text": "notes\n"},
            {"op": "xh", "args": ["submit", "--final", "v2 final"]},
        ],
    ]}
    if not adversarial:
        hostile = ("probe", "write_outside", "symlink")
        plan["sessions"][0] = [step for step in plan["sessions"][0]
                               if step["op"] not in hostile and step.get("expect") != [2]]
    return plan


def _scan_for_seeds(directories: list[Path], seeds: set[int]) -> list[str]:
    import re

    pattern = re.compile(r"(?<![\w.])(" + "|".join(str(s) for s in sorted(seeds)) + r")(?![\w.])")
    hits = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.stat().st_size < 5_000_000:
                text = path.read_text(encoding="utf-8", errors="replace")
                match = pattern.search(text)
                if match:
                    hits.append(f"{path}: {match.group(1)}")
    return hits


def run_dry_run(out_dir: str | Path | None = None, *, isolation: str = "auto",
                log: Callable[[str], None] = lambda message: None) -> dict[str, Any]:
    base_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(
        prefix="xh-dryrun-", dir="/var/tmp" if os.access("/var/tmp", os.W_OK) else None))
    base_dir = base_dir.resolve()  # the adapter stores absolute sandbox paths
    base_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(base_dir, 0o755)
    mode = pick_isolation(isolation)
    clock = Clock()
    base = TaskAdapter.load(template_adapter_path())
    world = build_world(base_dir / "world")
    adapter = make_dry_run_adapter(base, world, isolation=mode)
    paths = RoundPaths(base_dir / "round-1")
    plan = default_plan(world, paths.root)
    plan_file = world["root"] / "plan" / "plan.json"
    plan_file.parent.mkdir(exist_ok=True)
    os.chmod(plan_file.parent, 0o755)
    plan_file.write_text(json.dumps(plan, indent=1))
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None, skipped: bool = False) -> None:
        checks.append({"check": name, "ok": bool(ok), "skipped": skipped, "detail": detail})
        log(f"{'SKIP' if skipped else ('ok  ' if ok else 'FAIL')} {name}")

    log(f"dry run in {base_dir} (isolation: {mode})")
    prepared = prepare_round(adapter, paths.root, clock=clock)
    lint = lint_file(paths.brief, adapter, holdout_seeds=list(HOLDOUT_SEEDS))
    check("brief_lint", lint.ok, lint.errors or lint.warnings)
    summary = RoundRunner(paths, adapter, clock=clock, tick_interval_s=0.1, poll_interval_s=0.2,
                          term_grace_s=3.0, drain_timeout_s=120.0).run()
    state = RoundState.load(paths.state).data
    check("round_completed", summary["status"] == "completed", summary.get("end_reason"))

    isolated = mode != "none"
    agent_iso = state["isolation"].get("agent", {})
    check("agent_sandbox_probe", agent_iso.get("ok"), agent_iso, skipped=not isolated)
    probe = read_json_or_none(paths.workspace / "isolation-probe.json") or {}
    check("agent_cannot_read_protected", probe and all(v.startswith("hidden") for v in probe.values()),
          probe, skipped=not isolated)
    writes = read_json_or_none(paths.workspace / "write-probe.json") or {}
    check("agent_cannot_write_outside_workspace", writes and all(v.startswith("refused") for v in writes.values()),
          writes, skipped=not isolated)
    sim_probes = {}
    for run_dir in sorted(paths.rollouts.iterdir()):
        sim_probes[run_dir.name] = read_json_or_none(run_dir / "native" / "probe.json")
    leaks = {k: v for k, v in sim_probes.items() if v and any(x == "readable" for x in v.values())}
    check("rollout_code_cannot_read_protected", bool(sim_probes) and not leaks and all(sim_probes.values()),
          leaks or "all hidden", skipped=not isolated)

    ledger = read_jsonl(paths.attempts)
    launches = {r["attempt_id"] for r in ledger if r["event"] == "launch"}
    ends = {r["attempt_id"] for r in ledger if r["event"] == "end"}
    counters = state["counters"]
    check("attempts_recorded_by_harness", launches == ends == {"A-0001", "A-0002", "A-0003", "A-0004"}
          and counters["counted"] == 3 and counters["infra"] == 1,
          {"launches": sorted(launches), "ends": sorted(ends), "counters": counters})
    kinds = sorted(v["kind"] for v in state["violations"])
    check("forbidden_requests_rejected_and_recorded",
          counters["holdout_seed"] == 1 and counters["unsafe_path"] == 2 and "holdout_seed_request" in kinds,
          {"violations": kinds, "rejected": counters["rejected"]})

    calls = read_calls(paths.agent_calls)
    sessions = state["sessions"]
    check("sessions_resumed_and_ledgered", len(calls) == 2 and len(sessions) == 2 and sessions[1]["resumed"]
          and sessions[1]["session_id"] == sessions[0]["session_id"]
          and all(c.component == "expert-harness/session" and c.cost_source == "provider" for c in calls),
          [{"session": s["index"], "session_id": s.get("session_id"), "cost": s.get("cost_usd"),
            "stop": s.get("stop_reason")} for s in sessions])
    subs = state["submissions"]
    check("submissions", [s["final"] for s in subs] == [False, True] and state["stop"]["reason"] == "done",
          [(s["submission_id"], s["final"]) for s in subs])

    manifest = state["manifest"]
    check("tamper_checks_passed", manifest["ok"] and manifest["checks"] >= 3, manifest)
    entries = manifest_entries(adapter, paths)
    extra = {"harness_code_sha256": harness_code_sha256()}
    baseline = build_manifest(entries, extra=extra)
    judge = world["judge"] / "fake_judge.py"
    original = judge.read_bytes()
    try:
        judge.write_bytes(original + b"\n# tampered\n")
        changed = diff_manifests(baseline, build_manifest(entries, extra=extra))
    finally:
        judge.write_bytes(original)
    check("tamper_drill_detects_modified_judge", any(c["path"] == str(judge) for c in changed), changed[:2])

    backend = make_backend(adapter, kind="local", masked=rollout_masks(adapter, paths, purpose="holdout"),
                           clock_now=clock.now)
    results = evaluate_holdout(paths, adapter, backend, clock=clock, poll_s=0.2,
                               manifest_entries=entries, manifest_extra=extra)
    curve = results["curve"]
    first, last = curve[0], curve[-1]
    check("holdout_scored_each_submission", len(curve) == 2 and first["holdout_scored"] == len(HOLDOUT_SEEDS)
          and last["holdout_scored"] == len(HOLDOUT_SEEDS) and last["holdout_passed"] > first["holdout_passed"],
          [(p["submission_id"], p["holdout_passed"], p["holdout_scored"], p["cost_usd_through_session"])
           for p in curve])
    agent_visible = [paths.workspace, paths.broker, paths.rollouts, paths.submissions, paths.ledger,
                     paths.bin_dir, paths.handoff_in]
    hits = _scan_for_seeds(agent_visible, (set(HOLDOUT_SEEDS) | set(SPARE_SEEDS)) - {LEAKED_SEED})
    check("holdout_invisible_to_agent", not hits and oct(paths.holdout_results.stat().st_mode & 0o777) == "0o600",
          hits[:5])

    handoff_dir = base_dir / "handoff"
    bundle = build_handoff(paths, adapter, handoff_dir, clock=clock)
    next_lint = read_json(handoff_dir / "next-brief.lint.json")
    check("handoff_bundle", bundle["frozen_expert"]["sha256"] == subs[-1]["expert_sha256"] and next_lint["ok"]
          and not _scan_for_seeds([handoff_dir], set(HOLDOUT_SEEDS) | set(SPARE_SEEDS)),
          {"frozen": bundle["frozen_expert"]["submission_id"], "next_brief_errors": next_lint["errors"]})

    library = ExpertLibrary(base_dir / "library")
    entry = add_from_round(library, paths, adapter, now=clock.now())
    ok = entry is not None and library.get(entry["entry_id"])[0]["expert_sha256"] == subs[-1]["expert_sha256"]
    check("library_admits_verified_expert", ok, entry and {"entry": entry["entry_id"], "holdout": entry["holdout"]})

    paths2 = RoundPaths(base_dir / "round-2")
    prepare_round(adapter, paths2.root, clock=clock, handoff_dir=str(handoff_dir), library_dir=str(base_dir / "library"),
                  references=1)
    brief2 = paths2.brief.read_text(encoding="utf-8")
    check("next_round_prepared_from_handoff",
          sha256_file(paths2.workspace / "expert.py") == subs[-1]["expert_sha256"] and "Frozen expert" in brief2
          and any(paths2.references.iterdir()), {"round": str(paths2.root)})

    usage = summarize(calls)
    check("call_ledger_summary", usage["calls"] == 2 and abs(usage["cost_usd_known"] - 0.74) < 1e-6
          and usage["calls_with_unknown_cost"] == 0, usage)
    failed = [c for c in checks if not c["ok"] and not c["skipped"]]
    return {"ok": not failed, "isolation": mode, "dir": str(base_dir), "round": prepared["round_id"],
            "checks": checks, "failed": [c["check"] for c in failed],
            "holdout_final_pass_rate": (results.get("final") or {}).get("pass_rate"),
            "agent_calls": usage}

