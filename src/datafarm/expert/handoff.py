"""Handoff bundle: everything round N+1 needs from round N, assembled by the harness.

Instead of an operator copying the previous expert, summarising the held-out evaluation and
appending known problems to the next brief, the bundle does this from harness records: the frozen expert (by digest), the agent's REPORT/NOTES
(read without following links), the harness attempt ledger, a failure taxonomy, an
agent-safe holdout summary (per-family pass counts and failure labels, never seeds or
per-seed states), costs from the agent-call ledger, and the known brief defects (operator
list plus defects detected from the round's own events). It then renders and lints the
next brief.
"""

from __future__ import annotations

import collections
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datafarm.expert.adapter import TaskAdapter, load_holdout_seeds
from datafarm.expert.brief import lint_brief, render_brief
from datafarm.expert.layout import RoundPaths
from datafarm.expert.state import ACTIVE, RoundState
from datafarm.expert.util import (Clock, HarnessError, UnsafePathError, atomic_write_bytes, atomic_write_json,
                   atomic_write_text, read_json, read_jsonl, read_regular_nofollow, sha256_bytes,
                   sha256_file, utc_now)

BUNDLE_SCHEMA = "datafarm.expert-harness.handoff/1"
DEFAULT_COPY = ("REPORT.md", "NOTES.md")


def _calls_summary(paths: RoundPaths) -> dict[str, Any]:
    try:
        from datafarm.agent.call_log import read_calls, summarize
    except ImportError:  # pragma: no cover - the shared ledger module ships with the repo
        return {}
    return summarize(read_calls(paths.agent_calls))


def failure_taxonomy(state: RoundState, adapter: TaskAdapter) -> dict[str, Any]:
    classes: collections.Counter[str] = collections.Counter()
    labels: collections.Counter[str] = collections.Counter()
    families: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    details: collections.Counter[str] = collections.Counter()
    failure_fields = adapter.rollout.get("verdict", {}).get("failure_fields", [])
    gpu_wall = 0.0
    for attempt in state.data["attempts"].values():
        outcome = attempt.get("outcome") or {}
        cls = outcome.get("outcome_class")
        if not cls:
            continue
        classes[cls] += 1
        gpu_wall += float(outcome.get("wall_s") or 0.0)
        if outcome.get("counted"):
            family = outcome.get("family") or "unknown"
            families[family][1] += 1
            if outcome.get("success"):
                families[family][0] += 1
            else:
                fields = outcome.get("fields") or {}
                for name in failure_fields:
                    value = fields.get(name)
                    items = value if isinstance(value, list) else [value]
                    for item in items:
                        for label in (item if isinstance(item, list) else [item]):
                            if label:
                                labels[f"{name}:{label}"] += 1
        if cls != "judged":
            details[f"{cls}: {str(outcome.get('detail'))[:120]}"] += 1
    return {"outcome_classes": dict(classes.most_common()),
            "families": {k: {"successes": v[0], "counted": v[1]} for k, v in sorted(families.items())},
            "failure_labels": dict(labels.most_common(25)),
            "non_judged_details": dict(details.most_common(15)),
            "rollout_wall_s": round(gpu_wall, 1)}


def detected_defects(paths: RoundPaths, state: RoundState) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    events = read_jsonl(paths.events)
    rejected = collections.Counter(e.get("code") for e in events if e.get("event") == "request_rejected")
    if rejected:
        out.append({"id": "rejected_requests", "detail": dict(rejected)})
    sessions = state.data["sessions"]
    capped = [s["index"] for s in sessions if s.get("stop_reason") == "wall_cap"]
    if capped:
        out.append({"id": "sessions_hit_wall_cap", "detail": capped})
    outages = [s["index"] for s in sessions if s.get("api_outage")]
    if outages:
        out.append({"id": "api_outage_sessions", "detail": outages})
    unknown = [s["index"] for s in sessions if s.get("cost_usd") is None]
    if unknown:
        out.append({"id": "sessions_with_unknown_cost", "detail": unknown})
    infra = collections.Counter((a.get("outcome") or {}).get("outcome_class")
                                for a in state.data["attempts"].values()
                                if (a.get("outcome") or {}).get("outcome_class") not in (None, "judged", "expert_error"))
    if infra:
        out.append({"id": "infrastructure_failures", "detail": dict(infra)})
    capacity = [a["attempt_id"] for a in state.data["attempts"].values()
                if "capacity" in str((a.get("outcome") or {}).get("detail", ""))]
    if capacity:
        out.append({"id": "trace_capacity_hits", "detail": capacity})
    finals = [s for s in state.data["submissions"] if s.get("final")]
    if finals and not finals[-1].get("dev_rollouts_same_expert"):
        out.append({"id": "final_expert_untested", "detail": finals[-1]["submission_id"]})
    if state.data.get("amendments"):
        out.append({"id": "mid_round_amendments",
                    "detail": [{"number": a["number"], "reason": a["reason"]} for a in state.data["amendments"]]})
    if state.data.get("violations"):
        kinds = collections.Counter(v["kind"] for v in state.data["violations"])
        out.append({"id": "violations", "detail": dict(kinds)})
    return out


def agent_safe_holdout_summary(results: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Aggregates only: per-family pass counts and failure labels; no seeds, no states."""

    if not results:
        return None
    final = results.get("final") or {}
    return {
        "verifier_version": results.get("verifier_version"),
        "final": {"passed": final.get("passed"), "scored": final.get("scored"),
                  "per_family": final.get("per_family"), "failure_labels": final.get("failure_labels")},
        "submissions": [{"submission_id": p["submission_id"], "final": p["final"],
                         "holdout_passed": p["holdout_passed"], "holdout_scored": p["holdout_scored"],
                         "active_wall_s": p["active_wall_s"],
                         "cost_usd_through_session": p["cost_usd_through_session"]}
                        for p in results.get("curve", [])],
    }


def assert_no_seed_leak(directory: Path, seeds: Iterable[int]) -> None:
    values = sorted(set(int(s) for s in seeds))
    if not values:
        return
    pattern = re.compile(r"(?<![\w.])(" + "|".join(str(v) for v in values) + r")(?![\w.])")
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in (".md", ".json", ".jsonl", ".txt", ".py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            match = pattern.search(text)
            if match:
                raise HarnessError(f"handoff file {path.name} contains holdout seed value {match.group(1)}; "
                                   "refusing to hand it to the next agent")


def handoff_markdown(bundle: Mapping[str, Any], *, handoff_dir: str = "handoff-in") -> str:
    lines = [f"Previous round: `{bundle.get('round_id')}` ({bundle.get('provenance', 'harness-recorded')}), "
             f"verifier {bundle.get('verifier_version')}, status `{bundle.get('status')}`."]
    frozen = bundle.get("frozen_expert")
    if frozen:
        lines.append(f"- **Frozen expert:** `{handoff_dir}/expert.py`, class `{frozen['class']}`, sha256 "
                     f"`{frozen['sha256'][:16]}…`. Your workspace starts with an exact copy as `expert.py`.")
    holdout = bundle.get("holdout_summary")
    if holdout and holdout.get("final"):
        final = holdout["final"]
        fam = ", ".join(f"{k} {v['passed']}/{v['scored']}" for k, v in (final.get("per_family") or {}).items())
        who = "the harness"
        lines.append(f"- **Holdout (hidden seeds, run by {who}):** the final expert passed "
                     f"{final.get('passed')}/{final.get('scored')} ({fam}).")
        if final.get("failure_labels"):
            top = ", ".join(f"{k} ×{v}" for k, v in list(final["failure_labels"].items())[:8])
            lines.append(f"  Failure labels on holdout: {top}.")
        subs = holdout.get("submissions") or []
        if len(subs) > 1:
            curve = "; ".join(f"{s['submission_id']}: {s['holdout_passed']}/{s['holdout_scored']}"
                              for s in subs if s.get("holdout_scored") is not None)
            lines.append(f"  Submissions over the round: {curve}.")
    elif bundle.get("holdout_note"):
        lines.append(f"- **Holdout:** {bundle['holdout_note']}")
    if bundle.get("narrative"):
        lines.append(f"- **Operator summary:** {bundle['narrative']}")
    attempts = bundle.get("attempts_summary") or {}
    if attempts:
        lines.append(f"- **Development:** counted rollouts {attempts.get('counted', 0)}, successes "
                     f"{attempts.get('successes', 0)}, infrastructure failures (not counted) "
                     f"{attempts.get('infra', 0)}.")
    taxonomy = bundle.get("failure_taxonomy") or {}
    if taxonomy.get("failure_labels"):
        top = ", ".join(f"{k} ×{v}" for k, v in list(taxonomy["failure_labels"].items())[:8])
        lines.append(f"- **Most frequent development failure labels:** {top}.")
    costs = bundle.get("costs") or {}
    if costs.get("calls"):
        lines.append(f"- **Cost of the previous round:** ${costs.get('cost_usd_known', 0):.2f} over "
                     f"{costs.get('calls')} sessions ({costs.get('calls_with_unknown_cost', 0)} with unknown cost).")
    # bundle.json keeps every recorded defect; the brief shows only problems observed in
    # (or reported for) the previous round itself.
    defects = list(bundle.get("open_issues") or []) + list(bundle.get("detected_defects") or [])
    if defects:
        lines.append("- **Problems observed in the previous round:**")
        for item in defects[:12]:
            text = item.get("text") or json.dumps(item.get("detail"), ensure_ascii=False)[:200]
            lines.append(f"  - `{item.get('id')}`: {text}")
    files = [f for f in bundle.get("files", []) if f != "bundle.json"]
    if files:
        lines.append(f"- Files in `{handoff_dir}/`: " + ", ".join(f"`{f}`" for f in files) + ".")
    return "\n".join(lines)


def build_handoff(paths: RoundPaths, adapter: TaskAdapter, out_dir: str | Path, *, clock: Clock,
                  copy_files: Sequence[str] | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    state = RoundState.load(paths.state)
    if state.status in ACTIVE:
        raise HarnessError(f"round is {state.status}; build the handoff after it ends")
    if out.exists() and any(out.iterdir()):
        raise HarnessError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    subs = state.data["submissions"]
    finals = [s for s in subs if s.get("final")]
    frozen_sub = finals[-1] if finals else (subs[-1] if subs else None)
    frozen = None
    if frozen_sub:
        data = (paths.submissions / frozen_sub["submission_id"] / "expert.py").read_bytes()
        if sha256_bytes(data) != frozen_sub["expert_sha256"]:
            raise HarnessError("frozen submission snapshot does not match its digest")
        atomic_write_bytes(out / "expert.py", data, mode=0o444)
        files.append("expert.py")
        frozen = {"submission_id": frozen_sub["submission_id"], "sha256": frozen_sub["expert_sha256"],
                  "class": frozen_sub["expert_class"], "file": "expert.py"}
    skipped = []
    for name in (copy_files or adapter.section("handoff").get("copy_workspace_files", DEFAULT_COPY)):
        try:
            data = read_regular_nofollow(paths.workspace, name, max_bytes=2_000_000)
        except (UnsafePathError, OSError) as exc:
            skipped.append({"file": name, "reason": str(exc)[:200]})
            continue
        atomic_write_bytes(out / Path(name).name, data, mode=0o444)
        files.append(Path(name).name)
    if paths.attempts.exists():
        shutil.copyfile(paths.attempts, out / "attempts.jsonl")
        files.append("attempts.jsonl")
    holdout = read_json(paths.holdout_results) if paths.holdout_results.exists() else None
    counters = state.counters
    bundle = {
        "schema": BUNDLE_SCHEMA, "provenance": "harness-recorded", "round_id": state.data["round_id"],
        "task_id": adapter.task_id, "verifier_version": adapter.verifier_version,
        "created_at": utc_now(clock.now()), "status": state.status, "stop": state.data.get("stop"),
        "flag": state.data.get("flag"), "frozen_expert": frozen,
        "submissions": [{k: s.get(k) for k in ("submission_id", "final", "source", "expert_sha256",
                                                "expert_class", "active_wall_s", "counted_rollouts",
                                                "dev_rollouts_same_expert", "note")} for s in subs],
        "attempts_summary": {"counted": counters.get("counted", 0), "successes": counters.get("successes", 0),
                             "infra": counters.get("infra", 0), "rejected_requests": counters.get("rejected", 0)},
        "failure_taxonomy": failure_taxonomy(state, adapter),
        "holdout_summary": agent_safe_holdout_summary(holdout),
        "holdout_note": None if holdout else "not evaluated yet",
        "costs": _calls_summary(paths),
        "known_brief_defects": adapter.raw.get("known_brief_defects", []),
        "open_issues": adapter.raw.get("open_issues", []),
        "detected_defects": detected_defects(paths, state),
        "skipped_files": skipped, "files": sorted(files + ["bundle.json"]),
    }
    atomic_write_json(out / "bundle.json", bundle, mode=0o444)
    if paths.holdout_seeds.exists():
        assert_no_seed_leak(out, [s.seed for s in load_holdout_seeds(paths.holdout_seeds)])
    write_next_brief(out, adapter, bundle)
    return bundle


def write_next_brief(out: Path, adapter: TaskAdapter, bundle: Mapping[str, Any]) -> dict[str, Any]:
    text = render_brief(adapter, handoff_md=handoff_markdown(bundle))
    report = lint_brief(text, adapter)
    atomic_write_text(out / "next-brief.preview.md", text)
    atomic_write_json(out / "next-brief.lint.json", report.to_dict())
    return report.to_dict()


def load_bundle(directory: str | Path) -> dict[str, Any]:
    bundle = read_json(Path(directory) / "bundle.json")
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise HarnessError(f"{directory} is not a {BUNDLE_SCHEMA} bundle")
    expert = Path(directory) / "expert.py"
    if bundle.get("frozen_expert") and sha256_file(expert) != bundle["frozen_expert"]["sha256"]:
        raise HarnessError("handoff expert.py does not match the bundle digest")
    return bundle
