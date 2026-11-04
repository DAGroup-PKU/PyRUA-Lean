"""Verified expert library: content-addressed experts that passed holdout evaluation.

Voyager / GenSim-style reuse, with our stricter admission: only an expert whose frozen
snapshot passed the harness's own holdout evaluation at or above the adapter's threshold
is stored, together with the task id, brief and Task-IR digests, verifier version,
holdout scores, model and cost that produced it. Objects are stored by SHA-256 and
verified on every read, so a tampered object is detected rather than reused. New rounds
can retrieve reference experts (``prepare-round --library L --references N``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from datafarm.common import append_jsonl
from datafarm.expert.adapter import TaskAdapter
from datafarm.expert.layout import RoundPaths
from datafarm.expert.state import RoundState
from datafarm.expert.util import (HarnessError, atomic_write_bytes, atomic_write_json, read_json,
                   sha256_bytes, utc_now)

ENTRY_SCHEMA = "datafarm.expert-harness.library-entry/1"


class ExpertLibrary:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.entries = self.root / "entries"
        self.index = self.root / "index.jsonl"

    def _object_path(self, sha: str) -> Path:
        return self.objects / sha[:2] / f"{sha}.py"

    def add(self, *, source: bytes, expert_class: str, task_id: str, verifier_version: str,
            holdout: dict[str, Any], brief_sha256: str | None, task_ir_sha256: str | None,
            model: str | None, backend: str | None, cost_usd: float | None, wall_s: float | None,
            counted_rollouts: int | None, round_id: str, added_at: str,
            tags: list[str] | None = None) -> dict[str, Any]:
        sha = sha256_bytes(source)
        path = self._object_path(sha)
        if not path.exists():
            atomic_write_bytes(path, source, mode=0o444)
        entry_id = re.sub(r"[^A-Za-z0-9._-]", "_", f"{task_id}-{sha[:12]}-{verifier_version}")
        entry = {"schema": ENTRY_SCHEMA, "entry_id": entry_id, "task_id": task_id,
                 "expert_sha256": sha, "expert_class": expert_class, "verifier_version": verifier_version,
                 "holdout": holdout, "brief_sha256": brief_sha256, "task_ir_sha256": task_ir_sha256,
                 "model": model, "backend": backend, "cost_usd": cost_usd, "wall_s": wall_s,
                 "counted_rollouts": counted_rollouts, "round_id": round_id, "added_at": added_at,
                 "tags": sorted(set(tags or []))}
        target = self.entries / f"{entry_id}.json"
        if target.exists():
            existing = read_json(target)
            if existing.get("round_id") == round_id:
                return existing
            entry_id = f"{entry_id}-{round_id}"
            entry["entry_id"] = entry_id
            target = self.entries / f"{entry_id}.json"
        atomic_write_json(target, entry, mode=0o444)
        append_jsonl(self.index, {"event": "add", "entry_id": entry_id, "expert_sha256": sha,
                                  "task_id": task_id, "added_at": added_at})
        return entry

    def get(self, entry_id: str) -> tuple[dict[str, Any], bytes]:
        entry = read_json(self.entries / f"{entry_id}.json")
        source = self._object_path(entry["expert_sha256"]).read_bytes()
        if sha256_bytes(source) != entry["expert_sha256"]:
            raise HarnessError(f"library object for {entry_id} does not match its digest (tampered)")
        return entry, source

    def all_entries(self) -> list[dict[str, Any]]:
        if not self.entries.exists():
            return []
        return [read_json(p) for p in sorted(self.entries.glob("*.json"))]

    def find(self, *, task_id: str | None = None, verifier_version: str | None = None,
             min_pass_rate: float = 0.0, limit: int = 5, exclude_task: str | None = None) -> list[dict[str, Any]]:
        found = []
        for entry in self.all_entries():
            rate = (entry.get("holdout") or {}).get("pass_rate") or 0.0
            if task_id and entry["task_id"] != task_id:
                continue
            if exclude_task and entry["task_id"] == exclude_task:
                continue
            if verifier_version and entry["verifier_version"] != verifier_version:
                continue
            if rate < min_pass_rate:
                continue
            found.append(entry)
        found.sort(key=lambda e: (-(e.get("holdout") or {}).get("pass_rate", 0.0),
                                  e.get("cost_usd") if e.get("cost_usd") is not None else float("inf"),
                                  e["entry_id"]))
        return found[:limit]

    def export_references(self, entries: list[dict[str, Any]], out_dir: Path) -> str:
        """Copy reference experts into a round and return the brief section text."""

        out_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for entry in entries:
            _, source = self.get(entry["entry_id"])
            name = f"{entry['entry_id']}.py"
            atomic_write_bytes(out_dir / name, source, mode=0o444)
            holdout = entry.get("holdout") or {}
            lines.append(f"- `references/{name}`: task `{entry['task_id']}`, class `{entry['expert_class']}`, "
                         f"holdout {holdout.get('passed')}/{holdout.get('scored')} under verifier "
                         f"{entry['verifier_version']} (reference only; your task may differ).")
        return "\n".join(lines) if lines else ""


def add_from_round(library: ExpertLibrary, paths: RoundPaths, adapter: TaskAdapter, *,
                   now: float, min_pass_rate: float | None = None) -> dict[str, Any] | None:
    """Admit the round's final expert if its holdout pass rate meets the threshold."""

    state = RoundState.load(paths.state)
    if not paths.holdout_results.exists():
        raise HarnessError("no holdout results; run eval-holdout first")
    if state.status == "invalid_tamper" or not state.data["manifest"].get("ok", True):
        raise HarnessError("round is invalid; its experts cannot enter the library")
    results = read_json(paths.holdout_results)
    final = results.get("final") or {}
    threshold = float(adapter.holdout.get("verified_pass_rate", 0.5) if min_pass_rate is None else min_pass_rate)
    rate = final.get("pass_rate")
    if rate is None or rate < threshold:
        return None
    sub = next(s for s in state.data["submissions"] if s["submission_id"] == results["final_submission"])
    source = (paths.submissions / sub["submission_id"] / "expert.py").read_bytes()
    round_info = read_json(paths.round_json)
    sessions = state.data["sessions"]
    cost = sum(float(s.get("cost_usd") or 0.0) for s in sessions)
    return library.add(source=source, expert_class=sub["expert_class"], task_id=adapter.task_id,
                       verifier_version=adapter.verifier_version,
                       holdout={"passed": final.get("passed"), "scored": final.get("scored"),
                                "pass_rate": rate, "per_family": final.get("per_family")},
                       brief_sha256=round_info.get("brief_sha256"),
                       task_ir_sha256=round_info.get("task_ir_sha256"),
                       model=round_info.get("agent", {}).get("model"),
                       backend=round_info.get("agent", {}).get("backend"),
                       cost_usd=round(cost, 4),
                       wall_s=(state.data["active_wall_s_final"] if state.data.get("active_wall_s_final") is not None
                               else state.active_wall_s(now)),
                       counted_rollouts=state.counters.get("counted"), round_id=state.data["round_id"],
                       added_at=utc_now(now), tags=[adapter.task_id])

