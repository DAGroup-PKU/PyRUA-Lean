"""Shared fixtures for the expert-harness tests: a toy task adapter in a temp directory,
fake experts and fake plans. Everything is hermetic (no network, GPU, simulator or model)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from datafarm.expert.adapter import HoldoutSeed, TaskAdapter, write_holdout_seeds
from datafarm.expert.util import Clock, deep_merge

TEMPLATE = """# Toy round {{ round.id }}

Workspace `{{ round.workspace }}`; rollouts `{{ round.rollouts }}`.

## Where the previous round stands

{{ handoff_md }}

## Task

Move the toy block onto the target.

## Expert interface

`ToyExpert(pose_bank, home=home)` with `next_plan(observation)` and `complete(...)`.

## Verifier ({{ verifier.version }})

{{ verifier.name }}; judged at {{ verifier.judge_points }}.

{{ verifier.criteria_md }}

## Runtime limits

{{ runtime_facts_md }}

## Budget

{{ budget.max_counted_rollouts }} counted rollouts, {{ budget.max_concurrent_rollouts }} at once,
{{ budget.total_wall }} in total, {{ budget.wall_per_session }} per session, {{ budget.max_sessions }} sessions,
{{ budget.max_usd }}. Infrastructure failures (infra) are not counted; at most {{ budget.max_infra_failures }}.

## Tools

`xh rollout`, `xh wait`, `xh status`, `xh result`, `xh submit`, `xh flag-for-human`.

## Readable paths

{{ readable_md }}

{{ masked_md }}

## Forbidden

{{ forbidden_md }}

## Deliverables

{{ deliverables_md }}

## Submit protocol

`xh submit "<note>"` at any time; `xh submit --final "<note>"` ends the round.

## Holdout policy

Holdout seeds are hidden and never shown to you.

## How to stop or abort

`xh flag-for-human "<reason>"`.

## References

{{ references_md }}
"""

HOLDOUT = (5000, 5001, 5002, 5003)
SPARES = (5100,)

EXPERT = '''"""Toy expert."""
import math

FAKE_QUALITY = {quality}


class ToyExpert:
    def __init__(self, pose_bank, home=None):
        self.done = False
        self.events = []
        self.phase = "start"
        self.fold_index = 0
        self._n = 0

    def next_plan(self, observation):
        self._n += 1
        if self._n > 2:
            self.done = True
            return None
        return {{"primitive_id": "p%d" % self._n, "angle": math.pi}}

    def complete(self, plan, observation, execution):
        self.events.append(plan["primitive_id"])
'''


def expert(quality: float = 0.5) -> str:
    return EXPERT.format(quality=quality)


def base_raw(root: Path) -> dict[str, Any]:
    (root / "protected").mkdir(exist_ok=True)
    (root / "protected" / "judge.py").write_text("# toy judge\n")
    (root / "readable").mkdir(exist_ok=True)
    (root / "readable" / "doc.md").write_text("toy doc\n")
    return {
        "schema": "datafarm.expert-harness.adapter/1",
        "task_id": "toy-task",
        "title": "Toy task",
        "vars": {"root": str(root)},
        "brief": {"template": "brief.md", "facts": {}},
        "verifier": {"version": "toy-v1", "name": "toy verifier", "judge_points": ["release", "settled"],
                     "criteria": ["Both points within 70 mm at release AND settled."],
                     "criteria_tokens": ["70 mm"], "files": ["{root}/protected/judge.py"],
                     "success_metric": "M2"},
        "runtime_facts": [{"id": "cap", "title": "Capacity", "text": "Trace capacity is 100 samples.",
                           "must_mention": ["100 samples"]}],
        "forbidden": ["No decompiling.", "No web search."],
        "deliverables": [{"file": "expert.py", "what": "the expert"}, {"file": "REPORT.md", "what": "the report"}],
        "seeds": {"dev": {"ranges": [[0, 99]]}},
        "holdout": {"seeds_file": str(root / "holdout.json"), "targets": "all", "max_targets": 4,
                    "max_concurrent": 4, "retry_infra": 1, "verified_pass_rate": 0.5, "timeout_s": 600},
        "budgets": {"wall_per_session_s": 60, "total_wall_s": 600, "max_counted_rollouts": 6,
                    "max_concurrent_rollouts": 2, "max_sessions": 4, "max_usd": 5.0, "max_infra_failures": 3,
                    "max_idle_sessions": 2, "final_grace_s": 30, "wait_cap_s": 30, "warn_before_wall_s": 60},
        "agent": {"backend": "fake", "model": "fake-model", "plan": str(root / "plan.json"),
                  "python": sys.executable},
        "expert": {"default_file": "expert.py", "default_class": "ToyExpert", "max_bytes": 100000,
                   "allowed_imports": ["__future__", "math"]},
        "workspace": {"readable": ["{root}/readable"], "protected": ["{root}/protected"]},
        "isolation": {"agent": {"mode": "none", "base": "allowlist"},
                      "rollout": {"mode": "none", "base": "root_ro"}},
        "rollout": {"backend": "fake"},
    }


def write_adapter(root: Path, overrides: dict[str, Any] | None = None, *, template: str = TEMPLATE) -> TaskAdapter:
    root.mkdir(parents=True, exist_ok=True)
    (root / "brief.md").write_text(template)
    raw = base_raw(root)
    if overrides:
        raw = deep_merge(raw, overrides)
    path = root / "adapter.json"
    path.write_text(json.dumps(raw, indent=1))
    write_holdout_seeds(root / "holdout.json",
                        [HoldoutSeed(s, "a" if s % 2 else "b") for s in HOLDOUT]
                        + [HoldoutSeed(s, "a", True) for s in SPARES], note="test")
    return TaskAdapter.load(path)


def write_plan(root: Path, sessions: list[list[dict[str, Any]]], cost: float = 0.25) -> Path:
    path = root / "plan.json"
    path.write_text(json.dumps({"cost_per_session": cost, "sessions": sessions}))
    return path


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


class FakeClock(Clock):
    """Deterministic clock: ``sleep`` advances time instantly."""

    def __init__(self, start: float = 1_790_000_000.0) -> None:
        self.t = float(start)

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.t += float(seconds)

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)
