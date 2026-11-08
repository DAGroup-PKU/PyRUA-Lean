# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Is a candidate library better than its parent on the same layouts?

Each side is a list of episode digests (:func:`.digest.digest`) run on the same
(task, eval seed, layout) cells.  Per task, the sides are compared on, in order:

1. successes;
2. mean stages passed (how far failed episodes got: RoboDojo's ordered checks);
3. mean score (RoboDojo's process score), when every episode of both sides carries one.

The first that differs decides the task.  The candidate is accepted when it wins at least one
task and loses none; anything else keeps the parent, a tie included.

The comparison refuses (:class:`IncompleteComparison`), rather than scores, when the sides
were not run on the same cells, when a run never became a judged episode (an
infrastructure failure: see :func:`.digest.classify`), when a run lacks its eval seed, or
when the runs were not made with the same execution settings (:func:`.digest.identity`):
an incomplete comparison is not a loss and not a tie, it is not a comparison.  A cell may
be run more than once on a side when each run carries its own ``repeat`` number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .digest import digest


class IncompleteComparison(ValueError):
    """The two sides were not run on the same cells, or an episode has no verdict."""


def _cell(facts: dict[str, Any]) -> tuple[str, int, int, int]:
    if facts.get("eval_seed") is None:
        raise IncompleteComparison(f"run without an eval seed: {facts.get('run_dir')}")
    return (
        str(facts["task"]),
        int(facts["eval_seed"]),
        int(facts["layout"]),
        int(facts.get("repeat") or 0),
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def task_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Successes, mean stages passed, mean score and episode count of one task's episodes."""

    stages = [float(e["stages_passed"]) for e in episodes if e.get("stages_passed") is not None]
    scores = [float(e["score"]) for e in episodes if e.get("score") is not None]
    normalized = [
        float(episode["stages_passed"]) / float(episode["stages_total"])
        for episode in episodes
        if episode.get("stages_passed") is not None and episode.get("stages_total")
    ]
    return {
        "episodes": len(episodes),
        "successes": sum(1 for e in episodes if e["success"]),
        "mean_stages": _mean(stages) if len(stages) == len(episodes) else None,
        "mean_stage_fraction": _mean(normalized) if len(normalized) == len(episodes) else None,
        "stages_total": next((e["stages_total"] for e in episodes if e.get("stages_total")), None),
        # only a score every episode carries is compared
        "mean_score": _mean(scores) if len(scores) == len(episodes) else None,
    }


def _compare(parent: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, str]:
    """+1 candidate better, -1 worse, 0 tie; with the criterion that decided."""

    if candidate["successes"] != parent["successes"]:
        better = candidate["successes"] > parent["successes"]
        return (1 if better else -1), "successes"
    for key, name in (("mean_stages", "mean stages passed"), ("mean_score", "mean score")):
        p, c = parent[key], candidate[key]
        if p is not None and c is not None and abs(c - p) > 1e-9:
            return (1 if c > p else -1), name
    return 0, "tie"


def _check_runs(side: str, runs: list[dict[str, Any]]) -> None:
    infra = [r["run_dir"] for r in runs if r.get("kind") == "infra"]
    if infra:
        raise IncompleteComparison(f"{side}: not judged episodes (infrastructure): {infra}")


def compare(parent: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    """The verdict of the gate over two lists of digests of the same cells."""

    _check_runs("parent", parent)
    _check_runs("candidate", candidate)
    if not parent or not candidate:
        raise IncompleteComparison("empty comparison")
    identities = {json.dumps(r.get("identity"), sort_keys=True) for r in [*parent, *candidate]}
    if len(identities) > 1:
        raise IncompleteComparison(f"runs differ in their execution settings: {sorted(identities)}")
    p_cells = {_cell(f): f for f in parent}
    c_cells = {_cell(f): f for f in candidate}
    if len(p_cells) != len(parent) or len(c_cells) != len(candidate):
        raise IncompleteComparison("a cell was run twice on one side without a repeat number")
    if set(p_cells) != set(c_cells):
        only_p = sorted(set(p_cells) - set(c_cells))
        only_c = sorted(set(c_cells) - set(p_cells))
        raise IncompleteComparison(f"cells differ: parent only {only_p}, candidate only {only_c}")
    for cell in p_cells:
        if p_cells[cell].get("instance") != c_cells[cell].get("instance"):
            raise IncompleteComparison("layout content or environment identity differs")
        if p_cells[cell].get("stages_total") != c_cells[cell].get("stages_total"):
            raise IncompleteComparison("stage totals differ")
    tasks = sorted({cell[0] for cell in p_cells})
    per_task = {}
    for task in tasks:
        cells = [cell for cell in p_cells if cell[0] == task]
        pm = task_metrics([p_cells[c] for c in cells])
        cm = task_metrics([c_cells[c] for c in cells])
        sign, decided_by = _compare(pm, cm)
        per_task[task] = {"parent": pm, "candidate": cm, "result": sign, "decided_by": decided_by}
    wins = [t for t, r in per_task.items() if r["result"] > 0]
    losses = [t for t, r in per_task.items() if r["result"] < 0]
    accepted = bool(wins) and not losses
    if losses:
        reason = f"regresses on {', '.join(losses)}"
    elif wins:
        reason = f"better on {', '.join(wins)}, no task worse"
    else:
        reason = "tie on every task: the parent is kept"
    return {
        "accepted": accepted,
        "reason": reason,
        "per_task": per_task,
        "identity": json.loads(next(iter(identities))) if identities else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyrualean.evolve.gate", description=__doc__)
    parser.add_argument("--parent", nargs="+", type=Path, required=True, help="parent run dirs")
    parser.add_argument("--candidate", nargs="+", type=Path, required=True, help="candidate runs")
    args = parser.parse_args(argv)
    verdict = compare([digest(p) for p in args.parent], [digest(c) for c in args.candidate])
    sys.stdout.write(json.dumps(verdict, indent=1) + "\n")
    return 0 if verdict["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
