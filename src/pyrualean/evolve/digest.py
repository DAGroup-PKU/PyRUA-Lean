# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""One episode's run directory, condensed for whoever diagnoses it.

A run directory (``pyrualean play`` / the XPolicyLab adapter) holds the verdict
(``result.json``), the cells (``workspace/cell_NNN.py``), the ledger of primitive calls
(``calls.jsonl``) and the per-step images (``steps/``).  :func:`digest` reads them and says
what a reviewer looks at first: did it succeed, how many of the task's ordered checks it
passed, where its native-step budget went, and which primitive calls did not do what was
asked (a motion that did not reach, a grasp that holds nothing, an object not found).

It reads, never writes into the run, and judges nothing it cannot read off the record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _failure(record: dict[str, Any]) -> str | None:
    """Why a ledger record shows a primitive call that did not do what was asked, or None."""

    if not record.get("ok", True):
        return f"error: {record.get('error')}"
    summary = record.get("summary") or {}
    tool = record.get("tool", "")
    if summary.get("reached") is False:
        dist = summary.get("dist_to_target_m")
        return "not reached" + (f" ({dist:.3f} m off)" if isinstance(dist, (int, float)) else "")
    if summary.get("blocked"):
        return "blocked"
    if summary.get("ik_error"):
        return f"ik: {summary['ik_error']}"
    if summary.get("found") is False:
        return "not found"
    if (
        tool in {"set_gripper", "grip"}
        and summary.get("holding") is False
        and summary.get("width_m") is not None
    ):
        return "holds nothing"
    # ``truncated`` is not a failed call: the benchmark ended the episode, by success as well
    # as by a spent budget (``digest`` reports where, ``render`` says which)
    return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _client_result(root: Path) -> Any:
    """RoboDojo's own ``_result.json`` for the run (its client writes it under
    ``bench/eval_result/``), or None when there is none.

    A run evaluates one layout, so it holds at most one; more is refused rather than guessed.
    """

    results = sorted((root / "bench" / "eval_result").rglob("_result.json"))
    if len(results) > 1:
        raise ValueError(f"{len(results)} RoboDojo results under {root / 'bench'}: {results}")
    return _read_json(results[0]) if results else None


def _ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


#: Statuses of a run that never became an episode the policy could be judged on.
INFRA_STATUSES = frozenset({"boot_error", "error", "killed"})


def identity(result: dict[str, Any]) -> dict[str, Any]:
    """What must be equal for two runs to be compared: everything but the library.

    The reasoning effort is the one the runtime sent: ``reasoning_effort`` in
    ``result.json`` is the codex/claude setting, the openai runtime sends
    ``extra_body.reasoning_effort``.
    """

    generation = result.get("generation") or {}
    options = generation.get("options") or {}
    runtime = (result.get("host") or {}).get("runtime", "codex")
    effort = result.get("reasoning_effort")
    if runtime == "openai":
        effort = (options.get("extra_body") or {}).get("reasoning_effort")
    return {
        "robot": result.get("robot"),
        "arm": result.get("arm"),
        "runtime": runtime,
        "model": result.get("model"),
        "reasoning_effort": effort,
        "primitive_set": (result.get("host") or {}).get("primitive_set"),
        "images": result.get("images"),
        "feedback": result.get("feedback"),
        "max_turns": result.get("max_turns"),
        "max_tokens": options.get("max_tokens"),
        "max_requests": options.get("max_requests"),
        "context_groups": options.get("context_groups"),
        "harness": result.get("harness"),
        "execution_sha256": result.get("execution_sha256"),
    }


def classify(result: dict[str, Any]) -> str:
    """``policy``: an episode the benchmark judged; ``infra``: one it never could."""

    if result.get("status") in INFRA_STATUSES:
        return "infra"
    if (result.get("generation") or {}).get("status") == "error":
        return "infra"
    if result.get("verdict") is None and not result.get("environment_success"):
        return "infra"  # the client never reported a verdict
    return "policy"


def claim(result: dict[str, Any]) -> str | None:
    """How the policy's own ``finish`` compares with the benchmark's verdict.

    ``overclaim``: it finished with success and the benchmark judged a failure;
    ``unclaimed_success``: the benchmark judged a success it finished without claiming;
    ``agreed``: both say the same; None: no ``finish`` (a success usually ends the episode
    before one), no judged episode, or a replay (whose ``finish`` the replay runtime makes,
    always ``failure``: nobody claimed anything).
    """

    finish = result.get("finish") or {}
    if not finish.get("status") or classify(result) != "policy":
        return None
    if (result.get("generation") or {}).get("runtime") == "replay":
        return None
    claimed = finish["status"] == "success"
    succeeded = bool(result.get("environment_success"))
    if claimed and not succeeded:
        return "overclaim"
    if succeeded and not claimed:
        return "unclaimed_success"
    return "agreed"


def _evidence(root: Path, calls: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    """Point each failed call at the cell that made it and the frames recorded during it."""

    turns = _read_json(root / "tool_turns.json") or []
    cells = [t for t in turns if t.get("tool") == "python" and t.get("calls")]
    steps = root / "steps"
    frames = sorted(steps.iterdir(), key=lambda p: p.name) if steps.is_dir() else []
    stamps = [(p.name, p.stat().st_mtime) for p in frames]
    for failure in failures:
        index = failure["index"]
        turn = next((t for t in cells if t["calls"][0] <= index < t["calls"][1]), None)
        if turn is None:
            continue
        number = cells.index(turn) + 1  # cell_NNN counts python turns from 1
        failure["cell"] = f"cell_{number:03d}.py"
        start = float(turn.get("started_at") or 0.0)
        end = start + float(turn.get("elapsed_s") or 0.0)
        during = [name for name, t in stamps if start <= t <= end + 1.0]
        before = [name for name, t in stamps if t < start]
        failure["frames"] = {"before": before[-1:] or None, "during": during or None}


def digest(run_dir: str | Path) -> dict[str, Any]:
    """The facts of one run directory a diagnosis starts from.

    Raises FileNotFoundError when ``run_dir`` has no ``result.json``.  A run that never
    became a judged episode (``classify`` = ``infra``) is reported as such, never as a
    policy failure.
    """

    root = Path(run_dir).expanduser().resolve()
    result = _read_json(root / "result.json")
    if result is None:
        raise FileNotFoundError(f"no result.json under {root}")
    verdict = result.get("verdict") or {}
    calls = _ledger(root / "calls.jsonl")
    failures = []
    for record in calls:
        why = _failure(record)
        if why is not None:
            failures.append(
                {
                    "index": record.get("index"),
                    "tool": record.get("tool"),
                    "why": why,
                    "kwargs": record.get("kwargs"),
                }
            )
    _evidence(root, calls, failures)
    ended_at = next(
        (r.get("index") for r in calls if (r.get("summary") or {}).get("truncated")), None
    )
    steps_by_tool: dict[str, int] = {}
    for record in calls:
        if record.get("stateful") and isinstance(record.get("env_steps"), int):
            tool = str(record.get("tool"))
            steps_by_tool[tool] = steps_by_tool.get(tool, 0) + record["env_steps"]
    cells = sorted((root / "workspace").glob("cell_*.py")) if (root / "workspace").is_dir() else []
    generation = result.get("generation") or {}
    # the benchmark's process score, as RoboDojo's own client recorded it for this run
    client = _client_result(root) or {}
    details = list((client.get("details") or {}).values())
    score = float(details[0]["score"]) if len(details) == 1 else verdict.get("score")
    return {
        "run_dir": str(root),
        "kind": classify(result),
        "instance": result.get("instance"),
        "split": result.get("split"),
        "revision": result.get("revision"),
        "identity": identity(result),
        "task": result.get("task"),
        "layout": result.get("seed"),
        "eval_seed": verdict.get("eval_seed"),
        "success": bool(result.get("environment_success")),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "error": result.get("error"),
        "stages_passed": verdict.get("stages_passed"),
        "stages_total": verdict.get("stages_total"),
        "blocking_stage": verdict.get("blocking_stage"),
        "score": score,
        "native_steps": verdict.get("step", result.get("env_steps")),
        "step_limit": verdict.get("step_limit", result.get("max_episode_steps")),
        "steps_by_tool": dict(sorted(steps_by_tool.items(), key=lambda kv: -kv[1])),
        "calls": len(calls),
        "failed_calls": failures,
        "ended_at_call": ended_at,
        "cells": [p.name for p in cells],
        "finish": result.get("finish"),
        "claim": claim(result),
        "library": (result.get("library") or {}).get("sha256"),
        "tokens": generation.get("total_tokens"),
        "runtime": generation.get("runtime"),
    }


def _check_name(name: str) -> str:
    """A compound check's ``a+b+a+b`` (one term per object it repeats over) as ``(a+b)*2``."""

    terms = name.split("+")
    for size in range(1, len(terms) // 2 + 1):
        times, rest = divmod(len(terms), size)
        if not rest and terms == terms[:size] * times:
            unit = "+".join(terms[:size])
            return f"({unit})*{times}" if size > 1 else f"{unit}*{times}"
    return name


def render(facts: dict[str, Any], *, max_failures: int = 12, scoring: bool = True) -> str:
    """The digest as the short text a reviewer reads before opening any image.

    ``scoring=False`` leaves out what tells how RoboDojo checks and scores the task: the score,
    the stages passed and the blocking stage's checks (success stays).
    """

    stages = (
        f"{facts['stages_passed']}/{facts['stages_total']}"
        if facts.get("stages_total") is not None
        else "n/a"
    )
    measures = f"   score: {facts.get('score')}   stages passed: {stages}" if scoring else ""
    lines = [
        f"# {facts['task']} layout {facts['layout']} (eval seed {facts['eval_seed']})",
        f"success: {facts['success']}{measures}   status: {facts['status']}"
        + (f" ({facts['reason']})" if facts.get("reason") else ""),
        f"native steps: {facts['native_steps']} / {facts['step_limit']}   "
        f"calls: {facts['calls']}   cells: {len(facts['cells'])}   tokens: {facts['tokens']}",
    ]
    if facts.get("kind") == "infra":
        lines.insert(1, f"INFRASTRUCTURE: not a judged episode ({facts.get('error')})")
    if scoring and facts.get("blocking_stage"):
        marks = {True: "holds", False: "FAILS", None: "not evaluated"}
        parts = [
            f"{_check_name(c['check'])}"
            f"{'(' + ','.join(c['labels'].values()) + ')' if c['labels'] else ''}: "
            f"{marks.get(c['holds'], c['holds'])}"
            for c in facts["blocking_stage"]
        ]
        counted = {part: parts.count(part) for part in parts}  # first-seen order
        lines.append(
            "blocking stage: "
            + "; ".join(p + (f" (x{n})" if n > 1 else "") for p, n in counted.items())
        )
    if facts["steps_by_tool"]:
        spent = ", ".join(f"{tool} {n}" for tool, n in facts["steps_by_tool"].items())
        lines.append(f"steps by primitive: {spent}")
    if facts.get("ended_at_call") is not None:
        steps, limit = facts.get("native_steps"), facts.get("step_limit")
        if facts["success"]:
            why = "all of the task's checks passed" if scoring else "success"
        elif isinstance(steps, int) and isinstance(limit, int) and steps >= limit:
            why = "native-action budget spent"
        else:
            why = "failed with budget left" + (" (e.g. a counted check overran)" if scoring else "")
        lines.append(
            f"benchmark ended the episode (truncated) by call #{facts['ended_at_call']}: {why}"
        )
    if facts.get("finish"):
        lines.append(f"finish: {facts['finish'].get('status')}: {facts['finish'].get('summary')}")
    if facts.get("claim") == "overclaim":
        lines.append(
            "OVERCLAIM: the policy finished with success, the benchmark judged a failure: find "
            "what it took for done (judgement), as well as what failed"
        )
    elif facts.get("claim") == "unclaimed_success":
        lines.append("unclaimed success: the benchmark judged a success the policy did not claim")
    failures = facts["failed_calls"]
    lines.append(f"failed primitive calls: {len(failures)}")
    for failure in failures[:max_failures]:
        where = ""
        if failure.get("cell"):
            frames = failure.get("frames") or {}
            seen = (frames.get("before") or []) + (frames.get("during") or [])
            where = (
                f"  [{failure['cell']}; frames {seen[0]}..{seen[-1]}]"
                if seen
                else (f"  [{failure['cell']}]")
            )
        lines.append(f"  #{failure['index']} {failure['tool']}: {failure['why']}{where}")
    if len(failures) > max_failures:
        lines.append(f"  ... {len(failures) - max_failures} more in calls.jsonl")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyrualean.evolve.digest", description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="episode run directories")
    parser.add_argument("--json", action="store_true", help="print the facts as JSON lines")
    args = parser.parse_args(argv)
    for run_dir in args.run_dirs:
        facts = digest(run_dir)
        sys.stdout.write(json.dumps(facts) + "\n" if args.json else render(facts) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
