# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Strict paired comparison of two systems on identical (backend, task, seed) cells.

A pair is only formed when both systems have exactly one episode for a key;
duplicate or unmatched keys invalidate the report instead of being averaged.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .metrics import EpisodeMetrics, load_episode_result, summarize

PAIR_SYSTEMS = ("rpent", "pyrualean")
PAIR_FIELDS = (
    "calls",
    "stateful_calls",
    "env_steps",
    "turns",
    "wall_time_s",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def _extract_records(value: Any, *, source: str) -> list[EpisodeMetrics]:
    if isinstance(value, list):
        out: list[EpisodeMetrics] = []
        for index, item in enumerate(value):
            out.extend(_extract_records(item, source=f"{source}[{index}]"))
        return out
    if not isinstance(value, dict):
        raise ValueError(f"expected a metrics object or list in {source}")
    if value.get("schema") == "pyrualean-episode-v2":
        return [load_episode_result(Path(source.split("[")[0]))]
    if "backend" in value and "task" in value:
        return [EpisodeMetrics.from_dict(value, source=source)]
    for key in ("metrics", "runs", "episodes"):
        if key in value:
            return _extract_records(value[key], source=f"{source}.{key}")
    raise ValueError(f"no episode metrics found in {source}")


def load_metrics_file(path: str | Path) -> list[EpisodeMetrics]:
    location = Path(path).expanduser().resolve()
    if location.is_dir():
        location = location / "result.json"
    value = json.loads(location.read_text(encoding="utf-8"))
    return _extract_records(value, source=str(location))


def _key(record: EpisodeMetrics) -> tuple[str, str, int | None]:
    return (record.backend, record.task, record.seed)


def _delta(left: Any, right: Any) -> float | int | None:
    if left is None or right is None:
        return None
    try:
        return right - left
    except TypeError:
        return None


def compare_runs(
    runs: Iterable[EpisodeMetrics], *, systems: tuple[str, str] = PAIR_SYSTEMS
) -> dict[str, Any]:
    """Pair episodes by (backend, task, seed); ``delta`` = right minus left."""

    if len(systems) != 2 or systems[0] == systems[1]:
        raise ValueError("systems must contain two distinct names")
    indexed: dict[tuple[str, str, int | None], dict[str, list[EpisodeMetrics]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_runs = list(runs)
    for run in all_runs:
        indexed[_key(run)][run.system].append(run)

    duplicates: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for key in sorted(indexed, key=lambda k: (k[0], k[1], k[2] is None, k[2] or 0)):
        by_system = indexed[key]
        left, right = by_system.get(systems[0], []), by_system.get(systems[1], [])
        key_dict = {"backend": key[0], "task": key[1], "seed": key[2]}
        if len(left) > 1 or len(right) > 1:
            duplicates.append(
                {**key_dict, "counts": {systems[0]: len(left), systems[1]: len(right)}}
            )
            continue
        if len(left) != 1 or len(right) != 1:
            unmatched.append({**key_dict, "present": sorted(by_system)})
            continue
        lrec, rrec = left[0], right[0]
        delta = {name: _delta(getattr(lrec, name), getattr(rrec, name)) for name in PAIR_FIELDS}
        delta["native_success_delta"] = (
            int(rrec.environment_success) - int(lrec.environment_success)
            if lrec.environment_success is not None and rrec.environment_success is not None
            else None
        )
        pairs.append(
            {**key_dict, systems[0]: lrec.to_dict(), systems[1]: rrec.to_dict(), "delta": delta}
        )

    paired_left = [EpisodeMetrics.from_dict(p[systems[0]]) for p in pairs]
    paired_right = [EpisodeMetrics.from_dict(p[systems[1]]) for p in pairs]
    return {
        "protocol": "pyrualean-paired-v2",
        "systems": list(systems),
        "pairing": {
            "valid": bool(pairs) and not duplicates and not unmatched,
            "matched_episodes": len(pairs),
            "input_episodes": len(all_runs),
            "duplicate_keys": duplicates,
            "unmatched_keys": unmatched,
            "key_definition": ["backend", "task", "seed"],
        },
        "summary": {systems[0]: summarize(paired_left), systems[1]: summarize(paired_right)},
        "pairs": pairs,
    }


def compare_metric_files(
    paths: Iterable[str | Path], *, systems: tuple[str, str] = PAIR_SYSTEMS
) -> dict[str, Any]:
    runs: list[EpisodeMetrics] = []
    for path in paths:
        runs.extend(load_metrics_file(path))
    return compare_runs(runs, systems=systems)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--left-system", default=PAIR_SYSTEMS[0])
    parser.add_argument("--right-system", default=PAIR_SYSTEMS[1])
    args = parser.parse_args(argv)
    report = compare_metric_files(args.metrics, systems=(args.left_system, args.right_system))
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["pairing"]["valid"] else 2


__all__ = ["compare_metric_files", "compare_runs", "load_metrics_file", "main"]
