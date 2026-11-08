"""Pair two evaluation roots layout by layout.

    python eval/compare.py ROOT_A ROOT_B [--criteria orig] [--json OUT]


For every task both roots judged, the episodes of the same layout are paired (the last judged
entry of each layout in each ledger): successes of each side, the layouts only one side solved,
and the two-sided exact sign test on those.  ``--criteria`` picks the verdict of a root built on
``criteria_v2`` (dual judging): ``faith`` (default) is the ledger's ``success``; ``orig`` is the
original criteria's ``v1 shadow final`` line of the episode's client log.  A task whose criteria
were not rewritten prints no shadow line: its ledger ``success`` is the original verdict.
"""

from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path

SHADOW = re.compile(r"\[criteria_v2\] v1 shadow final: env 0 success=(True|False)")


def sign_test(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def verdicts(root: Path, criteria: str) -> dict[tuple[str, int], bool]:
    """(task, layout) -> success of the last judged episode of each layout."""

    found: dict[tuple[str, int], bool] = {}
    for line in (root / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["kind"] != "policy":
            continue
        success = bool(entry["success"])
        if criteria == "orig":
            log = Path(entry["run_dir"]) / "robodojo_eval_client.log"
            shadow = SHADOW.findall(log.read_text(errors="replace")) if log.exists() else []
            if shadow:
                success = shadow[-1] == "True"
        found[(entry["task"], int(entry["layout"]))] = success
    return found


def compare(a: Path, b: Path, criteria: str) -> dict:
    left, right = verdicts(a, criteria), verdicts(b, criteria)
    rows: dict[str, dict[str, int]] = {}
    for key in sorted(set(left) & set(right)):
        row = rows.setdefault(key[0], {"pairs": 0, "a": 0, "b": 0, "a_only": 0, "b_only": 0})
        row["pairs"] += 1
        row["a"] += left[key]
        row["b"] += right[key]
        row["a_only"] += left[key] and not right[key]
        row["b_only"] += right[key] and not left[key]
    total = {k: sum(r[k] for r in rows.values()) for k in ("pairs", "a", "b", "a_only", "b_only")}
    total["p_sign"] = sign_test(total["a_only"], total["b_only"])
    return {"criteria": criteria, "a": str(a), "b": str(b), "tasks": rows, "total": total}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--criteria", choices=("faith", "orig"), default="faith")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    out = compare(args.a, args.b, args.criteria)
    print(f"A = {args.a}\nB = {args.b}\ncriteria: {args.criteria}")
    print(f"  {'task':32s} {'pairs':>5s} {'A':>4s} {'B':>4s} {'A only':>6s} {'B only':>6s}")
    for task, r in out["tasks"].items():
        counts = f"{r['pairs']:5d} {r['a']:4d} {r['b']:4d} {r['a_only']:6d} {r['b_only']:6d}"
        print(f"  {task:32s} {counts}")
    t = out["total"]
    counts = f"{t['pairs']:5d} {t['a']:4d} {t['b']:4d} {t['a_only']:6d} {t['b_only']:6d}"
    print(f"  {'total':32s} {counts}   sign test p = {t['p_sign']:.3g}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
