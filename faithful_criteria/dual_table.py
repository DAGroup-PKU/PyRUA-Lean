"""The 2x2 of the two criteria on the same episodes, from dual-judged evaluation roots.

    python faithful_criteria/dual_table.py ROOT [ROOT ...] [--json OUT]

Each ROOT is an ``eval/robodojo/run.py`` root whose config's ``source_root`` is a build of this directory
(``faithful_criteria/build_source.sh``): every episode is judged by the instruction-faithful criteria
(C^faith, the ledger's ``success`` and the ``dual final`` line of the client log) and, alongside,
by the original cb2a37d criteria (C^orig, the ``v1 shadow final`` line).  Per root and task:
n11 (both succeed), n10 (C^orig succeeds, C^faith fails: a success the instruction does not
support), n01 (C^orig fails, C^faith succeeds: a completion the original criteria reject), n00;
both success rates; the spurious-success rate n10 / (n10 + n11); the exact McNemar p-value of n10
against n01; minutes and tokens of the judged episodes.  Episodes still running are not counted.
"""

from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path

SHADOW = re.compile(r"\[criteria_v2\] v1 shadow final: env 0 success=(True|False)")
DUAL = re.compile(r"\[criteria_v2\] dual final: env 0 v2 success=(True|False)")
# the kind of change of each task (README.md), fixed before the runs
KIND = {
    "fasten_screws": "tightened", "press_by_number": "tightened", "swap_blocks": "tightened",
    "pour_by_language": "tightened", "sweep_blocks": "tightened", "sweep_blocks_random": "tightened",
    "insert_key": "tightened", "fill_pen_holder": "relaxed+tightened", "organize_table": "relaxed+tightened",
    "make_toast": "relaxed+fixed", "make_toast_random": "relaxed+fixed",
    "hang_mugs": "fixed", "hang_mugs_random": "fixed", "solve_equation": "fixed", "pour_balls_into_vase": "relaxed+fixed",
    "fill_egg_holder": "fixed", "build_tower": "relaxed", "make_kong": "relaxed", "play_tic_tac_toe": "fixed",
    "pour_liquid_into_cup": "relaxed", "pour_liquid_into_cup_random": "relaxed", "store_tools_in_toolbox": "relaxed",
    "pack_objects_into_box": "relaxed", "pack_objects_into_box_random": "relaxed",
    "store_laptop_and_headphones": "relaxed", "store_laptop_and_headphones_random": "relaxed",
}  # fmt: skip


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def compute(root: Path) -> list[dict]:
    last = {}
    ledger = root / "episodes.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                last[(e["task"], e["layout"])] = e
    cells: dict[str, dict] = {}
    for e in last.values():
        c = cells.setdefault(e["task"], {"n11": 0, "n10": 0, "n01": 0, "n00": 0, "other": 0, "minutes": [],
                                         "input": 0, "cached": 0, "output": 0, "requests": 0})  # fmt: skip
        if e["kind"] != "policy":
            c["other"] += 1
            continue
        text = (Path(e["run_dir"]) / "robodojo_eval_client.log").read_text(errors="replace")
        v1, dual = SHADOW.findall(text), DUAL.findall(text)
        if not v1 or not dual:
            raise SystemExit(f"{e['run_dir']}: no shadow/dual line (not a dual-judge episode?)")
        if (dual[-1] == "True") != e["success"]:
            raise SystemExit(f"{e['run_dir']}: ledger success differs from the dual line")
        c["n" + ("1" if v1[-1] == "True" else "0") + ("1" if e["success"] else "0")] += 1
        c["minutes"].append(e["seconds"] / 60)
        c["input"] += e.get("input_tokens") or 0
        c["cached"] += e.get("cached_input_tokens") or 0
        c["output"] += e.get("output_tokens") or 0
        c["requests"] += e.get("requests") or 0
    rows = []
    for task in sorted(cells):
        c = cells[task]
        n = c["n11"] + c["n10"] + c["n01"] + c["n00"]
        m = sorted(c.pop("minutes"))
        rows.append({"task": task, "n": n, **c, "median_minutes": m[len(m) // 2] if m else None,
                     "v1_rate": (c["n11"] + c["n10"]) / n if n else None, "v2_rate": (c["n11"] + c["n01"]) / n if n else None,
                     "spurious": c["n10"] / (c["n10"] + c["n11"]) if c["n10"] + c["n11"] else None,
                     "p_mcnemar": mcnemar_exact(c["n10"], c["n01"])})  # fmt: skip
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="eval/robodojo/run.py roots")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    out = {}
    for name in args.roots:
        rows = compute(name)
        out[str(name)] = rows
        print(f"== {name}")
        print(f"  {'task':30s} {'n':>3s} {'n11':>4s} {'n10':>4s} {'n01':>4s} {'n00':>4s} {'other':>5s}  {'v1':>5s} {'v2':>5s}  med min")
        for r in rows:
            v1 = f"{r['v1_rate']:.2f}" if r["n"] else "  -  "
            v2 = f"{r['v2_rate']:.2f}" if r["n"] else "  -  "
            med = f"{r['median_minutes']:.0f}" if r["median_minutes"] is not None else "-"
            print(f"  {r['task']:30s} {r['n']:3d} {r['n11']:4d} {r['n10']:4d} {r['n01']:4d} {r['n00']:4d} {r['other']:5d}  {v1} {v2}  {med:>7s}")
        t = {k: sum(r[k] for r in rows) for k in ("n", "n11", "n10", "n01", "n00", "other", "input", "cached", "output", "requests")}
        print(f"  total judged {t['n']} (n11 {t['n11']} n10 {t['n10']} n01 {t['n01']} n00 {t['n00']}, other {t['other']}); "
              f"tokens in {t['input'] / 1e6:.1f}M (cached {t['cached'] / 1e6:.1f}M) out {t['output'] / 1e6:.2f}M, {t['requests']} requests")  # fmt: skip
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
