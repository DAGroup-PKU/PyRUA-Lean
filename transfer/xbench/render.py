"""Render the system prompt a policy of one benchmark reads, the way the eval harness builds it.

Run with the benchmark's own interpreter and code (``sys.path`` gets ``--src``):

    python render.py --src REPO/src --card EP/card.json --robot R --primitives P --images I
        --feedback F --budget-s S [--library DIR] [--check EP/prompt.txt] [--out FILE]

``pyrualean.play.build_prompt`` is called as ``play_episode`` calls it for the harness's runs
(arm "cells", ``max_programs`` None, budget mode "calls", no guides).  ``--check`` compares the full
prompt with an episode's ``prompt.txt`` and prints the first differing line; ``--out`` writes the
prompt without its task card (the part that is the same for every cell of the benchmark).
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
for name in ("--src", "--card", "--robot", "--primitives", "--images", "--feedback"):
    p.add_argument(name, required=True)
p.add_argument("--budget-s", type=int, required=True)
p.add_argument("--library")
p.add_argument("--check", type=Path)
p.add_argument("--out", type=Path)
a = p.parse_args()

sys.path.insert(0, a.src)
from pyrualean.play import build_prompt  # noqa: E402
from pyrualean.prompt import TaskCard  # noqa: E402
from pyrualean.robots import get_adapter  # noqa: E402

raw = json.loads(Path(a.card).read_text())
known = {f.name for f in dataclasses.fields(TaskCard)}
unknown = set(raw) - known
if unknown:
    raise SystemExit(f"card fields this TaskCard does not have: {sorted(unknown)}")
card = TaskCard(**raw)
if card.robot != a.robot:
    raise SystemExit(f"card.robot is {card.robot!r}, not {a.robot!r}")
prompt = build_prompt(
    "cells",
    card,
    budget_s=a.budget_s,
    max_programs=None,
    images=a.images,
    feedback=a.feedback,
    guides=False,
    budget_mode="calls",
    adapter=get_adapter(a.robot),
    primitives=a.primitives,
    library=a.library,
)
tail = "\n\n" + card.render()
if not prompt.endswith(tail):
    raise SystemExit("the prompt does not end with the task card")
if a.check:
    want = a.check.read_text(encoding="utf-8")
    if prompt == want:
        print(f"identical to {a.check} ({len(prompt)} chars)")
    else:
        got, exp = prompt.splitlines(), want.splitlines()
        i = next((i for i, (x, y) in enumerate(zip(got, exp, strict=False)) if x != y), min(len(got), len(exp)))
        print(f"DIFFERS from {a.check} at line {i + 1}: {len(prompt)} vs {len(want)} chars")
        print("  rendered:", got[i][:200] if i < len(got) else "<end>")
        print("  episode: ", exp[i][:200] if i < len(exp) else "<end>")
if a.out:
    a.out.write_text(prompt[: -len(tail)] + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({len(prompt) - len(tail)} chars, task card removed)")
