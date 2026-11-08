"""Static acceptance checks of a RoboDojo distilled library: the BRIEF's size limits, the card
format, no source task name in a card's do:/applies: or in skills, every library.<module>[.<name>]
a card names exists, skills parse and import.

    python transfer/robodojo/check.py DIR [DIR ...]           # checks DIR/out; names: DIR/SOURCES.json
    python transfer/robodojo/check.py --lib LIB --names S.json  # any library (lessons/, skills/)

--names adds the task names of more SOURCES.json files (e.g. of every campaign set one distils
from, so that no library names any of them).  Exit status 1 when any PROBLEM is found."""

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.dont_write_bytecode = True  # importing a library's skills leaves no __pycache__ in it
FIELDS = ("stage", "symptom", "applies", "do", "evidence")


def check(lib, names, label):
    lessons = sorted((lib / "lessons").glob("*.md")) if (lib / "lessons").is_dir() else []
    skills = sorted((lib / "skills").glob("*.py")) if (lib / "skills").is_dir() else []
    lc = sum(len(p.read_text()) for p in lessons)
    sc = sum(len(p.read_text()) for p in skills)
    probs = []
    if lc > 12000:
        probs.append(f"lessons {lc} > 12000 chars")
    if len(skills) > 2 or sc > 15000:
        probs.append(f"skills {len(skills)} files / {sc} chars")
    if not lessons and not skills:
        probs.append("no lessons or skills")
    cards, modules = [], {}
    for p in skills:
        src = p.read_text()
        try:
            ast.parse(src)
        except SyntaxError as e:
            probs.append(f"{p.name}: {e}")
            continue
        hit = [t for t in names if t.lower() in src.lower()]
        if hit:
            probs.append(f"{p.name}: task names {hit}")
        try:
            spec = importlib.util.spec_from_file_location(f"library.{p.stem}", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = m
            spec.loader.exec_module(m)
            fns = [
                n
                for n, v in vars(m).items()
                if callable(v)
                and not n.startswith("_")
                and getattr(v, "__module__", "") == spec.name
            ]
            modules[p.stem] = m
        except Exception as e:
            probs.append(f"{p.name}: import failed {e!r}")
            fns = []
        print(f"   {p.name}: {len(src)} chars, functions {fns}")
    for p in lessons:
        for block in re.split(r"(?m)^(?=## )", p.read_text())[1:]:
            cid = block.splitlines()[0][3:].strip()
            f = {}
            for line in block.splitlines()[1:]:
                k, sep, v = line.partition(":")
                if sep and not k.startswith(" ") and k.strip().lower() in FIELDS:
                    f[k.strip().lower()] = v.strip()
            miss = [k for k in FIELDS if k not in f]
            if miss:
                probs.append(f"{cid}: missing {miss}")
            hit = [
                t for t in names if t.lower() in (f.get("do", "") + f.get("applies", "")).lower()
            ]
            if hit:
                probs.append(f"{cid}: task name in do/applies {hit}")
            for mod, attr in set(re.findall(r"\blibrary\.(\w+)(?:\.(\w+))?", block)):
                if mod not in modules:
                    probs.append(
                        f"{cid}: names library.{mod}, which is not a skill module that imports"
                    )
                elif attr and not hasattr(modules[mod], attr):
                    probs.append(
                        f"{cid}: names library.{mod}.{attr}, which {mod}.py does not define"
                    )
            cards.append(cid)
    print(
        f"{label}: {len(cards)} cards, lessons {lc} chars, skills {len(skills)} files {sc} chars;",
        "OK" if not probs else "PROBLEMS:",
    )
    for x in probs:
        print("   -", x)
    return not probs


ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="*", type=Path)
ap.add_argument("--lib", type=Path)
ap.add_argument(
    "--names",
    nargs="+",
    type=Path,
    default=[],
    help="SOURCES.json files whose task names are forbidden",
)
a = ap.parse_args()
extra = {t for s in a.names for t in json.loads(s.read_text())}
ok = True
for d in a.dirs:
    ok &= check(
        d / "out", sorted(extra | set(json.loads((d / "SOURCES.json").read_text()))), d.name
    )
if a.lib:
    if not extra:
        raise SystemExit("--lib needs --names (the source tasks' SOURCES.json)")
    ok &= check(a.lib, sorted(extra), str(a.lib))
sys.exit(0 if ok else 1)
