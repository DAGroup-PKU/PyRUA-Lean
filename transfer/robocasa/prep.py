"""Build the inputs of a RoboCasa365 in-benchmark distillation (the rc10 libraries) from finished
learn campaigns:

    python transfer/robocasa/prep.py OUT_DIR CAMPAIGN_ROOT [CAMPAIGN_ROOT ...] [--tasks T [T ...]]
        [--card CARD.json]

- inputs/<task>/: the campaign's active library revision as the policy read it (its view: skills/,
  lessons/ without retired cards), the last worker's FINDINGS.md and the dev successes per round
  with each round's revision, as transfer/robodojo/prep.py does for RoboDojo;
- inputs/robot/system_prompt.md: what a robocasa_rgb policy reads besides library and task card,
  rendered by transfer/xbench/render.py with eval_robocasa's settings (as xbench's robocasa target;
  --card replaces its card); inputs/robot/api/: robocasa_rgb.py, _robot.py.
Also writes OUT_DIR/SOURCES.json and OUT_DIR/MANIFEST.json.  The source tasks are all tasks of the
given campaigns, or those named by --tasks; use the campaigns of one setting (all
--worker-scoring off, or all on).
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

XBENCH = Path(__file__).resolve().parents[1] / "xbench"
sys.path.insert(0, str(XBENCH))
from prep import SRC, TARGETS, render  # noqa: E402

ROBOT = TARGETS["robocasa"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def config(campaign):
    return json.loads((campaign / "config.json").read_text())


def copy_task(c, task, dst):
    rev = json.loads((c / "library" / task / "active.json").read_text())["revision"]
    src = c / "library" / task / "views" / rev / task
    for part in ("skills", "lessons"):
        if (src / part).is_dir():
            shutil.copytree(src / part, dst / part, ignore=shutil.ignore_patterns("__pycache__"))
    # worker-rN.void-<stamp>: a round the campaign's requeue voided, not a finished worker
    workers = sorted(
        (p for p in (c / "tasks" / task).glob("worker-r*") if p.name[len("worker-r") :].isdigit()),
        key=lambda p: int(p.name[len("worker-r") :]),
    )
    if not workers:
        raise SystemExit(f"{task}: no worker-r<N> in {c / 'tasks' / task}")
    shutil.copyfile(workers[-1] / "FINDINGS.md", dst / "FINDINGS-last.md")
    dev = json.loads((c / "report.json").read_text())["tasks"][task]["dev"]
    rounds = [r["round"] for r in dev if r["revision"] == rev]
    if not rounds:
        raise SystemExit(f"{task}: active revision {rev} was never evaluated on dev")
    n = len(config(c)["dev"]["layouts"])
    (dst / "dev_rounds.txt").write_text(
        f"dev successes per round ({n} layouts): "
        + " ".join(str(r["successes"]) for r in dev)
        + "\nrevision per round: "
        + " ".join(r["revision"] for r in dev)
        + f"\nfinal active revision (the library in this directory): {rev}, evaluated in round(s) "
        + " ".join(map(str, rounds))
        + (
            ""
            if rounds[-1] == dev[-1]["round"]
            else f"; the last worker's FINDINGS-last.md describes revision {dev[-1]['revision']}, which was not kept"
        )
        + "\n"
    )
    return dict(
        campaign=c.name,
        revision=rev,
        findings=f"{workers[-1].name}/FINDINGS.md",
        dev=[r["successes"] for r in dev],
        active_rounds=rounds,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("campaigns", nargs="+", type=Path)
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--card", type=Path)
    a = ap.parse_args()
    out = a.out.resolve()
    inputs = out / "inputs"
    if inputs.exists():
        raise SystemExit(f"{inputs} exists")
    found = {}
    for c in (c.resolve() for c in a.campaigns):
        for t in config(c)["tasks"]:
            if t in found:
                raise SystemExit(f"{t} in two campaigns")
            found[t] = c
    names = a.tasks or list(found)
    missing = [t for t in names if t not in found]
    if missing:
        raise SystemExit(f"tasks in none of the campaigns: {missing}")
    if "robot" in names:
        raise SystemExit("a task named robot would collide with inputs/robot/")
    layouts = {len(config(found[t])["dev"]["layouts"]) for t in names}
    if len(layouts) != 1:
        raise SystemExit(f"the campaigns differ in their number of dev layouts: {sorted(layouts)}")
    meta = {t: copy_task(found[t], t, inputs / t) for t in names}
    robot = inputs / "robot"
    (robot / "api").mkdir(parents=True)
    for name in ROBOT["api"]:
        shutil.copy(SRC / "pyrualean" / name, robot / "api" / name)
    entry = render(ROBOT, a.card.resolve() if a.card else ROBOT["card"], robot / "system_prompt.md")
    (out / "SOURCES.json").write_text(json.dumps(meta, indent=1) + "\n")
    manifest = {
        "campaigns": sorted({str(found[t]) for t in names}),
        "dev_layouts": layouts.pop(),
        "render": entry,
        "files": {},
    }
    for p in sorted(inputs.rglob("*")):
        if p.is_file():
            manifest["files"][str(p.relative_to(out))] = sha(p)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for t, m in meta.items():
        print(t, m["campaign"], m["revision"], m["dev"], "active in", m["active_rounds"])
    print(f"{out}: {len(manifest['files'])} input files")


if __name__ == "__main__":
    main()
