"""Build the inputs of a RoboDojo cross-task distillation from finished learn campaigns.

    python transfer/robodojo/prep.py OUT_DIR CAMPAIGN_ROOT [CAMPAIGN_ROOT ...] [--tasks T [T ...]]

Each source task's inputs are its campaign's active library revision as the policy read it (its
view: skills/, lessons/ without retired cards), the last worker's FINDINGS.md and its dev
successes per round.  The source tasks are all tasks of the given campaigns, or those named by
--tasks; a task may be in only one of the campaigns.  Writes OUT_DIR/inputs/<task>/,
OUT_DIR/SOURCES.json (per task: campaign, revision, findings, dev) and OUT_DIR/MANIFEST.json
(campaign roots, number of dev layouts, sha256 of every input).  Use the campaigns of one setting
(all --worker-scoring off, or all on).
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def config(campaign):
    return json.loads((campaign / "config.json").read_text())


def sources(campaigns):
    found = {}
    for c in campaigns:
        for t in config(c)["tasks"]:
            if t in found:
                raise SystemExit(f"{t} in both {found[t]} and {c}")
            found[t] = c
    return found


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
    n = len(config(c)["dev"]["layouts"])
    (dst / "dev_rounds.txt").write_text(
        f"dev successes per round ({n} layouts): "
        + " ".join(str(r["successes"]) for r in dev)
        + f"\nfinal active revision: {rev}\n"
    )
    return dict(
        campaign=c.name,
        revision=rev,
        findings=f"{workers[-1].name}/FINDINGS.md",
        dev=[r["successes"] for r in dev],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("campaigns", nargs="+", type=Path)
    ap.add_argument("--tasks", nargs="+")
    a = ap.parse_args()
    out = a.out.resolve()
    campaigns = [c.resolve() for c in a.campaigns]
    found = sources(campaigns)
    names = sorted(a.tasks or found)
    missing = [t for t in names if t not in found]
    if missing:
        raise SystemExit(f"tasks in none of the campaigns: {missing}")
    layouts = {len(config(found[t])["dev"]["layouts"]) for t in names}
    if len(layouts) != 1:
        raise SystemExit(f"the campaigns differ in their number of dev layouts: {sorted(layouts)}")
    d = out / "inputs"
    if d.exists():
        raise SystemExit(f"{d} exists")
    d.mkdir(parents=True)
    meta = {t: copy_task(found[t], t, d / t) for t in names}
    (out / "SOURCES.json").write_text(json.dumps(meta, indent=1) + "\n")
    manifest = {
        "campaigns": sorted({str(found[t]) for t in names}),
        "dev_layouts": layouts.pop(),
        "files": {},
    }
    for p in sorted(d.rglob("*")):
        if p.is_file():
            manifest["files"][str(p.relative_to(out))] = sha(p)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for t, m in meta.items():
        print(t, m["campaign"], m["revision"], m["dev"])
    print(f"{d}: {len(names)} tasks, {len(manifest['files'])} input files")


if __name__ == "__main__":
    main()
