"""Write DIR/BRIEF.md (DIR made by prep.py) from BRIEF.wo_ce.md or BRIEF.w_ce.md: only the number of
source tasks, their list, the number of dev layouts, the API source path and the working
directory are filled in.

    python transfer/robodojo/brief.py DIR --setting wo_ce|w_ce

wo_ce: the source campaigns ran with --worker-scoring off; w_ce: with --worker-scoring on.  The
two templates differ in one passage on what the learners saw."""

import argparse
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parents[1] / "src" / "pyrualean"
WORD = dict(
    enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
)
WORD.update({30: "thirty", 40: "forty", 50: "fifty"})

ap = argparse.ArgumentParser()
ap.add_argument("dir", type=Path)
ap.add_argument("--setting", choices=("wo_ce", "w_ce"), required=True)
a = ap.parse_args()
d = a.dir.resolve()
text = (HERE / f"BRIEF.{a.setting}.md").read_text()
tasks = sorted(json.loads((d / "SOURCES.json").read_text()), key=str.lower)
n = len(tasks)
word = WORD.get(n, str(n))
listing = ", ".join(f"`{t}`" for t in tasks)
para = textwrap.fill(f"For each of the {word} source tasks ({listing}):", 96)
dev = json.loads((d / "MANIFEST.json").read_text())["dev_layouts"]
for key, value in (
    ("{TASKS}", para),
    ("{N_WORD}", word),
    ("{N_DEV}", str(dev)),
    ("{API}", f"{API}/"),
    ("{DIR}", f"{d}/"),
):
    assert text.count(key) == 1, key
    text = text.replace(key, value)
(d / "BRIEF.md").write_text(text)
print(f"wrote {d / 'BRIEF.md'} ({len(text)} chars, {n} source tasks)")
