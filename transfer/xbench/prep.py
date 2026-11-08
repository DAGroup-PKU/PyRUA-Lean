"""Build inputs/ for a cross-benchmark transfer: a distilled RoboDojo library -> LIBERO-PRO
(libero_rgb) or RoboCasa365 (robocasa_rgb).

    python transfer/xbench/prep.py SOURCE OUT_DIR [--target libero|robocasa]
        [--source-card CARD.json] [--target-card CARD.json]

SOURCE is a frozen RoboDojo library (a DEST of transfer/freeze.sh: library-v1/, NOTES.md,
SOURCES.json); TARGET is libero (default) or robocasa.  Writes OUT_DIR/inputs/ and
OUT_DIR/MANIFEST.json:

- source/library/   the source library-v1 (lessons/, skills/) as is; source/NOTES.md, SOURCES.json
- source/system_prompt.md   what a RoboDojo policy reads besides library and task card, rendered by
  render.py with the RoboDojo evaluation's settings (robodojo_xpl_rgb, no-vla, on-demand images,
  pure feedback, 5400 s) and an empty library
- source/api/       robodojo_xpl_rgb.py, robodojo.py (its base), _robot.py
- target/system_prompt.md   the same for LIBERO-PRO with eval_libero's settings (libero_rgb,
  no-vla, on-demand, pure, 7200 s)
- target/api/       libero_rgb.py, _robot.py
  (robocasa: target/system_prompt.md with eval_robocasa's settings: robocasa_rgb, no-vla,
  on-demand, pure, 7200 s, and an atomic task's card, so its native-action budget is that task's
  horizon; target/api/ robocasa_rgb.py, _robot.py)

Everything is rendered from this repository's src/ with the running interpreter.  The cards (the
default ones are in cards/) only give the robot and the native-action budget: the task card itself
is cut off the prompt.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
EMPTY = (
    HERE.parents[1] / "eval_libero/libraries/empty"
)  # lessons/none.md, empty: the prompt gets LIBRARY_NOTE, no lessons
BENCH = {
    "source": {
        # any RoboDojo card: only max_env_steps is used, and the card is cut off the prompt
        "card": HERE / "cards/robodojo_xpl_rgb.json",
        "args": [
            "--robot",
            "robodojo_xpl_rgb",
            "--primitives",
            "no-vla",
            "--images",
            "on-demand",
            "--feedback",
            "pure",
            "--budget-s",
            "5400",
        ],
        "api": ["robodojo_xpl_rgb.py", "robodojo.py", "_robot.py"],
    },
}
TARGETS = {
    "libero": {
        "card": HERE / "cards/libero_rgb.json",
        "args": [
            "--robot",
            "libero_rgb",
            "--primitives",
            "no-vla",
            "--images",
            "on-demand",
            "--feedback",
            "pure",
            "--budget-s",
            "7200",
        ],
        "api": ["libero_rgb.py", "_robot.py"],
    },
    "robocasa": {
        # an atomic task (CloseBlenderLid): its horizon is the budget the prompt shows
        "card": HERE / "cards/robocasa_rgb.json",
        "args": [
            "--robot",
            "robocasa_rgb",
            "--primitives",
            "no-vla",
            "--images",
            "on-demand",
            "--feedback",
            "pure",
            "--budget-s",
            "7200",
        ],
        "api": ["robocasa_rgb.py", "_robot.py"],
    },
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def render(b, card, out):
    """Write out (system_prompt.md) for bench settings b; returns its MANIFEST entry."""
    cmd = [
        sys.executable,
        "-I",
        str(HERE / "render.py"),
        "--src",
        str(SRC),
        "--card",
        str(card),
        *b["args"],
        "--library",
        str(EMPTY),
        "--out",
        str(out),
    ]
    print(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip())
    return {
        "python": sys.executable,
        "src": str(SRC),
        "card": str(card),
        "args": b["args"],
        "library": str(EMPTY),
    }


def main(source, out, target, cards):
    lib = source / "library-v1"
    if not lib.is_dir():
        raise SystemExit(f"{lib} is not a directory")
    inputs = out / "inputs"
    if inputs.exists():
        raise SystemExit(f"{inputs} exists")
    manifest = {
        "source": source.name,
        "source_library": str(lib),
        "target": target,
        "files": {},
        "render": {},
    }
    dst = inputs / "source"
    shutil.copytree(lib, dst / "library", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("NOTES.md", "SOURCES.json"):
        shutil.copy(source / name, dst / name)
    for side, b in {**BENCH, "target": TARGETS[target]}.items():
        d = inputs / side
        (d / "api").mkdir(parents=True, exist_ok=True)
        for name in b["api"]:
            shutil.copy(SRC / "pyrualean" / name, d / "api" / name)
        manifest["render"][side] = render(b, cards[side] or b["card"], d / "system_prompt.md")
    for p in sorted(inputs.rglob("*")):
        if p.is_file():
            manifest["files"][str(p.relative_to(out))] = sha(p)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{out}: {len(manifest['files'])} input files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--target", choices=sorted(TARGETS), default="libero")
    ap.add_argument("--source-card", type=Path)
    ap.add_argument("--target-card", type=Path)
    a = ap.parse_args()
    main(
        a.source.resolve(),
        a.out.resolve(),
        a.target,
        {
            "source": a.source_card and a.source_card.resolve(),
            "target": a.target_card and a.target_card.resolve(),
        },
    )
