"""The worker's replay command: saved cells on a dev layout, against the candidate, no model.

``python -m pyrualean.learn.replay CELLS_DIR --layout N`` (inside a worker session; the session
sets ``PYRUALEAN_LEARN_SESSION``).  The cells run in order through the same toolkit, sandbox
and benchmark verdict as a model's; the candidate library in the session is frozen into a
scratch view so ``from library.<module> import ...`` imports the candidate's skills.

Limits are enforced here, not by the worker's prompt: dev layouts only, at most
``max_replays`` per session and ``max_replays_per_layout`` per layout, and every replay
counts against the campaign's launch budget (:class:`.runner.Runner`).

With the session's ``scoring`` off the run itself goes outside the session
(``ROOT/.replays/``); the session gets its filtered copy (:func:`.worker.mirror_run`) and the
printed digest and ``replays.json`` give only success.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from ..evolve.digest import digest, render
from ..evolve.library import _live_cards, validate
from .config import load
from .runner import BudgetExhausted, Runner
from .worker import evidence_files, mirror_run


def candidate_view(candidate: Path, task: str, target: Path) -> Path:
    """The flat ``skills/`` + ``lessons/`` view of a candidate tree, as an episode gets it."""

    problems = validate(candidate, allow_empty=True)
    if problems:
        raise ValueError("candidate does not validate: " + "; ".join(problems))
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for kind in ("skills", "lessons"):
        for scope in (candidate / "shared" / kind, candidate / "tasks" / task / kind):
            for path in sorted(scope.glob("*")) if scope.is_dir() else []:
                (target / kind).mkdir(exist_ok=True)
                if kind == "lessons":  # retired cards are not shown, as in Library.view
                    (target / kind / path.name).write_text(_live_cards(path.read_text()))
                else:
                    shutil.copyfile(path, target / kind / path.name)
    if not any(target.rglob("*.*")):
        (target / "lessons").mkdir()
        (target / "lessons" / "none.md").write_text("")
    return target


def replay(session: Path, cells: Path, layout: int, *, runner: Runner | None = None) -> str:
    spec = json.loads((session / "session.json").read_text())
    config = load(spec["config"])
    task = spec["task"]
    if layout not in config["dev"]["layouts"]:
        raise ValueError(f"layout {layout} is not a dev layout {config['dev']['layouts']}")
    cells = cells.resolve()
    if not cells.is_relative_to(session.resolve()):
        raise ValueError("replay cells must be under the session directory")
    used_path = session / "replays.json"
    used = json.loads(used_path.read_text()) if used_path.exists() else []
    limits = config["worker"]
    if len(used) >= limits["max_replays"]:
        raise BudgetExhausted(f"replay budget of this session spent ({limits['max_replays']})")
    if sum(1 for u in used if u["layout"] == layout) >= limits["max_replays_per_layout"]:
        raise BudgetExhausted(f"replays of layout {layout} spent")
    number = len(used)
    while (session / "replays" / f"{number:02d}").exists():  # left by an interrupted attempt
        number += 1
    snapshot = session / "replays" / f"{number:02d}"
    shutil.copytree(cells, snapshot / "cells")
    view = candidate_view(session / "candidate", task, snapshot / "view")
    blind = spec.get("scoring", "on") == "off"
    output_root = snapshot
    if blind:  # the run's benchmark tree and verdict stay where the worker cannot read them
        output_root = (
            Path(spec["campaign"]) / ".replays" / task / session.name / "replays" / snapshot.name
        )
    runner = runner or Runner(spec["campaign"], config)
    entry = runner.episode(
        split="dev",
        task=task,
        revision=f"replay-{spec['round']}",
        layout=layout,
        view=view,
        replay_dir=snapshot / "cells",
        output_root=output_root,
    )
    run_dir = shown = Path(entry["run_dir"]) if entry is not None else None
    record = entry
    if blind and entry is not None:
        shown = snapshot / run_dir.parent.name / "episode"
        mirror_run(run_dir, shown, evidence_files(config))
        record = {k: v for k, v in entry.items() if k not in ("stages", "score")}
        record["run_dir"] = str(shown)
    used.append({"layout": layout, "cells": str(cells), "entry": record})
    used_path.write_text(json.dumps(used, indent=1) + "\n")
    if entry is None:
        return "INFRASTRUCTURE: the replay never became a judged episode (retries spent).\n"
    text = render(digest(run_dir), scoring=not blind)
    results = run_dir / "replay_results.json"
    if results.exists():
        for item in json.loads(results.read_text()):
            lines = [f"\n--- {item.get('cell')}"]
            for key in ("stdout", "error", "exception"):
                if item.get(key):
                    lines.append(f"{key}: {str(item[key])[-3000:]}")
            text += "\n".join(lines) + "\n"
    return text + f"\nrun dir (frames under steps/): {shown}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyrualean.learn.replay", description=__doc__)
    parser.add_argument("cells", type=Path, help="directory of cell_NNN.py under the session")
    parser.add_argument("--layout", type=int, required=True, help="dev layout index")
    args = parser.parse_args(argv)
    session = os.environ.get("PYRUALEAN_LEARN_SESSION")
    if not session:
        sys.stderr.write("PYRUALEAN_LEARN_SESSION is not set: run inside a worker session\n")
        return 2
    try:
        sys.stdout.write(replay(Path(session), args.cells, args.layout))
    except (ValueError, BudgetExhausted) as exc:
        sys.stderr.write(f"refused: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
