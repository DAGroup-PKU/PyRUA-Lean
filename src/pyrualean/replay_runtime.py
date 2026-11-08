# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Replay saved cells without a model: the fourth runtime of :mod:`pyrualean.play`.

``--runtime replay`` runs the ``cell_NNN.py`` files of a directory (a cells episode's
``workspace/``, or cells written by hand) through the same :class:`~pyrualean.arms.ArmToolkit`
an agent calls, in order, then calls ``finish``.  Boot, sandbox, ledger, budgets and the
benchmark's verdict are the agent's; only the author of the cells differs.  It is how a
fix to a skill or a lesson is checked on the layout it failed on before it goes into a
library, at no model cost.

The cells are not adapted to the scene: a replay on another layout than the one the cells
were written on runs the same code against a different scene.  Code that re-locates what it
manipulates (``robo.locate``) carries over; code with numbers read off one scene does not.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .arms import ArmToolkit

#: The cells of an episode, in order (``arms.ArmToolkit._python`` writes them).
CELL_PATTERN = re.compile(r"cell_(\d{3,})\.py")


def replay_cells(source: str | Path) -> list[Path]:
    """The ``cell_NNN.py`` files of ``source`` in execution order; refuses an empty one."""

    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"replay directory not found: {root}")
    cells = sorted(
        (p for p in root.iterdir() if CELL_PATTERN.fullmatch(p.name)),
        key=lambda p: int(CELL_PATTERN.fullmatch(p.name).group(1)),  # type: ignore[union-attr]
    )
    if not cells:
        raise FileNotFoundError(f"no cell_NNN.py files under {root}")
    return cells


def run_replay_agent(
    arm_toolkit: ArmToolkit,
    *,
    source: str | Path,
    budget_s: float,
    out_dir: str | Path,
    arm: str = "cells",
    on_timeout: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the saved cells of ``source`` in order; returns the ``generation`` record.

    Stops after the last cell, once ``should_stop`` answers true (the benchmark's client
    has ended the episode, e.g. the task was solved), or when ``budget_s`` is spent
    (``on_timeout`` is called first).  Every cell's result goes to ``replay_results.json``.
    ``status`` is ``completed`` or ``timeout``, as for the other runtimes.
    """

    if arm != "cells" or arm_toolkit.arm != "cells":
        raise ValueError("--runtime replay replays cells: run it with --arm cells")
    cells = replay_cells(source)
    out_dir = Path(out_dir)
    started = time.perf_counter()
    status, returncode = "completed", 0
    results: list[dict[str, Any]] = []
    ran = 0
    stopped_by = "end of cells"
    for path in cells:
        if should_stop is not None and should_stop():
            stopped_by = "episode over"
            break
        if time.perf_counter() - started > budget_s:
            status, returncode = "timeout", None
            stopped_by = "timeout"
            if on_timeout is not None:
                on_timeout()
            break
        reply = arm_toolkit.execute_tool("python", {"code": path.read_text(encoding="utf-8")})
        ran += 1
        results.append({"cell": path.name, **reply.result})
        # a cell that raised leaves the scene as the cells after it did not expect: stop there
        # (the toolkit reports a raised cell as "exception", a refused call as "error")
        hit = next(
            (k for k in ("episode_finished", "error", "exception") if reply.result.get(k)), None
        )
        if hit is not None:
            stopped_by = f"{hit} in {path.name}"
            break
    if status == "completed" and arm_toolkit.finish is None:
        arm_toolkit.execute_tool(
            "finish",
            {
                "status": "failure",
                "summary": f"replayed {ran} of {len(cells)} cells ({stopped_by})",
            },
        )
    (out_dir / "replay_results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return {
        "runtime": "replay",
        "model": None,
        "models": [],
        "source": str(Path(source).expanduser().resolve()),
        "cells": [p.name for p in cells],
        "cells_run": ran,
        "stopped_by": stopped_by,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
        "turns": len(arm_toolkit.turns),
        "tool_calls": {"python": ran},
        "errors": [],
        "warnings": [],
        "max_turns_hit": False,
        "status": status,
        "returncode": returncode,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


__all__ = ["CELL_PATTERN", "replay_cells", "run_replay_agent"]
