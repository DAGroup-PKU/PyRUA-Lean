# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Run episodes of a library revision on a split, within hard budgets, and keep the record.

A campaign directory holds ``campaign.json`` (what may be run and the budgets),
``state.json`` (what has been spent; rewritten after every episode, so a restart resumes
rather than starts over) and ``episodes.jsonl`` (one line per episode, appended).  The runner
is the deterministic part of the loop: it never diagnoses a run and never edits a library.

Rules it enforces, not a prompt:

* nothing starts unless ``launch_enabled`` is true;
* a budget (episodes, hours, dollars) is checked before every episode and never exceeded by
  more than the episode in flight;
* a run that never became a judged episode (:func:`.digest.classify` = ``infra``) is recorded,
  retried at most ``max_infra_retries`` times, and never counted as a policy failure;
* the held-out split runs only when asked explicitly, once per revision.

Episodes run through a launcher (``launcher`` in ``campaign.json``) called as
``<launcher> <task> <eval_seed> <layout> <gpu> <tag> <library_view>``; its run directory is
found under ``<runs_root>/<tag>/results/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .digest import digest
from .library import Library

SPLITS = ("dev", "val", "heldout")


class BudgetExhausted(RuntimeError):
    """A campaign budget is spent: nothing more starts."""


class CampaignError(ValueError):
    """A request the campaign's rules refuse."""


@dataclass(frozen=True)
class Cell:
    task: str
    eval_seed: int
    layout: int
    repeat: int = 0

    @property
    def key(self) -> str:
        return f"{self.task}/s{self.eval_seed}/l{self.layout}/r{self.repeat}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Campaign:
    """One campaign directory (see the module docstring)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.config = json.loads((self.root / "campaign.json").read_text())
        for key in ("library", "launcher", "runs_root", "tasks", "splits", "budgets", "gpu"):
            if key not in self.config:
                raise CampaignError(f"campaign.json lacks {key!r}")
        self.library = Library(self.config["library"])
        state = self.root / "state.json"
        self.state: dict[str, Any] = (
            json.loads(state.read_text())
            if state.is_file()
            else {
                "episodes": 0,
                "infra_runs": 0,
                "seconds": 0.0,
                "usd": 0.0,
                "done": {},
                "heldout_revisions": [],
                "created": _now(),
            }
        )
        self.state.setdefault("launch_attempts", 0)
        self.state.setdefault("attempts_by_cell", {})
        self.state.setdefault("in_flight", None)

    # -- record --------------------------------------------------------------

    def _save(self) -> None:
        path = self.root / "state.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True) + "\n")
        tmp.replace(path)

    def _append(self, entry: dict[str, Any]) -> None:
        if entry.get("tag") and any(row.get("tag") == entry["tag"] for row in self.episodes()):
            return
        with open(self.root / "episodes.jsonl", "a", encoding="utf-8") as log:
            log.write(json.dumps(entry, sort_keys=True) + "\n")

    def episodes(self) -> list[dict[str, Any]]:
        path = self.root / "episodes.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines() if x] if path.is_file() else []

    # -- rules ---------------------------------------------------------------

    def cells(self, split: str, tasks: list[str] | None = None, repeats: int = 1) -> list[Cell]:
        """The cells of ``split`` (``campaign.json``: ``{"eval_seed": s, "layouts": [..]}``)."""

        if split not in SPLITS:
            raise CampaignError(f"split must be one of {SPLITS}")
        spec = self.config["splits"][split]
        chosen = tasks or self.config["tasks"]
        unknown = sorted(set(chosen) - set(self.config["tasks"]))
        if unknown:
            raise CampaignError(f"tasks not in this campaign: {unknown}")
        return [
            Cell(task, int(spec["eval_seed"]), int(layout), repeat)
            for task in chosen
            for layout in spec["layouts"]
            for repeat in range(repeats)
        ]

    def check_budget(self) -> None:
        budgets = self.config["budgets"]
        spent = self.state
        cap_attempts = budgets.get(
            "max_online_attempts",
            int(budgets["max_episodes"]) * (1 + int(budgets.get("max_infra_retries", 1))),
        )
        if spent["launch_attempts"] >= cap_attempts:
            raise BudgetExhausted(
                "online launch attempts exhausted (including infrastructure failures)"
            )
        if spent["episodes"] >= int(budgets["max_episodes"]):
            raise BudgetExhausted(f"episodes: {spent['episodes']} of {budgets['max_episodes']}")
        if spent["seconds"] >= float(budgets["max_hours"]) * 3600:
            raise BudgetExhausted(f"hours: {spent['seconds'] / 3600:.2f} of {budgets['max_hours']}")
        cap = budgets.get("max_usd")
        if cap is not None and spent["usd"] >= float(cap):
            raise BudgetExhausted(f"dollars: {spent['usd']:.2f} of {cap}")

    def _done_key(self, revision: str, split: str, cell: Cell) -> str:
        return f"{revision}|{split}|{cell.key}"

    # -- running -------------------------------------------------------------

    def run(
        self,
        revision: str,
        split: str,
        cells: list[Cell],
        *,
        allow_heldout: bool = False,
        launch: Callable[[list[str]], int] | None = None,
        spend: Callable[[], float | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run each cell not yet done for ``revision`` on ``split``; return their digests.

        ``launch`` runs the launcher's argv and returns its exit code (default: a
        subprocess); ``spend`` reads the API key's cumulative spend in dollars, or None when
        it cannot (default: no reading, dollars not tracked).
        """

        if not self.config.get("launch_enabled"):
            raise CampaignError("launch_enabled is false in campaign.json: nothing is started")
        if self.state["in_flight"]:
            raise CampaignError(
                "interrupted launch is unresolved; run the explicit recover command"
            )
        if split not in SPLITS:
            raise CampaignError("unknown split")
        allowed = {(cell.task, cell.eval_seed, cell.layout) for cell in self.cells(split)}
        if any(
            (cell.task, cell.eval_seed, cell.layout) not in allowed or cell.repeat < 0
            for cell in cells
        ):
            raise CampaignError("cell is outside the declared split")
        if len({cell.key for cell in cells}) != len(cells):
            raise CampaignError("duplicate cells in a request")
        if split == "heldout" and any(cell.repeat != 0 for cell in cells):
            raise CampaignError("heldout has one attempt per instance")
        if split == "heldout":
            if not allow_heldout:
                raise CampaignError("the held-out split runs only with allow_heldout")
            if revision in self.state["heldout_revisions"]:
                raise CampaignError(f"held-out was already run on revision {revision}")
        self.library.verify(revision)
        launch = launch or _subprocess_launch(Path(self.config["runs_root"]))
        results = []
        for cell in cells:
            key = self._done_key(revision, split, cell)
            if key in self.state["done"]:
                facts = digest(self.state["done"][key])
                facts["repeat"] = cell.repeat
                results.append(facts)
                continue
            results.append(self._run_cell(revision, split, cell, launch, spend))
        if split == "heldout":
            self.state["heldout_revisions"].append(revision)
            self._save()
        return results

    def adopt(self, revision: str, split: str, cell: Cell, run_dir: str | Path) -> dict[str, Any]:
        """Count a run made outside the runner as ``cell``'s episode, when it is one.

        The run must be a judged episode of ``cell`` (task, eval seed, layout) made with
        ``revision``'s view of the task: its ``result.json`` library hash must be that view's.
        Recorded in ``episodes.jsonl`` as adopted; it counts against the episode budget.
        """

        if split == "heldout":
            raise CampaignError("a held-out episode is never adopted: run it with the runner")
        facts = digest(run_dir)
        if facts["kind"] != "policy":
            raise CampaignError(f"{run_dir}: not a judged episode")
        got = (facts["task"], facts["eval_seed"], facts["layout"])
        if got != (cell.task, cell.eval_seed, cell.layout):
            raise CampaignError(f"{run_dir} is {got}, not {cell.key}")
        from ..play import copy_library

        view = self.library.view(revision, cell.task)
        probe = self.root / ".adopt_probe"
        expected = copy_library(view, probe / "library")["sha256"]
        import shutil

        shutil.rmtree(probe)
        if facts["library"] != expected:
            raise CampaignError(
                f"{run_dir} ran library {facts['library']}, not revision {revision}'s view"
            )
        key = self._done_key(revision, split, cell)
        if key in self.state["done"]:
            raise CampaignError(f"{cell.key} already has an episode for {revision} on {split}")
        self._append(
            {
                "at": _now(),
                "revision": revision,
                "split": split,
                "cell": cell.key,
                "attempt": 0,
                "kind": "policy",
                "adopted": True,
                "run_dir": str(Path(run_dir).resolve()),
                "success": facts["success"],
                "stages": [facts["stages_passed"], facts["stages_total"]],
            }
        )
        self.state["episodes"] += 1
        self.state["done"][key] = str(Path(run_dir).resolve())
        self._save()
        return facts

    def _run_cell(self, revision, split, cell, launch, spend) -> dict[str, Any]:
        view = self.library.view(revision, cell.task)
        retries = int(self.config["budgets"].get("max_infra_retries", 1))
        if split == "heldout":
            retries = 0
        done_key = self._done_key(revision, split, cell)
        start = self.state["attempts_by_cell"].get(done_key, 0)
        for attempt in range(start, retries + 1):
            self.check_budget()
            tag = (
                f"{split}-{revision[:8]}-{cell.task}"
                f"-s{cell.eval_seed}-l{cell.layout}-r{cell.repeat}-a{attempt}"
                f"-{uuid.uuid4().hex[:8]}"
            )
            argv = [
                str(self.config["launcher"]),
                cell.task,
                str(cell.eval_seed),
                str(cell.layout),
                str(self.config["gpu"]),
                tag,
                str(view),
            ]
            usd_before = spend() if spend else None
            started = time.time()
            self.state["launch_attempts"] += 1
            self.state["attempts_by_cell"][done_key] = attempt + 1
            self.state["in_flight"] = {
                "tag": tag,
                "key": done_key,
                "started": started,
                "revision": revision,
                "split": split,
                "cell": cell.key,
            }
            self._save()
            code = launch(argv)
            elapsed = time.time() - started
            usd_after = spend() if spend else None
            usd = (
                max(0.0, usd_after - usd_before)
                if usd_before is not None and usd_after is not None
                else None
            )
            self.state["seconds"] += elapsed
            self.state["in_flight"] = None
            if usd is not None:
                self.state["usd"] += usd
            run_dir = _find_run(Path(self.config["runs_root"]) / tag)
            facts = digest(run_dir) if run_dir else None
            kind = facts["kind"] if facts else "infra"
            entry = {
                "at": _now(),
                "tag": tag,
                "revision": revision,
                "split": split,
                "cell": cell.key,
                "attempt": attempt,
                "kind": kind,
                "launcher_exit": code,
                "run_dir": str(run_dir) if run_dir else None,
                "seconds": round(elapsed, 1),
                "usd_key_delta": usd,
                "success": facts["success"] if facts else None,
                "stages": [facts["stages_passed"], facts["stages_total"]] if facts else None,
            }
            self._append(entry)
            if kind == "policy":
                self.state["episodes"] += 1
                facts["repeat"] = cell.repeat
                self.state["done"][self._done_key(revision, split, cell)] = str(run_dir)
                self._save()
                return facts
            self.state["infra_runs"] += 1
            self._save()
        raise CampaignError(
            f"{cell.key}: no judged episode after {retries + 1} attempts (infrastructure); "
            "see episodes.jsonl"
        )


def _find_run(tag_dir: Path) -> Path | None:
    """The one run directory a launcher call produced under ``<tag>/results/``."""

    results = tag_dir / "results"
    runs = sorted(p for p in results.iterdir() if p.is_dir()) if results.is_dir() else []
    if len(runs) > 1:
        raise CampaignError(f"{results} holds {len(runs)} runs: a tag must be used once")
    if not runs or not (runs[0] / "result.json").is_file():
        return None
    return runs[0]


def _subprocess_launch(runs_root: Path) -> Callable[[list[str]], int]:
    """Run the launcher, its output in ``<runs_root>/<tag>.log`` (the tag is argv[5]).

    The launcher is told where to put the run (``EVOLVE_RUNS_ROOT``): the runner looks for
    it under ``<runs_root>/<tag>/results/``.
    """

    def launch(argv: list[str]) -> int:
        runs_root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, EVOLVE_RUNS_ROOT=str(runs_root))
        with open(runs_root / f"{argv[5]}.log", "a", encoding="utf-8") as out:
            out.write(f"[{_now()}] {' '.join(argv)}\n")
            out.flush()
            return subprocess.call(
                argv, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )

    return launch


def key_spend(env_var: str, base_url: str, model: str) -> Callable[[], float | None]:
    """A reader of the key's cumulative spend (``x-litellm-key-spend``), None on failure.

    An upper bound for one campaign's cost: anything else using the key is counted too.
    The key is read from the environment and never logged.
    """

    def read() -> float | None:
        import httpx

        key = os.environ.get(env_var)
        if not key:
            return None
        try:
            reply = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ok"}],
                    "max_completion_tokens": 1,
                },
                timeout=30,
            )
            return float(reply.headers["x-litellm-key-spend"])
        except Exception:  # noqa: BLE001 - dollars are then simply not tracked
            return None

    return read


def summary(campaign: Campaign) -> dict[str, Any]:
    """What a campaign has spent and run, per (revision, split, task)."""

    table: dict[str, dict[str, Any]] = {}
    for entry in campaign.episodes():
        task = entry["cell"].split("/")[0]
        row = table.setdefault(
            f"{entry['revision'][:8]} {entry['split']} {task}",
            {"episodes": 0, "successes": 0, "infra": 0},
        )
        if entry["kind"] == "policy":
            row["episodes"] += 1
            row["successes"] += int(bool(entry["success"]))
        else:
            row["infra"] += 1
    spent = {k: campaign.state[k] for k in ("episodes", "infra_runs", "seconds", "usd")}
    return {"spent": spent, "budgets": campaign.config["budgets"], "by_revision": table}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyrualean.evolve.campaign", description=__doc__
    )
    parser.add_argument("campaign", type=Path)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a revision on a split")
    run.add_argument("--revision", default=None, help="default: the library's active revision")
    run.add_argument("--split", choices=SPLITS, required=True)
    run.add_argument("--tasks", nargs="*", default=None)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--allow-heldout", action="store_true")
    sub.add_parser("summary", help="what the campaign has spent and run")
    args = parser.parse_args(argv)
    campaign = Campaign(args.campaign)
    if args.cmd == "summary":
        sys.stdout.write(json.dumps(summary(campaign), indent=1) + "\n")
        return 0
    revision = args.revision or campaign.library.active()
    if revision is None:
        raise CampaignError("the library has no active revision: pass --revision")
    cfg = campaign.config
    spend = (
        key_spend(cfg["usd_key_env"], cfg["base_url"], cfg["model"])
        if cfg.get("usd_key_env")
        else None
    )
    results = campaign.run(
        revision,
        args.split,
        campaign.cells(args.split, args.tasks, args.repeats),
        allow_heldout=args.allow_heldout,
        spend=spend,
    )
    for facts in results:
        stages = f"{facts['stages_passed']}/{facts['stages_total']}"
        sys.stdout.write(
            f"{facts['task']} l{facts['layout']}: success={facts['success']} stages={stages}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
