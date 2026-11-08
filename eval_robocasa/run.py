"""RoboCasa365 evaluation of robocasa_rgb on the PyRUA-Lean paper's cells (resumable).

``python eval_robocasa/run.py ROOT init --config CONFIG``   freeze the config
``python eval_robocasa/run.py ROOT run [--tasks ...]``   episodes not judged yet
``python eval_robocasa/run.py ROOT report [--with ROOT ...]``   successes, paired with the paper

The scheduling, ledger and bookkeeping are ``eval/run.py``'s :class:`Evaluation` (imported, not
changed): at most ``concurrency`` episodes at once on ``gpus`` (``per_gpu``, ``need_gib``,
``stagger_seconds``); an episode without a verdict is infrastructure, retried at most
``budgets.max_infra_retries`` times, then missing; ``pause_after_provider_failures`` provider
failures in a row pause the run.

An evaluation task is a RoboCasa task name, its layout ``n`` the scene seed
``seeds_of(config, task)[n]`` (``seed_overrides`` replaces ``seeds`` for a task: the paper ran
CoffeeSetupMug on seed 6 instead of 4).  Every task has every seed, or only those
``only_seeds[task]`` names (a retest of some cells); no layout replaces another.
RoboCasa judges by the task's ``_check_success`` after every native action: there is no verdict
record and no stages, ``environment_success`` is the outcome.  There is no skill library (the
paper's runs had none, and a library adds a note to the prompt), unless ``arms`` (``{name:
{library: DIR}}``) names some: then every task runs once per arm as ``<arm>/<task>``, with the
arm's library copied by ``init`` to ``libraries/<arm>/<task>`` (its hash checked against every
episode), and the report adds each arm's successes and its pairing with ``v0``, the episodes of
roots without arms (``report --with`` a library-free root of the same identity: the library is
not part of the identity).

One root runs on one machine (``gpus`` are its local GPUs); an evaluation spread over machines is
one root per machine, each with its share of the tasks and the same settings, and
``report --with`` adds the other roots.  ``reference`` names the paper's run directories
(``*-cells-<split>-<Task>-s<seed>/result.json``): the report pairs every judged episode with the
paper's outcome on the same task and seed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robocasa_episode import (  # noqa: E402
    identity,
    library_sha256,
    seeds_of,
    split_task,
)


def _load_robodojo_run() -> Any:
    """``eval/run.py`` under its own module name (its ``episode`` is eval/episode.py)."""

    spec = importlib.util.spec_from_file_location(
        "robodojo_eval_run", HERE.parent / "eval" / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_robodojo_run()
Stop = base.Stop
PROVIDER_FAILURE, INFRA_STATUSES = base.PROVIDER_FAILURE, base.INFRA_STATUSES

#: Settings ``init`` requires in ``execution``.
EXECUTION_FIELDS = (
    "robot", "primitives", "split", "runtime", "model", "extra_body", "images", "feedback",
    "max_cells", "max_model_requests", "max_tokens", "nudges", "provider_retries",
    "policy_seconds", "max_episode_steps", "base_url", "api_key_env",
)  # fmt: skip
PATH_FIELDS = ("python", "lean", "sim_python", "sim_packages", "assets", "sam3_checkpoint")
#: A paper run directory: ``<prefix>-cells-<split>-<Task>-s<seed>``.
REFERENCE_RUN = re.compile(r"-cells-(?P<split>[a-z]+)-(?P<task>[A-Za-z]+)-s(?P<seed>\d+)$")


class ConfigError(ValueError):
    """A configuration that cannot run."""


def check(config: dict[str, Any]) -> None:
    seeds, names = config.get("seeds"), config.get("task_names")
    if not seeds or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) for s in seeds):
        raise ConfigError("seeds: a nonempty list of distinct integers")
    if not names or len(set(names)) != len(names) or not all(n.isalnum() for n in names):
        raise ConfigError("task_names: a nonempty list of distinct RoboCasa task names")
    for task, override in (config.get("seed_overrides") or {}).items():
        if task not in names:
            raise ConfigError(f"seed_overrides.{task}: not in task_names")
        if len(override) != len(seeds) or len(set(override)) != len(override):
            raise ConfigError(f"seed_overrides.{task}: {len(seeds)} distinct seeds")
    for task, only in (config.get("only_seeds") or {}).items():
        if task not in names:
            raise ConfigError(f"only_seeds.{task}: not in task_names")
        allowed = (config.get("seed_overrides") or {}).get(task, seeds)
        if not only or len(set(only)) != len(only) or not set(only) <= set(allowed):
            raise ConfigError(f"only_seeds.{task}: distinct seeds among {allowed}")
    execution = config.get("execution") or {}
    missing = [field for field in EXECUTION_FIELDS if field not in execution]
    if missing:
        raise ConfigError(f"execution: missing {missing}")
    if execution["robot"] != "robocasa_rgb":
        raise ConfigError("execution.robot: robocasa_rgb (the colour primitives)")
    if execution["runtime"] == "codex" and not execution["base_url"]:
        if not (execution.get("codex_bin") and execution.get("codex_home")):
            raise ConfigError("runtime codex without base_url: codex_bin and codex_home log in")
    elif not (execution["base_url"] and execution["api_key_env"]):
        raise ConfigError("execution.base_url and execution.api_key_env name the endpoint")
    paths = config.get("paths") or {}
    for field in PATH_FIELDS:
        if not paths.get(field) or not Path(paths[field]).exists():
            raise ConfigError(f"paths.{field} does not exist")
    for part in ("objects", "fixtures"):
        if not (Path(paths["assets"]) / part).is_dir():
            raise ConfigError(f"paths.assets has no {part}/")
    for field in ("concurrency", "per_gpu", "need_gib", "gpus", "budgets", "sam3_sha256"):
        if not config.get(field):
            raise ConfigError(f"{field} is required")
    for field in ("max_launches", "max_hours", "max_infra_retries", "episode_watchdog_seconds"):
        if not isinstance(config["budgets"].get(field), (int, float)):
            raise ConfigError(f"budgets.{field} is required")
    if config["budgets"]["episode_watchdog_seconds"] <= execution["policy_seconds"]:
        raise ConfigError("the watchdog must leave the episode its policy wall clock and boot")
    for directory in (config.get("reference") or {}).get("dirs", []):
        if not Path(directory).is_dir():
            raise ConfigError(f"reference dir {directory} does not exist")
    arms = config.get("arms")
    if arms is not None:
        if not arms or any(not arm.isidentifier() or arm == "v0" for arm in arms):
            raise ConfigError("arms: {name: {library}}, names identifiers other than v0")
        for arm, spec in arms.items():
            if not Path(spec.get("library", "")).is_dir():
                raise ConfigError(f"arms.{arm}.library is not a directory")


def init(root: Path, config_path: Path, *, sam3_hash: Callable[[Path], str] | None = None) -> dict:
    """Freeze ``config_path`` into ``root``: identity, task order, the lean repo's commit."""

    config = json.loads(Path(config_path).read_text())
    check(config)
    root = Path(root)
    if (root / "config.json").exists():
        raise ConfigError(f"{root} is initialised already")
    checkpoint = Path(config["paths"]["sam3_checkpoint"])
    found = (sam3_hash or _sha256_file)(checkpoint)
    if found != config["sam3_sha256"]:
        raise ConfigError(
            f"{checkpoint}: sha256 {found[:12]}, the config says {config['sam3_sha256'][:12]}"
        )
    names = config["task_names"]
    if config.get("arms"):
        # every task once per arm, the arms of a task side by side in the order
        tasks, libraries, order = {}, {}, []
        for task in names:
            for arm, spec in config["arms"].items():
                name = f"{arm}/{task}"
                source = Path(spec["library"]).resolve()
                view = root / "libraries" / name
                for kind, pattern in (("skills", "*.py"), ("lessons", "*.md")):
                    for path in (
                        sorted((source / kind).glob(pattern)) if (source / kind).is_dir() else []
                    ):
                        (view / kind).mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, view / kind / path.name)
                sha = library_sha256(view)
                if sha != library_sha256(source):
                    raise RuntimeError(f"{name}: the copied library differs from {source}")
                tasks[name] = {"layouts": len(seeds_of(config, task))}
                libraries[name] = {
                    "sha256": sha, "source": str(source), "revision": arm, "source_test": None,
                }  # fmt: skip
                order.append(name)
    else:
        tasks = {task: {"layouts": len(seeds_of(config, task))} for task in names}
        # no library: the base class only reads these fields
        libraries = {
            task: {"sha256": None, "source": None, "revision": "none", "source_test": None}
            for task in names
        }
        order = list(names)
    frozen = {
        **config,
        "created": base._now(),
        "eval_seed": None,
        "tasks": tasks,
        "libraries": libraries,
        "order": order,
        "lean_commit": _git_head(Path(config["paths"]["lean"])),
        "identity": identity(config),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(frozen, indent=1) + "\n")
    return frozen


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repo: Path) -> dict[str, Any] | None:
    """The commit and whether the tree is clean (informational; the identity hashes the code)."""

    env = {
        **os.environ,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "*",
    }
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, env=env,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", "src"],
            capture_output=True, text=True, check=True, env=env,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"head": head, "src_clean": not dirty}


def classify(result: dict[str, Any]) -> str:
    """``policy``: an episode RoboCasa judged (its predicate ran to the end); ``infra``: one it
    never could."""

    # the kitchen ended the episode (solved, or its native-action budget spent): judged,
    # whatever the planner did after (a provider failure then is not a reason to run it again)
    if (result.get("verdict") or {}).get("over"):
        return "policy"
    if result.get("status") in INFRA_STATUSES or result.get("status") == PROVIDER_FAILURE:
        return "infra"
    if (result.get("generation") or {}).get("status") == "error":
        return "infra"
    return "policy"


def sign_test(wins: int, losses: int) -> float | None:
    """Two-sided exact sign test on the discordant pairs."""

    n = wins + losses
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def reference_outcomes(config: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """The paper's outcome per (task, seed), from ``reference.dirs``."""

    split = config["execution"]["split"]
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for directory in (config.get("reference") or {}).get("dirs", []):
        for run_dir in sorted(Path(directory).iterdir()):
            match = REFERENCE_RUN.search(run_dir.name)
            if not match or match["split"] != split or not (run_dir / "result.json").is_file():
                continue
            key = (match["task"], int(match["seed"]))
            if key in found:
                raise RuntimeError(f"two paper runs of {key}: {found[key]['run']} and {run_dir}")
            result = json.loads((run_dir / "result.json").read_text())
            found[key] = {
                "success": bool(result.get("environment_success")),
                "tool_turns": result.get("tool_turns"),
                "run": str(run_dir),
            }
    return found


class RobocasaEvaluation(base.Evaluation):
    def __init__(self, root: str | Path, *, launch: Any = None, gpus: Any = None) -> None:
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / "config.json").read_text())
        if identity(self.config) != self.config["identity"]:
            raise RuntimeError("code, benchmark or execution settings changed since init")
        self.launch = launch or self.subprocess_launch
        self.gpus = gpus or base.Gpus(self.config)
        self.lock = base.threading.Lock()

    def subprocess_launch(
        self, output: Path, env: dict[str, str], timeout_s: float, started: Callable[[int], None]
    ) -> int:
        """The episode in a fresh process, killed with its tree (kitchen, SAM3) at the watchdog."""

        command = [
            self.config["paths"]["python"],
            str(HERE / "robocasa_episode.py"),
            str(output / "request.json"),
        ]
        with (output / "episode.log").open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=output,
                start_new_session=True,
            )
            started(process.pid)
            try:
                return process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                base.stop_tree(process)
                (output / "watchdog.json").write_text(json.dumps({"killed_after_s": timeout_s}))
                return 124
            except BaseException:
                base.stop_tree(process)
                raise

    def _environment(self, output: Path) -> dict[str, str]:
        paths = self.config["paths"]
        cache = Path(paths["lean"]) / ".cache"
        env = base.environment(self.config)
        # robosuite refuses an EGL device outside a CUDA_VISIBLE_DEVICES mask, and SAM3 pins
        # the request's physical GPU only when there is none
        env.pop("CUDA_VISIBLE_DEVICES", None)
        tmp = output / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env.update(
            TMPDIR=str(tmp),
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1",
            PYTHONPATH=str(Path(paths["lean"]) / "src"),
            SAM3_CHECKPOINT_PATH=paths["sam3_checkpoint"],
            XDG_CACHE_HOME=str(cache / "xdg"),
            HF_HOME=str(cache / "huggingface"),
            TORCH_HOME=str(cache / "torch"),
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
        )
        return env

    def _entry(
        self,
        key: str,
        request: dict[str, Any],
        output: Path,
        code: int,
        seconds: float,
        attempt: int,
    ) -> dict[str, Any]:
        run_dir = output / "episode"
        task = request["task"]
        entry: dict[str, Any] = {
            "at": base._now(),
            "key": key,
            "task": task,
            "arm": split_task(task)[0],
            "layout": request["layout"],
            "seed": seeds_of(self.config, task)[int(request["layout"])],
            "eval_seed": None,
            "gpu": request["gpu"],
            "attempt": attempt,
            "exit": code,
            "seconds": round(seconds, 1),
            "run_dir": str(run_dir),
        }
        path = run_dir / "result.json"
        result = json.loads(path.read_text()) if path.exists() else None
        if result is None or result.get("identity") != self.config["identity"]:
            entry.update(kind="infra", why="no result" if result is None else "identity")
            return entry
        generation = result.get("generation") or {}
        entry.update(
            kind=classify(result),
            success=bool(result.get("environment_success")),
            stages=[None, None],
            status=result.get("status"),
            prompt_sha256=(result.get("prompt") or {}).get("sha256"),
            library_sha256=(result.get("library") or {}).get("sha256"),
            requests=generation.get("requests"),
            tool_turns=result.get("tool_turns"),
            env_steps=result.get("env_steps"),
            input_tokens=generation.get("input_tokens"),
            cached_input_tokens=generation.get("cached_input_tokens"),
            output_tokens=generation.get("output_tokens"),
        )
        if entry["kind"] == "infra":
            entry["why"] = result.get("status") or "no verdict"
        elif entry["arm"] is None and result.get("library") is not None:
            raise RuntimeError(f"{key}: the episode read a library; this evaluation has none")
        elif entry["arm"] is not None and (
            entry["library_sha256"] != self.config["libraries"][task]["sha256"]
        ):
            raise RuntimeError(f"{key}: the episode read another library than init copied")
        return entry

    def report(self, others: list[RobocasaEvaluation] | None = None) -> dict[str, Any]:
        report = super().report()
        everyone = [self, *(others or [])]
        for other in everyone[1:]:
            if other.config["identity"] != self.config["identity"]:
                raise RuntimeError(f"{other.root}: another identity than {self.root}")
        entries = [e for evaluation in everyone for e in evaluation.entries()]
        names = [t for evaluation in everyone for t in evaluation.config["tasks"]]
        if len(set(names)) != len(names):
            raise RuntimeError("the roots share tasks; each task belongs to one root")
        judged = {e["key"]: e for e in entries if e["kind"] == "policy"}
        paper = reference_outcomes(self.config)
        rows: dict[str, Any] = {}
        ours_only = paper_only = both = neither = unpaired = 0
        for evaluation in everyone:
            config = evaluation.config
            for task in config["tasks"]:
                seeds = seeds_of(config, task)
                row = {"seeds": seeds, "ours": [], "paper": []}
                for n, seed in enumerate(seeds):
                    mine = judged.get(f"{task}/l{n}")
                    theirs = paper.get((split_task(task)[1], seed))
                    row["ours"].append(None if mine is None else int(mine["success"]))
                    row["paper"].append(None if theirs is None else int(theirs["success"]))
                    if mine is None or theirs is None:
                        unpaired += mine is not None
                        continue
                    if mine["success"] and theirs["success"]:
                        both += 1
                    elif mine["success"]:
                        ours_only += 1
                    elif theirs["success"]:
                        paper_only += 1
                    else:
                        neither += 1
                rows[task] = row
        mine = [e for e in judged.values() if e["task"] in rows]
        report.update(
            roots=[str(evaluation.root) for evaluation in everyone],
            identity=self.config["identity"],
            lean_commit=self.config.get("lean_commit"),
            ours={
                "episodes": len(mine),
                "successes": sum(1 for e in mine if e["success"]),
                "planned": sum(len(r["seeds"]) for r in rows.values()),
            },
            paper={
                "label": (self.config.get("reference") or {}).get("label"),
                "cells": sum(1 for r in rows.values() for v in r["paper"] if v is not None),
                "successes": sum(1 for r in rows.values() for v in r["paper"] if v),
            },
            paired={
                "pairs": ours_only + paper_only + both + neither,
                "ours_only": ours_only,
                "paper_only": paper_only,
                "both": both,
                "neither": neither,
                "judged_without_paper": unpaired,
                "sign_test_p": sign_test(ours_only, paper_only),
            },
            per_task=rows,
        )
        if any(split_task(task)[0] for task in rows):
            report.update(arms_report(rows))
        return report


def arms_report(rows: dict[str, Any]) -> dict[str, Any]:
    """Each arm's (``v0``: no library) episodes and successes, and every arm paired with ``v0``
    on the same (task, seed)."""

    arms: dict[str, dict[str, Any]] = {}
    cells: dict[str, dict[tuple[str, int], int]] = {}
    for task, row in rows.items():
        arm, name = split_task(task)
        arm = arm or "v0"
        summary = arms.setdefault(arm, {"episodes": 0, "successes": 0})
        judged = [
            (seed, v) for seed, v in zip(row["seeds"], row["ours"], strict=True) if v is not None
        ]
        summary["episodes"] += len(judged)
        summary["successes"] += sum(v for _, v in judged)
        cells.setdefault(arm, {}).update({(name, seed): v for seed, v in judged})
    paired = {}
    for arm in arms:
        if arm == "v0" or "v0" not in cells:
            continue
        pairs = [(v, cells["v0"][cell]) for cell, v in cells[arm].items() if cell in cells["v0"]]
        wins = sum(1 for v, b in pairs if v and not b)
        losses = sum(1 for v, b in pairs if b and not v)
        paired[f"{arm}_vs_v0"] = {
            "pairs": len(pairs),
            f"{arm}_only": wins,
            "v0_only": losses,
            "both": sum(1 for v, b in pairs if v and b),
            "neither": sum(1 for v, b in pairs if not v and not b),
            "sign_test_p": sign_test(wins, losses),
        }
    return {"arms": arms, "paired_with_v0": paired}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--config", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--tasks", nargs="+")
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--with", dest="others", nargs="+", type=Path, default=[])
    args = parser.parse_args(argv)
    if args.action == "init":
        frozen = init(args.root, args.config)
        planned = sum(spec["layouts"] for spec in frozen["tasks"].values())
        sys.stdout.write(
            f"initialised {args.root}: {planned} episodes, identity {frozen['identity'][:12]}\n"
        )
        return 0
    evaluation = RobocasaEvaluation(args.root)
    if args.action == "run":
        execution = evaluation.config["execution"]
        replay = bool(evaluation.config.get("replay_dir"))
        key = execution["api_key_env"]
        if not replay and key and not os.environ.get(key):
            raise SystemExit(f"environment variable {key} is empty")
        evaluation.resume(check=(lambda config: []) if replay else base.check_policy)
        report = evaluation.run(args.tasks)
        if evaluation.paused():
            sys.stderr.write("PAUSED: the model service failed; fix it and run again\n")
            return 3
        sys.stdout.write(json.dumps({"overall": report["overall"]}, indent=1) + "\n")
    else:
        report = evaluation.report([RobocasaEvaluation(other) for other in args.others])
        (evaluation.root / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        sys.stdout.write(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
