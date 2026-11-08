"""LIBERO-PRO evaluation of several libraries (arms) on the same tasks and seeds (resumable).

``python eval_libero/run.py ROOT init --config CONFIG``   freeze the config, copy the libraries
``python eval_libero/run.py ROOT run [--tasks ...]``      episodes not judged yet
``python eval_libero/run.py ROOT report``                 successes per arm, paired by task and seed

The scheduling, ledger and bookkeeping are ``eval/run.py``'s :class:`Evaluation` (imported, not
changed): at most ``concurrency`` episodes at once on ``gpus`` (``per_gpu``, ``need_gib``,
``stagger_seconds``); an episode without a verdict is infrastructure, retried at most
``budgets.max_infra_retries`` times, then missing; ``pause_after_provider_failures`` provider
failures in a row pause the run.

An evaluation task is ``<arm>/<suite>-t<index>``, its layout ``n`` the LIBERO seed ``seeds[n]``;
every task has every seed (LIBERO has no unstable scene, so no layout replaces another).  The
order interleaves the arms task by task, so that both arms of a task run side by side.  LIBERO
judges by its task predicate, checked after every simulator step: there is no verdict record and
no stages, ``environment_success`` is the outcome.  Each episode is ``libero_episode.py`` in the
configured RPent LIBERO Python (``paths.python``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from libero_episode import identity, library_sha256, parse_task  # noqa: E402


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
    "robot", "primitives", "runtime", "model", "extra_body", "images", "feedback", "max_cells",
    "max_model_requests", "max_tokens", "nudges", "provider_retries", "policy_seconds",
    "max_episode_steps", "base_url", "api_key_env",
)  # fmt: skip
PATH_FIELDS = (
    "python",
    "lean",
    "rpent_code",
    "liberopro",
    "assets",
    "libero_config",
    "sam3_checkpoint",
)


class ConfigError(ValueError):
    """A configuration that cannot run."""


def check(config: dict[str, Any]) -> None:
    seeds, indices, arms = config.get("seeds"), config.get("task_indices"), config.get("arms")
    if not seeds or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) for s in seeds):
        raise ConfigError("seeds: a nonempty list of distinct integers")
    if not indices or len(set(indices)) != len(indices):
        raise ConfigError("task_indices: a nonempty list of distinct task indices of the suite")
    if not arms or any("/" in arm or "-t" in arm for arm in arms):
        raise ConfigError("arms: {name: {library}}, names without '/' or '-t'")
    for arm, spec in arms.items():
        if not Path(spec.get("library", "")).is_dir():
            raise ConfigError(f"arms.{arm}.library is not a directory")
    execution = config.get("execution") or {}
    missing = [field for field in EXECUTION_FIELDS if field not in execution]
    if missing:
        raise ConfigError(f"execution: missing {missing}")
    if execution["robot"] not in ("libero_rgb", "libero"):
        raise ConfigError("execution.robot: libero_rgb (the colour primitives) or libero (RPent's)")
    if execution["runtime"] == "codex" and not execution["base_url"]:
        if not (execution.get("codex_bin") and execution.get("codex_home")):
            raise ConfigError("runtime codex without base_url: codex_bin and codex_home log in")
    elif not (execution["base_url"] and execution["api_key_env"]):
        raise ConfigError("execution.base_url and execution.api_key_env name the endpoint")
    paths = config.get("paths") or {}
    for field in PATH_FIELDS:
        if not paths.get(field) or not Path(paths[field]).exists():
            raise ConfigError(f"paths.{field} does not exist")
    for part in ("bddl_files", "init_files"):
        if not (Path(paths["liberopro"]) / part / config.get("suite", "")).is_dir():
            raise ConfigError(f"suite {config.get('suite')!r} has no {part} under paths.liberopro")
    for field in ("concurrency", "per_gpu", "need_gib", "gpus", "budgets", "sam3_sha256"):
        if not config.get(field):
            raise ConfigError(f"{field} is required")
    for field in ("max_launches", "max_hours", "max_infra_retries", "episode_watchdog_seconds"):
        if not isinstance(config["budgets"].get(field), (int, float)):
            raise ConfigError(f"budgets.{field} is required")
    if config["budgets"]["episode_watchdog_seconds"] <= execution["policy_seconds"]:
        raise ConfigError("the watchdog must leave the episode its policy wall clock and boot")


def task_name(arm: str, suite: str, index: int) -> str:
    return f"{arm}/{suite}-t{index}"


def init(root: Path, config_path: Path, *, sam3_hash: Callable[[Path], str] | None = None) -> dict:
    """Freeze ``config_path`` into ``root``: libraries copied per task, identity, task order."""

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
    suite, arms = config["suite"], config["arms"]
    tasks, libraries, order = {}, {}, []
    for index in config["task_indices"]:
        for arm, spec in arms.items():
            name = task_name(arm, suite, index)
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
            tasks[name] = {"layouts": len(config["seeds"]), "pool": len(config["seeds"])}
            libraries[name] = {
                "sha256": sha,
                "source": str(source),
                "revision": arm,
                "source_test": None,
            }
            order.append(name)
    frozen = {
        **config,
        "created": base._now(),
        "eval_seed": None,
        "tasks": tasks,
        "libraries": libraries,
        "order": order,
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


def classify(result: dict[str, Any]) -> str:
    """``policy``: an episode LIBERO judged (its predicate ran to the end); ``infra``: one it
    never could."""

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


class LiberoEvaluation(base.Evaluation):
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
        """The episode in a fresh RPent Python process, killed with its tree at the watchdog."""

        command = [
            self.config["paths"]["python"],
            str(HERE / "libero_episode.py"),
            str(output / "request.json"),
        ]
        with (output / "episode.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
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
        env = base.environment(self.config)
        # RPent's servers pin the request's GPU themselves; a mask here would renumber it
        env.pop("CUDA_VISIBLE_DEVICES", None)
        tmp = output / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env.update(
            TMPDIR=str(tmp),
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            # the servers RPent spawns import rpent from the code copy too, not the venv's
            PYTHONPATH=os.pathsep.join((str(Path(paths["lean"]) / "src"), paths["rpent_code"])),
            RPENT_ROOT=paths["rpent_code"],
            LIBERO_PRO_ASSET_PATH=paths["assets"],
            LIBERO_CONFIG_PATH=paths["libero_config"],
            LIBERO_TYPE="pro",
            ROBOT_PLATFORM="LIBERO",
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            SAM3_CHECKPOINT_PATH=paths["sam3_checkpoint"],
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
        arm, _, index = parse_task(request["task"])
        entry: dict[str, Any] = {
            "at": base._now(),
            "key": key,
            "task": request["task"],
            "arm": arm,
            "task_index": index,
            "layout": request["layout"],
            "seed": self.config["seeds"][int(request["layout"])],
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
        elif entry["library_sha256"] != self.config["libraries"][request["task"]]["sha256"]:
            raise RuntimeError(f"{key}: the episode read another library than init copied")
        return entry

    def report(self) -> dict[str, Any]:
        report = super().report()
        entries = self.entries()
        judged = {e["key"]: e for e in entries if e["kind"] == "policy"}
        arms = list(self.config["arms"])
        suite, seeds = self.config["suite"], self.config["seeds"]
        per_arm = {}
        for arm in arms:
            mine = [e for e in judged.values() if e["arm"] == arm]
            per_arm[arm] = {
                "episodes": len(mine),
                "successes": sum(1 for e in mine if e["success"]),
                "missing": len(self.config["task_indices"]) * len(seeds) - len(mine),
            }
        paired: dict[str, Any] = {}
        reference = arms[0]
        for other in arms[1:]:
            rows, wins, losses, both, neither = {}, 0, 0, 0, 0
            for index in self.config["task_indices"]:
                row = {
                    "pairs": 0,
                    f"{reference}_only": 0,
                    f"{other}_only": 0,
                    "both": 0,
                    "neither": 0,
                }
                for n in range(len(seeds)):
                    a = judged.get(f"{task_name(reference, suite, index)}/l{n}")
                    b = judged.get(f"{task_name(other, suite, index)}/l{n}")
                    if a is None or b is None:
                        continue
                    row["pairs"] += 1
                    if a["success"] and b["success"]:
                        row["both"] += 1
                    elif a["success"]:
                        row[f"{reference}_only"] += 1
                    elif b["success"]:
                        row[f"{other}_only"] += 1
                    else:
                        row["neither"] += 1
                rows[f"t{index}"] = row
                wins += row[f"{other}_only"]
                losses += row[f"{reference}_only"]
                both += row["both"]
                neither += row["neither"]
            paired[f"{other}_vs_{reference}"] = {
                "tasks": rows,
                "pairs": wins + losses + both + neither,
                f"{other}_only": wins,
                f"{reference}_only": losses,
                "both": both,
                "neither": neither,
                "sign_test_p": sign_test(wins, losses),
            }
        report.update(suite=suite, seeds=seeds, arms=per_arm, paired=paired)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--config", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--tasks", nargs="+")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.action == "init":
        frozen = init(args.root, args.config)
        planned = len(frozen["order"]) * len(frozen["seeds"])
        sys.stdout.write(
            f"initialised {args.root}: {planned} episodes, identity {frozen['identity'][:12]}\n"
        )
        return 0
    evaluation = LiberoEvaluation(args.root)
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
        sys.stdout.write(
            json.dumps({"overall": report["overall"], "arms": report["arms"]}, indent=1) + "\n"
        )
    else:
        report = evaluation.report()
        (evaluation.root / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        sys.stdout.write(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
