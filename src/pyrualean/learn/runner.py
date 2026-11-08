"""Episodes as jobs: a GPU slot, a watchdog, infra retries, a ledger and a launch budget.

Every launch of a campaign (dev, test, and the workers' replays, from any process) goes
through :meth:`Runner.episode`.  The ledger ``episodes.jsonl`` holds one line per finished
launch; ``state.json`` the counters.  Both are guarded by a file lock, so the coordinator's
threads and the workers' replay commands share one budget.

A job is identified by ``key`` (split/task/revision/layout/repeat).  A key with a judged
episode in the ledger is not run again: rerunning the campaign resumes it.  An
infrastructure failure (no verdict: boot error, crash, watchdog) is retried at most
``max_infra_retries`` times and never counted as a policy failure.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ..evolve.digest import classify, digest
from ..evolve.launcher import stop_tree
from .episode import execution_identity
from .slots import slot


class BudgetExhausted(RuntimeError):
    """A campaign budget is spent; nothing new starts."""


class ServiceDown(RuntimeError):
    """A model service (the policy's endpoint or the worker's) is failing: the campaign pauses.

    Nothing new starts while ``PAUSED.json`` exists; ``campaign run`` checks both services,
    clears it and resumes.  Work lost to the outage is not counted as an attempt.
    """


#: Ledger ``why`` of an episode the policy's model provider ended (after its own retries).
PROVIDER_FAILURE = "planner_error"


def pause(root: Path, reason: str) -> None:
    path = Path(root) / "PAUSED.json"
    if not path.exists():
        path.write_text(json.dumps({"at": _now(), "reason": reason}, indent=1) + "\n")


def paused(root: Path) -> dict[str, Any] | None:
    path = Path(root) / "PAUSED.json"
    return json.loads(path.read_text()) if path.exists() else None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Runner:
    def __init__(
        self,
        root: str | Path,
        config: dict[str, Any],
        *,
        launch: Callable[[Path, dict[str, str], float], int] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.config = config
        self.identity = execution_identity(config)
        self._launch = launch or subprocess_launch

    # -- ledger --------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        with (self.root / ".ledger.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            path = self.root / "state.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            state.setdefault("started", time.time())
            state.setdefault("launches", 0)
            state.setdefault("infra", 0)
            yield state
            path.with_suffix(".tmp").write_text(json.dumps(state, indent=1) + "\n")
            path.with_suffix(".tmp").replace(path)

    def entries(self) -> list[dict[str, Any]]:
        path = self.root / "episodes.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def done(self, key: str) -> dict[str, Any] | None:
        return next((e for e in self.entries() if e["key"] == key and e["kind"] == "policy"), None)

    def _reserve(self, key: str) -> int:
        """Count a launch against the budget; returns the attempt number of ``key``."""

        budgets = self.config["budgets"]
        with self._locked() as state:
            if (why := paused(self.root)) is not None:
                raise ServiceDown(f"campaign paused: {why['reason']}")
            if state["launches"] >= budgets["max_launches"]:
                raise BudgetExhausted(f"launch budget spent ({budgets['max_launches']})")
            hours = (time.time() - state["started"]) / 3600
            if hours >= budgets["max_hours"]:
                raise BudgetExhausted(f"campaign wall clock spent ({budgets['max_hours']} h)")
            attempts = state.setdefault("attempts", {})
            if attempts.get(key, 0) > budgets["max_infra_retries"]:
                raise BudgetExhausted(f"{key}: infrastructure retries spent")
            attempt = attempts.get(key, 0)
            attempts[key] = attempt + 1
            state["launches"] += 1
            return attempt

    def _record(self, entry: dict[str, Any]) -> None:
        limit = int(self.config["learning"].get("pause_after_provider_failures", 3))
        with self._locked() as state:
            if entry["kind"] == "infra":
                state["infra"] += 1
            if entry.get("why") == PROVIDER_FAILURE:
                # an outage is not the job's fault: it does not use up the job's retries
                state["attempts"][entry["key"]] -= 1
                state["provider_failures"] = state.get("provider_failures", 0) + 1
                if state["provider_failures"] >= limit:
                    pause(
                        self.root,
                        f"{state['provider_failures']} episodes in a row ended by the policy's "
                        "model provider",
                    )
            elif entry["kind"] == "policy":
                state["provider_failures"] = 0
            with (self.root / "episodes.jsonl").open("a") as ledger:
                ledger.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _refund(self, key: str) -> None:
        with self._locked() as state:
            state["attempts"][key] -= 1
            state["launches"] -= 1

    def reconcile(self) -> None:
        """Recount each job's attempts from the ledger (launches killed by a stop do not count)."""

        counts: dict[str, int] = {}
        for entry in self.entries():
            if entry["kind"] == "infra" and entry.get("why") != PROVIDER_FAILURE:
                counts[entry["key"]] = counts.get(entry["key"], 0) + 1
        with self._locked() as state:
            state["attempts"] = counts
            state["provider_failures"] = 0

    # -- one job -------------------------------------------------------------

    def episode(
        self,
        *,
        split: str,
        task: str,
        revision: str,
        layout: int,
        view: Path,
        repeat: int = 0,
        replay_dir: Path | None = None,
        output_root: Path | None = None,
    ) -> dict[str, Any] | None:
        """Run one job until it is judged; its ledger entry, or None when infra won."""

        eval_seed = self.config["test" if split == "test" else "dev"]["eval_seed"]
        key = f"{split}/{task}/{revision}/s{eval_seed}l{layout}/r{repeat}"
        if replay_dir is not None:
            key = f"replay/{task}/{replay_dir.resolve()}/s{eval_seed}l{layout}"
        if replay_dir is None and (previous := self.done(key)):
            return previous
        while True:
            try:
                attempt = self._reserve(key)
            except BudgetExhausted as exc:
                if "infrastructure retries" in str(exc):
                    return None
                raise
            base = output_root or self.root / "runs" / split / task
            output = base / f"{revision}-l{layout}-r{repeat}-a{attempt}"
            if replay_dir is not None:
                output = base / f"{replay_dir.name}-l{layout}-a{attempt}"
            request = {
                "config": self.config,
                "identity": self.identity,
                "split": split,
                "task": task,
                "revision": revision,
                "eval_seed": eval_seed,
                "layout": layout,
                "view": str(view),
                "replay_dir": str(replay_dir) if replay_dir else None,
                "output": str(output),
            }
            while output.exists():  # left by a launch killed with an earlier coordinator
                output = output.with_name(output.name + "+")
            request["output"] = str(output)
            started = time.time()
            with slot(self.root, self.config["slots"]) as gpu:
                if (why := paused(self.root)) is not None:  # paused while waiting for a slot
                    self._refund(key)
                    raise ServiceDown(f"campaign paused: {why['reason']}")
                request["gpu"] = gpu
                output.mkdir(parents=True, exist_ok=False)
                (output / "request.json").write_text(json.dumps(request, indent=1) + "\n")
                code = self._launch(
                    output,
                    self._environment(output),
                    float(
                        self.config["worker"]["replay_watchdog_seconds"]
                        if replay_dir is not None
                        else self.config["budgets"]["episode_watchdog_seconds"]
                    ),
                )
            entry = self._entry(key, request, output, code, time.time() - started, attempt)
            self._record(entry)
            if entry["kind"] == "policy":
                return entry
            if entry.get("why") == PROVIDER_FAILURE and paused(self.root):
                raise ServiceDown(f"campaign paused: {paused(self.root)['reason']}")

    def _environment(self, output: Path) -> dict[str, str]:
        env = environment(self.config)
        tmp = output / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        repo_src = str(Path(__file__).resolve().parents[2])
        env.update(
            TMPDIR=str(tmp),
            OMNI_KIT_ACCEPT_EULA="YES",
            PYTHONUNBUFFERED="1",
            PYTHONPATH=os.pathsep.join(p for p in (repo_src, env.get("PYTHONPATH", "")) if p),
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
        entry: dict[str, Any] = {
            "at": _now(),
            "key": key,
            "split": request["split"],
            "task": request["task"],
            "revision": request["revision"],
            "eval_seed": request["eval_seed"],
            "layout": request["layout"],
            "replay": request["replay_dir"] is not None,
            "attempt": attempt,
            "exit": code,
            "seconds": round(seconds, 1),
            "run_dir": str(run_dir),
        }
        result_path = run_dir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        if result is None or result.get("identity") != self.identity:
            entry.update(kind="infra", why="no result" if result is None else "identity")
            return entry
        kind = classify(result)
        if result.get("status") == "planner_error" and request["replay_dir"] is None:
            kind = "infra"  # the provider dropped the session; not the policy's doing
        facts = digest(run_dir)
        generation = result.get("generation") or {}
        entry.update(
            kind=kind,
            success=bool(result.get("environment_success")),
            stages=[facts.get("stages_passed"), facts.get("stages_total")],
            score=facts.get("score"),
            end=result.get("end"),
            claim=facts.get("claim"),
            cells=len(facts.get("cells") or []),
            requests=generation.get("requests"),
            input_tokens=generation.get("input_tokens"),
            cached_input_tokens=generation.get("cached_input_tokens"),
            output_tokens=generation.get("output_tokens"),
        )
        if kind == "infra":
            entry["why"] = result.get("status")
        return entry


def environment(config: dict[str, Any]) -> dict[str, str]:
    """This process's environment with the config's ``environment`` applied (null: unset)."""

    env = dict(os.environ)
    for name, value in config.get("environment", {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = str(value)
    return env


def subprocess_launch(output: Path, env: dict[str, str], timeout_s: float) -> int:
    """The episode in a fresh Python process, killed with its tree at the watchdog."""

    command = [sys.executable, "-m", "pyrualean.learn.episode", str(output / "request.json")]
    with (output / "episode.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            stop_tree(process)
            (output / "watchdog.json").write_text(json.dumps({"killed_after_s": timeout_s}))
            return 124
        except BaseException:
            stop_tree(process)
            raise
