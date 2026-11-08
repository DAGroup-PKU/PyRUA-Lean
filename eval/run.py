"""Run an exported evaluation: every task's library on its layouts, once each (resumable).

``python eval/run.py ROOT run [--tasks ...]``   episodes not judged yet
``python eval/run.py ROOT report``              successes per task, against the source test

At most ``concurrency`` episodes run at once (the model API's limit).  An episode starts on a
GPU of ``gpus`` that runs fewer than ``per_gpu`` of ours and has ``need_gib`` free (all users)
after what our episodes already started there may still take: each counts ``need_gib`` less
what its processes hold now.  Starts are ``stagger_seconds`` apart.

The ledger ``episodes.jsonl`` holds one line per finished launch; a key (task, layout) with a
judged episode is not run again.  An episode that never got a verdict (boot error, crash,
watchdog) is infrastructure: retried at most ``budgets.max_infra_retries`` times, then
missing, never a failure.  ``pause_after_provider_failures`` episodes in a row ended by the
model provider write ``PAUSED.json`` and nothing new starts; ``run`` again checks the endpoint
and resumes.  The model key comes from the variable ``execution.api_key_env``; with
``execution.runtime`` ``codex`` and no ``base_url`` there is none: the endpoint check is then
``codex login status`` in ``execution.codex_home``.

A layout whose scene the benchmark finds unstable (it judges nothing and counts it under
``Unstable nums``, as its own evaluation skips it) is not retried.  A task with ``pool`` above
``layouts`` takes layouts 0, 1, ... below ``pool`` until ``layouts`` of them are judged: a
layout left without a verdict is replaced by the next one, as RoboDojo's own evaluation does.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from episode import identity  # noqa: E402

EPISODE = Path(__file__).resolve().parent / "episode.py"
#: ``result.status`` of a run the benchmark never judged.
INFRA_STATUSES = frozenset({"boot_error", "error", "killed"})
#: ``result.status`` of an episode the model provider ended (after its own retries).
PROVIDER_FAILURE = "planner_error"
#: ``why`` of a launch whose layout the benchmark found unstable: never retried.
UNSTABLE = "unstable"
#: The tally the benchmark's client prints after each batch.
TALLY = re.compile(r"Success nums: (\d+), Fail nums: (\d+), Unstable nums: (\d+)")


class Stop(RuntimeError):
    """A budget is spent or the run is paused: nothing new starts."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def environment(config: dict[str, Any]) -> dict[str, str]:
    """This process's environment with the config's ``environment`` applied (null: unset)."""

    env = dict(os.environ)
    for name, value in config.get("environment", {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = str(value)
    return env


def descendants(pid: int) -> list[int]:
    found = []
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except OSError:
        return found
    for child in children:
        found += [int(child), *descendants(int(child))]
    return found


def stop_tree(process: subprocess.Popen) -> None:
    pids = [*descendants(process.pid), process.pid]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def _smi(query: str) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout  # fmt: skip
    return [[part.strip() for part in line.split(",")] for line in out.splitlines() if line]


def free_gib() -> dict[int, float]:
    """Free memory per GPU (all users)."""

    return {int(index): float(mib) / 1024 for index, mib in _smi("gpu=index,memory.free")}


def held_gib() -> dict[int, float]:
    """Memory per process on any GPU."""

    held: dict[int, float] = {}
    for pid, mib in _smi("compute-apps=pid,used_memory"):
        held[int(pid)] = held.get(int(pid), 0.0) + float(mib) / 1024
    return held


class Gpus:
    """GPU admission for our episodes (see the module docstring)."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        free: Callable[[], dict[int, float]] = free_gib,
        held: Callable[[], dict[int, float]] = held_gib,
        poll_s: float = 10.0,
    ) -> None:
        self.gpus = [int(g) for g in config["gpus"]]
        self.per_gpu = int(config["per_gpu"])
        self.need = float(config["need_gib"])
        self.stagger = float(config.get("stagger_seconds", 0))
        self.free, self.held, self.poll_s = free, held, poll_s
        self.lock = threading.Lock()
        self.running: dict[str, dict[str, Any]] = {}
        self.last_start = 0.0

    def _pending(self, gpu: int, held: dict[int, float]) -> float:
        """What our episodes on ``gpu`` may still take beyond what they hold."""

        total = 0.0
        for job in self.running.values():
            if job["gpu"] != gpu:
                continue
            pid = job.get("pid")
            now = 0.0 if pid is None else sum(held.get(p, 0.0) for p in [pid, *descendants(pid)])
            total += max(0.0, self.need - now)
        return total

    def acquire(self, key: str) -> int:
        while True:
            with self.lock:
                free = self.free()
                if not free:
                    raise RuntimeError("nvidia-smi reported no GPU")
                held = self.held() if self.running else {}
                room = {
                    gpu: free.get(gpu, 0.0) - self._pending(gpu, held)
                    for gpu in self.gpus
                    if sum(1 for j in self.running.values() if j["gpu"] == gpu) < self.per_gpu
                }
                fits = [gpu for gpu, gib in room.items() if gib >= self.need]
                if fits:
                    gpu = max(fits, key=lambda g: room[g])
                    wait = self.last_start + self.stagger - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    self.last_start = time.time()
                    self.running[key] = {"gpu": gpu, "pid": None}
                    return gpu
            time.sleep(self.poll_s)

    def started(self, key: str, pid: int) -> None:
        with self.lock:
            self.running[key]["pid"] = pid

    def release(self, key: str) -> None:
        with self.lock:
            self.running.pop(key, None)


def subprocess_launch(
    output: Path, env: dict[str, str], timeout_s: float, started: Callable[[int], None]
) -> int:
    """The episode in a fresh Python process, killed with its tree at the watchdog."""

    command = [sys.executable, str(EPISODE), str(output / "request.json")]
    with (output / "episode.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        started(process.pid)
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            stop_tree(process)
            (output / "watchdog.json").write_text(json.dumps({"killed_after_s": timeout_s}))
            return 124
        except BaseException:
            stop_tree(process)
            raise


def check_policy(config: dict[str, Any]) -> list[str]:
    """The policy's endpoint answers from this host, with the run's environment and key."""

    import httpx

    env = environment(config)
    execution = config["execution"]
    codex = execution.get("runtime", "openai") == "codex"
    if codex and not execution["base_url"]:
        # the subscription: the clean home must hold a login (made by the user, never copied)
        status = subprocess.run(
            [execution["codex_bin"], "login", "status"],
            env={**env, "CODEX_HOME": execution["codex_home"]},
            capture_output=True,
            text=True,
            timeout=60,
        )
        said = (status.stdout + status.stderr).strip()
        if status.returncode != 0 or not said.startswith("Logged in"):
            return [f"codex home {execution['codex_home']}: {said or 'login status failed'}"]
        return []
    proxy = env.get("HTTPS_PROXY") or env.get("https_proxy")
    try:
        with httpx.Client(proxy=proxy, trust_env=False, timeout=60) as client:
            reply = client.post(
                execution["base_url"].rstrip("/") + ("/responses" if codex else "/chat/completions"),
                headers={"Authorization": f"Bearer {env.get(execution['api_key_env'], '')}"},
                json=(
                    {"model": execution["model"], "input": "Reply with: ok", "max_output_tokens": 16}
                    if codex
                    else {
                        "model": execution["model"],
                        "messages": [{"role": "user", "content": "Reply with: ok"}],
                        "max_tokens": 16,
                    }
                ),
            )
    except httpx.HTTPError as exc:
        return [f"policy endpoint: {type(exc).__name__}: {exc}"]
    return [] if reply.status_code == 200 else [f"policy endpoint: HTTP {reply.status_code}"]


def classify(result: dict[str, Any]) -> str:
    """``policy``: an episode the benchmark judged; ``infra``: one it never could."""

    if result.get("status") in INFRA_STATUSES or result.get("status") == PROVIDER_FAILURE:
        return "infra"
    if (result.get("generation") or {}).get("status") == "error":
        return "infra"
    if result.get("verdict") is None and not result.get("environment_success"):
        return "infra"  # the client never reported a verdict
    return "policy"


def unstable(run_dir: Path) -> bool:
    """The benchmark's client judged nothing and counted the scene unstable."""

    log = run_dir / "robodojo_eval_client.log"
    tallies = TALLY.findall(log.read_text(errors="replace")) if log.exists() else []
    if not tallies:
        return False
    successes, failures, unstables = (int(n) for n in tallies[-1])
    return successes + failures == 0 and unstables > 0


class Evaluation:
    def __init__(self, root: str | Path, *, launch: Any = None, gpus: Gpus | None = None) -> None:
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / "config.json").read_text())
        if identity(self.config) != self.config["identity"]:
            raise RuntimeError("code, benchmark or execution settings changed since export")
        self.launch = launch or subprocess_launch
        self.gpus = gpus or Gpus(self.config)
        self.lock = threading.Lock()

    # -- ledger --------------------------------------------------------------

    def log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        with self.lock:
            with (self.root / "eval.log").open("a") as handle:
                handle.write(line + "\n")
        print(line, flush=True)

    def entries(self) -> list[dict[str, Any]]:
        path = self.root / "episodes.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def _state(self) -> dict[str, Any]:
        path = self.root / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        state.setdefault("started", time.time())
        state.setdefault("launches", 0)
        state.setdefault("attempts", {})
        state.setdefault("provider_failures", 0)
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        path = self.root / "state.json"
        path.with_suffix(".tmp").write_text(json.dumps(state, indent=1) + "\n")
        path.with_suffix(".tmp").replace(path)

    def paused(self) -> dict[str, Any] | None:
        path = self.root / "PAUSED.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _reserve(self, key: str) -> int | None:
        """Count a launch; its attempt number, or None when the key's retries are spent."""

        budgets = self.config["budgets"]
        with self.lock:
            if (why := self.paused()) is not None:
                raise Stop(f"paused: {why['reason']}")
            state = self._state()
            if state["launches"] >= budgets["max_launches"]:
                raise Stop(f"launch budget spent ({budgets['max_launches']})")
            if (time.time() - state["started"]) / 3600 >= budgets["max_hours"]:
                raise Stop(f"wall clock spent ({budgets['max_hours']} h)")
            attempt = state["attempts"].get(key, 0)
            if attempt > budgets["max_infra_retries"]:
                return None
            state["attempts"][key] = attempt + 1
            state["launches"] += 1
            self._save_state(state)
            return attempt

    def _record(self, entry: dict[str, Any]) -> None:
        limit = int(self.config.get("pause_after_provider_failures", 3))
        with self.lock:
            state = self._state()
            if entry.get("why") == PROVIDER_FAILURE:
                # an outage is not the job's fault: it does not use up the job's retries
                state["attempts"][entry["key"]] -= 1
                state["provider_failures"] += 1
                if state["provider_failures"] >= limit and self.paused() is None:
                    (self.root / "PAUSED.json").write_text(
                        json.dumps(
                            {
                                "at": _now(),
                                "reason": f"{state['provider_failures']} episodes in a row "
                                "ended by the model provider",
                            }
                        )
                        + "\n"
                    )
            elif entry["kind"] == "policy":
                state["provider_failures"] = 0
            self._save_state(state)
            with (self.root / "episodes.jsonl").open("a") as ledger:
                ledger.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def resume(self, check: Any = None) -> None:
        """Before a run: the endpoint answers, the pause is lifted, retries are recounted."""

        problems = (check or check_policy)(self.config)
        if problems:
            raise Stop("; ".join(problems))
        if (why := self.paused()) is not None:
            self.log(f"resuming after pause: {why['reason']}")
            (self.root / "PAUSED.json").rename(
                self.root / f"PAUSED-{time.strftime('%Y%m%d-%H%M%S')}.json"
            )
        counts: dict[str, int] = {}
        for entry in self.entries():
            if entry["kind"] == "infra" and entry.get("why") != PROVIDER_FAILURE:
                counts[entry["key"]] = counts.get(entry["key"], 0) + 1
            if entry.get("why") == UNSTABLE:  # never retried
                counts[entry["key"]] = int(self.config["budgets"]["max_infra_retries"]) + 1
        with self.lock:
            state = self._state()
            state["attempts"] = counts
            state["provider_failures"] = 0
            self._save_state(state)

    # -- one episode ---------------------------------------------------------

    def episode(self, task: str, layout: int) -> dict[str, Any] | None:
        """Run ``task`` on ``layout`` until it is judged; its entry, or None when infra won."""

        key = f"{task}/l{layout}"
        done = {e["key"]: e for e in self.entries() if e["kind"] == "policy"}
        if key in done:
            return done[key]
        if self.skipped(task).get(layout) == UNSTABLE:
            return None
        while True:
            attempt = self._reserve(key)
            if attempt is None:
                return None
            output = self.root / "runs" / task / f"l{layout}-a{attempt}"
            while output.exists():  # left by a launch killed with an earlier coordinator
                output = output.with_name(output.name + "+")
            gpu = self.gpus.acquire(key)
            try:
                if (why := self.paused()) is not None:  # paused while waiting for a GPU
                    with self.lock:
                        state = self._state()
                        state["attempts"][key] -= 1
                        state["launches"] -= 1
                        self._save_state(state)
                    raise Stop(f"paused: {why['reason']}")
                request = {
                    "config": self.config,
                    "identity": self.config["identity"],
                    "task": task,
                    "layout": layout,
                    "eval_seed": self.config["eval_seed"],
                    "library_name": task,
                    "view": str(self.root / "libraries" / task),
                    "gpu": gpu,
                    "output": str(output),
                }
                output.mkdir(parents=True)
                (output / "request.json").write_text(json.dumps(request, indent=1) + "\n")
                started = time.time()
                code = self.launch(
                    output,
                    self._environment(output),
                    float(self.config["budgets"]["episode_watchdog_seconds"]),
                    lambda pid: self.gpus.started(key, pid),
                )
            finally:
                self.gpus.release(key)
            entry = self._entry(key, request, output, code, time.time() - started, attempt)
            self._record(entry)
            if entry["kind"] == "policy":
                return entry
            self.log(f"{task} l{layout}: infrastructure ({entry['why']}), attempt {attempt}")
            if entry["why"] == UNSTABLE:
                return None

    def _environment(self, output: Path) -> dict[str, str]:
        import pyrualean

        env = environment(self.config)
        tmp = output / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        source = str(Path(pyrualean.__file__).resolve().parents[1])
        env.update(
            TMPDIR=str(tmp),
            OMNI_KIT_ACCEPT_EULA="YES",
            PYTHONUNBUFFERED="1",
            PYTHONPATH=os.pathsep.join(p for p in (source, env.get("PYTHONPATH", "")) if p),
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
            "task": request["task"],
            "layout": request["layout"],
            "eval_seed": request["eval_seed"],
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
        verdict = result.get("verdict") or {}
        generation = result.get("generation") or {}
        entry.update(
            kind=classify(result),
            success=bool(result.get("environment_success")),
            stages=[verdict.get("stages_passed"), verdict.get("stages_total")],
            status=result.get("status"),
            prompt_sha256=(result.get("prompt") or {}).get("sha256"),
            library_sha256=(result.get("library") or {}).get("sha256"),
            requests=generation.get("requests"),
            input_tokens=generation.get("input_tokens"),
            cached_input_tokens=generation.get("cached_input_tokens"),
            output_tokens=generation.get("output_tokens"),
        )
        if entry["kind"] == "infra":
            entry["why"] = UNSTABLE if unstable(run_dir) else result.get("status") or "no verdict"
        elif entry["library_sha256"] != self.config["libraries"][request["task"]]["sha256"]:
            raise RuntimeError(f"{key}: the episode read another library than the export")
        return entry

    # -- all of them ---------------------------------------------------------

    def pool(self, task: str) -> int:
        """Layouts ``task`` may take: ``layouts`` of them are judged, the rest replace."""

        spec = self.config["tasks"][task]
        return int(spec.get("pool", spec["layouts"]))

    def skipped(self, task: str) -> dict[int, str]:
        """Layouts of ``task`` left without a verdict, and why (their last launch's reason)."""

        judged = {e["layout"] for e in self.entries() if e["task"] == task and e["kind"] == "policy"}
        why: dict[int, str] = {}
        for entry in self.entries():
            if entry["task"] == task and entry["kind"] == "infra" and entry["layout"] not in judged:
                if entry.get("why") != PROVIDER_FAILURE:
                    why[entry["layout"]] = entry["why"]
        attempts = self._state()["attempts"]
        retries = int(self.config["budgets"]["max_infra_retries"])
        return {
            layout: reason
            for layout, reason in why.items()
            if reason == UNSTABLE or attempts.get(f"{task}/l{layout}", 0) > retries
        }

    def jobs(self, tasks: list[str] | None = None) -> list[tuple[str, int]]:
        """One job per judged episode wanted: (task, its index among them)."""

        unknown = set(tasks or ()) - set(self.config["libraries"])
        if unknown:
            raise ValueError(f"not tasks of this evaluation: {sorted(unknown)}")
        return [
            (task, slot)
            for task in self.config["order"]
            if not tasks or task in tasks
            for slot in range(self.config["tasks"][task]["layouts"])
        ]

    def run(self, tasks: list[str] | None = None) -> dict[str, Any]:
        jobs = self.jobs(tasks)
        self.log(f"run: {len(jobs)} episodes, at most {self.config['concurrency']} at once")
        stopped: dict[str, int] = {}
        # each task's layouts in order: a job takes the next one until one of them is judged
        queues: dict[str, list[int]] = {}
        for task in {task for task, _ in jobs}:
            judged = {e["layout"] for e in self.entries() if e["task"] == task and e["kind"] == "policy"}
            skipped = self.skipped(task)
            # judged layouts first, so that a resumed run gives each job its old one back
            # "only": these layouts and no other, in this order
            listed = self.config["tasks"][task].get("only") or range(self.pool(task))
            rest = [n for n in listed if n not in judged and n not in skipped]
            queues[task] = sorted(judged) + rest

        def take(task: str) -> int | None:
            with self.lock:
                return queues[task].pop(0) if queues[task] else None

        def one(job: tuple[str, int]) -> None:
            task, _ = job
            while (layout := take(task)) is not None:
                try:
                    entry = self.episode(task, layout)
                except Stop as exc:
                    with self.lock:
                        stopped[str(exc)] = stopped.get(str(exc), 0) + 1
                    return
                if entry is not None:
                    outcome = "success" if entry["success"] else "failure"
                    self.log(f"{task} l{layout}: {outcome} ({entry['seconds'] / 60:.0f} min)")
                    return
                spec = self.config["tasks"][task]
                if "only" in spec or self.pool(task) == spec["layouts"]:
                    self.log(f"{task} l{layout}: missing (infrastructure retries spent)")
                    return
                self.log(f"{task} l{layout}: skipped, replaced by the next layout")
            self.log(f"{task}: missing (no layout left below {self.pool(task)})")

        with ThreadPoolExecutor(max_workers=int(self.config["concurrency"])) as pool:
            list(pool.map(one, jobs))
        for reason, count in stopped.items():
            self.log(f"{count} episodes not run: {reason}")
        report = self.report()
        (self.root / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        return report

    def report(self) -> dict[str, Any]:
        entries = self.entries()
        judged = {e["key"]: e for e in entries if e["kind"] == "policy"}
        tasks: dict[str, Any] = {}
        for task in self.config["order"]:
            planned = self.config["tasks"][task]["layouts"]
            done = [judged[k] for n in range(self.pool(task)) if (k := f"{task}/l{n}") in judged]
            stages = [e["stages"][0] for e in done if e["stages"][0] is not None]
            successes = sum(1 for e in done if e["success"])
            tasks[task] = {
                "layouts": planned,
                "episodes": len(done),
                "missing": planned - len(done),
                "successes": successes,
                "rate": round(successes / len(done), 4) if done else None,
                "mean_stages": round(sum(stages) / len(stages), 2) if stages else None,
                "judged_layouts": sorted(e["layout"] for e in done),
                "skipped_layouts": {str(n): why for n, why in sorted(self.skipped(task).items())},
                "revision": self.config["libraries"][task]["revision"],
                "source_test": self.config["libraries"][task]["source_test"],
            }
        scored = [t for t in tasks.values() if t["episodes"]]
        policy = [e for e in entries if e["kind"] == "policy"]
        prompts = sorted({e["prompt_sha256"] for e in policy if e.get("prompt_sha256")})
        return {
            "eval_seed": self.config["eval_seed"],
            "tasks": tasks,
            "overall": {
                "episodes": sum(t["episodes"] for t in tasks.values()),
                "missing": sum(t["missing"] for t in tasks.values()),
                "successes": sum(t["successes"] for t in tasks.values()),
                "macro_rate": (
                    round(sum(t["rate"] for t in scored) / len(scored), 4) if scored else None
                ),
                "tasks": len(scored),
            },
            "spend": {
                "launches": len(entries),
                "infra": sum(1 for e in entries if e["kind"] == "infra"),
                "input_tokens": sum(e.get("input_tokens") or 0 for e in policy),
                "cached_input_tokens": sum(e.get("cached_input_tokens") or 0 for e in policy),
                "output_tokens": sum(e.get("output_tokens") or 0 for e in policy),
                "episode_hours": round(sum(e["seconds"] for e in entries) / 3600, 2),
            },
            "distinct_prompts": len(prompts),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--tasks", nargs="+")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    evaluation = Evaluation(args.root)
    if args.action == "run":
        key = evaluation.config["execution"]["api_key_env"]
        subscription = evaluation.config["execution"].get("runtime") == "codex" and not key
        if not subscription and not os.environ.get(key or ""):
            raise SystemExit(f"environment variable {key} is empty")
        evaluation.resume()
        report = evaluation.run(args.tasks)
        if evaluation.paused():
            sys.stderr.write("PAUSED: the model service failed; fix it and run again\n")
            return 3
        sys.stdout.write(json.dumps(report["overall"], indent=1) + "\n")
    else:
        report = evaluation.report()
        (evaluation.root / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        sys.stdout.write(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
