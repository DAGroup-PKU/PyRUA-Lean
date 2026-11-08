"""Learn per task on dev, then test the final revision against v0 on unseen layouts.

``python -m pyrualean.learn.campaign ROOT init --config CONFIG``   create ROOT, freeze config
``python -m pyrualean.learn.campaign ROOT plan``                   planned episode counts
``python -m pyrualean.learn.campaign ROOT run [--tasks ...]``      learn + test (resumable)
``python -m pyrualean.learn.campaign ROOT report``                 dev curves and test table

When a model service fails (several policy episodes in a row ended by the provider, or the
worker CLI cannot reach its API) the campaign writes ``PAUSED.json`` and stops starting work;
nothing lost to the outage counts as an attempt.  ``run`` again (with a new credential if
needed): it checks both services, lifts the pause, and puts rounds whose worker died of the
outage back in the queue.

Each task is its own pipeline (one thread): round 0 runs v0 on the dev layouts; every later
round runs a worker on the previous batch, freezes its candidate and runs that on the dev
layouts.  The active revision is rolled back to the best earlier one when a batch's successes
fall ``rollback_if_worse_by`` or more below it.  After the last round the active revision and
v0 run on the test layouts, interleaved, once each.  Nothing from test reaches a worker.

The policy's credential is read at ``run`` from the variable ``execution.api_key_env`` into
this process's environment only; it is never written under ROOT and is withheld from workers.
The worker CLI uses its own login unless ``worker.api_key_env`` names a variable, whose key it
then gets as ``worker.token_env`` (``ANTHROPIC_API_KEY`` by default).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..evolve.digest import digest, render
from ..evolve.library import Library, diff
from . import config as configuration
from . import worker as workers
from .runner import BudgetExhausted, Runner, ServiceDown, environment, paused

V0 = "v0"


class WorkerFailed(RuntimeError):
    """A round's worker left no acceptable candidate; the task waits for a person.

    Inspect ``tasks/<task>/worker-r<N>/``; to retry, rename that directory (a fresh session
    starts on the next ``run``), or fix the candidate and delete ``accepted.json``.
    """


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    path.with_suffix(".tmp").replace(path)


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def scoring_mode(root: Path, requested: str | None) -> str:
    """Whether this campaign's workers see how RoboDojo scores a task (``on``) or not (``off``).

    Recorded in ``ROOT/worker_scoring`` at the first ``run`` that names it and fixed after; a
    campaign that never named it is ``on`` (as before the option).  A different value later, or
    ``off`` for a campaign whose workers already ran sighted, is refused.
    """

    path = root / "worker_scoring"
    recorded = path.read_text().strip() if path.exists() else None
    if requested is None:
        return recorded or "on"
    if requested not in workers.SCORING_MODES:
        raise ValueError(f"worker scoring must be one of {workers.SCORING_MODES}")
    if recorded is not None and recorded != requested:
        raise ValueError(f"{root}: worker scoring is {recorded}, fixed at its first run")
    if recorded is None:
        if requested != "on" and any((root / "tasks").glob("*/worker-r*/session.json")):
            raise ValueError(f"{root}: workers already ran with scoring on")
        path.write_text(requested + "\n")
    return requested


class Campaign:
    def __init__(
        self,
        root: str | Path,
        *,
        launch: Any = None,
        worker_launch: Any = None,
        scoring: str | None = None,
    ):
        self.root = Path(root).resolve()
        self.config_path = self.root / "config.json"
        self.config = configuration.load(self.config_path)
        self.scoring = scoring_mode(self.root, scoring)
        self.runner = Runner(self.root, self.config, launch=launch)
        self.worker_launch = worker_launch
        self.worker_slots = threading.Semaphore(int(self.config["worker"]["concurrency"]))
        self.log_lock = threading.Lock()

    # -- per task ------------------------------------------------------------

    def library(self, task: str) -> Library:
        library = Library(self.root / "library" / task)
        if library.active() is None:
            empty = self.root / "library" / ".empty"
            empty.mkdir(parents=True, exist_ok=True)
            revision = library.freeze(empty, parent=None, note="v0: empty library")
            library.set_active(revision)
            library.record({"round": 0, "revision": revision, "event": "v0"})
        return library

    def v0(self, task: str) -> str:
        library = self.library(task)
        return next(e["revision"] for e in library.promotions() if e.get("event") == "v0")

    def log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        with self.log_lock:
            with (self.root / "campaign.log").open("a") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    def batch(self, split: str, task: str, revision: str, repeat: int = 0) -> list[dict]:
        """One revision on every layout of a split, in parallel; judged entries only."""

        library = self.library(task)
        view = library.view(revision, task)
        layouts = self.config["test" if split == "test" else "dev"]["layouts"]
        with ThreadPoolExecutor(max_workers=len(layouts)) as pool:
            futures = [
                pool.submit(
                    self.runner.episode,
                    split=split,
                    task=task,
                    revision=revision,
                    layout=layout,
                    view=view,
                    repeat=repeat,
                )
                for layout in layouts
            ]
            entries = [f.result() for f in futures]
        return [e for e in entries if e is not None]

    def learn(self, task: str) -> str:
        """All rounds of ``task``; returns the revision to test."""

        library = self.library(task)
        state_path = self.root / "tasks" / task / "learn.json"
        state = _load(state_path) or {"rounds": []}
        for round_index in range(self.config["rounds"]):
            if round_index < len(state["rounds"]) and state["rounds"][round_index].get("done"):
                continue
            if round_index == 0:
                record = {"round": 0, "parent": None, "revision": self.v0(task)}
            else:
                record = self._improve(task, round_index, state)
            self.log(f"{task} r{round_index}: dev on {record['revision']}")
            entries = self.batch("dev", task, record["revision"], repeat=round_index)
            record["dev"] = summarize(entries)
            record["runs"] = [e["run_dir"] for e in entries]
            # success counts compare only between complete batches: an episode lost to
            # infrastructure is missing, not a failure, so a short batch neither rolls back
            # nor becomes the best one
            planned = len(self.config["dev"]["layouts"])
            best = max(
                (
                    r
                    for r in state["rounds"][:round_index]
                    if r.get("done") and r["dev"]["episodes"] == planned
                ),
                key=lambda r: (r["dev"]["successes"], r["dev"]["mean_stages"] or 0),
                default=None,
            )
            threshold = self.config["learning"]["rollback_if_worse_by"]
            if (
                best is not None
                and record["dev"]["episodes"] == planned
                and best["dev"]["successes"] - record["dev"]["successes"] >= threshold
            ):
                record["rolled_back_to"] = best["revision"]
                library.set_active(best["revision"])
            else:
                library.set_active(record["revision"])
            library.record(
                {
                    "round": round_index,
                    "event": "dev",
                    "revision": record["revision"],
                    "dev": record["dev"],
                    "active": library.active(),
                }
            )
            record["active"] = library.active()
            record["done"] = True
            state["rounds"][round_index : round_index + 1] = [record]
            _save(state_path, state)
            self.log(
                f"{task} r{round_index}: {record['dev']['successes']}/{record['dev']['episodes']}"
                f" successes; active {record['active']}"
            )
        return library.active()

    def _improve(self, task: str, round_index: int, state: dict) -> dict[str, Any]:
        library = self.library(task)
        previous = state["rounds"][round_index - 1]
        parent = previous["active"]
        session = self.root / "tasks" / task / f"worker-r{round_index}"
        blind = self.scoring == "off"
        if not (session / "session.json").exists():
            workers.prepare(
                session,
                task=task,
                round_index=round_index,
                parent=parent,
                library=library,
                campaign=self.root,
                config_path=self.config_path,
                config=self.config,
                episodes=episodes_text(
                    source_round(state["rounds"][:round_index], parent),
                    session=session if blind else None,
                ),
                history=history_text(
                    library, state["rounds"][:round_index], self.root, task, scoring=not blind
                ),
                scoring=self.scoring,
            )
        prepared = json.loads((session / "session.json").read_text()).get("scoring", "on")
        if prepared != self.scoring:
            raise ValueError(f"{session}: prepared with scoring {prepared}, not {self.scoring}")
        with self.worker_slots:
            self.log(f"{task} r{round_index}: worker on {parent}")
            outcome = workers.run(self.config, session, launch=self.worker_launch)
        accepted = _load(session / "accepted.json") or workers.accept(session, library)
        if accepted["accepted"]:
            _save(session / "accepted.json", accepted)
        else:
            # not accepted.json (a refusal is re-checked on the next run), but kept so that
            # ``requeue`` finds the held round on resume and gives it to a new worker
            _save(session / "refused.json", accepted)
        record = {
            "round": round_index,
            "parent": parent,
            "revision": accepted["revision"],
            "worker": {
                "exit": outcome.get("exit"),
                "seconds": outcome.get("seconds"),
                "cost_usd": outcome.get("total_cost_usd"),
                "accepted": accepted["accepted"],
                "problems": accepted.get("problems"),
                "replays": len(_load(session / "replays.json") or []),
            },
            "changes": diff(library, parent, accepted["revision"]),
        }
        if not accepted["accepted"]:
            self.log(f"{task} r{round_index}: candidate refused: {accepted['problems']}")
            # running the parent again would spend a round on no change: hold this task
            raise WorkerFailed(f"{task} r{round_index}: {accepted['problems']}")
        return record

    def test(self, task: str, final: str) -> dict[str, Any]:
        """v0 and ``final`` on the test layouts, interleaved; paired by layout."""

        v0 = self.v0(task)
        self.log(f"{task}: test {final} vs v0 {v0}")
        library = self.library(task)
        views = {rev: library.view(rev, task) for rev in {v0, final}}
        jobs = []
        for index, layout in enumerate(self.config["test"]["layouts"]):
            order = [v0, final] if index % 2 == 0 else [final, v0]
            jobs += [(rev, layout) for rev in dict.fromkeys(order)]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {
                job: pool.submit(
                    self.runner.episode,
                    split="test",
                    task=task,
                    revision=job[0],
                    layout=job[1],
                    view=views[job[0]],
                )
                for job in jobs
            }
            results = {job: future.result() for job, future in futures.items()}
        record = {"task": task, "v0": v0, "final": final, "pairs": []}
        for layout in self.config["test"]["layouts"]:
            a, b = results.get((v0, layout)), results.get((final, layout))
            record["pairs"].append(
                {
                    "layout": layout,
                    "v0": None if a is None else a["success"],
                    "final": None if b is None else b["success"],
                    "v0_stages": None if a is None else a["stages"],
                    "final_stages": None if b is None else b["stages"],
                }
            )
        _save(self.root / "tasks" / task / "test.json", record)
        return record

    def pipeline(self, task: str) -> None:
        try:
            final = self.learn(task)
            if self.config.get("run_test", True):
                self.test(task, final)
            self.log(f"{task}: done")
        except (BudgetExhausted, ServiceDown, WorkerFailed) as exc:
            self.log(f"{task}: stopped, {exc}")

    def report(self) -> dict[str, Any]:
        return report_of(self.root, self.config, self.runner)

    def resume(self, check: Any = None) -> None:
        """Before a run: both services answer, the pause is lifted, retries are recounted, and
        rounds whose worker died of a service failure are put back in the queue."""

        problems = (check or check_services)(self.config)
        if problems:
            raise ServiceDown("; ".join(problems))
        if (why := paused(self.root)) is not None:
            self.log(f"resuming after pause: {why['reason']}")
            (self.root / "PAUSED.json").rename(
                self.root / f"PAUSED-{time.strftime('%Y%m%d-%H%M%S')}.json"
            )
        self.runner.reconcile()
        for task in self.config["tasks"]:
            self.requeue(task)

    def requeue(self, task: str) -> None:
        """Void the first round whose worker failed for a service reason, and every later one.

        Their dev episodes stay in the ledger (and are reused if the redone round reaches the
        same revision); the voided records and worker sessions are kept, renamed.
        """

        state_path = self.root / "tasks" / task / "learn.json"
        state = _load(state_path)
        if not state:
            return
        task_dir = self.root / "tasks" / task
        # recorded rounds, and the next one (its worker may have finished before a stop)
        candidates = [r["round"] for r in state["rounds"] if r["round"] > 0]
        candidates.append(len(state["rounds"]))
        first = next((n for n in candidates if _service_failed(task_dir, n)), None)
        if first is None:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        voided = state["rounds"][first:]
        if voided:
            _save(self.root / "tasks" / task / f"voided-r{first}-{stamp}.json", voided)
        state["rounds"] = state["rounds"][:first]
        _save(state_path, state)
        library = self.library(task)
        library.set_active(state["rounds"][-1]["active"])
        library.record({"event": "requeue", "from_round": first, "active": library.active()})
        for session in (self.root / "tasks" / task).glob("worker-r*"):
            if re.fullmatch(r"worker-r\d+", session.name) and int(session.name[8:]) >= first:
                session.rename(session.with_name(f"{session.name}.void-{stamp}"))
        self.log(f"{task}: rounds {first}+ requeued (its worker left no change)")

    def run(self, tasks: list[str] | None = None) -> None:
        tasks = tasks or self.config["tasks"]
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            for future in [pool.submit(self.pipeline, task) for task in tasks]:
                future.result()
        _save(self.root / "report.json", self.report())


def _service_failed(task_dir: Path, round_index: int) -> bool:
    """The round ran without a change from its worker: an outage, or a worker that left nothing.

    Such a round measured the parent again instead of a change; it is redone.
    """

    session = task_dir / f"worker-r{round_index}"
    record = session / "worker.json"
    if not record.exists():
        return False
    if (session / "refused.json").exists():  # the candidate was refused: the task was held
        return True
    if workers.succeeded(json.loads(record.read_text())):
        return False
    attempts = [record, *session.glob("worker-a*.json")]
    if any(workers.service_error(json.loads(path.read_text())) for path in attempts):
        return True
    accepted = _load(session / "accepted.json")
    return accepted is not None and not accepted["accepted"]


def check_services(config: dict[str, Any], *, verbose: bool = False) -> list[str]:
    """Both model services answer from this host, with the campaign's environment and key.

    The worker check runs the CLI exactly as a worker runs (same environment, token and
    settings isolation), and confirms which token the gateway saw by its fingerprint."""

    import subprocess

    import httpx

    env = environment(config)
    problems = []
    execution = config["execution"]
    proxy = env.get("HTTPS_PROXY") or env.get("https_proxy")
    if execution.get("runtime") == "codex":
        # the CLI (e.g. tools/codex-mcp-only) logs in on its own home
        try:
            out = subprocess.run(
                [execution["codex_bin"], "login", "status"],
                env=env, capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL,
            )  # fmt: skip
            if out.returncode != 0:
                problems.append(f"policy CLI: {(out.stdout + out.stderr).strip()[:200]}")
            elif verbose:
                print(f"policy CLI ok: {execution['model']}, {out.stdout.strip()}")
        except (subprocess.TimeoutExpired, OSError) as exc:
            problems.append(f"policy CLI: {type(exc).__name__}: {exc}")
    else:
        key = os.environ.get(execution["api_key_env"], "")
        try:
            with httpx.Client(proxy=proxy, trust_env=False, timeout=60) as client:
                reply = client.post(
                    execution["base_url"].rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": execution["model"],
                        "messages": [{"role": "user", "content": "Reply with: ok"}],
                        "max_tokens": 16,
                    },
                )
            if reply.status_code != 200:
                problems.append(f"policy endpoint: HTTP {reply.status_code}")
        except httpx.HTTPError as exc:
            problems.append(f"policy endpoint: {type(exc).__name__}: {exc}")
    worker_env = workers.worker_environment(config, env)
    try:
        out = subprocess.run(
            [config["worker"]["command"], "-p", "Reply with: ok", "--output-format", "json",
             "--no-session-persistence", "--tools", "", *workers.isolation(config)],
            env=worker_env, capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL,
        )  # fmt: skip
        reply = json.loads(out.stdout or "{}")
        if out.returncode != 0 or reply.get("is_error"):
            problems.append(f"worker CLI: {str(reply.get('result') or out.stderr)[:200]}")
        elif verbose:
            print(
                f"worker CLI ok: model {list((reply.get('modelUsage') or {}).keys())}, "
                f"token ...{worker_env.get(config['worker']['token_env'], '')[-4:]}"
            )
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        problems.append(f"worker CLI: {type(exc).__name__}: {exc}")
    if verbose and not problems and execution.get("runtime") != "codex":
        print(
            f"policy ok: {execution['model']}, key ...{os.environ[execution['api_key_env']][-4:]}"
        )
    return problems


# -- texts for the worker -----------------------------------------------------------


def source_round(rounds: list[dict[str, Any]], parent: str) -> dict[str, Any]:
    """The latest dev batch of ``parent`` (after a rollback: the batch of the revision kept)."""

    return next(r for r in reversed(rounds) if r["revision"] == parent)


def episodes_text(record: dict[str, Any], *, session: Path | None = None) -> str:
    """The batch as the worker reads it; with ``session`` (scoring off): success only, the
    instruction, and run dirs in the session's filtered ``evidence/``."""

    measures = (
        f", mean score {record['dev'].get('mean_score')}, "
        f"mean stages {record['dev']['mean_stages']}"
        if session is None
        else ""
    )
    lines = [
        f"# Dev episodes of revision {record['revision']} (round {record['round']})",
        f"successes {record['dev']['successes']}/{record['dev']['episodes']}{measures}"
        + _overclaims(record["dev"]),
        "",
    ]
    for run_dir in record["runs"]:
        facts = digest(run_dir)
        result = json.loads((Path(run_dir) / "result.json").read_text())
        text = render(facts, scoring=session is None).rstrip().split("\n")
        if session is not None:
            text.insert(1, f"instruction: {result.get('task_language')}")
        lines += [
            *text,
            f"end: {result.get('end')}",
            f"run dir: {run_dir if session is None else workers.evidence_path(session, run_dir)}",
            "",
        ]
    return "\n".join(lines) + "\n"


def history_text(
    library: Library, rounds: list[dict], root: Path, task: str, *, scoring: bool = True
) -> str:
    lines = [f"# History of {task}", ""]
    for record in rounds:
        lines.append(
            f"## round {record['round']}: revision {record['revision']} (parent "
            f"{record.get('parent')})"
        )
        lines.append(
            f"dev: {record['dev']['successes']}/{record['dev']['episodes']} successes"
            + _overclaims(record["dev"])
            + (
                f", mean score {record['dev'].get('mean_score')}, "
                f"mean stages {record['dev']['mean_stages']}"
                if scoring
                else ""
            )
        )
        if record.get("rolled_back_to"):
            lines.append(f"ROLLED BACK to {record['rolled_back_to']}: this change made dev worse")
        if record.get("changes"):
            lines.append(f"changes: {json.dumps(record['changes'])}")
        findings = root / "tasks" / task / f"worker-r{record['round']}" / "FINDINGS.md"
        if findings.exists():
            lines.append(f"worker findings: {findings}")
        lines.append("")
    return "\n".join(lines) + "\n"


# -- report -----------------------------------------------------------------------


def _overclaims(dev: dict[str, Any]) -> str:
    """``, N overclaimed`` when the policy claimed success on N episodes the benchmark failed
    (absent in batches summarized before it was counted)."""

    return f", {dev['overclaims']} overclaimed" if dev.get("overclaims") else ""


def _claim(entry: dict[str, Any]) -> str | None:
    """The entry's ``claim``; read from its run dir for an entry recorded before the ledger
    carried it."""

    if "claim" in entry:
        return entry["claim"]
    return digest(entry["run_dir"])["claim"] if entry.get("run_dir") else None


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    stages = [e["stages"][0] for e in entries if e.get("stages") and e["stages"][0] is not None]
    scores = [float(e["score"]) for e in entries if e.get("score") is not None]
    return {
        "episodes": len(entries),
        "successes": sum(1 for e in entries if e["success"]),
        "mean_stages": round(sum(stages) / len(stages), 3) if stages else None,
        # RoboDojo's own per-episode score (1 on success, its partial credit otherwise); only
        # a mean every episode carries, so a missing record never reads as a lower score
        "mean_score": (
            round(sum(scores) / len(scores), 3) if scores and len(scores) == len(entries) else None
        ),
        "ends": _count(e.get("end") for e in entries),
        # the policy finished with success and the benchmark judged a failure
        "overclaims": sum(1 for e in entries if _claim(e) == "overclaim"),
    }


def _count(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def sign_test(wins: int, losses: int) -> float | None:
    """Two-sided exact binomial p over discordant pairs (None without any)."""

    n = wins + losses
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [p for p in pairs if p["v0"] is not None and p["final"] is not None]
    wins = sum(1 for p in complete if p["final"] and not p["v0"])
    losses = sum(1 for p in complete if p["v0"] and not p["final"])
    return {
        "pairs": len(complete),
        "missing": len(pairs) - len(complete),
        "v0_successes": sum(1 for p in complete if p["v0"]),
        "final_successes": sum(1 for p in complete if p["final"]),
        "final_only": wins,
        "v0_only": losses,
        "sign_test_p": sign_test(wins, losses),
    }


def report_of(root: Path, config: dict[str, Any], runner: Runner) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    all_pairs: list[dict[str, Any]] = []
    for task in config["tasks"]:
        learn = _load(root / "tasks" / task / "learn.json") or {"rounds": []}
        test = _load(root / "tasks" / task / "test.json")
        tasks[task] = {
            "dev": [
                {
                    "round": r["round"],
                    "revision": r["revision"],
                    **r["dev"],
                    "rolled_back_to": r.get("rolled_back_to"),
                    "worker": r.get("worker"),
                }
                for r in learn["rounds"]
                if r.get("done")
            ],
            "test": paired(test["pairs"]) if test else None,
            "final": test and test["final"],
        }
        if test:
            all_pairs += test["pairs"]
    entries = runner.entries()
    policy = [e for e in entries if e["kind"] == "policy" and not e["replay"]]
    tested = [t["test"] for t in tasks.values() if t["test"] and t["test"]["pairs"]]
    return {
        "tasks": tasks,
        "test_overall": paired(all_pairs) if all_pairs else None,
        "test_macro": {
            "v0": _mean([t["v0_successes"] / t["pairs"] for t in tested]),
            "final": _mean([t["final_successes"] / t["pairs"] for t in tested]),
            "tasks": len(tested),
        },
        "spend": {
            "launches": len(entries),
            "infra": sum(1 for e in entries if e["kind"] == "infra"),
            "replays": sum(1 for e in entries if e["replay"]),
            "policy_episodes": len(policy),
            "input_tokens": sum(e.get("input_tokens") or 0 for e in policy),
            "cached_input_tokens": sum(e.get("cached_input_tokens") or 0 for e in policy),
            "output_tokens": sum(e.get("output_tokens") or 0 for e in policy),
            "episode_hours": round(sum(e["seconds"] for e in entries) / 3600, 2),
            "worker_usd": round(
                sum(
                    (r.get("worker") or {}).get("cost_usd") or 0
                    for t in tasks.values()
                    for r in t["dev"]
                ),
                2,
            ),
        },
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


# -- CLI --------------------------------------------------------------------------


#: The variables the Claude CLI takes a credential from.
WORKER_TOKEN_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def use_key(config: dict[str, Any], key: str | None, worker_key: str | None = None) -> None:
    """This process (and so its episodes and workers) uses ``key`` for the policy and
    ``worker_key`` for the worker, as ``worker.token_env`` (the other credential variable of
    the CLI is dropped, and so is the worker's own variable, so that it reaches the worker only
    as its token).  Without ``worker_key`` the worker CLI uses its own login."""

    if key is not None:
        os.environ[config["execution"]["api_key_env"]] = key
    if worker_key is not None:
        for name in WORKER_TOKEN_VARIABLES:
            os.environ.pop(name, None)
        os.environ.pop(config["worker"]["api_key_env"], None)
        os.environ[config["worker"]["token_env"]] = worker_key


def key_from_environment(name: str) -> str:
    """A key given to this process as the environment variable ``name``."""

    if not os.environ.get(name):
        raise ValueError(f"environment variable {name} is empty")
    return os.environ[name]


def init(root: Path, config_path: Path) -> None:
    config = configuration.load(config_path)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} is not empty: a campaign is never re-initialized")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, root / "config.json")
    for task in config["tasks"]:
        (root / "tasks" / task).mkdir(parents=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyrualean.learn.campaign", description=__doc__)
    parser.add_argument("root", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init").add_argument("--config", type=Path, required=True)
    sub.add_parser("plan")
    run = sub.add_parser("run")
    run.add_argument("--tasks", nargs="+")
    run.add_argument(
        "--worker-scoring",
        choices=workers.SCORING_MODES,
        help="on: workers read RoboDojo's checks of the task and each episode's score and stages;"
        " off: only the instruction and success. Fixed at the first run (default: as recorded,"
        " else on)",
    )
    sub.add_parser("check-services", help="both models answer with these keys; no robot")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.action == "init":
        init(args.root, args.config)
        sys.stdout.write(json.dumps(configuration.planned(configuration.load(args.config))) + "\n")
        return 0
    campaign = Campaign(args.root, scoring=getattr(args, "worker_scoring", None))
    if args.action == "plan":
        sys.stdout.write(json.dumps(configuration.planned(campaign.config), indent=1) + "\n")
    elif args.action in ("run", "check-services"):
        execution = campaign.config["execution"]
        # runtime codex: the policy CLI logs in on its own home (configuration.check)
        key = None if execution.get("runtime") == "codex" else execution["api_key_env"]
        worker_env = campaign.config["worker"].get("api_key_env")
        use_key(
            campaign.config,
            key_from_environment(key) if key else None,
            key_from_environment(worker_env) if worker_env else None,
        )
        if args.action == "check-services":
            found = check_services(campaign.config, verbose=True)
            sys.stdout.write("\n".join(found or ["both services answer"]) + "\n")
            return 1 if found else 0
        campaign.resume()
        campaign.run(args.tasks)
        if paused(campaign.root):
            sys.stderr.write("PAUSED: a model service failed; fix it and run again\n")
            return 3
        sys.stdout.write(json.dumps(campaign.report()["test_overall"], indent=1) + "\n")
    else:
        report = campaign.report()
        _save(campaign.root / "report.json", report)
        sys.stdout.write(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
