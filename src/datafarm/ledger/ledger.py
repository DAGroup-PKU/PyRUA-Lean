"""Pipeline run ledger: one tracked record per requested pipeline invocation.

A *run* is created whenever an operator requests a full pipeline pass or a
partial pass (a cohort of items from an entry stage to an exit stage). The run
directory is the single operational authority for that request:

``run.json``
    Immutable request and plan: who asked for what, the stage range, the fixed
    item denominator, code/config fingerprints, and optional native-state sync.
``matrix.json``
    Item x stage(/substage) status table. Gaps are read from here.
``workers.json``
    Every process or scheduler lease that works on the run, with liveness hints.
``events.jsonl``
    Append-only history: creation, launches, exits, declarations, finalization.

Run status is *derived* from these files plus live probes; a process claiming
``running`` is never trusted on its own. The module is standard-library only.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import importlib
import json
import os
import re
import socket
import subprocess
from datafarm.common import atomic_write_json, read_json, sha256_file, utc_now
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

RUN_SCHEMA = "datafarm.pipeline-run/1.0.0"
MATRIX_SCHEMA = "datafarm.pipeline-run-matrix/1.0.0"
WORKERS_SCHEMA = "datafarm.pipeline-run-workers/1.0.0"

MAIN_STAGES = ("grounding", "task", "episode", "render", "admission", "training")
CELL_STATES = ("pending", "running", "passed", "rejected", "failed", "skipped")
DECLARABLE = ("running", "blocked", "failed", "cancelled")
RUN_ID_PATTERN = re.compile(r"^PRUN-\d{8}-\d{4}-[a-z0-9][a-z0-9-]{0,63}$")
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
DEFAULT_STALE_SECONDS = 900


class LedgerError(RuntimeError):
    pass


def ledger_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    env = os.environ.get("DATAFARM_LEDGER_ROOT")
    if env:
        return Path(env).resolve()
    return (Path.cwd() / "runs" / "ledger").resolve()


def make_run_id(slug: str, *, now: dt.datetime | None = None) -> str:
    moment = (now or dt.datetime.now().astimezone()).strftime("%Y%m%d-%H%M")
    run_id = f"PRUN-{moment}-{slug}"
    if not RUN_ID_PATTERN.match(run_id):
        raise LedgerError(f"invalid run slug {slug!r}; use lowercase kebab-case")
    return run_id


def git_fingerprint(root: str | Path) -> dict[str, Any]:
    """Record the code version. A repo without commits is recorded, not hidden."""

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise LedgerError(result.stderr.strip() or "git failed")
        return result.stdout

    try:
        status = git("status", "--porcelain")
    except (LedgerError, OSError, subprocess.TimeoutExpired) as exc:
        return {"root": str(root), "available": False, "error": str(exc)[:200]}
    try:
        commit: str | None = git("rev-parse", "HEAD").strip()
    except LedgerError:
        commit = None
    return {
        "root": str(Path(root).resolve()),
        "available": True,
        "commit": commit,
        "dirty_paths": len([line for line in status.splitlines() if line.strip()]),
    }


@dataclass(frozen=True)
class StagePlan:
    stage: str
    substages: tuple[str, ...] = ()

    def columns(self) -> tuple[str, ...]:
        if not self.substages:
            return (self.stage,)
        return tuple(f"{self.stage}/{name}" for name in self.substages)

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "substages": list(self.substages)}


def plan_stages(
    entry: str, exit: str, substages: Mapping[str, Sequence[str]] | None = None
) -> tuple[StagePlan, ...]:
    for name in (entry, exit):
        if name not in MAIN_STAGES:
            raise LedgerError(f"unknown stage {name!r}; expected one of {MAIN_STAGES}")
    start, stop = MAIN_STAGES.index(entry), MAIN_STAGES.index(exit)
    if start > stop:
        raise LedgerError(f"entry stage {entry!r} comes after exit stage {exit!r}")
    selected = MAIN_STAGES[start : stop + 1]
    substages = dict(substages or {})
    unknown = set(substages) - set(selected)
    if unknown:
        raise LedgerError(f"substages given for stages outside the range: {sorted(unknown)}")
    plans = []
    for stage in selected:
        names = tuple(substages.get(stage, ()))
        for name in names:
            if not NAME_PATTERN.match(name):
                raise LedgerError(f"invalid substage name {name!r}")
        if len(set(names)) != len(names):
            raise LedgerError(f"duplicate substage under {stage!r}")
        plans.append(StagePlan(stage, names))
    return tuple(plans)


@dataclass(frozen=True)
class CellUpdate:
    item: str
    column: str
    status: str
    attempt: int | None = None
    failure: str | None = None
    artifact: str | None = None
    note: str | None = None


# ---------------------------------------------------------------------------
# Liveness probes. Hosts may differ from the one running the status check, so
# remote PIDs are probed over SSH only when an SSH config is supplied.


def local_host_names() -> set[str]:
    names = {"localhost", socket.gethostname(), socket.getfqdn()}
    return {name for name in names if name}


def probe_pid(host: str | None, pid: int | None) -> bool | None:
    """Return True/False when the PID state is known, None when it cannot be probed."""

    if not pid:
        return None
    if not host or host in local_host_names():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    config = os.environ.get("DATAFARM_SSH_CONFIG")
    if not config:
        return None
    env = dict(os.environ)
    if os.environ.get("DATAFARM_SSH_HOME"):
        env["HOME"] = os.environ["DATAFARM_SSH_HOME"]
    env.pop("SSH_AUTH_SOCK", None)
    command = [
        "ssh", "-F", config, "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
        host, f"kill -0 {int(pid)} 2>/dev/null && echo alive || echo dead",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    if output.endswith("alive"):
        return True
    if output.endswith("dead"):
        return False
    return None


def probe_lease(lease_id: str | None) -> bool | None:
    directory = os.environ.get("DATAFARM_LEASE_DIR")
    if not lease_id or not directory or not Path(directory).is_dir():
        return None
    for path in Path(directory).glob("*.json"):
        try:
            if read_json(path).get("id") == lease_id:
                return True
        except (OSError, ValueError, AttributeError):
            continue
    return False


def newest_mtime(paths: Iterable[str]) -> float | None:
    newest = None
    for value in paths:
        try:
            mtime = Path(value).stat().st_mtime
        except OSError:
            continue
        newest = mtime if newest is None else max(newest, mtime)
    return newest


# ---------------------------------------------------------------------------


class RunLedger:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        if not (self.directory / "run.json").is_file():
            raise LedgerError(f"no pipeline run at {self.directory}")

    # -- construction -----------------------------------------------------
    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        run_id: str,
        title: str,
        request: str,
        task_family: str,
        stages: Sequence[StagePlan],
        items: Sequence[str],
        authorization: str | None = None,
        cohort_manifest: str | None = None,
        code: Mapping[str, Any] | None = None,
        configs: Sequence[str] = (),
        parent_run: str | None = None,
        links: Sequence[str] = (),
        sync: Sequence[Mapping[str, Any]] = (),
        stale_seconds: int = DEFAULT_STALE_SECONDS,
        created_at: str | None = None,
        backfilled_from: str | None = None,
    ) -> "RunLedger":
        if not RUN_ID_PATTERN.match(run_id):
            raise LedgerError(f"invalid run id {run_id!r}")
        if not request.strip():
            raise LedgerError("a run must record the operator request")
        if not stages:
            raise LedgerError("a run needs at least one stage")
        items = [str(item) for item in items]
        if not items:
            raise LedgerError("a run needs a fixed, non-empty item denominator")
        if len(set(items)) != len(items):
            raise LedgerError("duplicate item ids in the denominator")
        directory = Path(root).resolve() / run_id
        if directory.exists():
            raise LedgerError(f"run {run_id} already exists")
        cohort: dict[str, Any] = {"items": items, "count": len(items)}
        if cohort_manifest:
            cohort["manifest"] = str(Path(cohort_manifest).resolve())
            cohort["manifest_sha256"] = sha256_file(cohort_manifest)
        created = created_at or utc_now()
        run = {
            "schema": RUN_SCHEMA,
            "run_id": run_id,
            "created_at": created,
            "recorded_at": utc_now(),
            "title": title,
            "request": request,
            "authorization": authorization,
            "task_family": task_family,
            "entry_stage": stages[0].stage,
            "exit_stage": stages[-1].stage,
            "stages": [stage.to_dict() for stage in stages],
            "cohort": cohort,
            "code": dict(code or {}),
            "configs": {str(Path(p).resolve()): sha256_file(p) for p in configs},
            "parent_run": parent_run,
            "links": list(links),
            "sync": [dict(entry) for entry in sync],
            "stale_seconds": int(stale_seconds),
            "backfilled_from": backfilled_from,
        }
        for entry in run["sync"]:
            if not isinstance(entry.get("callable"), str) or ":" not in entry["callable"]:
                raise LedgerError("sync entries need callable='module:function'")
        columns = [column for stage in stages for column in stage.columns()]
        matrix = {
            "schema": MATRIX_SCHEMA,
            "columns": columns,
            "cells": {item: {column: {"status": "pending"} for column in columns} for item in items},
            "updated_at": created,
        }
        directory.mkdir(parents=True)
        atomic_write_json(directory / "run.json", run)
        atomic_write_json(directory / "matrix.json", matrix)
        atomic_write_json(directory / "workers.json", {"schema": WORKERS_SCHEMA, "workers": []})
        ledger = cls(directory)
        ledger.record_event(
            "created",
            title=title,
            entry_stage=run["entry_stage"],
            exit_stage=run["exit_stage"],
            items=len(items),
            backfilled_from=backfilled_from,
        )
        if parent_run:
            with contextlib.suppress(LedgerError):
                cls(Path(root).resolve() / parent_run).record_event("child_run", run_id=run_id)
        return ledger

    # -- basic io ---------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self.directory.name

    def run(self) -> dict[str, Any]:
        return read_json(self.directory / "run.json")

    def matrix(self) -> dict[str, Any]:
        return read_json(self.directory / "matrix.json")

    def workers(self) -> list[dict[str, Any]]:
        return list(read_json(self.directory / "workers.json")["workers"])

    def events(self) -> list[dict[str, Any]]:
        path = self.directory / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with open(self.directory / ".lock", "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def record_event(self, kind: str, **data: Any) -> dict[str, Any]:
        event = {"at": utc_now(), "kind": kind, **{k: v for k, v in data.items() if v is not None}}
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self._locked(), open(self.directory / "events.jsonl", "a", encoding="utf-8") as out:
            out.write(line + "\n")
            out.flush()
            os.fsync(out.fileno())
        return event

    # -- item matrix ------------------------------------------------------
    def set_cells(self, updates: Iterable[CellUpdate], *, source: str) -> int:
        """Apply cell updates; returns how many cells changed status."""

        updates = list(updates)
        changes: list[list[Any]] = []
        with self._locked():
            matrix = self.matrix()
            columns = set(matrix["columns"])
            for update in updates:
                if update.status not in CELL_STATES:
                    raise LedgerError(f"unknown cell status {update.status!r}")
                if update.item not in matrix["cells"]:
                    raise LedgerError(f"item {update.item!r} is outside the run denominator")
                if update.column not in columns:
                    raise LedgerError(f"column {update.column!r} is not in the run plan")
                cell = matrix["cells"][update.item][update.column]
                previous = cell.get("status")
                for key in ("attempt", "failure", "artifact", "note"):
                    value = getattr(update, key)
                    if value is not None:
                        cell[key] = value
                if update.status == "passed":
                    cell.pop("failure", None)
                if previous != update.status:
                    cell["status"] = update.status
                    cell["updated_at"] = utc_now()
                    changes.append([update.item, update.column, previous, update.status])
            if changes:
                matrix["updated_at"] = utc_now()
                atomic_write_json(self.directory / "matrix.json", matrix)
        if changes:
            self.record_event("cells", source=source, changes=changes)
        return len(changes)

    # -- workers ----------------------------------------------------------
    def register_worker(
        self,
        *,
        role: str,
        host: str | None = None,
        pid: int | None = None,
        sid: int | None = None,
        lease_id: str | None = None,
        log: str | None = None,
        progress_paths: Sequence[str] = (),
        command: Sequence[str] | None = None,
        started_at: str | None = None,
        note: str | None = None,
    ) -> str:
        with self._locked():
            data = read_json(self.directory / "workers.json")
            worker_id = f"w{len(data['workers']) + 1:03d}-{role}"
            data["workers"].append(
                {
                    "worker_id": worker_id,
                    "role": role,
                    "host": host,
                    "pid": pid,
                    "sid": sid,
                    "lease_id": lease_id,
                    "log": log,
                    "progress_paths": list(progress_paths),
                    "command": list(command) if command else None,
                    "started_at": started_at or utc_now(),
                    "ended_at": None,
                    "exit_code": None,
                    "note": note,
                }
            )
            atomic_write_json(self.directory / "workers.json", data)
        self.record_event("worker_registered", worker=worker_id, role=role, host=host, pid=pid,
                          lease_id=lease_id)
        return worker_id

    def update_worker(self, worker_id: str, **values: Any) -> None:
        allowed = {"pid", "sid", "host", "log", "progress_paths", "ended_at", "exit_code", "note",
                   "lease_id"}
        if set(values) - allowed:
            raise LedgerError(f"cannot update worker fields {sorted(set(values) - allowed)}")
        with self._locked():
            data = read_json(self.directory / "workers.json")
            for worker in data["workers"]:
                if worker["worker_id"] == worker_id:
                    worker.update(values)
                    break
            else:
                raise LedgerError(f"unknown worker {worker_id}")
            atomic_write_json(self.directory / "workers.json", data)

    def close_worker(self, worker_id: str, *, reason: str, exit_code: int | None = None) -> bool:
        """End a worker whose own process no longer reports (native terminal state)."""

        for worker in self.workers():
            if worker["worker_id"] == worker_id and not worker.get("ended_at"):
                self.end_worker(worker_id, exit_code, reason=reason)
                return True
        return False

    def end_worker(self, worker_id: str, exit_code: int | None, *, reason: str | None = None) -> None:
        self.update_worker(worker_id, ended_at=utc_now(), exit_code=exit_code)
        self.record_event("worker_exited", worker=worker_id, exit_code=exit_code, reason=reason)

    def heartbeat_path(self, worker_id: str) -> Path:
        return self.directory / "heartbeats" / f"{worker_id}.hb"

    def heartbeat(self, worker_id: str) -> None:
        path = self.heartbeat_path(worker_id)
        path.parent.mkdir(exist_ok=True)
        path.write_text(utc_now() + "\n", encoding="utf-8")

    # -- declarations -----------------------------------------------------
    def declare(self, status: str, *, reason: str, source: str) -> None:
        if status not in DECLARABLE:
            raise LedgerError(f"declared status must be one of {DECLARABLE}")
        self.record_event("declared", status=status, reason=reason, source=source)

    def mirror_declaration(self, status: str, *, reason: str, source: str) -> bool:
        """Declare only when the native state changed; keeps the event log quiet."""

        current = effective_declaration(self.events())
        if current and (current["status"], current.get("reason")) == (status, reason):
            return False
        self.declare(status, reason=reason, source=source)
        return True

    def finalize(self, verification: str | Path, *, source: str, allow_gaps: bool = False) -> None:
        verification = Path(verification)
        if not verification.is_file():
            raise LedgerError(f"final verification {verification} does not exist")
        gaps = self.gaps()
        if gaps and not allow_gaps:
            raise LedgerError(f"{len(gaps)} items have not passed the exit column; "
                              "resolve them or finalize with allow_gaps")
        self.record_event(
            "finalized",
            source=source,
            verification=str(verification.resolve()),
            verification_sha256=sha256_file(verification),
            gap_items=sorted(gaps),
        )

    def supersede(self, by_run: str, *, reason: str) -> None:
        self.record_event("superseded", by_run=by_run, reason=reason)

    # -- derived views ----------------------------------------------------
    def exit_column(self) -> str:
        return self.matrix()["columns"][-1]

    def gaps(self) -> dict[str, dict[str, Any]]:
        """Items that have not passed the final column, with where they stopped."""

        matrix = self.matrix()
        columns = matrix["columns"]
        result = {}
        for item, cells in matrix["cells"].items():
            if cells[columns[-1]]["status"] == "passed":
                continue
            stopped = next(
                (c for c in columns if cells[c]["status"] != "passed"), columns[-1]
            )
            cell = cells[stopped]
            result[item] = {"column": stopped, **cell}
        return result

    def column_counts(self) -> dict[str, dict[str, int]]:
        matrix = self.matrix()
        counts = {column: {} for column in matrix["columns"]}
        for cells in matrix["cells"].values():
            for column, cell in cells.items():
                counts[column][cell["status"]] = counts[column].get(cell["status"], 0) + 1
        return counts

    def sync(self) -> list[dict[str, Any]]:
        """Run the run's native-state sync callables; errors are recorded, not raised."""

        results = []
        for entry in self.run().get("sync", []):
            target = entry["callable"]
            try:
                module_name, function_name = target.split(":", 1)
                function: Callable[..., Any] = getattr(
                    importlib.import_module(module_name), function_name
                )
                outcome = function(self, **dict(entry.get("args", {}))) or {}
                results.append({"callable": target, "ok": True, **outcome})
            except Exception as exc:  # noqa: BLE001 - a broken probe must not hide status
                results.append({"callable": target, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
        return results

    def status(self, *, now: dt.datetime | None = None, probe: bool = True) -> dict[str, Any]:
        now = now or dt.datetime.now(dt.timezone.utc)
        run = self.run()
        stale = int(run.get("stale_seconds", DEFAULT_STALE_SECONDS))
        events = self.events()
        worker_views = []
        for worker in self.workers():
            view = dict(worker)
            paths = list(worker.get("progress_paths") or [])
            paths.append(str(self.heartbeat_path(worker["worker_id"])))
            if worker.get("log"):
                paths.append(worker["log"])
            mtime = newest_mtime(paths)
            view["last_progress_age_s"] = (
                None if mtime is None else int(now.timestamp() - mtime)
            )
            if worker.get("ended_at"):
                view["liveness"] = "exited"
            else:
                pid_state = probe_pid(worker.get("host"), worker.get("pid")) if probe else None
                lease_state = probe_lease(worker.get("lease_id")) if probe else None
                fresh = view["last_progress_age_s"] is not None and view["last_progress_age_s"] <= stale
                if pid_state is True:
                    view["liveness"] = "alive" if fresh else "alive_stalled"
                elif pid_state is False:
                    view["liveness"] = "dead"
                elif lease_state is True:
                    view["liveness"] = "lease_held" if fresh else "lease_stalled"
                elif lease_state is False and worker.get("lease_id"):
                    view["liveness"] = "dead"
                else:
                    view["liveness"] = "unknown_fresh" if fresh else "unknown_stale"
                view["pid_probe"] = pid_state
                view["lease_probe"] = lease_state
            worker_views.append(view)

        finalized = [e for e in events if e["kind"] == "finalized"]
        superseded = [e for e in events if e["kind"] == "superseded"]
        last_declared = effective_declaration(events)
        gaps = self.gaps()
        live = [w for w in worker_views if w["liveness"] in
                {"alive", "alive_stalled", "lease_held", "lease_stalled", "unknown_fresh"}]
        stalled = [w for w in live if w["liveness"] in {"alive_stalled", "lease_stalled"}]
        lost = [w for w in worker_views if w["liveness"] in {"dead", "unknown_stale"}]
        failed_exits = [w for w in worker_views if w["liveness"] == "exited" and w.get("exit_code") not in (0, None)]

        attention: list[str] = []
        if finalized:
            state = "complete" if not finalized[-1].get("gap_items") else "complete_with_gaps"
        elif superseded:
            state = "superseded"
        elif last_declared and last_declared["status"] == "cancelled":
            state = "cancelled"
        elif live:
            state = "running_stalled" if len(stalled) == len(live) else "running"
            if lost:
                attention.append(f"{len(lost)} worker(s) lost while others still run")
        elif last_declared and last_declared["status"] in {"blocked", "failed"}:
            state = last_declared["status"]
            attention.append(f"declared {last_declared['status']}: {last_declared.get('reason')}")
        elif lost:
            state = "lost"
            attention.append("no live worker, at least one died without an exit record")
        elif failed_exits:
            state = "exited_with_error"
        elif not worker_views:
            state = "registered"
        else:
            state = "idle_incomplete" if gaps else "idle_awaiting_finalize"
        if failed_exits:
            attention.append(
                "non-zero exits: "
                + ", ".join(f"{w['worker_id']}={w['exit_code']}" for w in failed_exits)
            )
        if gaps and state in {"lost", "exited_with_error", "idle_incomplete", "blocked", "failed"}:
            attention.append(f"{len(gaps)} item(s) not passed at {self.exit_column()}")

        by_stop: dict[str, dict[str, int]] = {}
        for gap in gaps.values():
            bucket = by_stop.setdefault(gap["column"], {})
            bucket[gap["status"]] = bucket.get(gap["status"], 0) + 1
        return {
            "run_id": run["run_id"],
            "title": run["title"],
            "task_family": run["task_family"],
            "range": f"{run['entry_stage']}->{run['exit_stage']}",
            "created_at": run["created_at"],
            "state": state,
            "attention": attention,
            "denominator": run["cohort"]["count"],
            "passed_exit": run["cohort"]["count"] - len(gaps),
            "gaps_by_stop": by_stop,
            "column_counts": self.column_counts(),
            "workers": worker_views,
            "last_declared": last_declared,
            "last_event_at": events[-1]["at"] if events else None,
            "parent_run": run.get("parent_run"),
            "children": [e["run_id"] for e in events if e["kind"] == "child_run"],
            "links": run.get("links", []),
        }


def effective_declaration(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Latest declaration, unless a worker was registered after it (a relaunch)."""

    current = None
    for event in events:
        if event["kind"] == "declared":
            current = event
        elif event["kind"] == "worker_registered":
            current = None
    return current


def list_runs(root: str | Path) -> list[RunLedger]:
    root = Path(root)
    if not root.is_dir():
        return []
    return [RunLedger(path) for path in sorted(root.iterdir()) if (path / "run.json").is_file()]


def attach_from_env() -> tuple[RunLedger, str] | None:
    """Supervisors launched through ``tools/datafarm_runs.py launch`` can report cells directly."""

    directory = os.environ.get("DATAFARM_LEDGER_RUN_DIR")
    worker = os.environ.get("DATAFARM_LEDGER_WORKER")
    if not directory or not worker:
        return None
    return RunLedger(directory), worker
