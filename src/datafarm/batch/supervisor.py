"""Generic batch supervisor: one bounded, resumable loop controller for per-item attempts.

What it replaces
----------------
A naive per-task supervisor loops ``while True`` over "missing or rejected" items. Without a round,
cost or zero-progress bound it can spin on no-ops indefinitely, it counts provider/upload crashes
against a sample's seed budget, and it may treat a failed lease query as "every lease was lost".
This module is the single, tested version of that loop. Task specifics (how to launch, poll and verify one attempt) are injected as
callables; the controller owns attempts, seeds, budgets, liveness semantics, durable state and the
terminal status.

Model
-----
* **Fixed denominator.** Item ids are given up front, stored in ``state.json`` and never change.
  Every report counts against all of them.
* **Attempt = one seed.** An attempt that *finished* is verified at once (streaming verification)
  and consumes one unit of the item's attempt budget, whatever the verdict (pass, reject,
  abstain).
* **Crash = ended without a result.** It is retried with the *same* attempt index and seed under a
  separate per-item crash budget, so crashes never burn seeds. Crashes from several distinct items
  inside a short window form a *burst* and trigger a self-resuming exponential backoff.
* **Unknown is not lost.** An attempt whose liveness cannot be established stays in flight and is
  never relaunched. It becomes a crash only when the poll has no positive evidence of life *and* a
  lease probe reports it dead on ``lost_confirmations`` consecutive polls. Too many consecutive
  unknowns pause the batch with ``paused_unknown_liveness``.
* **Provider down** pauses all launches. Paused time is excluded from the wall budget; the total
  pause is bounded (``paused_provider_down``).
* **Rounds are attempt generations.** Round *g* closes once every open item is past attempt *g*.
  ``zero_progress_rounds`` consecutive closed rounds without a newly admitted item stop the batch.
* **Systematic failure.** Over the latest verdict of the most recent distinct items, a single gate
  failing a large share stops new launches (``stopped_systematic_failure``) with a report.
* **Durability.** ``state.json`` is written atomically (temp file, fsync, rename). ``events.jsonl``
  is append-only. The launch *intent* is persisted before ``launch`` is called, so a supervisor that
  dies mid-launch reconciles on restart instead of launching twice.

The module is standard-library only and contains no task, provider or path knowledge.
"""

from __future__ import annotations

from datafarm import common
from datafarm.common import utc_now as iso
import contextlib
import dataclasses
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from datafarm.common import sha256_text

STATE_SCHEMA = "datafarm.batch-supervisor-state/1.0.0"
REPORT_SCHEMA = "datafarm.batch-supervisor-report/1.0.0"
POLICY_SCHEMA = "datafarm.batch-policy/1.0.0"

# -- item states -------------------------------------------------------------------------------
PENDING = "pending"  # never launched
LAUNCHING = "launching"  # intent persisted, launch outcome not yet known
RUNNING = "running"  # handle persisted, attempt in flight
UNKNOWN = "unknown"  # in flight, liveness cannot be established; never relaunched
CRASHED_RETRYABLE = "crashed_retryable"  # same attempt and seed will be relaunched
REJECTED_RETRYABLE = "rejected_retryable"  # next attempt (new seed) will be launched
ADMITTED = "admitted"  # terminal: verifier passed
REJECTED_FINAL = "rejected_final"  # terminal: final reject or permanent error
EXHAUSTED = "exhausted"  # terminal: attempt budget, crash budget or max rounds used up

ITEM_STATES = (PENDING, LAUNCHING, RUNNING, UNKNOWN, CRASHED_RETRYABLE, REJECTED_RETRYABLE,
               ADMITTED, REJECTED_FINAL, EXHAUSTED)
IN_FLIGHT = frozenset({LAUNCHING, RUNNING, UNKNOWN})
ELIGIBLE = frozenset({PENDING, CRASHED_RETRYABLE, REJECTED_RETRYABLE})
TERMINAL_ITEM_STATES = frozenset({ADMITTED, REJECTED_FINAL, EXHAUSTED})

# -- batch statuses ----------------------------------------------------------------------------
RUNNING_STATUS = "running"
STOPPING = "stopping"  # a stop was requested; in-flight attempts are drained, nothing launches
COMPLETE = "complete"
COMPLETE_WITH_GAPS = "complete_with_gaps"
STOPPED_ZERO_PROGRESS = "stopped_zero_progress"
STOPPED_SYSTEMATIC_FAILURE = "stopped_systematic_failure"
STOPPED_COST_CAP = "stopped_cost_cap"
STOPPED_WALL_BUDGET = "stopped_wall_budget"
STOPPED_OPERATOR = "stopped_operator"
PAUSED_PROVIDER_DOWN = "paused_provider_down"
PAUSED_UNKNOWN_LIVENESS = "paused_unknown_liveness"
PAUSED_COST_UNKNOWN = "paused_cost_unknown"

TERMINAL_STATUSES = frozenset({
    COMPLETE, COMPLETE_WITH_GAPS, STOPPED_ZERO_PROGRESS, STOPPED_SYSTEMATIC_FAILURE,
    STOPPED_COST_CAP, STOPPED_WALL_BUDGET, STOPPED_OPERATOR, PAUSED_PROVIDER_DOWN,
    PAUSED_UNKNOWN_LIVENESS, PAUSED_COST_UNKNOWN,
})
PAUSED_STATUSES = frozenset({PAUSED_PROVIDER_DOWN, PAUSED_UNKNOWN_LIVENESS,
                            PAUSED_COST_UNKNOWN})

POLL_STATUSES = ("running", "finished", "crashed", "unknown")
CRASH_KINDS = ("transient", "provider_down", "permanent", "capacity")
DECISIONS = ("pass", "reject", "abstain")
LEASE_STATES = ("alive", "dead", "unknown")

# Pipeline run ledger cell status for each item state (see ledger.CELL_STATES).
LEDGER_CELL_STATUS = {
    PENDING: "pending", LAUNCHING: "running", RUNNING: "running", UNKNOWN: "running",
    CRASHED_RETRYABLE: "running", REJECTED_RETRYABLE: "rejected", ADMITTED: "passed",
    REJECTED_FINAL: "failed", EXHAUSTED: "failed",
}


class BatchError(RuntimeError):
    pass


class DenominatorMismatch(BatchError):
    """A resume tried to change the fixed item denominator."""


class SupervisorBusy(BatchError):
    """Another live supervisor holds the state directory lock."""


class ProviderDown(Exception):
    """Raised by ``launch`` when the provider endpoint is unavailable; nothing was started."""


class LaunchNotStarted(Exception):
    """Raised by ``launch`` when nothing was started; counted as a transient crash."""


class LaunchDeferred(Exception):
    """Raised by ``launch`` when capacity is unavailable; nothing started, no budget is used."""


class LaunchOutcomeUnknown(Exception):
    """Raised by ``launch`` when the attempt may or may not have started."""


def write_json_atomic(path: str | Path, value: Any, *, fsync: bool = True) -> None:
    """Write JSON through a temporary file in the same directory, fsync it, then rename."""
    common.atomic_write_json(path, value, fsync=fsync, fsync_dir=fsync)


def read_events(state_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(state_dir) / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def default_seed(base: int, item: str, attempt: int, modulus: int = 2_147_483_647) -> int:
    """Deterministic, restart-stable seed in ``[0, modulus)``; distinct per item and attempt.

    The default modulus is the int32 maximum some providers enforce: an unbounded linear seed
    formula once started failing permanently after enough retry rounds.
    """

    digest = hashlib.sha256(f"{int(base)}\0{item}\0{int(attempt)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(modulus)


def launch_key(item: str, attempt: int, tries: int) -> str:
    """Deterministic id of one launch: ``<item>.aNN.tNN`` (crash retries get new ``tNN``)."""

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(item))[:80] or "item"
    if safe != item:  # keep ids unique when sanitizing changed the item id
        safe = f"{safe[:60]}-{sha256_text(str(item))[:8]}"
    return f"{safe}.a{int(attempt):02d}.t{int(tries):02d}"


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"launch handles must be JSON-serializable: {exc}") from exc


def _short(value: Any, limit: int = 300) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _backoff(base: float, level: int, cap: float) -> float:
    """``base * 2**(level-1)`` capped; the exponent is bounded so long crash loops cannot
    overflow a float (a replayed cohort had 1113 consecutive crash retries of one item)."""

    return min(float(base) * 2.0 ** min(max(int(level) - 1, 0), 60), float(cap))


def _accepts_keyword(function: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD or
               (p.name == name and p.kind in (inspect.Parameter.KEYWORD_ONLY,
                                               inspect.Parameter.POSITIONAL_OR_KEYWORD))
               for p in parameters)


# -- policy ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BatchPolicy:
    """Loop-control policy. Every bound is explicit; ``from_dict`` rejects unknown keys."""

    # budgets
    max_attempts_per_item: int = 8  # finished (verified) attempts = seeds per item
    max_crash_retries_per_item: int = 8  # relaunches after crashes; never consume seeds
    max_concurrency: int = 10
    max_rounds: int = 12  # hard ceiling on attempt generations (defence in depth)
    zero_progress_rounds: int = 3  # consecutive closed rounds without a new admission; 0 = off
    max_total_cost: float | None = None
    launch_cost_estimate: float = 0.0  # reserve this much per in-flight attempt before launching
    max_wall_seconds: float | None = None  # active time; provider-down pauses are excluded
    # crashes
    crash_retry_delay_s: float = 30.0  # per-item delay before a crash retry, doubling per crash
    crash_retry_delay_cap_s: float = 900.0
    crash_burst_items: int = 5  # distinct crashed items inside the window form a burst; 0 = off
    crash_burst_window_s: float = 600.0
    burst_backoff_s: float = 60.0  # first global backoff; doubles while bursts repeat
    burst_backoff_cap_s: float = 1800.0
    # provider availability
    provider_probe_interval_s: float = 120.0
    max_provider_pause_s: float = 6 * 3600.0
    # liveness
    max_consecutive_unknown: int = 20  # polls of one attempt with unknown liveness before pausing
    lost_confirmations: int = 2  # consecutive dead lease probes (without life evidence) = lost
    # streaming verification
    systematic_min_items: int = 8  # distinct items needed before the check runs; 0 = off
    systematic_window: int = 20  # most recent distinct items considered
    systematic_share: float = 0.6  # one gate failing this share of the window stops launches
    max_verify_errors: int = 3  # verifier exceptions before the attempt counts as abstain
    # loop mechanics
    poll_interval_s: float = 30.0
    max_tick_gap_s: float = 600.0  # longer gaps (supervisor down) are not charged to budgets
    drain_on_stop: bool = True  # wait for in-flight attempts before a stopped_* status
    seed_base: int = 0
    seed_modulus: int = 2_147_483_647

    def __post_init__(self) -> None:
        positive_ints = ("max_attempts_per_item", "max_concurrency", "max_rounds",
                         "max_consecutive_unknown", "lost_confirmations", "systematic_window",
                         "max_verify_errors")
        non_negative_ints = ("max_crash_retries_per_item", "zero_progress_rounds",
                             "crash_burst_items", "systematic_min_items", "seed_base")
        non_negative_floats = ("launch_cost_estimate", "crash_retry_delay_s",
                               "crash_retry_delay_cap_s", "crash_burst_window_s", "burst_backoff_s",
                               "burst_backoff_cap_s", "provider_probe_interval_s",
                               "max_provider_pause_s", "poll_interval_s", "max_tick_gap_s")
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"policy.{name} must be an integer >= 1, got {value!r}")
        for name in non_negative_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"policy.{name} must be an integer >= 0, got {value!r}")
        for name in non_negative_floats:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0):
                raise ValueError(f"policy.{name} must be a number >= 0, got {value!r}")
        for name in ("max_total_cost", "max_wall_seconds"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value <= 0):
                raise ValueError(f"policy.{name} must be null or a number > 0, got {value!r}")
        if not 0 < float(self.systematic_share) <= 1:
            raise ValueError("policy.systematic_share must be in (0, 1]")
        if self.systematic_min_items > self.systematic_window:
            raise ValueError("policy.systematic_min_items cannot exceed systematic_window")
        if isinstance(self.seed_modulus, bool) or not isinstance(self.seed_modulus, int) \
                or self.seed_modulus < 2:
            raise ValueError("policy.seed_modulus must be an integer >= 2")
        if self.poll_interval_s <= 0:
            raise ValueError("policy.poll_interval_s must be > 0")
        if not isinstance(self.drain_on_stop, bool):
            raise ValueError("policy.drain_on_stop must be a boolean")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "BatchPolicy":
        data = dict(data or {})
        schema = data.pop("schema", POLICY_SCHEMA)
        if schema != POLICY_SCHEMA:
            raise ValueError(f"unsupported policy schema {schema!r}; expected {POLICY_SCHEMA}")
        data.pop("description", None)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"unknown policy keys {unknown}; allowed: {sorted(names)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": POLICY_SCHEMA, **dataclasses.asdict(self)}

    def sha256(self) -> str:
        return common.sha256_json(self.to_dict(), allow_nan=True)

    def replace(self, **changes: Any) -> "BatchPolicy":
        return dataclasses.replace(self, **changes)


# -- callable results --------------------------------------------------------------------------
@dataclass(frozen=True)
class Poll:
    """What ``poll(handle)`` reports. ``running`` requires positive evidence of life."""

    status: str
    result: Any = None
    detail: str | None = None
    kind: str = "transient"  # for crashed: transient | provider_down | permanent
    cost: float | None = None

    def __post_init__(self) -> None:
        if self.status not in POLL_STATUSES:
            raise ValueError(f"poll status must be one of {POLL_STATUSES}, got {self.status!r}")
        if self.kind not in CRASH_KINDS:
            raise ValueError(f"crash kind must be one of {CRASH_KINDS}, got {self.kind!r}")

    @classmethod
    def coerce(cls, value: Any) -> "Poll":
        if isinstance(value, Poll):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            allowed = {f.name for f in dataclasses.fields(cls)}
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"unknown poll keys {sorted(unknown)}")
            return cls(**dict(value))
        if isinstance(value, tuple) and value:
            return cls(*value)
        raise TypeError(f"cannot interpret poll result {value!r}")


@dataclass(frozen=True)
class Verdict:
    """What ``verify(item, result)`` reports. ``abstain`` is a first-class non-admission."""

    decision: str
    reason: str | None = None
    gate: str | None = None
    final: bool = False  # a final reject ends the item (rejected_final)
    evidence: Any = None

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"verdict must be one of {DECISIONS}, got {self.decision!r}")

    @classmethod
    def passed(cls, evidence: Any = None) -> "Verdict":
        return cls("pass", evidence=evidence)

    @classmethod
    def rejected(cls, reason: str, gate: str | None = None, *, final: bool = False,
                 evidence: Any = None) -> "Verdict":
        return cls("reject", reason=reason, gate=gate, final=final, evidence=evidence)

    @classmethod
    def abstained(cls, reason: str, gate: str | None = None) -> "Verdict":
        return cls("abstain", reason=reason, gate=gate)

    @classmethod
    def coerce(cls, value: Any) -> "Verdict":
        if isinstance(value, Verdict):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            allowed = {f.name for f in dataclasses.fields(cls)}
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"unknown verdict keys {sorted(unknown)}")
            return cls(**dict(value))
        if isinstance(value, tuple) and value:
            return cls(*value)
        raise TypeError(f"cannot interpret verdict {value!r}")


def coerce_lease(value: Any) -> str:
    if value is True:
        return "alive"
    if value is False:
        return "dead"
    if value is None:
        return "unknown"
    if isinstance(value, str) and value in LEASE_STATES:
        return value
    raise ValueError(f"lease probe must return one of {LEASE_STATES}, got {value!r}")


@dataclass
class BatchTask:
    """Task-specific behaviour, injected as callables.

    ``launch(item, attempt, seed)`` starts one attempt and returns a JSON-serializable handle. When
    it also accepts a ``launch_id`` keyword it receives a deterministic id for this launch
    (``<item>.aNN.tNN``); crash retries of one attempt get new ``tNN`` suffixes.

    ``poll(handle)`` returns ``running | finished | crashed | unknown`` (or a :class:`Poll`).

    ``verify(item, result)`` returns ``pass | reject | abstain`` (or a :class:`Verdict`).

    Optional: ``lease_probe(handle) -> alive|dead|unknown``, ``cost(result) -> float``,
    ``reconcile(item, attempt, seed, launch_id) -> handle | None | "unknown"`` (look up a launch
    whose intent was persisted but whose handle was not), ``provider_probe() -> up|down|unknown``
    and ``seed_for(item, attempt) -> int``.
    """

    launch: Callable[..., Any]
    poll: Callable[[Any], Any]
    verify: Callable[[str, Any], Any]
    lease_probe: Callable[[Any], Any] | None = None
    cost: Callable[[Any], Any] | None = None
    reconcile: Callable[[str, int, int, str], Any] | None = None
    provider_probe: Callable[[], Any] | None = None
    seed_for: Callable[[str, int], int] | None = None


# -- optional ledger mirroring -----------------------------------------------------------------
class LedgerHook:
    """Mirror item states into a pipeline run ledger through its public API.

    Reporting never stops the batch: ledger errors are returned to the caller, which records
    them as events.
    """

    def __init__(self, ledger: Any, worker: str, column: str,
                 item_map: Callable[[str], str] | None = None) -> None:
        self.ledger = ledger
        self.worker = worker
        self.column = column
        self.item_map = item_map or (lambda item: item)

    @classmethod
    def from_env(cls, column: str | None,
                 item_map: Callable[[str], str] | None = None) -> "LedgerHook | None":
        from datafarm.ledger import ledger as ledger_module

        attached = ledger_module.attach_from_env()
        if attached is None or not column:
            return None
        ledger, worker = attached
        return cls(ledger, worker, column, item_map)

    def update_cells(self, rows: Sequence[Mapping[str, Any]]) -> int:
        from datafarm.ledger.ledger import CellUpdate

        updates = [
            CellUpdate(self.item_map(row["item"]), self.column, row["status"],
                       attempt=row.get("attempt"), failure=row.get("failure"),
                       artifact=row.get("artifact"), note=row.get("note"))
            for row in rows
        ]
        return self.ledger.set_cells(updates, source=f"batch-supervisor:{self.worker}")

    def declare_blocked(self, reason: str) -> None:
        self.ledger.mirror_declaration("blocked", reason=reason,
                                       source=f"batch-supervisor:{self.worker}")


# -- the controller ----------------------------------------------------------------------------
class BatchSupervisor:
    """Drive a fixed set of items to admission under an explicit :class:`BatchPolicy`."""

    def __init__(
        self,
        state_dir: str | Path,
        items: Sequence[str],
        task: BatchTask,
        policy: BatchPolicy | None = None,
        *,
        batch_id: str | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        ledger: LedgerHook | str | None = None,
        ledger_column: str | None = None,
        lock: bool = True,
        fsync: bool = True,
        heartbeat_s: float = 60.0,
        stop_check: Callable[[], str | None] | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.policy = policy or BatchPolicy()
        self._clock = clock or time.time
        self._sleep = sleep or time.sleep
        self._fsync = fsync
        self._heartbeat_s = float(heartbeat_s)
        self._stop_check = stop_check
        self._state_path = self.state_dir / "state.json"
        self._events_path = self.state_dir / "events.jsonl"
        self._report_path = self.state_dir / "report.json"
        self._lock_handle = self._acquire_lock() if lock else None
        self._launch_takes_id = _accepts_keyword(task.launch, "launch_id")
        self._dirty = True
        self._last_save = -float("inf")
        self._ledger_dirty: set[str] = set()
        self._ledger_errors: set[str] = set()
        items = [self._check_item(item) for item in items]
        if not items:
            raise BatchError("a batch needs a fixed, non-empty item denominator")
        if len(set(items)) != len(items):
            raise BatchError("duplicate item ids in the denominator")
        self._items = items
        self._order = {item: index for index, item in enumerate(items)}
        try:
            if self._state_path.is_file():
                self._state = self._load_existing(items, batch_id)
            else:
                self._state = self._fresh_state(items, batch_id)
                self._event("created", items=len(items), policy_sha256=self.policy.sha256())
            self._ledger = self._resolve_ledger(ledger, ledger_column)
            self._ledger_dirty = set(items)
            self._save(force=True)
        except BaseException:
            self.close()
            raise

    # -- construction -----------------------------------------------------------------------
    @staticmethod
    def _check_item(item: Any) -> str:
        if not isinstance(item, str) or not item.strip():
            raise BatchError(f"item ids must be non-empty strings, got {item!r}")
        return item

    def _acquire_lock(self) -> Any:
        handle = open(self.state_dir / "supervisor.lock", "a+", encoding="utf-8")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise SupervisorBusy(f"another supervisor owns {self.state_dir}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        return handle

    def close(self) -> None:
        handle, self._lock_handle = getattr(self, "_lock_handle", None), None
        if handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def __enter__(self) -> "BatchSupervisor":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _resolve_ledger(self, ledger: LedgerHook | str | None,
                        column: str | None) -> LedgerHook | None:
        if ledger is None:
            return None
        if isinstance(ledger, LedgerHook):
            return ledger
        if ledger != "env":
            raise BatchError("ledger must be None, 'env' or a LedgerHook")
        try:
            hook = LedgerHook.from_env(column)
        except Exception as exc:  # noqa: BLE001 - a broken ledger must not stop the batch
            self._event("ledger_error", error=_short(f"{type(exc).__name__}: {exc}"))
            return None
        if hook is None and os.environ.get("DATAFARM_LEDGER_RUN_DIR"):
            self._event("ledger_not_mirrored",
                        reason="running under a ledger run but no ledger column was configured")
        return hook

    def _new_item(self, item: str) -> dict[str, Any]:
        return {
            "state": PENDING, "attempt": 1, "seed": self._seed_for(item, 1), "attempts_used": 0,
            "crash_retries": 0, "provider_crashes": 0, "tries": 0, "max_launched_attempt": 0,
            "launch_id": None, "handle": None, "intent_at": None, "launched_at": None,
            "retry_not_before": None, "unknown_streak": 0, "dead_streak": 0, "verify_errors": 0,
            "cost_recorded": False, "attempt_limit": None, "attempts": [], "crash_log": [],
            "terminal_reason": None, "admitted_attempt": None, "evidence": None, "last": None,
        }

    def _fresh_state(self, items: list[str], batch_id: str | None) -> dict[str, Any]:
        now = self._now()
        return {
            "schema": STATE_SCHEMA,
            "batch_id": batch_id,
            "items": list(items),
            "created_at": iso(now),
            "updated_at": iso(now),
            "status": RUNNING_STATUS,
            "stop_request": None,
            "terminal": None,
            "policy": self.policy.to_dict(),
            "policy_sha256": self.policy.sha256(),
            "counters": {"launch_attempts": 0, "launches": 0, "finished": 0, "crashes": 0,
                         "provider_crashes": 0, "verified": 0, "cost": 0.0, "active_seconds": 0.0,
                         "provider_paused_seconds": 0.0, "ticks": 0, "resumes": 0},
            "rounds": {"closed": 0, "admitted_by_round": {}, "zero_progress_streak": 0,
                       "history": []},
            "burst": new_burst_state(),
            "provider": {"down": False, "since": None, "next_probe": None, "half_open": False,
                         "probe_item": None, "outages": 0, "pause_baseline": 0.0},
            "recent_verdicts": [],
            "last_tick_at": None,
            "item_state": {item: self._new_item(item) for item in items},
        }

    def _load_existing(self, items: list[str], batch_id: str | None) -> dict[str, Any]:
        state = json.loads(self._state_path.read_text(encoding="utf-8"))
        if state.get("schema") != STATE_SCHEMA:
            raise BatchError(f"unsupported state schema {state.get('schema')!r}")
        if list(state["items"]) != list(items):
            stored, given = set(state["items"]), set(items)
            raise DenominatorMismatch(
                "the item denominator is fixed for the life of a batch; "
                f"missing={sorted(stored - given)[:10]} extra={sorted(given - stored)[:10]} "
                f"(order changes are rejected too)")
        if batch_id is not None and state.get("batch_id") not in (None, batch_id):
            raise BatchError(f"state belongs to batch {state.get('batch_id')!r}, not {batch_id!r}")
        if state.get("policy_sha256") != self.policy.sha256():
            self._event("policy_changed", old=state.get("policy_sha256"),
                        new=self.policy.sha256(), old_policy=state.get("policy"),
                        new_policy=self.policy.to_dict())
            state["policy"] = self.policy.to_dict()
            state["policy_sha256"] = self.policy.sha256()
        state["counters"]["resumes"] += 1
        state["last_tick_at"] = None  # supervisor downtime is not charged to any budget
        in_flight = {item: rec["state"] for item, rec in state["item_state"].items()
                     if rec["state"] in IN_FLIGHT}
        self._event("resumed", status=state["status"], in_flight=in_flight)
        return state

    # -- persistence ------------------------------------------------------------------------
    def _now(self) -> float:
        return float(self._clock())

    def _event(self, event: str, /, **data: Any) -> None:
        now = self._now()
        record = {key: value for key, value in data.items() if value is not None}
        record.update({"at": iso(now), "t": round(now, 3), "kind": event})
        common.append_jsonl(self._events_path, record, lock=False, fsync=self._fsync, allow_nan=True, default=str)

    def _save(self, *, force: bool = False) -> None:
        now = self._now()
        if not force and not self._dirty and now - self._last_save < self._heartbeat_s:
            return
        self._state["updated_at"] = iso(now)
        write_json_atomic(self._state_path, self._state, fsync=self._fsync)
        self._dirty = False
        self._last_save = now

    def _touch(self, item: str | None = None) -> None:
        self._dirty = True
        if item is not None:
            self._ledger_dirty.add(item)

    # -- public accessors -------------------------------------------------------------------
    @property
    def status(self) -> str:
        return str(self._state["status"])

    @property
    def items(self) -> list[str]:
        return list(self._items)

    def item(self, item: str) -> dict[str, Any]:
        return json.loads(json.dumps(self._state["item_state"][item]))

    def state(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._state))

    def summary(self) -> dict[str, Any]:
        return summarize_state(self._state)

    @classmethod
    def read_summary(cls, state_dir: str | Path) -> dict[str, Any]:
        """Summarize a batch without taking its lock (read-only status)."""

        state = json.loads((Path(state_dir) / "state.json").read_text(encoding="utf-8"))
        return summarize_state(state)

    def request_stop(self, reason: str, status: str = STOPPED_OPERATOR,
                     report: Mapping[str, Any] | None = None) -> None:
        if status not in TERMINAL_STATUSES - {COMPLETE, COMPLETE_WITH_GAPS}:
            raise BatchError(f"cannot request stop status {status!r}")
        existing = self._state["stop_request"]
        if self._state["status"] in TERMINAL_STATUSES:
            return
        escalates = status in PAUSED_STATUSES and existing is not None and \
            existing["status"] not in PAUSED_STATUSES
        if existing and not escalates:
            return  # first stop wins, except that a pause (nothing drains) overrides a drain
        self._state["stop_request"] = {"status": status, "reason": reason, "at": iso(self._now()),
                                       "report": dict(report or {})}
        self._event("stop_requested", status=status, reason=reason, report=report)
        self._touch()

    def acknowledge(self, reason: str) -> str:
        """Clear a stopped/paused status after the operator has looked at it; returns the old one.

        Streaks (zero progress, unknown liveness) and the systematic-failure window restart from
        zero, and a paused provider gets a fresh pause allowance. Budgets are not reset; use
        :meth:`reopen` for that.
        """

        state = self._state
        previous = state["status"]
        if previous == COMPLETE:
            raise BatchError("the batch is complete; nothing to acknowledge")
        if previous not in TERMINAL_STATUSES and not state["stop_request"]:
            return previous
        state["status"] = RUNNING_STATUS
        state["stop_request"] = None
        state["terminal"] = None
        state["rounds"]["zero_progress_streak"] = 0
        state["recent_verdicts"] = []
        state["provider"]["pause_baseline"] = state["counters"]["provider_paused_seconds"]
        for rec in state["item_state"].values():
            rec["unknown_streak"] = 0
        self._event("acknowledged", previous=previous, reason=reason)
        self._touch()
        self._save(force=True)
        return previous

    def reopen(self, items: Iterable[str], *, reason: str, extra_attempts: int = 1) -> list[str]:
        """Give exhausted / finally rejected items another chance (explicit operator decision).

        An item exhausted by its *crash* budget gets a fresh crash budget and relaunches the same
        attempt and seed. Any other item moves to its next attempt (new seed) with
        ``extra_attempts`` more seeds; ``max_rounds`` still applies.
        """

        if extra_attempts < 1:
            raise BatchError("extra_attempts must be >= 1")
        items = list(items)
        for item in items:
            rec = self._rec(item)
            if rec["state"] not in (EXHAUSTED, REJECTED_FINAL):
                raise BatchError(f"{item} is {rec['state']}; only exhausted or rejected_final "
                                 "items can be reopened")
            if rec["terminal_reason"] != "crash_budget" and \
                    rec["attempt"] + 1 > self.policy.max_rounds:
                raise BatchError(f"{item} would exceed max_rounds={self.policy.max_rounds}; "
                                 "raise it in the policy first")
        now = self._now()
        reopened = []
        for item in items:
            rec = self._rec(item)
            rec["crash_retries"] = 0
            if rec["terminal_reason"] == "crash_budget":
                rec["state"] = CRASHED_RETRYABLE  # same attempt and seed
            else:
                rec["attempt"] += 1
                rec["seed"] = self._seed_for(item, rec["attempt"])
                rec["tries"] = 0
                rec["state"] = REJECTED_RETRYABLE
                rec["attempt_limit"] = rec["attempts_used"] + extra_attempts
            rec["retry_not_before"] = now
            rec["terminal_reason"] = None
            rec["last"] = {"kind": "reopened", "reason": reason}
            reopened.append(item)
            self._touch(item)
        if reopened:
            if self._state["status"] in TERMINAL_STATUSES:
                self._state["status"] = RUNNING_STATUS
                self._state["terminal"] = None
                self._state["stop_request"] = None
            self._event("reopened", items=reopened, reason=reason, extra_attempts=extra_attempts)
            self._save(force=True)
        return reopened

    def revoke_admission(self, item: str, *, reason: str, gate: str = "cohort_audit") -> None:
        """A later (e.g. cohort-level) verifier withdrew an admission: treat it as a rejection."""

        rec = self._rec(item)
        if rec["state"] != ADMITTED:
            raise BatchError(f"{item} is {rec['state']}, not admitted")
        now = self._now()
        if rec["attempts"]:
            rec["attempts"][-1]["revoked"] = {"reason": reason, "gate": gate}
        rec["evidence"] = None
        rec["admitted_attempt"] = None
        rec["terminal_reason"] = None
        self._event("admission_revoked", item=item, attempt=rec["attempt"], reason=reason,
                    gate=gate)
        self._after_consumed(item, rec, now, gate=gate, reason=reason, final=False)
        self._note_verdict(item, gate)
        if self._state["status"] in (COMPLETE, COMPLETE_WITH_GAPS):
            self._state["status"] = RUNNING_STATUS
            self._state["terminal"] = None
        self._touch(item)
        self._save(force=True)

    def resolve_unknown(self, item: str, *, outcome: str, reason: str) -> None:
        """Operator verdict on an attempt whose liveness could not be established.

        ``not_launched``: nothing is running; relaunch the same attempt and seed.
        ``crashed``: count it as a crash (crash budget, same seed).
        """

        rec = self._rec(item)
        if rec["state"] not in IN_FLIGHT:
            raise BatchError(f"{item} is {rec['state']}, not in flight")
        if outcome not in ("not_launched", "crashed"):
            raise BatchError("outcome must be 'not_launched' or 'crashed'")
        now = self._now()
        self._event("resolved_unknown", item=item, outcome=outcome, reason=reason,
                    launch_id=rec["launch_id"])
        if outcome == "not_launched":
            previous = rec.get("pre_launch_state") or self._idle_state(rec)
            self._clear_attempt(rec)
            rec["state"] = previous
            rec["retry_not_before"] = now
        else:
            self._crash(item, rec, now, Poll("crashed", detail=f"operator: {reason}"))
        self._touch(item)
        self._save(force=True)

    @staticmethod
    def _idle_state(rec: Mapping[str, Any]) -> str:
        """The eligible state an item returns to when its launch did not happen."""

        if rec["attempt"] > 1:
            return REJECTED_RETRYABLE
        return CRASHED_RETRYABLE if rec["crash_retries"] else PENDING

    # -- the loop ---------------------------------------------------------------------------
    def run(self, *, max_ticks: int | None = None) -> str:
        """Tick until a terminal status (or ``max_ticks``); returns the status."""

        ticks = 0
        while True:
            status = self.step()
            if status in TERMINAL_STATUSES:
                return status
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                return self._state["status"]
            self._sleep(self._next_delay())

    def step(self) -> str | None:
        """One control tick. Returns the terminal status once reached, otherwise ``None``."""

        state = self._state
        if state["status"] in TERMINAL_STATUSES:
            return state["status"]
        now = self._now()
        self._advance_clock(now)
        state["counters"]["ticks"] += 1
        if self._stop_check is not None:
            reason = self._stop_check()
            if reason:
                self.request_stop(f"operator: {reason}", STOPPED_OPERATOR)
        self._check_provider(now)
        for item in self._items:
            rec = state["item_state"][item]
            if rec["state"] == LAUNCHING:
                self._resolve_intent(item, rec, now)
            elif rec["state"] in (RUNNING, UNKNOWN):
                self._poll_item(item, rec, now)
        self._check_unknown()
        self._close_rounds(now)
        self._check_guards()
        status = self._maybe_terminal(now)
        if status is None:
            self._dispatch(now)
            self._save()
        self._flush_ledger()
        return status

    def _next_delay(self) -> float:
        """Sleep until the next poll, or earlier when a backoff/probe/retry time comes due.

        Only *future* deadlines count: an item that is due but blocked (capacity, backoff,
        provider down) must not turn the loop into a busy poll of every in-flight attempt.
        """

        now = self._now()
        state = self._state
        deadlines = [state["burst"]["until"]]
        if state["provider"]["down"]:
            deadlines.append(state["provider"]["next_probe"])
        deadlines += [rec["retry_not_before"] for rec in state["item_state"].values()
                      if rec["state"] in ELIGIBLE]
        candidates = [self.policy.poll_interval_s]
        candidates += [deadline - now for deadline in deadlines
                       if deadline is not None and deadline > now]
        return max(min(1.0, self.policy.poll_interval_s), min(candidates))

    # -- time accounting --------------------------------------------------------------------
    def _advance_clock(self, now: float) -> None:
        state = self._state
        last = state["last_tick_at"]
        if last is not None and now > last:
            delta = min(now - last, self.policy.max_tick_gap_s)
            key = "provider_paused_seconds" if state["provider"]["down"] else "active_seconds"
            state["counters"][key] += delta
        state["last_tick_at"] = now

    # -- per-item handling ------------------------------------------------------------------
    def _rec(self, item: str) -> dict[str, Any]:
        try:
            return self._state["item_state"][item]
        except KeyError as exc:
            raise BatchError(f"{item!r} is outside the batch denominator") from exc

    def _seed_for(self, item: str, attempt: int) -> int:
        if self.task.seed_for is not None:
            seed = self.task.seed_for(item, attempt)
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise BatchError(f"seed_for must return a non-negative int, got {seed!r}")
            return seed
        return default_seed(self.policy.seed_base, item, attempt, self.policy.seed_modulus)

    @staticmethod
    def _clear_attempt(rec: dict[str, Any]) -> None:
        rec.update(handle=None, launched_at=None, intent_at=None, unknown_streak=0,
                   dead_streak=0, verify_errors=0, cost_recorded=False)
        rec.pop("pre_launch_state", None)

    def _set_terminal(self, item: str, rec: dict[str, Any], state: str, reason: str) -> None:
        rec["state"] = state
        rec["terminal_reason"] = reason
        rec["retry_not_before"] = None
        self._event("item_terminal", item=item, state=state, reason=reason,
                    attempts_used=rec["attempts_used"], crash_retries=rec["crash_retries"])

    def _resolve_intent(self, item: str, rec: dict[str, Any], now: float,
                        launch_error: str | None = None) -> None:
        """A launch intent exists without a handle (supervisor died mid-launch, or the launch
        raised). ``reconcile`` decides; without it the attempt stays unknown, never relaunched."""

        if self.task.reconcile is None:
            self._mark_unknown(item, rec, launch_error or "launch intent persisted but no handle; "
                                                          "no reconcile callable to look it up")
            return
        try:
            found = self.task.reconcile(item, rec["attempt"], rec["seed"], rec["launch_id"])
        except Exception as exc:  # noqa: BLE001 - reconcile failure means unknown, never lost
            found = "unknown"
            self._event("reconcile_error", item=item, error=_short(f"{type(exc).__name__}: {exc}"))
        if found is None or found == "not_launched":
            self._event("reconcile_not_launched", item=item, launch_id=rec["launch_id"])
            if launch_error is not None:
                # The launch itself failed: that is a crash (budgeted), or a raising launcher
                # would be retried on every tick forever.
                self._crash(item, rec, now, Poll("crashed", detail=launch_error), launched=False)
                return
            previous = rec.get("pre_launch_state") or self._idle_state(rec)
            self._clear_attempt(rec)
            rec["state"] = previous
            rec["retry_not_before"] = now
        elif found == "unknown":
            self._mark_unknown(item, rec, "reconcile could not establish the launch outcome")
        else:
            rec["handle"] = _jsonable(found)
            rec["state"] = RUNNING
            rec["launched_at"] = now
            rec["unknown_streak"] = 0
            rec.pop("pre_launch_state", None)
            self._state["counters"]["launches"] += 1
            self._event("reconcile_found", item=item, launch_id=rec["launch_id"],
                        handle=rec["handle"])
        self._touch(item)

    def _mark_unknown(self, item: str, rec: dict[str, Any], detail: str) -> None:
        rec["unknown_streak"] += 1
        if rec["state"] == RUNNING:
            rec["state"] = UNKNOWN
        if rec["unknown_streak"] == 1:
            self._event("unknown", item=item, attempt=rec["attempt"], launch_id=rec["launch_id"],
                        detail=_short(detail))
        rec["last"] = {"kind": "unknown", "detail": _short(detail)}
        self._touch(item)

    def _poll_item(self, item: str, rec: dict[str, Any], now: float) -> None:
        handle = rec["handle"]
        try:
            poll = Poll.coerce(self.task.poll(handle))
        except Exception as exc:  # noqa: BLE001 - a failed status query is "unknown", not "lost"
            poll = Poll("unknown", detail=f"poll raised {type(exc).__name__}: {exc}")
        if poll.status == "finished":
            self._finished(item, rec, now, poll)
            return
        if poll.status == "crashed":
            self._crash(item, rec, now, poll)
            return
        if poll.status == "running":
            # Positive evidence of life; a lease is not even consulted (a lease can vanish while
            # the process still runs on an unreachable host).
            self._alive(item, rec)
            return
        lease = "unknown"
        if self.task.lease_probe is not None:
            try:
                lease = coerce_lease(self.task.lease_probe(handle))
            except Exception as exc:  # noqa: BLE001 - lease query failure is unknown, not dead
                lease = "unknown"
                if rec["unknown_streak"] == 0:
                    self._event("lease_probe_error", item=item,
                                error=_short(f"{type(exc).__name__}: {exc}"))
        if lease == "alive":
            self._alive(item, rec)
            return
        if lease == "dead":
            rec["dead_streak"] += 1
            if rec["dead_streak"] >= self.policy.lost_confirmations:
                self._event("lost", item=item, launch_id=rec["launch_id"],
                            confirmations=rec["dead_streak"], detail=_short(poll.detail))
                self._crash(item, rec, now, Poll(
                    "crashed", detail=f"lost: poll {poll.status} and lease dead on "
                                      f"{rec['dead_streak']} consecutive probes; "
                                      f"{poll.detail or ''}".strip()))
                return
        else:
            rec["dead_streak"] = 0
        self._mark_unknown(item, rec, poll.detail or f"poll unknown, lease {lease}")

    def _alive(self, item: str, rec: dict[str, Any]) -> None:
        if rec["state"] == UNKNOWN:
            self._event("liveness_restored", item=item, after_unknown_polls=rec["unknown_streak"])
            self._touch(item)
        rec["state"] = RUNNING
        rec["unknown_streak"] = 0
        rec["dead_streak"] = 0

    def _finished(self, item: str, rec: dict[str, Any], now: float, poll: Poll) -> None:
        state = self._state
        result = poll.result if poll.result is not None else rec["handle"]
        if not rec["cost_recorded"]:
            cost = poll.cost
            cost_error = None
            if cost is None and self.task.cost is not None:
                try:
                    cost = self.task.cost(result)
                except Exception as exc:  # noqa: BLE001
                    cost_error = _short(f"{type(exc).__name__}: {exc}")
            cost_required = self.task.cost is not None or self.policy.max_total_cost is not None
            if cost is None and cost_required:
                cost_error = cost_error or "finished attempt has no cost evidence"
            elif cost is not None and (isinstance(cost, bool) or
                                       not isinstance(cost, (int, float)) or
                                       not math.isfinite(cost) or cost < 0):
                cost_error = f"invalid cost {cost!r}; expected a finite non-negative number"
            if cost_error is not None:
                self._event("cost_error", item=item, error=_short(cost_error))
                rec["last"] = {"kind": "cost_unknown", "detail": _short(cost_error)}
                self.request_stop(
                    f"cost for finished attempt {item} is unknown: {cost_error}",
                    PAUSED_COST_UNKNOWN,
                    {"item": item, "launch_id": rec["launch_id"], "error": _short(cost_error)})
                self._touch(item)
                return  # retain the result and attempt; resume re-reads cost before verification
            amount = float(cost) if cost is not None else 0.0
            state["counters"]["cost"] += amount
            rec["attempt_cost"] = amount
            rec["cost_recorded"] = True
            state["counters"]["finished"] += 1
            self._provider_evidence_up(item, now)
        try:
            verdict = Verdict.coerce(self.task.verify(item, result))
        except Exception as exc:  # noqa: BLE001 - verifier failure is not the render's failure
            rec["verify_errors"] += 1
            self._event("verify_error", item=item, attempt=rec["attempt"],
                        errors=rec["verify_errors"], error=_short(f"{type(exc).__name__}: {exc}"))
            self._touch(item)
            if rec["verify_errors"] < self.policy.max_verify_errors:
                return  # keep the attempt; verification is retried on the next tick
            verdict = Verdict.abstained(f"verifier failed {rec['verify_errors']} times: "
                                        f"{type(exc).__name__}: {exc}", gate="verifier_error")
        self._apply_verdict(item, rec, now, verdict)

    def _apply_verdict(self, item: str, rec: dict[str, Any], now: float, verdict: Verdict) -> None:
        state = self._state
        state["counters"]["verified"] += 1
        entry = {"attempt": rec["attempt"], "seed": rec["seed"], "tries": rec["tries"],
                 "launch_id": rec["launch_id"], "decision": verdict.decision, "gate": verdict.gate,
                 "reason": _short(verdict.reason), "at": iso(now),
                 "cost": rec.get("attempt_cost", 0.0)}
        rec["attempts"].append(entry)
        rec["last"] = {"kind": "verdict", "decision": verdict.decision, "gate": verdict.gate,
                       "reason": _short(verdict.reason)}
        self._event("verdict", item=item, attempt=rec["attempt"], seed=rec["seed"],
                    decision=verdict.decision, gate=verdict.gate, reason=_short(verdict.reason),
                    final=verdict.final or None)
        self._clear_attempt(rec)
        if verdict.decision == "pass":
            rec["attempts_used"] += 1
            rec["state"] = ADMITTED
            rec["admitted_attempt"] = rec["attempt"]
            rec["terminal_reason"] = "admitted"
            rec["retry_not_before"] = None
            try:
                rec["evidence"] = _jsonable(verdict.evidence)
            except TypeError:
                rec["evidence"] = _short(verdict.evidence)
            rounds = state["rounds"]["admitted_by_round"]
            key = str(rec["attempt"])
            rounds[key] = rounds.get(key, 0) + 1
            self._note_verdict(item, None)
        else:
            gate = verdict.gate or ("abstain" if verdict.decision == "abstain" else "unspecified")
            self._note_verdict(item, gate)
            rec["attempts_used"] += 1
            self._after_consumed(item, rec, now, gate=gate, reason=verdict.reason,
                                 final=verdict.final)
        self._touch(item)

    def _after_consumed(self, item: str, rec: dict[str, Any], now: float, *, gate: str | None,
                        reason: str | None, final: bool) -> None:
        """An attempt was used up without admission: decide between retry and a terminal state.

        The caller has already counted the attempt in ``attempts_used``.
        """

        limit = rec["attempt_limit"] or self.policy.max_attempts_per_item
        if final:
            self._set_terminal(item, rec, REJECTED_FINAL, f"final reject by {gate}: {reason}")
        elif rec["attempts_used"] >= limit:
            self._set_terminal(item, rec, EXHAUSTED, "attempt_budget")
        elif rec["attempt"] + 1 > self.policy.max_rounds:
            self._set_terminal(item, rec, EXHAUSTED, "max_rounds")
        else:
            rec["attempt"] += 1
            rec["seed"] = self._seed_for(item, rec["attempt"])
            rec["tries"] = 0
            rec["state"] = REJECTED_RETRYABLE
            rec["retry_not_before"] = now
        self._touch(item)

    def _crash(self, item: str, rec: dict[str, Any], now: float, poll: Poll,
               *, launched: bool = True) -> None:
        state = self._state
        detail = poll.detail or "crashed without a result"
        if poll.kind == "capacity":
            state["counters"]["capacity_waits"] = state["counters"].get("capacity_waits", 0) + 1
        else:
            state["counters"]["crashes"] += 1
        if poll.cost:
            state["counters"]["cost"] += float(poll.cost)
        rec["crash_log"] = (rec["crash_log"] + [{
            "attempt": rec["attempt"], "try": rec["tries"], "kind": poll.kind,
            "detail": _short(detail), "at": iso(now), "launched": launched,
        }])[-5:]
        rec["last"] = {"kind": "crash", "crash_kind": poll.kind, "detail": _short(detail)}
        launch_id = rec["launch_id"]
        self._clear_attempt(rec)
        probe_item = state["provider"]["probe_item"] == item
        if poll.kind == "capacity":
            rec["state"] = CRASHED_RETRYABLE
            rec["retry_not_before"] = now + self.policy.provider_probe_interval_s
            rec["last"] = {"kind": "capacity_wait", "detail": _short(detail)}
            self._event("capacity_wait", item=item, launch_id=launch_id,
                        detail=_short(detail), seed_kept=rec["seed"])
            if probe_item:
                self._provider_up(now, "allocator responds; capacity wait is not a provider outage")
        elif poll.kind == "permanent":
            self._event("crashed", item=item, attempt=rec["attempt"], launch_id=launch_id,
                        crash_kind=poll.kind, detail=_short(detail))
            self._note_verdict(item, "permanent_error")
            self._set_terminal(item, rec, REJECTED_FINAL, f"permanent error: {_short(detail)}")
            if probe_item:
                self._provider_up(now, "half-open probe reached the provider")
        elif poll.kind == "provider_down":
            state["counters"]["provider_crashes"] += 1
            rec["provider_crashes"] += 1
            rec["state"] = CRASHED_RETRYABLE
            rec["retry_not_before"] = now
            self._event("crashed", item=item, attempt=rec["attempt"], launch_id=launch_id,
                        crash_kind=poll.kind, detail=_short(detail))
            self._provider_down(now, detail)
        else:
            rec["crash_retries"] += 1
            self._event("crashed", item=item, attempt=rec["attempt"], launch_id=launch_id,
                        crash_kind=poll.kind, detail=_short(detail),
                        crash_retries=rec["crash_retries"], seed_kept=rec["seed"])
            if probe_item:
                self._provider_up(now, "half-open probe crashed for a non-provider reason")
            self._note_crash(item, now)
            if rec["crash_retries"] > self.policy.max_crash_retries_per_item:
                self._set_terminal(item, rec, EXHAUSTED, "crash_budget")
            else:
                delay = _backoff(self.policy.crash_retry_delay_s, rec["crash_retries"],
                                 self.policy.crash_retry_delay_cap_s)
                rec["state"] = CRASHED_RETRYABLE
                rec["retry_not_before"] = now + delay
        self._touch(item)

    # -- crash bursts -----------------------------------------------------------------------
    def _note_crash(self, item: str, now: float) -> None:
        trigger = advance_burst(self._state["burst"], now, item, self.policy)
        if trigger is not None:
            self._event("crash_burst_backoff", **trigger)

    # -- provider availability --------------------------------------------------------------
    def _provider_down(self, now: float, detail: str) -> None:
        provider = self._state["provider"]
        if not provider["down"]:
            provider.update(down=True, since=now, half_open=False, probe_item=None,
                            next_probe=now + self.policy.provider_probe_interval_s)
            provider["outages"] += 1
            self._event("provider_down", detail=_short(detail))
        else:
            provider.update(half_open=False, probe_item=None,
                            next_probe=now + self.policy.provider_probe_interval_s)
        self._touch()

    def _provider_up(self, now: float, why: str) -> None:
        provider = self._state["provider"]
        if not provider["down"]:
            return
        outage = now - provider["since"] if provider["since"] is not None else None
        provider.update(down=False, since=None, half_open=False, probe_item=None, next_probe=None)
        self._event("provider_up", why=why, outage_s=None if outage is None else round(outage, 1))
        self._touch()

    def _provider_evidence_up(self, item: str, now: float) -> None:
        if self._state["provider"]["down"]:
            self._provider_up(now, f"attempt {item} finished")

    def _check_provider(self, now: float) -> None:
        state = self._state
        provider = state["provider"]
        if not provider["down"]:
            return
        paused = state["counters"]["provider_paused_seconds"] - provider["pause_baseline"]
        if paused >= self.policy.max_provider_pause_s:
            self.request_stop(
                f"provider unavailable for {paused:.0f}s of pause (bound "
                f"{self.policy.max_provider_pause_s:.0f}s)", PAUSED_PROVIDER_DOWN,
                {"outages": provider["outages"], "paused_seconds": round(paused, 1)})
            return
        if provider["probe_item"] is not None or now < (provider["next_probe"] or 0):
            return
        if self.task.provider_probe is not None:
            try:
                answer = self.task.provider_probe()
            except Exception as exc:  # noqa: BLE001
                answer = "unknown"
                self._event("provider_probe_error", error=_short(f"{type(exc).__name__}: {exc}"))
            if answer == "up":
                self._provider_up(now, "provider probe")
            else:
                provider["next_probe"] = now + self.policy.provider_probe_interval_s
        else:
            provider["half_open"] = True  # let exactly one launch probe the provider
        self._touch()

    # -- verification window ----------------------------------------------------------------
    def _note_verdict(self, item: str, gate: str | None) -> None:
        window = [entry for entry in self._state["recent_verdicts"] if entry[0] != item]
        window.append([item, gate])
        self._state["recent_verdicts"] = window[-max(4 * self.policy.systematic_window, 50):]

    # -- guards -----------------------------------------------------------------------------
    def _check_unknown(self) -> None:
        stuck = {item: rec["unknown_streak"] for item, rec in self._state["item_state"].items()
                 if rec["state"] in IN_FLIGHT
                 and rec["unknown_streak"] >= self.policy.max_consecutive_unknown}
        if stuck:
            self.request_stop(
                f"liveness of {len(stuck)} attempt(s) unknown for >= "
                f"{self.policy.max_consecutive_unknown} consecutive polls; nothing was relaunched",
                PAUSED_UNKNOWN_LIVENESS,
                {"items": {item: {"streak": streak,
                                  "launch_id": self._state["item_state"][item]["launch_id"],
                                  "last": self._state["item_state"][item]["last"]}
                           for item, streak in sorted(stuck.items())}})

    def _close_rounds(self, now: float) -> None:
        state = self._state
        rounds = state["rounds"]
        recs = state["item_state"].values()
        open_attempts = [rec["attempt"] for rec in recs if rec["state"] not in TERMINAL_ITEM_STATES]
        if open_attempts:
            closable = min(open_attempts) - 1
        else:
            closable = max((rec["max_launched_attempt"] for rec in recs), default=0)
        while rounds["closed"] < closable:
            number = rounds["closed"] + 1
            admitted = int(rounds["admitted_by_round"].get(str(number), 0))
            rounds["closed"] = number
            rounds["zero_progress_streak"] = 0 if admitted else rounds["zero_progress_streak"] + 1
            rounds["history"].append({"round": number, "admitted": admitted, "closed_at": iso(now)})
            self._event("round_closed", round=number, admitted=admitted,
                        zero_progress_streak=rounds["zero_progress_streak"])
            self._touch()
            limit = self.policy.zero_progress_rounds
            work_left = any(rec["state"] not in TERMINAL_ITEM_STATES for rec in recs)
            if limit and rounds["zero_progress_streak"] >= limit and work_left:
                self.request_stop(
                    f"{rounds['zero_progress_streak']} consecutive closed rounds without a newly "
                    f"admitted item (rounds "
                    f"{number - rounds['zero_progress_streak'] + 1}-{number})",
                    STOPPED_ZERO_PROGRESS,
                    {"rounds": rounds["history"][-rounds["zero_progress_streak"]:],
                     "open_items": sorted(item for item, rec in state["item_state"].items()
                                          if rec["state"] not in TERMINAL_ITEM_STATES)})

    def _check_guards(self) -> None:
        state = self._state
        if state["stop_request"]:
            return
        policy = self.policy
        if policy.systematic_min_items:
            window = state["recent_verdicts"][-policy.systematic_window:]
            if len(window) >= policy.systematic_min_items:
                counts = Counter(gate for _, gate in window if gate is not None)
                if counts:
                    gate, failures = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    share = failures / len(window)
                    if share >= policy.systematic_share:
                        failing = [item for item, g in window if g == gate]
                        self.request_stop(
                            f"gate {gate!r} rejected {failures}/{len(window)} of the most recent "
                            f"distinct items (share {share:.2f} >= {policy.systematic_share})",
                            STOPPED_SYSTEMATIC_FAILURE,
                            {"gate": gate, "failures": failures, "window": len(window),
                             "share": round(share, 3), "threshold": policy.systematic_share,
                             "items": [{"item": item,
                                        "reason": (state["item_state"][item]["last"] or {})
                                        .get("reason") or (state["item_state"][item]["last"] or {})
                                        .get("detail")} for item in failing]})
                        return
        counters = state["counters"]
        if policy.max_total_cost is not None and counters["cost"] >= policy.max_total_cost:
            self.request_stop(f"cost {counters['cost']:.3f} reached the cap "
                              f"{policy.max_total_cost}", STOPPED_COST_CAP,
                              {"cost": counters["cost"]})
            return
        if policy.max_wall_seconds is not None and \
                counters["active_seconds"] >= policy.max_wall_seconds:
            self.request_stop(f"active wall time {counters['active_seconds']:.0f}s reached "
                              f"{policy.max_wall_seconds}s (provider pauses excluded)",
                              STOPPED_WALL_BUDGET,
                              {"active_seconds": round(counters["active_seconds"], 1),
                               "provider_paused_seconds":
                                   round(counters["provider_paused_seconds"], 1)})

    def _maybe_terminal(self, now: float) -> str | None:
        state = self._state
        in_flight = [item for item, rec in state["item_state"].items() if rec["state"] in IN_FLIGHT]
        stop = state["stop_request"]
        if stop:
            if stop["status"] in PAUSED_STATUSES or not self.policy.drain_on_stop or not in_flight:
                return self._finish(stop["status"], stop["reason"], stop.get("report"))
            if state["status"] != STOPPING:
                state["status"] = STOPPING
                self._event("draining", status=stop["status"], in_flight=sorted(in_flight))
                self._touch()
            return None
        eligible = any(rec["state"] in ELIGIBLE for rec in state["item_state"].values())
        if in_flight or eligible:
            return None
        if all(rec["state"] == ADMITTED for rec in state["item_state"].values()):
            return self._finish(COMPLETE, "every item admitted", None)
        gaps = {item: rec["terminal_reason"] for item, rec in state["item_state"].items()
                if rec["state"] != ADMITTED}
        return self._finish(COMPLETE_WITH_GAPS, f"{len(gaps)} item(s) not admitted: budgets "
                            "exhausted or final rejects", {"gaps": gaps})

    def _finish(self, status: str, reason: str | None,
                report: Mapping[str, Any] | None) -> str:
        state = self._state
        now = self._now()
        state["status"] = status
        state["stop_request"] = None
        state["terminal"] = {"status": status, "reason": reason, "at": iso(now),
                             "report": dict(report or {})}
        self._event("terminal", status=status, reason=reason)
        self._touch()
        self._save(force=True)
        write_json_atomic(self._report_path, self.summary(), fsync=self._fsync)
        self._ledger_dirty.update(self._items)
        self._flush_ledger()
        if self._ledger is not None and (status.startswith("stopped_") or
                                         status.startswith("paused_")):
            try:
                self._ledger.declare_blocked(f"{status}: {reason}")
            except Exception as exc:  # noqa: BLE001
                self._ledger_error(exc)
        return status

    # -- dispatch ---------------------------------------------------------------------------
    def _dispatch(self, now: float) -> None:
        state = self._state
        if state["stop_request"]:
            return
        burst_until = state["burst"]["until"]
        if burst_until is not None and now < burst_until:
            return
        provider = state["provider"]
        if provider["down"] and not provider["half_open"]:
            return
        recs = state["item_state"]
        in_flight = sum(rec["state"] in IN_FLIGHT for rec in recs.values())
        capacity = self.policy.max_concurrency - in_flight
        if capacity <= 0:
            return
        ready = sorted(
            (item for item, rec in recs.items() if rec["state"] in ELIGIBLE
             and (rec["retry_not_before"] is None or rec["retry_not_before"] <= now)),
            key=lambda item: (recs[item]["attempt"], self._order[item]))
        for item in ready:
            if in_flight >= self.policy.max_concurrency:
                break
            if not self._cost_allows(in_flight):
                break
            self._launch(item, recs[item], now, half_open=provider["down"])
            in_flight = sum(rec["state"] in IN_FLIGHT for rec in recs.values())
            if state["provider"]["down"] and recs[item]["state"] in IN_FLIGHT:
                break  # provider went down, or the single half-open probe is out
            if state["provider"]["down"] and not state["provider"]["half_open"]:
                break
            if state["burst"]["until"] is not None and now < state["burst"]["until"]:
                break

    def _cost_allows(self, in_flight: int) -> bool:
        policy = self.policy
        if policy.max_total_cost is None or policy.launch_cost_estimate <= 0:
            return True
        projected = self._state["counters"]["cost"] + (in_flight + 1) * policy.launch_cost_estimate
        if projected <= policy.max_total_cost:
            return True
        if in_flight == 0:
            self.request_stop(
                f"the next attempt (estimate {policy.launch_cost_estimate}) would exceed the cost "
                f"cap {policy.max_total_cost}", STOPPED_COST_CAP,
                {"cost": self._state["counters"]["cost"]})
        return False

    def _call_launch(self, item: str, rec: dict[str, Any]) -> Any:
        if self._launch_takes_id:
            return self.task.launch(item, rec["attempt"], rec["seed"], launch_id=rec["launch_id"])
        return self.task.launch(item, rec["attempt"], rec["seed"])

    def _launch(self, item: str, rec: dict[str, Any], now: float, *, half_open: bool) -> None:
        state = self._state
        rec["pre_launch_state"] = rec["state"]
        rec["tries"] += 1
        rec["launch_id"] = launch_key(item, rec["attempt"], rec["tries"])
        rec["state"] = LAUNCHING
        rec["intent_at"] = now
        rec["max_launched_attempt"] = max(rec["max_launched_attempt"], rec["attempt"])
        state["counters"]["launch_attempts"] += 1
        self._event("launch_intent", item=item, attempt=rec["attempt"], seed=rec["seed"],
                    tries=rec["tries"], launch_id=rec["launch_id"], half_open=half_open or None)
        self._touch(item)
        self._save(force=True)  # the intent is durable before anything can start
        try:
            handle = self._call_launch(item, rec)
        except LaunchDeferred as exc:
            rec["state"] = rec.pop("pre_launch_state")
            rec["tries"] -= 1
            rec["retry_not_before"] = now + self.policy.poll_interval_s
            self._event("launch_deferred", item=item, reason=_short(str(exc)))
        except ProviderDown as exc:
            rec["state"] = rec.pop("pre_launch_state")
            rec["tries"] -= 1
            rec["retry_not_before"] = now
            self._event("launch_provider_down", item=item, reason=_short(str(exc)))
            self._provider_down(now, str(exc))
        except LaunchNotStarted as exc:
            self._crash(item, rec, now, Poll("crashed", detail=f"launch failed: {exc}"),
                        launched=False)
        except LaunchOutcomeUnknown as exc:
            detail = f"launch outcome unknown: {exc}"
            self._event("launch_unknown", item=item, launch_id=rec["launch_id"],
                        reason=_short(detail))
            self._resolve_intent(item, rec, now, launch_error=detail)
        except Exception as exc:  # noqa: BLE001
            detail = f"launch raised {type(exc).__name__}: {exc}"
            if self.task.reconcile is not None:
                self._event("launch_unknown", item=item, launch_id=rec["launch_id"],
                            reason=_short(detail))
                self._resolve_intent(item, rec, now, launch_error=detail)
            else:
                # Without reconcile a raising launcher is assumed not to have started anything
                # (adapters that can partially launch must provide reconcile).
                self._crash(item, rec, now, Poll("crashed", detail=detail), launched=False)
        else:
            try:
                rec["handle"] = _jsonable(handle)
            except TypeError:
                self._mark_unknown(item, rec, "launch returned a handle that cannot be persisted")
                self._save(force=True)
                raise
            rec["state"] = RUNNING
            rec["launched_at"] = now
            rec.pop("pre_launch_state", None)
            state["counters"]["launches"] += 1
            if half_open:
                state["provider"]["half_open"] = False
                state["provider"]["probe_item"] = item
            self._event("launched", item=item, launch_id=rec["launch_id"], handle=rec["handle"])
        self._touch(item)
        self._save(force=True)

    # -- ledger -----------------------------------------------------------------------------
    def _ledger_error(self, exc: BaseException) -> None:
        message = _short(f"{type(exc).__name__}: {exc}")
        if message not in self._ledger_errors:
            self._ledger_errors.add(message)
            self._event("ledger_error", error=message)

    def _flush_ledger(self) -> None:
        if self._ledger is None or not self._ledger_dirty:
            self._ledger_dirty.clear()
            return
        rows = []
        for item in sorted(self._ledger_dirty, key=self._order.get):
            rec = self._state["item_state"][item]
            last = rec["last"] or {}
            failure = None
            if rec["state"] in (REJECTED_RETRYABLE, REJECTED_FINAL, EXHAUSTED):
                failure = rec["terminal_reason"] if rec["state"] == EXHAUSTED else None
                failure = ":".join(str(part) for part in (failure, last.get("gate"),
                                                          last.get("reason") or last.get("detail"))
                                   if part) or rec["state"]
            rows.append({
                "item": item, "status": LEDGER_CELL_STATUS[rec["state"]],
                "attempt": rec["attempt"], "failure": _short(failure, 200),
                "artifact": rec["evidence"] if isinstance(rec["evidence"], str) else None,
                "note": f"batch:{rec['state']}",
            })
        self._ledger_dirty.clear()
        try:
            self._ledger.update_cells(rows)
        except Exception as exc:  # noqa: BLE001 - the ledger observes; it never steers
            self._ledger_error(exc)


# -- crash-burst rule (pure, shared with offline analysis) --------------------------------------
def new_burst_state() -> dict[str, Any]:
    return {"recent": [], "level": 0, "until": None, "last_end": None, "count": 0}


def advance_burst(burst: dict[str, Any], now: float, item: str,
                  policy: BatchPolicy) -> dict[str, Any] | None:
    """Record one transient crash in ``burst`` (mutated); return a backoff trigger or ``None``.

    A burst is ``crash_burst_items`` *distinct* items crashing within ``crash_burst_window_s``: a
    single item crash-looping is an item problem (its own crash budget and retry delay handle it),
    many items crashing together is a provider/uploader problem. Bursts that follow each other
    within the window double the backoff up to ``burst_backoff_cap_s``.
    """

    if policy.crash_burst_items <= 0:
        return None
    recent = [entry for entry in burst["recent"] if now - entry[0] <= policy.crash_burst_window_s]
    recent.append([now, item])
    burst["recent"] = recent
    distinct = {entry[1] for entry in recent}
    in_backoff = burst["until"] is not None and now < burst["until"]
    if len(distinct) < policy.crash_burst_items or in_backoff:
        return None
    repeated = burst["last_end"] is not None and now - burst["last_end"] <= \
        policy.crash_burst_window_s
    burst["level"] = burst["level"] + 1 if repeated else 1
    delay = _backoff(policy.burst_backoff_s, burst["level"], policy.burst_backoff_cap_s)
    burst["until"] = now + delay
    burst["last_end"] = burst["until"]
    burst["count"] += 1
    burst["recent"] = []
    return {"level": burst["level"], "backoff_s": delay, "distinct_items": len(distinct),
            "window_s": policy.crash_burst_window_s, "items": sorted(distinct)[:20]}


# -- reporting ---------------------------------------------------------------------------------
def summarize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    items = state["item_state"]
    counts = Counter(rec["state"] for rec in items.values())
    admitted = [item for item in state["items"] if items[item]["state"] == ADMITTED]
    gaps = [
        {"item": item, "state": rec["state"], "reason": rec["terminal_reason"],
         "attempts_used": rec["attempts_used"], "crash_retries": rec["crash_retries"],
         "last": rec["last"]}
        for item, rec in ((item, items[item]) for item in state["items"])
        if rec["state"] in (REJECTED_FINAL, EXHAUSTED)
    ]
    in_flight = [
        {"item": item, "state": rec["state"], "attempt": rec["attempt"],
         "launch_id": rec["launch_id"], "unknown_streak": rec["unknown_streak"]}
        for item, rec in ((item, items[item]) for item in state["items"])
        if rec["state"] in IN_FLIGHT
    ]
    counters = state["counters"]
    return {
        "schema": REPORT_SCHEMA,
        "batch_id": state.get("batch_id"),
        "status": state["status"],
        "terminal": state.get("terminal"),
        "stop_request": state.get("stop_request"),
        "denominator": len(state["items"]),
        "admitted": len(admitted),
        "counts": {name: counts.get(name, 0) for name in ITEM_STATES},
        "admitted_items": admitted,
        "gaps": gaps,
        "in_flight": in_flight,
        "rounds": {"closed": state["rounds"]["closed"],
                   "admitted_by_round": state["rounds"]["admitted_by_round"],
                   "zero_progress_streak": state["rounds"]["zero_progress_streak"]},
        "counters": dict(counters),
        "crash_bursts": state["burst"]["count"],
        "provider_outages": state["provider"]["outages"],
        "provider_down": state["provider"]["down"],
        "policy_sha256": state["policy_sha256"],
        "policy": state["policy"],
        "created_at": state["created_at"],
        "updated_at": state["updated_at"],
        "items": {item: {"state": items[item]["state"], "attempt": items[item]["attempt"],
                         "attempts_used": items[item]["attempts_used"],
                         "crash_retries": items[item]["crash_retries"],
                         "admitted_attempt": items[item]["admitted_attempt"],
                         "terminal_reason": items[item]["terminal_reason"]}
                  for item in state["items"]},
    }


__all__ = [
    "ADMITTED", "BatchError", "BatchPolicy", "BatchSupervisor", "BatchTask", "COMPLETE",
    "COMPLETE_WITH_GAPS", "CRASHED_RETRYABLE", "DenominatorMismatch", "EXHAUSTED", "IN_FLIGHT",
    "ITEM_STATES", "LAUNCHING", "LaunchDeferred", "LaunchNotStarted", "LaunchOutcomeUnknown",
    "LedgerHook", "PAUSED_COST_UNKNOWN", "PAUSED_PROVIDER_DOWN", "PAUSED_UNKNOWN_LIVENESS", "PENDING", "Poll",
    "ProviderDown", "REJECTED_FINAL", "REJECTED_RETRYABLE", "RUNNING", "STOPPED_COST_CAP",
    "STOPPED_OPERATOR", "STOPPED_SYSTEMATIC_FAILURE", "STOPPED_WALL_BUDGET",
    "STOPPED_ZERO_PROGRESS", "SupervisorBusy", "TERMINAL_STATUSES", "UNKNOWN", "Verdict",
    "advance_burst", "default_seed", "launch_key", "new_burst_state",
    "read_events", "summarize_state", "write_json_atomic",
]
