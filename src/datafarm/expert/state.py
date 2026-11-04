"""Persistent round state (harness-private) and the two event logs.

The state file is rewritten atomically after every mutation so a harness restart resumes
the same round: running attempts are re-polled from their run directories and the agent
session is resumed by its recorded session id. Ledgers are append-only and never
rewritten; the state is a cache that can be cross-checked against them.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from datafarm.common import append_jsonl
from datafarm.expert.layout import RoundPaths
from datafarm.expert.util import HarnessError, atomic_write_json, read_json, utc_now

STATE_SCHEMA = "datafarm.expert-harness.state/1"
EVENT_SCHEMA = "datafarm.expert-harness.event/1"

TERMINAL = frozenset({"completed", "flagged_for_human", "wall_exhausted", "usd_exhausted",
                      "rollouts_exhausted", "sessions_exhausted", "stalled", "infra_exhausted",
                      "invalid_tamper", "error"})
ACTIVE = frozenset({"prepared", "running", "paused_model_down"})


class RoundState:
    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self.data = data

    @property
    def bounded_unknown_costs(self) -> dict[int, float]:
        """Operator-audited liabilities retain budget without pretending to be receipts."""
        result = {}
        sessions = {s['index']: s for s in self.data['sessions']}
        for record in self.data.get('bounded_unknown_costs', []):
            index = record['session_index']
            amount = float(record['reserved_usd'])
            session = sessions.get(index)
            if (index in result or not math.isfinite(amount) or amount <= 0
                    or not session or not session.get('ended_at')
                    or session.get('cost_usd') is not None
                    or not record.get('evidence')):
                raise HarnessError('invalid operator-bounded unknown cost record')
            result[index] = amount
        return result

    @classmethod
    def create(cls, path: Path, *, round_id: str, created_at: str) -> "RoundState":
        data = {
            "schema": STATE_SCHEMA,
            "round_id": round_id,
            "status": "prepared",
            "created_at": created_at,
            "wall_started_at": None,
            "paused_s": 0.0,
            "paused_at": None,
            "last_tick_at": None,
            "sessions": [],
            "agent_session_id": None,
            "next_attempt": 1,
            "next_submission": 1,
            "attempts": {},
            "submissions": [],
            "counters": {"counted": 0, "infra": 0, "successes": 0, "rejected": 0,
                         "non_dev_seed": 0, "holdout_seed": 0, "audit_rejected": 0,
                         "unsafe_path": 0, "invalid_request": 0, "submit_retries": 0},
            "stop": None,
            "violations": [],
            "max_in_flight_observed": 0,
            "outage": None,
            "idle_sessions": 0,
            "manifest": {"start_digest": None, "checks": 0, "ok": True, "amendments": 0},
            "isolation": {},
            "cost_usd_known": 0.0,
            "sessions_unknown_cost": 0,
        }
        state = cls(path, data)
        state.save()
        return state

    @classmethod
    def load(cls, path: Path) -> "RoundState":
        try:
            data = read_json(path)
        except (OSError, ValueError) as exc:
            raise HarnessError(f"cannot read round state {path}: {exc}") from exc
        if data.get("schema") != STATE_SCHEMA:
            raise HarnessError(f"{path}: not a {STATE_SCHEMA} file")
        return cls(path, data)

    def save(self) -> None:
        atomic_write_json(self.path, self.data, mode=0o600)

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    # convenience accessors -------------------------------------------------------
    @property
    def status(self) -> str:
        return self.data["status"]

    @status.setter
    def status(self, value: str) -> None:
        self.data["status"] = value

    @property
    def counters(self) -> dict[str, int]:
        return self.data["counters"]

    def bump(self, name: str, by: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + by

    def violation(self, kind: str, detail: str, utc: str) -> None:
        self.data["violations"].append({"kind": kind, "detail": detail, "utc": utc})

    def active_wall_s(self, now: float) -> float:
        started = self.data.get("wall_started_at")
        if started is None:
            return 0.0
        paused = float(self.data.get("paused_s") or 0.0)
        if self.data.get("paused_at") is not None:
            paused += max(0.0, now - float(self.data["paused_at"]))
        elif self.data.get("outage"):
            # an ongoing provider outage is not charged to the round
            paused += max(0.0, now - float(self.data["outage"]["started_at"]))
        return max(0.0, now - float(started) - paused)

    def request_stop(self, reason: str, detail: str, utc: str) -> None:
        if self.data.get("stop") is None:
            self.data["stop"] = {"reason": reason, "detail": detail, "utc": utc}

    @property
    def stop_reason(self) -> str | None:
        stop = self.data.get("stop")
        return stop["reason"] if stop else None


class EventLog:
    """``ledger/events.jsonl`` (agent-readable) and ``harness/events.jsonl`` (private)."""

    def __init__(self, paths: RoundPaths, round_id: str, clock_now) -> None:
        self.paths = paths
        self.round_id = round_id
        self.now = clock_now

    def emit(self, event: str, *, private: bool = False, **fields: Any) -> dict[str, Any]:
        record = {"schema": EVENT_SCHEMA, "utc": utc_now(self.now()), "round_id": self.round_id,
                  "event": event, **fields}
        append_jsonl(self.paths.private_events if private else self.paths.events, record)
        return record
