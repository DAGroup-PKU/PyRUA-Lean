"""The broker: the harness side of the agent's only tool (``xh``).

The agent writes a request file into ``broker/requests/``; the harness (outside the
sandbox) validates it, launches the rollout through the adapter's backend, and writes
the response. Launching rollouts, keeping the attempt log and counting concurrency happen
here, and every
decision (accepted, rejected and why) is recorded in harness-owned ledgers the agent can
read but not write.

Rules enforced per request: development seeds only (a holdout seed is a recorded
violation), static expert audit, concurrency counted from live attempts (not from
scheduler leases, which can vanish and let rollouts exceed the limit),
counted-rollout budget including rollouts in flight, infrastructure-failure cap, wall
budget, and round state (closing after a final submission or a flag for a human).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from datafarm.common import append_jsonl
from datafarm.expert.adapter import TaskAdapter
from datafarm.expert.audit import AuditReport, audit_expert_source
from datafarm.expert.layout import RoundPaths
from datafarm.expert.rollout import (AttemptSpec, LaunchError, RolloutBackend, classify_run_dir, job_record,
                      summarize_outcome)
from datafarm.expert.state import EventLog, RoundState
from datafarm.expert.util import (Clock, UnsafePathError, atomic_write_bytes, atomic_write_json,
                   clip_text, read_regular_nofollow, safe_relpath, sha256_bytes, sha256_json,
                   utc_now)

REQUEST_SCHEMA = "datafarm.expert-harness.request/1"
RESPONSE_SCHEMA = "datafarm.expert-harness.response/1"
RESULT_SCHEMA = "datafarm.expert-harness.result/1"
STATUS_SCHEMA = "datafarm.expert-harness.status/1"
ATTEMPT_SCHEMA = "datafarm.expert-harness.attempt/1"
SUBMISSION_SCHEMA = "datafarm.expert-harness.submission/1"

MAX_REQUEST_BYTES = 64 * 1024
MAX_REQUESTS_PER_TICK = 20
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,80}$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")
IN_FLIGHT = ("pending_submit", "running")


class Broker:
    def __init__(self, *, paths: RoundPaths, adapter: TaskAdapter, state: RoundState,
                 backend: RolloutBackend, events: EventLog, clock: Clock,
                 holdout_seeds: set[int] | frozenset[int] = frozenset(),
                 cost_fn: Callable[[], float] = lambda: 0.0, poll_interval_s: float = 10.0,
                 submit_retry_s: float = 60.0, submit_window_s: float = 1800.0,
                 revive_window_s: float = 7200.0, status_every_s: float = 10.0) -> None:
        self.paths = paths
        self.adapter = adapter
        self.state = state
        self.backend = backend
        self.events = events
        self.clock = clock
        self.holdout_seeds = set(holdout_seeds)
        self.cost_fn = cost_fn
        self.poll_interval_s = poll_interval_s
        self.submit_retry_s = submit_retry_s
        self.submit_window_s = submit_window_s
        self.revive_window_s = revive_window_s
        self.status_every_s = status_every_s
        self.policy = adapter.expert_policy
        self.budgets = adapter.budgets
        self.dev_seeds = adapter.dev_seeds
        self.session_index = 0
        self._last_poll = float("-inf")
        self._last_status = float("-inf")
        self._last_status_digest: str | None = None

    # ------------------------------------------------------------------ helpers

    @property
    def attempts(self) -> dict[str, dict[str, Any]]:
        return self.state.data["attempts"]

    def now(self) -> float:
        return self.clock.now()

    def utc(self) -> str:
        return utc_now(self.now())

    def in_flight_ids(self) -> list[str]:
        now = self.now()
        ids = [aid for aid, a in self.attempts.items() if a["status"] in IN_FLIGHT]
        for aid, a in self.attempts.items():
            if a["status"] == "lost" and self._alive_again(a, now):
                ids.append(aid)
        return sorted(ids)

    def _alive_again(self, attempt: Mapping[str, Any], now: float) -> bool:
        """A job declared lost whose heartbeat moved again still occupies a GPU.
        (File times are wall-clock times, so this uses the real clock.)"""

        try:
            mtime = (Path(attempt["run_dir"]) / "heartbeat").stat().st_mtime
        except OSError:
            return False
        lost_at = float(attempt.get("lost_at_real") or attempt.get("ended_at_ts") or 0.0)
        stale = float(self.adapter.rollout.get("stale_after_s", 900))
        return mtime > lost_at and time.time() - mtime < stale

    def counted(self) -> int:
        return int(self.state.counters.get("counted", 0))

    def infra(self) -> int:
        return int(self.state.counters.get("infra", 0))

    def active_wall(self) -> float:
        return self.state.active_wall_s(self.now())

    def accepting_rollouts(self) -> tuple[bool, str, str]:
        stop = self.state.data.get("stop")
        if stop:
            return False, "round_closed", f"the round is closing ({stop['reason']})"
        if self.state.status != "running":
            return False, "round_closed", f"the round is {self.state.status}"
        if self.active_wall() >= self.budgets.total_wall_s:
            return False, "round_closed", "the wall-clock budget is exhausted"
        in_flight = len(self.in_flight_ids())
        if self.counted() + in_flight >= self.budgets.max_counted_rollouts:
            return False, "budget", (f"counted-rollout budget exhausted ({self.counted()} used, "
                                     f"{in_flight} in flight, max {self.budgets.max_counted_rollouts}); "
                                     "freeze your best expert and submit it")
        if self.infra() >= self.budgets.max_infra_failures:
            return False, "infra_budget", "too many infrastructure failures; an operator must look"
        if in_flight >= self.budgets.max_concurrent_rollouts:
            return False, "concurrency", (f"{in_flight} rollouts are running (limit "
                                          f"{self.budgets.max_concurrent_rollouts}); wait with "
                                          "`xh wait` and retry")
        return True, "ok", ""

    # ------------------------------------------------------------------ tick

    def tick(self, *, force_poll: bool = False) -> None:
        """Handle requests, poll rollouts (throttled in real time: it touches the shared
        filesystem), refresh the agent-visible status."""

        self.process_requests()
        now = self.now()
        real = time.monotonic()
        if force_poll or real - self._last_poll >= self.poll_interval_s:
            self._last_poll = real
            self.poll(now)
        self.write_status(now)

    # ------------------------------------------------------------------ requests

    def process_requests(self) -> int:
        try:
            names = sorted(n for n in os.listdir(self.paths.requests)
                           if n.endswith(".json") and not n.startswith("."))
        except FileNotFoundError:
            return 0
        handled = 0
        for name in names[:MAX_REQUESTS_PER_TICK]:
            rid = name[: -len(".json")]
            response = self._process_one(name, rid)
            if response is not None and _REQUEST_ID.match(rid):
                target = self.paths.responses / f"{rid}.json"
                if not target.exists():
                    atomic_write_json(target, response, mode=0o644)
            try:
                os.replace(self.paths.requests / name,
                           self.paths.processed / f"{rid[:80]}.{uuid.uuid4().hex[:8]}.json")
            except OSError:
                pass
            handled += 1
        if handled:
            self.state.save()
        return handled

    def _reject(self, rid: str, kind: str, code: str, reason: str, **extra: Any) -> dict[str, Any]:
        self.state.bump("rejected")
        self.events.emit("request_rejected", request_id=rid, kind=kind, code=code, reason=reason,
                         session_index=self.session_index, **extra)
        return {"schema": RESPONSE_SCHEMA, "request_id": rid, "kind": kind, "status": "rejected",
                "code": code, "reason": reason, "utc": self.utc()}

    def _process_one(self, name: str, rid: str) -> dict[str, Any] | None:
        if not _REQUEST_ID.match(rid):
            self.state.bump("invalid_request")
            self.events.emit("request_rejected", request_id=None, kind=None, code="invalid_request",
                             reason="request file name is not a valid request id")
            return None
        if (self.paths.responses / f"{rid}.json").exists():
            return None  # duplicate id: the first answer stands
        try:
            raw = read_regular_nofollow(self.paths.requests, name, max_bytes=MAX_REQUEST_BYTES)
        except (UnsafePathError, OSError) as exc:
            self.state.bump("unsafe_path")
            self.state.violation("unsafe_request_file", str(exc), self.utc())
            return self._reject(rid, "unknown", "unsafe_path", f"request file rejected: {exc}")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            self.state.bump("invalid_request")
            return self._reject(rid, "unknown", "invalid_request", f"not JSON: {exc}")
        if not isinstance(body, dict) or body.get("schema") != REQUEST_SCHEMA \
                or body.get("request_id") != rid or not isinstance(body.get("args", {}), dict):
            self.state.bump("invalid_request")
            return self._reject(rid, str(body.get("kind") if isinstance(body, dict) else None),
                                "invalid_request", "malformed request (schema, request_id or args)")
        kind = body.get("kind")
        args = body.get("args", {})
        if kind == "rollout":
            return self._rollout(rid, args)
        if kind == "submit":
            return self._submit(rid, args)
        if kind == "flag-for-human":
            return self._flag(rid, args)
        self.state.bump("invalid_request")
        return self._reject(rid, str(kind), "invalid_request", f"unknown request kind {kind!r}")

    def _read_expert(self, rid: str, kind: str, args: Mapping[str, Any]
                     ) -> tuple[str, bytes, str, AuditReport] | dict[str, Any]:
        rel = args.get("file") or self.policy.get("default_file", "expert.py")
        class_name = args.get("class") or self.policy.get("default_class")
        if not isinstance(class_name, str) or not _IDENT.match(class_name):
            self.state.bump("invalid_request")
            return self._reject(rid, kind, "invalid_request", "class must be a Python identifier")
        try:
            rel = safe_relpath(rel)
            data = read_regular_nofollow(self.paths.workspace, rel,
                                         max_bytes=int(self.policy.get("max_bytes", 400_000)))
        except (UnsafePathError, OSError) as exc:
            self.state.bump("unsafe_path")
            self.state.violation("unsafe_expert_path", f"{rel!r}: {exc}", self.utc())
            return self._reject(rid, kind, "unsafe_path",
                                f"expert file must be a regular file inside the workspace: {exc}")
        audit = audit_expert_source(data, class_name, self.policy)
        if not audit.ok:
            self.state.bump("audit_rejected")
            return self._reject(rid, kind, "audit", "static expert audit failed: " + audit.summary(),
                                violations=audit.violations[:20])
        return rel, data, class_name, audit

    def _rollout(self, rid: str, args: Mapping[str, Any]) -> dict[str, Any]:
        seed = args.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            self.state.bump("invalid_request")
            return self._reject(rid, "rollout", "invalid_request", "seed must be an integer")
        if seed not in self.dev_seeds:
            if seed in self.holdout_seeds:
                self.state.bump("holdout_seed")
                self.state.violation("holdout_seed_request", f"seed {seed}", self.utc())
                code = "holdout_seed"
            else:
                self.state.bump("non_dev_seed")
                self.state.violation("non_dev_seed_request", f"seed {seed}", self.utc())
                code = "not_dev_seed"
            return self._reject(rid, "rollout", code,
                                f"seed {seed} is not a development seed (allowed: "
                                f"{self.dev_seeds.describe()}); this request is recorded")
        ok, code, reason = self.accepting_rollouts()
        if not ok:
            return self._reject(rid, "rollout", code, reason)
        expert = self._read_expert(rid, "rollout", args)
        if isinstance(expert, dict):
            return expert
        rel, data, class_name, _ = expert
        digest = sha256_bytes(data)
        if self.policy.get("reject_unchanged_rollouts") and args.get("repeat") is not True:
            previous = next((a for a in reversed(list(self.attempts.values()))
                if a["seed"] == seed and a["expert_class"] == class_name
                and a["expert_sha256"] == digest and a["status"] == "finished"
                and (a.get("outcome") or {}).get("counted")), None)
            if previous:
                return self._reject(rid, "rollout", "unchanged_expert",
                    "Identical expert, class and seed were already tested in " + previous["attempt_id"] +
                    ". Check that your edit command succeeded and inspect the source. "
                    "Read the existing result, or use --repeat for intentional replication.",
                    previous_attempt_id=previous["attempt_id"], expert_sha256=digest)
        number = int(self.state.data["next_attempt"])
        self.state.data["next_attempt"] = number + 1
        attempt_id = f"A-{number:04d}"
        run_dir = self.paths.rollouts / f"{attempt_id}-s{seed}"
        digest = sha256_bytes(data)
        now = self.now()
        attempt = {
            "attempt_id": attempt_id, "request_id": rid, "session_index": self.session_index,
            "seed": seed, "expert_class": class_name, "expert_file": rel, "expert_sha256": digest,
            "expert_bytes": len(data), "note": clip_text(args.get("note"), 1000),
            "hypothesis": clip_text(args.get("hypothesis"), 1000), "run_dir": str(run_dir),
            "status": "pending_submit", "requested_at_ts": now, "last_try_ts": None,
            "submit_tries": 0, "receipt": None, "launched_at_ts": None, "ended_at_ts": None,
            "outcome": None,
        }
        self.attempts[attempt_id] = attempt
        spec = self._spec(attempt_id)
        from datafarm.expert.rollout import prepare_run_dir

        prepare_run_dir(spec, data, self.adapter)
        self._launch(attempt_id, now)
        self.state.data["max_in_flight_observed"] = max(
            int(self.state.data.get("max_in_flight_observed", 0)), len(self.in_flight_ids()))
        self.state.save()
        return {"schema": RESPONSE_SCHEMA, "request_id": rid, "kind": "rollout",
                "status": "accepted", "attempt_id": attempt_id, "seed": seed,
                "expert_class": class_name, "expert_sha256": digest,
                "state": "running" if attempt["status"] == "running" else "queued",
                "queue_reason": attempt.get("last_submit_error"), "run_dir": str(run_dir),
                "budget": self.budget_view(now), "utc": self.utc()}

    def _submit(self, rid: str, args: Mapping[str, Any]) -> dict[str, Any]:
        stop = self.state.data.get("stop")
        if stop and stop["reason"] == "flagged_for_human":
            return self._reject(rid, "submit", "round_closed", "the round was flagged for a human")
        if any(s.get("final") for s in self.state.data["submissions"]):
            return self._reject(rid, "submit", "round_closed", "a final submission was already made")
        final = args.get("final", False)
        if not isinstance(final, bool):
            return self._reject(rid, "submit", "invalid_request", "final must be true or false")
        expert = self._read_expert(rid, "submit", args)
        if isinstance(expert, dict):
            return expert
        rel, data, class_name, _ = expert
        return self.record_submission(data, class_name, rel, note=args.get("note"), final=final,
                                      request_id=rid, source="agent")

    def record_submission(self, data: bytes, class_name: str, rel: str, *, note: Any, final: bool,
                          request_id: str | None, source: str) -> dict[str, Any]:
        number = int(self.state.data["next_submission"])
        self.state.data["next_submission"] = number + 1
        sid = f"S-{number:02d}"
        digest = sha256_bytes(data)
        target = self.paths.submissions / sid
        target.mkdir(parents=True, exist_ok=False)
        atomic_write_bytes(target / "expert.py", data, mode=0o444)
        same = [a for a in self.attempts.values()
                if a["expert_sha256"] == digest and a["status"] == "finished"
                and (a.get("outcome") or {}).get("counted")]
        successes = sum(1 for a in same if (a.get("outcome") or {}).get("success"))
        now = self.now()
        record = {
            "schema": SUBMISSION_SCHEMA, "submission_id": sid, "request_id": request_id,
            "source": source, "final": final, "note": clip_text(note, 2000),
            "expert_class": class_name, "expert_file": rel, "expert_sha256": digest,
            "utc": utc_now(now), "ts": now, "active_wall_s": round(self.active_wall(), 1),
            "session_index": self.session_index, "counted_rollouts": self.counted(),
            "cost_usd_completed_sessions": round(self.cost_fn(), 6),
            "dev_rollouts_same_expert": len(same), "dev_successes_same_expert": successes,
        }
        atomic_write_json(target / "submission.json", record, mode=0o444)
        self.state.data["submissions"].append(record)
        warnings = []
        if not same:
            warnings.append("this exact expert has no completed development rollout; its "
                            "behaviour is unverified")
        if final:
            self.state.request_stop("done", f"final submission {sid}", utc_now(now))
        self.events.emit("submission", submission_id=sid, expert_sha256=digest, final=final,
                         source=source, session_index=self.session_index,
                         dev_rollouts_same_expert=len(same))
        self.state.save()
        return {"schema": RESPONSE_SCHEMA, "request_id": request_id, "kind": "submit",
                "status": "accepted", "submission_id": sid, "expert_sha256": digest,
                "final": final, "dev_rollouts_same_expert": len(same),
                "dev_successes_same_expert": successes, "warnings": warnings,
                "note": "Submissions are scored later by the harness on hidden holdout seeds; "
                        "no holdout score is ever returned to you.", "utc": utc_now(now)}

    def _flag(self, rid: str, args: Mapping[str, Any]) -> dict[str, Any]:
        reason = args.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return self._reject(rid, "flag-for-human", "invalid_request", "a reason is required")
        now = self.now()
        self.state.data["flag"] = {"reason": clip_text(reason, 4000), "utc": utc_now(now),
                                   "session_index": self.session_index}
        self.state.request_stop("flagged_for_human", clip_text(reason, 300), utc_now(now))
        self.events.emit("flag_for_human", reason=clip_text(reason, 4000),
                         session_index=self.session_index)
        self.state.save()
        return {"schema": RESPONSE_SCHEMA, "request_id": rid, "kind": "flag-for-human",
                "status": "accepted", "message": "Recorded. The round ends after this session; an "
                                                 "operator will read your reason.", "utc": utc_now(now)}

    # ------------------------------------------------------------------ attempts

    def _spec(self, attempt_id: str) -> AttemptSpec:
        a = self.attempts[attempt_id]
        short = re.sub(r"[^A-Za-z0-9]", "", self.state.data["round_id"])[-10:]
        return AttemptSpec(attempt_id=attempt_id, purpose="dev", seed=int(a["seed"]),
                           expert_class=a["expert_class"], run_dir=Path(a["run_dir"]),
                           job_name=f"xh-{short}-{attempt_id}")

    def _launch(self, attempt_id: str, now: float) -> None:
        a = self.attempts[attempt_id]
        a["last_try_ts"] = now
        a["submit_tries"] = int(a.get("submit_tries", 0)) + 1
        try:
            receipt = self.backend.launch(self._spec(attempt_id))
        except LaunchError as exc:
            a["last_submit_error"] = str(exc)[:300]
            if a["submit_tries"] > 1:
                self.state.bump("submit_retries")
            if now - float(a["requested_at_ts"]) >= self.submit_window_s:
                self._finish(attempt_id, "submit_failed", now)
            return
        a["status"] = "running"
        a["receipt"] = dict(receipt)
        a["launched_at_ts"] = now
        a["last_submit_error"] = None
        append_jsonl(self.paths.attempts, {
            "schema": ATTEMPT_SCHEMA, "event": "launch", "utc": utc_now(now),
            "round_id": self.state.data["round_id"], **{k: a[k] for k in (
                "attempt_id", "request_id", "session_index", "seed", "expert_class", "expert_file",
                "expert_sha256", "expert_bytes", "note", "hypothesis", "run_dir", "submit_tries")},
            "receipt": {k: v for k, v in receipt.items() if k in ("backend", "id", "host", "pid", "gpus")},
        })

    def poll(self, now: float) -> None:
        changed = False
        for attempt_id, a in list(self.attempts.items()):
            if a["status"] == "pending_submit":
                if a.get("last_try_ts") is None or now - float(a["last_try_ts"]) >= self.submit_retry_s:
                    self._launch(attempt_id, now)
                    changed = True
                continue
            if a["status"] == "running":
                status = self.backend.poll(self._spec(attempt_id), a["receipt"] or {}, now)
                if status != "running":
                    self._finish(attempt_id, status, now)
                    changed = True
            elif a["status"] in ("lost", "timeout") \
                    and time.time() - float(a.get("lost_at_real") or 0.0) < self.revive_window_s \
                    and (Path(a["run_dir"]) / "harness-job.json").exists():
                self._revive(attempt_id, now)
                changed = True
        in_flight = len(self.in_flight_ids())
        if in_flight > int(self.state.data.get("max_in_flight_observed", 0)):
            self.state.data["max_in_flight_observed"] = in_flight
            changed = True
        if in_flight > self.budgets.max_concurrent_rollouts:
            self.state.violation("concurrency", f"{in_flight} rollouts alive (limit "
                                                f"{self.budgets.max_concurrent_rollouts})", utc_now(now))
        if changed:
            self.state.save()

    def _finish(self, attempt_id: str, status: str, now: float) -> None:
        a = self.attempts[attempt_id]
        spec = self._spec(attempt_id)
        if status == "finished":
            outcome = self.backend.outcome(spec, a.get("receipt") or {}, status)
        else:
            outcome = classify_run_dir(spec.run_dir, {}, status=status)
        a["status"] = "finished" if status == "finished" else status
        a["outcome"] = summarize_outcome(outcome)
        a["ended_at_ts"] = now
        if status in ("lost", "timeout"):
            a["lost_at_real"] = time.time()
        job = job_record(spec.run_dir)
        if job is not None:
            a["isolation"] = job.get("isolation_backend") or job.get("isolation")
            if job.get("isolation") == "none":
                self.state.bump("unisolated_rollouts")
                if self.state.counters["unisolated_rollouts"] == 1:
                    self.state.violation("isolation", "rollouts run without OS isolation", utc_now(now))
        self._count(outcome.outcome_class, outcome.counted, outcome.success, +1)
        self._record_end(attempt_id, "end", now)
        if self.infra() >= self.budgets.max_infra_failures:
            self.state.request_stop("infra_exhausted", f"{self.infra()} infrastructure failures",
                                    utc_now(now))

    def _count(self, cls: str, counted: bool, success: bool | None, sign: int) -> None:
        if counted:
            self.state.bump("counted", sign)
            if success:
                self.state.bump("successes", sign)
        elif cls != "submit_failed":
            self.state.bump("infra", sign)

    def _revive(self, attempt_id: str, now: float) -> None:
        a = self.attempts[attempt_id]
        old = a.get("outcome") or {}
        self._count(old.get("outcome_class", "lost"), bool(old.get("counted")), old.get("success"), -1)
        outcome = self.backend.outcome(self._spec(attempt_id), a.get("receipt") or {}, "finished")
        a["status"] = "finished"
        a["revived"] = True
        a["outcome"] = summarize_outcome(outcome)
        self._count(outcome.outcome_class, outcome.counted, outcome.success, +1)
        self.state.violation("late_finish", f"{attempt_id} finished after being declared "
                                            f"{old.get('outcome_class')}", utc_now(now))
        self._record_end(attempt_id, "revived", now)

    def _record_end(self, attempt_id: str, event: str, now: float) -> None:
        a = self.attempts[attempt_id]
        outcome = a["outcome"] or {}
        append_jsonl(self.paths.attempts, {
            "schema": ATTEMPT_SCHEMA, "event": event, "utc": utc_now(now),
            "round_id": self.state.data["round_id"], "attempt_id": attempt_id,
            "seed": a["seed"], "expert_sha256": a["expert_sha256"], "session_index": a["session_index"],
            "status": a["status"], "wall_s": outcome.get("wall_s"),
            "elapsed_s": None if a.get("launched_at_ts") is None else round(now - float(a["launched_at_ts"]), 1),
            **{k: outcome.get(k) for k in ("outcome_class", "counted", "success", "family",
                                           "init_valid", "fields", "sim_exit", "detail")},
        })
        result = {
            "schema": RESULT_SCHEMA, "attempt_id": attempt_id, "seed": a["seed"],
            "expert_class": a["expert_class"], "expert_sha256": a["expert_sha256"],
            "note": a["note"], "status": a["status"], "run_dir": a["run_dir"], "utc": utc_now(now),
            **{k: outcome.get(k) for k in ("outcome_class", "counted", "success", "family",
                                           "init_valid", "fields", "wall_s", "detail")},
        }
        atomic_write_json(self.paths.results / f"{attempt_id}.json", result, mode=0o644)
        self.events.emit("attempt_" + event, attempt_id=attempt_id, seed=a["seed"],
                         outcome_class=outcome.get("outcome_class"), counted=outcome.get("counted"),
                         success=outcome.get("success"))

    def drain(self, timeout_s: float, *, sleep_s: float = 5.0) -> list[str]:
        """Wait for rollouts in flight (round end); what is still running is abandoned."""

        deadline = self.now() + timeout_s
        while self.in_flight_ids() and self.now() < deadline:
            self.tick(force_poll=True)
            if self.in_flight_ids():
                self.clock.sleep(sleep_s)
        abandoned = [aid for aid in self.in_flight_ids() if self.attempts[aid]["status"] in IN_FLIGHT]
        for aid in abandoned:
            self._finish(aid, "abandoned", self.now())
        self.state.save()
        self.write_status(self.now(), force=True)
        return abandoned

    # ------------------------------------------------------------------ views

    def budget_view(self, now: float) -> dict[str, Any]:
        b = self.budgets
        in_flight = len(self.in_flight_ids())
        active = self.state.active_wall_s(now)
        return {
            "counted_rollouts": {"used": self.counted(), "in_flight": in_flight,
                                 "max": b.max_counted_rollouts,
                                 "remaining": max(0, b.max_counted_rollouts - self.counted() - in_flight)},
            "concurrency": {"running": in_flight, "max": b.max_concurrent_rollouts},
            "wall_s": {"active": round(active), "max": b.total_wall_s,
                       "remaining": round(max(0.0, b.total_wall_s - active))},
            "usd": {"spent_completed_sessions": round(self.cost_fn(), 6),
                    "unknown_liability_reserved": sum(self.state.bounded_unknown_costs.values()),
                    "budget_used": self.cost_fn() + sum(self.state.bounded_unknown_costs.values()),
                    "max": b.max_usd},
            "sessions": {"used": sum(1 for s in self.state.data["sessions"] if not s.get("api_outage")),
                         "max": b.max_sessions},
            "infra_failures": {"used": self.infra(), "max": b.max_infra_failures},
            "successes": int(self.state.counters.get("successes", 0)),
        }

    def status_view(self, now: float) -> dict[str, Any]:
        budget = self.budget_view(now)
        running = []
        for aid in self.in_flight_ids():
            a = self.attempts[aid]
            start = a.get("launched_at_ts") or a.get("requested_at_ts")
            running.append({"attempt_id": aid, "seed": a["seed"],
                            "state": "queued" if a["status"] == "pending_submit" else "running",
                            "elapsed_s": round(now - float(start)), "note": a["note"][:120]})
        finished = [a for a in self.attempts.values() if a["status"] not in IN_FLIGHT and a.get("outcome")]
        finished.sort(key=lambda a: a.get("ended_at_ts") or 0)
        recent = [{"attempt_id": a["attempt_id"], "seed": a["seed"], "family": a["outcome"].get("family"),
                   "outcome_class": a["outcome"].get("outcome_class"),
                   "counted": a["outcome"].get("counted"), "success": a["outcome"].get("success"),
                   "fields": a["outcome"].get("fields")} for a in finished[-12:]]
        messages = []
        remaining_wall = budget["wall_s"]["remaining"]
        if remaining_wall < self.budgets.warn_before_wall_s:
            messages.append(f"Less than {int(remaining_wall // 60) + 1} min of wall-clock budget "
                            "left: freeze your best expert and run `xh submit --final`.")
        if 0 < budget["counted_rollouts"]["remaining"] <= 5:
            messages.append(f"Only {budget['counted_rollouts']['remaining']} counted rollouts left.")
        if budget["counted_rollouts"]["remaining"] == 0 and not running:
            messages.append("No counted rollouts left: submit your final expert now.")
        stop = self.state.data.get("stop")
        if stop:
            messages.append(f"The round is closing: {stop['reason']}.")
        return {
            "schema": STATUS_SCHEMA, "round_id": self.state.data["round_id"],
            "task_id": self.adapter.task_id, "state": self.state.status, "stop": stop,
            "budget": budget, "running": running, "recent": recent,
            "submissions": [{"submission_id": s["submission_id"], "expert_sha256": s["expert_sha256"],
                             "final": s["final"], "utc": s["utc"]} for s in self.state.data["submissions"]],
            "messages": messages, "wait_cap_s": self.budgets.wait_cap_s,
            "dev_seeds": self.dev_seeds.describe(),
        }

    def write_status(self, now: float, *, force: bool = False) -> None:
        view = self.status_view(now)
        # clocks tick every second; rewrite on real changes, or every status_every_s
        stable = {k: v for k, v in view.items() if k not in ("running", "budget")}
        stable["budget"] = {k: v for k, v in view["budget"].items() if k != "wall_s"}
        stable["running"] = [(r["attempt_id"], r["state"]) for r in view["running"]]
        digest = sha256_json(stable)
        real = time.monotonic()
        if not force and digest == self._last_status_digest and real - self._last_status < self.status_every_s:
            return
        view["updated_at"] = utc_now(now)
        atomic_write_json(self.paths.status, view, mode=0o644)
        self._last_status = real
        self._last_status_digest = digest
