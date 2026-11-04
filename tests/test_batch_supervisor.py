"""Hermetic tests for the generic BatchSupervisor (fake clock, fake provider, no processes)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

from datafarm.batch.supervisor import (
    ADMITTED,
    COMPLETE,
    COMPLETE_WITH_GAPS,
    CRASHED_RETRYABLE,
    EXHAUSTED,
    LAUNCHING,
    PAUSED_PROVIDER_DOWN,
    PAUSED_COST_UNKNOWN,
    PAUSED_UNKNOWN_LIVENESS,
    PENDING,
    REJECTED_FINAL,
    REJECTED_RETRYABLE,
    RUNNING,
    STOPPED_COST_CAP,
    STOPPED_OPERATOR,
    STOPPED_SYSTEMATIC_FAILURE,
    STOPPED_WALL_BUDGET,
    STOPPED_ZERO_PROGRESS,
    UNKNOWN,
    BatchError,
    BatchPolicy,
    BatchSupervisor,
    BatchTask,
    DenominatorMismatch,
    LaunchDeferred,
    LaunchNotStarted,
    LedgerHook,
    Poll,
    ProviderDown,
    SupervisorBusy,
    Verdict,
    default_seed,
    launch_key,
    read_events,
)


class FakeClock:
    def __init__(self, start: float = 1_790_000_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


class SupervisorDied(BaseException):
    """Simulates the supervisor process dying (not an ordinary launch error)."""


class World:
    """Scripted fake provider on a fake clock.

    ``script[item]`` is a list of steps consumed one per launch: ``(kind, duration[, extra])``
    with kind in pass | reject | abstain | crash | provider_down | permanent. ``duration=None``
    means the job never ends by itself.
    """

    def __init__(self, script: dict[str, list[tuple]] | None = None,
                 default: tuple = ("pass", 60.0)) -> None:
        self.clock = FakeClock()
        self.script = {item: list(steps) for item, steps in (script or {}).items()}
        self.default = default
        self.jobs: dict[int, dict[str, Any]] = {}
        self.launches: list[dict[str, Any]] = []
        self.peak = 0
        self.poll_hook: Callable[[dict], Any] | None = None
        self.lease_hook: Callable[[dict], Any] | None = None
        self.verify_hook: Callable[[str, dict], Any] | None = None
        self.step_hook: Callable[[str], tuple | None] | None = None
        self.launch_errors: dict[str, list[BaseException | None]] = {}
        self.provider_down_until = -1.0

    def running_now(self) -> int:
        now = self.clock()
        return sum(1 for job in self.jobs.values() if job["start"] <= now < job["end"])

    def launch(self, item: str, attempt: int, seed: int, launch_id: str | None = None) -> dict:
        errors = self.launch_errors.get(item)
        if errors:
            error = errors.pop(0)
            if error is not None:
                raise error
        step = self.step_hook(item) if self.step_hook else None
        if step is None:
            steps = self.script.get(item)
            step = steps.pop(0) if steps else self.default
        kind, duration = step[0], step[1]
        extra = step[2] if len(step) > 2 else None
        now = self.clock()
        job_id = len(self.jobs) + 1
        self.jobs[job_id] = {"id": job_id, "item": item, "attempt": attempt, "seed": seed,
                             "launch_id": launch_id, "kind": kind, "extra": extra, "start": now,
                             "end": float("inf") if duration is None else now + duration}
        self.launches.append({"t": now, "item": item, "attempt": attempt, "seed": seed,
                              "launch_id": launch_id})
        self.peak = max(self.peak, self.running_now())
        return {"job": job_id}

    def poll(self, handle: dict) -> Any:
        job = self.jobs[handle["job"]]
        if self.poll_hook is not None:
            value = self.poll_hook(job)
            if value is not None:
                return value
        if self.clock() < job["end"]:
            return "running"
        kind = job["kind"]
        if kind == "crash":
            return Poll("crashed", detail=job["extra"] or "RuntimeError: bounded upload attempts "
                                                          "exhausted")
        if kind == "provider_down":
            return Poll("crashed", kind="provider_down", detail="503 Service Unavailable")
        if kind == "capacity":
            return Poll("crashed", kind="capacity", detail="no matching free GPU found")
        if kind == "permanent":
            return Poll("crashed", kind="permanent", detail="InvalidParameter: seed too large")
        return Poll("finished", result=job, cost=1.0)

    def verify(self, item: str, job: dict) -> Any:
        if self.verify_hook is not None:
            value = self.verify_hook(item, job)
            if value is not None:
                return value
        if job["kind"] == "pass":
            return "pass"
        if job["kind"] == "abstain":
            return Verdict.abstained("verifier cannot decide")
        gate = job["extra"] or "gate"
        return Verdict.rejected(f"{gate} failed", gate)

    def lease_probe(self, handle: dict) -> Any:
        if self.lease_hook is not None:
            return self.lease_hook(self.jobs[handle["job"]])
        return "alive"

    def reconcile(self, item: str, attempt: int, seed: int, launch_id: str) -> Any:
        for job in self.jobs.values():
            if job["launch_id"] == launch_id:
                return {"job": job["id"]}
        return None

    def provider_probe(self) -> str:
        return "down" if self.clock() < self.provider_down_until else "up"

    def task(self, **overrides: Any) -> BatchTask:
        values: dict[str, Any] = {"launch": self.launch, "poll": self.poll, "verify": self.verify}
        values.update(overrides)
        return BatchTask(**values)

    def launches_for(self, item: str) -> list[dict[str, Any]]:
        return [row for row in self.launches if row["item"] == item]


QUIET = dict(poll_interval_s=10.0, crash_retry_delay_s=0.0, systematic_min_items=0,
             zero_progress_rounds=0)


def policy(**overrides: Any) -> BatchPolicy:
    values = dict(QUIET)
    values.update(overrides)
    return BatchPolicy(**values)


class BatchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.state_dir = self.tmp / "state"
        self.supervisors: list[BatchSupervisor] = []

    def tearDown(self) -> None:
        for supervisor in self.supervisors:
            supervisor.close()
        self._tmp.cleanup()

    def make(self, world: World, items: list[str], batch_policy: BatchPolicy | None = None,
             task: BatchTask | None = None, **kwargs: Any) -> BatchSupervisor:
        kwargs.setdefault("fsync", False)
        supervisor = BatchSupervisor(self.state_dir, items, task or world.task(),
                                     batch_policy or policy(), clock=world.clock,
                                     sleep=world.clock.sleep, **kwargs)
        self.supervisors.append(supervisor)
        return supervisor

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        events = read_events(self.state_dir)
        return [e for e in events if kind is None or e["kind"] == kind]

    def launches_after(self, t: float) -> list[dict[str, Any]]:
        return [e for e in self.events("launch_intent") if e["t"] > t]


class PolicyTests(unittest.TestCase):
    def test_defaults_are_bounded_and_hash_is_stable(self) -> None:
        default = BatchPolicy()
        self.assertEqual(default.sha256(), BatchPolicy.from_dict(default.to_dict()).sha256())
        self.assertGreater(default.max_attempts_per_item, 0)
        self.assertGreater(default.zero_progress_rounds, 0)
        self.assertLessEqual(default.seed_modulus, 2_147_483_647)

    def test_unknown_or_invalid_keys_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            BatchPolicy.from_dict({"max_attempt_per_item": 3})  # typo
        with self.assertRaises(ValueError):
            BatchPolicy(max_attempts_per_item=0)
        with self.assertRaises(ValueError):
            BatchPolicy(systematic_share=0.0)
        for invalid in (float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                BatchPolicy(max_total_cost=invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                BatchPolicy(launch_cost_estimate=invalid)
        with self.assertRaises(ValueError):
            BatchPolicy(max_total_cost=-1.0)
        with self.assertRaises(ValueError):
            BatchPolicy.from_dict({"schema": "other/1.0.0"})

    def test_default_seed_is_deterministic_bounded_and_distinct(self) -> None:
        seeds = {default_seed(7, f"item-{i}", attempt) for i in range(50) for attempt in (1, 2, 3)}
        self.assertEqual(len(seeds), 150)
        self.assertTrue(all(0 <= seed < 2_147_483_647 for seed in seeds))
        self.assertEqual(default_seed(7, "x", 2), default_seed(7, "x", 2))

    def test_launch_key_is_unique_after_sanitizing(self) -> None:
        self.assertEqual(launch_key("s01", 2, 3), "s01.a02.t03")
        self.assertNotEqual(launch_key("a/b", 1, 1), launch_key("a_b", 1, 1))


class CompletionTests(BatchTestCase):
    def test_capacity_wait_keeps_seed_and_does_not_pause_other_items(self):
        world=World({'a':[('capacity',0),('pass',10)],'b':[('pass',10)]})
        supervisor=self.make(world,['a','b'],policy(max_concurrency=1,
                             max_crash_retries_per_item=0,provider_probe_interval_s=60))
        self.assertEqual(supervisor.run(),COMPLETE)
        self.assertEqual(supervisor.item('a')['crash_retries'],0)
        self.assertEqual(len({x['seed'] for x in world.launches_for('a')}),1)
        self.assertEqual(supervisor.summary()['counters']['crashes'],0)
        self.assertFalse(self.events('provider_down'))
        self.assertEqual([x['item'] for x in world.launches],['a','b','a'])

    def test_all_pass_completes_and_respects_concurrency(self) -> None:
        world = World()
        items = [f"s{i:02d}" for i in range(12)]
        supervisor = self.make(world, items, policy(max_concurrency=4))
        self.assertEqual(supervisor.run(), COMPLETE)
        summary = supervisor.summary()
        self.assertEqual((summary["denominator"], summary["admitted"]), (12, 12))
        self.assertEqual(len(world.launches), 12)
        self.assertLessEqual(world.peak, 4)
        self.assertEqual(summary["rounds"]["closed"], 1)
        report = json.loads((self.state_dir / "report.json").read_text())
        self.assertEqual(report["status"], COMPLETE)

    def test_concurrency_cap_holds_through_retries_and_crashes(self) -> None:
        script = {f"s{i}": [("reject", 20.0 + i, "seam"), ("crash", 5.0), ("pass", 30.0)]
                  for i in range(9)}
        world = World(script)
        supervisor = self.make(world, list(script), policy(max_concurrency=3, crash_burst_items=0))
        observed = []
        while supervisor.step() is None:
            observed.append(sum(1 for item in script
                                if supervisor.item(item)["state"] in (LAUNCHING, RUNNING, UNKNOWN)))
            world.clock.sleep(5.0)
        self.assertEqual(supervisor.status, COMPLETE)
        self.assertLessEqual(max(observed), 3)
        self.assertLessEqual(world.peak, 3)

    def test_abstain_consumes_an_attempt_and_changes_seed(self) -> None:
        world = World({"a": [("abstain", 10.0), ("pass", 10.0)]})
        supervisor = self.make(world, ["a"])
        self.assertEqual(supervisor.run(), COMPLETE)
        record = supervisor.item("a")
        self.assertEqual((record["attempts_used"], record["admitted_attempt"]), (2, 2))
        seeds = [row["seed"] for row in world.launches_for("a")]
        self.assertNotEqual(seeds[0], seeds[1])

    def test_permanent_error_is_rejected_final_without_retry(self) -> None:
        world = World({"a": [("permanent", 5.0)]})
        supervisor = self.make(world, ["a", "b"])
        self.assertEqual(supervisor.run(), COMPLETE_WITH_GAPS)
        record = supervisor.item("a")
        self.assertEqual(record["state"], REJECTED_FINAL)
        self.assertEqual(record["attempts_used"], 0)
        self.assertEqual(len(world.launches_for("a")), 1)

    def test_verifier_errors_are_retried_then_become_abstain(self) -> None:
        calls = {"a": 0}

        def flaky(item: str, job: dict) -> Any:
            if item == "a":
                calls["a"] += 1
                if calls["a"] <= 2:
                    raise RuntimeError("verifier crashed")
            if item == "b":
                raise RuntimeError("verifier always crashes")
            return None

        world = World()
        world.verify_hook = flaky
        supervisor = self.make(world, ["a", "b"], policy(max_verify_errors=3,
                                                         max_attempts_per_item=2))
        self.assertEqual(supervisor.run(), COMPLETE_WITH_GAPS)
        self.assertEqual(supervisor.item("a")["state"], ADMITTED)
        self.assertEqual(supervisor.item("a")["attempts_used"], 1)
        b = supervisor.item("b")
        self.assertEqual((b["state"], b["terminal_reason"], b["attempts_used"]),
                         (EXHAUSTED, "attempt_budget", 2))
        self.assertEqual({row["gate"] for row in b["attempts"]}, {"verifier_error"})
        self.assertGreaterEqual(len(self.events("verify_error")), 2 + 6)


class CrashTests(BatchTestCase):
    def test_crash_burst_backs_off_without_consuming_seeds(self) -> None:
        items = [f"s{i}" for i in range(8)]
        script = {item: [("crash", 30.0, "RuntimeError: bounded upload attempts exhausted"),
                         ("pass", 60.0)] for item in items[:6]}
        world = World(script)
        supervisor = self.make(world, items, policy(
            max_concurrency=8, crash_burst_items=5, crash_burst_window_s=600,
            burst_backoff_s=120, burst_backoff_cap_s=960))
        self.assertEqual(supervisor.run(), COMPLETE)
        bursts = self.events("crash_burst_backoff")
        self.assertEqual(len(bursts), 1)
        start, backoff = bursts[0]["t"], bursts[0]["backoff_s"]
        self.assertEqual(backoff, 120)
        during = [e for e in self.events("launch_intent") if start <= e["t"] < start + backoff]
        self.assertEqual(during, [], "nothing may launch during the backoff")
        for item in items[:6]:
            record = supervisor.item(item)
            self.assertEqual((record["attempts_used"], record["crash_retries"],
                              record["admitted_attempt"]), (1, 1, 1))
            launches = world.launches_for(item)
            self.assertEqual([row["attempt"] for row in launches], [1, 1])
            self.assertEqual(len({row["seed"] for row in launches}), 1, "crash kept the seed")
        self.assertEqual(sum(supervisor.item(i)["attempts_used"] for i in items), 8)
        self.assertEqual(supervisor.summary()["rounds"]["closed"], 1)

    def test_repeated_bursts_escalate_and_self_resume(self) -> None:
        items = [f"s{i}" for i in range(6)]
        script = {item: [("crash", 10.0), ("crash", 10.0), ("pass", 20.0)] for item in items}
        world = World(script)
        supervisor = self.make(world, items, policy(
            max_concurrency=6, crash_burst_items=3, crash_burst_window_s=600,
            burst_backoff_s=60, burst_backoff_cap_s=1000, max_crash_retries_per_item=5))
        self.assertEqual(supervisor.run(), COMPLETE)
        bursts = self.events("crash_burst_backoff")
        self.assertEqual([(b["level"], b["backoff_s"]) for b in bursts], [(1, 60), (2, 120)])
        self.assertTrue(all(supervisor.item(i)["attempts_used"] == 1 for i in items))

    def test_single_item_crash_loop_uses_crash_budget_not_global_backoff(self) -> None:
        world = World({"a": [("crash", 5.0, "stale verdict returned")] * 20})
        supervisor = self.make(world, ["a", "b", "c"], policy(
            max_crash_retries_per_item=3, crash_burst_items=2, crash_retry_delay_s=10.0))
        self.assertEqual(supervisor.run(), COMPLETE_WITH_GAPS)
        record = supervisor.item("a")
        self.assertEqual((record["state"], record["terminal_reason"]), (EXHAUSTED, "crash_budget"))
        self.assertEqual((record["crash_retries"], record["attempts_used"]), (4, 0))
        launches = world.launches_for("a")
        self.assertEqual(len(launches), 4)
        self.assertEqual({(row["attempt"], row["seed"]) for row in launches},
                         {(1, record["seed"])})
        self.assertEqual(self.events("crash_burst_backoff"), [])
        gaps = supervisor.summary()["gaps"]
        self.assertEqual([gap["item"] for gap in gaps], ["a"])

    def test_very_long_crash_loop_does_not_overflow_the_backoff(self) -> None:
        world = World({"a": [("crash", 1.0)] * 1200 + [("pass", 10.0)]})
        supervisor = self.make(world, ["a"], policy(max_crash_retries_per_item=5000,
                                                    crash_retry_delay_s=1.0,
                                                    crash_retry_delay_cap_s=2.0))
        self.assertEqual(supervisor.run(), COMPLETE)
        self.assertEqual(supervisor.item("a")["crash_retries"], 1200)

    def test_launch_failures_are_budgeted_and_deferrals_are_free(self) -> None:
        world = World()
        world.launch_errors = {
            "a": [LaunchNotStarted("upload refused"), None],
            "b": [LaunchDeferred("no free GPU"), LaunchDeferred("no free GPU"), None],
            "c": [RuntimeError("submit exploded")] * 10,
        }
        supervisor = self.make(world, ["a", "b", "c"], policy(max_crash_retries_per_item=2),
                               task=world.task(reconcile=world.reconcile))
        self.assertEqual(supervisor.run(), COMPLETE_WITH_GAPS)
        self.assertEqual(supervisor.item("a")["crash_retries"], 1)
        self.assertEqual(supervisor.item("b")["crash_retries"], 0)
        self.assertEqual(supervisor.item("b")["state"], ADMITTED)
        c = supervisor.item("c")
        self.assertEqual((c["state"], c["terminal_reason"], c["crash_retries"]),
                         (EXHAUSTED, "crash_budget", 3))
        self.assertEqual(world.launches_for("c"), [])


class LivenessTests(BatchTestCase):
    def test_lease_query_failure_is_unknown_and_never_resubmitted(self) -> None:
        world = World(default=("pass", 100.0))
        outage_end = world.clock() + 300.0

        def status_file_unreadable(job: dict) -> Any:
            if job["item"] == "a" and world.clock() < outage_end:
                return Poll("unknown", detail="no exit record; worker host unreachable")
            return None

        def failing_lease_query(job: dict) -> Any:
            raise RuntimeError("gpu-scheduler leases exited 1")

        world.poll_hook = status_file_unreadable
        world.lease_hook = failing_lease_query
        supervisor = self.make(world, ["a", "b"], policy(max_consecutive_unknown=100),
                               task=world.task(lease_probe=world.lease_probe))
        supervisor.run(max_ticks=15)
        self.assertEqual(supervisor.item("a")["state"], UNKNOWN)
        self.assertEqual(supervisor.item("b")["state"], ADMITTED)
        self.assertEqual(supervisor.run(), COMPLETE)
        self.assertEqual(len(world.launches_for("a")), 1, "unknown must never be resubmitted")
        self.assertEqual(self.events("lost"), [])
        self.assertTrue(self.events("lease_probe_error"))
        self.assertTrue([e for e in self.events("unknown") if e["item"] == "a"])

    def test_unknown_liveness_is_bounded_then_resumes_after_acknowledgement(self) -> None:
        world = World(default=("pass", 50.0))
        blackout = {"on": True}
        world.poll_hook = lambda job: Poll("unknown", detail="status query failed") \
            if blackout["on"] else None
        world.lease_hook = lambda job: "unknown"
        task = world.task(lease_probe=world.lease_probe)
        supervisor = self.make(world, ["a"], policy(max_consecutive_unknown=5), task=task)
        self.assertEqual(supervisor.run(), PAUSED_UNKNOWN_LIVENESS)
        self.assertEqual(supervisor.item("a")["state"], UNKNOWN)
        self.assertEqual(len(world.launches), 1)
        supervisor.close()
        blackout["on"] = False
        resumed = self.make(world, ["a"], policy(max_consecutive_unknown=5), task=task)
        self.assertEqual(resumed.run(), PAUSED_UNKNOWN_LIVENESS, "a pause needs acknowledgement")
        resumed.acknowledge("scheduler reachable again")
        self.assertEqual(resumed.run(), COMPLETE)
        self.assertEqual(len(world.launches), 1)

    def test_dead_lease_needs_confirmation_and_positive_life_wins(self) -> None:
        world = World(default=("pass", 1000.0))
        world.poll_hook = lambda job: Poll("unknown", detail="no exit record") \
            if job["item"] == "a" and job["id"] == 1 else None
        world.lease_hook = lambda job: "dead"
        supervisor = self.make(world, ["a", "b"], policy(lost_confirmations=2),
                               task=world.task(lease_probe=world.lease_probe))
        self.assertEqual(supervisor.run(), COMPLETE)
        a_launches = world.launches_for("a")
        self.assertEqual(len(a_launches), 2)
        self.assertEqual(a_launches[0]["seed"], a_launches[1]["seed"])
        record = supervisor.item("a")
        self.assertTrue(record["crash_log"][0]["detail"].startswith("lost:"))
        self.assertEqual(len(world.launches_for("b")), 1, "running + dead lease is not lost")
        lost = self.events("lost")
        self.assertEqual([e["item"] for e in lost], ["a"])
        self.assertEqual(lost[0]["confirmations"], 2)


class StopRuleTests(BatchTestCase):
    def test_zero_progress_stops_after_n_rounds(self) -> None:
        world = World(default=("reject", 50.0, "topology"))
        supervisor = self.make(world, ["a", "b", "c"], policy(
            zero_progress_rounds=3, max_attempts_per_item=10, max_rounds=10))
        self.assertEqual(supervisor.run(), STOPPED_ZERO_PROGRESS)
        summary = supervisor.summary()
        self.assertEqual(summary["rounds"]["closed"], 3)
        self.assertEqual(len(world.launches), 9)
        self.assertEqual({supervisor.item(i)["state"] for i in "abc"}, {REJECTED_RETRYABLE})
        self.assertEqual(summary["terminal"]["report"]["open_items"], ["a", "b", "c"])

    def test_zero_progress_streak_resets_on_admission(self) -> None:
        script = {"a": [("reject", 10.0, "seam"), ("pass", 10.0)]}
        world = World(script, default=("reject", 10.0, "seam"))
        supervisor = self.make(world, ["a", "b", "c", "d"], policy(
            zero_progress_rounds=2, max_attempts_per_item=10, max_rounds=10))
        self.assertEqual(supervisor.run(), STOPPED_ZERO_PROGRESS)
        streaks = [e["zero_progress_streak"] for e in self.events("round_closed")]
        self.assertEqual(streaks, [1, 0, 1, 2])

    def test_cost_cap_stops_launching_and_drains(self) -> None:
        script = {f"s{i}": [("reject", 30.0 + 7 * i, "seam")] * 5 for i in range(6)}
        world = World(script)
        supervisor = self.make(world, list(script), policy(
            max_concurrency=3, max_total_cost=5.0, max_attempts_per_item=5, max_rounds=5))
        self.assertEqual(supervisor.run(), STOPPED_COST_CAP)
        stop = self.events("stop_requested")[0]
        self.assertEqual(self.launches_after(stop["t"]), [])
        summary = supervisor.summary()
        self.assertGreaterEqual(summary["counters"]["cost"], 5.0)
        self.assertEqual(summary["in_flight"], [])
        self.assertTrue(self.events("draining") or summary["counters"]["cost"] >= 5.0)

    def test_cost_reservation_prevents_overshoot(self) -> None:
        world = World(default=("reject", 30.0, "seam"))
        supervisor = self.make(world, [f"s{i}" for i in range(10)], policy(
            max_concurrency=10, max_total_cost=3.0, launch_cost_estimate=1.0))
        self.assertEqual(supervisor.run(), STOPPED_COST_CAP)
        self.assertEqual(len(world.launches), 3)

    def test_invalid_reported_cost_pauses_without_verdict_or_new_launch(self) -> None:
        world = World(default=("pass", 10.0))
        report = {"cost": float("nan")}
        world.poll_hook = lambda job: (Poll("finished", result=job, cost=report["cost"])
                                       if world.clock() >= job["end"] else None)
        supervisor = self.make(world, ["a", "b"], policy(max_concurrency=1,
                                                          max_total_cost=1.0))
        self.assertEqual(supervisor.run(), PAUSED_COST_UNKNOWN)
        self.assertEqual(len(world.launches), 1)
        self.assertEqual(supervisor.item("a")["attempts_used"], 0)
        self.assertEqual(supervisor.summary()["counters"]["cost"], 0.0)
        report["cost"] = 1.0
        supervisor.acknowledge("fixed provider cost evidence")
        self.assertEqual(supervisor.run(), STOPPED_COST_CAP)
        self.assertEqual(supervisor.item("a")["state"], ADMITTED)
        self.assertEqual(len(world.launches), 1)

    def test_item_exhaustion_is_complete_with_gaps(self) -> None:
        world = World({"b": [("reject", 30.0, "integrity")] * 10})
        supervisor = self.make(world, ["a", "b", "c"], policy(max_attempts_per_item=3))
        self.assertEqual(supervisor.run(), COMPLETE_WITH_GAPS)
        b = supervisor.item("b")
        self.assertEqual((b["state"], b["terminal_reason"], b["attempts_used"]),
                         (EXHAUSTED, "attempt_budget", 3))
        self.assertEqual(len({row["seed"] for row in world.launches_for("b")}), 3)
        summary = supervisor.summary()
        self.assertEqual((summary["denominator"], summary["admitted"]), (3, 2))
        self.assertEqual([gap["item"] for gap in summary["gaps"]], ["b"])

    def test_max_rounds_is_a_hard_ceiling(self) -> None:
        world = World({"b": [("reject", 30.0, "integrity")] * 10})
        supervisor = self.make(world, ["a", "b"], policy(max_attempts_per_item=10, max_rounds=2))
        self.assertEqual(supervisor.run(), COMPLETE_WITH_GAPS)
        b = supervisor.item("b")
        self.assertEqual((b["terminal_reason"], b["attempts_used"]), ("max_rounds", 2))

    def test_systematic_failure_stops_new_launches_early(self) -> None:
        items = [f"s{i:02d}" for i in range(40)]
        script = {item: [("reject", 60.0 + i, "topology")] for i, item in enumerate(items[:20])}
        world = World(script)
        supervisor = self.make(world, items, policy(
            max_concurrency=5, systematic_min_items=8, systematic_window=20,
            systematic_share=0.6))
        self.assertEqual(supervisor.run(), STOPPED_SYSTEMATIC_FAILURE)
        report = supervisor.summary()["terminal"]["report"]
        self.assertEqual(report["gate"], "topology")
        self.assertGreaterEqual(report["share"], 0.6)
        self.assertGreaterEqual(report["window"], 8)
        stop = self.events("stop_requested")[0]
        self.assertEqual(self.launches_after(stop["t"]), [])
        pending = [i for i in items if supervisor.item(i)["state"] == PENDING]
        self.assertGreaterEqual(len(pending), 25, "most of the batch was never burned")

    def test_systematic_check_needs_enough_distinct_items(self) -> None:
        world = World(default=("reject", 30.0, "topology"))
        supervisor = self.make(world, [f"s{i}" for i in range(5)], policy(
            systematic_min_items=8, max_attempts_per_item=3))
        self.assertEqual(supervisor.run(), COMPLETE_WITH_GAPS)
        self.assertEqual(self.events("stop_requested"), [])

    def test_operator_stop_drains_then_acknowledge_resumes(self) -> None:
        world = World(default=("pass", 100.0))
        items = [f"s{i}" for i in range(6)]
        requested = {"on": False}
        supervisor = self.make(world, items, policy(max_concurrency=2),
                               stop_check=lambda: "operator asked" if requested["on"] else None)
        supervisor.run(max_ticks=3)
        requested["on"] = True
        self.assertEqual(supervisor.run(), STOPPED_OPERATOR)
        self.assertEqual(len(world.launches), 2)
        self.assertEqual(supervisor.summary()["counts"][PENDING], 4)
        requested["on"] = False
        supervisor.acknowledge("resume")
        self.assertEqual(supervisor.run(), COMPLETE)
        self.assertEqual(len(world.launches), 6)

    def test_wall_budget_stops(self) -> None:
        world = World(default=("reject", 600.0, "seam"))
        supervisor = self.make(world, ["a"], policy(max_wall_seconds=1000.0,
                                                    max_attempts_per_item=10, max_rounds=10))
        self.assertEqual(supervisor.run(), STOPPED_WALL_BUDGET)
        self.assertLess(supervisor.summary()["counters"]["active_seconds"], 1300.0)


class ProviderTests(BatchTestCase):
    def test_provider_down_pause_is_not_charged_to_the_wall_budget(self) -> None:
        world = World()
        world.provider_down_until = world.clock() + 7200.0
        world.step_hook = lambda item: ("provider_down", 5.0) \
            if world.clock() < world.provider_down_until else ("pass", 60.0)
        items = ["a", "b", "c", "d"]
        supervisor = self.make(world, items, policy(
            max_wall_seconds=1800.0, provider_probe_interval_s=300.0,
            max_provider_pause_s=4 * 3600.0), task=world.task(provider_probe=world.provider_probe))
        self.assertEqual(supervisor.run(), COMPLETE)
        counters = supervisor.summary()["counters"]
        self.assertGreater(counters["provider_paused_seconds"], 7000.0)
        self.assertLess(counters["active_seconds"], 1800.0)
        # Due-but-blocked retries must not turn the loop into a busy poll.
        self.assertLess(counters["ticks"], 7400 / 10 + 50)
        for item in items:
            record = supervisor.item(item)
            self.assertEqual((record["crash_retries"], record["attempts_used"],
                              record["provider_crashes"]), (0, 1, 1))
        down, up = self.events("provider_down"), self.events("provider_up")
        self.assertEqual((len(down), len(up)), (1, 1))
        between = [e for e in self.events("launch_intent") if down[0]["t"] < e["t"] < up[0]["t"]]
        self.assertEqual(between, [])

    def test_provider_pause_is_bounded(self) -> None:
        world = World(default=("provider_down", 5.0))
        world.provider_down_until = float("inf")
        supervisor = self.make(world, ["a", "b", "c", "d"], policy(
            provider_probe_interval_s=300.0, max_provider_pause_s=3600.0),
            task=world.task(provider_probe=world.provider_probe))
        self.assertEqual(supervisor.run(), PAUSED_PROVIDER_DOWN)
        self.assertEqual(len(world.launches), 4)
        self.assertTrue(all(supervisor.item(i)["state"] == CRASHED_RETRYABLE for i in "abcd"))

    def test_half_open_probe_without_a_probe_callable(self) -> None:
        world = World()
        world.provider_down_until = world.clock() + 600.0
        world.step_hook = lambda item: ("provider_down", 5.0) \
            if world.clock() < world.provider_down_until else ("pass", 60.0)
        items = [f"s{i}" for i in range(5)]
        supervisor = self.make(world, items, policy(provider_probe_interval_s=300.0,
                                                    max_concurrency=5))
        self.assertEqual(supervisor.run(), COMPLETE)
        down = self.events("provider_down")[0]
        up = self.events("provider_up")[0]
        probes = [e for e in self.events("launch_intent") if down["t"] < e["t"] < up["t"]]
        self.assertTrue(probes)
        self.assertTrue(all(e.get("half_open") for e in probes))
        gaps = [b["t"] - a["t"] for a, b in zip(probes, probes[1:])]
        self.assertTrue(all(gap >= 300.0 for gap in gaps))

    def test_launch_raising_provider_down_pauses(self) -> None:
        world = World()
        world.launch_errors = {"a": [ProviderDown("endpoint 503"), None]}
        world.provider_down_until = world.clock() + 200.0
        supervisor = self.make(world, ["a", "b"], policy(provider_probe_interval_s=100.0,
                                                         max_concurrency=1),
                               task=world.task(provider_probe=world.provider_probe))
        self.assertEqual(supervisor.run(), COMPLETE)
        self.assertEqual(supervisor.item("a")["crash_retries"], 0)
        self.assertEqual(len(self.events("provider_down")), 1)


class ResumeTests(BatchTestCase):
    def test_supervisor_death_between_intent_and_handle_does_not_double_launch(self) -> None:
        world = World(default=("pass", 100.0))

        def launch_then_die(item: str, attempt: int, seed: int, launch_id: str | None = None):
            handle = world.launch(item, attempt, seed, launch_id)
            if item == "b":
                raise SupervisorDied()
            return handle

        first = self.make(world, ["a", "b", "c"], policy(max_concurrency=3),
                          task=world.task(launch=launch_then_die, reconcile=world.reconcile))
        with self.assertRaises(SupervisorDied):
            first.run()
        first.close()
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertEqual(state["item_state"]["b"]["state"], LAUNCHING)
        self.assertIsNotNone(state["item_state"]["b"]["launch_id"])
        second = self.make(world, ["a", "b", "c"], policy(max_concurrency=3),
                           task=world.task(reconcile=world.reconcile))
        self.assertEqual(second.run(), COMPLETE)
        self.assertEqual(len(world.launches_for("b")), 1)
        self.assertEqual(len(world.launches), 3)
        self.assertEqual([e["item"] for e in self.events("reconcile_found")], ["b"])
        self.assertEqual(len(self.events("resumed")), 1)

    def test_intent_without_reconcile_stays_unknown_and_is_never_relaunched(self) -> None:
        world = World(default=("pass", 100.0))

        def launch_then_die(item: str, attempt: int, seed: int, launch_id: str | None = None):
            world.launch(item, attempt, seed, launch_id)
            raise SupervisorDied()

        first = self.make(world, ["a"], policy(), task=world.task(launch=launch_then_die))
        with self.assertRaises(SupervisorDied):
            first.run()
        first.close()
        second = self.make(world, ["a"], policy(max_consecutive_unknown=4))
        self.assertEqual(second.run(), PAUSED_UNKNOWN_LIVENESS)
        self.assertEqual(len(world.launches), 1)
        second.acknowledge("checked by hand: the job did finish, but nothing records it")
        second.resolve_unknown("a", outcome="crashed", reason="no handle to poll")
        self.assertEqual(second.run(), COMPLETE)
        self.assertEqual(second.item("a")["crash_retries"], 1)

    def test_reconcile_not_launched_relaunches_same_attempt_without_budget(self) -> None:
        world = World(default=("pass", 100.0))
        died = {"once": False}

        def die_before_provider(item: str, attempt: int, seed: int, launch_id: str | None = None):
            if item == "b" and not died["once"]:
                died["once"] = True
                raise SupervisorDied()
            return world.launch(item, attempt, seed, launch_id)

        first = self.make(world, ["a", "b"], policy(),
                          task=world.task(launch=die_before_provider, reconcile=world.reconcile))
        with self.assertRaises(SupervisorDied):
            first.run()
        first.close()
        intent = json.loads((self.state_dir / "state.json").read_text())["item_state"]["b"]
        second = self.make(world, ["a", "b"], policy(),
                           task=world.task(reconcile=world.reconcile))
        self.assertEqual(second.run(), COMPLETE)
        launches = world.launches_for("b")
        self.assertEqual(len(launches), 1)
        self.assertEqual((launches[0]["attempt"], launches[0]["seed"]),
                         (intent["attempt"], intent["seed"]))
        self.assertEqual(second.item("b")["crash_retries"], 0)
        self.assertEqual(len(self.events("reconcile_not_launched")), 1)

    def test_resume_polls_persisted_handles(self) -> None:
        world = World(default=("pass", 300.0))
        items = [f"s{i}" for i in range(4)]
        first = self.make(world, items, policy(max_concurrency=4))
        first.run(max_ticks=5)
        first.close()
        world.clock.sleep(1000.0)  # supervisor down for a while
        second = self.make(world, items, policy(max_concurrency=4))
        self.assertEqual(second.run(), COMPLETE)
        self.assertEqual(len(world.launches), 4)
        self.assertLess(second.summary()["counters"]["active_seconds"], 900.0)

    def test_denominator_is_fixed_for_the_life_of_the_batch(self) -> None:
        world = World({"b": [("reject", 10.0, "seam")] * 3})
        first = self.make(world, ["a", "b", "c"], policy(max_attempts_per_item=2))
        self.assertEqual(first.run(), COMPLETE_WITH_GAPS)
        first.close()
        for changed in (["a", "c"], ["a", "b", "c", "d"], ["c", "b", "a"]):
            with self.assertRaises(DenominatorMismatch):
                BatchSupervisor(self.state_dir, changed, world.task(), policy(),
                                clock=world.clock, fsync=False)
        again = self.make(world, ["a", "b", "c"], policy(max_attempts_per_item=2))
        summary = again.summary()
        self.assertEqual((summary["denominator"], summary["admitted"]), (3, 2))
        self.assertEqual(sum(summary["counts"].values()), 3)

    def test_second_supervisor_is_locked_out(self) -> None:
        world = World()
        self.make(world, ["a"])
        with self.assertRaises(SupervisorBusy):
            BatchSupervisor(self.state_dir, ["a"], world.task(), policy(), clock=world.clock)

    def test_state_is_atomic_and_events_append_only(self) -> None:
        world = World({"b": [("crash", 5.0), ("reject", 10.0, "seam"), ("pass", 10.0)]})
        first = self.make(world, ["a", "b"], policy(), fsync=True)
        first.run(max_ticks=2)
        first.close()
        before = (self.state_dir / "events.jsonl").read_text().splitlines()
        second = self.make(world, ["a", "b"], policy(), fsync=True)
        self.assertEqual(second.run(), COMPLETE)
        after = (self.state_dir / "events.jsonl").read_text().splitlines()
        self.assertEqual(after[: len(before)], before)
        events = [json.loads(line) for line in after]
        self.assertEqual(events[0]["kind"], "created")
        self.assertEqual(events[-1]["kind"], "terminal")
        self.assertEqual(list(self.state_dir.glob("*.tmp")), [])
        self.assertEqual(list(self.state_dir.glob(".*.tmp")), [])
        state = json.loads((self.state_dir / "state.json").read_text())
        self.assertEqual(state["schema"], "datafarm.batch-supervisor-state/1.0.0")
        self.assertEqual(state["status"], COMPLETE)

    def test_policy_change_on_resume_is_recorded(self) -> None:
        world = World()
        first = self.make(world, ["a"], policy(max_attempts_per_item=3))
        first.close()
        self.make(world, ["a"], policy(max_attempts_per_item=5))
        changes = self.events("policy_changed")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["new_policy"]["max_attempts_per_item"], 5)


class OperatorActionTests(BatchTestCase):
    def test_reopen_and_revoke(self) -> None:
        world = World({"b": [("reject", 10.0, "seam"), ("reject", 10.0, "seam"),
                             ("pass", 10.0)],
                       "a": [("pass", 10.0), ("pass", 10.0)]})
        supervisor = self.make(world, ["a", "b"], policy(max_attempts_per_item=2))
        self.assertEqual(supervisor.run(), COMPLETE_WITH_GAPS)
        supervisor.reopen(["b"], reason="gate fixed", extra_attempts=1)
        self.assertEqual(supervisor.run(), COMPLETE)
        self.assertEqual(supervisor.item("b")["admitted_attempt"], 3)
        supervisor.revoke_admission("a", reason="cohort audit: motion structure")
        self.assertEqual(supervisor.item("a")["state"], REJECTED_RETRYABLE)
        self.assertEqual(supervisor.run(), COMPLETE)
        a = supervisor.item("a")
        self.assertEqual((a["attempts_used"], a["admitted_attempt"]), (2, 2))
        with self.assertRaises(BatchError):
            supervisor.reopen(["a"], reason="admitted items cannot be reopened")

class LedgerTests(BatchTestCase):
    def make_ledger(self):
        from datafarm.ledger.ledger import RunLedger, plan_stages

        ledger = RunLedger.create(
            self.tmp / "ledger", run_id="PRUN-20260926-1200-batch-fixture", title="fixture",
            request="render three items", task_family="fixture",
            stages=plan_stages("render", "admission", {"render": ["generate"]}),
            items=["a", "b", "c"])
        worker = ledger.register_worker(role="batch-supervisor", pid=os.getpid())
        return ledger, worker

    def test_cells_are_mirrored_through_the_public_api(self) -> None:
        ledger, worker = self.make_ledger()
        world = World({"b": [("reject", 10.0, "seam"), ("pass", 10.0)],
                       "c": [("reject", 10.0, "integrity")] * 3})
        supervisor = self.make(world, ["a", "b", "c"], policy(max_attempts_per_item=2),
                               ledger=LedgerHook(ledger, worker, "render/generate"))
        self.assertEqual(supervisor.run(), COMPLETE_WITH_GAPS)
        cells = ledger.matrix()["cells"]
        self.assertEqual(cells["a"]["render/generate"]["status"], "passed")
        self.assertEqual(cells["b"]["render/generate"]["status"], "passed")
        self.assertEqual(cells["b"]["render/generate"]["attempt"], 2)
        self.assertEqual(cells["c"]["render/generate"]["status"], "failed")
        self.assertIn("attempt_budget", cells["c"]["render/generate"]["failure"])
        self.assertEqual(cells["a"]["admission"]["status"], "pending", "other columns untouched")
        sources = {e.get("source") for e in ledger.events() if e["kind"] == "cells"}
        self.assertEqual(sources, {f"batch-supervisor:{worker}"})

    def test_attach_from_env_and_blocked_declaration(self) -> None:
        ledger, worker = self.make_ledger()
        world = World(default=("reject", 10.0, "topology"))
        env = {"DATAFARM_LEDGER_RUN_DIR": str(ledger.directory), "DATAFARM_LEDGER_WORKER": worker}
        with mock.patch.dict(os.environ, env):
            supervisor = self.make(world, ["a", "b", "c"], policy(
                zero_progress_rounds=2, max_attempts_per_item=5, max_rounds=5),
                ledger="env", ledger_column="render/generate")
            self.assertEqual(supervisor.run(), STOPPED_ZERO_PROGRESS)
        cells = ledger.matrix()["cells"]
        self.assertEqual({cells[i]["render/generate"]["status"] for i in "abc"}, {"rejected"})
        declared = [e for e in ledger.events() if e["kind"] == "declared"]
        self.assertEqual(declared[-1]["status"], "blocked")
        self.assertIn(STOPPED_ZERO_PROGRESS, declared[-1]["reason"])

    def test_ledger_errors_never_stop_the_batch(self) -> None:
        ledger, worker = self.make_ledger()
        world = World()
        supervisor = self.make(world, ["a", "b", "c"],
                               ledger=LedgerHook(ledger, worker, "render/no-such-column"))
        self.assertEqual(supervisor.run(), COMPLETE)
        self.assertTrue(self.events("ledger_error"))

    def test_env_without_column_is_recorded_not_guessed(self) -> None:
        ledger, worker = self.make_ledger()
        env = {"DATAFARM_LEDGER_RUN_DIR": str(ledger.directory), "DATAFARM_LEDGER_WORKER": worker}
        world = World()
        with mock.patch.dict(os.environ, env):
            supervisor = self.make(world, ["a", "b", "c"], ledger="env")
            self.assertEqual(supervisor.run(), COMPLETE)
        self.assertTrue(self.events("ledger_not_mirrored"))
        self.assertEqual(ledger.matrix()["cells"]["a"]["render/generate"]["status"], "pending")


class CrashBurstTests(unittest.TestCase):
    def test_burst_rule_counts_distinct_items_and_escalates(self) -> None:
        from datafarm.batch.supervisor import advance_burst, new_burst_state

        policy = BatchPolicy(crash_burst_items=3, crash_burst_window_s=600, burst_backoff_s=60,
                             burst_backoff_cap_s=100)
        burst = new_burst_state()
        self.assertIsNone(advance_burst(burst, 0, "a", policy))
        self.assertIsNone(advance_burst(burst, 10, "a", policy), "same item twice is no burst")
        self.assertIsNone(advance_burst(burst, 20, "b", policy))
        first = advance_burst(burst, 30, "c", policy)
        self.assertEqual((first["level"], first["backoff_s"]), (1, 60.0))
        for t, item in ((100, "d"), (110, "e")):
            self.assertIsNone(advance_burst(burst, t, item, policy))
        second = advance_burst(burst, 120, "f", policy)
        self.assertEqual((second["level"], second["backoff_s"]), (2, 100.0), "capped doubling")



if __name__ == "__main__":
    unittest.main()
