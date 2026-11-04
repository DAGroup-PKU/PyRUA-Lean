"""Expert harness rounds: prepare, the session loop and its stop conditions, outages,
tamper checks, crash recovery and spend guards. Scripted agent (real child process, no
model), fake rollout backend, fake clock for budgets and backoff."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from datafarm.agent.call_log import read_calls

from datafarm.expert.adapter import TaskAdapter
from datafarm.expert.layout import RoundPaths
from datafarm.expert.rollout import FakeRolloutBackend
from datafarm.expert.round import EXIT_PAUSED, RoundRunner, load_round_adapter, prepare_round, round_status
from datafarm.expert.session import make_agent_backend
from datafarm.expert.state import RoundState
from xh_helpers import FakeClock
from datafarm.expert.util import HarnessError, read_jsonl
from xh_helpers import HOLDOUT, expert, write_adapter, write_plan

WRITE = {"op": "write", "path": "expert.py", "text": expert(0.99)}


class RoundFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xh-round-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clock = FakeClock()

    def prepare(self, sessions, overrides=None, cost: float = 0.25) -> tuple[TaskAdapter, RoundPaths]:
        adapter = write_adapter(self.tmp / "task", overrides)
        write_plan(self.tmp / "task", sessions, cost=cost)
        paths = RoundPaths(self.tmp / "round")
        prepare_round(adapter, paths.root, clock=self.clock, round_id="XR-toy-test")
        return adapter, paths

    def runner(self, adapter, paths, **kwargs) -> RoundRunner:
        defaults = dict(clock=self.clock, rollout_backend=FakeRolloutBackend(), tick_interval_s=0.05,
                        poll_interval_s=0, term_grace_s=1.0, drain_timeout_s=5.0, flag_grace_s=0.0)
        defaults.update(kwargs)
        return RoundRunner(paths, adapter, **defaults)


class PrepareTests(RoundFixture):
    def test_prepare_writes_public_and_private_parts(self) -> None:
        adapter, paths = self.prepare([[WRITE]])
        round_json = json.loads(paths.round_json.read_text())
        self.assertEqual(round_json["holdout"], {"count": 4, "policy": "hidden; chosen and run only by the harness"})
        self.assertNotRegex(paths.round_json.read_text(), rf"\b{HOLDOUT[0]}\b")
        self.assertNotRegex(paths.brief.read_text(), rf"\b{HOLDOUT[0]}\b")
        self.assertEqual(oct(paths.harness.stat().st_mode & 0o777), "0o700")
        self.assertEqual(oct(paths.holdout_seeds.stat().st_mode & 0o777), "0o600")
        self.assertTrue(paths.tool.exists())
        self.assertTrue((paths.agent_home / ".codex").is_dir())
        self.assertEqual(RoundState.load(paths.state).status, "prepared")
        self.assertEqual(load_round_adapter(paths).sha256, adapter.sha256)
        with self.assertRaises(HarnessError):
            prepare_round(adapter, paths.root, clock=self.clock)

    def test_prepare_fails_closed(self) -> None:
        adapter = write_adapter(self.tmp / "task")
        with self.assertRaises(HarnessError):  # round dir inside a protected path
            prepare_round(adapter, self.tmp / "task" / "protected" / "round", clock=self.clock)
        (self.tmp / "task" / "leaky.json").write_text(json.dumps(
            {"schema": "datafarm.expert-harness.holdout-seeds/1", "seeds": [{"seed": 7}]}))
        with self.assertRaises(HarnessError):  # holdout seed is a development seed
            prepare_round(adapter, self.tmp / "r2", clock=self.clock,
                          holdout_seeds_path=str(self.tmp / "task" / "leaky.json"))
        self.assertFalse((self.tmp / "r2").exists())


class LoopTests(RoundFixture):
    def test_operator_diagnostic_is_snapshotted_and_audited(self) -> None:
        adapter, paths = self.prepare([[WRITE, {"op": "xh", "args": ["submit", "--final", "done"]}]])
        note = "Measured geometry suggests revisiting the target orientation; no success claim."
        (paths.harness / 'operator-diagnostic.txt').write_text(note)
        runner = self.runner(adapter, paths)
        original = runner.agent_backend.command
        delivered = []
        def command(spec):
            delivered.append(spec.prompt)
            return original(spec)
        runner.agent_backend.command = command
        self.assertEqual(runner.run()['status'], 'completed')
        self.assertIn(note, delivered[0])
        saved = paths.sessions / 'session-001.operator-diagnostic.txt'
        self.assertEqual(saved.read_text(), note)
        self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        events = [e for e in read_jsonl(paths.events) if e['event'] == 'operator_diagnostic']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['bytes'], len(note.encode()))

    def test_codex_cannot_claim_an_unenforceable_usd_cap(self) -> None:
        adapter, paths = self.prepare([[WRITE]])
        codex = make_agent_backend({"backend": "codex", "model": "codex-test-model"}, {})
        runner = self.runner(adapter, paths, agent_backend=codex, approve_agent_spend=True)
        with self.assertRaisesRegex(HarnessError, "no per-session USD hard cap"):
            runner.run()
        self.assertEqual(RoundState.load(paths.state).status, "prepared")
        self.assertFalse(paths.agent_calls.exists())

    def test_complete_round_with_resumed_sessions(self) -> None:
        adapter, paths = self.prepare([
            [WRITE, {"op": "xh", "args": ["rollout", "--seed", "3"]},
             {"op": "xh", "args": ["wait", "A-0001", "--timeout", "20"]},
             {"op": "xh", "args": ["submit", "first"]}],
            [{"op": "xh", "args": ["submit", "--final", "done"]}],
        ])
        summary = self.runner(adapter, paths).run()
        self.assertEqual(summary["status"], "completed")
        state = RoundState.load(paths.state).data
        self.assertEqual([s["resumed"] for s in state["sessions"]], [False, True])
        self.assertEqual(state["sessions"][0]["session_id"], state["sessions"][1]["session_id"])
        calls = read_calls(paths.agent_calls)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c.component == "expert-harness/session" and c.status == "ok" and c.cost_usd == 0.25
                            and c.cost_source == "provider" and c.run_id == "XR-toy-test" for c in calls))
        self.assertEqual(state["cost_usd_known"], 0.5)
        self.assertEqual(state["counters"]["counted"], 1)
        self.assertTrue(state["manifest"]["ok"])
        self.assertGreaterEqual(state["manifest"]["checks"], 3)
        events = [e["event"] for e in read_jsonl(paths.events)]
        for name in ("round_started", "isolation_probe", "session_started", "session_ended", "submission",
                     "round_finished"):
            self.assertIn(name, events)
        status = round_status(paths)
        self.assertEqual((status["final_submission"], status["agent_calls"]["calls"]), ("S-02", 2))
        self.assertEqual(self.runner(adapter, paths).run()["status"], "completed")  # idempotent

    def test_flag_for_human_ends_the_round(self) -> None:
        adapter, paths = self.prepare([[{"op": "xh", "args": ["flag-for-human", "physics cannot lift"]}],
                                       [WRITE]])
        summary = self.runner(adapter, paths).run()
        self.assertEqual(summary["status"], "flagged_for_human")
        self.assertEqual(summary["flag"]["reason"], "physics cannot lift")
        self.assertEqual(summary["sessions"], 1)
        self.assertEqual(summary["submissions"], [])  # no auto-final for a flagged round

    def test_api_outage_backs_off_then_pauses_and_resumes(self) -> None:
        adapter, paths = self.prepare([[{"op": "api_error", "times": 5}, WRITE,
                                        {"op": "xh", "args": ["submit", "--final", "done"]}]])
        first = self.runner(adapter, paths).run()
        self.assertEqual(first["exit_code"], EXIT_PAUSED)
        state = RoundState.load(paths.state).data
        self.assertEqual(state["status"], "paused_model_down")
        self.assertEqual(len(state["sessions"]), 5)
        self.assertTrue(all(s["api_outage"] for s in state["sessions"]))
        backoffs = [e["sleep_s"] for e in read_jsonl(paths.events) if e["event"] == "api_outage_backoff"]
        self.assertEqual(backoffs, [60.0, 120.0, 240.0, 300.0])
        self.clock.advance(3600)  # provider down for an hour: not charged
        second = self.runner(adapter, paths).run()
        self.assertEqual(second["status"], "completed")
        self.assertLess(second["active_wall_s"], 60)
        calls = read_calls(paths.agent_calls)
        self.assertEqual([c.error_type for c in calls].count("api_outage"), 5)
        self.assertEqual(calls[-1].status, "ok")

    def test_resume_without_pause_timestamp_accepts_rollout(self) -> None:
        adapter, paths = self.prepare([[{"op": "api_error", "times": 5}, WRITE,
            {"op": "xh", "args": ["rollout", "--seed", "3"]},
            {"op": "xh", "args": ["wait", "A-0001", "--timeout", "20"]},
            {"op": "xh", "args": ["submit", "--final", "done"]}]])
        self.assertEqual(self.runner(adapter, paths).run()["status"], "paused_model_down")
        state = RoundState.load(paths.state)
        state.data["paused_s"] += self.clock.now() - state.data["paused_at"]
        state.data["paused_at"] = None  # timestamp already cleared during recovery
        state.save()
        self.clock.advance(3600)
        result = self.runner(adapter, paths).run()
        self.assertEqual(result["status"], "completed")
        state = RoundState.load(paths.state).data
        self.assertEqual(state["counters"]["counted"], 1)
        self.assertEqual(state["counters"]["rejected"], 0)
        self.assertLess(result["active_wall_s"], 60)

    def test_round_wall_terminates_the_session_and_auto_submits(self) -> None:
        adapter, paths = self.prepare([[WRITE, {"op": "sleep", "seconds": 30}]])
        runner = self.runner(adapter, paths)
        original = runner._tick

        def tick() -> None:
            original()
            self.clock.advance(100)  # 100 fake seconds per tick against a 600 s budget

        runner._tick = tick
        summary = runner.run()
        self.assertEqual(summary["status"], "wall_exhausted")
        self.assertEqual(read_calls(paths.agent_calls)[0].status, "aborted")
        state = RoundState.load(paths.state).data
        self.assertEqual(state["submissions"][-1]["source"], "harness_auto_final")

    def test_session_wall_cap_and_stall(self) -> None:
        adapter, paths = self.prepare([[{"op": "sleep", "seconds": 30}]],
                                      overrides={"budgets": {"wall_per_session_s": 1}})
        summary = self.runner(adapter, paths).run()
        self.assertEqual(summary["status"], "stalled")
        calls = read_calls(paths.agent_calls)
        self.assertEqual([c.status for c in calls], ["timeout", "timeout"])
        self.assertEqual([c.extra["stop_reason"] for c in calls], ["wall_cap", "wall_cap"])

    def test_auth_error_is_fatal(self) -> None:
        adapter, paths = self.prepare([[{"op": "auth_error"}]])
        summary = self.runner(adapter, paths).run()
        self.assertEqual(summary["status"], "error")
        self.assertEqual(read_calls(paths.agent_calls)[0].error_type, "auth_error")

    def test_tampering_invalidates_the_round(self) -> None:
        adapter, paths = self.prepare([[WRITE, {"op": "xh", "args": ["submit", "--final", "done"]}]])
        runner = self.runner(adapter, paths)
        original = runner._tick
        judge = self.tmp / "task" / "protected" / "judge.py"

        def tick() -> None:
            original()
            judge.write_text("# changed mid-round\n")

        runner._tick = tick
        summary = runner.run()
        self.assertEqual(summary["status"], "invalid_tamper")
        self.assertIn("tamper", {v["kind"] for v in summary["violations"]})

    def test_budget_exhaustion_gets_one_final_call(self) -> None:
        sessions = [[WRITE] + [{"op": "xh", "args": ["rollout", "--seed", str(s)]} for s in (1, 2)]
                    + [{"op": "xh", "args": ["wait", "A-0001", "A-0002", "--timeout", "20"]}]
                    + [{"op": "xh", "args": ["rollout", "--seed", str(s)]} for s in (3, 4)]
                    + [{"op": "xh", "args": ["wait", "A-0003", "A-0004", "--timeout", "20"]}],
                    [{"op": "xh", "args": ["submit", "--final", "last call"]}]]
        adapter, paths = self.prepare(sessions, overrides={"budgets": {"max_counted_rollouts": 4}})
        summary = self.runner(adapter, paths).run()
        self.assertEqual(summary["status"], "completed")
        state = RoundState.load(paths.state).data
        self.assertTrue(state["sessions"][1]["final_call"])
        self.assertEqual(state["counters"]["counted"], 4)

    def test_restart_closes_a_dangling_session(self) -> None:
        adapter, paths = self.prepare([[WRITE, {"op": "xh", "args": ["submit", "--final", "x"]}]])
        state = RoundState.load(paths.state)
        state.status = "running"
        state.data["wall_started_at"] = self.clock.now()
        state.data["sessions"].append({"index": 1, "started_at": "2026-09-26T00:00:00Z", "started_at_ts": 0,
                                       "planned_session_id": None, "resumed": False, "ended_at": None,
                                       "prompt_sha256": "0" * 64})
        state.save()
        runner = self.runner(adapter, paths)
        from datafarm.expert.manifest import ManifestGuard
        from datafarm.expert.round import harness_code_sha256, manifest_entries
        from datafarm.expert.state import EventLog

        ManifestGuard(paths, manifest_entries(adapter, paths), state, EventLog(paths, "XR-toy-test", self.clock.now),
                      self.clock.now, extra={"harness_code_sha256": harness_code_sha256()}).record_start()
        summary = runner.run()
        self.assertEqual(summary["status"], "completed")
        calls = read_calls(paths.agent_calls)
        self.assertEqual((calls[0].status, calls[0].error_type), ("aborted", "harness_restart"))

    def test_spend_guards_and_adapter_drift(self) -> None:
        adapter, paths = self.prepare([[WRITE]])
        claude = TaskAdapter.load(self.tmp / "task" / "adapter.json",
                                  overrides={"agent": {"backend": "claude", "model": "claude-opus-5-5"}})
        claude_paths = RoundPaths(self.tmp / "round-claude")
        prepare_round(claude, claude_paths.root, clock=self.clock)
        with self.assertRaisesRegex(HarnessError, "approve-agent-spend"):
            RoundRunner(claude_paths, claude, clock=self.clock, rollout_backend=FakeRolloutBackend()).run()
        with self.assertRaisesRegex(HarnessError, "approve-gpu"):
            self.runner(adapter, paths, rollout_backend=None, rollout_backend_kind="command").run()
        drifted = TaskAdapter.load(self.tmp / "task" / "adapter.json", overrides={"title": "changed"})
        with self.assertRaises(HarnessError):
            self.runner(drifted, paths).run()


if __name__ == "__main__":
    unittest.main()
