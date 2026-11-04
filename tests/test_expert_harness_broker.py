"""Expert harness broker (``xh`` requests), rollout job scripts and outcome classification.
Fake rollout backend and fake clock; the local job backend runs the fake simulator/judge."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

from datafarm.expert.layout import RoundPaths
from datafarm.expert.rollout import (AttemptSpec, FakeRolloutBackend, LocalRolloutBackend, Outcome,
                                          build_job_script, classify_run_dir, prepare_run_dir)
from datafarm.expert.state import EventLog, RoundState
from datafarm.expert.toolbox import REQUEST_SCHEMA, Broker
from xh_helpers import FakeClock
from datafarm.expert.util import read_jsonl, sha256_bytes
from xh_helpers import HOLDOUT, expert, write_adapter

PKG = Path(__file__).resolve().parents[1] / "src" / "datafarm" / "expert"


class BrokerFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xh-broker-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.adapter = write_adapter(self.tmp / "task")
        self.paths = RoundPaths(self.tmp / "round")
        self.paths.create()
        self.clock = FakeClock()
        self.state = RoundState.create(self.paths.state, round_id="XR-test", created_at="2026-09-26T00:00:00Z")
        self.state.status = "running"
        self.state.data["wall_started_at"] = self.clock.now()
        self.events = EventLog(self.paths, "XR-test", self.clock.now)
        self.backend = FakeRolloutBackend(polls_to_finish=2)
        self.broker = self.make_broker()
        (self.paths.workspace / "expert.py").write_text(expert(0.99))

    def make_broker(self, **kwargs) -> Broker:
        return Broker(paths=self.paths, adapter=self.adapter, state=self.state, backend=self.backend,
                      events=self.events, clock=self.clock, holdout_seeds=set(HOLDOUT), poll_interval_s=0,
                      **kwargs)

    def request(self, kind: str, args: dict, rid: str | None = None) -> str:
        rid = rid or f"20260926T000000-{uuid.uuid4().hex[:12]}"
        body = {"schema": REQUEST_SCHEMA, "request_id": rid, "kind": kind, "args": args}
        (self.paths.requests / f"{rid}.json").write_text(json.dumps(body))
        return rid

    def ask(self, kind: str, args: dict) -> dict:
        rid = self.request(kind, args)
        self.broker.tick()
        return json.loads((self.paths.responses / f"{rid}.json").read_text())

    def finish_all(self) -> None:
        for _ in range(5):
            self.clock.advance(1)
            self.broker.tick(force_poll=True)


class BrokerTests(BrokerFixture):
    def test_rollout_is_snapshotted_launched_and_ledgered(self) -> None:
        answer = self.ask("rollout", {"seed": 3, "note": "first", "hypothesis": "passes"})
        self.assertEqual((answer["status"], answer["attempt_id"], answer["state"]), ("accepted", "A-0001", "running"))
        run_dir = Path(answer["run_dir"])
        data = (self.paths.workspace / "expert.py").read_bytes()
        self.assertEqual((run_dir / "expert_snapshot.py").read_bytes(), data)
        self.assertEqual(answer["expert_sha256"], sha256_bytes(data))
        (self.paths.workspace / "expert.py").write_text(expert(0.0))  # later edits do not touch the snapshot
        self.finish_all()
        ledger = read_jsonl(self.paths.attempts)
        self.assertEqual([r["event"] for r in ledger], ["launch", "end"])
        self.assertEqual(ledger[1]["outcome_class"], "judged")
        self.assertTrue(ledger[1]["counted"])
        result = json.loads((self.paths.results / "A-0001.json").read_text())
        self.assertEqual(result["family"], "family_d")
        status = json.loads(self.paths.status.read_text())
        self.assertEqual(status["budget"]["counted_rollouts"]["used"], 1)
        self.assertFalse(list(self.paths.requests.iterdir()))
        self.assertEqual(len(list(self.paths.processed.iterdir())), 1)

    def test_unchanged_expert_guard_preserves_trials_and_allows_replication(self):
        self.broker.policy["reject_unchanged_rollouts"] = True
        first = self.ask("rollout", {"seed": 3})
        self.finish_all()
        count = self.broker.counted()
        rejected = self.ask("rollout", {"seed": 3, "note": "I lowered the grasp"})
        self.assertEqual(rejected["code"], "unchanged_expert")
        self.assertIn(first["attempt_id"], rejected["reason"])
        self.assertEqual(self.broker.counted(), count)
        self.assertEqual(len(self.backend.launched), 1)
        repeated = self.ask("rollout", {"seed": 3, "repeat": True})
        self.assertEqual(repeated["status"], "accepted")
        self.finish_all()
        (self.paths.workspace / "expert.py").write_text(expert(0.1))
        changed = self.ask("rollout", {"seed": 3})
        self.assertEqual(changed["status"], "accepted")

    def test_non_dev_and_holdout_seeds_are_rejected_and_recorded(self) -> None:
        self.assertEqual(self.ask("rollout", {"seed": 500})["code"], "not_dev_seed")
        self.assertEqual(self.ask("rollout", {"seed": HOLDOUT[1]})["code"], "holdout_seed")
        self.assertEqual(self.ask("rollout", {"seed": "3"})["code"], "invalid_request")
        self.assertEqual(self.ask("rollout", {"seed": True})["code"], "invalid_request")
        counters = self.state.counters
        self.assertEqual((counters["non_dev_seed"], counters["holdout_seed"], counters["counted"]), (1, 1, 0))
        self.assertEqual({v["kind"] for v in self.state.data["violations"]},
                         {"non_dev_seed_request", "holdout_seed_request"})
        self.assertEqual(self.backend.launched, [])

    def test_concurrency_counts_live_attempts(self) -> None:
        self.backend.polls_to_finish = 100
        self.assertEqual(self.ask("rollout", {"seed": 1})["status"], "accepted")
        self.assertEqual(self.ask("rollout", {"seed": 2})["status"], "accepted")
        third = self.ask("rollout", {"seed": 3})
        self.assertEqual(third["code"], "concurrency")
        self.assertEqual(self.state.counters["counted"], 0)

    def test_counted_budget_includes_rollouts_in_flight(self) -> None:
        self.state.counters["counted"] = 5
        self.backend.polls_to_finish = 100
        self.assertEqual(self.ask("rollout", {"seed": 1})["status"], "accepted")
        self.assertEqual(self.ask("rollout", {"seed": 2})["code"], "budget")

    def test_infra_failures_are_not_counted_and_capped(self) -> None:
        for _ in range(3):
            self.assertEqual(self.ask("rollout", {"seed": 13})["status"], "accepted")  # fake invalid start
            self.finish_all()
        self.assertEqual((self.state.counters["counted"], self.state.counters["infra"]), (0, 3))
        self.assertEqual(self.state.stop_reason, "infra_exhausted")
        self.assertEqual(self.ask("rollout", {"seed": 1})["code"], "round_closed")

    def test_unsafe_expert_paths_and_audit(self) -> None:
        secret = self.tmp / "secret.py"
        secret.write_text(expert(0.5))
        (self.paths.workspace / "evil.py").symlink_to(secret)
        self.assertEqual(self.ask("rollout", {"seed": 1, "file": "evil.py"})["code"], "unsafe_path")
        self.assertEqual(self.ask("rollout", {"seed": 1, "file": "../harness/state.json"})["code"], "unsafe_path")
        (self.paths.workspace / "bad.py").write_text("import os\nclass ToyExpert: pass\n")
        audit = self.ask("rollout", {"seed": 1, "file": "bad.py"})
        self.assertEqual(audit["code"], "audit")
        self.assertIn("os", audit["reason"])
        self.assertEqual(self.ask("rollout", {"seed": 1, "class": "Missing"})["code"], "audit")
        self.assertEqual(self.ask("rollout", {"seed": 1, "class": "not valid"})["code"], "invalid_request")
        self.assertEqual(self.backend.launched, [])
        self.assertEqual(self.state.counters["unsafe_path"], 2)

    def test_malformed_requests(self) -> None:
        rid = f"20260926T000000-{uuid.uuid4().hex[:12]}"
        (self.paths.requests / f"{rid}.json").write_text("{not json")
        other = self.request("rollout", {"seed": 1}, rid=f"20260926T000001-{uuid.uuid4().hex[:12]}")
        body = json.loads((self.paths.requests / f"{other}.json").read_text())
        body["request_id"] = "mismatch-0000000"
        (self.paths.requests / f"{other}.json").write_text(json.dumps(body))
        unknown = self.request("teleport", {})
        (self.paths.requests / "x.json").write_text("{}")
        self.broker.tick()
        for request_id in (rid, other, unknown):
            self.assertEqual(json.loads((self.paths.responses / f"{request_id}.json").read_text())["code"],
                             "invalid_request")
        self.assertFalse(list(self.paths.requests.iterdir()))

    def test_submissions_and_final(self) -> None:
        first = self.ask("submit", {"note": "baseline"})
        self.assertEqual((first["status"], first["submission_id"], first["final"]), ("accepted", "S-01", False))
        self.assertTrue(first["warnings"])  # never ran
        self.assertNotIn("holdout_pass_rate", json.dumps(first))
        self.ask("rollout", {"seed": 2})
        self.finish_all()
        final = self.ask("submit", {"note": "done", "final": True})
        self.assertEqual((final["final"], final["dev_rollouts_same_expert"], final["warnings"]), (True, 1, []))
        self.assertEqual(self.state.stop_reason, "done")
        self.assertEqual(self.ask("submit", {"note": "again"})["code"], "round_closed")
        self.assertEqual(self.ask("rollout", {"seed": 3})["code"], "round_closed")
        self.assertEqual((self.paths.submissions / "S-02" / "expert.py").read_text(), expert(0.99))

    def test_flag_for_human(self) -> None:
        answer = self.ask("flag-for-human", {"reason": "the grasp model cannot lift two layers"})
        self.assertEqual(answer["status"], "accepted")
        self.assertEqual(self.state.stop_reason, "flagged_for_human")
        self.assertEqual(self.ask("submit", {"note": "late"})["code"], "round_closed")
        self.assertEqual(self.ask("flag-for-human", {"reason": ""})["code"], "invalid_request")

    def test_failed_launch_is_queued_and_retried(self) -> None:
        self.backend.fail_launches = 1
        answer = self.ask("rollout", {"seed": 4})
        self.assertEqual((answer["status"], answer["state"]), ("accepted", "queued"))
        self.assertIn("no matching free GPU", answer["queue_reason"])
        self.broker.tick(force_poll=True)
        self.assertEqual(self.state.data["attempts"]["A-0001"]["status"], "pending_submit")
        self.clock.advance(61)
        self.broker.tick(force_poll=True)
        self.assertEqual(self.state.data["attempts"]["A-0001"]["status"], "running")

    def test_submit_window_expires_without_counting(self) -> None:
        self.backend.fail_launches = 100
        self.ask("rollout", {"seed": 4})
        self.clock.advance(1801)
        self.broker.tick(force_poll=True)
        attempt = self.state.data["attempts"]["A-0001"]
        self.assertEqual(attempt["outcome"]["outcome_class"], "submit_failed")
        self.assertEqual((self.state.counters["counted"], self.state.counters["infra"]), (0, 0))

    def test_lost_attempt_revives_when_its_result_appears(self) -> None:
        self.backend.lose = {"A-0001"}
        self.ask("rollout", {"seed": 2})
        self.finish_all()
        attempt = self.state.data["attempts"]["A-0001"]
        self.assertEqual(attempt["status"], "lost")
        self.assertEqual(self.state.counters["infra"], 1)
        (Path(attempt["run_dir"]) / "harness-job.json").write_text("{}")
        self.broker.tick(force_poll=True)
        self.assertEqual(attempt["status"], "finished")
        self.assertTrue(attempt["revived"])
        self.assertEqual((self.state.counters["infra"], self.state.counters["counted"]), (0, 1))
        self.assertIn("late_finish", {v["kind"] for v in self.state.data["violations"]})

    def test_drain_abandons_what_is_still_running(self) -> None:
        self.backend.polls_to_finish = 10_000
        self.ask("rollout", {"seed": 2})
        abandoned = self.broker.drain(10, sleep_s=5)
        self.assertEqual(abandoned, ["A-0001"])
        self.assertEqual(self.state.data["attempts"]["A-0001"]["outcome"]["outcome_class"], "abandoned")


class AgentToolTests(BrokerFixture):
    """The real ``xh`` script against a broker ticking in a thread."""

    def run_tool(self, *args: str) -> subprocess.CompletedProcess[str]:
        tool = self.paths.tool
        tool.write_bytes((PKG / "agent_tool.py").read_bytes())
        env = dict(os.environ, XH_ROUND_DIR=str(self.paths.root))
        return subprocess.run([sys.executable, str(tool), *args], capture_output=True, text=True, timeout=60, env=env)

    def test_tool_round_trip(self) -> None:
        stop = threading.Event()

        def serve() -> None:
            while not stop.is_set():
                self.broker.tick(force_poll=True)
                time.sleep(0.05)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            ok = self.run_tool("rollout", "--seed", "3", "--note", "hi")
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            self.assertEqual(json.loads(ok.stdout)["attempt_id"], "A-0001")
            rejected = self.run_tool("rollout", "--seed", str(HOLDOUT[0]))
            self.assertEqual(rejected.returncode, 2)
            waited = self.run_tool("wait", "A-0001", "--timeout", "20")
            self.assertEqual(waited.returncode, 0, waited.stdout)
            self.assertEqual(json.loads(waited.stdout)["finished"][0]["outcome_class"], "judged")
            self.assertEqual(json.loads(self.run_tool("result", "A-0001").stdout)["seed"], 3)
            self.assertIn("counted_rollouts", json.loads(self.run_tool("budget").stdout))
            submitted = self.run_tool("submit", "note", "--final")
            self.assertEqual(json.loads(submitted.stdout)["submission_id"], "S-01")
        finally:
            stop.set()
            thread.join(timeout=5)

    def test_tool_reads_status_without_the_harness(self) -> None:
        self.paths.status.write_text(json.dumps({"wait_cap_s": 1, "running": []}))
        status = self.run_tool("status")
        self.assertEqual(status.returncode, 0)


class JobScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xh-job-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        fakes = PKG / "fakes"
        self.adapter = write_adapter(self.tmp / "task", {
            "vars": {"py": sys.executable, "fakes": str(fakes)},
            "rollout": {
                "backend": "local", "heartbeat_interval_s": 1, "stale_after_s": 60, "max_wall_s": 120,
                "sim": {"argv": ["{py}", "{fakes}/fake_sim.py", "--expert", "{run_dir}/expert_snapshot.py:{expert_class}",
                                 "--seed", "{seed}", "--output-dir", "{run_dir}/native"],
                        "cwd": "{run_dir}/native_workdir", "log": "native.log", "timeout_s": 60,
                        "requires_output": "native/trace.json"},
                "judge": [{"name": "fake", "argv": ["{py}", "{fakes}/fake_judge.py", "{run_dir}"],
                           "stdout": "verdict.stdout", "outputs": ["verdict.json"]}],
                "verdict": {"file": "verdict.json", "success": "m2", "fields": {"m2": "m2", "err": "fold_max_xy_mm"},
                            "failure_fields": []},
                "init_state": {"file": "native/init.json", "valid": "valid", "family": "family"},
                "classify": {"infra_regex": ["CUDA error"], "expert_error_regex": ["expert_snapshot\\.py",
                                                                                   "trace exceeded"]},
                "record_digests": ["{fakes}/fake_sim.py"]},
            "isolation": {"rollout": {"mode": "none", "base": "root_ro", "writable": ["{env:HOME}/.cache-xh-test"],
                                      "extra_masked": ["{env:HOME}/.ssh"]}}})

    def spec(self, seed: int, name: str = "A-0001") -> AttemptSpec:
        return AttemptSpec(name, "dev", seed, "ToyExpert", self.tmp / "runs" / f"{name}-s{seed}", f"xh-{name}")

    def test_script_shape(self) -> None:
        spec = self.spec(3)
        text = build_job_script(self.adapter, spec, masked=["/secret", "${HOME}/.ssh"])
        self.assertIn("--seed 3", text)
        self.assertIn("expert_snapshot.py:ToyExpert", text)
        self.assertIn('mask "${HOME}/.ssh"', text)
        self.assertIn("timeout --kill-after=30 60", text)
        self.assertIn("harness-job.json", text)
        self.assertIn("rm -f", text)
        self.assertIn("unset", text)
        self.assertEqual(subprocess.run(["bash", "-n"], input=text, text=True).returncode, 0)

    def test_allowlist_runtime_readable_mount(self) -> None:
        self.adapter.raw["isolation"]["rollout"].update(
            mode="bwrap", base="allowlist", readable=["{fakes}"])
        text = build_job_script(self.adapter, self.spec(3), masked=["/secret"])
        self.assertIn('ro "' + str(PKG / "fakes") + '"', text)

    def run_local(self, seed: int, source: str, name: str = "A-0001") -> tuple[str, Outcome]:
        backend = LocalRolloutBackend(self.adapter, masked=[])
        spec = self.spec(seed, name)
        prepare_run_dir(spec, source.encode(), self.adapter)
        receipt = backend.launch(spec)
        deadline = time.time() + 60
        status = "running"
        while status == "running" and time.time() < deadline:
            time.sleep(0.2)
            status = backend.poll(spec, receipt, time.time())
        return status, backend.outcome(spec, receipt, status)

    def test_local_job_runs_sim_and_judge(self) -> None:
        status, outcome = self.run_local(3, expert(1.0))
        self.assertEqual(status, "finished")
        self.assertEqual((outcome.outcome_class, outcome.counted, outcome.success, outcome.family),
                         ("judged", True, True, "family_d"))
        run_dir = self.spec(3).run_dir
        job = json.loads((run_dir / "harness-job.json").read_text())
        self.assertEqual((job["isolation"], job["sim_exit"], job["judge_exits"]), ("none", 0, [0]))
        self.assertIn("fake_sim.py", (run_dir / "environment.txt").read_text())

    def test_local_job_outcome_classes(self) -> None:
        status, invalid = self.run_local(13, expert(1.0), "A-0002")
        self.assertEqual((invalid.outcome_class, invalid.counted), ("init_invalid", False))
        raising = expert(1.0).replace("FAKE_QUALITY", "FAKE_RAISE = True\nFAKE_QUALITY").replace(
            "self._n += 1", "self._n += 1\n        raise RuntimeError('boom')")
        _, error = self.run_local(5, raising, "A-0003")
        self.assertEqual((error.outcome_class, error.counted, error.success), ("expert_error", True, False))
        long = expert(1.0).replace("FAKE_QUALITY", "FAKE_STEPS = 99999\nFAKE_QUALITY")
        _, capacity = self.run_local(6, long, "A-0004")
        self.assertEqual(capacity.outcome_class, "expert_error")
        self.assertIn("capacity", capacity.detail)

    def test_classification_of_run_dirs(self) -> None:
        cfg = self.adapter.rollout
        run = self.tmp / "manual"
        (run / "native").mkdir(parents=True)

        def job(**fields):
            (run / "harness-job.json").write_text(json.dumps({"isolation": "ok", "sim_exit": 0, **fields}))

        job(isolation="failed")
        self.assertEqual(classify_run_dir(run, cfg).outcome_class, "isolation_failed")
        job(isolation="unavailable")
        self.assertEqual(classify_run_dir(run, cfg).outcome_class, "isolation_unavailable")
        job(sim_exit=1)
        (run / "native.log").write_text("RuntimeError: CUDA error: out of memory\n")
        self.assertEqual(classify_run_dir(run, cfg).outcome_class, "sim_failure")
        (run / "native.log").write_text("segfault somewhere\n")
        self.assertEqual(classify_run_dir(run, cfg).outcome_class, "sim_failure")
        job(sim_exit=139)
        (run / "native.log").write_text("[Fatal] commandLine = python expert_snapshot.py:Expert\n")
        crash = classify_run_dir(run, cfg)
        self.assertEqual(crash.outcome_class, "sim_failure")
        self.assertFalse(crash.counted)
        job(sim_exit=143)
        self.assertFalse(classify_run_dir(run, cfg).counted)
        job(sim_exit=124)
        self.assertFalse(classify_run_dir(run, cfg).counted)
        (run / "native.log").write_text('File "/sandbox/expert_snapshot.py", line 7\nRuntimeError: broken policy')
        self.assertTrue(classify_run_dir(run, cfg).counted)
        job(sim_exit=0)
        (run / "native.log").write_text("")
        (run / "native" / "trace.json").write_text("{}")
        self.assertEqual(classify_run_dir(run, cfg).outcome_class, "judge_failure")
        (run / "verdict.json").write_text(json.dumps({"m2": False, "fold_max_xy_mm": [80, 20]}))
        judged = classify_run_dir(run, cfg)
        self.assertEqual((judged.outcome_class, judged.success, judged.fields["err"]), ("judged", False, [80, 20]))
        for status in ("lost", "timeout", "abandoned"):
            self.assertEqual(classify_run_dir(run, cfg, status=status).counted, False)

    def test_pre_expert_failure_record_overrides_echoed_command_line(self) -> None:
        cfg = self.adapter.rollout
        run = self.tmp / 'structured-startup'
        (run / 'native').mkdir(parents=True)
        (run / 'harness-job.json').write_text(json.dumps({'isolation':'ok','sim_exit':0}))
        (run / 'native.log').write_text('commandLine: python expert_snapshot.py:Expert\n')
        failure={'expert_phase':None,'completed_primitives':0,
                 'exception_message':'camera overscan unavailable',
                 'traceback':'File "annotator_utils.py", line 454\nTypeError: None - None'}
        (run / 'native/production_failure.json').write_text(json.dumps(failure))
        outcome=classify_run_dir(run,cfg)
        self.assertEqual(outcome.outcome_class,'sim_failure')
        self.assertFalse(outcome.counted)
        failure['traceback']='File "/sandbox/expert_snapshot.py", line 3\nRuntimeError: broken plugin'
        (run / 'native/production_failure.json').write_text(json.dumps(failure))
        (run / 'native.log').write_text(failure['traceback'])
        self.assertTrue(classify_run_dir(run,cfg).counted)

    def test_stale_heartbeat_means_lost(self) -> None:
        backend = LocalRolloutBackend(self.adapter, masked=[])
        spec = self.spec(7, "A-0009")
        spec.run_dir.mkdir(parents=True)
        (spec.run_dir / "heartbeat").write_text("1")
        old = time.time() - 3600
        os.utime(spec.run_dir / "heartbeat", (old, old))
        self.assertEqual(backend.poll(spec, {"launched_at_ts": time.time() - 100}, time.time()), "lost")
        os.utime(spec.run_dir / "heartbeat", None)
        self.assertEqual(backend.poll(spec, {"launched_at_ts": time.time() - 100}, time.time()), "running")
        self.assertEqual(backend.poll(spec, {"launched_at_ts": time.time() - 1000}, time.time()), "timeout")


if __name__ == "__main__":
    unittest.main()
