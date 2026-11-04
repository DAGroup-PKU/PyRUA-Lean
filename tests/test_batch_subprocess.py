"""End-to-end tests of the subprocess task adapter and the generic ``supervise_batch`` CLI.

The attempts are tiny local Python scripts; no provider, GPU, network or credentials.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from datafarm.batch.supervisor import BatchPolicy, BatchSupervisor, LaunchDeferred, read_events
from datafarm.batch.subprocess_task import (
    SubprocessTask,
    TaskConfigError,
    load_task_config,
    pid_alive,
)

ROOT = Path(__file__).resolve().parents[1]

WORKER = r'''
import json, pathlib, sys
item, attempt, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = pathlib.Path(sys.argv[4])
marker = out.parents[1] / f"crashed-{item}"
if item == "c" and not marker.exists():
    marker.write_text("once")
    print("RuntimeError: bounded upload attempts exhausted", file=sys.stderr)
    sys.exit(1)
if item == "stale":
    attempt = 99  # a result left over from another attempt
status = "rejected_seam" if (item == "b" and attempt == 1) else "admitted"
if item == "provisional":
    status = "generated_needs_semantic_review"  # finalize died after its first write
(out / "result.json").write_text(json.dumps({"status": status, "attempt": attempt, "seed": seed}))
'''


def load_cli():
    spec = importlib.util.spec_from_file_location("supervise_batch_cli",
                                                  ROOT / "tools" / "supervise_batch.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SubprocessAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        (self.tmp / "worker.py").write_text(WORKER, encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_config(self, items: list, **overrides) -> Path:
        config = {
            "schema": "datafarm.batch-task-config/1.0.0",
            "batch_id": "fixture",
            "items": items,
            "work_dir": "batch",
            "launch": {"argv": [sys.executable, "worker.py", "{item}", "{attempt}", "{seed}",
                                "{attempt_dir}"], "cwd": "."},
            "result": {"path": "{attempt_dir}/result.json", "required_keys": ["status"],
                       "match": {"attempt": "{attempt}"}},
            "verify": {"result_status": {"key": "status", "pass": ["admitted"],
                                         "reject": {"rejected_seam": "seam"}}},
            "crash_classes": [{"pattern": "bounded upload attempts exhausted",
                               "kind": "transient"}],
            "cost": {"per_finished_attempt": 1.0},
            "seed": {"mode": "linear", "base": 1000, "item_stride": 1, "attempt_stride": 100},
            "policy": {"poll_interval_s": 0.05, "crash_retry_delay_s": 0.0,
                       "crash_burst_items": 0, "max_attempts_per_item": 3,
                       "max_crash_retries_per_item": 2},
        }
        config.update(overrides)
        path = self.tmp / "task.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def run_cli(self, *argv: str) -> tuple[int, str]:
        cli = load_cli()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(list(argv))
        return code, out.getvalue()

    def test_upstream_wait_does_not_spawn_or_spend_retry_and_ready_item_runs(self):
        upstream = self.tmp / 'upstream.json'
        upstream.write_text(json.dumps({'item_state': {'a': {'state': 'pending'},
                                                     'b': {'state': 'admitted'}}}))
        config_path = self.write_config(['a', 'b'], upstream_batch_state=str(upstream))
        config, base, items = load_task_config(config_path)
        adapter = SubprocessTask(config, base_dir=base, items=items)
        with self.assertRaises(LaunchDeferred):
            adapter.launch('a', 1, 1000, launch_id='a.a01.t01')
        self.assertFalse((adapter.attempts_dir / 'a.a01.t01').exists())
        supervisor = BatchSupervisor(adapter.work_dir / 'state', items, adapter.task(),
                                     BatchPolicy(max_concurrency=1, poll_interval_s=.05))
        with supervisor:
            supervisor.run(max_ticks=4)
            self.assertEqual(supervisor.item('a')['crash_retries'], 0)
            self.assertEqual(supervisor.item('a')['attempts_used'], 0)
            self.assertGreater(supervisor.item('b')['tries'], 0)
            upstream.write_text(json.dumps({'item_state': {
                k: {'state': 'admitted'} for k in items}}))
            self.assertEqual(supervisor.run(), 'complete')

    def test_cli_runs_a_batch_to_completion(self) -> None:
        config = self.write_config(["a", "b", "c"])
        code, output = self.run_cli("run", str(config))
        self.assertEqual(code, 0, output)
        summary = json.loads(output)
        self.assertEqual((summary["status"], summary["admitted"]), ("complete", 3))
        state_dir = self.tmp / "batch" / "state"
        state = json.loads((state_dir / "state.json").read_text())["item_state"]
        self.assertEqual((state["b"]["attempts_used"], state["b"]["admitted_attempt"]), (2, 2))
        self.assertEqual((state["c"]["crash_retries"], state["c"]["attempts_used"]), (1, 1))
        attempts = sorted((self.tmp / "batch" / "attempts").iterdir())
        c_seeds = {json.loads((path / "attempt.json").read_text())["seed"]
                   for path in attempts if path.name.startswith("c.")}
        self.assertEqual(c_seeds, {1000 + 2}, "linear seed; the crash retry kept it")
        b_seeds = sorted(json.loads((path / "attempt.json").read_text())["seed"]
                         for path in attempts if path.name.startswith("b."))
        self.assertEqual(b_seeds, [1001, 1101])
        for path in attempts:
            self.assertTrue((path / "exit.json").is_file())
            self.assertTrue((path / "started.json").is_file())
        crashed = [e for e in read_events(state_dir) if e["kind"] == "crashed"]
        self.assertEqual(len(crashed), 1)
        self.assertIn("bounded upload attempts exhausted", crashed[0]["detail"])
        code, output = self.run_cli("status", str(config))
        self.assertEqual((code, json.loads(output)["status"]), (0, "complete"))

    def test_stale_result_is_a_crash_not_a_verdict(self) -> None:
        config = self.write_config(["stale"])
        code, output = self.run_cli("run", str(config))
        self.assertEqual(code, 3, output)
        summary = json.loads(output)
        gap = summary["gaps"][0]
        self.assertEqual((gap["item"], gap["reason"], gap["attempts_used"]),
                         ("stale", "crash_budget", 0))
        self.assertIn("stale result", gap["last"]["detail"])

    def test_provisional_result_is_a_crash_not_a_verdict(self) -> None:
        config = self.write_config(["provisional"])
        data = json.loads(config.read_text())
        data["result"]["incomplete_values"] = {"status": ["generated_needs_semantic_review"]}
        config.write_text(json.dumps(data))
        code, output = self.run_cli("run", str(config))
        self.assertEqual(code, 3, output)
        gap = json.loads(output)["gaps"][0]
        self.assertEqual((gap["reason"], gap["attempts_used"]), ("crash_budget", 0))
        self.assertIn("provisional", gap["last"]["detail"])

    def test_verifier_nonzero_exit_cannot_admit_even_after_pass_json(self) -> None:
        verifier = self.tmp / "verifier.py"
        verifier.write_text(
            'import sys\nprint(\'{"decision": "pass"}\')\n'
            'print("judge failed after verdict", file=sys.stderr)\nsys.exit(7)\n',
            encoding="utf-8")
        config = self.write_config(
            ["a"], verify={"argv": [sys.executable, str(verifier)]},
            policy={"poll_interval_s": 0.05, "crash_retry_delay_s": 0.0,
                    "crash_burst_items": 0, "max_attempts_per_item": 1,
                    "max_verify_errors": 1})
        code, output = self.run_cli("run", str(config))
        self.assertEqual(code, 3, output)
        summary = json.loads(output)
        self.assertEqual((summary["admitted"], summary["gaps"][0]["reason"]),
                         (0, "attempt_budget"))
        self.assertEqual(summary["gaps"][0]["last"]["gate"], "verifier_error")
        events = read_events(self.tmp / "batch" / "state")
        self.assertTrue(any(event["kind"] == "verify_error" and "exit 7" in event["error"]
                            for event in events))

    def test_launch_timeout_pauses_without_relaunching_unknown_remote_effects(self) -> None:
        worker = self.tmp / "slow_worker.py"
        descendant_pid = self.tmp / "descendant.pid"
        worker.write_text(
            "import pathlib, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
            "time.sleep(30)\n", encoding="utf-8")
        config = self.write_config(
            ["a"], launch={"argv": [sys.executable, str(worker), str(descendant_pid)],
                           "timeout_s": 1.0},
            policy={"poll_interval_s": 0.05, "max_consecutive_unknown": 1,
                    "max_attempts_per_item": 2, "crash_retry_delay_s": 0.0,
                    "crash_burst_items": 0})
        code, output = self.run_cli("run", str(config))
        self.assertEqual(code, 5, output)
        summary = json.loads(output)
        self.assertEqual(summary["status"], "paused_unknown_liveness")
        attempts = list((self.tmp / "batch" / "attempts").iterdir())
        self.assertEqual(len(attempts), 1)
        exit_record = json.loads((attempts[0] / "exit.json").read_text())
        self.assertTrue(exit_record["timed_out"])
        self.assertEqual(exit_record["timeout_s"], 1.0)
        state = json.loads((self.tmp / "batch" / "state" / "state.json").read_text())
        self.assertEqual(state["item_state"]["a"]["attempts_used"], 0)
        self.assertEqual(state["item_state"]["a"]["crash_retries"], 0)
        self.assertFalse(pid_alive(exit_record["child_pid"]))
        self.assertTrue(descendant_pid.is_file())
        self.assertFalse(pid_alive(int(descendant_pid.read_text())))

    def test_unresolved_remote_submission_marker_pauses_without_retry(self) -> None:
        worker = self.tmp / "ambiguous_worker.py"
        worker.write_text(
            "import pathlib, sys\n"
            "pathlib.Path(sys.argv[1], 'remote-submit-pending').write_text('POST sent; no ID')\n"
            "print('ConnectionError: response lost', file=sys.stderr)\n"
            "sys.exit(1)\n", encoding="utf-8")
        config = self.write_config(
            ["a"], launch={"argv": [sys.executable, str(worker), "{attempt_dir}"]},
            unresolved_submission_marker="remote-submit-pending",
            lease_probe={"argv": [sys.executable, "-c", "print('dead')"]},
            policy={"poll_interval_s": 0.05, "max_consecutive_unknown": 1,
                    "max_attempts_per_item": 2, "crash_retry_delay_s": 0.0,
                    "crash_burst_items": 0, "lost_confirmations": 1})
        code, output = self.run_cli("run", str(config))
        self.assertEqual(code, 5, output)
        self.assertEqual(json.loads(output)["status"], "paused_unknown_liveness")
        attempts = list((self.tmp / "batch" / "attempts").iterdir())
        self.assertEqual(len(attempts), 1)
        self.assertEqual(json.loads((attempts[0] / "exit.json").read_text())["returncode"], 1)
        state_dir = self.tmp / "batch" / "state"
        events = read_events(state_dir)
        self.assertTrue(any(e["kind"] == "unknown" for e in events))
        self.assertFalse(any(e["kind"] in ("crashed", "lost") for e in events))
        state = json.loads((state_dir / "state.json").read_text())["item_state"]["a"]
        self.assertEqual((state["attempts_used"], state["crash_retries"]), (0, 0))
        code, output = self.run_cli("run", str(config))
        self.assertEqual(code, 5, output)
        self.assertEqual(len(list((self.tmp / "batch" / "attempts").iterdir())), 1)

    def test_unresolved_remote_submission_marker_must_be_local_filename(self) -> None:
        for marker in ("", "../pending", "/tmp/pending", ".", 4):
            config = self.write_config(["a"], unresolved_submission_marker=marker)
            with self.subTest(marker=marker), self.assertRaises(TaskConfigError):
                load_task_config(config)

    def test_unresolved_submission_survives_wrapper_disappearance(self) -> None:
        config = self.write_config(
            ["a"], unresolved_submission_marker="remote-submit-pending",
            lease_probe={"argv": [sys.executable, "-c", "print('dead')"]})
        loaded, base_dir, items = load_task_config(config)
        adapter = SubprocessTask(loaded, base_dir=base_dir, items=items)
        values = adapter.values("a", 1, 1000, "a.1")
        attempt_dir = Path(values["attempt_dir"])
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "remote-submit-pending").write_text("request may have reached provider")
        handle = adapter._handle(values, 2**31 - 1, None)
        self.assertEqual(adapter.poll(handle).status, "unknown")
        self.assertEqual(adapter.lease_probe(handle), "unknown")

    def test_launch_timeout_must_be_positive_and_finite(self) -> None:
        for timeout in (0, -1, float("inf"), float("nan"), True):
            config = self.write_config(["a"], launch={"argv": ["true"],
                                                       "timeout_s": timeout})
            with self.subTest(timeout=timeout), self.assertRaises(TaskConfigError):
                load_task_config(config)

    def test_missing_result_cost_pauses_before_next_launch_and_resumes_same_attempt(self) -> None:
        config = self.write_config(
            ["a", "b"], cost={"result_key": "cost_usd"},
            policy={"poll_interval_s": 0.05, "max_concurrency": 1,
                    "max_total_cost": 1.0, "launch_cost_estimate": 0.25})
        code, output = self.run_cli("run", str(config))
        summary = json.loads(output)
        self.assertEqual((code, summary["status"], summary["admitted"]),
                         (5, "paused_cost_unknown", 0))
        self.assertEqual(summary["counters"]["cost"], 0.0)
        attempts = list((self.tmp / "batch" / "attempts").iterdir())
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0].name.startswith("a."))
        state = json.loads((self.tmp / "batch/state/state.json").read_text())
        self.assertFalse(state["item_state"]["a"]["cost_recorded"])
        self.assertEqual(state["item_state"]["a"]["attempts_used"], 0)
        result_path = attempts[0] / "result.json"
        result = json.loads(result_path.read_text())
        result["cost_usd"] = 1.0
        result_path.write_text(json.dumps(result))
        self.assertEqual(self.run_cli("acknowledge", str(config), "--reason",
                                      "restored provider charge")[0], 0)
        code, output = self.run_cli("run", str(config))
        summary = json.loads(output)
        self.assertEqual((code, summary["status"], summary["admitted"]),
                         (4, "stopped_cost_cap", 1))
        self.assertEqual(summary["counters"]["cost"], 1.0)
        self.assertEqual(len(list((self.tmp / "batch" / "attempts").iterdir())), 1)

    def test_cost_configuration_rejects_invalid_values(self) -> None:
        for value in (-1, float("inf"), float("nan"), True, "1"):
            config = self.write_config(["a"], cost={"per_finished_attempt": value})
            with self.subTest(value=value), self.assertRaises(TaskConfigError):
                load_task_config(config)
        for value in (None, "", 4):
            config = self.write_config(["a"], cost={"result_key": value})
            with self.subTest(value=value), self.assertRaises(TaskConfigError):
                load_task_config(config)

    def test_revoke_reopens_an_admitted_item(self) -> None:
        config = self.write_config(["a", "b"])
        self.assertEqual(self.run_cli("run", str(config))[0], 0)
        code, _ = self.run_cli("revoke", str(config), "--items", "a", "--reason",
                               "cohort audit: three_view_motion_structure")
        self.assertEqual(code, 0)
        code, output = self.run_cli("run", str(config))
        self.assertEqual(code, 0, output)
        state = json.loads((self.tmp / "batch/state/state.json").read_text())["item_state"]["a"]
        self.assertEqual((state["attempts_used"], state["admitted_attempt"]), (2, 2))
        self.assertEqual(state["attempts"][0]["revoked"]["gate"], "cohort_audit")

    def test_template_batch_config_loads(self) -> None:
        config, base_dir, items = load_task_config(ROOT / "configs/batch/template.json")
        self.assertEqual(len(items), 3)
        self.assertGreater(BatchPolicy.from_dict(config["policy"]).max_rounds, 0)
        self.assertIsNotNone(SubprocessTask(config, base_dir=base_dir, items=items).task())

    def test_per_item_variables(self) -> None:
        config = self.write_config(
            ["a", "b"], item_vars={"a": {"prompt": "prompts/a.txt"}, "b": {"prompt": "b.txt"}},
            launch={"argv": [sys.executable, "worker.py", "{item}", "{attempt}", "{seed}",
                             "{attempt_dir}", "--prompt", "{vars[prompt]}"], "cwd": "."})
        loaded, base_dir, items = load_task_config(config)
        adapter = SubprocessTask(loaded, base_dir=base_dir, items=items)
        argv, _, _ = adapter.command(loaded["launch"], adapter.values("a", 1, 5, "a.a01.t01"))
        self.assertEqual(argv[-2:], ["--prompt", "prompts/a.txt"])
        missing = self.write_config(["a", "b"], item_vars={"a": {"prompt": "x"}})
        with self.assertRaises(TaskConfigError):
            load_task_config(missing)

    def test_validate_and_bad_configs(self) -> None:
        config = self.write_config(["7", 8])
        code, output = self.run_cli("validate", str(config))
        self.assertEqual(code, 0)
        plan = json.loads(output)
        self.assertEqual(plan["items"], 2)
        self.assertIn("1007", plan["example_launch"]["argv"])
        bad = self.write_config(["a"], launch={"argv": ["x", "{not_a_field}"]})
        with self.assertRaises(TaskConfigError):
            load_task_config(bad)
        typo = self.write_config(["a"], policy={"max_attempt_per_item": 2})
        code, _ = self.run_cli("validate", str(typo))
        self.assertEqual(code, 2)

    def test_stop_file_and_acknowledge(self) -> None:
        config = self.write_config(["a", "b"])
        code, _ = self.run_cli("stop", str(config), "--reason", "inspect first outputs")
        self.assertEqual(code, 0)
        code, output = self.run_cli("run", str(config))
        self.assertEqual(code, 4)
        self.assertEqual(json.loads(output)["status"], "stopped_operator")
        code, _ = self.run_cli("acknowledge", str(config), "--reason", "looked fine")
        self.assertEqual(code, 0)
        code, output = self.run_cli("run", str(config))
        self.assertEqual((code, json.loads(output)["status"]), (0, "complete"))

    def test_reconcile_after_supervisor_death_finds_the_attempt(self) -> None:
        config_path = self.write_config(["a"])
        config, base_dir, items = load_task_config(config_path)
        adapter = SubprocessTask(config, base_dir=base_dir, items=items)
        task = adapter.task()
        handle = task.launch("a", 1, task.seed_for("a", 1), launch_id="a.a01.t01")
        deadline = time.time() + 30
        while adapter.poll(handle).status == "running" and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(adapter.poll(handle).status, "finished")
        found = adapter.reconcile("a", 1, task.seed_for("a", 1), "a.a01.t01")
        self.assertEqual(found["launch_id"], "a.a01.t01")
        self.assertIsNone(adapter.reconcile("a", 1, task.seed_for("a", 1), "a.a01.t09"))
        self.assertIs(pid_alive(None), None)

    def test_supervisor_uses_the_adapter_directly(self) -> None:
        config_path = self.write_config(["a", "b"])
        config, base_dir, items = load_task_config(config_path)
        adapter = SubprocessTask(config, base_dir=base_dir, items=items)
        policy = BatchPolicy.from_dict(config["policy"])
        with BatchSupervisor(adapter.work_dir / "state", items, adapter.task(), policy) as sup:
            self.assertEqual(sup.run(), "complete")
            self.assertEqual(sup.summary()["counters"]["cost"], 3.0)


if __name__ == "__main__":
    unittest.main()
