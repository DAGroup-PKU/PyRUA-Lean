"""Expert harness after the round: holdout evaluation and curve, handoff bundles, the
verified expert library, the eval suite (pass@k / pass^k), the CLI and the end-to-end
dry run (scripted agent + fake simulator in real sandboxes when available)."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from datafarm.expert.adapter import TaskAdapter
from datafarm.expert.cli import main as cli_main
from datafarm.expert.evalsuite import pass_at_k, pass_hat_k, run_suite
from datafarm.expert.handoff import build_handoff, load_bundle
from datafarm.expert.holdout import draw_holdout_seeds, evaluate_holdout, select_targets
from datafarm.expert.layout import RoundPaths
from datafarm.expert.library import ExpertLibrary, add_from_round
from datafarm.expert.round import RoundRunner, harness_code_sha256, manifest_entries, prepare_round
from datafarm.expert.rollout import FakeRolloutBackend, Outcome, fake_outcome
from datafarm.expert.state import RoundState
from xh_helpers import FakeClock
from datafarm.expert.util import HarnessError
from xh_helpers import HOLDOUT, SPARES, expert, write_adapter, write_plan

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "expert-harness"


def finished_round(tmp: Path, clock: FakeClock, qualities=(0.0, 1.0), report: str = "# report\n"):
    adapter = write_adapter(tmp / "task")
    sessions = [[{"op": "write", "path": "expert.py", "text": expert(q)},
                 {"op": "xh", "args": ["rollout", "--seed", str(i + 1)]},
                 {"op": "xh", "args": ["wait", f"A-{i + 1:04d}", "--timeout", "20"]},
                 {"op": "xh", "args": ["submit", f"q={q}"] + (["--final"] if i == len(qualities) - 1 else [])}]
                for i, q in enumerate(qualities)]
    sessions[-1].insert(0, {"op": "write", "path": "REPORT.md", "text": report})
    write_plan(tmp / "task", sessions)
    paths = RoundPaths(tmp / "round")
    prepare_round(adapter, paths.root, clock=clock, round_id="XR-toy-eval")
    RoundRunner(paths, adapter, clock=clock, rollout_backend=FakeRolloutBackend(), tick_interval_s=0.05,
                poll_interval_s=0, term_grace_s=1.0, drain_timeout_s=5.0).run()
    return adapter, paths


def holdout(paths, adapter, clock, backend=None, **kwargs):
    return evaluate_holdout(paths, adapter, backend or FakeRolloutBackend(), clock=clock, poll_s=1,
                            manifest_entries=manifest_entries(adapter, paths),
                            manifest_extra={"harness_code_sha256": harness_code_sha256()}, **kwargs)


class HoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xh-eval-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clock = FakeClock()
        self.adapter, self.paths = finished_round(self.tmp, self.clock)

    def test_every_submission_is_scored_and_the_curve_is_ordered(self) -> None:
        result = holdout(self.paths, self.adapter, self.clock)
        curve = result["curve"]
        self.assertEqual([p["submission_id"] for p in curve], ["S-01", "S-02"])
        self.assertEqual([p["holdout_scored"] for p in curve], [4, 4])
        self.assertEqual((curve[0]["holdout_passed"], curve[1]["holdout_passed"]), (0, 4))
        self.assertEqual((curve[0]["cost_usd_before_session"], curve[0]["cost_usd_through_session"]), (0.0, 0.25))
        self.assertEqual(curve[1]["cost_usd_through_session"], 0.5)
        self.assertEqual(result["final_submission"], "S-02")
        self.assertEqual(result["verifier_version"], "toy-v1")
        self.assertEqual(oct(self.paths.holdout_results.stat().st_mode & 0o777), "0o600")
        public = [e for e in (self.paths.ledger / "events.jsonl").read_text().splitlines() if "holdout" in e]
        self.assertTrue(public and all("pass_rate" not in e for e in public))

    def test_spares_replace_invalid_starts_and_infra_is_retried(self) -> None:
        tries: dict[int, int] = {}

        def outcome(spec, source) -> Outcome:
            tries[spec.seed] = tries.get(spec.seed, 0) + 1
            if spec.seed == HOLDOUT[1]:
                return Outcome("init_invalid", False, None, {}, "a", False, 0, 1.0, "invalid")
            if spec.seed == HOLDOUT[2] and tries[spec.seed] == 1:
                return Outcome("sim_failure", False, None, {}, None, None, 1, 1.0, "flaky node")
            return fake_outcome(spec, source)

        result = holdout(self.paths, self.adapter, self.clock, FakeRolloutBackend(outcome_fn=outcome), targets="final")
        final = result["final"]
        self.assertEqual(final["scored"], 4)
        seeds = {o["seed"] for o in final["outcomes"] if o["counted"]}
        self.assertIn(SPARES[0], seeds)
        self.assertNotIn(HOLDOUT[1], seeds)
        self.assertEqual(tries[HOLDOUT[2]], 2)

    def test_refusals(self) -> None:
        state = RoundState.load(self.paths.state)
        state.status = "running"
        state.save()
        with self.assertRaises(HarnessError):
            holdout(self.paths, self.adapter, self.clock)
        state.status = "completed"
        state.save()
        (self.tmp / "task" / "protected" / "judge.py").write_text("# edited after the round\n")
        with self.assertRaises(HarnessError):
            holdout(self.paths, self.adapter, self.clock)
        self.assertEqual(RoundState.load(self.paths.state).status, "invalid_tamper")

    def test_target_selection(self) -> None:
        subs = [{"submission_id": f"S-0{i}", "final": i == 3} for i in (1, 2, 3, 4)]
        self.assertEqual([s["submission_id"] for s in select_targets(subs, "final", 6)], ["S-03"])
        self.assertEqual([s["submission_id"] for s in select_targets(subs, "all", 2)], ["S-03", "S-04"])
        self.assertEqual(len(select_targets(subs, "all", 1)), 1)

    def test_draw_holdout_seeds_is_balanced_and_disjoint(self) -> None:
        adapter = TaskAdapter.load(self.tmp / "task" / "adapter.json", overrides={"seeds": {
            "families": ["a", "b"], "family_command": ["{root}/fam", "{seeds}"]}})

        class Proc:
            def __init__(self, argv):
                self.returncode = 0
                self.stdout = "\n".join(f"{s} {'a' if int(s) % 2 else 'b'}" for s in argv[1:])
                self.stderr = ""

        seeds = draw_holdout_seeds(adapter, per_family=2, spares=1, range_start=50, range_size=10,
                                   run=lambda argv, **_: Proc(argv))
        self.assertEqual(len(seeds), 6)
        self.assertEqual(sum(1 for s in seeds if s.spare), 2)
        self.assertTrue(all(s.seed not in adapter.dev_seeds for s in seeds))


class HandoffLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xh-handoff-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clock = FakeClock()

    def test_bundle_next_round_and_library(self) -> None:
        adapter, paths = finished_round(self.tmp, self.clock)
        (paths.workspace / "NOTES.md").symlink_to(paths.holdout_seeds)  # must not be followed
        holdout(paths, adapter, self.clock)
        bundle = build_handoff(paths, adapter, self.tmp / "handoff", clock=self.clock)
        self.assertEqual(bundle["frozen_expert"]["submission_id"], "S-02")
        self.assertIn("REPORT.md", bundle["files"])
        self.assertNotIn("NOTES.md", bundle["files"])
        self.assertEqual(bundle["holdout_summary"]["final"]["passed"], 4)
        self.assertNotIn("outcomes", json.dumps(bundle["holdout_summary"]))
        lint = json.loads((self.tmp / "handoff" / "next-brief.lint.json").read_text())
        self.assertTrue(lint["ok"], lint["errors"])
        self.assertEqual(load_bundle(self.tmp / "handoff")["round_id"], "XR-toy-eval")
        next_paths = RoundPaths(self.tmp / "round-2")
        prepare_round(adapter, next_paths.root, clock=self.clock, handoff_dir=str(self.tmp / "handoff"))
        self.assertEqual((next_paths.workspace / "expert.py").read_text(), expert(1.0))
        self.assertIn("Frozen expert", next_paths.brief.read_text())

        library = ExpertLibrary(self.tmp / "library")
        entry = add_from_round(library, paths, adapter, now=self.clock.now())
        self.assertIsNotNone(entry)
        self.assertEqual(library.find(task_id="toy-task")[0]["entry_id"], entry["entry_id"])
        _, source = library.get(entry["entry_id"])
        self.assertEqual(source.decode(), expert(1.0))
        self.assertIsNone(add_from_round(library, paths, adapter, now=self.clock.now(), min_pass_rate=1.01))
        obj = library._object_path(entry["expert_sha256"])
        os.chmod(obj, 0o644)
        obj.write_text("tampered")
        with self.assertRaises(HarnessError):
            library.get(entry["entry_id"])

    def test_bundle_refuses_to_leak_holdout_seeds(self) -> None:
        adapter, paths = finished_round(self.tmp, self.clock, report=f"seed {HOLDOUT[2]} looked hard\n")
        with self.assertRaises(HarnessError):
            build_handoff(paths, adapter, self.tmp / "handoff", clock=self.clock)

class SuiteTests(unittest.TestCase):
    def test_estimators(self) -> None:
        self.assertEqual(pass_at_k(3, 0, 1), 0.0)
        self.assertAlmostEqual(pass_at_k(3, 1, 1), 1 / 3)
        self.assertAlmostEqual(pass_at_k(4, 2, 2), 1 - 1 / 6)
        self.assertEqual(pass_at_k(3, 2, 2), 1.0)
        self.assertAlmostEqual(pass_hat_k(4, 2, 2), 1 / 6)
        self.assertEqual(pass_hat_k(3, 1, 2), 0.0)
        self.assertIsNone(pass_at_k(1, 1, 3))

    def test_fake_suite_run_and_report(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="xh-suite-", dir="/var/tmp" if os.access("/var/tmp", os.W_OK) else None))
        try:
            write_adapter(tmp / "task")
            suite = {"schema": "datafarm.expert-harness.suite/1", "suite_id": "toy-suite", "verified_pass_rate": 0.5,
                     "k": [1, 2], "repeats": 2, "models": [{"backend": "fake", "model": "fake-a"}],
                     "tasks": [{"task_id": "toy-task", "adapter": "task/adapter.json"},
                               {"task_id": "later", "adapter": None,
                                "unavailable": {"missing_interfaces": ["plugin loader"]}}]}
            (tmp / "suite.json").write_text(json.dumps(suite))
            report = run_suite(tmp / "suite.json", tmp / "out", clock=FakeClock(), fake=True, isolation="none")
            row = report["rows"][0]
            self.assertEqual((row["rounds"], row["evaluated"], row["verified"]), (2, 2, 2))
            self.assertEqual((row["pass@1"], row["pass^2"]), (1.0, 1.0))
            self.assertAlmostEqual(row["cost_per_verified_expert"], 0.74)
            self.assertEqual(report["unavailable_tasks"][0]["task_id"], "later")
            self.assertTrue((tmp / "out" / "report.md").read_text().startswith("# Task-expert eval suite"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class CliAndDryRunTests(unittest.TestCase):
    def cli(self, *args: str) -> tuple[int, dict]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli_main(list(args))
        return code, json.loads(buffer.getvalue())

    def test_lint_brief_cli(self) -> None:
        code, report = self.cli("lint-brief", "--adapter", str(CONFIGS / "template.json"))
        self.assertEqual(code, 0, report)
        self.assertTrue(report["ok"])
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "brief.md"
            bad.write_text("# nothing here\n")
            code, report = self.cli("lint-brief", "--adapter", str(CONFIGS / "template.json"), "--brief", str(bad))
            self.assertEqual(code, 1)
            self.assertFalse(report["ok"])

    def test_prepare_status_and_refusals_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = write_adapter(Path(tmp) / "task")
            round_dir = Path(tmp) / "round"
            code, prepared = self.cli("prepare-round", "--adapter", str(adapter.path), "--round-dir", str(round_dir))
            self.assertEqual(code, 0, prepared)
            code, status = self.cli("status", "--round-dir", str(round_dir))
            self.assertEqual((code, status["status"]), (0, "prepared"))
            code, refused = self.cli("eval-holdout", "--round-dir", str(round_dir), "--rollout-backend", "command")
            self.assertEqual(code, 1)
            self.assertIn("approve-gpu", refused["message"])

    def test_smoke_rollout_cli(self) -> None:
        from datafarm.expert.dryrun import EXPERT_TEMPLATE, build_world, make_dry_run_adapter

        base = Path(tempfile.mkdtemp(prefix="xh-smoke-", dir="/var/tmp" if os.access("/var/tmp", os.W_OK) else None))
        try:
            world = build_world(base / "world")
            adapter = make_dry_run_adapter(TaskAdapter.load(CONFIGS / "template.json"), world, isolation="none")
            expert_file = base / "expert.py"
            expert_file.write_text(EXPERT_TEMPLATE.format(quality=1.0))
            args = ["smoke-rollout", "--adapter", str(adapter.path), "--expert", str(expert_file), "--class",
                    "FakeExpert", "--out", str(base / "out"), "--poll", "0.2"]
            code, result = self.cli(*args, "--seed", "3")
            self.assertEqual(code, 0, result)
            self.assertEqual((result["outcome"]["outcome_class"], result["job"]["isolation"]), ("judged", "none"))
            code, refused = self.cli(*args[:-4], "--out", str(base / "out2"), "--seed", "5000")
            self.assertEqual(code, 1)
            self.assertIn("not a development seed", refused["message"])
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_dry_run_end_to_end(self) -> None:
        from datafarm.expert.dryrun import run_dry_run

        base = Path(tempfile.mkdtemp(prefix="xh-dry-", dir="/var/tmp" if os.access("/var/tmp", os.W_OK) else None))
        try:
            result = run_dry_run(base / "run", isolation="auto")
            failed = [c for c in result["checks"] if not c["ok"] and not c["skipped"]]
            self.assertTrue(result["ok"], json.dumps(failed, indent=1, default=str)[:4000])
            names = {c["check"] for c in result["checks"]}
            for name in ("attempts_recorded_by_harness", "holdout_scored_each_submission", "handoff_bundle",
                         "library_admits_verified_expert", "call_ledger_summary", "tamper_checks_passed"):
                self.assertIn(name, names)
            if result["isolation"] != "none":
                self.assertFalse(any(c["skipped"] for c in result["checks"]))
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
