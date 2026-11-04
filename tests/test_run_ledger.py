from __future__ import annotations

import datetime as dt
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from datafarm.ledger.runs_cli import main
from datafarm.ledger.ledger import (
    CellUpdate,
    LedgerError,
    RunLedger,
    make_run_id,
    plan_stages,
)

RUN_ID = "PRUN-20260921-2300-fixture"


def make(root: str, **overrides) -> RunLedger:
    values = dict(
        run_id=RUN_ID,
        title="fixture",
        request="run render to admission for three items",
        task_family="fixture",
        stages=plan_stages("render", "admission", {"render": ["generate", "verify"]}),
        items=["a", "b", "c"],
    )
    values.update(overrides)
    return RunLedger.create(root, **values)


def sync_fixture(ledger: RunLedger, *, status: str) -> dict:
    ledger.set_cells([CellUpdate("a", "render/generate", status)], source="fixture-sync")
    return {"synced": 1}


def sync_broken(ledger: RunLedger) -> dict:
    raise ValueError("native state missing")


class PlanTests(unittest.TestCase):
    def test_range_and_substage_columns(self) -> None:
        stages = plan_stages("episode", "training", {"render": ["style", "guided", "verify"]})
        columns = [c for stage in stages for c in stage.columns()]
        self.assertEqual(
            columns,
            ["episode", "render/style", "render/guided", "render/verify", "admission", "training"],
        )

    def test_rejects_reversed_range_and_outside_substages(self) -> None:
        with self.assertRaises(LedgerError):
            plan_stages("admission", "render")
        with self.assertRaises(LedgerError):
            plan_stages("render", "admission", {"training": ["x"]})

    def test_run_id_format(self) -> None:
        moment = dt.datetime(2026, 9, 21, 23, 5)
        self.assertEqual(make_run_id("my-task", now=moment), "PRUN-20260921-2305-my-task")
        with self.assertRaises(LedgerError):
            make_run_id("Bad Slug", now=moment)


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_denominator_is_fixed_and_duplicates_rejected(self) -> None:
        ledger = make(self.root)
        with self.assertRaises(LedgerError):
            ledger.set_cells([CellUpdate("z", "admission", "passed")], source="t")
        with self.assertRaises(LedgerError):
            ledger.set_cells([CellUpdate("a", "render", "passed")], source="t")
        with self.assertRaises(LedgerError):
            make(self.root, run_id="PRUN-20260921-2300-dup", items=["a", "a"])
        with self.assertRaises(LedgerError):
            make(self.root)  # same run id twice

    def test_gaps_report_where_each_item_stopped(self) -> None:
        ledger = make(self.root)
        ledger.set_cells(
            [CellUpdate(i, c, "passed") for i in "ab" for c in ("render/generate", "render/verify")]
            + [CellUpdate("a", "admission", "passed"),
               CellUpdate("b", "admission", "rejected", attempt=2, failure="appearance"),
               CellUpdate("c", "render/generate", "running")],
            source="t",
        )
        gaps = ledger.gaps()
        self.assertEqual(sorted(gaps), ["b", "c"])
        self.assertEqual(gaps["b"]["column"], "admission")
        self.assertEqual(gaps["b"]["failure"], "appearance")
        self.assertEqual(gaps["c"]["column"], "render/generate")
        status = ledger.status(probe=False)
        self.assertEqual(status["passed_exit"], 1)
        self.assertEqual(status["state"], "registered")

    def test_passing_clears_failure_and_changes_are_logged(self) -> None:
        ledger = make(self.root)
        ledger.set_cells([CellUpdate("a", "admission", "rejected", failure="seam")], source="t")
        ledger.set_cells([CellUpdate("a", "admission", "passed")], source="t")
        self.assertNotIn("failure", ledger.matrix()["cells"]["a"]["admission"])
        self.assertEqual(ledger.set_cells([CellUpdate("a", "admission", "passed")], source="t"), 0)
        cell_events = [e for e in ledger.events() if e["kind"] == "cells"]
        self.assertEqual(len(cell_events), 2)

    def test_dead_worker_without_exit_record_is_lost(self) -> None:
        ledger = make(self.root)
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        ledger.register_worker(role="supervisor", pid=dead.pid)
        status = ledger.status()
        self.assertEqual(status["state"], "lost")
        self.assertTrue(any("died" in note for note in status["attention"]))

    def test_live_worker_with_stale_progress_is_flagged(self) -> None:
        ledger = make(self.root, stale_seconds=60)
        worker = ledger.register_worker(role="supervisor", pid=os.getpid())
        ledger.heartbeat(worker)
        self.assertEqual(ledger.status()["state"], "running")
        later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=600)
        self.assertEqual(ledger.status(now=later)["state"], "running_stalled")

    def test_nonzero_exit_and_blocked_declaration(self) -> None:
        ledger = make(self.root)
        worker = ledger.register_worker(role="supervisor", pid=12345)
        ledger.end_worker(worker, 3)
        self.assertEqual(ledger.status(probe=False)["state"], "exited_with_error")
        ledger.declare("blocked", reason="3 appearance rejects", source="t")
        status = ledger.status(probe=False)
        self.assertEqual(status["state"], "blocked")
        self.assertTrue(any("appearance" in note for note in status["attention"]))

    def test_relaunch_after_blocked_declaration_is_not_masked(self) -> None:
        ledger = make(self.root)
        ledger.declare("blocked", reason="waiting", source="t")
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        ledger.register_worker(role="repair", pid=dead.pid)
        self.assertEqual(ledger.status()["state"], "lost")

    def test_native_helpers_are_idempotent(self) -> None:
        ledger = make(self.root)
        first = ledger.register_worker(role="supervisor", host="h1", pid=7)
        self.assertTrue(ledger.close_worker(first, reason="native blocked"))
        self.assertFalse(ledger.close_worker(first, reason="again"))
        self.assertTrue(ledger.mirror_declaration("blocked", reason="r", source="t"))
        self.assertFalse(ledger.mirror_declaration("blocked", reason="r", source="t"))
        self.assertEqual(ledger.status(probe=False)["state"], "blocked")
        ledger.register_worker(role="relaunch", pid=8)
        self.assertTrue(ledger.mirror_declaration("blocked", reason="r", source="t"))

    def test_finalize_requires_verification_and_no_gaps(self) -> None:
        ledger = make(self.root)
        report = Path(self.root) / "verification.json"
        with self.assertRaises(LedgerError):
            ledger.finalize(report, source="t")
        report.write_text("{}", encoding="utf-8")
        with self.assertRaises(LedgerError):
            ledger.finalize(report, source="t")
        ledger.finalize(report, source="t", allow_gaps=True)
        self.assertEqual(ledger.status(probe=False)["state"], "complete_with_gaps")

    def test_child_run_links_to_parent(self) -> None:
        make(self.root)
        make(self.root, run_id="PRUN-20260921-2310-repair", items=["b"], parent_run=RUN_ID)
        parent = RunLedger(Path(self.root) / RUN_ID)
        self.assertEqual(parent.status(probe=False)["children"], ["PRUN-20260921-2310-repair"])

    def test_sync_errors_are_reported_not_raised(self) -> None:
        ledger = make(
            self.root,
            sync=[
                {"callable": "tests.test_run_ledger:sync_fixture", "args": {"status": "running"}},
                {"callable": "tests.test_run_ledger:sync_broken"},
            ],
        )
        results = ledger.sync()
        self.assertTrue(results[0]["ok"])
        self.assertFalse(results[1]["ok"])
        self.assertIn("native state missing", results[1]["error"])
        self.assertEqual(ledger.matrix()["cells"]["a"]["render/generate"]["status"], "running")


class CliTests(unittest.TestCase):
    def test_create_launch_records_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["--root", root, "create", "--run-id", RUN_ID, "--title", "t",
                          "--request", "full run", "--task-family", "fixture", "--from", "render",
                          "--to", "render", "--item-pattern", "sample-%03d:0:3"]),
                    0,
                )
                code = main(["--root", root, "launch", RUN_ID, "--role", "worker",
                             "--settle-seconds", "0.5", "--heartbeat-seconds", "5", "--",
                             sys.executable, "-c", "import time; time.sleep(2); raise SystemExit(4)"])
            self.assertEqual(code, 0)
            ledger = RunLedger(Path(root) / RUN_ID)
            deadline = time.time() + 30
            while time.time() < deadline and ledger.workers()[0]["exit_code"] is None:
                time.sleep(0.5)
            worker = ledger.workers()[0]
            self.assertEqual(worker["exit_code"], 4)
            self.assertEqual(ledger.status()["state"], "exited_with_error")
            self.assertEqual(sorted(ledger.matrix()["cells"]), ["sample-000", "sample-001", "sample-002"])
            kinds = [e["kind"] for e in ledger.events()]
            self.assertIn("persistence_check", kinds)
            self.assertIn("worker_exited", kinds)

    def test_status_json_lists_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            make(root)
            out = io.StringIO()
            with redirect_stdout(out):
                main(["--root", root, "status", "--json", "--gaps", "--no-probe"])
            report = json.loads(out.getvalue())
            self.assertEqual(report[0]["run_id"], RUN_ID)
            self.assertEqual(sorted(report[0]["gaps"]), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
