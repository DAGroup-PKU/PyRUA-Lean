from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from datafarm.agent.call_log import AgentCallRecord, append_call
from datafarm.agent.call_report import (
    build_report,
    expand_paths,
    group_rows,
    load_records,
    render_table,
)

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "agent_calls_report_tool", ROOT / "tools/agent_calls_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(call_id: str, status: str, **kwargs) -> AgentCallRecord:
    values = {
        "component": "task-expert-review",
        "backend": "openai-compatible-chat",
        "model": "codex-test-model",
        "started_at": "2026-09-26T00:00:00.000Z",
        "latency_s": 1.0,
    }
    values.update(kwargs)
    return AgentCallRecord(call_id=call_id, status=status, **values)


RECORDS = [
    record("call-1", "timeout", latency_s=600.0, extra={"logical_call_id": "l1"}),
    record("call-2", "ok", input_tokens=1000, output_tokens=200, cost_usd=0.01,
           cost_source="price_table", latency_s=100.0, extra={"logical_call_id": "l1"}),
    record("call-3", "ok", input_tokens=500, output_tokens=100, latency_s=50.0,
           extra={"logical_call_id": "l2"}),
    record("call-4", "cache_hit", cost_usd=0.0, cost_source="cache", latency_s=0.01,
           extra={"logical_call_id": "l3"}),
    record("call-5", "invalid_response", model="other-model", input_tokens=10,
           output_tokens=5, latency_s=2.0),
]


class CallReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.ledger = self.tmp / "run-a" / "agent-calls.jsonl"
        for item in RECORDS:
            append_call(self.ledger, item)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_totals_keep_denominator_and_unknown_costs(self) -> None:
        records, problems = load_records(expand_paths([self.tmp]))
        self.assertEqual(problems, [])
        totals = build_report(records)["totals"]
        self.assertEqual(totals["calls"], 5)
        self.assertEqual(totals["logical_calls"], 4)  # l1 has a retry
        self.assertEqual(totals["cache_hits"], 1)
        self.assertEqual(totals["input_tokens"], 1510)
        self.assertEqual(totals["cost_usd_known"], 0.01)
        self.assertEqual(totals["calls_with_unknown_cost"], 3)
        self.assertEqual(totals["by_status"],
                         {"timeout": 1, "ok": 2, "cache_hit": 1, "invalid_response": 1})
        self.assertEqual(totals["latency_p50_s"], 50.0)  # nearest rank of 5 values
        self.assertEqual(totals["latency_max_s"], 600.0)

    def test_grouping(self) -> None:
        rows = group_rows(RECORDS)
        keyed = {(r["model"], r["status"]): r for r in rows}
        self.assertEqual(keyed[("codex-test-model", "ok")]["calls"], 2)
        self.assertEqual(keyed[("codex-test-model", "ok")]["calls_with_unknown_cost"], 1)
        self.assertEqual(keyed[("other-model", "invalid_response")]["output_tokens"], 5)
        by_model = group_rows(RECORDS, ["model"])
        self.assertEqual([r["calls"] for r in by_model], [4, 1])
        with self.assertRaises(ValueError):
            group_rows(RECORDS, ["prompt_sha256"])

    def test_duplicates_and_invalid_lines(self) -> None:
        copy = self.tmp / "run-b" / "agent-calls.jsonl"
        copy.parent.mkdir()
        shutil.copy(self.ledger, copy)
        records, problems = load_records(expand_paths([self.tmp]))
        self.assertEqual(len(records), 5)
        self.assertEqual(sum(p["kind"] == "duplicate_call_id" for p in problems), 5)
        with copy.open("a") as handle:
            handle.write("{truncated\n")
        with self.assertRaises(ValueError):
            load_records([copy], strict=True)
        records, problems = load_records([copy], strict=False)
        self.assertEqual(len(records), 5)
        self.assertEqual([p["kind"] for p in problems], ["invalid_line"])

    def test_table_and_cli_json(self) -> None:
        table = render_table(build_report(RECORDS))
        self.assertIn("TOTAL", table)
        self.assertIn("task-expert-review", table)
        self.assertIn("by status: cache_hit=1, invalid_response=1, ok=2, timeout=1", table)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = _tool().main([str(self.tmp), "--json", "-"])
        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["schema"], "datafarm.agent-calls-report/1")
        self.assertEqual(report["totals"]["calls"], 5)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(_tool().main([str(self.tmp / "missing")]), 2)


if __name__ == "__main__":
    unittest.main()
