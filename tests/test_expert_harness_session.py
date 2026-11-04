"""Expert harness sessions: stream parsing, cost accounting, error classification, headless
command construction, and the wall-capped process runner (real child processes, no model)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

from datafarm.expert.session import (ClaudeCliBackend, ClaudeStreamParser, CodexExecBackend,
                                          CodexJsonParser, SessionRunner, SessionSpec, classify_errors,
                                          estimate_cost, make_agent_backend)

PRICES = {"claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_write_5m": 5.0, "cache_write_1h": 8.0,
                              "cache_read": 0.2}}


def events(*items: dict) -> list[str]:
    return [json.dumps(item) for item in items]


class ParserTests(unittest.TestCase):
    def test_result_event_supplies_cost_usage_turns_and_session(self) -> None:
        parser = ClaudeStreamParser(PRICES)
        for line in events(
                {"type": "system", "subtype": "init", "session_id": "s-1", "model": "claude-opus-5-5"},
                {"type": "assistant", "session_id": "s-1", "message": {"id": "m1", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}], "usage": {"input_tokens": 5}}},
                {"type": "assistant", "session_id": "s-1", "message": {"id": "m1", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}], "usage": {"input_tokens": 5}}},
                {"type": "result", "subtype": "success", "is_error": False, "num_turns": 7, "session_id": "s-1",
                 "total_cost_usd": 1.25, "result": "done",
                 "usage": {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 30,
                           "cache_creation_input_tokens": 40}}):
            parser.feed(line)
        parser.feed("not json")
        parsed = parser.finish()
        self.assertEqual((parsed.session_id, parsed.model, parsed.turns), ("s-1", "claude-opus-5-5", 7))
        self.assertEqual((parsed.cost_usd, parsed.cost_source), (1.25, "provider"))
        self.assertEqual((parsed.input_tokens, parsed.output_tokens, parsed.cache_read_tokens,
                          parsed.cache_write_tokens), (10, 20, 30, 40))
        self.assertEqual(parsed.tool_uses, 1)
        self.assertEqual(parsed.bad_lines, 1)
        self.assertEqual(classify_errors(parsed, "", 0), (False, False, None))

    def test_killed_session_is_priced_from_per_message_usage(self) -> None:
        parser = ClaudeStreamParser(PRICES)
        for line in events(
                {"type": "system", "subtype": "init", "session_id": "s-2", "model": "claude-opus-5-5"},
                {"type": "assistant", "message": {"id": "a", "usage": {
                    "input_tokens": 1_000_000, "output_tokens": 0, "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0}}},
                {"type": "assistant", "message": {"id": "b", "usage": {
                    "input_tokens": 0, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
                    "cache_creation_input_tokens": 2_000_000,
                    "cache_creation": {"ephemeral_1h_input_tokens": 1_000_000}}}}):
            parser.feed(line)
        parsed = parser.finish()
        self.assertFalse(parsed.saw_result)
        self.assertEqual(parsed.cost_source, "price_table")
        self.assertAlmostEqual(parsed.cost_usd, 4.0 + 20.0 + 0.2 + 5.0 + 8.0)
        self.assertIsNone(estimate_cost("unknown-model", {"input_tokens": 5}, PRICES))

    def test_api_outage_and_auth_classification(self) -> None:
        outage = ClaudeStreamParser()
        outage.feed(json.dumps({"type": "result", "is_error": True, "subtype": "success",
                                "result": "API Error: 529 overloaded_error", "session_id": "x"}))
        self.assertEqual(classify_errors(outage.finish(), "", 1), (True, False, "api_outage"))
        auth = ClaudeStreamParser()
        auth.feed(json.dumps({"type": "result", "is_error": True, "result": "Invalid API key · Please run /login"}))
        self.assertEqual(classify_errors(auth.finish(), "", 1), (False, True, "auth_error"))
        silent = ClaudeStreamParser()
        self.assertEqual(classify_errors(silent.finish(), "Error: fetch failed ECONNRESET", 1)[0], True)
        busy = ClaudeStreamParser()
        busy.feed(json.dumps({"type": "assistant", "message": {"id": "m", "content": [
            {"type": "tool_use", "id": "t", "name": "Bash", "input": {}}]}}))
        # a session that did work before the API failed is a normal (resumable) end, not an outage
        self.assertEqual(classify_errors(busy.finish(), "API Error: ECONNRESET", 1)[0], False)

    def test_codex_parser(self) -> None:
        parser = CodexJsonParser(PRICES, model="claude-opus-5-5")
        for line in events({"type": "thread.started", "thread_id": "th-1"},
                           {"type": "item.completed", "item": {"type": "command_execution"}},
                           {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
                           {"type": "turn.completed", "usage": {"input_tokens": 1_000_000,
                                                                "cached_input_tokens": 1_000_000,
                                                                "output_tokens": 0}}):
            parser.feed(line)
        parsed = parser.finish()
        self.assertEqual((parsed.session_id, parsed.tool_uses, parsed.turns), ("th-1", 1, 1))
        self.assertEqual((parsed.input_tokens, parsed.cache_read_tokens), (0, 1_000_000))
        self.assertAlmostEqual(parsed.cost_usd, 0.2)
        unpriced = CodexJsonParser({}, model="codex-test-model")
        unpriced.feed(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 4}}))
        self.assertIsNone(unpriced.finish().cost_usd)


class CommandTests(unittest.TestCase):
    def test_claude_first_and_resumed_sessions(self) -> None:
        backend = ClaudeCliBackend({"cli": "/usr/local/bin/claude", "model": "claude-opus-5-5",
                                    "tools": ["Bash", "Read"], "disallowed_tools": ["WebFetch"],
                                    "permission_mode": "bypassPermissions", "settings_file": "/r/bin/s.json"})
        first = backend.command(SessionSpec(1, "PROMPT", "/ws", new_session_id="uuid-1", max_budget_usd=12.5))
        self.assertEqual(first[:3], ["/usr/local/bin/claude", "-p", "PROMPT"])
        for flag, value in (("--output-format", "stream-json"), ("--model", "claude-opus-5-5"),
                            ("--session-id", "uuid-1"), ("--tools", "Bash,Read"),
                            ("--allowedTools", "Bash,Read"), ("--disallowedTools", "WebFetch"),
                            ("--max-budget-usd", "12.50"), ("--setting-sources", "user"),
                            ("--settings", "/r/bin/s.json")):
            self.assertEqual(first[first.index(flag) + 1], value)
        self.assertIn("--verbose", first)
        self.assertIn("--strict-mcp-config", first)
        resumed = backend.command(SessionSpec(2, "GO ON", "/ws", resume_id="uuid-1"))
        self.assertEqual(resumed[resumed.index("--resume") + 1], "uuid-1")
        self.assertNotIn("--session-id", resumed)
        self.assertEqual(backend.env()["IS_SANDBOX"], "1")

    def test_codex_exec_and_resume(self) -> None:
        backend = make_agent_backend({"backend": "codex", "model": "codex-test-model",
                                      "codex": {"cli": "/usr/local/bin/codex"}}, {})
        self.assertIsInstance(backend, CodexExecBackend)
        first = backend.command(SessionSpec(1, "PROMPT", "/ws"))
        self.assertEqual(first[:3], ["/usr/local/bin/codex", "exec", "--json"])
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", first)
        self.assertEqual(first[-3:], ["-C", "/ws", "-"])  # prompt via stdin
        self.assertTrue(backend.uses_stdin_prompt)
        self.assertEqual(first[first.index("-m") + 1], "codex-test-model")
        resumed = backend.command(SessionSpec(2, "GO ON", "/ws", resume_id="th-1"))
        self.assertEqual(resumed[-3:], ["resume", "th-1", "-"])
        defaulted = make_agent_backend({"backend": "codex", "model": "claude-opus-5-5",
                                        "codex": {"model": "codex-test-model"}}, {})
        self.assertEqual(defaulted.model, "codex-test-model")


FAKE_CLI = r'''
import json, os, sys, time
mode = sys.argv[1]
print(json.dumps({"type": "system", "subtype": "init", "session_id": "fake", "model": "m"}), flush=True)
if mode == "hang":
    signal_file = sys.argv[2]
    import signal
    def on_term(*_):
        open(signal_file, "w").write("TERM")
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_term)
    time.sleep(120)
elif mode == "ignore":
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.system("sleep 120 &")
    time.sleep(120)
else:
    data = sys.stdin.read()
    print(json.dumps({"type": "result", "is_error": False, "num_turns": 1, "session_id": "fake",
                      "total_cost_usd": 0.5, "result": data[:20]}), flush=True)
'''


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xh-session-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cli = self.tmp / "cli.py"
        self.cli.write_text(FAKE_CLI)

    def run_cli(self, mode: str, *, wall: float, stop=lambda: None, grace: float = 1.0, extra=()):
        ticks = []
        runner = SessionRunner(tick=lambda: ticks.append(1), should_stop=stop, tick_interval_s=0.05,
                               term_grace_s=grace)
        parser = ClaudeStreamParser()
        info = runner.run([sys.executable, str(self.cli), mode, *extra], env=dict(os.environ), cwd=str(self.tmp),
                          stdin_text="hello prompt", transcript=self.tmp / "t.jsonl",
                          stderr_path=self.tmp / "e.log", wall_s=wall, parser=parser)
        return info, parser.finish(), ticks

    def test_normal_completion_streams_transcript(self) -> None:
        info, parsed, ticks = self.run_cli("ok", wall=30)
        self.assertEqual(info.stop_reason, "completed")
        self.assertEqual(info.exit_code, 0)
        self.assertEqual((parsed.cost_usd, parsed.result_text), (0.5, "hello prompt"))
        self.assertEqual(len((self.tmp / "t.jsonl").read_text().splitlines()), 2)

    def test_wall_cap_sends_sigterm_to_the_group(self) -> None:
        marker = self.tmp / "term"
        start = time.monotonic()
        info, _, ticks = self.run_cli("hang", wall=0.5, extra=[str(marker)])
        self.assertEqual(info.stop_reason, "wall_cap")
        self.assertLess(time.monotonic() - start, 20)
        self.assertEqual(marker.read_text(), "TERM")
        self.assertGreater(len(ticks), 2)

    def test_sigkill_after_grace_and_no_stragglers(self) -> None:
        info, _, _ = self.run_cli("ignore", wall=0.3, grace=0.5)
        self.assertEqual(info.stop_reason, "wall_cap")
        self.assertTrue(info.killed)
        self.assertEqual(info.exit_code, -signal.SIGKILL)

    def test_stop_callback_terminates(self) -> None:
        calls = []

        def stop():
            calls.append(1)
            return "flagged_for_human" if len(calls) > 3 else None

        info, _, _ = self.run_cli("hang", wall=60, stop=stop, extra=[str(self.tmp / "m")])
        self.assertEqual(info.stop_reason, "flagged_for_human")


if __name__ == "__main__":
    unittest.main()
