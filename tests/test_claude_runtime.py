"""Offline checks of the Claude Agent SDK runtime (``pyrualean.claude_runtime``).

The SDK is replaced by a fake module whose ``query`` is an async generator of
message-like dataclasses named as the SDK's; nothing touches the network.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from pyrualean import claude_runtime as cr

# -- message-like fakes (same class names as claude_agent_sdk.types) ------------


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: bool | None = None


@dataclass
class UserMessage:
    content: Any
    uuid: str | None = None
    parent_tool_use_id: str | None = None
    tool_use_result: dict[str, Any] | None = None


@dataclass
class AssistantMessage:
    content: list[Any]
    model: str
    parent_tool_use_id: str | None = None
    error: str | None = None
    usage: dict[str, Any] | None = None
    message_id: str | None = None


@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any]


@dataclass
class ResultMessage:
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    stop_reason: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    result: str | None = None
    model_usage: dict[str, Any] | None = None
    permission_denials: list[Any] | None = None
    errors: list[str] | None = None
    api_error_status: int | None = None
    terminal_reason: str | None = None


class ClaudeAgentOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def fake_sdk(monkeypatch, messages, *, hang_after=False, raise_after=None):
    """Install a fake ``claude_agent_sdk``; returns what ``query`` captured."""

    captured: dict[str, Any] = {}
    module = types.ModuleType("claude_agent_sdk")

    async def query(*, prompt, options):
        captured["options"] = options
        captured["prompt"] = [message async for message in prompt]
        for message in messages:
            yield message
        if raise_after is not None:
            raise raise_after
        if hang_after:
            await asyncio.sleep(3600)

    module.ClaudeAgentOptions = ClaudeAgentOptions  # type: ignore[attr-defined]
    module.query = query  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return captured


def usage(inp, read, create, out, think=None):
    record = {
        "input_tokens": inp,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": create,
        "output_tokens": out,
    }
    if think is not None:
        record["output_tokens_details"] = {"thinking_tokens": think}
    return record


#: True per-request usage of the three model requests ...
ITERATIONS = [usage(100, 0, 900, 50, 10), usage(20, 900, 100, 30, 5), usage(10, 1000, 50, 20, 0)]
#: ... and what the CLI stamps on the assistant messages: the request's
#: ``message_start`` usage (prompt side right, ``output_tokens`` a placeholder).
PLACEHOLDERS = [usage(100, 0, 900, 1), usage(20, 900, 100, 1), usage(10, 1000, 50, 1)]
MODEL = "claude-opus-5-5"
SIDE_MODEL = "claude-haiku-4-5-20251001"
INIT = SystemMessage(
    "init",
    {
        "session_id": "s1",
        "model": MODEL,
        "tools": ["Read", "Glob", "Grep", "mcp__pyrualean__python", "mcp__pyrualean__finish"],
        "mcp_servers": [{"name": "pyrualean", "status": "connected"}],
    },
)


def episode_messages(*, with_result=True):
    """One Read of a guide, one python cell, one finish; three model requests."""

    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "A" * 1000},
    }
    messages = [
        INIT,
        AssistantMessage(
            [ToolUseBlock("t1", "Read", {"file_path": "guides/x.md"})],
            MODEL,
            usage=PLACEHOLDERS[0],
            message_id="m1",
        ),
        UserMessage([ToolResultBlock("t1", "guide text")]),
        # One response, two SDK messages with the same id (text, then tool use).
        AssistantMessage(
            [TextBlock("Plan: grasp the bowl.")], MODEL, usage=PLACEHOLDERS[1], message_id="m2"
        ),
        AssistantMessage(
            [ToolUseBlock("t2", "mcp__pyrualean__python", {"code": "robo.state()"})],
            MODEL,
            usage=PLACEHOLDERS[1],
            message_id="m2",
        ),
        UserMessage([ToolResultBlock("t2", [{"type": "text", "text": "{}"}, image])]),
        AssistantMessage(
            [
                ToolUseBlock(
                    "t3", "mcp__pyrualean__finish", {"status": "success", "summary": "done"}
                )
            ],
            MODEL,
            usage=PLACEHOLDERS[2],
            message_id="m3",
        ),
        UserMessage([ToolResultBlock("t3", "ok")]),
    ]
    if with_result:
        # Like Claude Code 2.1.280: top-level totals of the main model, and an
        # `iterations` list holding only the last request of the tool loop.
        total = usage(130, 1900, 1050, 100, 15)
        total["iterations"] = [ITERATIONS[-1]]
        messages.append(
            ResultMessage(
                "success",
                5000,
                4000,
                False,
                3,
                "s1",
                stop_reason="end_turn",
                usage=total,
                result="Bowl placed.",
                model_usage={
                    SIDE_MODEL: {"inputTokens": 8869, "outputTokens": 15},
                    MODEL: {"inputTokens": 130, "outputTokens": 100},
                },
                permission_denials=[],
            )
        )
    return messages


def run(tmp_path, monkeypatch, messages, *, sdk=None, **overrides):
    captured = fake_sdk(monkeypatch, messages, **(sdk or {}))
    png = tmp_path / "agentview.png"
    png.write_bytes(b"\x89PNG not really")
    kwargs: dict[str, Any] = dict(
        model=MODEL,
        reasoning_effort="high",
        max_turns=40,
        budget_s=30,
        workspace=tmp_path / "ws",
        out_dir=tmp_path / "out",
        cli_path="/opt/claude",
        env={"HOME": "/h", "CLAUDE_RELAY_BASE_URL": "https://relay.example/claude"},
        arm="cells",
    )
    kwargs.update(overrides)
    generation = cr.run_claude_agent("PROMPT TEXT", [png], "http://127.0.0.1:1/mcp/", **kwargs)
    return generation, captured, png


def test_usage_totals_from_result_and_profile_from_assistant_messages(tmp_path, monkeypatch):
    generation, _, _ = run(tmp_path, monkeypatch, episode_messages())
    assert generation["runtime"] == "claude" and generation["status"] == "completed"
    assert generation["returncode"] == 0 and generation["errors"] == []
    assert generation["model"] == MODEL and generation["session_id"] == "s1"
    assert generation["models"] == [SIDE_MODEL, MODEL]  # Claude Code's own Haiku side calls
    assert generation["model_usage"][SIDE_MODEL] == {"inputTokens": 8869, "outputTokens": 15}
    # prompt tokens per request = input + cache_read + cache_creation, like Codex's input_tokens;
    # one entry per assistant message id, the placeholder output tokens ignored
    assert generation["request_profile"] == [1000, 2020, 3080]
    assert generation["requests"] == 3 and generation["input_tokens"] == 3080
    assert generation["cached_input_tokens"] == 1900
    assert generation["cache_creation_input_tokens"] == 1050
    assert generation["output_tokens"] == 100 and generation["reasoning_output_tokens"] == 15
    assert generation["total_tokens"] == 3180 and generation["usage_source"] == "result+stream"
    assert generation["tool_calls"] == {"Read": 1, "python": 1, "finish": 1}
    assert generation["tool_requests"] == [2, 3]  # which request issued each MCP call
    assert generation["tool_results"] == 3 and generation["tool_errors"] == 0
    assert generation["turns"] == 3 and generation["max_turns_hit"] is False
    assert generation["permission_denials"] == 0 and generation["compactions"] == 0
    assert len(generation["request_latency_s"]) == 3
    assert generation["init"]["mcp_servers"] == [{"name": "pyrualean", "status": "connected"}]
    assert generation["options"]["max_turns"] == 45 and generation["options"]["images"] == 1
    out = tmp_path / "out"
    lines = [json.loads(line) for line in (out / "claude_events.jsonl").read_text().splitlines()]
    assert len(lines) == 9 and lines[0]["type"] == "SystemMessage" and "t" in lines[0]
    assert lines[-1]["type"] == "ResultMessage" and lines[-1]["usage"]["output_tokens"] == 100
    dumped = (out / "claude_events.jsonl").read_text()
    assert "AAAA" not in dumped and '"data_bytes": 1000' in dumped  # images reduced to their size
    assert (out / "last_message.md").read_text() == "Bowl placed."
    assert (out / "claude_stderr.txt").is_file()


def test_options_and_initial_message(tmp_path, monkeypatch):
    _, captured, png = run(tmp_path, monkeypatch, episode_messages())
    options = captured["options"]
    assert options.model == MODEL and options.cwd == str(tmp_path / "ws")
    assert options.max_turns == 45  # decision budget + allowance for finish
    assert options.tools == ["Read", "Glob", "Grep"]
    assert options.allowed_tools == [
        "Read",
        "Glob",
        "Grep",
        "mcp__pyrualean__python",
        "mcp__pyrualean__finish",
    ]
    assert options.mcp_servers == {"pyrualean": {"type": "http", "url": "http://127.0.0.1:1/mcp/"}}
    assert options.strict_mcp_config is True and options.setting_sources == []
    assert options.effort == "high" and options.thinking is None
    assert options.cli_path == "/opt/claude"
    assert options.env == {"HOME": "/h", "CLAUDE_RELAY_BASE_URL": "https://relay.example/claude"}
    assert not hasattr(options, "permission_mode") and not hasattr(options, "system_prompt")
    assert callable(options.stderr)
    # The one user message, in the SDK's streaming-input shape.
    assert len(captured["prompt"]) == 1
    message = captured["prompt"][0]
    assert message["type"] == "user" and message["session_id"] == ""
    assert message["parent_tool_use_id"] is None
    assert message["message"]["role"] == "user"
    text, image = message["message"]["content"]
    assert text == {"type": "text", "text": "PROMPT TEXT"}
    assert image["type"] == "image" and image["source"]["type"] == "base64"
    assert image["source"]["media_type"] == "image/png"
    assert base64.b64decode(image["source"]["data"]) == png.read_bytes()


def test_effort_none_disables_thinking_and_program_arm_tools(tmp_path, monkeypatch):
    _, captured, _ = run(
        tmp_path,
        monkeypatch,
        episode_messages(),
        reasoning_effort="none",
        max_turns=None,
        arm="program",
    )
    options = captured["options"]
    assert options.thinking == {"type": "disabled"} and options.effort is None
    assert options.max_turns == cr.UNBOUNDED_TURNS
    assert options.allowed_tools[-2:] == ["mcp__pyrualean__run_program", "mcp__pyrualean__finish"]


def test_wall_clock_timeout_cancels_and_reports(tmp_path, monkeypatch):
    stopped: list[str] = []
    generation, _, _ = run(
        tmp_path,
        monkeypatch,
        [INIT],
        sdk={"hang_after": True},
        budget_s=0.3,
        on_timeout=lambda: stopped.append("timeout"),
    )
    assert generation["status"] == "timeout" and generation["returncode"] is None
    assert stopped == ["timeout"] and generation["errors"] == []
    assert generation["requests"] == 0 and generation["request_profile"] == []
    lines = (tmp_path / "out" / "claude_events.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_budget_subtypes_are_warnings_provider_failures_are_errors(tmp_path, monkeypatch):
    ceiling = ResultMessage("error_max_turns", 1, 1, True, 45, "s1", usage=usage(1, 2, 3, 4))
    generation, _, _ = run(tmp_path, monkeypatch, [INIT, ceiling])
    assert generation["max_turns_hit"] is True and generation["returncode"] == 1
    assert generation["errors"] == [] and generation["warnings"] == ["result error_max_turns"]
    failure = ResultMessage(
        "error_during_execution",
        1,
        1,
        True,
        0,
        "s1",
        errors=["API Error: 403 Forbidden"],
        api_error_status=403,
    )
    generation, _, _ = run(tmp_path, monkeypatch, [INIT, failure])
    assert generation["errors"] == [
        "result error_during_execution (http 403)",
        "API Error: 403 Forbidden",
    ]
    assert generation["returncode"] == 1 and generation["max_turns_hit"] is False


def test_sdk_exception_and_failed_mcp_server_are_errors(tmp_path, monkeypatch):
    generation, _, _ = run(tmp_path, monkeypatch, [INIT], sdk={"raise_after": RuntimeError("boom")})
    assert generation["status"] == "error" and generation["returncode"] == 2
    assert generation["errors"] == ["RuntimeError: boom"]
    failed = SystemMessage(
        "init", {"session_id": "s2", "mcp_servers": [{"name": "pyrualean", "status": "failed"}]}
    )
    generation, _, _ = run(tmp_path, monkeypatch, [failed])
    assert "mcp server pyrualean: failed" in generation["errors"]
    assert generation["returncode"] == 3  # the stream ended without a result message
    assert "stream ended without a result message" in generation["errors"]


def test_usage_falls_back_to_assistant_messages_without_a_result(tmp_path, monkeypatch):
    generation, _, _ = run(tmp_path, monkeypatch, episode_messages(with_result=False))
    assert generation["usage_source"] == "stream"
    assert generation["request_profile"] == [1000, 2020, 3080]  # duplicates by message id collapse
    assert generation["requests"] == 3 and generation["input_tokens"] == 3080
    assert generation["cached_input_tokens"] == 1900
    # the assistant messages only carry the streaming placeholder for the output side
    assert generation["output_tokens"] is None and generation["reasoning_output_tokens"] is None
    assert generation["total_tokens"] is None
    assert any("output tokens unknown" in w for w in generation["warnings"])
    assert generation["model"] == MODEL and generation["models"] == [MODEL]
    assert generation["turns"] is None and generation["returncode"] == 3


def test_replay_events_reproduces_the_live_audit(tmp_path, monkeypatch):
    live, _, _ = run(tmp_path, monkeypatch, episode_messages())
    replayed = cr.replay_events(tmp_path / "out" / "claude_events.jsonl")
    for key in (
        "model",
        "models",
        "session_id",
        "requests",
        "request_profile",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "tool_calls",
        "tool_requests",
        "tool_results",
        "turns",
        "errors",
        "warnings",
        "result_subtype",
        "init",
    ):
        assert replayed[key] == live[key], key
    assert "status" not in replayed and "options" not in replayed
    accounting = cr.usage_from_events(tmp_path / "out" / "claude_events.jsonl")
    assert set(accounting) == set(cr.USAGE_KEYS)
    assert accounting["request_profile"] == [1000, 2020, 3080]
    assert accounting["input_tokens"] == 3080 and accounting["output_tokens"] == 100
    assert accounting["model"] == MODEL and accounting["models"] == [SIDE_MODEL, MODEL]


def test_helpers(tmp_path):
    assert cr.sdk_max_turns(40) == 45 and cr.sdk_max_turns(None) == cr.UNBOUNDED_TURNS
    assert cr.arm_tools("cells") == ["mcp__pyrualean__python", "mcp__pyrualean__finish"]
    assert cr.arm_tools("program") == ["mcp__pyrualean__run_program", "mcp__pyrualean__finish"]
    assert cr.resolve_cli(None) is None and cr.resolve_cli("/nonexistent/claude-xyz") is None
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    assert cr.resolve_cli(str(binary)) == str(binary)
    record = cr.message_to_json(SimpleNamespace(a=b"\x00" * 5, b=[{"type": "image", "data": "xx"}]))
    assert record == {
        "type": "SimpleNamespace",
        "a": {"type": "bytes", "size": 5},
        "b": [{"type": "image", "data_bytes": 2}],
    }


def test_play_episode_wires_the_claude_runtime(tmp_path, monkeypatch, backend):
    """``play_episode(runtime="claude")`` on a fake host: files, fields and options."""

    # play_episode serves the arm's tools through the real in-process MCP server
    for module in ("mcp", "uvicorn", "httpx"):
        pytest.importorskip(module, reason=f"the MCP server needs {module} (the dev extra has it)")
    from pyrualean import play
    from pyrualean.robots import get_adapter

    captured = fake_sdk(monkeypatch, episode_messages())
    monkeypatch.chdir(tmp_path)
    png = tmp_path / "agentview.png"
    png.write_bytes(b"\x89PNG not really")
    card = SimpleNamespace(
        robot="libero",
        max_env_steps=600,
        task_language="put the bowl on the plate",
        images={"agentview": str(png)},
        render=lambda: "# Task\n\nput the bowl on the plate",
    )
    host = SimpleNamespace(
        boot=lambda cell, out_dir, rpent_root=None, **options: (backend, []),
        dump_card=lambda toolkit, cell, out_dir: card,
        make_backend=lambda toolkit: toolkit,
    )
    adapter = SimpleNamespace(**vars(get_adapter("libero")), host_module=lambda: host)
    monkeypatch.setattr(play, "get_adapter", lambda name: adapter)
    monkeypatch.setattr(play, "resolve_rpent_root", lambda explicit=None: tmp_path)
    cell = SimpleNamespace(
        suite="libero_object_swap", task=2, seed=0, max_episode_steps=600, cuda_device=0, tag="t2s0"
    )
    out = tmp_path / "out"
    result = play.play_episode(
        cell,
        out,
        arm="cells",
        model=MODEL,
        reasoning_effort="high",
        budget_s=30,
        max_turns=10,
        images="on-demand",
        feedback="pure",
        runtime="claude",
        claude_cli="/nonexistent/claude-xyz",
    )
    assert result["status"] == "completed" and result["environment_success"] is False
    assert result["host"]["runtime"] == "claude" and result["host"]["claude_cli"] == "bundled"
    assert "codex_context" not in result["host"] and "codex_returncode" not in result
    assert result["model"] == MODEL and result["claude_returncode"] == 0
    assert result["planner_attempts"] == 1 and result["finish"] is None
    generation = result["generation"]
    assert generation["runtime"] == "claude" and generation["request_profile"] == [1000, 2020, 3080]
    assert generation["sandbox_flags"] == [] and generation["options"]["max_turns"] == 15
    options = captured["options"]
    assert options.cwd == str(out / "workspace")
    assert options.mcp_servers["pyrualean"]["url"].startswith("http://127.0.0.1:")
    assert options.env["no_proxy"].endswith("localhost,127.0.0.1,::1")
    prompt = captured["prompt"][0]["message"]["content"]
    assert prompt[0]["text"] == (out / "prompt.txt").read_text()
    assert prompt[1]["source"]["media_type"] == "image/png"
    assert result["prompt"]["images"] == [str(png)]
    for name in ("result.json", "tool_turns.json", "claude_events.jsonl", "claude_stderr.txt"):
        assert (out / name).is_file(), name
    assert json.loads((out / "result.json").read_text())["generation"]["requests"] == 3
