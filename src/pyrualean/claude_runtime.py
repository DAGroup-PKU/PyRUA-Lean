# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Drive one code-arm episode with Claude Code through the Claude Agent SDK.

The second agent runtime of :mod:`pyrualean.play` (``--runtime claude``).
Everything around the agent loop is the Codex runtime's: the same HTTP MCP
server over :class:`~pyrualean.arms.ArmToolkit`, the same prompt text and
step-0 card images.  Claude Code owns the loop (``claude_agent_sdk.query``)
and keeps its default system prompt, as RPent's Claude planner does: our
prompt is the first user message, the card images ride along as base64
image blocks.  This module owns the wall clock, the event dump
(``claude_events.jsonl``) and the usage audit that becomes
``result["generation"]``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import mimetypes
import shutil
import subprocess
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .mcp_server import SERVER_NAME

#: Read-only Claude Code built-ins the agent keeps.  The Codex runtime ran with
#: a read-only sandbox shell; RPent's Claude arm gets the same three.
BUILTIN_TOOLS = ("Read", "Glob", "Grep")
#: SDK turns granted on top of the decision budget: :class:`ArmToolkit` already
#: refuses calls past the budget and the agent needs a turn to call ``finish``.
TURN_ALLOWANCE = 5
#: SDK turn ceiling of an episode without a decision budget (wall clock only).
UNBOUNDED_TURNS = 10_000
#: One stream-json line may carry a tool result with several PNG images.
MAX_BUFFER_BYTES = 64 * 1024 * 1024
#: Result subtypes that end the run by a budget, not by a provider failure.
_BUDGET_SUBTYPES = ("error_max_turns", "error_max_budget_usd")
_MCP_PREFIX = f"mcp__{SERVER_NAME}__"


def sdk_max_turns(max_turns: int | None) -> int:
    """The ``--max-turns`` handed to Claude Code (one turn per model request)."""

    return max_turns + TURN_ALLOWANCE if max_turns else UNBOUNDED_TURNS


def arm_tools(arm: str) -> list[str]:
    """The MCP tool names of ``arm`` as Claude Code sees them."""

    main = "python" if arm == "cells" else "run_program"
    return [f"{_MCP_PREFIX}{main}", f"{_MCP_PREFIX}finish"]


def allowed_tools(arm: str) -> list[str]:
    return [*BUILTIN_TOOLS, *arm_tools(arm)]


def resolve_cli(cli: str | None) -> str | None:
    """Absolute path of the ``claude`` executable, ``None`` for the SDK's bundled one."""

    if not cli:
        return None
    path = Path(cli).expanduser()
    if path.is_file():
        return str(path.absolute())
    found = shutil.which(cli)
    return str(Path(found).absolute()) if found else None


def cli_version(cli_path: str | None) -> str | None:
    if not cli_path:
        return None
    try:
        proc = subprocess.run(
            [cli_path, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None


def image_block(path: str | Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    media_type = mimetypes.guess_type(str(path))[0] or "image/png"
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def initial_message(prompt_text: str, image_paths: list[str | Path]) -> dict[str, Any]:
    """The episode's one user message, in the SDK's streaming-input shape.

    The dict is what ``claude_agent_sdk.query`` writes verbatim to Claude
    Code's stdin (``--input-format stream-json``); ``message`` is an API
    ``MessageParam`` whose content holds the prompt text and one base64
    image block per card image.
    """

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    content += [image_block(path) for path in image_paths]
    return {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }


def build_options(
    sdk: Any,
    *,
    arm: str,
    model: str,
    reasoning_effort: str,
    max_turns: int | None,
    mcp_url: str,
    workspace: str | Path,
    cli_path: str | None,
    env: dict[str, str],
    stderr: Callable[[str], None] | None = None,
) -> Any:
    """``ClaudeAgentOptions`` of one episode.

    Effort maps like RPent's planner: ``none`` disables thinking, any other
    level is passed as ``effort``.  ``setting_sources=[]`` keeps user and
    project configuration (settings, CLAUDE.md) out of the session;
    ``strict_mcp_config`` keeps every MCP server but ours out.  No
    ``permission_mode``: the MCP tools are pre-approved through
    ``allowed_tools`` and the built-ins are read-only, so Claude Code never
    has to ask.
    """

    thinking = {"type": "disabled"} if reasoning_effort == "none" else None
    effort = None if reasoning_effort == "none" else reasoning_effort
    return sdk.ClaudeAgentOptions(
        model=model,
        cwd=str(workspace),
        max_turns=sdk_max_turns(max_turns),
        tools=list(BUILTIN_TOOLS),
        allowed_tools=allowed_tools(arm),
        mcp_servers={SERVER_NAME: {"type": "http", "url": mcp_url}},
        strict_mcp_config=True,
        setting_sources=[],
        thinking=thinking,
        effort=effort,
        cli_path=cli_path,
        env=dict(env),
        max_buffer_size=MAX_BUFFER_BYTES,
        stderr=stderr,
    )


# -- event dump ---------------------------------------------------------------


def _kind(value: Any) -> str:
    return type(value).__name__


def _jsonable(value: Any) -> Any:
    """JSON-ready copy of a message payload, images reduced to their size."""

    if isinstance(value, dict):
        source = value.get("source")
        if value.get("type") == "image" and isinstance(source, dict) and "data" in source:
            # API-shaped block: {"type": "image", "source": {"type": "base64", "data": ...}}
            rest = {k: v for k, v in value.items() if k != "source"}
            slim = {k: v for k, v in source.items() if k != "data"}
            slim["data_bytes"] = len(str(source["data"]))
            return {**rest, "source": slim}
        if value.get("type") == "image" and "data" in value:
            # MCP-shaped block: {"type": "image", "data": ..., "mimeType": ...}
            rest = {k: v for k, v in value.items() if k != "data"}
            return {**rest, "data_bytes": len(str(value["data"]))}
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    return value


def message_to_json(message: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(message) and not isinstance(message, type):
        data = dataclasses.asdict(message)
    elif hasattr(message, "__dict__"):
        data = dict(vars(message))
    else:
        data = {"value": repr(message)}
    return {"type": _kind(message), **_jsonable(data)}


# -- usage audit ----------------------------------------------------------------


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _prompt_tokens(usage: dict[str, Any]) -> int:
    """Prompt tokens of one request, counted the way Codex counts ``input_tokens``."""

    return (
        _int(usage.get("input_tokens"))
        + _int(usage.get("cache_read_input_tokens"))
        + _int(usage.get("cache_creation_input_tokens"))
    )


def _thinking_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("output_tokens_details")
    if isinstance(details, dict) and details.get("thinking_tokens") is not None:
        return _int(details["thinking_tokens"])
    return None


class _Audit:
    """Accumulate usage, tool calls and errors from the SDK's message stream."""

    def __init__(self) -> None:
        self.requests: dict[str, dict[str, Any]] = {}
        self.request_order: list[str] = []
        self.request_arrival: list[float] = []
        self.request_latency: list[float] = []
        self._boundary: float = 0.0
        self._anonymous = 0
        self.model: str | None = None
        self.session_id: str | None = None
        self.init: dict[str, Any] | None = None
        self.tool_calls: dict[str, int] = {}
        self.tool_requests: list[int] = []
        self.tool_results = 0
        self.tool_errors = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.compactions = 0
        self.last_text: str | None = None
        self.result: Any = None

    def observe(self, message: Any, at: float) -> None:
        kind = _kind(message)
        if kind == "AssistantMessage":
            self._assistant(message, at)
        elif kind == "UserMessage":
            self._user(message, at)
        elif kind == "ResultMessage":
            self._result(message)
        elif hasattr(message, "subtype") and hasattr(message, "data"):
            self._system(message)  # SystemMessage and its task / hook subclasses

    def _system(self, message: Any) -> None:
        subtype = str(getattr(message, "subtype", ""))
        data = getattr(message, "data", None)
        data = data if isinstance(data, dict) else {}
        if subtype == "init":
            self.session_id = data.get("session_id") or self.session_id
            keys = ("model", "claude_code_version", "permissionMode", "tools", "mcp_servers")
            self.init = {key: data.get(key) for key in keys}
            for server in data.get("mcp_servers") or []:
                if isinstance(server, dict) and server.get("status") not in (None, "connected"):
                    self.errors.append(f"mcp server {server.get('name')}: {server.get('status')}")
        elif subtype == "compact_boundary":
            self.compactions += 1
        elif "retry" in subtype or "error" in subtype:
            self.warnings.append(f"system {subtype}: {json.dumps(_jsonable(data))[:300]}")

    def _assistant(self, message: Any, at: float) -> None:
        if getattr(message, "parent_tool_use_id", None):
            return  # a subagent's message; none are expected (no Task tool)
        self.model = self.model or getattr(message, "model", None)
        key = getattr(message, "message_id", None)
        if not key:
            self._anonymous += 1
            key = f"anonymous-{self._anonymous}"
        if key not in self.requests:
            self.request_order.append(key)
            self.request_arrival.append(at)
            self.request_latency.append(round(at - self._boundary, 3))
        usage = getattr(message, "usage", None)
        if isinstance(usage, dict):
            self.requests[key] = usage
        else:
            self.requests.setdefault(key, {})
        request_index = len(self.request_order)
        for block in getattr(message, "content", None) or []:
            block_kind = _kind(block)
            if block_kind == "TextBlock":
                text = str(getattr(block, "text", "")).strip()
                if text:
                    self.last_text = text
            elif block_kind == "ToolUseBlock":
                name = str(getattr(block, "name", "?"))
                if name.startswith(_MCP_PREFIX):
                    name = name[len(_MCP_PREFIX) :]
                    self.tool_requests.append(request_index)
                self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        if error := getattr(message, "error", None):
            self.errors.append(f"assistant error: {error}")

    def _user(self, message: Any, at: float) -> None:
        content = getattr(message, "content", None)
        blocks = content if isinstance(content, list) else []
        results = [b for b in blocks if _kind(b) == "ToolResultBlock"]
        if results or getattr(message, "parent_tool_use_id", None):
            self._boundary = at
        self.tool_results += len(results)
        self.tool_errors += sum(1 for b in results if getattr(b, "is_error", None))

    def _result(self, message: Any) -> None:
        self.result = message
        self.session_id = getattr(message, "session_id", None) or self.session_id
        subtype = str(getattr(message, "subtype", ""))
        reported = [str(e) for e in getattr(message, "errors", None) or []]
        if getattr(message, "is_error", False):
            status = getattr(message, "api_error_status", None)
            text = f"result {subtype}" + (f" (http {status})" if status else "")
            if subtype in _BUDGET_SUBTYPES:
                self.warnings.append(text)
            else:
                self.errors.append(text)
            self.errors.extend(reported)
        else:
            self.warnings.extend(reported)

    # -- summary -----------------------------------------------------------------

    def usage(self) -> dict[str, Any]:
        """Token usage in the Codex audit's vocabulary.

        One assistant message id is one model request.  Per request only the
        prompt side is trustworthy: the CLI stamps every assistant message
        with the request's ``message_start`` usage, whose ``output_tokens``
        is a streaming placeholder.  The run totals come from the result
        message (verified on Claude Code 2.1.280: its top-level ``usage``
        sums the main model's requests, while ``usage.iterations`` holds a
        single entry for a whole tool loop, so it is ignored).  Without a
        result (timeout, crash) the prompt sums are rebuilt from the
        assistant messages and the output side is reported as unknown.
        """

        entries = [self.requests[k] for k in self.request_order]
        per_request = [_prompt_tokens(u) for u in entries]
        profile: list[int] = []
        for tokens in per_request:
            profile.append(profile[-1] + tokens if profile else tokens)
        out: dict[str, Any] = {"requests": len(self.request_order), "request_profile": profile}
        result_usage = getattr(self.result, "usage", None)
        if isinstance(result_usage, dict) and result_usage:
            out["input_tokens"] = _prompt_tokens(result_usage)
            out["cached_input_tokens"] = _int(result_usage.get("cache_read_input_tokens"))
            out["cache_creation_input_tokens"] = _int(
                result_usage.get("cache_creation_input_tokens")
            )
            out["output_tokens"] = _int(result_usage.get("output_tokens"))
            out["reasoning_output_tokens"] = _thinking_tokens(result_usage)
            out["usage_source"] = "result+stream"
        else:
            out["input_tokens"] = sum(per_request)
            out["cached_input_tokens"] = sum(
                _int(u.get("cache_read_input_tokens")) for u in entries
            )
            out["cache_creation_input_tokens"] = sum(
                _int(u.get("cache_creation_input_tokens")) for u in entries
            )
            out["output_tokens"] = None
            out["reasoning_output_tokens"] = None
            out["usage_source"] = "stream"
        out["total_tokens"] = (
            out["input_tokens"] + out["output_tokens"] if out["output_tokens"] is not None else None
        )
        return out

    def generation(self) -> dict[str, Any]:
        result = self.result
        model_usage = getattr(result, "model_usage", None)
        model_usage = model_usage if isinstance(model_usage, dict) else {}
        # The agent's model is the one on its assistant messages; Claude Code's
        # own side calls (Haiku) only show up in the per-model breakdown.
        model = self.model or next(iter(model_usage), None)
        usage = self.usage()
        warnings = list(self.warnings)
        if usage["output_tokens"] is None and usage["requests"]:
            warnings.append(
                "no result message: output tokens unknown (assistant usage carries only "
                "the streaming placeholder)"
            )
        out: dict[str, Any] = {
            "runtime": "claude",
            "model": model,
            "models": sorted(model_usage) if model_usage else ([model] if model else []),
            "session_id": self.session_id,
            **usage,
            "model_usage": model_usage,
            "request_latency_s": list(self.request_latency),
            "turns": getattr(result, "num_turns", None),
            "tool_calls": dict(self.tool_calls),
            "tool_requests": list(self.tool_requests),
            "tool_results": self.tool_results,
            "tool_errors": self.tool_errors,
            "errors": list(self.errors),
            "warnings": warnings,
            "compactions": self.compactions,
            "result_subtype": getattr(result, "subtype", None),
            "is_error": getattr(result, "is_error", None),
            "max_turns_hit": getattr(result, "subtype", None) == "error_max_turns",
            "stop_reason": getattr(result, "stop_reason", None),
            "terminal_reason": getattr(result, "terminal_reason", None),
            "duration_ms": getattr(result, "duration_ms", None),
            "duration_api_ms": getattr(result, "duration_api_ms", None),
            "total_cost_usd": getattr(result, "total_cost_usd", None),
            "permission_denials": len(getattr(result, "permission_denials", None) or []),
            "init": self.init,
        }
        return out


# -- replay ---------------------------------------------------------------------

#: Content-block classes of the SDK, recognised by their dumped keys.
_BLOCK_KEYS = (
    ("ToolUseBlock", {"id", "name", "input"}),
    ("ToolResultBlock", {"tool_use_id"}),
    ("ThinkingBlock", {"thinking"}),
    ("TextBlock", {"text"}),
)


def _named(name: str, fields: dict[str, Any]) -> Any:
    return type(name, (SimpleNamespace,), {})(**fields)


def _revive_block(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    keys = set(block)
    for name, needed in _BLOCK_KEYS:
        if needed <= keys:
            return _named(name, block)
    return block


def _revive(record: dict[str, Any]) -> Any:
    """A message-like object from one ``claude_events.jsonl`` record."""

    fields = {k: v for k, v in record.items() if k not in ("type", "t")}
    if isinstance(fields.get("content"), list):
        fields["content"] = [_revive_block(block) for block in fields["content"]]
    return _named(str(record.get("type") or "Unknown"), fields)


def replay_events(path: str | Path) -> dict[str, Any]:
    """Re-audit a ``claude_events.jsonl``: the ``generation`` record without
    the run's ``status`` / ``returncode`` / ``options`` (which only the live
    run knows).  Lets a recorded episode be re-audited with the current usage mapping."""

    audit = _Audit()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            audit.observe(_revive(record), float(record.get("t") or 0.0))
    return audit.generation()


#: The token-accounting keys of a ``generation`` record.
USAGE_KEYS = (
    "model",
    "models",
    "requests",
    "request_profile",
    "input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
    "usage_source",
    "model_usage",
    "tool_requests",
)


def usage_from_events(path: str | Path) -> dict[str, Any]:
    """Token accounting of a ``claude_events.jsonl``: ``request_profile`` is the
    cumulative prompt tokens (input + cache read + cache creation) over the
    unique assistant message ids, the totals come from the result message
    (``model_usage`` as-is; Claude Code's Haiku helper calls only appear
    there)."""

    generation = replay_events(path)
    return {key: generation.get(key) for key in USAGE_KEYS}


# -- the episode ----------------------------------------------------------------


async def _prompt_stream(message: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    yield message


async def _drive(
    sdk: Any,
    options: Any,
    message: dict[str, Any],
    *,
    budget_s: float,
    audit: _Audit,
    events_path: Path,
    on_timeout: Callable[[], None] | None,
) -> tuple[str, int | None]:
    """Consume one ``query`` under the wall clock; returns ``(status, returncode)``."""

    started = time.perf_counter()
    stream = sdk.query(prompt=_prompt_stream(message), options=options)

    async def consume() -> None:
        with open(events_path, "a", encoding="utf-8") as events_file:
            async for msg in stream:
                at = time.perf_counter() - started
                audit.observe(msg, at)
                record = {"t": round(at, 3), **message_to_json(msg)}
                events_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                events_file.flush()

    status, returncode = "completed", None
    consumer = asyncio.ensure_future(consume())
    try:
        await asyncio.wait_for(asyncio.shield(consumer), timeout=budget_s)
    except asyncio.TimeoutError:
        status = "timeout"
        if on_timeout is not None:
            with contextlib.suppress(Exception):
                on_timeout()
        consumer.cancel()
        with contextlib.suppress(BaseException):
            await consumer
    except Exception as exc:  # noqa: BLE001 - recorded, the episode result decides
        status = "error"
        returncode = 2
        audit.errors.append(f"{type(exc).__name__}: {str(exc)[:500]}")
    finally:
        with contextlib.suppress(BaseException):
            await stream.aclose()
    if status == "completed":
        if audit.result is None:
            returncode = 3
            audit.errors.append("stream ended without a result message")
        else:
            returncode = 1 if getattr(audit.result, "is_error", False) else 0
    return status, returncode


def run_claude_agent(
    prompt_text: str,
    image_paths: list[str | Path],
    mcp_url: str,
    *,
    model: str,
    reasoning_effort: str,
    max_turns: int | None,
    budget_s: float,
    workspace: str | Path,
    out_dir: str | Path,
    cli_path: str | None,
    env: dict[str, str],
    arm: str = "cells",
    on_timeout: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run one episode's agent loop; returns the ``generation`` record.

    Every SDK message is appended as one JSON line to
    ``out_dir/claude_events.jsonl`` (``t`` = seconds since the query
    started), Claude Code's stderr to ``claude_stderr.txt`` and the final
    assistant text to ``last_message.md``.  ``status`` is ``completed``,
    ``timeout`` (the wall clock ``budget_s`` expired: ``on_timeout`` is
    called, then the query is cancelled, which terminates Claude Code) or
    ``error`` (the SDK raised); ``returncode`` is 0 for a clean result, 1
    for an ``is_error`` result, 2 after an exception, 3 when the stream
    ended without a result and ``None`` on timeout.
    """

    import claude_agent_sdk as sdk

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "claude_events.jsonl"
    stderr_path = out_dir / "claude_stderr.txt"
    audit = _Audit()
    started = time.perf_counter()
    with open(stderr_path, "a", encoding="utf-8") as stderr_file:

        def stderr(line: str) -> None:
            stderr_file.write(line.rstrip("\n") + "\n")
            stderr_file.flush()

        options = build_options(
            sdk,
            arm=arm,
            model=model,
            reasoning_effort=reasoning_effort,
            max_turns=max_turns,
            mcp_url=mcp_url,
            workspace=workspace,
            cli_path=cli_path,
            env=env,
            stderr=stderr,
        )
        message = initial_message(prompt_text, image_paths)
        status, returncode = asyncio.run(
            _drive(
                sdk,
                options,
                message,
                budget_s=budget_s,
                audit=audit,
                events_path=events_path,
                on_timeout=on_timeout,
            )
        )
    final_text = getattr(audit.result, "result", None) or audit.last_text or ""
    (out_dir / "last_message.md").write_text(str(final_text), encoding="utf-8")
    generation = audit.generation()
    generation.update(
        {
            "status": status,
            "returncode": returncode,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "options": {
                "model": model,
                "effort": getattr(options, "effort", None),
                "thinking": getattr(options, "thinking", None),
                "max_turns": getattr(options, "max_turns", None),
                "tools": list(BUILTIN_TOOLS),
                "allowed_tools": allowed_tools(arm),
                "permission_mode": getattr(options, "permission_mode", None),
                "cli_path": cli_path or "bundled",
                "cli_version": cli_version(cli_path),
                "images": len(image_paths),
            },
        }
    )
    return generation


__all__ = [
    "BUILTIN_TOOLS",
    "TURN_ALLOWANCE",
    "UNBOUNDED_TURNS",
    "allowed_tools",
    "arm_tools",
    "build_options",
    "initial_message",
    "message_to_json",
    "replay_events",
    "resolve_cli",
    "run_claude_agent",
    "sdk_max_turns",
    "usage_from_events",
]
