# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Drive one episode of a code arm with an agent runtime: Codex CLI or Claude Code.

The host boots RPent's stack for the chosen robot (``--robot``), wraps it in
the adapter's robot class (:class:`LiberoRobot` for LIBERO),
exposes the arm's tools (``python`` / ``run_program`` + ``finish``) through
RPent's in-process HTTP MCP server and launches the agent against it:
``codex exec`` (``--runtime codex``, the default) or Claude Code through the
Claude Agent SDK (``--runtime claude``, :mod:`pyrualean.claude_runtime`).
The runtime owns the agent loop; this module owns the budget, the audit of
the event stream and ``result.json``.  One episode per process.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ._backend import Ledger
from ._env import setting
from ._robot import PRIMITIVE_SETS
from .arms import ARMS, ArmToolkit, arm_contract, robot_words
from .claude_runtime import resolve_cli, run_claude_agent, sdk_max_turns
from .generate import codex_catalog_flags, provider_overrides
from .hosts.common import RESULT_SCHEMA, RpentBackend, resolve_rpent_root, stop_all
from .prompt import api_reference, knowledge_text
from .robots import ROBOTS, RobotAdapter, get_adapter
from .sandbox import DESCRIPTION as SANDBOX_DESCRIPTION

#: RPent's LIBERO guides (the default set for ``--guides`` on that robot).
GUIDE_NAMES = ("strict_hybrid_guide.md", "pro_hybrid_guide.md", "env_calibration.md")
#: Agent runtimes: the Codex CLI (``codex exec``) or Claude Code (Claude Agent SDK).
RUNTIMES = ("codex", "claude")


def _make_backend(host: Any, toolkit: Any) -> Any:
    """The host's own backend when it defines ``make_backend``, else :class:`RpentBackend`."""

    factory = getattr(host, "make_backend", None)
    return factory(toolkit) if callable(factory) else RpentBackend(toolkit)


def copy_guides(source: Path, target: Path, names: tuple[str, ...] | None = None) -> dict[str, str]:
    """Copy the operating guides ``names`` (default: LIBERO's three) from
    ``source`` into ``target``.

    Returns ``{name: sha256}`` so the run records exactly which text the
    agent could read (RPent's originals live under ``robots/<robot>/guides``).
    ``names=()`` copies every ``*.md`` file.
    """

    source = Path(source)
    if names is None:
        names = GUIDE_NAMES
    if not names:
        names = tuple(sorted(path.name for path in source.glob("*.md")))
    missing = [name for name in names if not (source / name).is_file()]
    if missing or not names:
        raise FileNotFoundError(f"guides missing under {source}: {missing or 'no *.md'}")
    target.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name in names:
        data = (source / name).read_bytes()
        (target / name).write_bytes(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    return digests


def build_prompt(
    arm: str,
    card: Any,
    *,
    budget_s: int,
    max_programs: int | None,
    images: str = "on-motion",
    feedback: str = "rich",
    guides: bool = False,
    budget_mode: str = "wall",
    adapter: RobotAdapter | None = None,
    primitives: str = "full",
) -> str:
    """The code arm's prompt: contract, API reference, knowledge, task card.

    ``primitives="no-vla"`` renders the reference and the knowledge without
    the robot's VLA primitives and names only the remaining services.
    """

    adapter = adapter or get_adapter(getattr(card, "robot", "libero"))
    contract = arm_contract(
        arm,
        budget_s=budget_s,
        max_env_steps=card.max_env_steps,
        max_programs=max_programs,
        images=images,
        feedback=feedback,
        guides=guides,
        budget_mode=budget_mode,
        robot=robot_words(adapter, primitives),
    )
    reference = api_reference(adapter.robot_cls, primitives=primitives)
    return "\n\n".join(
        [
            contract.rstrip(),
            "# API reference (the `robo` object)\n\n" + reference,
            knowledge_text(adapter.knowledge, primitives=primitives).strip(),
            card.render(),
        ]
    )


def _codex_argv(
    *,
    codex_bin: str,
    model: str,
    reasoning_effort: str,
    mcp_url: str,
    workspace: Path,
    last_message: Path,
    images: list[str],
    base_url: str | None,
    api_key_env: str | None,
) -> list[str]:
    argv = [
        codex_bin,
        "exec",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-c",
        f"mcp_servers.pyrualean.url={json.dumps(mcp_url)}",
        "-c",
        "project_doc_max_bytes=0",
        # GPT-6 Astra: the comparison's direct function calls, not code mode (docs/guide.md).
        *codex_catalog_flags(model),
        # No cached login-shell snapshot: every sandbox command starts from the
        # minimal environment we pass, so no profile can re-point CODEX_HOME.
        "--disable",
        "shell_snapshot",
        "--disable",
        "memories",
        *(
            [
                "-c",
                f"model_auto_compact_token_limit={int(os.environ['CODEX_AUTO_COMPACT_TOKEN_LIMIT'])}",
            ]
            if os.environ.get("CODEX_AUTO_COMPACT_TOKEN_LIMIT")
            else []
        ),
        *provider_overrides(base_url, api_key_env),
        "-s",
        "workspace-write",
        "-C",
        str(workspace),
        "--skip-git-repo-check",
        "--json",
        "-o",
        str(last_message),
    ]
    for image in images:
        argv += ["-i", image]
    argv.append("-")
    return argv


#: Environment variables the agent process inherits.  Everything else (the
#: launcher's own tooling variables, other Codex homes, transcripts) is
#: withheld so the agent cannot discover prior sessions through its shell.
_AGENT_ENV_KEEP = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "TZ",
    "XDG_CACHE_HOME",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
)

#: What the Claude runtime needs on top of :data:`_AGENT_ENV_KEEP`: the relay
#: that fronts the model when one is used (``CLAUDE_RELAY_*``), its CA bundle
#: for Node, and the standard Claude Code endpoint variables.
_CLAUDE_ENV_KEEP = (
    "CLAUDE_RELAY_BASE_URL",
    "CLAUDE_RELAY_KEY_FILE",
    "NODE_EXTRA_CA_CERTS",
    "CLAUDE_CONFIG_DIR",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
)


def _agent_environment(
    *, codex_home: str | None, api_key_env: str | None, runtime: str = "codex"
) -> dict[str, str]:
    """Build a minimal, leak-free environment for the agent process.

    Codex: the sandbox shell Codex gives the model is a login shell; it must
    not inherit ``ZDOTDIR``/``CODEX_HOME*`` pointers to other Codex homes
    (whose memories the model would otherwise read) nor the launcher's own
    session variables.  ``SHELL`` is pinned to bash so the user's zsh profile
    is not sourced inside the sandbox.

    Claude: the same allow-list plus :data:`_CLAUDE_ENV_KEEP`.  The Claude
    Agent SDK merges this dict *over* the launcher's environment when it
    spawns ``claude`` (it cannot withhold variables), so the list is the
    explicit contract of what the runtime needs; the agent gets no shell
    there, so nothing else can be discovered through it.
    """

    env = {k: v for k, v in os.environ.items() if k in _AGENT_ENV_KEEP}
    if api_key_env and api_key_env in os.environ:
        env[api_key_env] = os.environ[api_key_env]
    if runtime == "claude":
        env.update({k: v for k, v in os.environ.items() if k in _CLAUDE_ENV_KEEP})
    else:
        env["SHELL"] = "/bin/bash"
        env["CODEX_HOME"] = (
            str(Path(codex_home).expanduser()) if codex_home else os.environ.get("CODEX_HOME", "")
        )
        if not env["CODEX_HOME"]:
            env.pop("CODEX_HOME")
    loopback = "localhost,127.0.0.1,::1"
    upstream = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    env["NO_PROXY"] = env["no_proxy"] = ",".join(filter(None, [upstream, loopback]))
    return env


def _usage_from_rollouts(codex_home: str | None, thread_ids: list[str]) -> dict[str, Any] | None:
    """Recover cumulative token usage from Codex's session rollout files.

    ``codex exec --json`` only reports usage on ``turn.completed``; a session
    that was killed at the budget never emits it.  The rollout Codex keeps
    under ``$CODEX_HOME/sessions`` records a ``token_count`` event after every
    model response, so the last one gives the total for the thread.
    """

    if not codex_home or not thread_ids:
        return None
    root = Path(codex_home).expanduser() / "sessions"
    if not root.is_dir():
        return None
    totals: dict[str, int] = {}
    quota: Any = None
    found = False
    compactions = 0
    for thread_id in thread_ids:
        for rollout in root.rglob(f"rollout-*{thread_id}.jsonl"):
            last: dict[str, Any] | None = None
            for line in rollout.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = event.get("payload") or {}
                if event.get("type") == "event_msg" and payload.get("type") == "token_count":
                    last = payload
                if event.get("type") == "compacted" or (
                    isinstance(payload, dict) and "compact" in str(payload.get("type", "")).lower()
                ):
                    compactions += 1
            if last is None:
                continue
            found = True
            usage = (last.get("info") or {}).get("total_token_usage") or {}
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "total_tokens",
            ):
                if usage.get(key) is not None:
                    totals[key] = totals.get(key, 0) + int(usage[key])
            quota = (last.get("rate_limits") or {}).get("primary") or quota
    if not found:
        return None
    out: dict[str, Any] = dict(totals)
    out["compactions"] = compactions
    if quota:
        out["rate_limit_used_percent"] = quota.get("used_percent")
    return out


#: Codex ``error`` items that are diagnostics, not failures of the session.
#: The gateway's model id (``openai/openai/gpt-6-astra``) is not in Codex's
#: model catalog, so every session starts with a "fallback metadata" notice.
_WARNING_PATTERNS = ("Model metadata for", "Defaulting to fallback metadata")


def _is_warning(message: str) -> bool:
    return any(pattern in message for pattern in _WARNING_PATTERNS)


_CONFIG_KEYS = ("model_context_window", "model_auto_compact_token_limit")


def _codex_context_config(codex_home: str | None) -> dict[str, Any]:
    """Effective context-window and auto-compaction settings of the agent.

    Codex applies ``$CODEX_HOME/config.toml`` first and ``-c`` overrides on
    top; :func:`_codex_argv` only passes ``model_auto_compact_token_limit``
    when ``CODEX_AUTO_COMPACT_TOKEN_LIMIT`` is set, so the value in the home's
    ``config.toml`` is what governs a run otherwise.  Recorded in
    ``result.json`` because compaction changes what the model sees.
    """

    values: dict[str, Any] = {key: None for key in _CONFIG_KEYS}
    values["source"] = None
    config = Path(codex_home).expanduser() / "config.toml" if codex_home else None
    if config is not None and config.is_file():
        text = config.read_text(encoding="utf-8", errors="replace")
        for key in _CONFIG_KEYS:
            match = re.search(rf"^\s*{key}\s*=\s*(\d+)", text, re.MULTILINE)
            if match:
                values[key] = int(match.group(1))
                values["source"] = "config.toml"
    override = os.environ.get("CODEX_AUTO_COMPACT_TOKEN_LIMIT")
    if override:
        values["model_auto_compact_token_limit"] = int(override)
        values["source"] = "CODEX_AUTO_COMPACT_TOKEN_LIMIT"
    return values


def _audit_events(path: Path) -> dict[str, Any]:
    """Summarise a Codex ``--json`` event stream."""

    usage: dict[str, Any] = {}
    turns = 0
    tool_calls: dict[str, int] = {}
    shell_commands: list[str] = []
    file_changes = 0
    errors: list[str] = []
    warnings: list[str] = []
    thread_ids: list[str] = []
    if not path.is_file():
        return {"turns": None, "tool_calls": {}, "shell_commands": [], "file_changes": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = str(event.get("type", ""))
        if kind == "thread.started" and event.get("thread_id"):
            thread_ids.append(str(event["thread_id"]))
        if kind == "turn.completed":
            turns += 1
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
        if kind == "turn.failed" or kind == "error":
            errors.append(json.dumps(event)[:300])
        item = event.get("item")
        if kind == "item.completed" and isinstance(item, dict):
            item_type = str(item.get("type", ""))
            if item_type in ("mcp_tool_call", "mcpToolCall"):
                name = str(item.get("tool", "?"))
                tool_calls[name] = tool_calls.get(name, 0) + 1
            elif item_type in ("command_execution", "commandExecution"):
                shell_commands.append(str(item.get("command", ""))[:200])
            elif item_type in ("file_change", "fileChange"):
                file_changes += 1
            elif item_type == "error":
                message = str(item.get("message", ""))[:300]
                if _is_warning(message):
                    warnings.append(message)
                else:
                    errors.append(message)
    out: dict[str, Any] = {
        "turns": turns or None,
        "tool_calls": tool_calls,
        "shell_commands": shell_commands,
        "file_changes": file_changes,
        "errors": errors,
        "warnings": warnings,
        "thread_ids": thread_ids,
    }
    for target, names in (
        ("input_tokens", ("input_tokens",)),
        ("cached_input_tokens", ("cached_input_tokens",)),
        ("output_tokens", ("output_tokens",)),
        ("reasoning_output_tokens", ("reasoning_output_tokens",)),
    ):
        for name in names:
            if usage.get(name) is not None:
                out[target] = int(usage[name])
    if "input_tokens" in out and "output_tokens" in out:
        out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    return out


def play_episode(
    cell: Any,
    out_dir: Path,
    *,
    arm: str,
    model: str,
    reasoning_effort: str = "xhigh",
    budget_s: float = 1200.0,
    max_programs: int | None = None,
    codex_bin: str = "codex",
    base_url: str | None = None,
    api_key_env: str | None = None,
    rpent_root: str | None = None,
    images: str = "on-motion",
    feedback: str = "rich",
    codex_home: str | None = None,
    max_attempts: int = 3,
    guides: bool = False,
    guides_dir: str | None = None,
    max_turns: int | None = None,
    robot: str = "libero",
    boot_options: dict[str, Any] | None = None,
    runtime: str = "codex",
    claude_cli: str | None = None,
    primitives: str = "full",
) -> dict[str, Any]:
    """Run one episode of ``arm`` on ``cell`` and write ``result.json``.

    ``robot`` selects the adapter (robot class, host module, knowledge);
    ``cell`` is that host's ``Cell`` and ``boot_options`` its ``boot``
    keyword arguments (shared servers, model paths).  ``primitives`` is the
    primitive set the policy may use (``full`` or ``no-vla``: the robot's VLA
    primitives are hidden from the prompt and refused at call time; the
    servers boot as usual); ``result.json`` records it as
    ``host.primitive_set``.

    ``runtime`` picks the agent: ``codex`` runs ``codex exec`` in a
    subprocess, ``claude`` runs Claude Code through the Claude Agent SDK
    (:func:`pyrualean.claude_runtime.run_claude_agent`; ``claude_cli`` names
    the ``claude`` executable, the SDK's bundled one when it cannot be
    found).  Boot, prompt, MCP server, budgets and ``result.json`` are the
    same for both; only the event stream files (``<runtime>_events.jsonl``,
    ``<runtime>_stderr.txt``) and ``generation`` are runtime-specific.

    ``codex_home`` points Codex at a clean ``CODEX_HOME`` (no memories, no
    history) so nothing from earlier sessions leaks into the episode.
    ``guides=True`` copies the robot's operating guides (LIBERO's three by
    default) into ``workspace/guides/`` and tells the agent they are there;
    ``guides_dir`` names the directory (default: the adapter's guides under
    RPent's checkout).  ``max_turns`` switches the episode to a decision
    budget: at most that many tool calls, ``budget_s`` then being only a
    safety ceiling that the prompt does not mention.
    """

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if runtime not in RUNTIMES:
        raise ValueError(f"runtime must be one of {RUNTIMES}")
    if primitives not in PRIMITIVE_SETS:
        raise ValueError(f"primitives must be one of {PRIMITIVE_SETS}")
    adapter = get_adapter(robot)
    host = adapter.host_module()
    boot_options = dict(boot_options or {})
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = out_dir / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()
    guide_record: dict[str, Any] | bool = False
    if guides:
        if guides_dir:
            source = Path(guides_dir).expanduser().resolve()
            names: tuple[str, ...] = ()
        elif adapter.guides_subdir:
            source = resolve_rpent_root(rpent_root) / adapter.guides_subdir
            names = adapter.guide_names
        else:
            raise ValueError(f"robot {robot!r} has no default guides; pass --guides-dir")
        guide_record = {
            "source": str(source),
            "files": copy_guides(source, workspace / "guides", names),
        }
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "system": "pyrualean",
        "arm": arm,
        "backend": adapter.name,
        "robot": adapter.name,
        "suite": cell.suite,
        "task": cell.task,
        "seed": cell.seed,
        "task_language": None,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_programs": max_programs,
        "images": images,
        "feedback": feedback,
        "guides": guide_record,
        "max_turns": max_turns,
        "budget_mode": "calls" if max_turns else "wall",
        "environment_success": False,
        "status": "boot_error",
        "reason": None,
        "error": None,
        "finish": None,
        "calls": 0,
        "stateful_calls": 0,
        "env_steps": 0,
        "tool_turns": 0,
        "policy_wall_s": None,
        "startup_s": None,
        "total_wall_s": None,
        "timeout_s": budget_s,
        "max_episode_steps": cell.max_episode_steps,
        "prompt": None,
        "generation": None,
        "host": {
            "python": sys.executable,
            "cuda_device": cell.cuda_device,
            "primitive_set": primitives,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": [],
    }
    if runtime == "claude":
        result["host"]["runtime"] = "claude"
    toolkit = None
    daemons: list[Any] = []
    server = None
    ledger = Ledger()
    arm_toolkit: ArmToolkit | None = None
    try:
        toolkit, daemons = host.boot(cell, out_dir, rpent_root=rpent_root, **boot_options)
        result["host"]["boot_options"] = boot_options
        if runtime == "codex":
            result["host"]["codex_context"] = _codex_context_config(codex_home)
        result["host"]["sandbox"] = SANDBOX_DESCRIPTION
        result["host"]["rpent_root"] = str(resolve_rpent_root(rpent_root))
        result["startup_s"] = time.perf_counter() - started
        backend = _make_backend(host, toolkit)
        card = host.dump_card(toolkit, cell, out_dir)
        result["task_language"] = card.task_language
        robo = adapter.robot_cls(backend, ledger=ledger, primitives=primitives)
        # The contract tells the agent its working directory is the scratch
        # workspace; make that true for code run by the `python` tool as well
        # (Codex's own shell already runs there via `-C`).  Everything the
        # host touches from here on uses absolute paths.
        os.chdir(workspace)
        arm_toolkit = ArmToolkit(
            robo,
            arm=arm,
            workspace=workspace,
            max_programs=max_programs,
            program_timeout_s=budget_s,
            images=images,
            feedback=feedback,
            max_turns=max_turns,
        )
        from .mcp_server import ArmMcpServer

        # Tools are published as read-only/non-destructive so Codex executes
        # them under ``approval_policy = "never"`` without a reviewer model.
        server = ArmMcpServer(
            arm_toolkit,
            annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False},
        )
        mcp_url = server.start()
        prompt = build_prompt(
            arm,
            card,
            budget_s=int(budget_s),
            max_programs=max_programs,
            images=images,
            feedback=feedback,
            guides=guides,
            budget_mode="calls" if max_turns else "wall",
            adapter=adapter,
            primitives=primitives,
        )
        (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        result["prompt"] = {
            "chars": len(prompt),
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "images": list(card.images.values()),
        }
        events_path = out_dir / f"{runtime}_events.jsonl"
        stderr_path = out_dir / f"{runtime}_stderr.txt"
        # Fresh event/stderr logs for this episode; retries below append.
        events_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        images = list(card.images.values())
        env = _agent_environment(codex_home=codex_home, api_key_env=api_key_env, runtime=runtime)
        cli_path: str | None = None
        if runtime == "codex":
            argv = _codex_argv(
                codex_bin=codex_bin,
                model=model,
                reasoning_effort=reasoning_effort,
                mcp_url=mcp_url,
                workspace=workspace,
                last_message=out_dir / "last_message.md",
                images=images,
                base_url=base_url,
                api_key_env=api_key_env,
            )
            if codex_home:
                result["host"]["codex_home"] = env["CODEX_HOME"]
        else:
            cli_path = resolve_cli(claude_cli)
            result["host"]["claude_cli"] = cli_path or "bundled"
        policy_started = time.perf_counter()
        attempts = 0
        generation: dict[str, Any] | None = None
        while True:
            attempts += 1
            remaining = budget_s - (time.perf_counter() - policy_started)
            if remaining <= 0:
                status, returncode = "timeout", None
                break
            if runtime == "claude":
                generation = run_claude_agent(
                    prompt,
                    images,
                    mcp_url,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    max_turns=max_turns,
                    budget_s=remaining,
                    workspace=workspace,
                    out_dir=out_dir,
                    cli_path=cli_path,
                    env=env,
                    arm=arm,
                    on_timeout=lambda: robo._stop_episode("timeout"),
                )
                status, returncode = generation["status"], generation["returncode"]
                if status == "error":
                    # The SDK raised (Claude Code did not start, transport died):
                    # recorded in generation["errors"] and judged below like
                    # any other provider failure.
                    status = "completed"
                audit = generation
            else:
                with (
                    open(events_path, "ab") as events_file,
                    open(stderr_path, "ab") as err_file,
                ):
                    proc = subprocess.Popen(
                        argv,
                        stdin=subprocess.PIPE,
                        stdout=events_file,
                        stderr=err_file,
                        env=env,
                        start_new_session=True,
                    )
                    assert proc.stdin is not None
                    proc.stdin.write(prompt.encode("utf-8"))
                    proc.stdin.close()
                    try:
                        returncode = proc.wait(timeout=remaining)
                        status = "completed"
                    except subprocess.TimeoutExpired:
                        robo._stop_episode("timeout")
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            returncode = proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            with contextlib.suppress(ProcessLookupError):
                                os.killpg(proc.pid, signal.SIGKILL)
                            returncode = proc.wait()
                        status = "timeout"
                audit = _audit_events(events_path)
            # A planner that died before touching the robot (provider overload,
            # 429, transport error) is retried; the episode state is untouched.
            if (
                status == "completed"
                and audit.get("errors")
                and not arm_toolkit.turns
                and attempts < max_attempts
            ):
                result["notes"].append(
                    f"attempt {attempts}: planner failed before any tool call; retrying"
                )
                time.sleep(15)
                continue
            break
        result["policy_wall_s"] = time.perf_counter() - policy_started
        result["status"] = status
        result["reason"] = "timeout" if status == "timeout" else None
        result[f"{runtime}_returncode"] = returncode
        result["planner_attempts"] = attempts
        if runtime == "codex":
            result["generation"] = _audit_events(events_path)
            recovered = _usage_from_rollouts(codex_home, result["generation"].get("thread_ids", []))
            if recovered:
                if result["generation"].get("input_tokens") is None:
                    result["generation"].update(recovered)
                    result["generation"]["usage_source"] = "codex_rollout"
                else:
                    result["generation"]["compactions"] = recovered.get("compactions")
        else:
            result["generation"] = generation or {
                "runtime": "claude",
                "errors": ["the wall clock expired before the agent started"],
                "tool_calls": {},
            }
        result["finish"] = arm_toolkit.finish
        result["generation"]["sandbox_flags"] = arm_toolkit.sandbox_flags
        if arm_toolkit.sandbox_flags:
            result["notes"].append(
                f"{len(arm_toolkit.sandbox_flags)} cell(s) raised sandbox audit flags; "
                "see tool_turns.json"
            )
        result["environment_success"] = bool(backend.solved())
        if (
            status == "completed"
            and arm_toolkit.turn_budget_exhausted
            and not result["environment_success"]
        ):
            result["status"] = "call_budget"
            result["reason"] = f"call budget exhausted ({max_turns})"
        elif (
            status == "completed"
            and result["generation"].get("max_turns_hit")
            and not result["environment_success"]
        ):
            # Claude Code's own turn ceiling (decision budget + allowance,
            # every model request counts) ended the run before finish.
            result["status"] = "call_budget"
            result["reason"] = f"sdk turn ceiling exhausted ({sdk_max_turns(max_turns)})"
        if (
            result["status"] == "completed"
            and result["generation"].get("errors")
            and arm_toolkit.finish is None
            and not result["environment_success"]
        ):
            # The provider dropped the session (overload, 429, transport) and
            # the agent never got to finish: an infrastructure failure, to be
            # rerun rather than counted against the arm.
            result["status"] = "planner_error"
            result["reason"] = "provider error mid-episode"
        result["env_steps_total"] = backend.env_steps()
        if status == "completed" and returncode != 0:
            result["notes"].append(f"{runtime} exited {returncode}")
        if result["generation"].get("errors"):
            result["notes"].append(f"{runtime} reported errors; see {events_path.name}")
        if result["generation"].get("compactions"):
            result["notes"].append(
                f"{result['generation']['compactions']} context compaction(s): older turns were "
                "replaced by a model-written summary"
            )
        if result["generation"].get("shell_commands"):
            result["notes"].append(
                f"{len(result['generation']['shell_commands'])} shell command(s) used; audit"
            )
    except BaseException as exc:  # noqa: BLE001 - the result records the failure
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if result["status"] == "boot_error":
            result["notes"].append("host failed before the agent started")
        else:
            result["status"] = "error"
    finally:
        result["calls"] = len(ledger)
        result["stateful_calls"] = ledger.stateful_calls()
        result["env_steps"] = ledger.env_steps()
        if arm_toolkit is not None:
            result["tool_turns"] = len(arm_toolkit.turns)
            (out_dir / "tool_turns.json").write_text(
                json.dumps(arm_toolkit.turns, indent=2, ensure_ascii=False) + "\n"
            )
        ledger.write_jsonl(out_dir / "calls.jsonl")
        if server is not None:
            with contextlib.suppress(Exception):
                server.stop()
        stop_all(toolkit, daemons)
        result["total_wall_s"] = time.perf_counter() - started
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--robot", default="libero", choices=ROBOTS)
    known, _ = pre.parse_known_args(argv)
    adapter = get_adapter(known.robot)
    host = adapter.host_module()
    parser = argparse.ArgumentParser(prog="python -m pyrualean.play", description=__doc__)
    parser.add_argument("--robot", default="libero", help="libero | robocasa | robotwin")
    parser.add_argument("--arm", choices=ARMS, required=True)
    host.add_cell_args(parser)
    host.add_boot_args(parser)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--rpent-root", default=None)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning", default="xhigh")
    parser.add_argument("--budget-s", type=float, default=1200.0)
    parser.add_argument("--max-programs", type=int, default=None)
    parser.add_argument(
        "--codex-bin",
        default=setting("CODEX_BIN", "codex"),
        help="Codex executable; use the real binary, not a wrapper that forces CODEX_HOME",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument(
        "--images",
        choices=("on-motion", "on-demand", "none"),
        default="on-motion",
        help="when camera images are attached to tool results",
    )
    parser.add_argument(
        "--feedback",
        choices=("pure", "rich"),
        default="rich",
        help="pure: only what the code output (python-tool semantics); rich: plus state and ledger",
    )
    parser.add_argument(
        "--codex-home",
        default=setting("CODEX_HOME"),
        help="clean CODEX_HOME for the agent (default: $PYRUALEAN_CODEX_HOME)",
    )
    parser.add_argument(
        "--guides",
        action="store_true",
        help="copy the robot's operating guides (RPent's) into the workspace and mention them",
    )
    parser.add_argument(
        "--guides-dir",
        default=None,
        help="directory holding the guides (default: the robot's guides dir in RPent's checkout)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="decision budget: at most this many tool calls (budget-s becomes a safety ceiling)",
    )
    parser.add_argument(
        "--runtime",
        choices=RUNTIMES,
        default=setting("RUNTIME", "codex"),
        help="agent runtime: codex (Codex CLI) or claude (Claude Code via the Claude Agent SDK)",
    )
    parser.add_argument(
        "--claude-cli",
        default=setting("CLAUDE_CLI", "claude"),
        help="claude executable for --runtime claude (default: $PYRUALEAN_CLAUDE_CLI or PATH)",
    )
    parser.add_argument(
        "--primitives",
        choices=PRIMITIVE_SETS,
        default=setting("PRIMITIVES", "full"),
        help="primitive set: full, or no-vla (the VLA primitives hidden and refused)",
    )
    args = parser.parse_args(argv)
    cell = host.cell_from_args(args)
    result = play_episode(
        cell,
        args.out.resolve(),
        arm=args.arm,
        model=args.model,
        reasoning_effort=args.reasoning,
        budget_s=args.budget_s,
        max_programs=args.max_programs,
        codex_bin=args.codex_bin,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        rpent_root=args.rpent_root,
        images=args.images,
        feedback=args.feedback,
        codex_home=args.codex_home,
        guides=args.guides,
        guides_dir=args.guides_dir,
        max_turns=args.max_turns,
        robot=known.robot,
        boot_options=host.boot_options(args),
        runtime=args.runtime,
        claude_cli=args.claude_cli,
        primitives=args.primitives,
    )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "arm",
                    "status",
                    "environment_success",
                    "finish",
                    "tool_turns",
                    "stateful_calls",
                    "env_steps",
                    "policy_wall_s",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] not in ("boot_error", "error") else 2


if __name__ == "__main__":
    raise SystemExit(main())
