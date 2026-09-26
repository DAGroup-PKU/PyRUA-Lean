# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Read-only parser for RPent run artifacts (the tool-calling arm).

Reads ``transcript_*.json`` and ``states.json`` without importing RPent.
Native success is the final recorded ``terminated`` flag; the agent's own
``finish.status`` is kept separately as ``status``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import EpisodeMetrics, env_steps_from_states


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _single_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern} under {directory}; found {len(matches)}")
    return matches[0]


def parse_rpent_run(
    run_dir: str | Path,
    *,
    transcript: str | Path | None = None,
    states: str | Path | None = None,
    backend: str = "libero",
    task: str | None = None,
    seed: int | None = None,
) -> EpisodeMetrics:
    """Parse one RPent output directory into :class:`EpisodeMetrics`."""

    root = Path(run_dir).expanduser().resolve()
    transcript_path = Path(transcript) if transcript else _single_match(root, "transcript_*.json")
    if not transcript_path.is_absolute():
        transcript_path = root / transcript_path
    record = _load_json(transcript_path)
    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    finish = record.get("finish") if isinstance(record.get("finish"), dict) else {}

    states_path = Path(states) if states is not None else root / "states.json"
    if not states_path.is_absolute():
        states_path = root / states_path
    steps: list[dict[str, Any]] = []
    if states_path.is_file():
        raw_steps = _load_json(states_path).get("steps")
        if isinstance(raw_steps, list):
            steps = [s for s in raw_steps if isinstance(s, dict)]

    notes: list[str] = []
    terminated = [s.get("terminated") for s in steps]
    if terminated and terminated[-1] is not None:
        environment_success: bool | None = bool(any(bool(t) for t in terminated))
    else:
        environment_success = None
        notes.append("no states.json: native outcome unknown")
    if finish.get("status") == "success" and environment_success is not True:
        notes.append("agent claimed success but the environment did not terminate")
    status = str(finish["status"]) if "status" in finish else None
    if status is None and steps:
        # The planner session ended without calling ``finish`` (timeout,
        # provider error, crash): the episode still counts, with its outcome
        # taken from the environment.
        status = "no_finish"
        notes.append("planner session ended without a finish call")

    if task is None:
        # LIBERO transcripts name the cell suite/task, RoboTwin's task_config/
        # task_name, RoboCasa's split/task_name; all become "<suite>:<task>",
        # the code arm's key.
        suite = record.get("suite", record.get("task_config", record.get("split")))
        task_id = record.get("task", record.get("task_name"))
        task = f"{suite}:{task_id}" if suite is not None else str(task_id)
    if seed is None:
        for key in ("seed", "requested_seed"):
            if isinstance(record.get(key), int):
                seed = record[key]
                break
    if backend == "libero" and isinstance(record.get("env"), str):
        backend = record["env"]
    elif backend == "libero" and "split" in record and "suite" not in record:
        # RoboCasa365 transcripts carry the Target50 split/task_name and no env.
        backend = "robocasa"

    input_tokens = _int(stats.get("total_input_tokens"))
    output_tokens = _int(stats.get("total_output_tokens"))
    cached_tokens = _int(stats.get("total_cached_input_tokens"))
    cache_read = _int(stats.get("total_cache_read_input_tokens"))
    if cache_read is not None:
        # Claude Agent SDK stats keep cached prompt tokens out of input_tokens;
        # the Codex stats and the code arm count them inside it (and report
        # the cached share separately), so normalise to that convention.
        cache_create = _int(stats.get("total_cache_creation_input_tokens")) or 0
        input_tokens = (input_tokens or 0) + cache_read + cache_create
        cached_tokens = cache_read
        exact = claude_stream_usage(root)
        if exact is not None:
            # RPent's recorder adds the usage of every message the SDK streams,
            # and one response arrives as one message per content block with the
            # same usage, so its stats over-count; the stream deduplicated by
            # message id is exact.
            input_tokens = exact["input"] + exact["cache_read"] + exact["cache_creation"]
            cached_tokens = exact["cache_read"]
            output_tokens = exact["output"]
            notes.append("tokens recomputed from the SDK stream (unique model requests)")
            if not exact["exact_output"]:
                notes.append(
                    "output tokens are the stream's partial snapshots (lower bound): "
                    "no Claude Code transcript found"
                )
    return EpisodeMetrics(
        system="rpent",
        backend=backend,
        task=task,
        seed=seed,
        environment_success=environment_success,
        status=status,
        calls=_int(stats.get("tool_calls")),
        stateful_calls=sum(1 for s in steps if _moves(s)),
        env_steps=env_steps_from_states(steps) if steps else None,
        turns=_int(stats.get("turns_used")),
        wall_time_s=_float(stats.get("elapsed_s", record.get("elapsed_s"))),
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        reasoning_output_tokens=_int(stats.get("total_reasoning_output_tokens")),
        total_tokens=(
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        ),
        source=str(transcript_path),
        notes=tuple(notes),
    )


def claude_stream_usage(run_dir: Path) -> dict[str, int] | None:
    """Token usage of a Claude Agent SDK run, one count per model request.

    Reads ``claude_*.stream.jsonl`` (every SDK message as JSON) and takes each
    assistant message id once (blocks of one response stream with the same
    usage).  Prompt tokens are exact there; the stream's ``output_tokens`` are
    partial snapshots, so the output count comes from Claude Code's own
    transcript of the session when one is found (``claude_*.transcript.jsonl``
    beside the stream, else ``~/.claude/projects/*/<session>.jsonl``) and
    ``exact_output`` says whether it was.  ``None`` when the run has no stream.
    """

    streams = sorted(Path(run_dir).glob("claude_*.stream.jsonl"))
    if not streams:
        return None
    per_message: dict[str, dict[str, int]] = {}
    session_id = ""
    for line in streams[0].read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not session_id and isinstance(event.get("session_id"), str):
            session_id = event["session_id"]
        if event.get("type") != "AssistantMessage" or event.get("parent_tool_use_id"):
            continue
        message_id = str(event.get("message_id") or event.get("uuid") or "")
        usage = event.get("usage") or {}
        if not message_id or not isinstance(usage, dict):
            continue
        entry = per_message.setdefault(
            message_id, {"input": 0, "cache_read": 0, "cache_creation": 0, "output": 0}
        )
        entry["input"] = int(usage.get("input_tokens") or 0)
        entry["cache_read"] = int(usage.get("cache_read_input_tokens") or 0)
        entry["cache_creation"] = int(usage.get("cache_creation_input_tokens") or 0)
        entry["output"] = max(entry["output"], int(usage.get("output_tokens") or 0))
    if not per_message:
        return None
    exact_output = claude_transcript_output(run_dir, session_id)
    for message_id, output in exact_output.items():
        if message_id in per_message:
            per_message[message_id]["output"] = output
    keys = ("input", "cache_read", "cache_creation", "output")
    totals = {key: sum(entry[key] for entry in per_message.values()) for key in keys}
    totals["requests"] = len(per_message)
    totals["exact_output"] = int(bool(exact_output))
    return totals


def claude_transcript_output(run_dir: Path, session_id: str = "") -> dict[str, int]:
    """Final output tokens per assistant message id from Claude Code's transcript.

    Looks for ``claude_*.transcript.jsonl`` under ``run_dir`` (a snapshot of
    the session transcript kept with the run), then for the live session file
    under ``~/.claude/projects``.  Empty when neither exists.
    """

    candidates = sorted(Path(run_dir).glob("claude_*.transcript.jsonl"))
    if not candidates and session_id:
        projects = Path.home() / ".claude" / "projects"
        candidates = sorted(projects.glob(f"*/{session_id}.jsonl")) if projects.is_dir() else []
    if not candidates:
        return {}
    output: dict[str, int] = {}
    for line in candidates[0].read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "assistant" or entry.get("isSidechain"):
            continue
        message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        message_id = str(message.get("id") or "")
        if message_id:
            seen = output.get(message_id, 0)
            output[message_id] = max(seen, int(usage.get("output_tokens") or 0))
    return output


def _moves(step: dict[str, Any]) -> bool:
    """A recorded step that moved the robot (RoboTwin also records the reset)."""

    command = step.get("command")
    return bool(command) and (command or {}).get("action") != "reset"


def _int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["claude_stream_usage", "claude_transcript_output", "parse_rpent_run"]
