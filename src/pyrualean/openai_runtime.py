# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""A plain OpenAI chat-completions agent loop for the code arms (``--runtime openai``).

The Codex and Claude runtimes hand the agent loop to a coding-agent product
with its own system prompt, tools and transcript format.  This runtime is the
third option: the loop lives here, talks to *any* OpenAI-compatible
``/v1/chat/completions`` endpoint (a gateway, a local SGLang / vLLM server, a
training framework's rollout engine) with function calling, and executes the
arm's tools in-process through :class:`pyrualean.arms.ArmToolkit` (no MCP
server, no subprocess).  It is what a small open model, a distillation
teacher and an RL rollout all share.

Protocol (one episode):

1. The first user message carries the full prompt text and the task card
   images as ``image_url`` data URIs (same content Codex / Claude get).
2. Each assistant reply may carry ``tool_calls``; they are executed in order
   through ``ArmToolkit.execute_tool`` and answered with one ``tool`` message
   each.  Images returned by a tool (``on-motion`` or ``robo.show()``) cannot
   travel in a ``tool`` message under the OpenAI schema, so they follow in one
   extra ``user`` message that names the tool call they belong to.
3. The episode ends when the agent calls ``finish``, when it replies without
   a tool call (after ``nudges`` reminders), when the decision budget is
   spent (``ArmToolkit`` refuses further ``python`` calls), when the wall
   clock ``budget_s`` runs out, or when the provider keeps failing.

Everything the model saw and said is written to ``messages.json`` (images
replaced by references to PNG files under ``images/``), every request to
``openai_events.jsonl``, and the usual ``generation`` record (Codex's token
conventions: ``input_tokens`` includes cached tokens) goes into
``result.json`` by :mod:`pyrualean.play`.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .arms import ArmResult, ArmToolkit

#: Retries of one chat request on transport / 5xx / 429 errors (env override).
DEFAULT_RETRIES = int(os.environ.get("PYRUALEAN_OPENAI_RETRIES", "10"))
#: A rate limit (HTTP 429) is waited out, not counted against ``retries``: backoff doubling from
#: 5 s to RATE_LIMIT_MAX_WAIT_S, with jitter, for as long as the episode's own time allows.  With
#: ~40 episodes on one gateway key, 39 of 162 episodes ended at their first request after ten
#: 429s two to twenty seconds apart.
RATE_LIMIT_MAX_WAIT_S = float(os.environ.get("PYRUALEAN_RATE_LIMIT_MAX_WAIT_S", "120"))
#: Seconds one chat request may take (a 40-call episode on a slow model needs it).
DEFAULT_TIMEOUT_S = float(os.environ.get("PYRUALEAN_OPENAI_TIMEOUT_S", "900"))
#: Reminders sent when the model replies without a tool call before the episode is over.
DEFAULT_NUDGES = 2
NUDGE_TEXT = (
    "The episode is not over: the simulator has not reported success and you have not "
    "called `finish`. Continue by calling the `python` tool, or call `finish`."
)
#: One message carries every image the turn produced; ``sources`` names the calls they came from.
IMAGE_NOTE = "Images returned by tool call {sources}; they are not part of the text result."

#: ``post(url, json_body, timeout_s) -> (status_code, parsed_json_or_text)``.
Transport = Callable[[str, dict[str, Any], float], tuple[int, Any]]


class OpenAIRuntimeError(RuntimeError):
    """The provider failed for good (after the retries)."""


# -- images ---------------------------------------------------------------------


def _resize_png(data: bytes, max_side: int) -> tuple[bytes, float]:
    """Downscale ``data`` so its longer side is ``max_side``; returns (png, scale)."""

    if not max_side:
        return data, 1.0
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    longest = max(image.size)
    if longest <= max_side:
        return data, 1.0
    scale = max_side / longest
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    )
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    return buffer.getvalue(), scale


def image_part(data: bytes) -> dict[str, Any]:
    """An OpenAI ``image_url`` content part holding a PNG as a data URI."""

    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")},
    }


def openai_tools(spec: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the arm's MCP-style tool specs (``input_schema``) to OpenAI ``tools``."""

    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in spec
    ]


# -- transport ------------------------------------------------------------------


def httpx_transport(api_key: str | None) -> Transport:
    """The default transport: one ``httpx`` client, bearer auth, JSON in and out."""

    import httpx

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    client = httpx.Client(headers=headers, timeout=None, trust_env=True)

    def post(url: str, body: dict[str, Any], timeout_s: float) -> tuple[int, Any]:
        response = client.post(url, json=body, timeout=timeout_s)
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        return response.status_code, payload

    return post


# -- the loop -------------------------------------------------------------------


class _Transcript:
    """The messages the model sees, plus a copy with images swapped for file references."""

    def __init__(self, out_dir: Path) -> None:
        self.messages: list[dict[str, Any]] = []  # what is sent to the model
        self.recorded: list[dict[str, Any]] = []  # same, images as {"image_ref": path}
        self.images_dir = out_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._image_index = 0

    def _save_image(self, data: bytes, name: str) -> str:
        self._image_index += 1
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        path = self.images_dir / f"{self._image_index:03d}_{safe}"
        if not path.suffix:
            path = path.with_suffix(".png")
        path.write_bytes(data)
        return str(path)

    def add(self, message: dict[str, Any], images: list[tuple[str, bytes]] | None = None) -> None:
        """Append ``message``; ``images`` (name, png) become content parts of a user message."""

        sent = dict(message)
        recorded = dict(message)
        if images:
            parts_sent: list[dict[str, Any]] = []
            parts_recorded: list[dict[str, Any]] = []
            text = message.get("content")
            if isinstance(text, str) and text:
                parts_sent.append({"type": "text", "text": text})
                parts_recorded.append({"type": "text", "text": text})
            elif isinstance(text, list):
                parts_sent.extend(text)
                parts_recorded.extend(text)
            for name, data in images:
                parts_sent.append(image_part(data))
                parts_recorded.append(
                    {"type": "image_ref", "image_ref": self._save_image(data, name)}
                )
            sent["content"] = parts_sent
            recorded["content"] = parts_recorded
        self.messages.append(sent)
        self.recorded.append(recorded)


def _usage_fields(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    reasoning = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
    return {"prompt": prompt, "completion": completion, "cached": cached, "reasoning": reasoning}


def _tool_result_text(result: ArmResult) -> str:
    """The text block of a tool result, exactly what the MCP server would send."""

    return result.content_blocks[0]["text"]


def _execute_with_deadline(
    arm_toolkit: ArmToolkit,
    name: str,
    arguments: dict[str, Any],
    remaining_s: float,
    on_timeout: Callable[[], None] | None,
) -> ArmResult:
    """Run one tool call, stopping the episode if it is still running at the deadline.

    A cell can loop over primitives for minutes; this loop *is* the host thread, so without a
    timer the wall-clock budget would only be noticed after the cell returned (the Codex and
    Claude runtimes stop the robot from the host while the agent process keeps running).  The
    timer fires ``on_timeout``, which makes the next primitive raise ``EpisodeFinished``.
    """

    if on_timeout is None:
        return arm_toolkit.execute_tool(name, arguments)
    if remaining_s <= 0:
        # Already out of budget: stop first, so the cell's very first primitive raises instead of
        # running unbounded (a timer with a non-positive delay is the same as none at all).
        on_timeout()
        return arm_toolkit.execute_tool(name, arguments)
    timer = threading.Timer(remaining_s, on_timeout)
    timer.daemon = True
    timer.start()
    try:
        return arm_toolkit.execute_tool(name, arguments)
    finally:
        timer.cancel()


def _parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw, dict):
        return raw, None
    if raw is None or raw == "":
        return {}, None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return None, f"tool arguments are not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "tool arguments must be a JSON object"
    return parsed, None


def run_openai_agent(
    prompt_text: str,
    image_paths: list[str | Path],
    arm_toolkit: ArmToolkit,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    max_turns: int | None,
    budget_s: float,
    out_dir: str | Path,
    arm: str = "cells",
    temperature: float | None = None,
    max_tokens: int | None = 4096,
    extra_body: dict[str, Any] | None = None,
    tool_choice: str = "auto",
    image_max_side: int = 0,
    nudges: int = DEFAULT_NUDGES,
    retries: int = DEFAULT_RETRIES,
    request_timeout_s: float = DEFAULT_TIMEOUT_S,
    transport: Transport | None = None,
    on_timeout: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run one episode's agent loop; returns the ``generation`` record.

    ``status`` is ``completed`` (the agent finished, stopped calling tools, or
    spent its decision budget), ``timeout`` (``budget_s`` expired:
    ``on_timeout`` is called first) or ``error`` (the provider failed after
    ``retries``); ``returncode`` mirrors the Claude runtime: 0 clean, 2 after
    a provider error, ``None`` on timeout.  ``extra_body`` is merged into
    every request (``reasoning_effort``, ``chat_template_kwargs`` ...).
    ``tool_choice`` is OpenAI's: ``auto`` lets the model answer in text,
    ``required`` makes every reply a tool call (``python`` or ``finish``),
    which small thinking models need (they otherwise think and emit nothing).
    ``image_max_side`` downscales every image the model receives (0 keeps the
    native 1024 px that the prompt's pixel coordinates refer to).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "openai_events.jsonl"
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    post = transport or httpx_transport(api_key)
    # The deadline can be noticed by the tool timer and again by the loop; the robot is stopped
    # once either way, so make the hook itself fire once (the Claude runtime's contract).
    timeout_fired = False

    def stop_once() -> None:
        nonlocal timeout_fired
        if timeout_fired or on_timeout is None:
            return
        timeout_fired = True
        on_timeout()

    tools = openai_tools(arm_toolkit.get_tools_spec())
    transcript = _Transcript(out_dir)
    started = time.perf_counter()

    def event(kind: str, **fields: Any) -> None:
        with open(events_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"t": round(time.perf_counter() - started, 3), "type": kind, **fields},
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    # 1. The prompt + the card images, as one user message.
    card_images: list[tuple[str, bytes]] = []
    scales: list[float] = []
    for path in image_paths:
        data, scale = _resize_png(Path(path).read_bytes(), image_max_side)
        card_images.append((Path(path).name, data))
        scales.append(scale)
    text = prompt_text
    if any(s != 1.0 for s in scales):
        factor = min(scales)
        text += (
            f"\n\nNote: every image you receive is downscaled by {factor:.3f} "
            "(multiply pixel coordinates you read off an image by "
            f"{1 / factor:.3f} before using them as 1024x1024 coordinates)."
        )
    transcript.add({"role": "user", "content": text}, card_images)

    totals = {"prompt": 0, "completion": 0, "cached": 0, "reasoning": 0}
    request_profile: list[int] = []
    request_latency: list[float] = []
    tool_calls: dict[str, int] = {}
    tool_requests: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    requests = 0
    nudges_left = max(0, nudges)
    malformed_streak = 0
    max_malformed_streak = 3
    status = "completed"
    returncode: int | None = 0
    last_text = ""
    finish_reasons: list[str | None] = []
    #: Replies allowed after the decision budget ended the episode on the robot
    #: (the refused call told the model to call ``finish``; give it one reply to do so).
    post_budget_replies = 1

    def remaining() -> float:
        return budget_s - (time.perf_counter() - started)

    def chat() -> dict[str, Any] | None:
        """One chat request with retries; ``None`` when the provider is down for good."""

        nonlocal requests
        body: dict[str, Any] = {
            "model": model,
            "messages": transcript.messages,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:  # 0 / None: do not send the field at all (o-series reject max_tokens)
            body["max_tokens"] = max_tokens
        if extra_body:
            body.update(extra_body)
        attempt = 0
        limited = 0
        while True:
            attempt += 1
            left = remaining()
            if left <= 0:
                return None
            t0 = time.perf_counter()
            try:
                code, payload = post(url, body, min(request_timeout_s, left))
            except Exception as exc:  # noqa: BLE001 - transport errors are retried
                code, payload = 0, f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - t0
            requests += 1
            ok = code == 200 and isinstance(payload, dict) and payload.get("choices")
            event(
                "request",
                attempt=attempt,
                status_code=code,
                latency_s=round(latency, 3),
                messages=len(transcript.messages),
                ok=bool(ok),
                error=None if ok else str(payload)[:2000],
            )
            if ok:
                request_latency.append(round(latency, 3))
                return payload
            message = f"request {requests} attempt {attempt}: HTTP {code}: {str(payload)[:500]}"
            if code == 429:
                # rate limited: wait it out (only the episode's time limit ends this)
                if limited < 5:
                    warnings.append(message)
                limited += 1
                attempt -= 1
                wait = min(RATE_LIMIT_MAX_WAIT_S, 5.0 * 2.0 ** min(limited - 1, 5))
                sleep(max(0.0, min(wait * (0.5 + random.random()), remaining())))
                continue
            # 4xx other than 408/429 is our fault (bad request, context overflow): no retry.
            retryable = code in (0, 408, 409, 425) or code >= 500
            if not retryable or attempt > retries:
                # Fatal: the episode ends on a provider failure (``errors`` is
                # what play.py judges as planner_error).
                errors.append(message)
                return None
            # Recovered by a retry: recorded, but not held against the episode.
            warnings.append(message)
            sleep(max(0.0, min(60.0, 2.0 * attempt, remaining())))

    try:
        while True:
            if remaining() <= 0:
                status, returncode = "timeout", None
                break
            if arm_toolkit.turn_budget_exhausted and arm_toolkit.finish is None:
                if post_budget_replies <= 0:
                    break
                post_budget_replies -= 1
            response = chat()
            if response is None:
                if remaining() <= 0:
                    status, returncode = "timeout", None
                else:
                    status, returncode = "error", 2
                break
            usage = _usage_fields(response.get("usage"))
            for key in totals:
                totals[key] += usage[key]
            request_profile.append(totals["prompt"])  # cumulative, as the Claude runtime reports it
            choice = response["choices"][0]
            finish_reasons.append(choice.get("finish_reason"))
            message = choice.get("message") or {}
            calls = message.get("tool_calls") or []
            content = message.get("content")
            # A reply with no tool calls must replay with a string content: providers reject
            # {"role": "assistant", "content": null} without tool_calls (a thinking model that
            # spent its whole budget reasoning produces exactly that).
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": content if (calls or isinstance(content, str)) else "",
            }
            if calls:
                assistant["tool_calls"] = calls
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            recorded_extra = {"reasoning_content": reasoning} if reasoning else {}
            transcript.messages.append(assistant)
            transcript.recorded.append({**assistant, **recorded_extra})
            if isinstance(assistant["content"], str) and assistant["content"]:
                last_text = assistant["content"]
            event(
                "assistant",
                tool_calls=[c.get("function", {}).get("name") for c in calls],
                text_chars=len(assistant["content"] or "")
                if isinstance(assistant["content"], str)
                else 0,
                reasoning_chars=len(reasoning or ""),
                finish_reason=choice.get("finish_reason"),
                usage=usage,
            )
            if not calls:
                if tool_choice == "required":
                    warnings.append(
                        f"request {requests}: no tool call despite tool_choice=required "
                        f"(finish_reason={choice.get('finish_reason')!r})"
                    )
                done = (
                    arm_toolkit.finish is not None
                    or arm_toolkit.robo._reported_success()
                    or arm_toolkit.turn_budget_exhausted
                )
                if done or nudges_left == 0:
                    break
                nudges_left -= 1
                transcript.add({"role": "user", "content": NUDGE_TEXT})
                event("nudge", left=nudges_left)
                continue
            # Images cannot ride in a `tool` message, and a `tool` message may not be separated
            # from its `tool_calls` by another role, so every call's images wait until the last
            # tool reply has been appended and then travel together in one `user` message.
            pending_images: list[tuple[str, bytes]] = []
            image_sources: list[str] = []
            malformed = 0
            for index, call in enumerate(calls):
                function = call.get("function") or {}
                name = function.get("name") or ""
                call_id = call.get("id") or f"call_{requests}_{index}"
                call["id"] = call_id  # the replayed assistant message must carry the id we answer
                tool_calls[name] = tool_calls.get(name, 0) + 1
                arguments, problem = _parse_arguments(function.get("arguments"))
                t0 = time.perf_counter()
                if problem:
                    malformed += 1
                    result = ArmResult({"error": problem})
                else:
                    result = _execute_with_deadline(
                        arm_toolkit, name, arguments or {}, remaining(), stop_once
                    )
                elapsed = time.perf_counter() - t0
                tool_requests.append(
                    {"request": requests, "tool": name, "elapsed_s": round(elapsed, 3)}
                )
                transcript.add(
                    {"role": "tool", "tool_call_id": call_id, "content": _tool_result_text(result)}
                )
                names = result.result.get("images") or []
                call_images: list[tuple[str, bytes]] = []
                for position, data in enumerate(result.images):
                    label = names[position] if position < len(names) else f"image_{position}"
                    png, _ = _resize_png(data, image_max_side)
                    call_images.append(
                        (f"{label}.png" if not label.endswith(".png") else label, png)
                    )
                if call_images:
                    image_sources.append(
                        f"{call_id} ({', '.join(name for name, _ in call_images)})"
                    )
                    pending_images.extend(call_images)
                event(
                    "tool",
                    tool=name,
                    call_id=call_id,
                    elapsed_s=round(elapsed, 3),
                    ok="error" not in result.result,
                    images=len(call_images),
                    terminated=arm_toolkit.robo._reported_success(),
                )
            if pending_images:
                transcript.add(
                    {
                        "role": "user",
                        "content": IMAGE_NOTE.format(sources="; ".join(image_sources)),
                    },
                    pending_images,
                )
            # `finish` only ends the episode when the toolkit actually recorded it: a call whose
            # arguments did not parse never reached the tool, and the agent must see the error.
            if arm_toolkit.finish is not None:
                break
            if malformed and malformed == len(calls):
                malformed_streak += 1
                if malformed_streak > max_malformed_streak:
                    warnings.append(
                        f"{malformed_streak} consecutive replies whose tool arguments did not parse"
                    )
                    status, returncode = "malformed", 2
                    break
            else:
                malformed_streak = 0
    finally:
        if status == "timeout":
            stop_once()
        (out_dir / "messages.json").write_text(
            json.dumps(transcript.recorded, indent=1, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (out_dir / "last_message.md").write_text(last_text or "", encoding="utf-8")

    return {
        "runtime": "openai",
        "model": model,
        "models": [model],
        "base_url": base_url,
        "input_tokens": totals["prompt"],
        "cached_input_tokens": totals["cached"],
        "output_tokens": totals["completion"],
        "reasoning_output_tokens": totals["reasoning"],
        "total_tokens": totals["prompt"] + totals["completion"],
        "requests": requests,
        "request_profile": request_profile,
        "request_latency_s": request_latency,
        "finish_reasons": finish_reasons,
        "turns": len(arm_toolkit.turns),
        "tool_calls": tool_calls,
        "tool_requests": tool_requests,
        "errors": errors,
        "warnings": warnings,
        "compactions": 0,
        "max_turns_hit": False,
        "status": status,
        "returncode": returncode,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "messages": len(transcript.recorded),
        "images_sent": transcript._image_index,
        "options": {
            "model": model,
            "max_turns": max_turns,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": extra_body or {},
            "tool_choice": tool_choice,
            "image_max_side": image_max_side,
            "nudges": nudges,
            "retries": retries,
            "request_timeout_s": request_timeout_s,
            "images": len(image_paths),
        },
    }


__all__ = [
    "DEFAULT_NUDGES",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT_S",
    "IMAGE_NOTE",
    "NUDGE_TEXT",
    "OpenAIRuntimeError",
    "httpx_transport",
    "image_part",
    "openai_tools",
    "run_openai_agent",
]
