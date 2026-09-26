# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Ask a model for a policy file (single shot) through the Codex CLI.

The generation is deliberately tool-free: the prompt tells the model to
answer directly, the Codex session runs read-only in an empty scratch
directory, and every event Codex emits is kept (``codex_events.jsonl``) so a
run can prove that no command or file read happened during generation.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ._env import setting
from .prompt import PromptBundle, TaskCard, render_prompt

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
#: Codex 0.155.1's bundled model catalog with only its ``gpt-6-astra`` entry set to the
#: values of Codex's fallback metadata (docs/guide.md, "Codex code mode").
CODEX_CATALOG = Path(__file__).resolve().with_name("codex-catalog-gpt-6-astra-fallback.json")
NO_TOOLS_NOTE = (
    "\n\nAnswer directly in this reply: do not run commands, do not read or "
    "write files, do not explore the filesystem.  Everything you need is in "
    "this prompt and the attached images."
)


def extract_python_block(text: str) -> str:
    """Return the largest fenced Python block in ``text``."""

    blocks = _FENCE.findall(text)
    if not blocks:
        raise ValueError("no fenced ```python block in the model response")
    return max(blocks, key=len).strip() + "\n"


def _usage_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    tool_events = 0
    turns = 0
    for event in events:
        kind = str(event.get("type", ""))
        if kind.endswith("turn.completed"):
            turns += 1
        if any(word in kind for word in ("command", "tool", "mcp", "file_change", "patch")):
            tool_events += 1
        for key in ("usage", "token_usage"):
            block = event.get(key)
            if isinstance(block, dict):
                usage = block
        item = event.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type", ""))
            if any(
                word in item_type for word in ("command", "tool", "mcp", "file_change", "patch")
            ):
                tool_events += 1
    out: dict[str, Any] = {"turns": turns or None, "tool_events": tool_events}
    for target, names in (
        ("input_tokens", ("input_tokens", "prompt_tokens")),
        ("cached_input_tokens", ("cached_input_tokens", "cached_tokens")),
        ("output_tokens", ("output_tokens", "completion_tokens")),
        ("reasoning_output_tokens", ("reasoning_output_tokens", "reasoning_tokens")),
        ("total_tokens", ("total_tokens",)),
    ):
        for name in names:
            if usage.get(name) is not None:
                out[target] = int(usage[name])
                break
    return out


def _retries() -> int:
    """Codex request/stream retries; large values let an episode wait out a network outage."""

    return int(setting("CODEX_RETRIES", "10"))


def provider_overrides(
    base_url: str | None,
    api_key_env: str | None,
    *,
    provider_id: str = "pyrualean",
    wire_api: str = "responses",
) -> list[str]:
    """Codex ``-c`` overrides that point the CLI at a custom OpenAI-compatible endpoint.

    Mirrors RPent's own Codex planner: a named ``model_providers`` entry with a
    Responses-style ``wire_api`` and an ``env_key`` naming the environment
    variable that holds the API key.  Returns ``[]`` when no base URL is given.
    """

    if not base_url:
        return []
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    settings = [
        ("model_provider", provider_id),
        (f"model_providers.{provider_id}.name", provider_id),
        (f"model_providers.{provider_id}.base_url", normalized),
        (f"model_providers.{provider_id}.wire_api", wire_api),
    ]
    if api_key_env:
        settings.append((f"model_providers.{provider_id}.env_key", api_key_env))
    settings.append((f"model_providers.{provider_id}.request_max_retries", _retries()))
    settings.append((f"model_providers.{provider_id}.stream_max_retries", _retries()))
    out: list[str] = []
    for key, value in settings:
        out += ["-c", f"{key}={json.dumps(value)}"]
    return out


def codex_catalog_flags(model: str) -> list[str]:
    """``-c model_catalog_json=<CODEX_CATALOG>`` when Codex would look ``model`` up as
    ``gpt-6-astra``, else ``[]`` (Codex's own catalog).

    Codex takes the entry whose slug the id starts with, or failing that, the one the
    id matches after a single leading ``namespace/``: ``gpt-6-astra`` and
    ``openai/gpt-6-astra`` get the catalog, ``openai/openai/gpt-6-astra`` (the
    comparison's gateway id, already on fallback metadata) and other models do not.
    """

    namespace, sep, rest = model.partition("/")
    unprefixed = rest if sep and "/" not in rest and re.fullmatch(r"[\w-]+", namespace) else model
    if not (model.startswith("gpt-6-astra") or unprefixed.startswith("gpt-6-astra")):
        return []
    return ["-c", f"model_catalog_json={json.dumps(str(CODEX_CATALOG))}"]


def generate_with_codex(
    card: TaskCard,
    out_dir: Path,
    *,
    model: str = "gpt-6-astra",
    reasoning_effort: str = "xhigh",
    budget_s: int = 1200,
    codex_bin: str = "codex",
    timeout_s: float = 3600.0,
    bundle: PromptBundle | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Generate ``policy.py`` under ``out_dir`` and return the generation record.

    ``base_url`` / ``api_key_env`` route the Codex CLI to a custom
    OpenAI-compatible gateway (the key itself is never written to disk).
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = bundle or render_prompt(card, budget_s=budget_s)
    prompt_text = bundle.text() + NO_TOOLS_NOTE
    (out_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    last_message = out_dir / "response.md"
    events_path = out_dir / "codex_events.jsonl"
    record: dict[str, Any] = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "codex_bin": codex_bin,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "mode": "transfer" if card.transfer else "per_cell",
        "card": {"suite": card.suite, "task": card.task, "seed": card.seed},
        "prompt_chars": len(prompt_text),
        "images": list(bundle.images),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with tempfile.TemporaryDirectory(prefix="pyrualean-gen-") as scratch:
        argv = [
            codex_bin,
            "exec",
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            *codex_catalog_flags(model),
            *provider_overrides(base_url, api_key_env),
            "-s",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "-C",
            scratch,
            "--json",
            "-o",
            str(last_message),
        ]
        for image in bundle.images:
            argv += ["-i", image]
        argv.append("-")
        started = time.perf_counter()
        proc = subprocess.run(  # noqa: S603 - trusted local CLI
            argv,
            input=prompt_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        record["wall_s"] = time.perf_counter() - started
        record["returncode"] = proc.returncode
    events: list[dict[str, Any]] = []
    with events_path.open("w", encoding="utf-8") as handle:
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            handle.write(line + "\n")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    (out_dir / "codex_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    record.update(_usage_from_events(events))
    response = last_message.read_text(encoding="utf-8") if last_message.is_file() else ""
    if not response:
        record["error"] = f"codex exited {proc.returncode} without a final message"
        (out_dir / "generation.json").write_text(json.dumps(record, indent=2) + "\n")
        raise RuntimeError(record["error"])
    try:
        code = extract_python_block(response)
    except ValueError as exc:
        record["error"] = str(exc)
        (out_dir / "generation.json").write_text(json.dumps(record, indent=2) + "\n")
        raise
    policy_path = out_dir / "policy.py"
    policy_path.write_text(code, encoding="utf-8")
    record["policy"] = str(policy_path)
    record["policy_lines"] = code.count("\n")
    (out_dir / "generation.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record


__all__ = [
    "CODEX_CATALOG",
    "codex_catalog_flags",
    "extract_python_block",
    "generate_with_codex",
    "provider_overrides",
]
