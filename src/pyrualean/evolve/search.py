"""Dev-only reflection with read-only evidence access and task-local proposals."""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

from .digest import digest, render
from .evidence import READ_EVIDENCE_TOOL, DevEvidence
from .gate import _compare
from .library import LESSON_RE, SKILL_RE, Library, LibraryError, parse_lessons, validate

INSTRUCTIONS = """Reflect on this task's DEV episodes only. Use read_dev_evidence to inspect
catalogued logs, cells and images when the preview is insufficient. It is read-only, scoped
to this task's dev episodes; there is no shell, robot, val/test or arbitrary file access.
Separate observed facts from hypotheses. Do not invent image contents. An infrastructure
failure is not robot evidence. Do not use layout IDs, seeds, absolute object poses, hidden
scores, or simulator internals as policy inputs. Keep the robot/perception/scoring API frozen.
Return ONE JSON object: {"observed": "...", "hypothesis": "...", "validation": "...",
"proposals": [{"strategy": "short description of the approach",
"difference": "what changes and why", "expected": "observable
prediction", "evidence": ["dev episode IDs"], "files": {"tasks/TASK/lessons/NAME.md":
"complete file content", "tasks/TASK/skills/NAME.py": "complete file content"}}]}.
If no useful change is warranted, return "proposals": [] and a nonempty "no_change" reason.
Do not invent a modification merely to fill a round. Previous attempts and their publication
outcomes are provided as history, not as validated instructions. Learn from unsuccessful tries.
Each proposal solves one problem: at most one function, two Markdown files, 24000 characters.
Skills are short functions accepting robo explicitly, using public methods, returning
done/needs_observation/failed plus evidence. No external IO, classes, decorators, import-time
calls, recursion or attribute writes. Normal loops are allowed within the episode wall budget;
return control for observations rather than running the whole task blindly.
Imports: math/numpy/statistics or library.
Markdown cards: ## id then stage, symptom, applies, do, status, evidence fields (one per line).
Stages: observe/locate/grasp/transport/manipulate/place/verify/recover, optional /substage.
New advice MUST have status: hypothesis. Preserve existing card IDs, including retired cards.
Revise an old card or mark it retired with a reason and evidence; never silently drop it.
An unchanged verified card may retain its status; altered advice becomes a hypothesis or is
retired. Existing accepted evidence must not be exaggerated.
Use relative relationships, not remembered object poses. Generalize successful cells into a
small optional skill; never paste the whole episode. When alternatives are requested, explain
different approaches; they may share a category. No modifications outside this task.
"""


def evidence_bundle(
    run_dirs: list[Path], task: str, *, preview_cells: int | None = 12
) -> dict[str, Any]:
    episodes = []
    instructions = ""
    for root in run_dirs:
        result = json.loads((root / "result.json").read_text())
        if result.get("split") != "dev" or result.get("task") != task:
            raise ValueError("reflection accepts only this task's dev episodes")
        facts = digest(root)
        if facts["kind"] != "policy":
            continue
        (root / "digest.md").write_text(render(facts))
        (root / "evidence.json").write_text(json.dumps(facts, indent=2) + "\n")
        prompt_file = root / "prompt.txt"
        if not instructions and prompt_file.exists():
            if not prompt_file.resolve().is_relative_to(root.resolve()):
                raise ValueError("dev prompt escaped its episode")
            instructions = prompt_file.read_text()
        cells = []
        paths = sorted((root / "workspace").glob("cell_*.py"))
        selected = (
            paths if preview_cells is None else paths[-preview_cells:] if preview_cells else []
        )
        for path in selected:
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("dev cell escaped its episode")
            cells.append({"cell": path.name, "source": path.read_text()})
        events = root / "events.jsonl"
        available = sorted((root / "images").glob("*.png"))
        pictures = []
        for path in dict.fromkeys(available[:1] + available[-1:]):
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("dev image escaped its episode")
            pictures.append(str(path.resolve()))
        episodes.append(
            {
                "id": result.get("episode_id", root.parent.parent.name),
                "success": facts["success"],
                "failures": facts["failed_calls"],
                "cells": cells,
                "omitted_cells": len(paths) - len(selected),
                "model_visible_images": pictures,
                "events": [json.loads(line) for line in events.read_text().splitlines()][-24:]
                if events.exists()
                else [],
            }
        )
    if not episodes:
        raise ValueError("no judged development evidence")
    return {"task": task, "fixed_episode_instructions": instructions, "episodes": episodes}


def should_search(history: list[dict[str, Any]], current: dict[str, Any]) -> bool:
    metrics = [*history[-2:], current]
    return len(metrics) == 3 and all(
        _compare(previous, following)[0] <= 0
        for previous, following in zip(metrics, metrics[1:], strict=False)
    )


def proposal_prompt(
    bundle: dict[str, Any],
    library: Library,
    revision: str,
    count: int,
    *,
    history: list[dict[str, Any]] | None = None,
    catalog: list[dict[str, Any]] | None = None,
) -> str:
    view = library.view(revision, bundle["task"])
    files = {
        str(path.relative_to(view)): path.read_text()
        for path in sorted(view.rglob("*"))
        if path.is_file()
    }
    source = library.revisions / revision / "tasks" / bundle["task"] / "lessons"
    for path in sorted(source.glob("*.md")):
        files[f"lessons/{path.name}"] = path.read_text()
    return (
        INSTRUCTIONS
        + f"\nReturn exactly {count} proposal(s), or no_change.\n"
        + json.dumps(
            {
                "evidence": bundle,
                "current_library": files,
                "history": history or [],
                "artifacts": catalog or [],
            },
            ensure_ascii=False,
        )
    )


def parse_response(text: str, *, task: str, count: int, evidence_ids: set[str]) -> dict[str, Any]:
    response = json.loads(text)
    if not isinstance(response, dict) or any(
        not isinstance(response.get(field), str) or not response[field].strip()
        for field in ("observed", "hypothesis", "validation")
    ):
        raise LibraryError("reflection must separate facts, hypothesis and validation")
    proposals = response.get("proposals", [])
    if not isinstance(proposals, list):
        raise LibraryError("proposals must be a list")
    if (
        proposals == []
        and isinstance(response.get("no_change"), str)
        and response["no_change"].strip()
    ):
        response["proposals"] = []
        return response
    if len(proposals) != count:
        raise LibraryError("wrong proposal count")
    shapes = set()
    for proposal in proposals:
        if (
            not isinstance(proposal.get("strategy"), str)
            or not proposal["strategy"].strip()
            or not proposal.get("difference")
            or not proposal.get("expected")
        ):
            raise LibraryError("proposal lacks strategy and observable prediction")
        evidence = proposal.get("evidence", [])
        if not evidence or not set(evidence) <= evidence_ids:
            raise LibraryError("proposal must cite supplied dev episode IDs")
        files = proposal.get("files")
        if not isinstance(files, dict) or not files or len(files) > 3:
            raise LibraryError("proposal must change one to three task-local files")
        if len(json.dumps(files)) > 24000:
            raise LibraryError("proposal too large")
        functions = 0
        skill_files = 0
        for name, source in files.items():
            if (
                not isinstance(source, str)
                or not name.startswith(f"tasks/{task}/")
                or not (SKILL_RE.fullmatch(name) or LESSON_RE.fullmatch(name))
            ):
                raise LibraryError("proposal path outside this task's skills/lessons")
            if SKILL_RE.fullmatch(name):
                skill_files += 1
                tree = ast.parse(source)
                functions += sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and (
                        not node.args.args or node.args.args[0].arg != "robo"
                    ):
                        raise LibraryError("skill must explicitly accept robo first")
        if skill_files > 1 or functions > 1:
            raise LibraryError("one skill function per proposal")
        shape = json.dumps(files, sort_keys=True)
        if shape in shapes:
            raise LibraryError("alternative proposals contain identical changes")
        shapes.add(shape)
    return response


def validate_lesson_update(library: Library, parent: str, proposal: dict[str, Any]) -> None:
    for name, source in proposal["files"].items():
        if not LESSON_RE.fullmatch(name):
            continue
        path = library.revisions / parent / name
        previous = (
            {card["id"]: card for card in parse_lessons(path.read_text())} if path.is_file() else {}
        )
        cards = parse_lessons(source)
        current = {card["id"]: card for card in cards}
        if len(current) != len(cards):
            raise LibraryError("duplicate lesson card IDs")
        if missing := previous.keys() - current.keys():
            raise LibraryError(f"keep or explicitly retire old lesson IDs: {sorted(missing)}")
        for identifier, card in current.items():
            if card.get("status") not in {"hypothesis", "retired"} and card != previous.get(
                identifier
            ):
                raise LibraryError("new or changed advice cannot claim to be verified")


def create_candidate(
    library: Library, parent: str, name: str, proposal: dict[str, Any]
) -> tuple[Path, str]:
    validate_lesson_update(library, parent, proposal)
    target = library.candidate(name, base=parent)
    for relative, text in proposal["files"].items():
        path = target / relative
        if not path.resolve().is_relative_to(target.resolve()):
            raise LibraryError("candidate path traversal")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    errors = validate(target)
    if errors:
        raise LibraryError("; ".join(errors))
    revision = library.freeze(target, parent=parent, note=proposal["difference"])
    return target, revision


def reflect(
    prompt: str,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    timeout_s: float,
    max_tokens: int,
    extra_body: dict[str, Any] | None = None,
    transport: Any = None,
    images: list[str] | None = None,
    evidence: DevEvidence | None = None,
    reserve_next: Any = None,
    record: Any = None,
) -> tuple[str, dict[str, Any]]:
    from ..openai_runtime import httpx_transport, image_part

    content: Any = prompt
    if images:
        content = [{"type": "text", "text": prompt}]
        for path in images:
            content.append({"type": "text", "text": f"Recorded dev observation: {path}"})
            content.append(image_part(Path(path).read_bytes()))
    messages = [{"role": "user", "content": content}]
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if extra_body and {"messages", "model", "tools", "tool_choice"} & extra_body.keys():
        raise ValueError("reflection options cannot change model or grant tools")
    body.update(extra_body or {})
    if evidence is not None:
        body["tools"] = [READ_EVIDENCE_TOOL]
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    deadline = time.monotonic() + timeout_s
    usages = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("reflection wall budget exhausted")
        if usages:
            if reserve_next is None:
                raise ValueError("evidence follow-up requests need an explicit budget callback")
            reserve_next()
        code, payload = (transport or httpx_transport(api_key))(url, body, remaining)
        if record:
            record({"type": "response", "http_status": code, "payload": payload})
        if code != 200:
            raise RuntimeError(f"reflection service error: HTTP {code}")
        usages.append(payload.get("usage", {}))
        message = payload["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            return message.get("content") or "", {"requests": len(usages), "per_request": usages}
        if evidence is None:
            raise ValueError("reflection requested an unavailable tool")
        messages.append(message)
        pictures = []
        for call in calls:
            picture = None
            try:
                if call["function"]["name"] != "read_dev_evidence":
                    raise ValueError("only read_dev_evidence is available")
                result, picture = evidence.read(**json.loads(call["function"]["arguments"]))
            except (ValueError, TypeError, OSError) as exc:
                result = {"error": str(exc)}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            if record:
                record({"type": "evidence_read", "tool_call": call, "result": result})
            if picture is not None:
                pictures.extend(
                    [
                        {"type": "text", "text": f"Dev artifact: {result['artifact']}"},
                        image_part(picture),
                    ]
                )
        if pictures:
            messages.append({"role": "user", "content": pictures})
