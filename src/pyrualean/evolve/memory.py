"""Task-local retrieval and evidence-labelled episode progress; no simulator truth."""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path
from typing import Any

from .library import STAGES, parse_lessons
from .safety import check_source

CONTRACT = """# Online iteration harness
Work in short cells: observe, act, verify, return control. Never run an entire task in a loop.
Update progress when entering a new phase. Import skills with `from library.MODULE import FUNCTION`.
`harness.recall(stage, symptom='')` retrieves matching lessons and available skill docstrings.
`harness.progress(stage, confirmed=[], pending=[], evidence=[], holding='unknown')` records
YOUR interpretation, not the environment verdict. Cite current `cell:NUMBER` evidence;
confirmed statements require evidence. Use the returned primitive feedback and fresh images.
After motion, old object locations are stale. Do not reuse them without re-observation.
Repeated identical failures produce warnings, not action bans. Reassess using fresh evidence;
meaningful parameter adjustments remain allowed within the episode budget.
Skills take `robo` explicitly and return control for observation. Skill success is not task success.
The library and low-level API are frozen for this episode; propose changes only afterwards.
"""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.lower()))


class EpisodeMemory:
    def __init__(
        self,
        view: Path | None,
        output: Path,
        *,
        candidate: bool = False,
        max_attempts: int = 3,
        lesson_limit: int | None = None,
        skill_limit: int | None = None,
    ) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.cards = []
        self.skills = []
        self.cell = 0
        self.max_attempts = max_attempts
        self.lesson_limit = lesson_limit
        self.skill_limit = skill_limit
        self.failures: dict[str, int] = {}
        self.state: dict[str, Any] = {
            "stage": "observe",
            "confirmed": [],
            "pending": [],
            "evidence": [],
            "holding": "unknown",
            "locations_stale": True,
            "source": "model_report",
        }
        if view:
            for path in sorted((view / "lessons").glob("*.md")):
                task_local = (
                    view.parent.parent.parent
                    / "revisions"
                    / view.parent.name
                    / "tasks"
                    / view.name
                    / "lessons"
                    / path.name
                ).is_file()
                for card in parse_lessons(path.read_text()):
                    if card.get("status") in (
                        {"accepted", "dev_verified", "hypothesis"}
                        if candidate
                        else {"accepted", "dev_verified"}
                    ):
                        self.cards.append(
                            {**card, "file": path.name, "scope": "task" if task_local else "shared"}
                        )
            for path in sorted((view / "skills").glob("*.py")):
                for node in ast.parse(path.read_text()).body:
                    if isinstance(node, ast.FunctionDef):
                        self.skills.append(
                            {
                                "call": f"library.{path.stem}.{node.name}",
                                "signature": f"({ast.unparse(node.args)})",
                                "description": ast.get_docstring(node) or "",
                            }
                        )
        self._save()

    def _save(self) -> None:
        (self.output / "progress.json").write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2) + "\n"
        )

    def event(self, kind: str, **fields: Any) -> None:
        with (self.output / "events.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {"type": kind, "cell": self.cell, "stage": self.state["stage"], **fields},
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    def recall(self, stage: str, symptom: str = "") -> dict[str, Any]:
        if stage.split("/")[0] not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        words = _tokens(symptom)
        cards = [
            card
            for card in self.cards
            if card.get("stage", "").split("/")[0] == stage.split("/")[0]
        ]
        cards.sort(
            key=lambda card: (
                card["scope"] != "task",
                -len(words & _tokens(json.dumps(card))),
                card["file"],
                card["id"],
            )
        )
        skills = sorted(
            self.skills,
            key=lambda item: -len(_tokens(stage + " " + symptom) & _tokens(json.dumps(item))),
        )[: self.skill_limit]
        cards = cards[: self.lesson_limit]
        result = {"lessons": cards, "skills": skills}
        self.event(
            "recall", entries=[f"{card['file']}#{card['id']}" for card in cards], skills=skills
        )
        return copy.deepcopy(result)

    def progress(
        self,
        stage: str,
        confirmed: list[str] | None = None,
        pending: list[str] | None = None,
        evidence: list[str] | None = None,
        holding: str = "unknown",
    ) -> dict[str, Any]:
        if stage.split("/")[0] not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        confirmed, pending, evidence = confirmed or [], pending or [], evidence or []
        if any(
            not isinstance(items, list) or any(not isinstance(item, str) for item in items)
            for items in (confirmed, pending, evidence)
        ):
            raise ValueError("progress fields must be lists of strings")
        if confirmed and not evidence:
            raise ValueError("confirmed progress requires evidence")
        for reference in evidence:
            if (
                not re.fullmatch(r"cell:\d+", reference)
                or not 0 < int(reference.split(":")[1]) <= self.cell
            ):
                raise ValueError("evidence must reference an existing cell:NUMBER")
        if len(json.dumps([confirmed, pending, evidence, holding])) > 6000:
            raise ValueError("progress is too long")
        self.state.update(
            copy.deepcopy(
                {
                    "stage": stage,
                    "confirmed": confirmed,
                    "pending": pending,
                    "evidence": evidence,
                    "holding": holding,
                }
            )
        )
        self.event("progress", progress=self.state)
        self._save()
        return copy.deepcopy(self.state)

    def begin_cell(self, code: str, index: int) -> None:
        self.cell = index
        errors = check_source(code)
        if errors:
            self.event("rejected_cell", errors=errors)
            raise ValueError("; ".join(errors))
        self.event("cell_start", code=f"workspace/cell_{index:03d}.py")

    def end_cell(self, records: list[Any], result: dict[str, Any]) -> None:
        from .digest import _failure

        failed = bool(result.get("exception") or result.get("error"))
        symptoms = []
        for record in records:
            symptom = _failure(
                {
                    "tool": record.tool,
                    "ok": record.ok,
                    "error": record.error,
                    "summary": record.summary,
                }
            )
            if symptom is not None:
                failed = True
                symptoms.append(f"{record.tool}: {symptom}")
        if failed:
            self.state["last_failure"] = symptoms or ["cell exception; inspect feedback"]
        elif records:
            self.state["last_failure"] = []
        if any(record.stateful for record in records):
            self.state["locations_stale"] = True
            self.state["holding"] = "unknown_after_motion"
        self.event(
            "cell_end",
            failed=failed,
            calls=[record.index for record in records],
            tools=[record.tool for record in records],
            observations="images are dated by transcript order; old poses expire after motion",
        )
        self._save()

    def context(self) -> str:
        failure = self.state.get("last_failure", [])
        relevant = self.recall(
            "recover" if failure else self.state["stage"],
            " ".join([*self.state["pending"], *failure]),
        )
        text = "Current episode progress (model report, not score):\n" + json.dumps(
            {"progress": self.state, "memory": relevant}, ensure_ascii=False
        )
        self.event("context", text=text)
        return text

    def robot_view(self, robot: Any) -> Any:
        return _RobotView(robot, self)

    def public_api(self) -> Any:
        return _HarnessView(self)


class _HarnessView:
    def __init__(self, memory: EpisodeMemory) -> None:
        object.__setattr__(self, "_memory", memory)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("harness exposes only recall and progress")

    def __getattr__(self, name: str) -> Any:
        if name not in {"recall", "progress"}:
            raise AttributeError("harness exposes only recall and progress")
        return getattr(self._memory, name)


def _parameters(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_parameters(item) for item in value]
    if isinstance(value, dict):
        return {key: _parameters(item) for key, item in sorted(value.items())}
    if hasattr(value, "tolist"):
        return _parameters(value.tolist())
    return value


class _RobotView:
    def __init__(self, robot: Any, memory: EpisodeMemory) -> None:
        object.__setattr__(self, "_robot", robot)
        object.__setattr__(self, "_memory", memory)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("the robot API is read-only")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError("only public robot API is available")
        method = getattr(self._robot, name)
        if not callable(method):
            return method

        def call(*args: Any, **kwargs: Any) -> Any:
            from .digest import _failure

            memory = self._memory
            key = json.dumps(
                [memory.state["stage"].split("/")[0], name, _parameters(args), _parameters(kwargs)],
                sort_keys=True,
                default=str,
            )
            if memory.failures.get(key, 0) >= memory.max_attempts:
                memory.event("retry_warning", action=name, failures=memory.failures[key])
                memory.state["retry_warning"] = {
                    "action": name,
                    "failed_attempts": memory.failures[key],
                    "advice": (
                        "Repeated identical failure; reconsider using fresh evidence. "
                        "Action remains allowed."
                    ),
                }
            start = len(self._robot._ledger)
            try:
                return method(*args, **kwargs)
            finally:
                records = self._robot._ledger.records[start:]
                failed = any(_failure(record.to_dict()) is not None for record in records)
                if failed:
                    memory.failures[key] = memory.failures.get(key, 0) + 1
                elif records:
                    memory.failures.pop(key, None)
                    memory.state.pop("retry_warning", None)
                if any(record.stateful for record in records):
                    memory.state["locations_stale"] = True
                    memory.state["holding"] = "unknown_after_motion"
                memory.event(
                    "action", action=name, calls=[record.index for record in records], failed=failed
                )

        return call


def compact_messages(
    messages: list[dict[str, Any]], context: str, keep: int | None = None
) -> list[dict[str, Any]]:
    """Keep the initial contract and complete assistant/tool groups, never orphan a tool."""
    if not messages:
        return []
    if keep is None:
        return [messages[0], {"role": "user", "content": context}, *messages[1:]]
    starts = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    begin = starts[-keep] if len(starts) > keep else 1
    initial = dict(messages[0])
    if begin > 1 and isinstance(initial.get("content"), list):
        initial["content"] = [
            part for part in initial["content"] if part.get("type") != "image_url"
        ]
    critical = []
    for position, start in enumerate(starts):
        if start >= begin:
            break
        end = starts[position + 1] if position + 1 < len(starts) else len(messages)
        group = messages[start:end]
        if any(
            message.get("role") == "tool"
            and any(
                marker in str(message.get("content", ""))
                for marker in ('"exception":', '"reached": false', '"ok": false')
            )
            for message in group
        ):
            critical = group
    historical = (
        [
            {
                "role": "user",
                "content": "Earlier failed interaction; images/poses are historical, not current.",
            },
            *critical,
        ]
        if critical
        else []
    )
    return [initial, {"role": "user", "content": context}, *historical, *messages[begin:]]
