# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Model-facing prompt built from the live library.

The API reference is generated from :class:`pyrualean.libero.LiberoRobot`'s
signatures and docstrings (``inspect``), so the model always sees exactly the
functions the host will inject.  The operating knowledge is a hand-written
markdown file that carries the same manipulation lessons RPent gives its
tool-calling agent, rewritten for someone who writes a program instead of
issuing one call per turn.
"""

from __future__ import annotations

import inspect
import json
import textwrap
from dataclasses import dataclass, field, fields, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from ._robot import PRIMITIVE_SETS, RobotBase
from .libero import RESULT_TYPES, LiberoRobot

_HOST_ONLY = frozenset({"ledger", "backend"})
_LINE = 88


def _annotation(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, type):
        return value.__name__
    return str(value).replace("typing.", "")


def _format_parameters(sig: inspect.Signature) -> list[str]:
    parts: list[str] = []
    seen_kw_only = False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY and not seen_kw_only:
            parts.append("*")
            seen_kw_only = True
        text = name
        annotation = _annotation(param.annotation)
        if annotation:
            text += f": {annotation}"
        if param.default is not inspect.Parameter.empty:
            text += f" = {param.default!r}" if annotation else f"={param.default!r}"
        parts.append(text)
    return parts


def _render_callable(owner: str, name: str, fn: Any) -> str:
    sig = inspect.signature(fn)
    params = _format_parameters(sig)
    ret = _annotation(sig.return_annotation)
    suffix = f" -> {ret}" if ret else ""
    head = f"{owner}.{name}({', '.join(params)}){suffix}"
    if len(head) > _LINE:
        joined = ",\n    ".join(params)
        head = f"{owner}.{name}(\n    {joined},\n){suffix}"
    doc = inspect.getdoc(fn) or ""
    return head + "\n" + textwrap.indent(doc, "    ") if doc else head


def _render_property(owner: str, name: str, prop: property) -> str:
    fn = prop.fget
    ret = ""
    if fn is not None:
        ret = _annotation(inspect.signature(fn).return_annotation)
    head = f"{owner}.{name}" + (f" -> {ret}" if ret else "") + "  (read-only attribute)"
    doc = inspect.getdoc(fn) if fn is not None else ""
    return head + ("\n" + textwrap.indent(doc, "    ") if doc else "")


def _render_result_type(cls: type) -> str:
    parts = []
    for item in fields(cls):
        parts.append(f"{item.name}: {_annotation(item.type)}")
    head = f"class {cls.__name__}({', '.join(parts)})"
    if len(head) > _LINE:
        head = f"class {cls.__name__}(\n    " + ",\n    ".join(parts) + ",\n)"
    extra = []
    for name, member in vars(cls).items():
        if isinstance(member, property) and not name.startswith("_"):
            ret = _annotation(inspect.signature(member.fget).return_annotation)
            doc = inspect.getdoc(member.fget) or ""
            extra.append(f"    .{name} -> {ret}: {doc}")
    doc = inspect.getdoc(cls) or ""
    body = textwrap.indent(doc, "    ") if doc else ""
    return "\n".join(part for part in [head, body, *extra] if part)


def _public_members(robot_cls: type) -> dict[str, Any]:
    """Public methods and properties in definition order, base classes first.

    An override keeps the position of the member it overrides, so a robot
    that re-documents ``image`` for its own cameras does not reorder the
    reference.  ``RobotBase``'s configuration attributes never show.
    """

    members: dict[str, Any] = {}
    for klass in reversed(robot_cls.__mro__):
        if klass is object:
            continue
        for name, member in vars(klass).items():
            if name.startswith("_") or name in _HOST_ONLY:
                continue
            if isinstance(member, property) or inspect.isfunction(member):
                members[name] = member
    return members


def _return_type_name(member: Any) -> str | None:
    fn = member.fget if isinstance(member, property) else member
    if fn is None:
        return None
    annotation = inspect.signature(fn).return_annotation
    if annotation is inspect.Signature.empty:
        return None
    return annotation if isinstance(annotation, str) else getattr(annotation, "__name__", None)


def vla_hidden(robot_cls: type, primitives: str = "full") -> tuple[frozenset[str], frozenset[str]]:
    """What the ``no-vla`` primitive set hides: ``(member names, result type names)``.

    The members are ``robot_cls.VLA_PRIMITIVES`` (methods, or constants of
    that name); the result types are those returned by a hidden method and
    by no remaining public member.  ``full`` hides nothing.
    """

    if primitives not in PRIMITIVE_SETS:
        raise ValueError(f"primitives must be one of {PRIMITIVE_SETS}, got {primitives!r}")
    if primitives == "full":
        return frozenset(), frozenset()
    names = frozenset(getattr(robot_cls, "VLA_PRIMITIVES", ()))
    members = _public_members(robot_cls)
    returned_by_vla = {_return_type_name(members[n]) for n in names if n in members}
    returned_by_rest = {_return_type_name(m) for n, m in members.items() if n not in names}
    return names, frozenset(t for t in returned_by_vla - returned_by_rest if t)


def _apply_edits(text: str, edits: tuple[tuple[str, str], ...], what: str) -> str:
    for old, new in edits:
        found = text.count(old)
        if found != 1:
            raise RuntimeError(f"{what} drifted: {old[:60]!r} found {found} times, expected once")
        text = text.replace(old, new)
    return text


def api_reference(
    robot_cls: type = LiberoRobot, *, var: str = "robo", primitives: str = "full"
) -> str:
    """Render the policy-facing API of ``robot_cls`` from its live definition.

    ``primitives="no-vla"`` leaves out the robot's ``VLA_PRIMITIVES``, the
    result types only they return, and (through ``VLA_REFERENCE_EDITS``) the
    sentences of the other docstrings that describe the VLA.
    """

    vla_members, vla_types = vla_hidden(robot_cls, primitives)
    sections: list[str] = []
    class_doc = inspect.getdoc(robot_cls) or ""
    sections.append(f"class {robot_cls.__name__}\n{textwrap.indent(class_doc, '    ')}")
    # Base-class configuration, ``RESULT_TYPES`` (rendered as the result-type
    # section below) and ``ClassVar``-annotated configuration are not
    # constants the policy needs.
    hidden = set(vars(RobotBase)) | {"RESULT_TYPES"} | set(vla_members)
    hidden |= {
        name
        for name, annotation in inspect.get_annotations(robot_cls).items()
        if str(annotation).replace("typing.", "").startswith("ClassVar")
    }
    constants = [
        f"{var}.{name} = {value!r}"
        for name, value in vars(robot_cls).items()
        if name.isupper() and not name.startswith("_") and name not in hidden
    ]
    if constants:
        sections.append("Constants:\n" + "\n".join(f"    {c}" for c in constants))
    members: list[str] = []
    for name, member in _public_members(robot_cls).items():
        if name in vla_members:
            continue
        if isinstance(member, property):
            members.append(_render_property(var, name, member))
        else:
            members.append(_render_callable(var, name, member))
    sections.append("Methods and attributes of `robo`:\n\n" + "\n\n".join(members))
    result_types = tuple(getattr(robot_cls, "RESULT_TYPES", ()) or RESULT_TYPES)
    results = [
        _render_result_type(cls)
        for cls in result_types
        if is_dataclass(cls) and cls.__name__ not in vla_types
    ]
    sections.append(
        "Result types (frozen dataclasses; access fields as attributes):\n\n" + "\n\n".join(results)
    )
    sections.append(
        "Exceptions:\n"
        "    ToolError(RuntimeError): a call was rejected or a service failed; "
        "catch it if you have a fallback.\n"
        "    ValueError: an argument was invalid (e.g. a planar move over "
        "0.30 m) - nothing was executed.\n"
        "    EpisodeFinished(BaseException): the episode is over "
        "(reason: terminated | truncated | timeout | cancelled); return."
    )
    text = "\n\n".join(sections)
    if primitives == "no-vla":
        edits = tuple(getattr(robot_cls, "VLA_REFERENCE_EDITS", ()))
        text = _apply_edits(text, edits, f"{robot_cls.__name__} docstrings")
    return text


def knowledge_stem(name: str, primitives: str = "full") -> str:
    """Knowledge file stem: ``<name>`` for the full primitive set, ``<name>-novla`` otherwise."""

    if primitives not in PRIMITIVE_SETS:
        raise ValueError(f"primitives must be one of {PRIMITIVE_SETS}, got {primitives!r}")
    return name if primitives == "full" else f"{name}-novla"


def knowledge_text(name: str = "libero", primitives: str = "full") -> str:
    """Return the packaged operating-knowledge markdown for a backend.

    ``primitives="no-vla"`` reads ``knowledge/<name>-novla.md``: the same
    document with the sections and sentences about the VLA removed.
    """

    stem = knowledge_stem(name, primitives)
    return resources.files("pyrualean").joinpath(f"knowledge/{stem}.md").read_text(encoding="utf-8")


@dataclass
class TaskCard:
    """What the policy author is told about one evaluation cell.

    Everything here is available to RPent's agent too (through its first
    ``view_env_state`` call); nothing privileged such as object poses is
    included.
    """

    suite: str
    task: int | str
    seed: int
    task_language: str
    object_names: list[str]
    #: Initial gripper pose of a single-arm robot (LIBERO).  Robots with
    #: several arms leave these ``None`` and describe each arm in ``facts``.
    eef_pos: list[float] | None = None
    eef_quat: list[float] | None = None
    gripper_opening: float | None = None
    images: dict[str, str] = field(default_factory=dict)
    max_env_steps: int = 10000
    #: ``False``: the program runs on exactly this cell (per-cell single shot).
    #: ``True``: the program will be run on other seeds of the same task, so
    #: the images are only an example layout (transfer mode).
    transfer: bool = False
    #: Which robot the card describes (``libero``, ``robocasa``, ``robotwin``).
    robot: str = "libero"
    #: Robot-specific initial facts rendered as extra bullets (non-LIBERO
    #: robots put their identifiers and initial pose here).
    facts: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str | Path) -> TaskCard:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def render(self) -> str:
        if self.robot != "libero" or self.eef_pos is None or self.gripper_opening is None:
            return self._render_generic()
        eef = ", ".join(f"{v:.3f}" for v in self.eef_pos)
        lines = [
            "# Task card",
            f"- suite: {self.suite}",
            f"- task: {self.task}",
            f"- seed: {self.seed}",
            f"- task instruction (authoritative): {self.task_language!r}",
            f"- movable objects in the scene: {', '.join(self.object_names)}",
            f"- initial gripper position [x, y, z]: [{eef}] m, opening "
            f"{self.gripper_opening:.3f} (open, pointing down)",
            f"- simulator step budget: {self.max_env_steps}",
        ]
        if self.images:
            names = ", ".join(f"{k} ({Path(v).name})" for k, v in self.images.items())
            if self.transfer:
                lines.append(
                    f"- attached images: {names}. They show ONE example layout "
                    f"(seed {self.seed}) of this task.  Your program will be run "
                    "on OTHER seeds of the same task, where the objects and the "
                    "container are placed differently, so the images only tell "
                    "you what the objects look like and what the task means.  "
                    "Do not hard-code pixel coordinates or world positions read "
                    "off these images; localise everything at runtime with "
                    "`robo.segment`, `robo.back_project`, `robo.region_center` "
                    "and `robo.world_map`."
                )
            else:
                lines.append(
                    f"- attached images: {names}. They are step 0 of your run, "
                    "exactly what `robo.image('agentview')` / "
                    "`robo.image('wrist')` return before any motion; a pixel "
                    "(row, col) you read off the 1024x1024 agentview image can be "
                    "turned into world coordinates at runtime with "
                    "`robo.back_project(row, col, step=0)`."
                )
        return "\n".join(lines)

    def _render_generic(self) -> str:
        """Card of a non-LIBERO robot: identifiers, instruction, facts, images."""

        lines = [
            "# Task card",
            f"- robot: {self.robot}",
            f"- task: {self.task}",
        ]
        if self.suite:
            lines.append(f"- variant: {self.suite}")
        lines += [
            f"- seed: {self.seed}",
            f"- task instruction (authoritative): {self.task_language!r}",
        ]
        if self.object_names:
            lines.append(f"- objects named by the benchmark: {', '.join(self.object_names)}")
        for key, value in self.facts.items():
            lines.append(f"- {key}: {value}")
        lines.append(f"- simulator step budget: {self.max_env_steps}")
        if self.images:
            names = ", ".join(f"{k} ({Path(v).name})" for k, v in self.images.items())
            cameras = ", ".join(f"`robo.image({k!r})`" for k in self.images)
            if self.transfer:
                lines.append(
                    f"- attached images: {names}. They show ONE example layout "
                    f"(seed {self.seed}) of this task; your program will run on "
                    "other seeds where objects are placed differently, so do "
                    "not hard-code pixels or world positions read off them."
                )
            else:
                lines.append(
                    f"- attached images: {names}. They are step 0 of your run, "
                    f"exactly what {cameras} return before any motion."
                )
        return "\n".join(lines)


@dataclass(frozen=True)
class PromptBundle:
    """System text, task text and image attachments for one generation."""

    system: str
    user: str
    images: tuple[str, ...] = ()

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def text(self) -> str:
        return self.system + "\n\n" + self.user


CONTRACT = """\
# What you are asked to do

You control {robot_blurb} by WRITING A
PROGRAM, not by chatting with the robot.  Reply with one complete Python file
that defines

    def run(robo):

The host imports the file, constructs `robo` (a `{cls}` bound to the
live simulator{services}) and calls
`run(robo)` exactly once.  No language model runs while the program executes:
every decision has to be made in code, from the return values and images
documented below.

# How the episode is scored

- Success = the benchmark's own task predicate fires (`terminated`) at any
  moment while `run` executes.  The simulator checks it, not you.
- The return value of `run` is ignored.
- The episode counts as a failure when `run` raises an uncaught exception,
  when the host's wall-clock budget of {budget_s} s runs out, or when the
  simulator step budget ({max_env_steps} steps) is exhausted.
- Once the task terminates, every further motion call raises
  `EpisodeFinished` (a BaseException): just return.

# Output format

Reply with the complete source of `policy.py` in a single ```python fenced
block and nothing else.  The program may import `math`, `time` and `numpy`
(as `np`); do not import anything else, read files or start threads.  Keep it
self-contained: helper functions live in the same file.  Print short progress
lines with `print(...)`; stdout is saved with the episode.
"""


def render_prompt(
    card: TaskCard | None = None,
    *,
    budget_s: int = 1200,
    knowledge: str | None = None,
    robot_cls: type | None = None,
    primitives: str = "full",
) -> PromptBundle:
    """Assemble the generation prompt for one cell (or a cell-less preview).

    The robot comes from ``card.robot`` (``libero`` when there is no card):
    its class supplies the API reference, its knowledge file the operating
    knowledge and its adapter the wording of the contract.  ``primitives``
    is the primitive set (:data:`pyrualean._robot.PRIMITIVE_SETS`).
    """

    from .robots import get_adapter

    adapter = get_adapter(card.robot if card is not None else "libero")
    max_env_steps = card.max_env_steps if card is not None else 10000
    services = adapter.services if primitives == "full" else adapter.services_novla
    contract = CONTRACT.format(
        budget_s=budget_s,
        max_env_steps=max_env_steps,
        robot_blurb=adapter.blurb,
        cls=(robot_cls or adapter.robot_cls).__name__,
        services=f", {services}" if services else "",
    )
    reference = api_reference(robot_cls or adapter.robot_cls, primitives=primitives)
    know = (
        knowledge
        if knowledge is not None
        else knowledge_text(adapter.knowledge, primitives=primitives)
    )
    system = "\n\n".join(
        [
            contract.rstrip(),
            "# API reference (generated from the library the host injects)\n\n" + reference,
            know.strip(),
        ]
    )
    if card is None:
        user = "# Task card\n(no cell selected; render with a TaskCard for a real run)"
        images: tuple[str, ...] = ()
    else:
        user = card.render()
        images = tuple(card.images.values())
    return PromptBundle(system=system, user=user, images=images)


__all__ = [
    "CONTRACT",
    "PromptBundle",
    "TaskCard",
    "api_reference",
    "knowledge_stem",
    "knowledge_text",
    "render_prompt",
    "vla_hidden",
]
