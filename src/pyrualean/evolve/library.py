# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""A skill library under version control: revisions, candidates, and what an episode is given.

A library root holds::

    revisions/<id>/            immutable; ``manifest.json`` names the parent and every file
      shared/skills/*.py       skills for every task (importable as ``library.<module>``)
      shared/lessons/*.md      lessons for every task (appended to the prompt)
      tasks/<task>/skills/*.py
      tasks/<task>/lessons/*.md
    candidates/<name>/         a working copy in the same layout; never what an episode runs
    active.json                the revision episodes run against (replaced atomically)
    promotions.jsonl           every promotion decided, accepted or not (appended, never rewritten)
    views/<id>/<task>/         one task's view of a revision: ``skills/`` + ``lessons/``, the
                               flat layout :func:`pyrualean.play.copy_library` reads

A revision's id is the hash of its content, so a revision is what it says it is and two
identical libraries are one revision.  :func:`validate` is the static check a candidate passes
before it can become a revision: only skill and lesson files, Python that parses, a module top
level that only defines (nothing runs, moves or opens a file at import), no name the policy
sandbox refuses, and lessons in the card format (:data:`LESSON_FIELDS`).

None of this says a library is better: that is :mod:`.gate`'s question, asked of episodes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import stat
import time
from pathlib import Path
from typing import Any

from ..sandbox import BLOCKED_NAMES
from .safety import check_source

#: Files a library may hold, relative to its root.
SKILL_RE = re.compile(r"(?:shared|tasks/[A-Za-z0-9_]+)/skills/[a-z_][a-z0-9_]*\.py")
LESSON_RE = re.compile(r"(?:shared|tasks/[A-Za-z0-9_]+)/lessons/[a-z0-9_\-]+\.md")

#: Fields every lesson card carries (``## <id>`` then ``field: value`` lines).
LESSON_FIELDS = ("stage", "symptom", "applies", "do", "status", "evidence")
#: A card's ``status``: a hypothesis is shown as one; retired cards are kept, not shown.
LESSON_STATUSES = ("hypothesis", "dev_verified", "accepted", "retired")
#: Stages a card may name (``stage:``); a task may refine one as ``grasp/handle``.
STAGES = ("observe", "locate", "grasp", "transport", "manipulate", "place", "verify", "recover")

#: Top-level statements a skill module may hold: definitions and constants, nothing that runs.
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)


class LibraryError(ValueError):
    """A library or candidate that breaks the rules above."""


# -- reading -------------------------------------------------------------------


def _files(root: Path) -> dict[str, bytes]:
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            found[path.relative_to(root).as_posix()] = path.read_bytes()
    return found


def content_id(files: dict[str, bytes]) -> str:
    """The revision id of a set of files: the hash of their names and contents."""

    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + hashlib.sha256(files[name]).digest())
    return digest.hexdigest()[:16]


def parse_lessons(text: str) -> list[dict[str, str]]:
    """The cards of one lessons file: ``## <id>`` headings, then ``field: value`` lines."""

    cards: list[dict[str, str]] = []
    for block in re.split(r"(?m)^## ", text)[1:]:
        heading, _, body = block.partition("\n")
        card = {"id": heading.strip()}
        for line in body.splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() and not key.startswith(" "):
                card[key.strip().lower()] = value.strip()
        cards.append(card)
    return cards


def _live_cards(text: str) -> str:
    """A lessons file without its retired cards (the text before the first card is kept)."""

    head, *blocks = re.split(r"(?m)^(?=## )", text)
    kept = [
        b for b in blocks if parse_lessons(b) and parse_lessons(b)[0].get("status") != "retired"
    ]
    return head + "".join(kept)


# -- validation ----------------------------------------------------------------


def _validate_skill(name: str, source: str) -> list[str]:
    problems = [f"{name}: {error}" for error in check_source(source, skill=True)]
    try:
        tree = ast.parse(source, filename=name)
    except SyntaxError as exc:
        return [f"{name}: does not parse: {exc}"]
    for node in tree.body:
        if isinstance(node, _DEFINITIONS):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # a docstring
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None or all(
                isinstance(n, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.Name,
                               ast.UnaryOp, ast.BinOp, ast.Load, ast.USub, ast.UAdd, ast.Add,
                               ast.Sub, ast.Mult, ast.Div, ast.keyword))
                for n in ast.walk(value)
            ):  # fmt: skip
                continue
        problems.append(
            f"{name}:{node.lineno}: only definitions and constants at module level "
            f"(a {type(node).__name__} would run at import)"
        )
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    for blocked in sorted(used & BLOCKED_NAMES):
        problems.append(f"{name}: names {blocked}, which the policy sandbox refuses")
    return problems


def _validate_lessons(name: str, text: str) -> list[str]:
    problems = []
    cards = parse_lessons(text)
    if not cards:
        problems.append(f"{name}: no '## <id>' card")
    seen = set()
    for card in cards:
        where = f"{name}#{card['id']}"
        if card["id"] in seen:
            problems.append(f"{where}: duplicate card id")
        seen.add(card["id"])
        missing = [f for f in LESSON_FIELDS if not card.get(f)]
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
        if card.get("status") and card["status"] not in LESSON_STATUSES:
            problems.append(f"{where}: status {card['status']!r} not in {LESSON_STATUSES}")
        stage = (card.get("stage") or "").split("/")[0]
        if stage and stage not in STAGES:
            problems.append(f"{where}: stage {card['stage']!r} not one of {STAGES}")
    return problems


def validate(root: str | Path, *, allow_empty: bool = False) -> list[str]:
    """Every rule a library tree at ``root`` breaks (empty when it may become a revision).

    An empty tree is a library only when ``allow_empty`` (the root revision of a campaign:
    the zero-shot baseline, nothing added to the prompt or the workspace).
    """

    root = Path(root)
    problems = []
    if any(path.is_symlink() for path in root.rglob("*")):
        return ["library must not contain symbolic links"]
    files = _files(root) if root.is_dir() else {}
    if not files:
        return [] if allow_empty and root.is_dir() else [f"{root}: empty library"]
    for name, data in files.items():
        if SKILL_RE.fullmatch(name):
            problems += _validate_skill(name, data.decode("utf-8"))
        elif LESSON_RE.fullmatch(name):
            problems += _validate_lessons(name, data.decode("utf-8"))
        else:
            problems.append(f"{name}: not a skill (…/skills/*.py) or lessons (…/lessons/*.md) file")
    return problems


# -- revisions -----------------------------------------------------------------


class Library:
    """One library root on disk (see the module docstring for its layout)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    @property
    def revisions(self) -> Path:
        return self.root / "revisions"

    def active(self) -> str | None:
        """The id of the revision episodes run against, None before the first."""

        path = self.root / "active.json"
        return json.loads(path.read_text())["revision"] if path.is_file() else None

    def manifest(self, revision: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{16}", revision):
            raise LibraryError("invalid revision id")
        return json.loads((self.revisions / revision / "manifest.json").read_text())

    def freeze(self, source: str | Path, *, parent: str | None, note: str = "") -> str:
        """Make ``source`` a revision (refused when it breaks a rule); return its id.

        The revision's files are copied, made read-only, and listed with their hashes in its
        manifest.  Freezing content that already is a revision returns that revision.  Only a
        root revision (``parent=None``) may be empty: the zero-shot baseline.
        """

        source = Path(source)
        problems = validate(source, allow_empty=parent is None)
        if problems:
            raise LibraryError("; ".join(problems))
        files = _files(source)
        revision = content_id(files)
        target = self.revisions / revision
        if target.is_dir():
            return revision
        staging = self.revisions / f".{revision}.tmp-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)  # a root revision may hold no file
        for name, data in files.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        manifest = {
            "revision": revision,
            "parent": parent,
            "note": note,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        staging.rename(target)
        return revision

    def verify(self, revision: str) -> None:
        """Refuse a revision whose files no longer match its manifest."""

        root = self.revisions / revision
        manifest = self.manifest(revision)
        files = _files(root)
        actual = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        if actual != manifest["files"] or content_id(files) != revision:
            raise LibraryError(f"revision {revision} was modified after it was frozen")

    def set_active(self, revision: str) -> None:
        self.verify(revision)
        path = self.root / "active.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"revision": revision}) + "\n")
        tmp.replace(path)

    def candidate(self, name: str, *, base: str | None = None) -> Path:
        """A fresh working copy of revision ``base`` (default: the active one) to edit."""

        if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            raise LibraryError(f"candidate name {name!r}")
        target = self.root / "candidates" / name
        if target.exists():
            raise LibraryError(f"candidate {name} exists: {target}")
        base = base or self.active()
        target.mkdir(parents=True)
        if base is not None:
            self.verify(base)
            for rel, data in _files(self.revisions / base).items():
                path = target / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)  # writable copies: the revision stays read-only
        return target

    def view(self, revision: str, task: str) -> Path:
        """Task ``task``'s view of ``revision``: shared and task files, flat for copy_library.

        A skill module or lessons file of the same name in shared and in the task is refused:
        a view never silently prefers one.
        """

        if not re.fullmatch(r"[A-Za-z0-9_]+", task):
            raise LibraryError("invalid task name")
        self.verify(revision)
        source = self.revisions / revision
        target = self.root / "views" / revision / task
        picked: dict[str, Path] = {}
        for scope in (source / "shared", source / "tasks" / task):
            for kind in ("skills", "lessons"):
                for path in sorted((scope / kind).glob("*")) if (scope / kind).is_dir() else []:
                    key = f"{kind}/{path.name}"
                    if key in picked:
                        raise LibraryError(f"{key} is both shared and {task}'s in {revision}")
                    picked[key] = path
        expected = {
            key: _live_cards(path.read_text()).encode()
            if key.startswith("lessons/")
            else path.read_bytes()
            for key, path in picked.items()
        } or {"lessons/none.md": b""}
        if target.is_dir():
            if _files(target) != expected or any(path.is_symlink() for path in target.rglob("*")):
                raise LibraryError("cached task view was modified")
            return target
        staging = target.parent / f".{task}.tmp-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        for key, path in picked.items():
            (staging / key).parent.mkdir(parents=True, exist_ok=True)
            if key.startswith("lessons/"):
                # retired cards stay in the revision (with their counter-evidence), not in
                # what an episode reads; a hypothesis is shown, marked by its status line
                (staging / key).write_text(_live_cards(path.read_text(encoding="utf-8")))
            else:
                shutil.copyfile(path, staging / key)
        if not picked:
            # an empty view is a library with nothing for this task: a lessons file says so,
            # since copy_library refuses an empty directory
            (staging / "lessons").mkdir(parents=True)
            (staging / "lessons" / "none.md").write_text("")
        staging.rename(target)
        for path in target.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        return target

    def record(self, entry: dict[str, Any]) -> None:
        """Append one promotion decision to ``promotions.jsonl`` (never rewritten)."""

        entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry}
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "promotions.jsonl", "a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(entry, sort_keys=True) + "\n")

    def promotions(self) -> list[dict[str, Any]]:
        path = self.root / "promotions.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def promote(
        self,
        candidate: str | Path,
        *,
        gate: dict[str, Any],
        evidence: list[str],
        note: str = "",
    ) -> str | None:
        """Freeze a candidate and, when ``gate`` accepted it, make it the active revision.

        Every call is recorded, accepted or not.  Returns the new active revision, or None
        when the parent stays.
        """

        parent = self.active()
        revision = self.freeze(candidate, parent=parent, note=note)
        accepted = bool(gate.get("accepted"))
        self.record(
            {
                "parent": parent,
                "candidate": revision,
                "accepted": accepted,
                "reason": gate.get("reason"),
                "gate": gate.get("per_task"),
                "evidence": evidence,
                "note": note,
            }
        )
        if accepted:
            self.set_active(revision)
            return revision
        return None


def diff(library: Library, old: str | None, new: str) -> dict[str, list[str]]:
    """Which files a revision adds, removes and changes against another."""

    before = library.manifest(old)["files"] if old else {}
    after = library.manifest(new)["files"]
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "changed": sorted(n for n in set(after) & set(before) if after[n] != before[n]),
    }


__all__ = [
    "LESSON_FIELDS",
    "LESSON_STATUSES",
    "STAGES",
    "Library",
    "LibraryError",
    "content_id",
    "diff",
    "parse_lessons",
    "validate",
]
