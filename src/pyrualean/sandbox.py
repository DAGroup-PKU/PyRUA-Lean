# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Execution sandbox and audit for model-written policy code.

In-process Python cannot be sealed, so the sandbox does three things: it
removes the convenient routes out of the documented ``robo`` API (importing
``os`` / ``subprocess`` / ``pathlib``, opening files outside the workspace,
``eval`` / ``exec`` with fresh builtins), it refuses to run code that names
Python's introspection routes (``__globals__``, ``__func__``,
``__subclasses__``, ...: :data:`BLOCKED_NAMES`, checked by
``blocked_names``), and it records what a cell *tried* (``audit_source``),
so an episode that reached for the simulator's internals is visible in
``result.json`` instead of silently counting as a success.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import os
from pathlib import Path
from typing import Any

#: Top-level modules policy code may import: pure computation only.
ALLOWED_MODULES = frozenset(
    {
        "__future__",
        "abc",
        "array",
        "bisect",
        "cmath",
        "collections",
        "contextlib",
        "copy",
        "cv2",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "io",
        "itertools",
        "json",
        "math",
        "numbers",
        "numpy",
        "operator",
        "PIL",
        "pprint",
        "random",
        "re",
        "scipy",
        "skimage",
        "statistics",
        "string",
        "struct",
        "textwrap",
        "time",
        "traceback",
        "types",
        "typing",
        "warnings",
    }
)

#: Builtins withheld from policy code (``open`` and ``__import__`` are
#: replaced by guarded versions).
REMOVED_BUILTINS = frozenset(
    {"exec", "eval", "compile", "__import__", "open", "breakpoint", "input", "help", "exit", "quit"}
)

#: Introspection routes out of the documented API that policy code may not
#: name at all: a cell or program that reads one of these (as an attribute, a
#: bare name or a string constant) is refused before it runs.  ``__globals__``
#: / ``__func__`` were once used to patch the library at runtime;
#: ``__subclasses__`` reaches every loaded class.
BLOCKED_NAMES = frozenset(
    {
        "__builtins__",
        "__closure__",
        "__code__",
        "__func__",
        "__globals__",
        "__import__",
        "__loader__",
        "__self__",
        "__spec__",
        "__subclasses__",
        "__wrapped__",
    }
)

DESCRIPTION = (
    "imports limited to pure-computation modules and workspace files, open() confined to "
    "the workspace, exec/eval/compile removed, code naming __globals__/__func__/__subclasses__-"
    "style introspection refused before it runs; every cell is audited (sandbox_flags)"
)


def _workspace_module(root: Path, top: str) -> bool:
    return bool(top) and ((root / f"{top}.py").is_file() or (root / top / "__init__.py").is_file())


def make_builtins(workspace: str | Path) -> dict[str, Any]:
    """Return a ``__builtins__`` mapping for code that runs on behalf of the model."""

    root = Path(workspace).resolve()
    real_import = _builtins.__import__
    real_open = _builtins.open

    def guarded_import(
        name: str,
        globals: Any = None,  # noqa: A002 - signature of builtins.__import__
        locals: Any = None,  # noqa: A002
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        top = name.split(".")[0] if name else ""
        if level == 0 and top not in ALLOWED_MODULES and not _workspace_module(root, top):
            raise ImportError(
                f"module {top!r} is not available inside the policy sandbox; available: "
                f"{', '.join(sorted(ALLOWED_MODULES))}, and .py files in your workspace"
            )
        return real_import(name, globals, locals, fromlist, level)

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, int):
            raise PermissionError("policy code may not open file descriptors")
        target = Path(os.fspath(file))
        resolved = target.resolve() if target.is_absolute() else (root / target).resolve()
        if resolved != root and root not in resolved.parents:
            raise PermissionError(
                f"policy code may only open files inside its workspace ({root}); got {file!r}"
            )
        return real_open(resolved, mode, *args, **kwargs)

    namespace = {k: v for k, v in vars(_builtins).items() if k not in REMOVED_BUILTINS}
    namespace["__import__"] = guarded_import
    namespace["open"] = guarded_open
    return namespace


_HOST_ATTRS = frozenset({"backend", "ledger", "toolkit", "primitives"})
_ESCAPE_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "__builtins__",
        "__subclasses__",
        "__class__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__closure__",
        "__code__",
        "__dict__",
        "__loader__",
        "__spec__",
        "__func__",
        "__self__",
        "__wrapped__",
    }
)
_PROBE_NAMES = frozenset({"getattr", "setattr", "delattr", "vars", "locals", "dir"})
MAX_FLAGS = 40


def audit_source(code: str, *, workspace: str | Path | None = None) -> list[str]:
    """Static flags for code that leaves the documented API.

    Categories: ``import:<module>`` (not allowed), ``host-attr:<name>`` (the
    host's handles), ``private-attr:<name>``, ``escape:<name>`` (introspection
    that reaches other objects), ``probe:<name>`` (``dir``/``getattr``...),
    ``path:<text>`` (absolute, parent-relative or artifact paths).  Flags are
    informational: the runtime guards decide what actually runs.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    root = Path(workspace).resolve() if workspace is not None else None
    flags: list[str] = []

    def add(flag: str) -> None:
        if flag not in flags and len(flags) < MAX_FLAGS:
            flags.append(flag)

    def check_import(name: str | None) -> None:
        top = (name or "").split(".")[0]
        if not top or top in ALLOWED_MODULES:
            return
        if root is not None and _workspace_module(root, top):
            return
        add(f"import:{top}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                check_import(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                check_import(node.module)
        elif isinstance(node, ast.Attribute):
            if node.attr in _HOST_ATTRS:
                add(f"host-attr:{node.attr}")
            elif node.attr in _ESCAPE_NAMES:
                add(f"escape:{node.attr}")
            elif node.attr.startswith("_"):
                add(f"private-attr:{node.attr}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _ESCAPE_NAMES:
                add(f"escape:{node.id}")
            elif node.id in _PROBE_NAMES:
                add(f"probe:{node.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if text in _ESCAPE_NAMES:  # getattr(obj, "__globals__")
                add(f"escape:{text}")
            parts = text.replace("\\", "/").split("/")
            if text.startswith(("/", "~")) or ".." in parts or "states.json" in text:
                add(f"path:{text[:60]}")
    return flags


def blocked_names(code: str) -> list[str]:
    """Members of :data:`BLOCKED_NAMES` that ``code`` reads, in first-seen order.

    Attribute access, bare names and string constants all count (the last
    covers ``getattr(obj, "__globals__")``).  Unparsable code returns ``[]``
    so the caller reports the ``SyntaxError`` instead.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name: Any = node.attr
        elif isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Constant):
            name = node.value
        else:
            continue
        if isinstance(name, str) and name in BLOCKED_NAMES and name not in found:
            found.append(name)
    return found


__all__ = [
    "ALLOWED_MODULES",
    "BLOCKED_NAMES",
    "DESCRIPTION",
    "REMOVED_BUILTINS",
    "audit_source",
    "blocked_names",
    "make_builtins",
]
