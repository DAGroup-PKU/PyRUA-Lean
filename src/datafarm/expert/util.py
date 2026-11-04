"""Shared helpers: canonical JSON, digests, atomic files, append-only JSONL, clocks, templates.

Everything here is standard library only, so the harness runs on the operator host without
the simulator or training environments.
"""

from __future__ import annotations

from datafarm import common
from datafarm.common import read_json, sha256_bytes, sha256_file, sha256_json, sha256_text, utc_now  # noqa: F401
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


class HarnessError(Exception):
    """Base class for every error the harness raises on purpose."""


class ConfigError(HarnessError):
    """An adapter, suite or round configuration is invalid (fail closed)."""


class UnsafePathError(HarnessError):
    """An agent-supplied path escapes its root, is a symlink, or is not a regular file."""


# --------------------------------------------------------------------------- time


class Clock:
    """Wall clock used for budgets and timestamps; tests substitute a deterministic subclass."""

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


def human_duration(seconds: float) -> str:
    """``17100`` -> ``"4 h 45 min"``; ``5400`` -> ``"1 h 30 min"``; ``600`` -> ``"10 min"``."""

    minutes = int(round(float(seconds) / 60.0))
    hours, rest = divmod(minutes, 60)
    if hours and rest:
        return f"{hours} h {rest} min"
    if hours:
        return f"{hours} h"
    return f"{rest} min"


# --------------------------------------------------------------------------- digests


def tree_digest(root: str | Path, *, exclude_names: Iterable[str] = ("__pycache__",),
                exclude_suffixes: Iterable[str] = (".pyc",)) -> dict[str, Any]:
    """Digest of a directory tree: sorted ``relpath<TAB>sha256`` lines, symlinks by target text.

    Symlinks are never followed, so a link that points somewhere else is itself a change.
    """

    base = Path(root)
    names = set(exclude_names)
    suffixes = tuple(exclude_suffixes)
    lines: list[str] = []
    total = 0
    for current, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in names)
        for name in sorted(files):
            if name in names or name.endswith(suffixes):
                continue
            path = Path(current) / name
            rel = path.relative_to(base).as_posix()
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                lines.append(f"{rel}\tlink:{os.readlink(path)}")
            elif stat.S_ISREG(info.st_mode):
                lines.append(f"{rel}\t{sha256_file(path)}")
                total += info.st_size
            else:
                lines.append(f"{rel}\tspecial:{stat.S_IFMT(info.st_mode)}")
        for name in dirs:
            path = Path(current) / name
            if os.path.islink(path):
                lines.append(f"{path.relative_to(base).as_posix()}/\tlink:{os.readlink(path)}")
    return {"kind": "dir", "sha256": sha256_text("\n".join(lines)), "files": len(lines),
            "bytes": total}


# --------------------------------------------------------------------------- files


def atomic_write_bytes(path: str | Path, data: bytes, mode: int = 0o644) -> None:
    common.atomic_write_bytes(path, data, mode=mode)


def atomic_write_text(path: str | Path, text: str, mode: int = 0o644) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode)


def atomic_write_json(path: str | Path, value: Any, mode: int = 0o644) -> None:
    atomic_write_text(path, json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False,
                                       allow_nan=False) + "\n", mode)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records = []
    for number, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"{target}:{number}: invalid JSON line: {exc}") from exc
        if not isinstance(value, dict):
            raise HarnessError(f"{target}:{number}: ledger line is not an object")
        records.append(value)
    return records


def safe_relpath(value: Any) -> str:
    """Validate an agent-supplied relative path (no absolute, no ``..``, no NUL)."""

    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 512:
        raise UnsafePathError("path must be a non-empty relative string")
    if value.startswith("/") or value.startswith("~"):
        raise UnsafePathError(f"path must be relative: {value!r}")
    parts = [p for p in value.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise UnsafePathError(f"path must stay inside its root: {value!r}")
    return "/".join(parts)


def read_regular_nofollow(root: str | Path, relpath: str, *, max_bytes: int) -> bytes:
    """Read ``root/relpath`` without following a symlink at any component.

    The agent owns the directories below ``root``; a symlink there could point the
    harness (which runs outside the sandbox) at a protected file, so every component is
    opened with ``O_NOFOLLOW`` relative to its parent descriptor.
    """

    parts = safe_relpath(relpath).split("/")
    dir_fd = os.open(os.path.realpath(root), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                  dir_fd=dir_fd)
            except OSError as exc:
                raise UnsafePathError(f"{relpath}: not a plain directory path ({exc.strerror})") \
                    from exc
            os.close(dir_fd)
            dir_fd = next_fd
        try:
            file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                              dir_fd=dir_fd)
        except OSError as exc:
            raise UnsafePathError(f"{relpath}: cannot open as a regular file ({exc.strerror})") \
                from exc
        try:
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode):
                raise UnsafePathError(f"{relpath}: not a regular file")
            if info.st_size > max_bytes:
                raise UnsafePathError(f"{relpath}: {info.st_size} bytes exceeds {max_bytes}")
            chunks = []
            remaining = max_bytes + 1
            while remaining > 0:
                block = os.read(file_fd, min(1 << 20, remaining))
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
            data = b"".join(chunks)
            if len(data) > max_bytes:
                raise UnsafePathError(f"{relpath}: grew beyond {max_bytes} bytes while reading")
            return data
        finally:
            os.close(file_fd)
    finally:
        os.close(dir_fd)


# --------------------------------------------------------------------------- templates

_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_.:-]*)\}")


def expand(template: str, values: Mapping[str, Any], *, keep_env: bool = False) -> str:
    """Replace ``{name}`` placeholders; ``{{``/``}}`` are literal braces.

    Unknown names raise :class:`ConfigError` (fail closed). With ``keep_env`` a
    ``{env:NAME}`` placeholder is left for the shell as ``${NAME}``.
    """

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        name = match.group(1)
        if name.startswith("env:"):
            if keep_env:
                return "${" + name[4:] + "}"
            if name[4:] in os.environ:
                return os.environ[name[4:]]
            raise ConfigError(f"environment variable {name[4:]} is not set")
        if name not in values:
            raise ConfigError(f"unknown placeholder {{{name}}} in {template!r}")
        return str(values[name])

    return _PLACEHOLDER.sub(replace, template)


def expand_all(value: Any, values: Mapping[str, Any], *, keep_env: bool = False) -> Any:
    if isinstance(value, str):
        return expand(value, values, keep_env=keep_env)
    if isinstance(value, list):
        return [expand_all(item, values, keep_env=keep_env) for item in value]
    if isinstance(value, dict):
        return {key: expand_all(item, values, keep_env=keep_env) for key, item in value.items()}
    return value


def get_dotted(value: Any, dotted: str) -> Any:
    """``get_dotted({"a": {"b": [1, 2]}}, "a.b.1") == 2``; missing -> ``None``."""

    current = value
    for part in dotted.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def clip_text(value: Any, limit: int = 1000) -> str:
    """Printable, length-limited text for agent-supplied notes."""

    text = "" if value is None else str(value)
    text = "".join(ch if ch.isprintable() or ch in "\n\t" else " " for ch in text)
    return text if len(text) <= limit else text[: limit - 1] + "…"
