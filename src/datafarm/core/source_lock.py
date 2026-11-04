"""Verify exported source trees without requiring a developer's Git checkout."""
from __future__ import annotations

from datafarm.common import sha256_file
import json
from pathlib import Path


def verify_source_lock(root: Path, manifest: Path, expected_sha256: str) -> dict:
    if sha256_file(manifest) != expected_sha256:
        raise ValueError("source lock manifest digest mismatch")
    value = json.loads(manifest.read_text())
    if value.get("schema") != "datafarm.source-lock/1" or not value.get("files"):
        raise ValueError("source lock must contain a nonempty file manifest")
    root = root.resolve()
    seen = set()
    for record in value["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("unsafe or duplicate source lock path")
        seen.add(str(relative))
        path = root / relative
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("source lock cannot resolve outside its tree or through a symlink")
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"source lock mismatch: {relative}")
    actual = {str(p.relative_to(root)) for folder in ("src", "scripts")
              for p in (root / folder).rglob("*.py")}
    if not actual.issubset(seen):
        raise ValueError("unlocked Python source present in exported backend")
    return value
