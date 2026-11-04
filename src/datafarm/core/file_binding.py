"""Bound file references: a ``{path, sha256}`` record must resolve to the exact bytes it names."""
from __future__ import annotations

from pathlib import Path

from datafarm.common import sha256_file


class ContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def bound_file(root: Path, ref: dict, *, external: bool = False) -> Path:
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str), 'invalid file reference')
    raw = Path(ref['path'])
    p = (root / raw).resolve(strict=True)
    require(external or (not raw.is_absolute() and p.is_relative_to(root.resolve())),
            'artifact escapes task workspace')
    require(p.is_file() and p.stat().st_size > 0, 'artifact must be a nonempty file')
    require(sha256_file(p) == ref.get('sha256'), f'artifact changed: {ref["path"]}')
    return p
