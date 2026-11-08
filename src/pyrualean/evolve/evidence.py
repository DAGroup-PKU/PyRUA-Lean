"""Read-only, explicitly registered model-visible artifacts from one task's dev episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

READ_EVIDENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_dev_evidence",
        "description": (
            "Read a catalogued dev artifact, including older attempts. Text is paged; "
            "images are returned visually. No val/test, scores, shell or robot access."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "length": {"type": "integer", "minimum": 1},
            },
            "required": ["artifact"],
            "additionalProperties": False,
        },
    },
}


class DevEvidence:
    def __init__(self, run_dirs: list[Path], task: str) -> None:
        self.files: dict[str, tuple[Path, str]] = {}
        self.episode_ids: set[str] = set()
        self.catalog: list[dict[str, Any]] = []
        for directory in dict.fromkeys(path.resolve() for path in run_dirs):
            result = json.loads((directory / "result.json").read_text())
            if result.get("split") != "dev" or result.get("task") != task:
                raise ValueError("reflection evidence must belong to this task's dev split")
            episode = result.get("episode_id", directory.parent.parent.name)
            self.episode_ids.add(episode)
            paths = [
                directory / name
                for name in (
                    "prompt.txt",
                    "calls.jsonl",
                    "tool_turns.json",
                    "messages.json",
                    "events.jsonl",
                )
            ]
            paths += sorted((directory / "workspace").glob("cell_*.py"))
            paths += sorted((directory / "images").glob("*.png"))
            for path in paths:
                if not path.is_file():
                    continue
                relative = str(path.relative_to(directory))
                artifact = f"{episode}/{relative}"
                self._path(directory, relative)
                if artifact in self.files and self.files[artifact] != (directory, relative):
                    raise ValueError("duplicate dev artifact ID")
                self.files[artifact] = (directory, relative)
                self.catalog.append(
                    {
                        "artifact": artifact,
                        "kind": "image" if path.suffix == ".png" else "text",
                        "revision": result.get("revision"),
                        "bytes": path.stat().st_size,
                    }
                )

    @staticmethod
    def _path(directory: Path, relative: str) -> Path:
        path = directory / relative
        if not path.resolve().is_relative_to(directory) or path.is_symlink():
            raise ValueError("dev artifact escaped its registered episode")
        return path

    def read(
        self, artifact: str, *, offset: int = 0, length: int = 20000
    ) -> tuple[dict[str, Any], bytes | None]:
        if artifact not in self.files:
            raise ValueError("unknown dev artifact; use an exact ID from the catalog")
        if type(offset) is not int or offset < 0 or type(length) is not int or length < 1:
            raise ValueError("offset must be nonnegative and length positive")
        directory, relative = self.files[artifact]
        path = self._path(directory, relative)
        if path.suffix == ".png":
            return {"artifact": artifact, "kind": "image"}, path.read_bytes()
        text = path.read_text()
        end = min(offset + length, len(text))
        return {
            "artifact": artifact,
            "kind": "text",
            "text": text[offset:end],
            "next_offset": end if end < len(text) else None,
            "total_characters": len(text),
        }, None
