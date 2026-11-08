"""Content-addressed private layouts. Official layouts are never an implicit fallback."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

TASKS = (
    "make_toast_random",
    "hang_mugs_random",
    "sweep_blocks_random",
    "plug_in_charger",
    "swap_blocks",
    "make_kong",
    "pour_by_language",
)
FORBIDDEN = {"Eval_Layout", "Train_Layout", "Traj", "eval_result"}
PHYSICS_CHECKS = ("startup", "stable", "objects_valid", "instruction_matches")


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def tree_hash(root: Path) -> str:
    return json_hash(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }
    )


class LayoutRegistry:
    def __init__(self, manifest: str | Path) -> None:
        self.path = Path(manifest).resolve()
        self.data = json.loads(self.path.read_text())
        self.rows = self.data["instances"]
        seen = set()
        keys = set()
        for row in self.rows:
            if row["task"] not in TASKS or row["split"] not in {"dev", "val", "heldout"}:
                raise ValueError("unknown task or split")
            key = (row["task"], row["split"], row["layout"])
            identity = (row["task"], row["content_sha256"])
            if key in keys or identity in seen:
                raise ValueError("duplicate layout content or index across splits")
            if type(row["layout"]) is not int or row["layout"] < 0:
                raise ValueError("invalid layout index")
            keys.add(key)
            seen.add(identity)
            self.file(row)

    def file(self, row: dict[str, Any]) -> Path:
        path = (self.path.parent / row["file"]).resolve()
        if not path.is_relative_to(self.path.parent) or FORBIDDEN.intersection(path.parts):
            raise ValueError("layout must be a private file inside the registry")
        if json_hash(json.loads(path.read_text())) != row["content_sha256"]:
            raise ValueError("layout content changed")
        return path

    def select(
        self, split: str, task: str, layout: int, *, require_physics: bool = True
    ) -> dict[str, Any]:
        found = [
            row
            for row in self.rows
            if (row["split"], row["task"], row["layout"]) == (split, task, layout)
        ]
        if len(found) != 1:
            raise ValueError("instance is absent from the declared split")
        row = found[0]
        self.file(row)
        if require_physics:
            receipt_path = (self.path.parent / row.get("physics_receipt", "MISSING")).resolve()
            if not receipt_path.is_relative_to(self.path.parent):
                raise ValueError("physics receipt escaped registry")
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("content_sha256") != row["content_sha256"]
                or receipt.get("env_config_sha256") != row["env_config_sha256"]
            ):
                raise ValueError("physics receipt identity mismatch")
            if not all(receipt.get("checks", {}).get(key) is True for key in PHYSICS_CHECKS):
                raise ValueError("physics/semantic validation has not passed")
            if not receipt.get("evidence") or not receipt.get("validator"):
                raise ValueError("physics validation requires evidence and validator identity")
        return row

    def identity(self, row: dict[str, Any]) -> dict[str, Any]:
        return {key: row[key] for key in ("task", "content_sha256", "env_config_sha256")}


def import_generated(
    batch: Path, output: Path, *, split: str, env_config: Path, previous: Path | None = None
) -> Path:
    """Import a guarded PhysicalRSI generation batch, without claiming physical validity."""
    batch = batch.resolve()
    generated = json.loads((batch / "manifest.json").read_text())
    expected = {"dev": "development", "val": "validation"}
    if split not in expected or generated["split"] != expected[split]:
        raise ValueError("generated batch split mismatch")
    if not generated.get("implementation"):
        raise ValueError("missing generator provenance")
    data = json.loads(previous.read_text()) if previous else {"instances": []}
    if previous and previous.parent.resolve() != output.resolve():
        raise ValueError("append only within the same registry root")
    output.mkdir(parents=True, exist_ok=True)
    for row in generated["entries"]:
        if row["state"] != "generated" or row.get("eligibility_blocker"):
            continue
        if any(FORBIDDEN.intersection(Path(name).parts) for name in row.get("inputs", {})):
            raise ValueError("generator accessed forbidden inputs")
        source = (batch / row["file"]).resolve()
        if not source.is_relative_to(batch):
            raise ValueError("batch path traversal")
        value = json.loads(source.read_text())
        content = json_hash(value)
        if content != row.get("content_sha256"):
            raise ValueError("generated layout changed before import")
        target = output / split / row["task"] / f"{row['index']}.json"
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False) + "\n")
        data["instances"].append(
            {
                "task": row["task"],
                "split": split,
                "layout": row["index"],
                "file": str(target.relative_to(output)),
                "content_sha256": content,
                "env_config_sha256": tree_hash(env_config),
                "generator": generated["implementation"],
                "inputs": row.get("inputs", {}),
                "physics_receipt": str(target.with_suffix(".physics.json").relative_to(output)),
            }
        )
    candidate = output / "manifest.pending.json"
    candidate.write_text(json.dumps(data, indent=2) + "\n")
    LayoutRegistry(candidate)
    candidate.replace(output / "manifest.json")
    return output / "manifest.json"


def private_assets(
    target: Path,
    assets: Path,
    layout: Path,
    *,
    task: str,
    suite: str,
    eval_seed: int,
    layout_index: int,
) -> None:
    """Expose meshes/textures and one private instance, never benchmark layouts/traces."""
    if target.exists() or target.is_symlink():
        raise FileExistsError("private Assets must start from an empty path")
    target.mkdir(parents=True)
    for path in assets.iterdir():
        if path.name not in FORBIDDEN:
            (target / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    folder = target / "Eval_Layout" / "RoboDojo" / suite / str(eval_seed)
    folder.mkdir(parents=True)
    shutil.copyfile(layout, folder / f"{task}_{layout_index}.json")
