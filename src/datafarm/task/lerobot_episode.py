"""One episode of a LeRobot v2.x dataset as a self-contained task input folder.

`pipeline_agent.py prepare --episode <dir>` hashes every file under the folder, so a whole dataset is the wrong
input.  This module copies (hard-links when it can) one episode's camera videos, its parquet log, the dataset's
`meta/` description and an optional robot package into a new folder, and writes `episode.json`: where each file came
from, its sha256, the camera mapping, the fps, the task text and the state and action fields.  It does not interpret
the robot's joints; grounding does, against the real views.

Layout read (codebase_version v2.0 / v2.1): meta/info.json (`data_path`, `video_path`, `chunks_size`, `fps`,
`features`), meta/episodes.jsonl (`episode_index`, `tasks`, `length`), meta/tasks.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

SCHEMA = "datafarm.lerobot-episode/1"
SUPPORTED = ("v2.0", "v2.1")


class EpisodeImportError(ValueError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parquet_columns(path: Path) -> dict | None:
    """Rows and column names of the episode log, or None when pyarrow is not installed."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    try:
        meta = pq.ParquetFile(path).metadata
    except Exception as e:  # noqa: BLE001 - an unreadable log is reported, the import still records the file
        return {"rows": None, "columns": None, "error": f"{type(e).__name__}: {e}"}
    return {"rows": meta.num_rows, "columns": [meta.schema.column(i).name for i in range(meta.num_columns)]}


def import_episode(dataset: Path, episode: int, out: Path, cameras: dict | None = None,
                   robot_package: Path | None = None) -> dict:
    """Write the episode folder; `cameras` maps an output camera name to a dataset video key or a video file."""
    dataset, out = Path(dataset), Path(out)
    if out.exists() and any(out.iterdir()):
        raise EpisodeImportError(f"{out} exists and is not empty")
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise EpisodeImportError(f"{dataset}: no meta/info.json (not a LeRobot dataset root)")
    info = json.loads(info_path.read_text())
    version = str(info.get("codebase_version", ""))
    if version not in SUPPORTED:
        raise EpisodeImportError(f"{dataset}: codebase_version {version!r}; supported: {', '.join(SUPPORTED)}")
    chunk = episode // int(info.get("chunks_size") or 1000)
    data = dataset / info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    if not data.is_file():
        raise EpisodeImportError(f"episode {episode}: {data} does not exist")
    features = info.get("features") or {}
    video_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    if cameras is None:
        cameras = {k.split(".")[-1]: k for k in video_keys}
    files, mapping = {}, {}
    for name, source in cameras.items():
        if Path(source).is_file():
            src, key = Path(source), None
        elif source in video_keys:
            if not info.get("video_path"):
                raise EpisodeImportError(f"{dataset}: video key {source} but no video_path in meta/info.json")
            src, key = dataset / info["video_path"].format(episode_chunk=chunk, episode_index=episode,
                                                         video_key=source), source
        else:
            raise EpisodeImportError(f"camera {name}: {source!r} is neither a file nor a video key {video_keys}")
        if not src.is_file():
            raise EpisodeImportError(f"camera {name}: {src} does not exist")
        dst = out / "videos" / f"{name}.mp4"
        _place(src, dst)
        files[str(dst.relative_to(out))] = {"source": str(src.resolve()), "sha256": _sha256(dst)}
        mapping[name] = {"video_key": key, "file": str(dst.relative_to(out))}
    dst = out / "data" / data.name
    _place(data, dst)
    files[str(dst.relative_to(out))] = {"source": str(data.resolve()), "sha256": _sha256(dst)}
    for f in ("info.json", "camera.json", "tasks.jsonl", "stats.json", "modality.json"):
        if (dataset / "meta" / f).is_file():
            _place(dataset / "meta" / f, out / "meta" / f)
            files[f"meta/{f}"] = {"source": str((dataset / "meta" / f).resolve()), "sha256": _sha256(out / "meta" / f)}
    row = next((r for r in _jsonl(dataset / "meta" / "episodes.jsonl") if r.get("episode_index") == episode), None)
    if row is not None:
        (out / "meta" / "episode.jsonl").write_text(json.dumps(row) + "\n")
    if robot_package is not None:
        robot_package = Path(robot_package)
        dst = out / "robot"
        if robot_package.is_dir():
            shutil.copytree(robot_package, dst, copy_function=os.link if _same_fs(robot_package, out) else shutil.copy2)
        else:
            _place(robot_package, dst / robot_package.name)
    log = parquet_columns(out / "data" / data.name)
    names = {k: {"shape": v.get("shape"), "names": v.get("names")} for k, v in features.items()
             if isinstance(v, dict) and v.get("dtype") != "video"}
    record = {"schema": SCHEMA, "dataset": str(dataset.resolve()), "codebase_version": version,
              "episode_index": episode, "fps": info.get("fps"), "robot_type": info.get("robot_type"),
              "tasks": (row or {}).get("tasks"), "length": (row or {}).get("length"), "cameras": mapping,
              "log": {"file": str(Path("data") / data.name), **(log or {"rows": None, "columns": None})},
              "features": names, "robot_package": "robot" if robot_package is not None else None, "files": files}
    (out / "episode.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    return record


def _same_fs(a: Path, b: Path) -> bool:
    b.mkdir(parents=True, exist_ok=True)
    return os.stat(a).st_dev == os.stat(b).st_dev
