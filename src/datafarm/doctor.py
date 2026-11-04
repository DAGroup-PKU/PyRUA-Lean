"""Check installation prerequisites without API or GPU calls."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

from datafarm.runtime import expand, load_bindings, source_root

PROFILES = ("core", "seedance", "training", "h3-control", "h3-ood")


def doctor(root: Path, profile: str) -> dict:
    bindings = load_bindings(root)
    checks = []
    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": "ok" if passed else detail})
    check("python", sys.version_info >= (3, 11), "Python >= 3.11 required")
    for name in ("configs", "tools", "vendor"):
        check(name, (root / name).is_dir(), f"missing {name}/: run from a complete source checkout")
    if profile != "core":
        for name in ("ffmpeg", "ffprobe"):
            check(name, shutil.which(name), "required media executable")
        for name in ("numpy", "cv2", "requests", "h5py", "pyarrow", "PIL"):
            check(name, importlib.util.find_spec(name) is not None, "install the workflow extra")
    required = {
        "core": [], "seedance": ["ARK_CREDENTIALS"],
        "training": ["TRAINING_PYTHON", "TRAINER", "OPENPI_CACHE", "GPU_SCHEDULER"],
        "h3-control": ["H3_PYTHON", "H3_MODELS", "H3_VIDEOX_FUN", "H3_TURBO_LORA", "H3_EVAL_DINO", "GPU_SCHEDULER"],
        "h3-ood": ["H3_PYTHON", "H3_SAM2"],
    }[profile]
    for name in required:
        path = Path(bindings[name])
        check(name, path.exists(), str(path))
        if name.endswith("CREDENTIALS") and path.exists():
            check(name + "-permissions", path.stat().st_mode & 0o077 == 0, "file must be mode 0600")
    manifest = root / "configs/external-assets.json"
    if manifest.is_file():
        from datafarm.core.source_lock import sha256_file
        for entry in json.loads(manifest.read_text())["files"]:
            if profile not in entry["profiles"]:
                continue
            path = Path(expand(entry["path"], bindings))
            passed = path.is_file() and sha256_file(path) == entry["sha256"]
            check("asset:" + entry["name"], passed, "exact file required: " + str(path))
    return {"schema": "datafarm.release-doctor/1", "profile": profile,
            "passed": all(c["passed"] for c in checks), "checks": checks,
            "scope": "installation prerequisites only; not scientific or production qualification"}


def add_parsers(subparsers) -> None:
    doc = subparsers.add_parser("doctor", help="check installation prerequisites without API/GPU calls")
    doc.add_argument("--workspace", type=Path, default=None, help="default: this checkout")
    doc.add_argument("--profile", choices=PROFILES, default="core")


def handle(args) -> int:
    result = doctor((args.workspace or source_root()).resolve(), args.profile)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 2
