"""Portable runtime bindings. Values are paths, never credentials themselves."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

TOKEN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def defaults(workspace: Path) -> dict[str, str]:
    w = str(workspace.resolve())
    return {
        "WORKSPACE": w, "TRAINER": w + "/vendor/trainer",
        "GPU_SCHEDULER": w + "/external/gpu-scheduler", "DATASETS": w + "/data/lerobot",
        "MODELS": w + "/external/models", "OPENPI_CACHE": w + "/external/openpi-cache",
        "ROBOTWIN": w + "/external/robotwin", "EXTERNAL": w + "/external",
        "TRAINING_RUN": w + "/runs/training",
        "ISAAC_PYTHON": w + "/environments/isaac/bin/python",
        "ISAAC_KIT": w + "/environments/isaac/lib/python3.11/site-packages/isaacsim/kit",
        "ANALYSIS_PYTHON": sys.executable, "ANALYSIS_ENV": sys.prefix,
        "TRAINING_PYTHON": w + "/environments/training/bin/python",
        "H3_PYTHON": w + "/environments/h3/bin/python", 
        # H3 control hybrid (src/datafarm/render/h3): model root, patched VideoX-Fun checkout, turbo LoRA,
        # EVAL-v1 DINOv2 and OOD SAM2.1 weights, optional HED annotator source; the harness itself never defaults them
        "H3_MODELS": w + "/external/h3-models", "H3_VIDEOX_FUN": w + "/external/videox-fun",
        "H3_TURBO_LORA": w + "/external/h3-models/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
        "H3_EVAL_DINO": w + "/external/models/dinov2-base", "H3_SAM2": w + "/external/models/sam2.1-hiera-large",
        "H3_HED_ANNOTATOR_REPO": w + "/external/controlnet-annotator",
        "ARK_CREDENTIALS": w + "/secrets/ark.env", "AGENT_CREDENTIALS": w + "/secrets/agent.env",
        "ISAAC_ENV": w + "/environments/isaac", "ROBOTWIN_PYTHON": w + "/environments/robotwin/bin/python",
        "EPISODES": w + "/data/episodes",
    }


def expand(value, bindings: dict[str, str]):
    if isinstance(value, str):
        def replace(match):
            key = match[1]
            if key not in bindings:
                raise ValueError(f"missing runtime binding: {key}")
            return str(bindings[key])
        result = TOKEN.sub(replace, value)
        if TOKEN.search(result):
            raise ValueError("nested runtime bindings are not supported")
        return result
    if isinstance(value, list):
        return [expand(item, bindings) for item in value]
    if isinstance(value, dict):
        return {expand(key, bindings): expand(item, bindings) for key, item in value.items()}
    return value


def explicit_bindings(workspace: Path | None = None) -> dict[str, str]:
    """Entries of the runtime bindings file only, without defaults or environment overrides."""
    root = workspace or Path(os.environ.get("DATAFARM_WORKSPACE", source_root()))
    path = Path(os.environ.get("DATAFARM_BINDINGS", root / "bindings.json"))
    if not path.is_file():
        return {}
    supplied = json.loads(path.read_text())
    if not isinstance(supplied, dict) or any(not isinstance(v, str) for v in supplied.values()):
        raise ValueError("bindings must be an object of path strings")
    return supplied


def load_bindings(workspace: Path | None = None) -> dict[str, str]:
    root = workspace or Path(os.environ.get("DATAFARM_WORKSPACE", source_root()))
    values = defaults(root)
    supplied = explicit_bindings(root)
    unknown = set(supplied) - set(values)
    if unknown:
        raise ValueError(f"unknown runtime bindings: {sorted(unknown)}")
    values.update(supplied)
    values.update({key: os.environ["DATAFARM_" + key] for key in values if "DATAFARM_" + key in os.environ})
    return values


def runtime_path(template: str) -> str:
    return expand(template, load_bindings())
