#!/usr/bin/env python3
"""Deployment bindings for the H3 control-hybrid render tools (stdlib only).

The harness has no machine defaults: the run directory, the H3 interpreter, model/repo roots and the scheduler are resolved
here, at the moment a tool needs them, and a missing or non-existent value fails closed.

  run directory   --run-dir <dir>  or  env H3_RUN            (data only: tasks/, recipes/, pkgs/, queue/, cache/,
                                                              screen/, reviews/, eval/realpool/; never code)
  bindings        env <KEY>, else env DATAFARM_<KEY>, else an explicit entry of the runtime bindings file
                  (DATAFARM_BINDINGS, else <DATAFARM_WORKSPACE or this checkout>/bindings.json); no built-in default
                  except H3_PYTHON, which falls back to the current interpreter.

Keys: H3_PYTHON (python with torch/diffusers/VideoX-Fun deps), H3_MODELS (contains MiniMax-H3/Ref2VA and
MiniMax-H3-Fun-Controlnet-Union/), H3_VIDEOX_FUN (VideoX-Fun checkout with vendor/videox_fun applied),
H3_TURBO_LORA (Sol-H3 Turbo 4-step Ref2VA LoRA file), GPU_SCHEDULER (scheduler executable), H3_EVAL_DINO (DINOv2-base
directory for the frozen EVAL-v1 scorer), H3_SAM2 (sam2.1-hiera-large directory, OOD tools only),
H3_HED_ANNOTATOR_REPO (ControlNet annotator source read by the soft-HED tool of the patched VideoX-Fun).

  python3 h3_paths.py get H3_MODELS        # prints the resolved value or exits 2 with the reason
  python3 h3_paths.py get H3_RUN          # the run directory (env H3_RUN)
  python3 h3_paths.py show                 # every key: value or why it is unresolved
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]  # src/datafarm/render/h3 -> the datafarm source checkout that ships this harness
EVAL_DIR = REPO / "evaluations" / "render-h3"
BINDING_KEYS = ("H3_PYTHON", "H3_MODELS", "H3_VIDEOX_FUN", "H3_TURBO_LORA", "GPU_SCHEDULER", "H3_EVAL_DINO", "H3_SAM2",
                "H3_HED_ANNOTATOR_REPO")
TOKEN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_RUN_DIR: Path | None = None


class MissingBinding(SystemExit):
    """A required deployment value is absent or does not exist (raised as SystemExit so CLIs stop cleanly)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


def add_run_dir_arg(parser) -> None:
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="H3 control-hybrid run directory (default: env H3_RUN; required, no implicit default)")


def set_run_dir(path) -> Path:
    """Pin the run directory for this process (CLIs call it with --run-dir before doing any work)."""
    global _RUN_DIR
    if path is None:
        return run_dir()
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise MissingBinding(f"--run-dir {p} is not an existing directory")
    _RUN_DIR = p
    os.environ["H3_RUN"] = str(p)  # child tools (enqueue -> package builders) inherit the same run directory
    return p


def run_dir(arg=None) -> Path:
    if arg is not None:
        return set_run_dir(arg)
    if _RUN_DIR is not None:
        return _RUN_DIR
    value = os.environ.get("H3_RUN")
    if not value:
        raise MissingBinding("H3 control-hybrid run directory is not set: pass --run-dir <dir> or export H3_RUN")
    p = Path(value).expanduser().resolve()
    if not p.is_dir():
        raise MissingBinding(f"H3_RUN={value} is not an existing directory")
    return p


def _bindings_file() -> dict:
    """Explicit entries of the runtime bindings file only (never datafarm.runtime defaults)."""
    datafarm_src()  # imported here so `--help` and environment-only lookups never touch the bindings file
    from datafarm.runtime import explicit_bindings
    try:
        return explicit_bindings()
    except ValueError as e:
        raise MissingBinding(str(e)) from e


def binding(key: str, *, required: bool = True, must_exist: bool = True) -> str | None:
    """Resolved value of `key`; None only when the binding is absent and required=False (use require() otherwise)."""
    if key not in BINDING_KEYS:
        raise MissingBinding(f"unknown H3 control-hybrid binding {key}; known: {', '.join(BINDING_KEYS)}")
    value = os.environ.get(key) or os.environ.get("DATAFARM_" + key) or _bindings_file().get(key)
    if not value:
        if required:
            raise MissingBinding(f"{key} is not set: export {key}=<path> (or DATAFARM_{key}, or add it to the runtime "
                                 f"bindings file); see docs/reference/rendering.md")
        return None
    if TOKEN.search(value):
        raise MissingBinding(f"{key}={value} still contains an unexpanded ${{...}} binding")
    if must_exist and not Path(value).exists():
        raise MissingBinding(f"{key}={value} does not exist")
    return value


def require(key: str, *, must_exist: bool = True) -> str:
    """A required binding: always a non-empty string, otherwise MissingBinding (never None)."""
    return binding(key, required=True, must_exist=must_exist)


def python_exe() -> str:
    return binding("H3_PYTHON", required=False) or sys.executable


def models_root() -> Path:
    return Path(require("H3_MODELS"))


def videox_fun_repo() -> Path:
    return Path(require("H3_VIDEOX_FUN"))


def turbo_lora() -> Path:
    return Path(require("H3_TURBO_LORA"))


def gpu_scheduler() -> str:
    return require("GPU_SCHEDULER")


def eval_dino() -> str:
    return require("H3_EVAL_DINO")


def sam2_model() -> str:
    return require("H3_SAM2")


def eval_script() -> Path:
    return EVAL_DIR / "h3eval.py"


_MODEL_LAYOUT = {  # key -> (binding root, path inside it)
    "base_model": ("H3_MODELS", ("MiniMax-H3", "Ref2VA")),
    "control_checkpoint": ("H3_MODELS", ("MiniMax-H3-Fun-Controlnet-Union", "MiniMax-H3-Fun-Controlnet-Union.safetensors")),
    "control_config": ("H3_VIDEOX_FUN", ("config", "minimax_h3", "minimax_h3_control.yaml")),
    "videox_fun_repo": ("H3_VIDEOX_FUN", ()),
    "hed_tool": ("H3_VIDEOX_FUN", ("tools", "prepare_hed_control.py")),
    "hed_checkpoint": ("H3_VIDEOX_FUN", ("models", "annotators", "ControlNetHED.pth")),
}


def default_model_path(key: str) -> str:
    """One model/repo file the recipe's `control` section uses when it does not name it explicitly."""
    root, parts = _MODEL_LAYOUT[key]
    return str(Path(require(root), *parts))


def default_model_paths() -> dict:
    return {key: default_model_path(key) for key in _MODEL_LAYOUT}


def expand(value):
    """Replace ${KEY} binding placeholders (configs/render/h3-*.json) by resolved values; unknown keys fail."""
    if isinstance(value, str):
        def sub(match):
            return python_exe() if match[1] == "H3_PYTHON" else require(match[1])
        out = TOKEN.sub(sub, value)
        if TOKEN.search(out):
            raise MissingBinding(f"nested binding placeholders are not supported: {value}")
        return out
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def datafarm_src() -> Path:
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="resolve H3 control-hybrid deployment bindings")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get")
    g.add_argument("key", choices=(*BINDING_KEYS, "H3_RUN", "REPO", "EVAL"))
    sub.add_parser("show")
    a = ap.parse_args(argv)
    if a.cmd == "get":
        try:
            if a.key == "H3_RUN":
                print(run_dir())
            elif a.key == "REPO":
                print(REPO)
            elif a.key == "EVAL":
                print(eval_script())
            elif a.key == "H3_PYTHON":
                print(python_exe())
            else:
                print(require(a.key))
        except MissingBinding as e:
            print(e, file=sys.stderr)
            return 2
        return 0
    for key in ("H3_RUN", *BINDING_KEYS):
        try:
            print(f"{key}={run_dir() if key == 'H3_RUN' else python_exe() if key == 'H3_PYTHON' else require(key)}")
        except MissingBinding as e:
            print(f"{key} unresolved: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
