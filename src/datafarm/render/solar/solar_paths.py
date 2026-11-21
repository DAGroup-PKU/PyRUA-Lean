#!/usr/bin/env python3
"""Deployment bindings for the SolarRenderer route (stdlib only).

No machine defaults: each value is resolved when a tool needs it, and a missing or non-existent value fails closed.
Resolution order is the same as the H3 harness: env <KEY>, else env DATAFARM_<KEY>, else an explicit entry of the runtime
bindings file (DATAFARM_BINDINGS, else <DATAFARM_WORKSPACE or this checkout>/bindings.json).

Keys:
  SOLAR_PYTHON   interpreter with SolarRenderer's dependencies (torch, diffusers, transformers, solarwm); default: the
                 current interpreter
  SOLAR_REPO     SolarRenderer checkout (https://github.com/Enderfga/SolarRenderer, pinned in configs/external-assets.json)
  SOLAR_H3       MiniMax-H3 directory prepared by SolarRenderer scripts/prepare_h3.py (transformer_ref/, text_encoder/,
                 vae/, ...)
  SOLAR_LORA     directory holding solarrenderer_{control,4step}_lora.safetensors (and the causal pair for --mode causal)

  python3 solar_paths.py get SOLAR_REPO   # prints the resolved value or exits 2 with the reason
  python3 solar_paths.py show             # every key: value or why it is unresolved
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]  # src/datafarm/render/solar -> the datafarm source checkout
BINDING_KEYS = ("SOLAR_PYTHON", "SOLAR_REPO", "SOLAR_H3", "SOLAR_LORA")
TOKEN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
LORA_PAIRS = {"bidir": ("solarrenderer_control_lora", "solarrenderer_4step_lora"),
              "causal": ("solarrenderer_causal_control_lora", "solarrenderer_causal_4step_lora")}


class MissingBinding(SystemExit):
    """A required deployment value is absent or does not exist (raised as SystemExit so CLIs stop cleanly)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


def _bindings_file() -> dict:
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from datafarm.runtime import explicit_bindings
    try:
        return explicit_bindings()
    except ValueError as e:
        raise MissingBinding(str(e)) from e


def binding(key: str, *, required: bool = True, must_exist: bool = True) -> str | None:
    if key not in BINDING_KEYS:
        raise MissingBinding(f"unknown SolarRenderer binding {key}; known: {', '.join(BINDING_KEYS)}")
    value = os.environ.get(key) or os.environ.get("DATAFARM_" + key) or _bindings_file().get(key)
    if not value:
        if required:
            raise MissingBinding(f"{key} is not set: export {key}=<path> (or DATAFARM_{key}, or add it to the runtime "
                                 f"bindings file); see src/datafarm/render/solar/README.md")
        return None
    if TOKEN.search(value):
        raise MissingBinding(f"{key}={value} still contains an unexpanded ${{...}} binding")
    if must_exist and not Path(value).exists():
        raise MissingBinding(f"{key}={value} does not exist")
    return value


def python_exe() -> str:
    return binding("SOLAR_PYTHON", required=False) or sys.executable


def solar_repo() -> Path:
    p = Path(binding("SOLAR_REPO"))
    if not (p / "scripts" / "render.py").is_file():
        raise MissingBinding(f"SOLAR_REPO={p} has no scripts/render.py (not a SolarRenderer checkout)")
    return p


def caption_skill() -> Path:
    p = solar_repo() / "skills" / "solarrenderer-caption"
    if not (p / "check.py").is_file():
        raise MissingBinding(f"{p} has no check.py (SolarRenderer caption skill missing)")
    return p


def h3_weights() -> Path:
    p = Path(binding("SOLAR_H3"))
    if not (p / "transformer_ref").is_dir():
        raise MissingBinding(f"SOLAR_H3={p} has no transformer_ref/ (run SolarRenderer scripts/prepare_h3.py)")
    return p


def lora_pair(mode: str = "bidir") -> tuple[Path, Path]:
    root = Path(binding("SOLAR_LORA"))
    out = tuple(root / f"{name}.safetensors" for name in LORA_PAIRS[mode])
    for p in out:
        if not p.is_file():
            raise MissingBinding(f"SOLAR_LORA={root} has no {p.name}")
    return out


def tool(name: str) -> Path:
    return HERE / name


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="resolve SolarRenderer deployment bindings")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get")
    g.add_argument("key", choices=BINDING_KEYS)
    sub.add_parser("show")
    a = ap.parse_args(argv)
    getters = {"SOLAR_PYTHON": python_exe, "SOLAR_REPO": solar_repo, "SOLAR_H3": h3_weights,
               "SOLAR_LORA": lambda: lora_pair("bidir")[0].parent}
    if a.cmd == "get":
        try:
            print(getters[a.key]())
        except MissingBinding as e:
            print(e, file=sys.stderr)
            return 2
        return 0
    for key in BINDING_KEYS:
        try:
            print(f"{key}={getters[key]()}")
        except MissingBinding as e:
            print(f"{key}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
