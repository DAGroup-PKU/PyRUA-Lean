#!/usr/bin/env python3
"""Command-line entry point of datafarm.training.launch (formal PI0.5 training preflight and launch)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datafarm.training.launch import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
