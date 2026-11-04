#!/usr/bin/env python3
"""Task-expert authoring harness CLI of this checkout; see docs/reference/expert-harness.md.

  tools/expert_harness.py lint-brief --adapter configs/expert-harness/template.json
  tools/expert_harness.py dry-run
  tools/expert_harness.py prepare-round | run-round | status | eval-holdout | handoff | suite ...

run-round and eval-holdout refuse to spend API money or GPU time without --approve-agent-spend / --approve-gpu.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))
sys.path.insert(0, str(ROOT / "src"))

from datafarm.expert.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
