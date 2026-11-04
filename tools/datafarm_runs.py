#!/usr/bin/env python3
"""Pipeline run ledger CLI of this checkout (a run pinned to a checkout keeps using that checkout's code).

The ledger defaults to <checkout>/runs/ledger (env DATAFARM_LEDGER_ROOT or --root overrides).  Host liveness probing uses
DATAFARM_SSH_CONFIG, DATAFARM_SSH_HOME and DATAFARM_LEASE_DIR when the deployment sets them.

  tools/datafarm_runs.py status             # open runs
  tools/datafarm_runs.py status RUN --gaps  # every item not yet passed at the exit column
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATAFARM_LEDGER_ROOT", str(ROOT / "runs" / "ledger"))
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))
sys.path.insert(0, str(ROOT / "src"))

from datafarm.ledger.runs_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
