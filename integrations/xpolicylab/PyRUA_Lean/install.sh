#!/bin/bash
# The policy environment of PyRUA_Lean: XPolicyLab's policy server, PyRUA-Lean, and SAM3, the
# one model the policy side loads.  Run it inside the environment the policy server will use
# (<policy_env> of eval.sh); that environment needs a CUDA build of torch first.
#
#   bash install.sh [<PyRUA-Lean checkout or pip requirement>]
#
# The simulator's environment needs nothing beyond RoboDojo's own installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYRUALEAN_SOURCE="${1:-${PYRUALEAN_SOURCE:-git+https://github.com/DAGroup-PKU/PyRUA-Lean.git}}"
# The revision of the sam3 package the masks were measured with.
SAM3_SOURCE="${SAM3_SOURCE:-git+https://github.com/RLinf/sam3.git@c1f6b1e83967c912f675d8975d437ccdc41e17d8}"

if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[install] this environment needs a CUDA build of torch before SAM3 can load" >&2
    exit 1
fi

# XPolicyLab itself: the websocket policy server and the image decoding it does.
python -m pip install -e "${XPL_ROOT}"
if [[ -d "${PYRUALEAN_SOURCE}" ]]; then
    python -m pip install -e "${PYRUALEAN_SOURCE}"
else
    python -m pip install "pyrualean @ ${PYRUALEAN_SOURCE}"
fi
# What PyRUA-Lean's episode driver imports (it declares no dependencies of its own).
python -m pip install "numpy>=1.23" "pillow" "pyyaml>=6" "httpx>=0.27" "mcp>=1.8,<2" "uvicorn"
python -m pip install "sam3 @ ${SAM3_SOURCE}"

echo "[install] done.  SAM3's checkpoint: bash ${SCRIPT_DIR}/download_assets.sh"
