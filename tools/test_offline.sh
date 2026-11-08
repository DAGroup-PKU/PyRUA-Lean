#!/usr/bin/env bash
# The offline tests: no simulator, no GPU, no model.  Deployment variables are cleared so that no
# test reaches a real checkout, run directory or model service; a `claude` or `codex` on PATH is
# never called.  Usage: tools/test_offline.sh [python]   (default: python3 on PATH)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${1:-python3}"
run() {
  env -i HOME="$HOME" PATH="$(dirname "$(command -v "$python")"):/usr/bin:/bin" \
    PYTHONPATH="$here/src" PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    "$python" -m pytest -q -p no:cacheprovider "$@"
}
cd "$here"
run tests
# each of these has its own `import run`: one pytest process per directory
for d in eval eval_libero eval_robocasa; do
  run --rootdir="$d" "$d/tests"
done
