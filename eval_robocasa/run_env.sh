#!/usr/bin/env bash
# Start one RoboCasa365 evaluation (eval_robocasa/run.py) detached: eval_robocasa/run_env.sh <root> [run args]
# Runs in the PyRUA-Lean Python the root's config names (paths.python); output to
# <root>/run.log.  Re-running resumes.  The model key is read from the variable the config names
# (execution.api_key_env), exported in this shell; the codex runtime without base_url logs in on
# execution.codex_home by itself.
set -euo pipefail
root="$(cd "$1" && pwd)"; shift
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["python"])' "$root/config.json")
if pgrep -u "$(id -u)" -f "eval_robocasa/run.py $root run" >/dev/null; then
  echo "already running: $root" >&2; exit 1
fi
unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY CUDA_VISIBLE_DEVICES
cd "$root"
nohup setsid "$python" "$here/run.py" "$root" run "$@" >> "$root/run.log" 2>&1 < /dev/null &
echo "started $root (pid $!)"
