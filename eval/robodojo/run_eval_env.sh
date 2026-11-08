#!/usr/bin/env bash
# Start one evaluation (eval/robodojo/run.py) whose policy key comes from the environment:
#   eval/robodojo/run_eval_env.sh <root> [--tasks ...]
# The exported config names it (execution.api_key_env); it must be set in the calling shell
# (none for execution.runtime codex with its own login: api_key_env null).
# Detached (setsid, output to <root>/run.log); re-running resumes the evaluation.
set -euo pipefail
root="$(cd "$1" && pwd)"; shift
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../tools/env.sh"
if pgrep -u "$(id -u)" -f "eval/robodojo/run.py $root run" >/dev/null; then
  echo "already running: $root" >&2; exit 1
fi
name=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["execution"]["api_key_env"] or "")' "$root/config.json")
if [ -n "$name" ]; then
  if [ -z "${!name:-}" ]; then echo "environment variable $name is empty" >&2; exit 1; fi
  echo "$name: ...${!name: -4}"
else
  # execution.runtime codex without base_url: the login of execution.codex_home (run.py checks it)
  echo "no API key: the codex login of execution.codex_home"
fi
cd "$root"
nohup setsid python "$here/run.py" "$root" run "$@" >> "$root/run.log" 2>&1 < /dev/null &
echo "started $root (pid $!)"
