#!/usr/bin/env bash
# Start one learn campaign detached from the terminal:
#   tools/run_learn.sh <campaign-root> [run options, e.g. --worker-scoring off]
# The coordinator runs in its own session (setsid), stdin from /dev/null, output appended to
# <root>/run.log.  The actor's key comes from the variable the config's execution.api_key_env
# names (e.g. ACTOR_API_KEY); the learner uses its CLI's own login, or ANTHROPIC_API_KEY.
# Re-running the same command resumes the campaign (finished episodes are reused).
set -euo pipefail
root="$(cd "$1" && pwd)"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh"
cd "$CE_ROOT"
if pgrep -u "$(id -u)" -f "pyrualean.learn.campaign $root run" >/dev/null; then
  echo "already running: $root" >&2; exit 1
fi
nohup setsid python -m pyrualean.learn.campaign "$root" run "${@:2}" \
  >> "$root/run.log" 2>&1 < /dev/null &
echo "started $root (pid $!); log: $root/run.log"
