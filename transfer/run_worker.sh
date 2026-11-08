#!/bin/sh
# One distillation (or translation) session for DIR, made by a stage's prep.py + brief.py: one
# headless Claude Code session reads DIR/BRIEF.md, may read DIR and write only DIR/out, with the
# tools Read, Glob, Grep, Write, Edit and the shell commands ls, wc and python3 -c; effort high.
#
#     sh transfer/run_worker.sh --api DIR   # robodojo/: the brief points at the API source,
#                                           # $CE_ROOT/src/pyrualean, which is readable too
#     sh transfer/run_worker.sh DIR         # xbench/, robocasa/: the API is copied into DIR/inputs
#
# `source tools/env.sh` first.  The CLI is $CLAUDE_BIN (default claude), with its own login or the
# caller's ANTHROPIC_API_KEY; the model is $LEARNER_MODEL (default claude-opus-5-5).  The session
# reads no user settings (--setting-sources project,local) and gets settings.json here: no hooks,
# no auto-memory, no git instructions, no plugins.  Output: DIR/worker.stream.jsonl (stream-json),
# DIR/worker.stderr, DIR/worker.exit; a DIR that has worker.exit is not run again.
set -eu
[ -n "${CE_ROOT:-}" ] || { echo "CE_ROOT is not set: source tools/env.sh" >&2; exit 2; }
api=
if [ "${1:-}" = "--api" ]; then
    api=$(cd "$CE_ROOT/src/pyrualean" && pwd -P)
    shift
fi
[ $# -eq 1 ] || { echo "usage: sh run_worker.sh [--api] DIR" >&2; exit 2; }
d=$(cd "$1" && pwd -P)
settings=$CE_ROOT/transfer/settings.json
[ ! -e "$d/worker.exit" ] || { echo "$d/worker.exit exists: already run" >&2; exit 2; }
[ -f "$d/BRIEF.md" ] && [ -d "$d/inputs" ] || { echo "$d has no BRIEF.md or inputs/" >&2; exit 2; }
if [ -n "$api" ]; then
    set -- "Read(/$api/**)" "Glob(/$api/**)" "Grep(/$api/**)" --add-dir "$api"
else
    set --
fi
cd "$d"
mkdir -p out
"${CLAUDE_BIN:-claude}" -p "$(cat BRIEF.md)
Start now. End with a one-paragraph summary of what you wrote." \
  --model "${LEARNER_MODEL:-claude-opus-5-5}" \
  --setting-sources project,local --settings "$(cat "$settings")" \
  --output-format stream-json --verbose \
  --tools Read,Glob,Grep,Write,Edit,Bash --permission-mode dontAsk \
  --allowedTools "Read(/$d/**)" "Glob(/$d/**)" "Grep(/$d/**)" "Write(/$d/out/**)" "Edit(/$d/out/**)" \
     "Bash(ls:*)" "Bash(wc:*)" "Bash(python3 -c:*)" "$@" \
  --no-session-persistence --effort high \
  < /dev/null > "$d/worker.stream.jsonl" 2> "$d/worker.stderr" && rc=0 || rc=$?
echo "exit $rc" > "$d/worker.exit"
