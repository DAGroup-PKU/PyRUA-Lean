#!/usr/bin/env python3
"""Scripted stand-in for the agent CLI (tests and ``tools/expert_harness.py dry-run``; no model calls).

It speaks the Claude Code ``stream-json`` format (system/init, assistant tool_use, user
tool_result, result with total_cost_usd/usage/num_turns/session_id), keeps its session
memory under ``$HOME`` like the real CLI, honours ``--session-id`` / ``--resume``, and
performs a JSON plan with the real ``xh`` tool inside the real sandbox.

Plan: ``{"sessions": [[step, ...], ...], "cost_per_session": 0.5}``; steps:
``{"op": "write", "path": "expert.py", "text": "..."}``,
``{"op": "xh", "args": [...], "expect": [0]}``,
``{"op": "probe", "paths": [...], "out": "isolation-probe.json"}``,
``{"op": "api_error", "times": N}`` (fail the first N invocations of this session),
``{"op": "auth_error"}``, ``{"op": "sleep", "seconds": S}``, ``{"op": "symlink", ...}``,
``{"op": "write_outside", "paths": [...]}``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid


def emit(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--session-index", type=int, required=True)
    parser.add_argument("--model", default="fake-model")
    parser.add_argument("--resume")
    parser.add_argument("--session-id")
    args = parser.parse_args()
    prompt = sys.stdin.read()
    with open(args.plan, encoding="utf-8") as handle:
        plan = json.load(handle)
    home = os.environ.get("HOME", ".")
    memory_path = os.path.join(home, ".fake-agent", "memory.json")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    try:
        with open(memory_path, encoding="utf-8") as handle:
            memory = json.load(handle)
    except (OSError, ValueError):
        memory = {"sessions": {}, "invocations": {}, "cursor": 0}
    if args.resume:
        if args.resume not in memory["sessions"]:
            emit({"type": "result", "subtype": "error_during_execution", "is_error": True,
                  "result": "No conversation found with session ID: " + args.resume,
                  "session_id": args.resume, "num_turns": 0, "total_cost_usd": 0.0,
                  "usage": {"input_tokens": 0, "output_tokens": 0}})
            return 1
        session_id = args.resume
    else:
        session_id = args.session_id or str(uuid.uuid4())
        memory["sessions"][session_id] = {"prompts": []}
    memory["sessions"][session_id]["prompts"].append(len(prompt))
    sessions_plan = plan.get("sessions", [])
    cursor = memory.get("cursor", 0)
    key = "plan-session-%d" % cursor  # repeated failures of one plan session share a counter
    memory["invocations"][key] = memory["invocations"].get(key, 0) + 1
    with open(memory_path, "w", encoding="utf-8") as handle:
        json.dump(memory, handle)  # like the real CLI, the conversation exists before any work
    emit({"type": "system", "subtype": "init", "session_id": session_id, "model": args.model,
          "cwd": os.getcwd(), "tools": ["Bash", "Read", "Edit", "Write"]})
    steps = sessions_plan[cursor] if cursor < len(sessions_plan) else [{"op": "noop"}]
    turns = 0
    tool = os.environ.get("XH_TOOL", "xh")
    for number, step in enumerate(steps):
        op = step.get("op")
        if op == "api_error" and memory["invocations"][key] <= int(step.get("times", 1)):
            with open(memory_path, "w", encoding="utf-8") as handle:
                json.dump(memory, handle)
            emit({"type": "result", "subtype": "success", "is_error": True, "num_turns": 1,
                  "result": 'API Error: 529 {"type":"error","error":{"type":"overloaded_error"}}',
                  "session_id": session_id, "total_cost_usd": 0.0,
                  "usage": {"input_tokens": 0, "output_tokens": 0}})
            return 1
        if op == "auth_error":
            emit({"type": "result", "subtype": "success", "is_error": True, "num_turns": 1,
                  "result": "Invalid API key · Please run /login", "session_id": session_id,
                  "total_cost_usd": 0.0, "usage": {"input_tokens": 0, "output_tokens": 0}})
            return 1
        if op in ("api_error", "noop"):
            continue
        tool_id = "toolu_%s_%d" % (uuid.uuid4().hex[:10], number)
        output = ""
        if op == "write":
            path = os.path.join(os.getcwd(), step["path"])
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(step["text"])
            output = "wrote %s" % step["path"]
        elif op == "xh":
            proc = subprocess.run([sys.executable, tool] + [str(a) for a in step["args"]],
                                  capture_output=True, text=True, timeout=600)
            output = proc.stdout[-3000:]
            if proc.returncode not in step.get("expect", [0]):
                output += "\nUNEXPECTED exit %d: %s" % (proc.returncode, proc.stderr[-500:])
                memory.setdefault("unexpected", []).append({"args": step["args"], "rc": proc.returncode})
        elif op == "probe":
            seen = {}
            for path in step["paths"]:
                try:
                    if os.path.isdir(path):
                        seen[path] = "readable" if os.listdir(path) else "hidden"
                    else:
                        with open(path, "rb") as handle:
                            seen[path] = "readable" if handle.read(1) else "hidden"
                except OSError as exc:
                    seen[path] = "hidden (%s)" % exc.strerror
            with open(os.path.join(os.getcwd(), step.get("out", "isolation-probe.json")), "w") as handle:
                json.dump(seen, handle, indent=1)
            output = json.dumps(seen)
        elif op == "sleep":
            time.sleep(float(step.get("seconds", 1)))
            output = "slept"
        elif op == "symlink":
            path = os.path.join(os.getcwd(), step["path"])
            if os.path.lexists(path):
                os.unlink(path)
            os.symlink(step["target"], path)
            output = "linked %s -> %s" % (step["path"], step["target"])
        elif op == "write_outside":
            seen = {}
            for path in step["paths"]:
                try:
                    with open(path, "w") as handle:
                        handle.write("agent was here\n")
                    seen[path] = "WRITTEN"
                except OSError as exc:
                    seen[path] = "refused (%s)" % exc.strerror
            with open(os.path.join(os.getcwd(), step.get("out", "write-probe.json")), "w") as handle:
                json.dump(seen, handle, indent=1)
            output = json.dumps(seen)
        turns += 1
        emit({"type": "assistant", "session_id": session_id, "message": {
            "id": "msg_%s" % uuid.uuid4().hex[:12], "model": args.model, "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": "Bash", "input": {"step": op}}],
            "usage": {"input_tokens": 10, "output_tokens": 50, "cache_read_input_tokens": 1000,
                      "cache_creation_input_tokens": 200}}})
        emit({"type": "user", "session_id": session_id, "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": output[-2000:]}]}})
    memory["cursor"] = cursor + 1
    with open(memory_path, "w", encoding="utf-8") as handle:
        json.dump(memory, handle)
    cost = float(plan.get("cost_per_session", 0.5))
    emit({"type": "result", "subtype": "success", "is_error": False, "num_turns": turns + 1,
          "result": "session %d done" % args.session_index, "session_id": session_id,
          "total_cost_usd": cost, "duration_ms": 10,
          "usage": {"input_tokens": 10 * (turns + 1), "output_tokens": 50 * (turns + 1),
                    "cache_read_input_tokens": 1000 * turns, "cache_creation_input_tokens": 200}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
