"""Progress of run_worker.sh sessions: tool calls, permission denials, out/ files, cost when done.
python transfer/status.py DIR [DIR ...]"""

import json
import sys
from pathlib import Path

for d in map(Path, sys.argv[1:]):
    s = d / "worker.stream.jsonl"
    if not s.exists():
        print(d.name, "not started")
        continue
    calls = denied = 0
    init = res = None
    tools = {}
    for line in s.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "system" and e.get("subtype") == "init":
            init = e
        if e.get("type") == "assistant":
            for c in e["message"].get("content", []):
                if c.get("type") == "tool_use":
                    calls += 1
                    tools[c["name"]] = tools.get(c["name"], 0) + 1
        if e.get("type") == "user":
            for c in (
                e["message"].get("content", [])
                if isinstance(e["message"].get("content"), list)
                else []
            ):
                if (
                    c.get("type") == "tool_result"
                    and c.get("is_error")
                    and "permission" in str(c.get("content")).lower()
                ):
                    denied += 1
        if e.get("type") == "result":
            res = e
    out = sorted(
        str(p.relative_to(d)) + f"({p.stat().st_size})"
        for p in (d / "out").rglob("*")
        if p.is_file()
    )
    print(
        d.name,
        "model",
        init and init.get("model"),
        "tools",
        init and init.get("tools"),
        "mcp",
        init and init.get("mcp_servers"),
    )
    print("  calls", calls, tools, "denied", denied, "out", out)
    if res:
        print(
            "  DONE",
            res.get("subtype"),
            "is_error",
            res.get("is_error"),
            "cost",
            res.get("total_cost_usd"),
            "min",
            round(res.get("duration_ms", 0) / 6e4, 1),
        )
    ex = d / "worker.exit"
    if ex.exists():
        print("  ", ex.read_text().strip())
