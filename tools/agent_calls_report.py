#!/usr/bin/env python3
"""Summarise agent-call ledgers (agent-calls.jsonl) per component / backend / model / status.

Every record counts: retries, failures, timeouts, invalid answers and cache hits appear
next to successes, and calls without a known cost are counted, never treated as free.

Examples:
  tools/agent_calls_report.py runs/                      # every agent-calls.jsonl below runs/
  tools/agent_calls_report.py a.jsonl b.jsonl --json report.json
  tools/agent_calls_report.py runs/ --group-by model,status --json -
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datafarm.agent.call_report import (  # noqa: E402
    GROUP_FIELDS,
    build_report,
    expand_paths,
    load_records,
    render_table,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="ledger files or directories")
    parser.add_argument(
        "--group-by",
        default=",".join(GROUP_FIELDS),
        help="comma-separated record fields (default: %(default)s)",
    )
    parser.add_argument("--json", dest="json_out", help="write the JSON report here ('-' = stdout)")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="report and skip malformed lines instead of failing",
    )
    args = parser.parse_args(argv)
    by = [name.strip() for name in args.group_by.split(",") if name.strip()]
    try:
        sources = expand_paths(args.paths)
        records, problems = load_records(sources, strict=not args.skip_invalid)
        report = build_report(records, by=by, sources=sources, problems=problems)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json_out == "-":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"{len(records)} call records from {len(sources)} ledger file(s)")
    print(render_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
